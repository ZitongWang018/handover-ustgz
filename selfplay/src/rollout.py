"""Shared rollout policy, token handling, and task-slot alignment helpers."""
from __future__ import annotations

import contextlib
import os
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


from reward import TrajectoryStats

# --------------------------------------------------------------------------- 配置


@dataclass
class RolloutConfig:
    group_size: int = 8
    max_steps: int = 30
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.05
    max_context_tokens: int = 32768
    keep_recent_turns: int = 12
    stop_mode: str = "tool_call"
    seed: Optional[int] = None
    log_every: int = 0
    max_batch_size: int = 0             # 0 processes the full request batch
    prompt_format: str = "raw"          # explicit chat_template in the current shared config


# --------------------------------------------------------------------------- 数据结构


@dataclass
class Trajectory:
    task_id: str
    task_type: str
    game_file: str
    messages: List[Dict[str, str]]
    input_ids: List[int]
    loss_mask: List[int]
    stats: TrajectoryStats
    reward: float = 0.0
    advantage: float = 0.0
    n_turns: int = 0
    n_tokens: int = 0
    # Solver segments are exact independent (prompt IDs, generated response IDs)
    # pairs. Rebuilt prompts are not a continuation of the prior turn's token prefix.
    segments: List[Tuple[List[int], List[int]]] = field(default_factory=list)

    @property
    def n_trainable_tokens(self) -> int:
        if self.segments:
            return sum(len(response_ids) for _prompt_ids, response_ids in self.segments)
        return sum(self.loss_mask[1:])

    @property
    def has_trainable_tokens(self) -> bool:
        return self.n_trainable_tokens > 0

    def summary(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "reward": round(self.reward, 4),
            "won": bool(self.stats.won),
            "steps": self.stats.n_steps,
            "invalid": self.stats.n_invalid,
            "format_bad": self.stats.n_format_violation,
            "repeats": self.stats.n_repeat,
            "tokens": self.n_tokens,
        }


# --------------------------------------------------------------------------- 策略

_ACTION_MAX_CHARS = 64


class HFPolicy:
    """把 HF 模型包成"给一批 prompt 返回一批生成结果"的策略。"""

    def __init__(self, model, tokenizer, device=None, enable_thinking: bool = False):
        import torch

        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch
        # ★ model=None 是**允许**的：vLLM 后端（ez_vllm_policy.VLLMPolicy）继承本类
        #   只为复用"chat 模板 + 分词"这一套，采样完全走 vLLM，不需要 HF 模型对象。
        #   这样分词面（initial_prompt_ids / assistant_end_ids / append_feedback_ids /
        #   next_generation_prefix_ids / truncate_messages）是**同一份代码**，
        #   两个后端产出的 prompt token 序列逐字节相同 —— 这是"算法逻辑不变"的前提。
        if device is not None:
            self.device = device
        elif model is not None:
            self.device = next(model.parameters()).device
        else:
            self.device = "cpu"
        self.enable_thinking = enable_thinking
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ---- chat 模板 ----
    def render(self, messages: Sequence[Dict[str, str]], add_generation_prompt: bool = False) -> str:
        kwargs = {"enable_thinking": bool(self.enable_thinking)}
        try:
            return self.tokenizer.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=add_generation_prompt, **kwargs
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=add_generation_prompt
            )

    def tokenize_text(self, text: str) -> List[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def initial_prompt_ids(self, messages: Sequence[Dict[str, str]], cfg: RolloutConfig) -> List[int]:
        truncated = self.truncate_messages(list(messages), cfg)
        return self.tokenize_text(self.render(truncated, add_generation_prompt=True))

    def append_feedback_ids(self, content: str) -> List[int]:
        return self.tokenize_text(self.render([{"role": "user", "content": content}]))

    def assistant_end_ids(self) -> List[int]:
        # TextWorld/Qwen3 template's completed assistant boundary.
        return self.tokenize_text("<|im_end|>\n")

    def next_generation_prefix_ids(self, feedback: str) -> List[int]:
        msgs = [{"role": "user", "content": feedback}]
        complete = self.tokenize_text(self.render(msgs))
        prompt = self.tokenize_text(self.render(msgs, add_generation_prompt=True))
        if prompt[:len(complete)] != complete:
            raise RuntimeError("chat template generation prefix is not a stable suffix after user feedback")
        return prompt[len(complete):]

    def truncate_messages(self, messages: List[Dict[str, str]], cfg: RolloutConfig) -> List[Dict[str, str]]:
        """Fit the prompt and reserved generation budget, dropping whole old turns."""
        budget = max(2, cfg.max_context_tokens - cfg.max_new_tokens)
        kept = list(messages)
        if len(kept) < 2:
            return kept
        while len(self.tokenize_text(self.render(kept, add_generation_prompt=True))) > budget:
            if len(kept) <= 2:
                raise ValueError("system/task prompt alone exceeds context budget")
            # Remove one complete old exchange: an assistant response plus all
            # following user feedback/nudges up to the next assistant response.
            assistant_boundaries = [i for i in range(2, len(kept))
                                    if kept[i]["role"] == "assistant"]
            end = assistant_boundaries[1] if len(assistant_boundaries) > 1 else len(kept)
            del kept[2:end]
        return kept

    # ---- 生成 ----
    def generate(self, messages_list: Sequence[Sequence[Dict[str, str]]], cfg: RolloutConfig) -> Tuple[List[str], List[List[int]], List[int]]:
        """Length-sort and split prompt batches to bound transient KV cache."""
        if not messages_list:
            return [], [], []
        chunk = len(messages_list) if int(cfg.max_batch_size) <= 0 else int(cfg.max_batch_size)
        rendered = [self.truncate_messages(list(m), cfg) for m in messages_list]
        lengths = [len(self.tokenize_text(self.render(m, add_generation_prompt=True))) for m in rendered]
        order = sorted(range(len(rendered)), key=lambda i: lengths[i])
        texts, ids, stops = [None] * len(order), [None] * len(order), [None] * len(order)
        reasons = [None] * len(order)
        for start in range(0, len(order), chunk):
            idx = order[start:start + chunk]
            out = self._generate_batch([rendered[i] for i in idx], cfg)
            for j, original in enumerate(idx):
                texts[original], ids[original], stops[original] = out[0][j], out[1][j], out[2][j]
                current = getattr(self, "last_finish_reasons", [])
                reasons[original] = current[j] if j < len(current) else None
        self.last_finish_reasons = reasons
        return texts, ids, stops

    def generate_from_token_ids(self, prompt_ids_list: Sequence[Sequence[int]],
                                cfg: RolloutConfig, *, request_seeds: Optional[Sequence[int]] = None) -> Tuple[List[str], List[List[int]], List[int]]:
        """Generate from exact token-native histories, preserving sampled prefixes."""
        if not prompt_ids_list:
            return [], [], []
        budget = int(cfg.max_context_tokens)
        if any(len(ids) + cfg.max_new_tokens > budget for ids in prompt_ids_list):
            raise ValueError("EZ conversation exceeded hard context budget; refusing off-policy truncation")
        if request_seeds is not None and len(request_seeds) != len(prompt_ids_list):
            raise ValueError("request_seeds must match prompt batch length")
        chunk = len(prompt_ids_list) if int(cfg.max_batch_size) <= 0 else int(cfg.max_batch_size)
        order = sorted(range(len(prompt_ids_list)), key=lambda i: len(prompt_ids_list[i]))
        texts, ids, stops = [None] * len(order), [None] * len(order), [None] * len(order)
        reasons = [None] * len(order)
        if request_seeds is not None:
            # Keep the HF reference path on Transformers' native sampler. HF does
            # not expose stable per-row generators across supported versions, so
            # seeded reference calls run one prompt at a time under isolated RNG.
            # The production vLLM path remains batched with native per-request seeds.
            for original in order:
                seed = int(request_seeds[original])
                out = self._generate_batch([], cfg,
                    prompt_ids_list=[prompt_ids_list[original]], request_seeds=[seed])
                texts[original], ids[original], stops[original] = out[0][0], out[1][0], out[2][0]
                current = getattr(self, "last_finish_reasons", [])
                reasons[original] = current[0] if current else None
            self.last_finish_reasons = reasons
            return texts, ids, stops
        for start in range(0, len(order), chunk):
            idx = order[start:start + chunk]
            out = self._generate_batch([], cfg, prompt_ids_list=[prompt_ids_list[i] for i in idx],
                                       request_seeds=None)
            for j, original in enumerate(idx):
                texts[original], ids[original], stops[original] = out[0][j], out[1][j], out[2][j]
                current = getattr(self, "last_finish_reasons", [])
                reasons[original] = current[j] if j < len(current) else None
        self.last_finish_reasons = reasons
        return texts, ids, stops

    def _generate_batch(self, messages_list: Sequence[Sequence[Dict[str, str]]], cfg: RolloutConfig,
                        prompt_ids_list: Optional[Sequence[Sequence[int]]] = None,
                        request_seeds: Optional[Sequence[int]] = None) -> Tuple[List[str], List[List[int]], List[int]]:
        """返回 (texts, gen_ids, stop_len)。

        `stop_len[i]` 是"该行已经可以停下"的生成长度（含触发停止的那个换行）。
        """
        if cfg.stop_mode not in {"tool_call", "action_tag", "eos"}:
            raise ValueError(f"unsupported stop_mode={cfg.stop_mode!r}")
        torch = self.torch
        # 生成必须用左 padding，否则右侧 pad 会被当成上下文
        prev_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            if prompt_ids_list is None:
                prompts = [self.render(self.truncate_messages(list(m), cfg), add_generation_prompt=True)
                           for m in messages_list]
                enc = self.tokenizer(list(prompts), return_tensors="pt", padding=True,
                                     add_special_tokens=False).to(self.device)
            else:
                enc = self.tokenizer.pad({"input_ids": [list(x) for x in prompt_ids_list]},
                                         padding=True, return_tensors="pt").to(self.device)
        finally:
            self.tokenizer.padding_side = prev_side
        # ⚠️⚠️ 左 padding 时"生成段起点"是**含 padding 的初始总长度**，
        #    不是 `attention_mask.sum()`（那个是**不含 pad** 的真实 prompt 长度）。
        #    两者在 batch 内各条长度不等时**不相等**，用错就会把 prompt 的尾巴
        #    当成模型生成解码出来 —— 实测出现过：
        #        raw='\n- inventory\n- look\nassistant\nThought: ... \nAction: go to cabinet 1'
        #    因为 parse_action 取的是**最后一个** `Action:`，动作还能解析对（所以
        #    invalid_rate 看不出异常），但 `gen_ids` 被污染 -> mask=1 覆盖了非生成
        #    token -> **训练目标被静默污染**。batch=1 时无 padding，不会触发，
        #    所以这个问题只在多卡/多槽训练时才暴露。
        gen_start = int(enc["input_ids"].shape[1])
        if cfg.stop_mode == "tool_call":
            # EZ：<tool_call>{...}</tool_call> 写完即停。主线判据（Action: 行）对这种
            # 输出永远不触发 -> 每轮白生成到 max_new_tokens（review P2-14）。
            criteria = _ToolCallStopCriteria(
                self.tokenizer, [gen_start] * enc["input_ids"].shape[0], cfg.max_new_tokens)
        elif cfg.stop_mode == "action_tag":
            criteria = _ActionTagStopCriteria(
                self.tokenizer, [gen_start] * enc["input_ids"].shape[0], cfg.max_new_tokens)

        gen_kwargs = dict(
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.temperature > 0,
            repetition_penalty=cfg.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if request_seeds is not None:
            if len(request_seeds) != int(enc["input_ids"].shape[0]):
                raise ValueError("request_seeds must match generated batch")
            if len(request_seeds) != 1:
                raise ValueError("seeded HF generation is intentionally single-request")
        if cfg.stop_mode != "eos":
            # transformers>=4.5x requires a list for custom criteria.
            gen_kwargs["stopping_criteria"] = [criteria]
        if cfg.temperature > 0:
            gen_kwargs.update(temperature=cfg.temperature, top_p=cfg.top_p,
                              top_k=max(0, int(cfg.top_k)))
            if cfg.top_k and cfg.top_k > 0:
                gen_kwargs["top_k"] = cfg.top_k
            if cfg.min_p and cfg.min_p > 0:
                gen_kwargs["min_p"] = cfg.min_p
        else:
            # Explicitly clear generation_config sampling defaults for deterministic
            # evaluation; greedy decoding must not inherit hidden temperature/top-k.
            gen_kwargs.update(temperature=None, top_p=None, top_k=None, min_p=None)
        # This is safe for Qwen3's selective lm_head path. Other model families
        # fall back to their ordinary generation path.
        import inspect
        lm = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        try:
            if "logits_to_keep" in inspect.signature(lm.forward).parameters:
                gen_kwargs["logits_to_keep"] = 1
        except (TypeError, ValueError):
            pass

        # ⚠️⚠️ gradient_checkpointing + generate() 会让 Qwen3 输出**变乱码**。
        #
        #    实测（env_setup/probe_gckpt.py 的 2×2×2 组合实验，8 组里只有 1 组炸）：
        #        train=True + GC=开 + use_cache=True（被 transformers 强制改 False）
        #        ->  'Thoughts = 1999999999999999...'   无 Action: 行，一路顶到 max_new_tokens
        #        GC=关（其它任意组合）
        #        ->  'Thought: I need to find the toiletpaper first...\nAction: go to cabinet 1'
        #
        #    后果极隐蔽：环境收到的动作 100% 非法（invalid_rate=1.0），
        #    奖励恒等于「全程非法」那一档 -> 组内 std=0 -> advantage 全 0 -> loss=0，
        #    **训练看起来在跑、日志也不报错，但一步都没学**。
        #
        #    训练需要 GC 省显存，所以这里只在**生成期间**临时关掉，生成完再打开。
        #    （所有 rank 都会走这条分支，DDP 不会不一致。）
        _gc_on = bool(getattr(self.model, "is_gradient_checkpointing", False))
        if _gc_on:
            gc_set(self.model, False)          # 幂等：外层已关时这里是 no-op

        # ⚠️⚠️ fp32 参数 + flash_attention_2 时，generate **必须**开 bf16 autocast。
        #
        #    训练侧有 `_autocast_bf16()` 包着，但 generate 是另一条独立路径：
        #    不开 autocast 的话 q/k/v 都是 fp32，而 flash-attn 内核只吃 fp16/bf16，
        #    会在第一次 forward 就抛
        #        "FlashAttention only support fp16 and bf16 data type"
        #    —— 而这个报错只会在**正式 fp32 配置**下出现（bf16 冒烟永远碰不到）。
        #
        #    更根本的理由：on-policy RL 要求「采样分布 == 被优化的分布」。
        #    采样跑 fp32、训练跑 bf16，会凭空引入一个 policy mismatch。
        try:
            _pdt = next(self.model.parameters()).dtype
        except StopIteration:
            _pdt = None
        # ⚠️ `cache_enabled=False`：autocast 默认会把 cast 过的 fp32 权重**缓存成 bf16**，
        #    4B 模型就是额外 8GB 常驻。fp32 参数 + AdamW 已经吃掉 64GB，
        #    A800 80G 上这 8GB 就是 OOM 与不 OOM 的区别。代价是每次 forward 重新 cast
        #    一遍（4B 参数 ~0.05s），而 RL 里时间大头在 rollout，可以忽略。
        _ac = torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                             enabled=(_pdt == torch.float32), cache_enabled=False)
        try:
            if request_seeds is None:
                with torch.no_grad(), _ac:
                    out = self.model.generate(**enc, use_cache=True, **gen_kwargs)
            else:
                device = enc["input_ids"].device
                devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
                with torch.random.fork_rng(devices=devices), torch.no_grad(), _ac:
                    torch.manual_seed(int(request_seeds[0]))
                    if device.type == "cuda":
                        torch.cuda.manual_seed(int(request_seeds[0]))
                    out = self.model.generate(**enc, use_cache=True, **gen_kwargs)
        finally:
            if _gc_on:
                gc_set(self.model, True)

        # 切片起点同样必须用 `gen_start`（含 padding 的总长），理由同上。
        plens = [gen_start] * enc["input_ids"].shape[0]
        texts, gen_ids, stops, finish_reasons = [], [], [], []
        for i in range(out.shape[0]):
            plen = int(plens[i])
            all_ids = out[i, plen:].tolist()

            # ---- 生成段精确定界：三道闸门，顺序不能反 ----
            # 1) eos 闸门：找到**第一个** eos 的位置。generate 对已 EOS 的行会继续
            #    填 pad 补齐到统一长度，所以 eos 之后的内容全是垃圾。
            eos_id = self.tokenizer.eos_token_id
            pad_id = self.tokenizer.pad_token_id
            eos_pos = len(all_ids)
            if eos_id is not None:
                for _j, _t in enumerate(all_ids):
                    if _t == eos_id:
                        eos_pos = _j
                        break
            # 2) 语义闸门：criterion 记录的"写完 Action: 那一行"的长度。
            #    以 eos_pos 作 fallback，避免 criterion 从未触发时把整段（含 pad）吞进来。
            semantic_stop = eos_pos if cfg.stop_mode == "eos" else criteria.stop_len(i, eos_pos)
            stop = min(int(semantic_stop), eos_pos)
            # 3) pad 过滤：整段过滤（而非只弹尾部），杜绝任何 pad 漏进 loss_mask=1。
            ids = [t for t in all_ids[:stop] if t != pad_id]

            gen_ids.append(ids)
            stops.append(len(ids))
            texts.append(self.tokenizer.decode(ids, skip_special_tokens=True))
            stop_reason = {"tool_call": "tool_call_stop", "action_tag": "action_tag_stop"}.get(cfg.stop_mode)
            finish_reasons.append(stop_reason if stop_reason and semantic_stop < eos_pos
                                  else "eos" if eos_pos < len(all_ids)
                                  else "max_new_tokens")
        self.last_finish_reasons = finish_reasons
        return texts, gen_ids, stops


class _ActionTagStopCriteria:
    """Stop each HF row at the first complete SkillRise `</action>` tag."""

    def __init__(self, tokenizer, prompt_lens: Sequence[int], max_new_tokens: int,
                 check_every: int = 1):
        self.tokenizer = tokenizer
        self.prompt_lens = list(prompt_lens)
        self.max_new_tokens = max_new_tokens
        self.check_every = check_every
        self._stop: Dict[int, int] = {}
        self._step = 0
        self.torch = __import__("torch")

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        from protocol import ACTION_CLOSE

        self._step += 1
        rows = input_ids.shape[0]
        if self._step % self.check_every == 0:
            for i in range(rows):
                if i in self._stop:
                    continue
                generated_len = input_ids.shape[1] - self.prompt_lens[i]
                if generated_len <= 0:
                    continue
                text = self.tokenizer.decode(input_ids[i, self.prompt_lens[i]:],
                                             skip_special_tokens=True)
                if ACTION_CLOSE in text:
                    self._stop[i] = generated_len
        if len(self._stop) == rows:
            return True
        if hasattr(input_ids, "new_zeros"):
            mask = input_ids.new_zeros((rows,), dtype=self.torch.bool)
            for i in self._stop:
                mask[i] = True
            return mask
        return False

    def stop_len(self, row: int, fallback: int) -> int:
        return self._stop.get(row, fallback)


class _ToolCallStopCriteria:
    """Stop at a closed tool call or the first complete bare JSON action."""

    CLOSE = "</tool_call>"

    def __init__(self, tokenizer, prompt_lens: Sequence[int], max_new_tokens: int,
                 start_after: int = 4, check_every: int = 1):
        self.tokenizer = tokenizer
        self.prompt_lens = list(prompt_lens)
        self.max_new_tokens = max_new_tokens
        self.start_after = start_after
        self.check_every = check_every
        self._stop: Dict[int, int] = {}
        self._step = 0
        self.torch = __import__("torch")

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        self._step += 1
        rows = input_ids.shape[0]
        if self._step % self.check_every != 0:
            return len(self._stop) >= rows if rows else True
        for i in range(rows):
            if i in self._stop:
                continue
            generated_len = input_ids.shape[1] - self.prompt_lens[i]
            if generated_len < self.start_after:
                continue
            text = self.tokenizer.decode(input_ids[i, self.prompt_lens[i]:], skip_special_tokens=True)
            if self.CLOSE in text or scan_bare_json_call(text) is not None:
                self._stop[i] = generated_len
        return self._finished(input_ids, rows)

    def _finished(self, input_ids, rows):
        if rows == 0:
            return True
        if hasattr(input_ids, "new_zeros"):
            mask = input_ids.new_zeros((rows,), dtype=self.torch.bool)
            for i in self._stop:
                mask[i] = True
            return mask
        return len(self._stop) >= rows

    def stop_len(self, row: int, fallback: int) -> int:
        return self._stop.get(row, fallback)


def scan_bare_json_call(text: str) -> Optional[Tuple[int, int]]:
    """Return the first balanced JSON object that looks like an EZ action/declaration."""
    n = len(text)
    i = 0
    while (i := text.find("{", i)) >= 0:
        depth = 0
        in_string = False
        escaped = False
        for j in range(i, n):
            ch = text[j]
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = not in_string
            elif not in_string and ch == "{":
                depth += 1
            elif not in_string and ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[i:j + 1]
                    if (any(key in candidate for key in ('"action_type"', '"task_description"',
                                                          '"name"', '"action"'))
                            and not re.search(r'"arguments"\s*:\s*$', text[:i].rstrip())):
                        return i, j + 1
                    break
        i += 1
    return None


# --------------------------------------------------------------------------- gradient checkpointing 开关

def gc_set(model, on: bool) -> bool:
    """Toggle gradient checkpointing only when its state changes."""
    cur = bool(getattr(model, "is_gradient_checkpointing", False))
    if cur == on:
        return False
    if on:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_disable()
    return True


@contextlib.contextmanager
def generation_mode(model):
    """Temporarily disable gradient checkpointing while generating."""
    was = bool(getattr(model, "is_gradient_checkpointing", False))
    if was:
        gc_set(model, False)
    try:
        yield
    finally:
        if was:
            gc_set(model, True)


def _align_slot_tasks(games: Sequence[str], used: Sequence[str], group_size: int) -> List[int]:
    """把 env 实际装载的槽位对齐回请求的 task 索引。

    textworld 的 batch env 在 `reset()` 时会对 gamefiles 做 shuffle
    （`textworld/gym/envs/textworld_batch.py:96-117`），所以 slot i 通常**不是**
    请求列表的第 i 项。这里按「同一 gamefile 的出现次数逐个消耗」做对齐，
    因此 `group_size > 1`（同一 task 在请求里重复出现）也能正确处理。

    Returns:
        `slot_task[s]` = env 第 s 个 slot 实际装载的是第几个 task。
    """
    n_slots = len(games)
    if [os.path.realpath(u) for u in used] == [os.path.realpath(x) for x in games]:
        return [i // group_size for i in range(n_slots)]
    avail: Dict[str, "deque[int]"] = defaultdict(deque)
    for idx, gf in enumerate(games):
        avail[os.path.realpath(gf)].append(idx // group_size)
    slot_task: List[int] = []
    for u in used:
        q = avail[os.path.realpath(u)]
        if not q:
            raise RuntimeError(
                f"env 实际装载的 gamefile 不在请求列表里（或出现次数超了）：{u}"
            )
        slot_task.append(int(q.popleft()))
    return slot_task
