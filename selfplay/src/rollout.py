"""ReAct rollout：让策略模型在 ALFWorld 里一局一局地玩，并记录可训练的词元序列。

关键约定
--------
* **环境侧**：用 `ALFWorldBatchEnv`（`AsyncBatchEnv`，每 slot 一个子进程）。每个 task 采 `group_size`
  条独立轨迹 —— 靠把同一个 game file 在 batch 里重复 `group_size` 次实现，再用
  `extra.gamefile` 反查确认。
* **模型侧**：只用 HF `generate`（不依赖 vLLM），DDP 下每个 rank 自己跑自己的轨迹，天然并行。
* **词元序列**：assistant 部分用**真实生成的 token id**（不是重新 tokenize 的文本），
  非 assistant 部分（chat 模板骨架 + 环境观测）用重新渲染得到的 token 并**置 mask=0**。
  这样 loss 只作用在模型自己产出的词元上 —— 多轮智能体 RL 的标准做法。
"""
from __future__ import annotations

import contextlib
import os
import random
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from prompts import ParsedAction, build_system_prompt, format_observation, format_turn_feedback, match_admissible, parse_action
from reward import RewardConfig, TrajectoryStats, compute_reward

# --------------------------------------------------------------------------- 配置


@dataclass
class RolloutConfig:
    group_size: int = 8
    max_steps: int = 30
    max_new_tokens: int = 96
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.05
    max_context_tokens: int = 8192
    keep_recent_turns: int = 12
    invalid_action_mode: str = "raw"       # "raw" | "best_match"
    fuzzy_threshold: float = 0.6
    # 停止条件模式：
    #   "action_line"（默认，主线）—— 等到 Action: 那行写完
    #   "tool_call"（EZ）—— 一看到 </tool_call> 就停，避免白生成到 max_new_tokens
    stop_mode: str = "action_line"
    seed: Optional[int] = None
    log_every: int = 0
    max_batch_size: int = 0             # 0 keeps mainline legacy batching


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
        for start in range(0, len(order), chunk):
            idx = order[start:start + chunk]
            out = self._generate_batch([rendered[i] for i in idx], cfg)
            for j, original in enumerate(idx):
                texts[original], ids[original], stops[original] = out[0][j], out[1][j], out[2][j]
        return texts, ids, stops

    def generate_from_token_ids(self, prompt_ids_list: Sequence[Sequence[int]],
                                cfg: RolloutConfig) -> Tuple[List[str], List[List[int]], List[int]]:
        """Generate from exact token-native histories, preserving sampled prefixes."""
        if not prompt_ids_list:
            return [], [], []
        budget = int(cfg.max_context_tokens)
        if any(len(ids) + cfg.max_new_tokens > budget for ids in prompt_ids_list):
            raise ValueError("EZ conversation exceeded hard context budget; refusing off-policy truncation")
        chunk = len(prompt_ids_list) if int(cfg.max_batch_size) <= 0 else int(cfg.max_batch_size)
        order = sorted(range(len(prompt_ids_list)), key=lambda i: len(prompt_ids_list[i]))
        texts, ids, stops = [None] * len(order), [None] * len(order), [None] * len(order)
        for start in range(0, len(order), chunk):
            idx = order[start:start + chunk]
            out = self._generate_batch([], cfg, prompt_ids_list=[prompt_ids_list[i] for i in idx])
            for j, original in enumerate(idx):
                texts[original], ids[original], stops[original] = out[0][j], out[1][j], out[2][j]
        return texts, ids, stops

    def _generate_batch(self, messages_list: Sequence[Sequence[Dict[str, str]]], cfg: RolloutConfig,
                        prompt_ids_list: Optional[Sequence[Sequence[int]]] = None) -> Tuple[List[str], List[List[int]], List[int]]:
        """返回 (texts, gen_ids, stop_len)。

        `stop_len[i]` 是"该行已经可以停下"的生成长度（含触发停止的那个换行）。
        """
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
        else:
            criteria = _ActionStopCriteria(
                self.tokenizer, [gen_start] * enc["input_ids"].shape[0], cfg.max_new_tokens)

        gen_kwargs = dict(
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.temperature > 0,
            temperature=max(cfg.temperature, 1e-5),
            top_p=cfg.top_p,
            top_k=max(0, int(cfg.top_k)),
            repetition_penalty=cfg.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            # ⚠️ transformers>=4.5x 不再自动把单个 StoppingCriteria 包成列表，必须显式给 list
            stopping_criteria=[criteria],
        )
        if cfg.top_k and cfg.top_k > 0:
            gen_kwargs["top_k"] = cfg.top_k
        if cfg.min_p and cfg.min_p > 0:
            gen_kwargs["min_p"] = cfg.min_p
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
            with torch.no_grad(), _ac:
                out = self.model.generate(**enc, use_cache=True, **gen_kwargs)
        finally:
            if _gc_on:
                gc_set(self.model, True)

        # 切片起点同样必须用 `gen_start`（含 padding 的总长），理由同上。
        plens = [gen_start] * enc["input_ids"].shape[0]
        texts, gen_ids, stops = [], [], []
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
            stop = min(int(criteria.stop_len(i, eos_pos)), eos_pos)
            # 3) pad 过滤：整段过滤（而非只弹尾部），杜绝任何 pad 漏进 loss_mask=1。
            ids = [t for t in all_ids[:stop] if t != pad_id]

            gen_ids.append(ids)
            stops.append(len(ids))
            texts.append(self.tokenizer.decode(ids, skip_special_tokens=True))
        return texts, gen_ids, stops


class _ActionStopCriteria:
    """等到**所有**行都写完了 `Action:` 那一行再停（HF generate 无法按行停）。

    记录每行第一次满足停止条件的**生成长度**（相对 prompt 的偏移），
    这样 `stop_len()` 能把"这一行实际想说的内容"精确切出来。
    """

    def __init__(self, tokenizer, prompt_lens: Sequence[int], max_new_tokens: int,
                 start_after: int = 6, check_every: int = 2):
        self.tokenizer = tokenizer
        self.prompt_lens = list(prompt_lens)
        self.max_new_tokens = max_new_tokens
        self.start_after = start_after
        self.check_every = check_every
        self._stop: Dict[int, int] = {}
        self._step = 0
        self.torch = __import__("torch")

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        from prompts import stop_text_finished

        self._step += 1
        rows = input_ids.shape[0]
        for i in range(rows):
            if i in self._stop:
                continue
            generated_len = input_ids.shape[1] - self.prompt_lens[i]
            if generated_len < self.start_after:
                continue
            if self._step % self.check_every != 0:
                continue
            text = self.tokenizer.decode(input_ids[i, self.prompt_lens[i]:], skip_special_tokens=True)
            if stop_text_finished(text):
                self._stop[i] = generated_len
        return self._finished(input_ids, rows)

    def _finished(self, input_ids, rows):
        if len(self._stop) == rows:
            return True
        if isinstance(input_ids, self.torch.Tensor):
            return self.torch.tensor([i in self._stop for i in range(rows)],
                                     dtype=self.torch.bool, device=input_ids.device)
        return False

    def stop_len(self, row: int, fallback: int) -> int:
        """返回该行的生成长度（相对 prompt 的偏移）。"""
        return self._stop.get(row, fallback)


class _ToolCallStopCriteria:
    """EZ 用：写完一个工具调用即停。

    ★ 为什么要认**裸 JSON**（2026-10-12 冒烟实测）：
    只按 `</tool_call>` 闭合标签停的话，Qwen3-4B-Instruct-2507 根本不吐标签——
    它直接把 `{"name":"alfworld_tool","arguments":{...}}` 甩出来，并且**一轮里
    连写 8 个动作**一路顶到 max_new_tokens。后果：
      * 解析侧 `<tool_call>...</tool_call>` 正则 100% 失配 -> fmt_bad=6/6 -> accept=0
      * 即使能解析，一次生成 8 个动作而循环只执行第 1 个，消息历史里的"8 个动作"
        与环境实际执行的 1 个动作对不上，训练信号被污染
    所以停止判据改成：**闭合标签 或 第一个括号配平的动作 JSON 对象**，
    这样"一步一动作"在生成侧就被强制住，不需要事后裁剪。
    """

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
        # Transformers accepts a per-row bool tensor and suppresses finished rows.
        # Older fakes/tests may pass ordinary lists and expect a scalar bool.
        if hasattr(input_ids, "new_zeros"):
            mask = input_ids.new_zeros((rows,), dtype=self.torch.bool) if hasattr(self, "torch") else None
            if mask is None:
                import torch
                mask = torch.zeros((rows,), dtype=torch.bool, device=input_ids.device)
            for i in self._stop:
                mask[i] = True
            return mask
        return len(self._stop) >= rows

    def stop_len(self, row: int, fallback: int) -> int:
        return self._stop.get(row, fallback)


def scan_bare_json_call(text: str) -> Optional[Tuple[int, int]]:
    """扫出**第一个括号配平、且含 `"action_type"` 的 JSON 对象**。

    返回 `(start, end)`（end 为开区间，指向对象右花括号之后）；没有则 None。

    用括号配平而不是正则：`arguments` 是嵌套对象，`\\{.*?\\}` 这类非贪婪正则在
    "有嵌套 + 后面还有别的对象"时会切错位置。字符串内部的花括号/引号必须跳过，
    否则 `{"task_description": "put an apple in {the} bowl"}` 会算错深度。

    ★ 必须排除 `"arguments"` 的**内层对象**（单测抓出来的 bug）：
    `{"name":"alfworld_tool","arguments":{"action_type":"look"}}` 里，内层
    `{"action_type":"look"}` 自己也是配平的、也含 `"action_type"`。外层还没闭合时
    （逐 token 增量判定下必然先出现这一幕），扫描会先命中内层对象就停下 ->
    **工具调用被从中间的 `}` 截断**，解析出来的是半个 JSON。
    所以：紧跟在 `"arguments":` 之后的对象要跳过，继续往外找。
    """
    n = len(text)
    i = 0
    while True:
        i = text.find("{", i)
        if i < 0:
            return None
        depth = 0
        in_str = False
        esc = False
        for j in range(i, n):
            c = text[j]
            if esc:
                esc = False
                continue
            if c == "\\":
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    cand = text[i:j + 1]
                    if _looks_like_tool_call(cand) and \
                            not re.search(r'"arguments"\s*:\s*$', text[:i].rstrip()):
                        return i, j + 1
                    break          # 不是完整工具调用，从下一个 '{' 继续找
        i += 1


#: 严格模式下，只有这两个键才算"工具调用"（原始行为）
_TOOL_KEYS_STRICT = ('"action_type"', '"task_description"')
#: 容错模式下额外认这两个 —— 模型把动作类型写进 `name`/`action` 的实测占比 14.6%
_TOOL_KEYS_TOLERANT = _TOOL_KEYS_STRICT + ('"name"', '"action"')


def _looks_like_tool_call(cand: str) -> bool:
    """候选对象是否"像一个工具调用"（决定 `scan_bare_json_call` 要不要把它切出来）。

    ★★ 2026-10-14 实测抓到的**真根因**：这里原来硬编码只认
       `'"action_type"'` / `'"task_description"'`。于是模型只要
         * 把动作类型写进 `name`（`{"name":"alfworld_tool","arguments":{"name":"go_to",...}}`，
           占 malformed 的 **14.6%**），或
         * 写扁平形式 `{"name":"go_to","receptacle_name":"cabinet 2"}`（占 **62.0%**）
       —— **在扫描阶段就返回 None**，`parse_tool_call` 后面那套容错解析**根本没机会执行**。
       所以光改 `parse_tool_call` 不够，这道前置闸门必须一起放宽。

    容错模式默认开；`EZ_PARSER_STRICT=1` 可恢复原始行为做对照。
    """
    keys = (_TOOL_KEYS_STRICT if os.environ.get("EZ_PARSER_STRICT", "0") == "1"
            else _TOOL_KEYS_TOLERANT)
    return any(k in cand for k in keys)


# --------------------------------------------------------------------------- gradient checkpointing 开关

def gc_set(model, on: bool) -> bool:
    """**幂等地**开/关 gradient_checkpointing，返回是否真的动了。

    ⚠️ 为什么必须幂等：`gradient_checkpointing_enable(use_reentrant=False)` 内部会调
    `enable_input_require_grads()`，后者每次都会往 input embeddings 上**再挂一个**
    forward hook（`make_inputs_require_grad`）。而 rollout 每一步都要 generate 一次，
    300 iter × 30 步 = 9000 次开关 -> 9000 个 hook，显存和耗时一起爆掉，
    而且报错信息完全不会指向这里。所以只有状态真的不同才允许动作。
    """
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
    """进入"生成模式"：整段关掉 gradient_checkpointing，退出时按原样恢复。

    ⚠️ GC 开着会让 Qwen3 的 generate 输出**乱码**（实测：'Thoughts = 1999999...'，
    没有 Action: 行，一路顶到 max_new_tokens）—— 环境收到的动作 100% 非法，
    奖励全在同一档 -> 组内 std=0 -> advantage=0 -> loss=0，
    **训练看起来在跑、日志不报错、但一步都没学**。

    把它做成上下文管理器、并在 `run_rollouts` 外面**一次性**开合，
    是为了不让每一步 generate 都去 toggle（见 `gc_set` 的 hook 累积问题）。
    """
    was = bool(getattr(model, "is_gradient_checkpointing", False))
    if was:
        gc_set(model, False)
    try:
        yield
    finally:
        if was:
            gc_set(model, True)


# --------------------------------------------------------------------------- 动作清洗


_WS_RE = re.compile(r"\s+")


def sanitize_action(action: str, admissible: Sequence[str], cfg: RolloutConfig) -> Tuple[str, bool]:
    """把模型输出的动作变成可以安全丢给 TextWorld 的字符串。

    Returns: (executed_action, is_exact_match)
    """
    action = (action or "").strip()
    action = action.split("\n")[0]
    action = _WS_RE.sub(" ", action)
    if len(action) > _ACTION_MAX_CHARS:
        action = action[:_ACTION_MAX_CHARS]
    if not action.isascii():
        action = action.encode("ascii", "ignore").decode()

    matched = match_admissible(action, admissible)
    if matched is not None:
        return matched, True
    if cfg.invalid_action_mode == "best_match":
        fuzzy = match_admissible(action, admissible, fuzzy_threshold=cfg.fuzzy_threshold)
        if fuzzy is not None:
            return fuzzy, False
    if action:
        return action, False
    # 完全空输出：用一个总是合法的命令兜底，保证 episode 还能往前走
    for cand in ("look", "inventory"):
        if cand in admissible:
            return cand, False
    return (admissible[0] if admissible else "look"), False


# --------------------------------------------------------------------------- 轨迹 → 词元


def build_token_sequence(
    messages: Sequence[Dict[str, str]],
    gen_ids_per_assistant: Sequence[List[int]],
    policy: HFPolicy,
) -> Tuple[List[int], List[int], List[int]]:
    """把一条（**已按上下文预算裁剪过的**）对话渲染成 (input_ids, loss_mask, turn_boundaries)。

    * **assistant 段**：模型**真实生成**的 token id（mask=1）——不是把生成文本重新 tokenize
      （那会引入 "解码→再编码" 的偏差，实测会多出 1 个 token 混进训练目标）；
    * **模板收尾**：assistant 内容之后的 `<|im_end|>` 等，mask=0；只有采样器返回的真实 token 可训练；
    * **其余段**（chat 模板骨架 + 环境观测）：模板重新渲染出来的 token，mask=0。

    实现要点：维护两个游标 —— `ids` 是我们真正输出的序列，`rlen` 是**纯模板渲染序列**上的游标。
    两者必须解耦，因为 assistant 段用的是真实 gen id，其长度通常 ≠ 同一段文本重新编码的长度。
    """
    ids: List[int] = []
    mask: List[int] = []
    boundaries: List[int] = []
    assistant_idx = 0
    rlen = 0
    prev_render: List[int] = []

    def _render_ids(msgs: Sequence[Dict[str, str]], gen_prompt: bool = False) -> List[int]:
        return policy.tokenize_text(policy.render(msgs, add_generation_prompt=gen_prompt))

    for i, msg in enumerate(messages):
        if msg["role"] == "assistant" and assistant_idx < len(gen_ids_per_assistant):
            gen = list(gen_ids_per_assistant[assistant_idx])
            assistant_idx += 1
            # 生成时模型真正看到的前缀（= render(messages[:i]) + 生成提示）
            prefix_r = _render_ids(messages[:i], gen_prompt=True)
            if prev_render and prefix_r[: len(prev_render)] != prev_render:
                raise RuntimeError("chat 模板不是前缀稳定的（assistant 前缀不匹配）—— 检查 tokenizer 配置")
            if len(prefix_r) < rlen:
                raise RuntimeError("chat 模板不是前缀稳定的（渲染长度回退）")
            gen_prompt_tokens = prefix_r[rlen:]          # `<|im_start|>assistant\n` 之类的骨架，mask=0
            full_r = _render_ids(messages[:i + 1], gen_prompt=False)
            if gen:
                # 模板收尾 token：用「同一前缀 + 空 assistant 内容」渲染，prefix 之后就是纯模板尾。
                # 这样拿到的是**精确**的收尾 token，不依赖"解码后再编码"的长度假设。
                empty_full = _render_ids(
                    list(messages[:i]) + [{"role": "assistant", "content": ""}], gen_prompt=False
                )
                if empty_full[: len(prefix_r)] != prefix_r:
                    raise RuntimeError("chat 模板不是前缀稳定的（空 assistant 渲染不以前缀开头）")
                tail = empty_full[len(prefix_r):]
                ids.extend(gen_prompt_tokens)
                mask.extend([0] * len(gen_prompt_tokens))
                ids.extend(gen)
                mask.extend([1] * len(gen))
                ids.extend(tail)
                mask.extend([0] * len(tail))
                boundaries.append(len(ids))
            else:
                # 没有生成任何 token：整段按 mask=0 处理（不该训练空输出）
                seq = gen_prompt_tokens + full_r[len(prefix_r):]
                ids.extend(seq)
                mask.extend([0] * len(seq))
            prev_render = full_r
            rlen = len(full_r)          # 渲染游标推进到该 assistant 消息在**纯模板**下的完整长度
        else:
            cur = _render_ids(messages[:i + 1], gen_prompt=False)
            if prev_render and cur[:len(prev_render)] != prev_render:
                raise RuntimeError(
                    "chat template is not prefix-stable; legacy transcript builder cannot "
                    "preserve the sampled token history. Use token-native EZ rollout or a "
                    "prefix-stable template."
                )
            new = cur[rlen:]
            if len(cur) < rlen:
                raise RuntimeError("chat template render length regressed; refusing lossy sequence")
            ids.extend(new)
            mask.extend([0] * len(new))
            prev_render = cur
            rlen = len(cur)
    if assistant_idx != len(gen_ids_per_assistant):
        raise RuntimeError("gen_ids count does not match assistant messages")
    return ids, mask, boundaries


# --------------------------------------------------------------------------- 主 rollout


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


def run_rollouts(
    env,
    policy,
    tasks: Sequence[Dict[str, Any]],
    cfg: RolloutConfig,
    reward_cfg: RewardConfig,
    system_prompt: Optional[str] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    tokenize: bool = True,
) -> List[Trajectory]:
    """跑一批 rollout。

    Args:
        env: `ALFWorldBatchEnv`，其 `batch_size` 必须等于 `len(tasks) * cfg.group_size`。
        policy: 需实现 `generate(messages_list, cfg) -> (texts, gen_ids, stops)`。HFPolicy 满足；
                走 HTTP API 的评测用策略把 `tokenize=False` 传入即可（不需要词元序列）。
        tasks: [{"game_file":..., "task_id":..., "task_type":...}, ...]

    Returns:
        返回顺序保证是 **task-major / 组连续**：第 i 个 task 的 `group_size` 条轨迹占据
        `[i*g, (i+1)*g)`。调用方可以直接用 `group_ids = [i // g for i in range(len(trajs))]`。
        （env 内部会 shuffle slot，这里已经重排回请求顺序。）
    """
    n_tasks = len(tasks)
    g = cfg.group_size
    n_slots = n_tasks * g
    if env.batch_size != n_slots:
        raise ValueError(f"env.batch_size={env.batch_size} 与 tasks×group={n_slots} 不一致")

    sys_prompt = system_prompt or build_system_prompt()
    games = [t["game_file"] for t in tasks for _ in range(g)]

    obs_list, views, admissible, used, _expert = env.reset(games)
    slot_task = _align_slot_tasks(games, used, g)

    max_steps = min(cfg.max_steps, env.max_episode_steps)

    messages: List[List[Dict[str, str]]] = []
    stats: List[TrajectoryStats] = []
    gen_ids_log: List[List[List[int]]] = []
    for s in range(n_slots):
        t = tasks[slot_task[s]]
        messages.append([{"role": "system", "content": sys_prompt}])
        v = views[s]
        messages[s].append({
            "role": "user",
            "content": format_observation(v.env_desc, admissible[s], first_turn=True, task_desc=v.task_desc),
        })
        stats.append(TrajectoryStats(task_type=t.get("task_type", "unknown"), task_id=t.get("task_id", "")))
        gen_ids_log.append([])

    active = [True] * n_slots
    last_action = [""] * n_slots
    t0 = time.time()

    for step in range(max_steps):
        if not any(active):
            break
        idxs = [s for s in range(n_slots) if active[s]]
        gen_messages = [messages[s] for s in idxs]
        texts, gen_ids, _ = policy.generate(gen_messages, cfg)

        actions = []
        for k, s in enumerate(idxs):
            parsed: ParsedAction = parse_action(texts[k])
            executed, exact = sanitize_action(parsed.action, admissible[s], cfg)
            actions.append((s, parsed, executed, exact))

        step_actions = ["look"] * n_slots
        for s, _p, executed, _e in actions:
            step_actions[s] = executed
        bs = env.step(step_actions)

        for k, (s, parsed, executed, exact) in enumerate(actions):
            st = stats[s]
            st.n_steps += 1
            if not parsed.has_marker:
                st.n_format_violation += 1
            if exact:
                st.n_valid_steps += 1
            else:
                st.n_invalid += 1
            if executed and executed.lower() == last_action[s].lower():
                st.n_repeat += 1
            last_action[s] = executed

            messages[s].append({"role": "assistant", "content": texts[k].strip()})
            gen_ids_log[s].append(list(gen_ids[k]))

            if bool(bs.dones[s]):
                st.won = bool(bs.won[s])
                active[s] = False
            else:
                messages[s].append({
                    "role": "user",
                    "content": format_turn_feedback(
                        executed, bs.obs[s], bs.admissible[s], done=False, won=False
                    ),
                })
        for s in range(n_slots):
            if active[s]:
                admissible[s] = bs.admissible[s]

    # 收尾：没结束的算 truncated。
    # ★ 关键：textworld 会打乱 slot，这里必须重排成 task-major（组连续）顺序，
    #   否则 train_grpo.py 的 group_ids = [i // g ...] 会把不同 task 混进同一组。
    order = sorted(range(n_slots), key=lambda s: (slot_task[s], s))
    trajectories: List[Trajectory] = []
    for s in order:
        st = stats[s]
        if active[s]:
            st.truncated = True
        t = tasks[slot_task[s]]
        if tokenize:
            # ★ 训练序列必须和「生成时实际看到的上下文」一致：生成时每轮都过
            #   `truncate_messages`，所以这里也要先裁再拼，否则 30 步的完整历史会到 16k+ token，
            #   而 per_token_logps 要物化 fp32 logits（R×W×vocab×4B）→ 直接 OOM。
            tr_msgs = policy.truncate_messages(messages[s], cfg)
            n_asst_kept = sum(1 for m in tr_msgs if m["role"] == "assistant")
            # 裁剪保留的是「尾部窗口」，所以留下的 assistant 正好是最后 n_asst_kept 次生成
            gen_kept = gen_ids_log[s][len(gen_ids_log[s]) - n_asst_kept:] if n_asst_kept else []
            ids, mask, _ = build_token_sequence(tr_msgs, gen_kept, policy)
            if not any(mask):
                raise RuntimeError(
                    f"训练序列里没有任何 mask=1 的 token（n_asst_kept={n_asst_kept}）"
                    f" —— 检查 keep_recent_turns={cfg.keep_recent_turns} 是否过小"
                )
        else:
            ids, mask = [], []
        traj = Trajectory(
            task_id=t.get("task_id", ""),
            task_type=t.get("task_type", "unknown"),
            game_file=used[s],          # ★ 用实际装载的那个，不是 games[s]
            messages=messages[s],
            input_ids=ids,
            loss_mask=mask,
            stats=st,
            n_turns=len(gen_ids_log[s]),
            n_tokens=len(ids),
        )
        traj.reward = compute_reward(st, reward_cfg, max_steps)
        trajectories.append(traj)

    if log_fn:
        won = sum(1 for t in trajectories if t.stats.won)
        log_fn(
            f"[rollout] {n_tasks} tasks × {g} = {n_slots} trajs | won={won}/{n_slots} "
            f"({won / max(1, n_slots):.1%}) | mean_steps={np.mean([t.stats.n_steps for t in trajectories]):.1f} "
            f"| mean_tokens={int(np.mean([t.n_tokens for t in trajectories]))} | {time.time() - t0:.1f}s"
        )
    return trajectories


# --------------------------------------------------------------------------- expert 参考点


def run_expert(
    env,
    tasks: Sequence[Dict[str, Any]],
    max_steps: int,
    reward_cfg: Optional[RewardConfig] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> List[Trajectory]:
    """用 ALFWorld **官方 handcoded expert** 直接驱动环境，作为"环境可解性上界"参考点。

    需要 `env` 是以 `with_expert=True` 构造的 `ALFWorldBatchEnv` —— 此时
    `reset()/step()` 的 infos 里会带 `extra.expert_plan`（单元素 list）。动作直接取
    expert 计划，拿不到时回退 `"look"`。

    这里**不经过任何 policy 抽象**：expert 计划只存在于 env infos 里，不经过 chat
    messages，所以 `messages`/`input_ids`/`loss_mask` 都留空，统计里 `n_invalid` /
    `n_format_violation` 恒为 0（expert 计划始终是 admissible 动作）。

    Args:
        env: `with_expert=True` 的 `ALFWorldBatchEnv`，`batch_size` 必须等于 `len(tasks)`。
        tasks: [{"game_file":..., "task_id":..., "task_type":...}, ...]
        max_steps: 每局最多走多少步（与 `env.max_episode_steps` 取 min）。
    """
    reward_cfg = reward_cfg or RewardConfig()
    n_slots = len(tasks)
    if env.batch_size != n_slots:
        raise ValueError(
            f"env.batch_size={env.batch_size} 与 tasks={n_slots} 不一致（expert 评测 group_size=1）"
        )
    if not getattr(env, "with_expert", False):
        raise ValueError("run_expert 需要以 with_expert=True 构造的 ALFWorldBatchEnv")

    games = [t["game_file"] for t in tasks]
    _obs_list, _views, _admissible, used, expert_plan = env.reset(games)
    slot_task = _align_slot_tasks(games, used, 1)
    max_steps = min(int(max_steps), env.max_episode_steps)

    stats: List[TrajectoryStats] = [
        TrajectoryStats(
            task_type=tasks[slot_task[s]].get("task_type", "unknown"),
            task_id=tasks[slot_task[s]].get("task_id", ""),
        )
        for s in range(n_slots)
    ]
    active = [True] * n_slots
    last_action = [""] * n_slots
    plan: List[Optional[str]] = list(expert_plan)
    t0 = time.time()

    for _step in range(max_steps):
        if not any(active):
            break
        actions = [(plan[s] or "look") for s in range(n_slots)]
        bs = env.step(actions)
        for s in range(n_slots):
            if not active[s]:
                continue
            st = stats[s]
            st.n_steps += 1
            st.n_valid_steps += 1        # expert 计划来自 admissible，始终算合法步
            if actions[s].lower() == last_action[s].lower():
                st.n_repeat += 1
            last_action[s] = actions[s]
            if bool(bs.dones[s]):
                st.won = bool(bs.won[s])
                active[s] = False
        plan = list(bs.expert_plan) if bs.expert_plan else [None] * n_slots

    # ★ 同 run_rollouts：env 会打乱 slot，这里重排回请求的 task-major 顺序，
    #   并用实际装载的 used[s] 作为 game_file（不是请求的 games[s]）。
    order = sorted(range(n_slots), key=lambda s: (slot_task[s], s))
    trajectories: List[Trajectory] = []
    for s in order:
        st = stats[s]
        if active[s]:
            st.truncated = True
        traj = Trajectory(
            task_id=st.task_id,
            task_type=st.task_type,
            game_file=used[s],
            messages=[],
            input_ids=[],
            loss_mask=[],
            stats=st,
            n_turns=st.n_steps,
            n_tokens=0,
        )
        traj.reward = compute_reward(st, reward_cfg, max_steps)
        trajectories.append(traj)

    if log_fn:
        won = sum(1 for t in trajectories if t.stats.won)
        log_fn(
            f"[expert] {n_slots} tasks | won={won}/{n_slots} ({won / max(1, n_slots):.1%}) "
            f"| mean_steps={np.mean([t.stats.n_steps for t in trajectories]):.1f} | {time.time() - t0:.1f}s"
        )
    return trajectories
