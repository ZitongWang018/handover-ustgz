"""vLLM sampler using the same token IDs and stop semantics as HFPolicy."""
from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, List, Optional, Sequence, Tuple

from protocol import action_tag_stop_len
from rollout import HFPolicy, RolloutConfig, scan_bare_json_call

_TC_CLOSE = "</tool_call>"
# 与 rollout._ToolCallStopCriteria 的默认值保持一致。
_TC_START_AFTER = 4


def _stop_triggered(text: str) -> bool:
    """`_ToolCallStopCriteria` 的判定式本身（闭合标签 **或** 配平的裸 JSON 调用）。"""
    return (_TC_CLOSE in text) or (scan_bare_json_call(text) is not None)


def toolcall_stop_len(tokenizer, gen_ids: Sequence[int],
                      start_after: int = _TC_START_AFTER) -> int:
    """事后复刻 `_ToolCallStopCriteria` 记录的 `stop_len`。

    返回"该行实际想说的 token 数"。语义与 HF 版逐条相同：
    * 只在 `generated_len >= start_after` 之后才判（HF 里是 `continue`）；
    * 取**第一个**满足判据的生成长度；
    * 一直没触发就返回全长（HF 侧对应 `eos` 闸门 fallback）。
    """
    n = len(gen_ids)
    if n <= start_after:
        return n
    if not _stop_triggered(tokenizer.decode(list(gen_ids), skip_special_tokens=True)):
        return n                      # 快路径：整段都没触发，不必逐前缀重解码
    for k in range(start_after, n + 1):
        if _stop_triggered(tokenizer.decode(list(gen_ids[:k]), skip_special_tokens=True)):
            return k
    return n


def export_lora_adapter(peft_model, adapter_name: str, out_dir: str) -> str:
    """把双 adapter 模型里的**单个** adapter 导出成 vLLM 能读的 PEFT 目录。

    ⚠️ PEFT 0.18 的 `save_pretrained(selected_adapters=[...])` **一定会**多套一层
    `out_dir/<adapter_name>/`，即使只选了一个 adapter。vLLM 只在给定目录下找
    `adapter_config.json`，所以这里必须把真正落盘的位置**探测出来返回**。

    实测踩坑（job 13008934）：直接返回 `out_dir` 会让 vLLM 在第一次请求时报
        ValueError: Loading lora solver failed: No adapter found for <dir>
    而且 EngineCore 会**直接死掉**，整个引擎报废 —— 不是可恢复的异常。

    返回：真正含 `adapter_config.json` 的目录。
    """
    os.makedirs(out_dir, exist_ok=True)
    peft_model.save_pretrained(
        out_dir, selected_adapters=[adapter_name],
        safe_serialization=True, save_embedding_layers=False)

    for cand in (out_dir, os.path.join(out_dir, adapter_name)):
        cfg = os.path.join(cand, "adapter_config.json")
        if os.path.isfile(cfg):
            with open(cfg, encoding="utf-8") as fh:
                meta = json.load(fh)
            r = int(meta.get("r", 0))
            if r <= 0:
                raise RuntimeError(f"导出的 adapter_conf 里 r={r} 不合理：{cfg}")
            # 哨兵写进**真正**的目录，供调用方（和排障）确认"这一份是完整的"。
            with open(os.path.join(cand, "_ready"), "w") as fh:
                fh.write(f"{adapter_name} r={r} alpha={meta.get('lora_alpha')}")
            if cand != out_dir:
                # 顺手把内层内容提升到外层的可读位置不是必需的，但把路径告诉调用方。
                pass
            return cand
    raise RuntimeError(
        f"adapter 导出后找不到 adapter_config.json（既不在 {out_dir} 也不在 "
        f"{out_dir}/{adapter_name}）—— 训练会以错误的 LoRA 采样，必须中止")


def build_vllm_engine(base_path: str, *, gpu_memory_utilization: float = 0.40,
                      max_model_len: int = 40960, max_lora_rank: int = 32,
                      max_num_seqs: int = 64, max_num_batched_tokens: int = 16384,
                      enable_prefix_caching: bool = True,
                      enforce_eager: bool = False, seed: int = 1234,
                      trust_remote_code: bool = True,
                      enable_lora: bool = True):
    """构造单进程（= 单卡 TP=1）vLLM 引擎。

    `gpu_memory_utilization` 控制 vLLM 在可见 GPU 上的显存预算；共享 GPU 时需
    按该设备上其他进程的占用共同规划。
    """
    from vllm import LLM
    kwargs = dict(
        model=base_path,
        tokenizer=base_path,
        dtype="bfloat16",
        trust_remote_code=bool(trust_remote_code),
        gpu_memory_utilization=float(gpu_memory_utilization),
        max_model_len=int(max_model_len),
        max_num_seqs=int(max_num_seqs),
        max_num_batched_tokens=int(max_num_batched_tokens),
        enable_prefix_caching=bool(enable_prefix_caching),
        enforce_eager=bool(enforce_eager),
        disable_log_stats=True,
        seed=int(seed),
    )
    if enable_lora:
        kwargs.update(enable_lora=True, max_lora_rank=int(max_lora_rank),
                      max_loras=2, max_cpu_loras=4)
    else:
        kwargs["enable_lora"] = False
    return LLM(**kwargs)


def normalize_sampling(cfg: RolloutConfig) -> Dict[str, Any]:
    """Map shared rollout sampling settings to valid vLLM arguments."""
    top_k = int(cfg.top_k)
    top_k = top_k if top_k >= 1 else -1
    top_p = float(cfg.top_p)
    top_p = 1.0 if top_p >= 1.0 else max(top_p, 1e-6)
    min_p = float(cfg.min_p)
    min_p = min(max(min_p, 0.0), 1.0)
    temp = float(cfg.temperature)
    rep = max(float(cfg.repetition_penalty), 1e-5)
    if temp == 0.0:
        # Greedy decoding must not inherit sampling filters from model defaults.
        top_k, top_p, min_p = -1, 1.0, 0.0

    # 兜底断言：任何一处超界都在**采样之前**报错并指出是哪个参数，
    # 而不是让 vLLM 在引擎深处抛异常把 EngineCore 打死（那样整个作业一起完）。
    if top_k != -1 and top_k < 1:
        raise ValueError(f"top_k 归一化失败：{top_k}")
    if not (0.0 <= temp):
        raise ValueError(f"temperature 必须 >= 0，得到 {temp}")
    if not (0.0 < rep):
        raise ValueError(f"repetition_penalty 必须 > 0，得到 {rep}")
    if not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p 必须落在 (0,1]，得到 {top_p}")
    if not (0.0 <= min_p <= 1.0):
        raise ValueError(f"min_p 必须落在 [0,1]，得到 {min_p}")
    return {"temperature": temp, "top_p": top_p, "top_k": top_k,
            "min_p": min_p, "repetition_penalty": rep}


class VLLMPolicy(HFPolicy):
    """`HFPolicy` 的 vLLM 版：分词面全复用，只有生成走连续批处理。"""

    def __init__(self, llm, tokenizer, enable_thinking: bool = False,
                 log_fn=None):
        super().__init__(None, tokenizer, device="cpu", enable_thinking=enable_thinking)
        self.llm = llm
        self._log = log_fn or (lambda _m: None)
        self._lora_req: Optional[Any] = None
        self.lora_version = 0

    # ------------------------------------------------------------- LoRA 管理
    def set_lora(self, adapter_name: str, adapter_dir: str) -> None:
        """挂载（或热替换）一个 adapter。每次调用都换新 int_id，强制 vLLM 重载。"""
        from vllm.lora.request import LoRARequest

        self.lora_version += 1
        self._lora_req = LoRARequest(adapter_name, self.lora_version, adapter_dir)
        self._log(f"[vllm] lora={adapter_name} version={self.lora_version} dir={adapter_dir}")

    def clear_lora(self) -> None:
        self._lora_req = None

    # ------------------------------------------------------------- 生成
    def generate_from_token_ids(self, prompt_ids_list: Sequence[Sequence[int]],
                                cfg: RolloutConfig, *, request_seeds: Optional[Sequence[int]] = None
                                ) -> Tuple[List[str], List[List[int]], List[int]]:
        if not prompt_ids_list:
            return [], [], []
        if cfg.stop_mode not in ("tool_call", "action_tag", "eos"):
            raise ValueError(
                f"VLLMPolicy supports stop_mode='tool_call', 'action_tag' or 'eos', got {cfg.stop_mode!r}")

        budget = int(cfg.max_context_tokens)
        # ★ 与 HF 版**同一道闸门**：绝不做"截断 prompt 重采样"这种 off-policy 操作。
        if any(len(ids) + cfg.max_new_tokens > budget for ids in prompt_ids_list):
            raise ValueError("EZ conversation exceeded hard context budget; refusing off-policy truncation")

        from vllm import SamplingParams

        # Qwen3 的 eos 可能是 int，也可能是 list（<|im_end|> + <|endoftext|>）。
        # HF 侧 `_generate_batch` 用 `eos_token_id=` 时能处理两种形态；这里必须显式归一，
        # 否则把 list 塞进 `stop_token_ids` 会变成嵌套 list 直接报错。
        eos = self.tokenizer.eos_token_id
        eos_ids = [int(eos)] if isinstance(eos, int) else [int(x) for x in (eos or [])]

        if request_seeds is not None and len(request_seeds) != len(prompt_ids_list):
            raise ValueError("request_seeds must match prompt batch length")
        base_params = dict(
            n=1,
            max_tokens=int(cfg.max_new_tokens),
            stop_token_ids=eos_ids or None,
            ignore_eos=False,
            # 只要 token id：解文本这一步我们自己做（停止判据本来就要逐前缀解码），
            # 让 vLLM 再解一遍纯属白烧 CPU。
            detokenize=False,
            logprobs=None,
            **normalize_sampling(cfg),
        )
        if cfg.stop_mode == "action_tag":
            # Match the quick evaluator: stop at the closing action tag and retain it.
            # vLLM needs text output enabled to match a stop string spanning tokens.
            base_params.update(stop=["</action>"], include_stop_str_in_output=True)
            base_params["detokenize"] = True
        sp = ([SamplingParams(**base_params, seed=int(seed)) for seed in request_seeds]
              if request_seeds is not None else SamplingParams(**base_params))
        prompts = [{"prompt_token_ids": [int(t) for t in ids]} for ids in prompt_ids_list]
        outs = self.llm.generate(prompts, sampling_params=sp,
                                 lora_request=self._lora_req, use_tqdm=False)

        pad_id = self.tokenizer.pad_token_id
        texts: List[str] = []
        gen_ids: List[List[int]] = []
        stops: List[int] = []
        for o in outs:
            raw = list(o.outputs[0].token_ids)
            # Keep the protocol's complete closing tag in returned text and IDs.
            if cfg.stop_mode == "tool_call":
                stop = toolcall_stop_len(self.tokenizer, raw, _TC_START_AFTER)
            elif cfg.stop_mode == "action_tag":
                # Native vLLM stop prevents spill generation; this guard trims any
                # backend-specific suffix while preserving the sampled close-tag IDs.
                stop = action_tag_stop_len(self.tokenizer, raw)
            else:
                stop = len(raw)
            ids = [int(t) for t in raw[:stop] if int(t) != pad_id]
            gen_ids.append(ids)
            stops.append(len(ids))
            texts.append(self.tokenizer.decode(ids, skip_special_tokens=True))
        self.last_finish_reasons = [getattr(o.outputs[0], "finish_reason", None) for o in outs]
        return texts, gen_ids, stops

    def generate(self, messages_list: Sequence[Sequence[Dict[str, str]]],
                 cfg: RolloutConfig) -> Tuple[List[str], List[List[int]], List[int]]:
        """message 入口：走与 HF 版同样的 truncate+render，再复用 token 入口。"""
        if not messages_list:
            return [], [], []
        ids = [self.initial_prompt_ids(m, cfg) for m in messages_list]
        return self.generate_from_token_ids(ids, cfg)


# --------------------------------------------------------------------------- 版本目录管理

def lora_workdir(run_dir: str, adapter_name: str, version: int,
                 rank: Optional[int] = None, keep: int = 2) -> str:
    """给某次 adapter 刷新分配一个目录，并**只保留最近 keep 份**。

    ⚠️ `rank` 必须给（8 卡训练时）：`run_dir` 是全 rank 共享的，多个 rank 同时
    `save_pretrained` 到同一路径会写出损坏的 safetensors —— 而且是**静默**的。

    为什么不能只留最新一份：vLLM 的 adapter 缓存按 `lora_int_id` 走 LRU，
    理论上不会回头读旧版本，但"文件已经没了 + 引擎因为某种原因要重读"会炸在
    训练步中间。多留一份的代价只有 ~0.26GB，换掉一个静默炸点很划算。
    """
    stem = os.path.join(run_dir, "vllm_lora")
    if rank is not None:
        stem = os.path.join(stem, f"rank{rank}")
    root = os.path.join(stem, adapter_name)
    out = os.path.join(root, f"v{version}")
    if os.path.isdir(root):
        olds = sorted((n for n in os.listdir(root)
                       if n.startswith("v") and n[1:].isdigit()
                       and os.path.isdir(os.path.join(root, n))),
                      key=lambda n: int(n[1:]))
        # out 稍后才写，所以磁盘上的旧版本里最多保留 keep-1 份。
        while len(olds) > max(0, keep - 1):
            shutil.rmtree(os.path.join(root, olds.pop(0)), ignore_errors=True)
    # ★ 目标目录先清空：`--resume` 之后 version 会从 1 重新开始，撞上上次运行留下的
    #   `v1`。不清的话可能把"上一版残留的 safetensors + 这一版新写的 config"混在一起，
    #   而 vLLM 加载残片是**静默**的（不报错，只是采样策略变了）。
    if os.path.isdir(out):
        shutil.rmtree(out, ignore_errors=True)
    return out
