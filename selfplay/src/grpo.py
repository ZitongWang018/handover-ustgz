"""Small shared runtime and GRPO math used by Explorer-Zero training."""
from __future__ import annotations

import inspect
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml

from grpo_norm import grpo_reduce

_LOGSOFTMAX_CHUNK = 512
def dist_info() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world, local

def setup_dist() -> Tuple[int, int, int]:
    rank, world, local = dist_info()
    if world > 1 and not torch.distributed.is_initialized():
        torch.cuda.set_device(local)
        # NCCL 默认 timeout 只有 10 分钟：某个 rank 卡在 rollout / 评测（其它 rank 在
        # barrier 干等）超过 10 分钟就会被 watchdog 判死直接 abort 整队。
        # ALFWorld 单步推理最长实测要好几分钟，统一放宽到 30 分钟。
        dist.init_process_group(backend="nccl", rank=rank, world_size=world,
                                timeout=timedelta(minutes=30))
    return rank, world, local

def is_main(rank: int) -> bool:
    return rank == 0

def log0(rank: int, msg: str) -> None:
    if is_main(rank):
        print(msg, flush=True)

def set_seed(seed: int, rank: int) -> None:
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_cfg(path: str) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    return cfg

def build_model_and_tokenizer(cfg: dict, local_rank: int, world: int,
                              wrap_ddp: bool = False):
    """Load the explicit local base without Hub or custom-code fallback."""
    if wrap_ddp:
        raise ValueError("Explorer-Zero synchronizes adapter gradients explicitly; DDP wrapping is disabled")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    mcfg = cfg["model"]
    model_path = Path(os.path.expandvars(os.path.expanduser(str(mcfg["path"])))).resolve()
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"local model directory is missing config.json: {model_path}")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}
    if mcfg.get("dtype") not in dtype_map:
        raise ValueError("unsupported explicit model dtype")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "right"
    load_kwargs = {}
    if cfg.get("runtime", {}).get("memory_profile") == "qwen3_moe_cpu" and torch.cuda.is_available():
        load_kwargs.update(device_map={"": torch.device("cuda", local_rank)}, low_cpu_mem_usage=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False,
        torch_dtype=dtype_map[mcfg["dtype"]],
        attn_implementation=mcfg["attn_implementation"], **load_kwargs)
    model.config.use_cache = False
    if mcfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda", local_rank))
    return model, tokenizer

def _autocast_bf16(model):
    """Use bf16 autocast for fp32 CUDA weights without retaining cast copies."""
    try:
        dt = next(model.parameters()).dtype
    except StopIteration:
        return nullcontext()
    if not torch.cuda.is_available() or dt != torch.float32:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, cache_enabled=False)

def per_token_logps(model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                    chunk: int = _LOGSOFTMAX_CHUNK,
                    loss_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """返回**每个位置**的 log p(target_token)。

    数学上就是 `logits.log_softmax(-1).gather(targets)`，但把 softmax 沿序列维分块算，
    避免一次性物化 fp32 的 B×W×vocab 中间张量（这是长上下文 RL 最典型的显存杀手）。
    """
    targets = input_ids[:, 1:]
    selected_positions = None
    model_kwargs = dict(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if loss_mask is not None:
        # logit position t predicts input token t+1. Select the union of required
        # positions across rows; masked positions are zeroed after gathering.
        positions = torch.nonzero(loss_mask[:, 1:].bool().any(dim=0), as_tuple=False).flatten()
        try:
            supports_select = "logits_to_keep" in inspect.signature(base.forward).parameters
        except (TypeError, ValueError):
            supports_select = False
        if supports_select:
            selected_positions = positions
            model_kwargs["logits_to_keep"] = selected_positions
    if loss_mask is not None and selected_positions is not None and selected_positions.numel() == 0:
        # Zero-mask batch: preserve a zero graph via a tiny logits probe at position 0.
        model_kwargs["logits_to_keep"] = torch.zeros((1,), dtype=torch.long, device=input_ids.device)
        selected_positions = model_kwargs["logits_to_keep"]
    with _autocast_bf16(model):
        out = model(**model_kwargs)
    logits = out.logits
    if selected_positions is None:
        logits = logits[:, :-1, :]
        positions = None
    else:
        positions = selected_positions
    width = targets.size(1)
    if positions is not None:
        selected_targets = targets.index_select(1, positions)
        selected_mask = loss_mask[:, 1:].index_select(1, positions).to(dtype=torch.float32)
        parts = []
        for s in range(0, logits.size(1), chunk):
            lc = logits[:, s:s + chunk, :]
            tc = selected_targets[:, s:s + chunk]
            if torch.is_grad_enabled():
                parts.append(torch.utils.checkpoint.checkpoint(_chunk_logp, lc, tc, use_reentrant=False))
            else:
                parts.append(_chunk_logp(lc, tc))
        selected_logps = torch.cat(parts, dim=1) if parts else logits.sum(dim=-1) * 0.0
        selected_logps = selected_logps * selected_mask
        dense = torch.zeros_like(targets, dtype=selected_logps.dtype)
        dense.index_copy_(1, positions, selected_logps)
        return dense
    if width <= chunk:
        return _chunk_logp(logits, targets)
    parts: List[torch.Tensor] = []
    for s in range(0, width, chunk):
        lc = logits[:, s:s + chunk, :]
        tc = targets[:, s:s + chunk]
        # ⚠️ 光"分块"其实**不省显存**：autograd 会把每块的 fp32 中间结果都留到 backward，
        #    累加起来 = 一次性物化 W×V×4。真正省的办法是对每块套 activation checkpoint，
        #    让 fp32 中间结果在反向时重算 —— 只保留 bf16 输入（本来就在图里）。
        #    W=8192 时峰值从 ~7.5G 降到 ~2.5G，对 fp32 参数（静态已 64G）很关键。
        if torch.is_grad_enabled():
            parts.append(torch.utils.checkpoint.checkpoint(
                _chunk_logp, lc, tc, use_reentrant=False))
        else:
            parts.append(_chunk_logp(lc, tc))
    return torch.cat(parts, dim=1)

def _chunk_logp(logits_chunk: "torch.Tensor", targets_chunk: "torch.Tensor"):
    """对一块 logits 算 log p(target)。输入保持 bf16，`.float()` 放在函数内部，
    这样套 checkpoint 时只需要保存 bf16 输入。"""
    lg = logits_chunk.float()
    return lg.log_softmax(dim=-1).gather(-1, targets_chunk.unsqueeze(-1)).squeeze(-1)

@dataclass
class BatchTensors:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor
    advantages: torch.Tensor
    n_active: int

def make_batch(trajs: Sequence[Trajectory], pad_id: int, device, max_len: Optional[int] = None) -> BatchTensors:
    lens = [len(t.input_ids) for t in trajs]
    width = max(lens) if max_len is None else min(max(lens), max_len)
    width = max(width, 2)
    n = len(trajs)
    ids = torch.full((n, width), pad_id, dtype=torch.long)
    attn = torch.zeros((n, width), dtype=torch.long)
    lmask = torch.zeros((n, width), dtype=torch.float32)
    adv = torch.zeros(n, dtype=torch.float32)
    for i, t in enumerate(trajs):
        L = min(len(t.input_ids), width)
        ids[i, :L] = torch.tensor(t.input_ids[:L], dtype=torch.long)
        attn[i, :L] = 1
        lmask[i, :L] = torch.tensor(t.loss_mask[:L], dtype=torch.float32)
        adv[i] = t.advantage
    return BatchTensors(
        input_ids=ids.to(device),
        attention_mask=attn.to(device),
        loss_mask=lmask.to(device),
        advantages=adv.to(device),
        # ⚠️ 分母口径必须和 `grpo_loss` 里的 `mask = loss_mask[:, 1:]` 一致：
        #    logits[:, t] 预测 token t+1，所以最后一个位置的 loss_mask 不参与。
        #    用全量 lmask.sum() 会让 micro-batch 的权重 `weight = n_active/total_active`
        #    略微偏大（每条轨迹多算 1 个位置），虽然数值上几乎看不出来，
        #    但两个口径混用就是埋雷。
        n_active=int(lmask[:, 1:].sum().item()),
    )

def grpo_loss(
    model,
    batch: BatchTensors,
    old_logps: torch.Tensor,
    clip_eps: float,
    loss_norm: str = "seq",
    ref_logps: Optional[torch.Tensor] = None,
    beta: float = 0.0,
    sequence_reduction: str = "mean",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute clipped sequence-level policy loss and optional reference KL."""
    new_logps = per_token_logps(model, batch.input_ids, batch.attention_mask,
                                loss_mask=batch.loss_mask)
    # 对齐 mask：logp[:, i] 预测的是 input_ids[:, i+1]
    mask = batch.loss_mask[:, 1:]
    adv = batch.advantages.unsqueeze(1)
    # Selected-logit path writes zero at masked positions, while legacy old_logps
    # may be dense. Zero the log-ratio before exp so masked entries stay finite.
    log_ratio = torch.where(mask.bool(), new_logps - old_logps,
                            torch.zeros_like(new_logps))
    ratio = torch.exp(log_ratio)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    per_tok = -torch.min(surr1, surr2)
    pg_per_tok = per_tok
    kl_per_tok = torch.zeros_like(per_tok)
    if beta and ref_logps is None:
        raise ValueError("beta > 0 requires frozen-reference log probabilities")
    if beta and ref_logps is not None:
        # Non-negative k3 estimate of KL(policy || frozen base), evaluated on
        # exactly the same generated-token mask as the policy objective.
        delta = torch.where(mask.bool(), ref_logps - new_logps,
                            torch.zeros_like(new_logps))
        kl_per_tok = torch.exp(delta) - delta - 1.0
        per_tok = per_tok + float(beta) * kl_per_tok
    denom = mask.sum().clamp(min=1.0)
    # 归约口径单独放在 `grpo_norm.grpo_reduce`（只依赖 torch），
    # 这样离线单测能直接测到**跑的这份代码**，而不是测一份复刻实现。
    loss = grpo_reduce(per_tok, mask, loss_norm=loss_norm,
                       sequence_reduction=sequence_reduction)

    with torch.no_grad():
        clipped = ((ratio - 1.0).abs() > clip_eps).float()
        valid_sequences = int((mask.sum(dim=1) > 0).sum().item())
        pg_sequence_sum = grpo_reduce(pg_per_tok.detach(), mask, loss_norm=loss_norm,
                                      sequence_reduction="sum")
        kl_sequence_sum = grpo_reduce(kl_per_tok.detach(), mask, loss_norm=loss_norm,
                                      sequence_reduction="sum") if beta else torch.zeros((), device=mask.device)
        objective_sequence_sum = pg_sequence_sum + float(beta) * kl_sequence_sum
        stats = {
            # ★ 这个式子看着像随手写的残差，其实是标准的 **k3 KL 估计**
            #   （Schulman）：令 Δ = new - old，则
            #       E_old[ exp(Δ) - 1 - Δ ] = Σ_old·(new/old) - 1 - E_old[Δ]
            #                               = 1 - 1 - E_old[Δ] = KL(old ‖ new)
            #   一阶项 -Δ 就是 k1，k3 比 k1 多了个非负的二阶项，方差更低且无偏。
            #   别"顺手改成 (old - new).mean()" —— 那退化成 k1 了。
            "kl_approx": float((((ratio - 1.0) - log_ratio) * mask).sum().item() / max(denom.item(), 1.0)) if mask.sum() > 0 else 0.0,
            "clip_frac": float((clipped * mask).sum().item() / max(denom.item(), 1.0)),
            "ratio_mean": float(((ratio * mask).sum() / denom).item()),
            "kl_ref": float(((torch.exp(torch.where(mask.bool(), ref_logps - new_logps, torch.zeros_like(new_logps))) - torch.where(mask.bool(), ref_logps - new_logps, torch.zeros_like(new_logps)) - 1.0) * mask).sum().item() / max(denom.item(), 1.0)) if ref_logps is not None else 0.0,
            # Exact no-grad decomposition using the same masked seq reduction as loss.
            "pg_sequence_sum": float(pg_sequence_sum.item()),
            "kl_sequence_sum": float(kl_sequence_sum.item()),
            "objective_sequence_sum": float(objective_sequence_sum.item()),
            "valid_sequence_count": valid_sequences,
        }
    return loss, stats
