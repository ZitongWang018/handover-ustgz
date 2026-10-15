"""Explorer-Zero training loop under the shared SkillRise ALFWorld protocol."""
from __future__ import annotations

import argparse
import atexit
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alfworld_compat import install as install_alfworld_compat  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from alfworld_env import ALFWorldBatchEnv, collect_game_files, load_alfworld_config  # noqa: E402
from ez_proposer import ProposerConfig, ProposerRollout, run_proposer_exploration  # noqa: E402
from ez_solver import EzSolverTrajectory, run_ez_solver_rollouts  # noqa: E402
from ez_state import StateSnapshot  # noqa: E402
from ez_verify import ProposedTask, proposer_reward, validity_filter  # noqa: E402
from ez_dist import (broadcast_pool_indices, check_adapter_sync, gather_group_values,
                     broadcast_adapter_parameters, global_group_advantages, local_group_size, select_adapter,
                     sync_active_adapter_gradients)  # noqa: E402
from rollout import HFPolicy, RolloutConfig  # noqa: E402
from grpo import (build_model_and_tokenizer, grpo_loss,  # noqa: E402
                        is_main, load_cfg, log0, make_batch, per_token_logps,
                        set_seed, setup_dist)


_LOG_FH = None    # 本 rank 的日志：train_log.rank{rank}.jsonl
_LOG_FH0 = None   # 仅 rank0：额外的 train_log.jsonl（让现成的判读脚本不用改）
RANK = 0
_REPORTER: Any = None


def log(msg: str) -> None:
    """同时落 stdout 和日志文件。

    ★ DDP 下 stdout 只有 rank0 会打（`log0`），但**每个 rank 都必须落自己的文件**：
    否则某个 rank 静默挂掉时，从 rank0 的日志里完全看不出来（8 卡验收第 3 条）。
    rank0 额外再写一份 `train_log.jsonl`，保持 `ez_check_smoke_ok.py` 无需改动。
    """
    log0(_RANK, msg)
    rec = json.dumps({"t": time.time(), "rank": _RANK, "msg": msg},
                     ensure_ascii=False) + "\n"
    for fh in (_LOG_FH, _LOG_FH0):
        if fh is not None:
            fh.write(rec)
            fh.flush()


class _MetricsReporter:
    """Write finite rank-zero scalar metrics to a local JSONL file."""

    def __init__(self, cfg: dict, run_dir: Path, rank: int) -> None:
        self._rank = rank
        self._fh = None
        if rank != 0:
            return
        self._fh = open(Path(run_dir) / "metrics.jsonl", "a", encoding="utf-8", newline="\n")

    def log(self, metrics: dict, step: int) -> None:
        if self._rank != 0:
            return
        clean = {}
        for key, value in metrics.items():
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(scalar):
                clean[str(key)] = scalar
        if self._fh is not None:
            self._fh.write(json.dumps({"step": int(step), "metrics": clean},
                                      sort_keys=True, allow_nan=False) + "\n")
            self._fh.flush()

    def finish(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()


def _cuda_sync_and_reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _cuda_peak_gib():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (torch.cuda.max_memory_allocated() / (1024 ** 3),
            torch.cuda.max_memory_reserved() / (1024 ** 3))


def _attention_backend(model, cfg):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return getattr(getattr(base, "config", None), "_attn_implementation",
                   cfg["model"].get("attn_implementation", "default")) or "default"


# --------------------------------------------------------------------------- LoRA


def build_dual_lora(model, cfg: dict, resume_checkpoint: Optional[Path] = None):
    """proposer 和 solver 是**两个独立策略**（论文 §3：同一基座初始化、各自训练）。

    实现：共享基座 + peft 两个 LoRA adapter（"proposer" / "solver"），
    相位切换用 `model.set_adapter(name)`。全参训练需两份完整权重，
    4B fp32 显存放不下 —— 这正是论文用 LoRA 的原因之一。
    """
    try:
        from peft import LoraConfig, PeftModel, get_peft_model
    except ImportError as exc:
        raise SystemExit("Explorer-Zero 需要双策略 LoRA：pip install peft") from exc
    lc = LoraConfig(r=cfg["model"].get("lora_r", 32),
                    lora_alpha=cfg["model"].get("lora_alpha", 32),
                    lora_dropout=0.0,
                    target_modules=cfg["model"].get("lora_targets", "all-linear"))
    if resume_checkpoint is None:
        model = get_peft_model(model, lc, adapter_name="proposer")
        model.add_adapter("solver", lc)
    else:
        proposer_path = resume_checkpoint / "proposer"
        solver_path = resume_checkpoint / "solver"
        if not (proposer_path / "adapter_config.json").is_file() or not (solver_path / "adapter_config.json").is_file():
            raise ValueError(f"checkpoint {resume_checkpoint} is missing proposer/solver adapter files")
        model = PeftModel.from_pretrained(model, proposer_path, adapter_name="proposer",
                                          is_trainable=True)
        model.load_adapter(solver_path, adapter_name="solver", is_trainable=True)
    model.set_adapter("proposer")
    # ★ adapter 参数必须 fp32：基座可以是 bf16（冻结，不参与更新，无舍入问题），
    #   但被更新的 adapter 要是 bf16，lr=1e-4 的更新同样会被 bf16 ulp 抹掉。
    n_cast = 0
    for name, p in model.named_parameters():
        if "lora_" in name and p.dtype != torch.float32:
            p.data = p.data.float()
            n_cast += 1
    if n_cast:
        print(f"[ez] cast {n_cast} 个 LoRA 张量到 fp32")
    # 两个 adapter 都先可训练，随后 select_adapter 在每个 phase 明确切换；EZ
    # 不包 DDP，而由 ez_dist 做固定顺序的手动梯度 all-reduce。
    n_tr = 0
    for _n, p in model.named_parameters():
        if "lora_" in _n and not p.requires_grad:
            p.requires_grad_(True)
            n_tr += 1
    if n_tr:
        print(f"[ez] 启用 {n_tr} 个 LoRA 张量（由 select_adapter 管理 phase trainability）")
    model.print_trainable_parameters()
    return model


# --------------------------------------------------------------------------- proposer 训练序列


def tokenize_proposer_rollout(ro: ProposerRollout) -> ProposerRollout:
    """Use the exact token-native history on which each action was sampled."""
    ids = list(getattr(ro, "training_ids", ()))
    mask = list(getattr(ro, "training_mask", ()))
    if not ids or len(ids) != len(mask):
        raise ValueError("proposer rollout is missing a well-formed token-native training history")
    if not any(mask):
        raise ValueError("proposer rollout has no generated tokens to train")
    ro.__dict__["input_ids"] = ids
    ro.__dict__["loss_mask"] = mask
    return ro


def update_probe(model, opt_params, optimizer, tag: str) -> float:
    """Report displacement across active trainable adapter parameters."""
    probes = [p for p in opt_params if p.grad is not None and p.requires_grad]
    if not probes:
        log(f"[{tag}] update_rel=0（没有任何参数带 grad，优化器无事可做）")
        return 0.0
    _nm = {id(p): n for n, p in model.named_parameters()}
    before = [p.detach().clone() for p in probes]
    optimizer.step()
    rows = []          # (rel, |Δp|, ‖p‖, name)
    with torch.no_grad():
        for p, b in zip(probes, before):
            d = float((p.detach() - b).norm())
            n = float(p.detach().norm())
            rows.append((d / (n + d) if (n + d) > 0 else 0.0, d, n, _nm.get(id(p), "?")))
    del before
    n_moved = sum(1 for r in rows if r[1] > 0.0)
    rows.sort(key=lambda r: -r[0])
    rel = rows[0][0] if rows else 0.0
    _dmax = max((r[1] for r in rows), default=0.0)
    log(f"[{tag}] update_rel={rel:.3e}"
        f"（=max ‖Δp‖/(‖p‖+‖Δp‖)；{n_moved}/{len(probes)} 张量真的动了；max‖Δp‖={_dmax:.3e}）")
    log(f"[{tag}] probe top3 -> " + " | ".join(
        f"{r[3][-42:]} rel={r[0]:.2e} |Δ|={r[1]:.2e} ‖p‖={r[2]:.2e}" for r in rows[:3]))
    return rel


def ez_save(model, tokenizer, optimizer, run_dir: Path, tag: str, cfg: dict,
            extra: dict, task_pool: list, rank: int, world: int,
            uid_counter: int = 0, dry_run: bool = False,
            protocol_manifest: Optional[dict] = None) -> None:
    if protocol_manifest is None:
        raise ValueError("a resolved training contract is required for every checkpoint")
    ez_save_v3(model, tokenizer, optimizer, run_dir, tag, cfg, extra, task_pool,
               rank, world, uid_counter, dry_run, protocol_manifest)


def ez_save_v3(model, tokenizer, optimizer, run_dir: Path, tag: str, cfg: dict,
               extra: dict, task_pool: list, rank: int, world: int,
               uid_counter: int, dry_run: bool, protocol_manifest: dict) -> None:
    """Seal a v3 checkpoint only after every rank has written its RNG state."""
    import shutil
    from checkpoint import (ContractError, seal_checkpoint,
                            validate_checkpoint_v3, validate_sealed_integrity)

    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "ckpt"
    tmp_dir = run_dir / "ckpt.tmp"
    backup_dir = run_dir / "ckpt.previous"

    def status_from_rank0(ok: bool, message: str = "") -> None:
        packet = [{"ok": ok, "message": message}]
        if world > 1:
            dist.broadcast_object_list(packet, src=0)
        if not packet[0]["ok"]:
            raise RuntimeError(f"v3 checkpoint {tag} failed: {packet[0]['message']}")

    if rank == 0:
        try:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True, exist_ok=False)
            active_adapter = getattr(model, "active_adapter", "proposer")
            model.save_pretrained(tmp_dir, selected_adapters=["proposer", "solver"])
            model.set_adapter(active_adapter if isinstance(active_adapter, str) else "proposer")
            tokenizer.save_pretrained(tmp_dir / "tokenizer")
            with (tmp_dir / "training_manifest.json").open("w", encoding="utf-8", newline="\n") as fh:
                json.dump(protocol_manifest, fh, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
                fh.write("\n")
            torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
            with (tmp_dir / "task_pool.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
                for task in task_pool:
                    fh.write(json.dumps(task.to_json(), ensure_ascii=False, sort_keys=True) + "\n")
            next_position = {
                "outer": int(extra["next_outer"]),
                "phase": str(extra["next_phase"]),
                "step": int(extra["next_step"]),
            }
            state = {
                "checkpoint_schema_version": 3,
                "tag": tag,
                "contract_sha256": protocol_manifest["contract_sha256"],
                "world_size": int(world),
                "dry_run": bool(dry_run),
                "next_position": next_position,
                "next_outer": next_position["outer"],
                "next_phase": next_position["phase"],
                "next_step": next_position["step"],
                "run_id": Path(run_dir).name.removeprefix(
                    str(protocol_manifest.get("resolved_config", {}).get("run", {}).get("name", "")) + "-"),
                "rng_policy": "python_numpy_torch_per_rank_saved; vllm_sampling_state_not_bitwise",
            }
            with (tmp_dir / "train_state.json").open("w", encoding="utf-8", newline="\n") as fh:
                json.dump(state, fh, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
                fh.write("\n")
            (tmp_dir / "rank_state").mkdir()
            status_from_rank0(True)
        except Exception as exc:  # ensure every rank exits instead of waiting in a barrier
            status_from_rank0(False, f"{type(exc).__name__}: {exc}")
            raise
    else:
        status_from_rank0(True)

    local_error = None
    try:
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (torch.cuda.get_rng_state(torch.cuda.current_device())
                           if torch.cuda.is_available() else None),
            "uid_counter": int(uid_counter),
        }
        torch.save(rng_state, tmp_dir / "rank_state" / f"rank{rank}.pt")
    except Exception as exc:  # noqa: BLE001
        local_error = f"rank {rank}: {type(exc).__name__}: {exc}"
    errors: list[Any] = [local_error]
    if world > 1:
        errors = [None] * world
        dist.all_gather_object(errors, local_error)
    failures = [item for item in errors if item]
    if failures:
        raise RuntimeError("v3 rank-local checkpoint write failed: " + "; ".join(failures))

    if rank == 0:
        try:
            seal_checkpoint(tmp_dir, protocol_manifest["contract_sha256"])
            if dry_run:
                validate_sealed_integrity(tmp_dir)
            else:
                validate_checkpoint_v3(tmp_dir, expected_contract=protocol_manifest, purpose="resume")
            # Keep the previous complete checkpoint until the new sealed directory is
            # ready; if the final rename fails, restore it without a missing-checkpoint gap.
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            if ckpt_dir.exists():
                os.replace(ckpt_dir, backup_dir)
            try:
                os.replace(tmp_dir, ckpt_dir)
            except Exception:
                if backup_dir.exists() and not ckpt_dir.exists():
                    os.replace(backup_dir, ckpt_dir)
                raise
        except Exception as exc:  # noqa: BLE001
            status_from_rank0(False, f"{type(exc).__name__}: {exc}")
            raise
        status_from_rank0(True)
    else:
        status_from_rank0(True)


def load_ez_resume(ckpt_dir: Path, cfg: dict, world: int, rank: int,
                   protocol_manifest: Optional[dict] = None):
    with open(ckpt_dir / "train_state.json", encoding="utf-8") as fh:
        state = json.load(fh)
    from checkpoint import ContractError, validate_checkpoint_v3
    try:
        validate_checkpoint_v3(ckpt_dir, expected_contract=protocol_manifest, purpose="resume")
    except ContractError as exc:
        raise ValueError(str(exc)) from None
    if state.get("dry_run", False):
        raise ValueError("smoke/dry-run checkpoints cannot be resumed as formal training")
    pool_path = ckpt_dir / "task_pool.jsonl"
    if not pool_path.is_file():
        raise ValueError("resume checkpoint is missing task_pool.jsonl")
    pool = []
    with open(pool_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                if int(row.get("schema_version", 1)) < 2:
                    raise ValueError("task pool schema is missing raw fact identity; refusing resume")
                task = ProposedTask.from_json(row)
                for snap in (task.s0, task.ref_final, *(s.ref_snapshot for s in task.subtasks)):
                    if snap.mode == "facts" and snap.facts and not snap.raw_facts:
                        raise ValueError("schema-2 task is missing raw facts; refusing unsafe resume")
                pool.append(task)
    rng_path = ckpt_dir / "rank_state" / f"rank{rank}.pt"
    if not rng_path.is_file():
        raise ValueError(f"resume checkpoint is missing rank-specific RNG state: {rng_path}")
    return state, pool, rng_path


def restore_ez_optimizer_rng(optimizer, optimizer_path: Path, rng_path: Path) -> int:
    """Restore the exact saved optimizer and rank-local Python/NumPy/Torch RNG state."""
    optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu", weights_only=False))
    rng_state = torch.load(rng_path, map_location="cpu", weights_only=False)
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch_cpu"])
    if torch.cuda.is_available() and rng_state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state(rng_state["torch_cuda"], torch.cuda.current_device())
    return int(rng_state.get("uid_counter", 0))


def grad_chain_selfcheck(model, tokenizer, device) -> Dict[str, float]:
    """启动自检：证明**两个 adapter 各自都真的能接到梯度**。

    这条自检刻意不依赖环境、不依赖奖励、不依赖模型能不能完成任务，几秒钟跑完。
    它回答的是一个和"奖励有没有区分度"**完全无关**的问题：

        前向 -> 该 adapter 的 LoRA 参数 -> loss -> 反向 -> p.grad 非空 ？

    为什么必须单独测：双 adapter + `set_adapter` 的隔离语义、peft 的包装、
    gradient_checkpointing、以及 adapter 切换后的 requires_grad 状态，
    都可能让**某一个 adapter 静默收不到梯度** —— 现象就是日志里 update_rel=0，
    但你无法区分"梯度链断了"和"这组奖励恰好全相同"。
    有了这条自检，两者就能一刀切开。

    非破坏性：只 backward，不 optimizer.step()，不改任何权重。
    """
    out: Dict[str, float] = {}
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device=device)
    attn = torch.ones_like(ids)
    for name in ("proposer", "solver"):
        try:
            select_adapter(model, name)
        except Exception as exc:  # noqa: BLE001
            log(f"[selfcheck] 切到 adapter {name!r} 失败：{exc}")
            out[f"{name}_grad_norm"] = float("nan")
            continue
        model.zero_grad(set_to_none=True)
        logits = model(input_ids=ids, attention_mask=attn).logits
        logits.float().pow(2).mean().backward()
        tot, n_grad = 0.0, 0
        for n, p in model.named_parameters():
            if "lora_" in n and name in n and p.grad is not None:
                n_grad += 1
                tot += float(p.grad.detach().float().norm())
        out[f"{name}_grad_norm"] = tot
        out[f"{name}_params_with_grad"] = float(n_grad)
    model.zero_grad(set_to_none=True)
    select_adapter(model, "proposer")
    return out


def backward_microbatches(raw, optimizer_model, trajectories, pad_id, device, *,
                          adapter: str, clip_eps: float, micro_batch_size: int,
                          world: int, loss_norm: str, beta: float, dist_obj=None,
                          group_weight: float = 1.0):
    """Accumulate exact seq-mean gradients over all global task groups in one step."""
    from grpo import BatchTensors, grpo_loss, make_batch, per_token_logps
    if loss_norm != "seq":
        raise ValueError("EZ microbatch accumulation currently requires train.loss_norm='seq'")
    mb = max(1, int(micro_batch_size))
    is_segmented = lambda t: bool(getattr(t, "segments", None))
    local_valid = sum(1 for t in trajectories if (
        bool(getattr(t, "has_trainable_tokens", False)) if is_segmented(t)
        else any(t.loss_mask[1:])))
    global_valid = torch.tensor([local_valid], dtype=torch.float64, device=device)
    if world > 1:
        dist_obj.all_reduce(global_valid, op=dist_obj.ReduceOp.SUM)
    denom = max(float(global_valid.item()), 1.0)
    loss_sum = 0.0
    stats = []
    for start in range(0, len(trajectories), mb):
        for trajectory in trajectories[start:start + mb]:
            if is_segmented(trajectory):
                segments = list(getattr(trajectory, "segments", ()))
                episode_tokens = sum(len(response) for _, response in segments)
                row = {"pg_sequence_sum": 0.0, "kl_sequence_sum": 0.0,
                       "objective_sequence_sum": 0.0, "valid_sequence_count": 0.0}
                for prompt_ids, response_ids in segments:
                    prompt_ids, response_ids = list(prompt_ids), list(response_ids)
                    if not response_ids:
                        continue
                    ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=device)
                    attention = torch.ones_like(ids)
                    mask = torch.zeros_like(ids, dtype=torch.float32)
                    mask[:, len(prompt_ids):] = 1.0
                    bt = BatchTensors(ids, attention, mask,
                                      torch.tensor([trajectory.advantage], dtype=torch.float32, device=device),
                                      len(response_ids))
                    with torch.no_grad():
                        old = per_token_logps(optimizer_model, ids, attention, loss_mask=mask)
                        ref = None
                        if beta:
                            with raw.disable_adapter():
                                ref = per_token_logps(raw, ids, attention, loss_mask=mask)
                    loss, stat = grpo_loss(optimizer_model, bt, old, clip_eps, loss_norm="seq",
                                           ref_logps=ref, beta=beta, sequence_reduction="sum")
                    weight = len(response_ids) / max(episode_tokens, 1)
                    (loss * weight * (world / denom) * group_weight).backward()
                    loss_sum += float(loss.detach().item()) * weight * group_weight
                    for key in ("pg_sequence_sum", "kl_sequence_sum", "objective_sequence_sum"):
                        row[key] += float(stat.get(key, 0.0)) * weight
                    del old, ref, bt, loss
                if episode_tokens:
                    row["valid_sequence_count"] = 1.0
                stats.append(row)
            else:
                bt = make_batch([trajectory], pad_id, device)
                with torch.no_grad():
                    old = per_token_logps(optimizer_model, bt.input_ids, bt.attention_mask,
                                          loss_mask=bt.loss_mask)
                    ref = None
                    if beta:
                        with raw.disable_adapter():
                            ref = per_token_logps(raw, bt.input_ids, bt.attention_mask,
                                                  loss_mask=bt.loss_mask)
                loss, stat = grpo_loss(optimizer_model, bt, old, clip_eps, loss_norm="seq",
                                       ref_logps=ref, beta=beta, sequence_reduction="sum")
                (loss * (world / denom) * group_weight).backward()
                loss_sum += float(loss.detach().item()) * group_weight
                stats.append(stat)
                del old, ref, bt, loss
    if not trajectories:
        raise RuntimeError("optimizer step requires at least one local trajectory")
    return loss_sum / denom, stats, int(global_valid.item())


def build_ez_rollout_configs(cfg: dict, local_group_size: int,
                             proposer_turns: int | None = None,
                             solver_turns: int | None = None):
    """Construct the exact proposer/solver generation configs used by EZ training."""
    p, s, train = cfg["proposer"], cfg["solver"], cfg["train"]
    p_cfg = ProposerConfig(
        group_size=local_group_size,
        max_turns=proposer_turns if proposer_turns is not None else p["max_turns"],
        max_new_tokens=p["max_new_tokens"], temperature=p["temperature"],
        top_p=p["top_p"], top_k=p["top_k"], min_p=p["min_p"],
        repetition_penalty=p["repetition_penalty"],
        max_context_tokens=p["max_context_tokens"],
        keep_recent_turns=p["keep_recent_turns"], seed=p["seed"],
        give_admissible=p["give_admissible"],
        finish_nudge=p["finish_nudge"],
        generation_max_batch_size=train["generation_max_batch_size"],
    )
    s_cfg = RolloutConfig(
        group_size=1,
        max_steps=solver_turns if solver_turns is not None else s["max_turns"],
        max_new_tokens=s["max_new_tokens"], temperature=s["temperature"],
        top_p=s["top_p"], top_k=s["top_k"], min_p=s["min_p"],
        repetition_penalty=s["repetition_penalty"],
        max_context_tokens=s["max_context_tokens"],
        keep_recent_turns=s["keep_recent_turns"], seed=s["seed"],
        stop_mode=s["stop_mode"], max_batch_size=train["generation_max_batch_size"],
    )
    return p_cfg, s_cfg


def vllm_engine_kwargs(cfg: dict, rank: int) -> dict:
    """Build vLLM settings from the resolved shared protocol."""
    vcfg = cfg["runtime"]["vllm"]
    return {"max_model_len": int(vcfg["max_model_len"]),
            "max_lora_rank": int(vcfg["max_lora_rank"]),
            "max_num_seqs": int(vcfg["max_num_seqs"]),
            "max_num_batched_tokens": int(vcfg["max_num_batched_tokens"]),
            "gpu_memory_utilization": float(vcfg["gpu_memory_utilization"]),
            "enable_prefix_caching": bool(vcfg["enable_prefix_caching"]),
            "enforce_eager": bool(vcfg["enforce_eager"]),
            "seed": int(vcfg["seed"]) + int(rank),
            "trust_remote_code": False}


# --------------------------------------------------------------------------- 主循环


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Explorer-Zero training runtime")
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--protocol-manifest", required=True)
    ap.add_argument("--protocol-smoke", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_cfg(args.config)
    with open(args.protocol_manifest, encoding="utf-8") as fh:
        protocol_manifest = json.load(fh)
    from checkpoint import ContractError, contract_signature
    if protocol_manifest.get("contract_sha256") != contract_signature(protocol_manifest):
        ap.error("training manifest checksum is invalid")
    if protocol_manifest.get("schema_version") != 3 or \
            protocol_manifest.get("protocol_name") != "explorer_zero_skillrise":
        ap.error("manifest must use the shared SkillRise schema-v3 contract")
    if cfg != protocol_manifest.get("resolved_config"):
        ap.error("runtime config differs from the resolved manifest")
    if bool(args.protocol_smoke) != (protocol_manifest.get("checkpoint_eligibility") == "smoke_only"):
        ap.error("smoke flag differs from manifest checkpoint eligibility")
    if args.resume and args.protocol_smoke:
        ap.error("smoke training cannot resume")
    install_alfworld_compat()
    os.environ["ALFWORLD_DATA"] = str(cfg["env"]["data_root"])
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    seed = cfg["run"].get("seed", 42)
    # ★ 必须用 setup_dist()（会 init_process_group），不能用 dist_info()：
    #   dist_info() 只读环境变量，从不建进程组 —— 8 进程照跑，但 all_reduce /
    #   broadcast 全部抛 "Default process group has not been initialized"。
    rank, world, local = setup_dist()
    global _RANK
    _RANK = rank
    set_seed(seed, rank)

    hashes = [None] * world
    if world > 1:
        dist.all_gather_object(hashes, protocol_manifest.get("contract_sha256"))
    else:
        hashes = [protocol_manifest.get("contract_sha256")]
    if any(item != protocol_manifest.get("contract_sha256") for item in hashes):
        raise ValueError("rank-local training manifests disagree before model load")

    resume_ckpt = Path(args.resume).expanduser().resolve() if args.resume else None
    if resume_ckpt is not None:
        run_dir = resume_ckpt.parent
    elif is_main(rank):
        run_dir = Path(cfg["run"]["output_dir"]) / f"{cfg['run']['name']}-{args.run_id}"
    else:
        run_dir = Path(".")          # 占位，马上被广播覆盖
    if world > 1:
        # ★ 各 rank 不能各自取时间命名：跨分钟边界时 rank0 与 rank7 会落到**两个目录**，
        #   表现为"rank0 有日志、rank7 的文件不存在"，很难查。统一由 rank0 决定。
        _rd = [str(run_dir)]
        dist.broadcast_object_list(_rd, src=0)
        run_dir = Path(_rd[0])
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol_file = run_dir / "run_protocol.json"
    if not protocol_file.is_file():
        raise ValueError("run directory is missing its immutable run_protocol.json")
    with protocol_file.open(encoding="utf-8") as fh:
        run_protocol = json.load(fh)
    if run_protocol.get("contract_sha256") != protocol_manifest.get("contract_sha256") or \
            run_protocol.get("run_id") != args.run_id:
        raise ValueError("run id/contract does not match immutable run_protocol.json")
    task_pool_path = run_dir / "task_pool.jsonl"
    resume_info = None
    resume_pool: List[ProposedTask] = []
    resume_rng_path = None
    if resume_ckpt is not None:
        resume_info, resume_pool, resume_rng_path = load_ez_resume(
            resume_ckpt, cfg, world, rank, protocol_manifest=protocol_manifest)
        if resume_info.get("run_id") != args.run_id:
            raise ValueError("v3 resume --run-id differs from sealed checkpoint")
        log(f"[resume] checkpoint={resume_ckpt} next={resume_info.get('next_outer')} "
            f"{resume_info.get('next_phase')}:{resume_info.get('next_step')} pool={len(resume_pool)}")
    global _REPORTER
    _REPORTER = _MetricsReporter(cfg, run_dir=run_dir, rank=rank)
    global _LOG_FH, _LOG_FH0
    _log_mode = "a" if resume_ckpt is not None else "w"
    _LOG_FH = open(run_dir / f"train_log.rank{rank}.jsonl", _log_mode)
    _LOG_FH0 = open(run_dir / "train_log.jsonl", _log_mode) if is_main(rank) else None
    log(f"[dist] rank={rank}/{world} local_rank={local} "
        f"device=cuda:{torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'} "
        f"| run_dir={run_dir}")
    if world > 1 and world != int(os.environ.get("WORLD_SIZE", "1")):
        log(f"[dist] ⚠️ world={world} 与 WORLD_SIZE={os.environ.get('WORLD_SIZE')} 不一致")
    if world == 1 and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        log("[dist] ⚠️ 检测到 WORLD_SIZE>1 但进程组未生效 —— torchrun 参数可能没传进去")

    tcfg = cfg["train"]
    # 用 is not None 而不是 or：--max-outer 0（只跑构建自检）不会被 falsy 吃掉
    N = tcfg.get("outer_loops", 4)
    M_P, M_S = tcfg.get("proposer_steps", 25), tcfg.get("solver_steps", 25)
    G = tcfg.get("group_size", 16)           # GRPO 组
    G_P = tcfg.get("solve_rate_rollouts", 8)  # 估 p̂(x) 用
    TPS = int(tcfg.get("tasks_per_step", 1))  # global task groups per step
    G_local = local_group_size(int(G), world)
    w_sub = tcfg.get("w_sub", 0.3)
    w_div = tcfg.get("w_div", 0.3)
    # ★ solve rate 每批并行处理的任务数（与 TPS 取 min：TPS=1 时退化为逐任务，避免补齐浪费）
    sr_tasks = max(1, min(int(tcfg.get("solve_rate_batch_tasks", 4)), TPS))

    def _game_unique_calls(items, key_of, max_per_call):
        """把 items 按 key_of（= game_file）轮转分组，保证**每次调用内 game 唯一**。

        `run_ez_solver_rollouts` 不允许同一批里同一 game_file 出现多个不同 task_desc
        （slot 按 gamefile 反查会混淆），所以同 game 的多个提案必须分到不同批。
        逐轮：每轮从每个 game 各取一个 → 同 game 的多个提案天然分到不同轮。
        yield (实参 sub, 补齐到 max_per_call 的 fill)：不足时用 sub[0] 补齐（同 game 同
        desc，不触发守卫；补出来的槽位结果调用方忽略）。
        """
        by: Dict[Any, List[Any]] = {}
        for it in items:
            by.setdefault(key_of(it), []).append(it)
        _ri = 0
        while True:
            _r = [q[_ri] for q in by.values() if len(q) > _ri]
            if not _r:
                break
            for _i in range(0, len(_r), max_per_call):
                _sub = _r[_i:_i + max_per_call]
                yield _sub, list(_sub) + [_sub[0]] * (max_per_call - len(_sub))
            _ri += 1

    # Smoke runs reduce update count and remain ineligible for formal resume or evaluation.
    if args.protocol_smoke:
        M_P = M_S = 1
    p_max_turns = cfg["proposer"].get("max_turns", 30)
    s_max_turns = cfg["solver"].get("max_turns", 30)

    if TPS < 1:
        raise ValueError("train.tasks_per_step must be positive")
    log(f"[dist] global task groups/step={TPS}; global G={G}; rank-local G={G_local}; "
        f"world={world}; adapter gradients use explicit all_reduce")

    # ---- 环境（facts 通道是 EZ 的命根子，必须开）----
    alf_cfg = load_alfworld_config(
        cfg["env"]["config_path"] if os.path.isabs(cfg["env"]["config_path"])
        else str((SRC_DIR.parent / cfg["env"]["config_path"]).resolve()))
    game_files = collect_game_files(alf_cfg, split=cfg["env"].get("split", "train"))
    log(f"[data] {len(game_files)} train games")

    p_cfg, s_cfg = build_ez_rollout_configs(
        cfg, G_local, proposer_turns=p_max_turns, solver_turns=s_max_turns)

    # ★ 批量探索规模：proposer 相位每次把 n_groups 个 game **一次性**交给
    #   run_proposer_exploration（批量接口），其前置条件是
    #       env.batch_size == group_size(=G_local) × len(game_files)
    #   n_groups = min(TPS, len(game_files)) 只依赖 TPS 和 game 池大小，二者此刻都已
    #   确定，所以这里直接把 env_p 的 batch 建成"实际会用到的规模"，运行期不存在
    #   batch 不匹配。若反过来把 batch 建成固定上界 G_local×TPS，当池子不够
    #   （len(game_files) < TPS）时就会在批量调用深处才断言失败，且已白白跑过一次
    #   reset —— 这里提前算、提前报，是更稳的写法。
    n_groups_eff = min(TPS, len(game_files))
    if n_groups_eff < 1:
        raise ValueError(
            f"train.tasks_per_step={TPS} 与 game 池 {len(game_files)} 不匹配，无法构造批量探索环境")
    if n_groups_eff < TPS:
        log(f"[warn] game 池仅 {len(game_files)} < tasks_per_step={TPS}，"
            f"批量探索降为 {n_groups_eff} 个 game/步（env_p.batch_size={G_local * n_groups_eff}）")
    env_p_batch = G_local * n_groups_eff
    env_p = ALFWorldBatchEnv(alf_cfg, split=cfg["env"].get("split", "train"),
                             batch_size=env_p_batch, max_episode_steps=p_cfg.max_turns,
                             with_facts=True, ignore_benchmark_done=True)
    env_s = ALFWorldBatchEnv(alf_cfg, split=cfg["env"].get("split", "train"),
                             batch_size=G_local * min(TPS, len(game_files)),
                             max_episode_steps=s_cfg.max_steps,
                             with_facts=True, ignore_benchmark_done=True)
    # ★ 估 solve rate 用独立 env（batch=G_P）：run_ez_solver_rollouts 要求 batch==len(tasks)，
    #   G_P(8) ≠ G(16) 时复用 env_s 会在第一个被接受提案处 ValueError（review P0-2）
    # ★ env_sr 懒建：只在 proposer 相位里 accepted 非空、真正要估 solve rate 时才创建，
    #   且只建一次（缓存到 env_sr）。省 1/5 slot，避免 8 卡全局 slot 静默超订。
    #   懒建只影响 env 对象是否存在，不影响任何 rank 的 backward 次数
    #   （backward 次数由 len(batch) 决定，与 env 无关），因此不违反红线 B。
    env_sr = None
    _envs_closed = False

    def _ensure_env_sr():
        # 懒建 env_sr：只建一次，创建成功后才注册 atexit.close
        nonlocal env_sr
        if _envs_closed:
            raise RuntimeError("[EZ] 环境已关闭，不能再懒建 env_sr")
        if env_sr is None:
            env_sr = ALFWorldBatchEnv(alf_cfg, split=cfg["env"].get("split", "train"),
                                      batch_size=G_P * sr_tasks, max_episode_steps=s_cfg.max_steps,
                                      with_facts=True, ignore_benchmark_done=True)
            atexit.register(env_sr.close)
        return env_sr

    def close_all():
        # 统一收尾：判空再 close，并把 env_sr 清空标记已关闭，
        # 避免收尾之后又被懒建出一个没人管的 env（子进程泄漏）
        nonlocal env_sr, _envs_closed
        for _e in (env_p, env_s, env_sr):
            if _e is not None:
                _e.close()
        env_sr = None
        _envs_closed = True

    # 异常退出也释放子进程/句柄（review P1-4）
    atexit.register(env_p.close)
    atexit.register(env_s.close)

    # ---- 模型（共享基座 + 双 LoRA adapter：proposer / solver）----
    # EZ explicitly synchronizes LoRA gradients and never wraps its raw PEFT model in DDP.
    raw, tokenizer = build_model_and_tokenizer(cfg, local, world, wrap_ddp=False)
    raw = build_dual_lora(raw, cfg, resume_checkpoint=resume_ckpt)
    broadcast_adapter_parameters(raw, world, dist)

    # ---- 启动自检：dtype / adapter / GC / facts（跑几小时前先炸掉所有致命项）----
    _p0 = next(raw.parameters())
    _adapters = [n for n, _ in raw.named_parameters() if "lora_" in n][:1]
    _lora_all = [p for n, p in raw.named_parameters() if "lora_" in n]
    _lora_tr = [p for p in _lora_all if p.requires_grad]
    log(f"[selfcheck] base dtype={_p0.dtype} | lora参数样例={_adapters} | "
        f"gc={getattr(raw, 'is_gradient_checkpointing', '?')} | "
        f"stop_mode=action_tag | base_frozen={not any(p.requires_grad for n, p in raw.named_parameters() if 'lora_' not in n)} "
        f"| LoRA 可训练 {len(_lora_tr)}/{len(_lora_all)} | lora dtype={_lora_all[0].dtype if _lora_all else '?'}")
    # facts 预检：EZ 没有 facts 就是废的，早炸比跑一半炸好
    from ez_state import assert_facts_mode as _afm
    from ez_proposer import _slot_snapshot as _snp
    _pre_games = [game_files[0]] * G_P
    # 预检临时建一个 env，验完即关；真正训练时再由懒建路径重建
    _pre_env = ALFWorldBatchEnv(alf_cfg, split=cfg["env"].get("split", "train"),
                                batch_size=G_P, max_episode_steps=s_cfg.max_steps,
                                with_facts=True, ignore_benchmark_done=True)
    try:
        _obs, _views, _adm, _used, _ = _pre_env.reset(_pre_games)
        _pre = _snp(getattr(_pre_env, "last_reset_facts", [None] * G_P)[0], _obs[0])
        _afm(_pre, "startup preflight")
        log(f"[selfcheck] facts 通道 OK（{len(_pre.facts)} 条事实）")
    finally:
        _pre_env.close()
    # env slot 数进启动自检日志（防 8 卡全局静默超订）：
    #   · env_p / env_s 已建，按真实 batch_size 计
    #   · env_sr 此刻必然还没建（懒建），但**预警必须按"将来会建"算**，
    #     否则这一行在启动时永远显示偏小的占用，预警形同虚设
    _slot_p = int(getattr(env_p, "batch_size", 0) or 0)
    _slot_s = int(getattr(env_s, "batch_size", 0) or 0)
    _slot_sr_built = env_sr is not None
    _slot_sr_show = _slot_sr_built and int(getattr(env_sr, "batch_size", 0) or 0) or G_P
    _slot_local = _slot_p + _slot_s + (0 if not _slot_sr_built else int(
        getattr(env_sr, "batch_size", 0) or 0))
    _slot_worst = _slot_p + _slot_s + G_P          # 懒建触发后的本 rank 占用
    _slot_global = world * _slot_worst
    _slot_warn = "" if _slot_global <= 256 else "  ⚠️ 全局 slot 占用偏高"
    log(f"[selfcheck] env slots: env_p={_slot_p} env_s={_slot_s} "
        f"env_sr={_slot_sr_show}{'' if _slot_sr_built else '(lazy)'} "
        f"→ 本 rank 当前 {_slot_local}，懒建后 {_slot_worst}；"
        f"{world} rank 全局 {_slot_global}{_slot_warn}")

    # ---- 梯度链自检：两个 adapter 是否各自都能接到梯度（不依赖环境/奖励）----
    #   在 phase 切换前跑：自检直接验证两个 adapter 的 forward/backward 梯度链
    #   会额外触发 reducer，日志更乱且没有诊断价值。device 见下一行。
    _gc = grad_chain_selfcheck(raw, tokenizer, next(raw.parameters()).device)
    log(f"[selfcheck] 梯度链 proposer grad_norm={_gc['proposer_grad_norm']:.4e} "
        f"({int(_gc.get('proposer_params_with_grad', 0))} 个张量有 grad) | "
        f"solver grad_norm={_gc['solver_grad_norm']:.4e} "
        f"({int(_gc.get('solver_params_with_grad', 0))} 个张量有 grad)")
    for _k in ("proposer_grad_norm", "solver_grad_norm"):
        _v = _gc.get(_k, 0.0)
        if not (_v > 0):
            raise SystemExit(
                f"[selfcheck] {_k}={_v!r} —— 该 adapter 收不到梯度，训练一定原地不动。"
                f" 检查：peft set_adapter 的 requires_grad 隔离 / 参数名里是否含 adapter 名")

    device = next(raw.parameters()).device

    # EZ deliberately uses the raw PEFT model: adapter trainability changes by phase,
    # and gradients are synchronized explicitly in a fixed parameter order.
    model = raw
    policy = HFPolicy(raw, tokenizer)

    # ------------------------------------------------------------------ rollout 后端
    # ★ ZTWANG_ROLLOUT_BACKEND=hf（默认，忠实路径）| vllm（工程加速）
    #
    #   换掉的**只有采样器**：`VLLMPolicy` 继承 `HFPolicy`，分词面完全复用，
    #   训练侧拿到的 `input_ids/loss_mask` 语义一字不改（见 src/ez_vllm_policy.py 顶部）。
    #
    #   LoRA 热同步：每次采样前把"当前要采样的那个 adapter"单独导出成 PEFT 目录并
    #   用新的 int_id 挂给 vLLM。三条相位边界都要刷：
    #       proposer 步内（每步权重都变）/ solve rate 前（→solver）/ 恢复（→proposer）
    #       solver 步内（每步权重都变）
    #   ⚠️ 漏刷 = 采样策略 ≠ 训练策略 = on-policy 静默破坏，所以刷完都打日志。
    rollout_backend = str(cfg["runtime"]["backend"]).strip().lower()
    vllm_policy = None
    _vllm_ver: Dict[str, int] = {"proposer": 0, "solver": 0}
    if rollout_backend == "vllm":
        from ez_vllm_policy import (VLLMPolicy, build_vllm_engine,
                                    export_lora_adapter, lora_workdir)

        _base_path = os.path.expandvars(os.path.expanduser(str(cfg["model"]["path"])))
        engine_kwargs = vllm_engine_kwargs(cfg, rank)
        vllm_policy = VLLMPolicy(
            build_vllm_engine(
                _base_path, **engine_kwargs),
            tokenizer, log_fn=log)
        policy = vllm_policy
        log(f"[selfcheck] rollout backend=vllm（引擎已就绪，engine_seed={engine_kwargs['seed']}，adapter 按需热刷）")
    else:
        log("[selfcheck] rollout backend=hf（默认）")

    def sync_rollout_adapter(adapter: str) -> None:
        """把指定 adapter 的**当前**权重刷进采样后端。HF 后端下是 no-op。"""
        if vllm_policy is None:
            return
        _t0 = time.time()
        _vllm_ver[adapter] += 1
        # ★ 必须**按 rank 分目录**：run_dir 是全 rank 共享的，8 个 rank 同时
        #   save_pretrained 到同一路径会写出损坏的 safetensors（且是静默的）。
        _d = lora_workdir(str(run_dir), adapter, _vllm_ver[adapter], rank=rank)
        # ★★ 必须用**返回值**：PEFT 0.18 的 save_pretrained 会多套一层 `<adapter>/`，
        #    export_lora_adapter 负责探测出真正含 adapter_config.json 的那一层。
        #    这里丢掉返回值 = 把父目录交给 vLLM ⇒ 第一次采样就
        #        ValueError: Loading lora ... failed: No adapter found for <dir>
        #    而且 EngineCore 会直接死，整个训练作业报废（engine failures）。
        _d = export_lora_adapter(raw, adapter, _d)
        vllm_policy.set_lora(adapter, _d)
        log(f"[vllm] 刷新 {adapter} adapter v{_vllm_ver[adapter]} | "
            f"{time.time() - _t0:.1f}s | {_d}")

    # ⚠️ 双 adapter 隔离由 peft 的 set_adapter 负责；optimizer 必须覆盖**两个**
    #    adapter 的全部参数（按名字里的 "lora_" 选），否则切到 solver 相位时新参数不在优化器里。
    #    raw.named_parameters() gives a stable fixed-order parameter set for manual sync.
    opt_params = [p for n, p in raw.named_parameters() if "lora_" in n]
    if not opt_params:
        raise SystemExit("没有找到任何 LoRA 参数，检查 peft 适配是否正常")
    optimizer = torch.optim.AdamW(opt_params, lr=tcfg.get("lr", 1e-4),
                                  weight_decay=tcfg.get("weight_decay", 0.0))
    if resume_ckpt is not None:
        optimizer_path = resume_ckpt / "optimizer.pt"
        if not optimizer_path.is_file():
            raise ValueError("resume checkpoint is missing optimizer.pt")
        resume_uid_counter = restore_ez_optimizer_rng(optimizer, optimizer_path, resume_rng_path)
    else:
        resume_uid_counter = 0
    clip_eps = tcfg.get("clip_eps", 0.3)
    max_grad_norm = tcfg.get("max_grad_norm", 1.0)
    pad_id = tokenizer.pad_token_id
    # GRPO surrogate 的归约口径：
    #   "seq"   = 论文语义（先每条序列内部按 token 平均，再组间等权）—— 默认
    #   "token" = 旧行为（把组里所有 token 混在一起平均 = 按长度加权，有 length bias）
    # 详见 src/grpo_norm.py 的模块 docstring，以及 scripts/ez_repro_lossnorm.py 的实测。
    loss_norm = tcfg.get("loss_norm", "seq")
    if loss_norm != "seq":
        raise ValueError("EZ global microbatch accumulation requires train.loss_norm='seq'")
    if loss_norm == "token" and is_main(rank):
        log("[warn] train.loss_norm='token'：组内按长度加权，会有 length bias "
            "（论文语义是 'seq'）。确认这是有意为之。")
    # 组内优势的标准差用无偏估计（ddof=1，TRL/DeepSeekMath 口径）还是
    # 总体标准差（ddof=0，numpy 默认）。ddof=0 会低估 std ⇒ 放大优势：
    # G=16 放大大 3.3%，G=2 放大 41%，跟 max_grad_norm 的阈值也对不上。
    adv_ddof = int(tcfg.get("adv_ddof", 1))
    if adv_ddof not in (0, 1):
        raise ValueError(f"train.adv_ddof 必须是 0 或 1，收到 {adv_ddof!r}")
    if is_main(rank):
        log(f"[selfcheck] loss_norm={loss_norm} adv_ddof={adv_ddof} "
            f"clip_eps={clip_eps} max_grad_norm={max_grad_norm}")

    # ★ rng 必须按 rank 偏移：`set_seed(seed, rank)` 里的 `seed + rank` 只是**局部**
    #   变量，改不到这里，直接用 `seed` 会让所有 rank 的 rng 状态完全一致
    #   （多卡采样重复的根因）。加上 rank 偏移，各 rank 的随机流才真正分开。
    micro_batch_size = int(tcfg.get("micro_batch_size", 1))
    kl_beta = float(tcfg.get("kl_beta", 0.001))
    sync_bucket_mb = int(tcfg.get("sync_bucket_mb", 25))
    check_sync_every = int(tcfg.get("check_adapter_sync_every", 5))
    task_pool: List[ProposedTask] = list(resume_pool)
    uid_counter = resume_uid_counter
    start_outer = int(resume_info.get("next_outer", 0)) if resume_info else 0
    resume_phase = str(resume_info.get("next_phase", "proposer")) if resume_info else "proposer"
    resume_step = int(resume_info.get("next_step", 0)) if resume_info else 0

    def persist_pool() -> None:
        """任务池落盘。★ 只 rank0 写：8 个 rank 同时 open(...,"w") 同一文件会互相截断。
        各 rank 的池子在 all_gather 之后本来就是同一份，所以 rank0 写的即全集。"""
        if not is_main(rank):
            return
        with open(task_pool_path, "w") as fh:
            for t in task_pool:
                fh.write(json.dumps(t.to_json(), ensure_ascii=False) + "\n")

    t_start = time.time()
    for outer in range(start_outer, N):
        # ⚠️ 冒烟只跑 1 个 outer 的判据必须放在**循环开头**。
        #    之前写在循环结尾（ez_save 之后），但 solver 空池那条路径会 `continue`
        #    直接跳到下一轮，把 break 绕过去 —— 实测日志里出现了 outer 0/1/2/3 各跑一遍。
        if args.protocol_smoke and outer > 0:
            break
        # ================= proposer 相位 =================
        model.eval()
        p_steps = M_P if not args.protocol_smoke else 1
        p_start = resume_step if outer == start_outer and resume_phase == "proposer" else 0
        p_range = range(p_start, p_steps) if not (outer == start_outer and resume_phase == "solver") else range(0)
        for step in p_range:
            st0 = time.time()
            _cuda_sync_and_reset_peak()
            model.eval()
            select_adapter(raw, "proposer")
            # ★ vLLM 后端：上一步 optimizer.step() 之后 proposer 权重已变，
            #   采样前必须刷一次，否则本轮探索用的是**旧策略**（on-policy 静默破坏）。
            sync_rollout_adapter("proposer")
            # All ranks process the same global task groups. Each group is sharded
            # rank-strided inside run_proposer_exploration.
            games: List[str] = []
            n_groups = min(TPS, len(game_files))
            selected_games = broadcast_pool_indices(
                len(game_files), n_groups, outer * max(M_P, 1) + step,
                seed + 13007, rank, world, dist)
            games = [game_files[i] for i in selected_games]
            local_games = games
            # 1) 探索 + 提案（**批量**：一次把 local_games 全部 game 一起探索）
            # ★ 批量接口返回的外层顺序 == 传入的 local_games 顺序，所以组按 game
            #   **逐个**收集即可（而不是事后按 G 切片）：切片一旦错位，组里会混进
            #   不同 game 的 rollout（红线 A：组是算法不可分单元）。
            cand: List[ProposerRollout] = []
            groups: List[List[ProposerRollout]] = []
            # env_p 按"实际规模"建好后应恒等于本次 game 数 × G_local；不符说明调用侧与
            # 环境侧对 n_groups 的理解漂移了，早炸好过让 ez_proposer 深处报错。
            if env_p.batch_size != G_local * len(local_games):
                raise ValueError(
                    f"批量探索环境不匹配：env_p.batch_size={env_p.batch_size} != "
                    f"G_local×len(local_games)={G_local * len(local_games)}"
                    f"（G_local={G_local}, len(local_games)={len(local_games)}）")
            grouped = run_proposer_exploration(env_p, policy, local_games, p_cfg, log_fn=log)
            for _one in grouped:
                cand.extend(_one)
                groups.append(_one)
            #   ★ 空组必须"所有 rank 一起炸"：某个 rank 的组是空的就会少一次
            #     backward，其它 rank 干等 → NCCL 卡 30 分钟。all_reduce MIN 让
            #     全员同时看到问题并同时退出。
            _mg = min((len(g) for g in groups), default=0)
            if world > 1:
                _t = torch.tensor([_mg], device=device)
                dist.all_reduce(_t, op=dist.ReduceOp.MIN)
                _mg = int(_t.item())
                del _t
            if _mg == 0 or not groups:
                raise SystemExit(
                    f"[EZ][红线B] 存在空的 GRPO 组（全局最小 {_mg}，本 rank {len(groups)} 组）——"
                    f" 各 rank 的 backward 次数会不一致，rank gradient synchronization would stall。"
                    f" 检查 run_proposer_exploration 是否有静默丢样本的路径")
            accepted = [c for c in cand if c.accepted and c.task is not None]
            # 2) 冻结 solver 估 solve rate —— **批量并行**（原为逐任务串行，TPS>1 时成主瓶颈）。
            #    每批 sr_tasks 个任务，把它们的 G_P 条 rollout 合并成一次调用（env_sr 一次跑
            #    sr_tasks×G_P 个 slot）；末尾不足一批时用最后一个任务补齐（补的槽位结果忽略）。
            new_tasks: List[ProposedTask] = []
            for c in accepted:
                # ★ uid 必须带 rank：每 rank 各自 proposals，不带 rank 会跨 rank 撞 uid，
                #   撞了之后 solver 相位的 `task_id == uid` 分组会把不同任务混成一组。
                c.task.meta["uid"] = f"o{outer}s{step}-r{rank}-{uid_counter}"
                uid_counter += 1
            if accepted:
                select_adapter(raw, "solver")
                sync_rollout_adapter("solver")
                for _sub, _fill in _game_unique_calls(accepted, lambda c: c.task.game_file, sr_tasks):
                    _input: List[ProposedTask] = []
                    for c in _fill:
                        _input.extend([c.task] * G_P)
                    _outs = run_ez_solver_rollouts(_ensure_env_sr(), policy, _input, s_cfg,
                                                   tokenize=False, log_fn=log)
                    for _j, c in enumerate(_sub):
                        _grp = _outs[_j * G_P:(_j + 1) * G_P]
                        _p_hat = sum(r.verify.final_match for r in _grp) / G_P
                        _rt, _parts = proposer_reward(c.task, _p_hat, w_sub=w_sub, w_div=w_div)
                        c.reward = _rt
                        c.__dict__["reward_parts"] = _parts
                        c.task.meta["solve_rate"] = _p_hat
                        new_tasks.append(c.task)
                select_adapter(raw, "proposer")
                sync_rollout_adapter("proposer")
            if world > 1:
                # ★ 任务池必须**每个 rank 完全一致**：solver 相位要从池里采样，
                #   各 rank 池子不同则采出的任务数/内容不同 → backward 次数不同 → 死锁。
                #   all_gather 之后所有人拿到同一份，顺带省掉一个单独的 gather 写盘逻辑。
                _g: List[Any] = [None] * world
                dist.all_gather_object(_g, new_tasks)
                new_tasks = [t for _sub in _g for t in _sub]
                del _g
            task_pool.extend(new_tasks)
            persist_pool()
            _rollout_s = time.time() - st0
            # 3) proposer GRPO update: each global game group is sharded across ranks;
            #    advantages are gathered/normalized globally before this local backward.
            model.train()
            optimizer.zero_grad(set_to_none=True)
            _diag = []
            _p_loss_sum = 0.0
            _p_components_local = np.zeros(5, dtype=np.float64)
            _p_reward_std_sum = 0.0
            _p_nonzero_adv_groups = 0.0
            _tokens_local = 0
            _backward_t0 = time.time()
            for group_idx, groll in enumerate(groups):
                for c in groll:
                    tokenize_proposer_rollout(c)
                    _tokens_local += int(sum(c.__dict__["loss_mask"][1:]))
                full_rewards = gather_group_values([c.reward for c in groll], rank, world, dist)
                full_advs = global_group_advantages(full_rewards, ddof=adv_ddof)
                _group_weight = 1.0 / max(len(groups), 1)
                _p_reward_std_sum += (float(np.std(full_rewards, ddof=0)) * _group_weight
                                      if rank == 0 else 0.0)
                _p_nonzero_adv_groups += (float(any(abs(float(a)) > 1e-12 for a in full_advs)) * _group_weight
                                          if rank == 0 else 0.0)
                advs = full_advs[rank::world]
                trajs_like = []
                from rollout import Trajectory as _T
                for c, a in zip(groll, advs):
                    trajs_like.append(_T(task_id=f"ez_proposer_{group_idx}", task_type="proposer",
                                         game_file="", messages=c.messages,
                                         input_ids=c.__dict__["input_ids"],
                                         loss_mask=c.__dict__["loss_mask"],
                                         stats=None, advantage=a))
                _diag.append((round(max(full_advs) - min(full_advs), 3),
                              [len(c.__dict__["input_ids"]) for c in groll],
                              [int(sum(c.__dict__["loss_mask"][1:])) for c in groll]))
                _g_loss, _g_stats, _g_valid = backward_microbatches(
                    raw, model, trajs_like, pad_id, device, adapter="proposer",
                    clip_eps=clip_eps, micro_batch_size=micro_batch_size, world=world,
                    loss_norm=loss_norm, beta=kl_beta, dist_obj=dist,
                    group_weight=_group_weight)
                _p_loss_sum += _g_loss
                _stats_local = {key: sum(float(row.get(key, 0.0)) for row in _g_stats)
                                for key in ("pg_sequence_sum", "kl_sequence_sum",
                                            "objective_sequence_sum", "valid_sequence_count")}
                _denom_group = max(int(_g_valid), 1)
                _p_components_local[0] += _stats_local["pg_sequence_sum"] / _denom_group * _group_weight
                _p_components_local[1] += _stats_local["kl_sequence_sum"] / _denom_group * _group_weight
                _p_components_local[2] += _stats_local["objective_sequence_sum"] / _denom_group * _group_weight
                _p_components_local[3] += _stats_local["objective_sequence_sum"] * _group_weight
                _p_components_local[4] += _stats_local["valid_sequence_count"]
            _cuda_sync()
            _backward_s = time.time() - _backward_t0
            _nof = {id(p): n for n, p in raw.named_parameters()}
            _sync_t0 = time.time()
            _cuda_sync()
            sync_active_adapter_gradients(raw, "proposer", world, sync_bucket_mb, dist)
            _cuda_sync()
            _sync_s = time.time() - _sync_t0
            _act = [p for p in opt_params if p.grad is not None and p.requires_grad]
            _gn = float(sum(p.grad.detach().float().norm() for p in _act)) if _act else 0.0
            # LoRA B starts at zero, so report A and B gradient norms separately.
            _gnA = float(sum(p.grad.detach().float().norm() for p in _act
                             if ".lora_A." in _nof.get(id(p), "")))
            _gnB = float(sum(p.grad.detach().float().norm() for p in _act
                             if ".lora_B." in _nof.get(id(p), "")))
            _step_t0 = time.time()
            torch.nn.utils.clip_grad_norm_(opt_params, max_grad_norm)
            _update_rel_p = update_probe(raw, opt_params, optimizer, f"outer {outer} P{step}")
            if step == 0 or (check_sync_every > 0 and (step + 1) % check_sync_every == 0):
                _sync_max = check_adapter_sync(raw, "proposer", world, dist)
                log(f"[adapter-sync] phase=proposer step={step} status=PASS max_abs={_sync_max:.3e}")
            optimizer.zero_grad(set_to_none=True)
            _cuda_sync()
            _optimizer_s = time.time() - _step_t0
            # Record group rewards to distinguish flat feedback from a broken gradient path.
            _rw = [round(float(c.reward), 3) for c in cand]
            # Summarize rejection reasons alongside the acceptance count.
            _rej = Counter(getattr(c, "reject_reason", "unknown") or "accepted"
                           for c in cand)
            log(f"[outer {outer}][P{step}] rank={rank} global_games={len(games)} local_G={G_local} "
                f"| pool={len(task_pool)} (+{len(new_tasks)}) "
                f"| accept={len(accepted)}/{len(cand)} | rew={_rw} "
                f"| rew_spread={max(_rw) - min(_rw) if _rw else 0:.3f} "
                f"| rej={dict(_rej)} | {time.time()-st0:.0f}s")
            # 断层定位：grad_norm>0 而 update_rel=0 -> 优化器/精度问题；
            # grad_norm=0 而 adv_spread>0 -> loss 没把梯度传出来（mask/图断）
            log(f"[outer {outer}][P{step}] 诊断 adv_spread/n_active/len_ids/mask_sum="
                f"{_diag} | 真实 backward grad_norm={_gn:.4e} "
                f"({len(_act)} 个张量有 grad；A={_gnA:.4e} B={_gnB:.4e})"
                f"{'  ← A=0 是正常的（B 初始全 0）' if _gnA == 0.0 and _gnB > 0 else ''}")
            _mem_alloc, _mem_reserved = _cuda_peak_gib()
            _token_metric = torch.tensor([_tokens_local], dtype=torch.int64, device=device)
            if world > 1:
                dist.all_reduce(_token_metric, op=dist.ReduceOp.SUM)
            _tokens_global = int(_token_metric.item())
            del _token_metric
            log(f"[telemetry] phase=proposer outer={outer} step={step} rollout_s={_rollout_s:.3f} "
                f"backward_s={_backward_s:.3f} sync_s={_sync_s:.3f} optimizer_s={_optimizer_s:.3f} "
                f"generated_tokens_global={_tokens_global} generated_tokens_local={_tokens_local} "
                f"effective_global_batch={len(games) * G} "
                f"cuda_peak_allocated_gib={_mem_alloc:.3f} cuda_peak_reserved_gib={_mem_reserved:.3f} "
                f"attention={_attention_backend(raw, cfg)}")
            _p_metric = torch.tensor(_p_components_local, device=device, dtype=torch.float64)
            if world > 1:
                dist.all_reduce(_p_metric, op=dist.ReduceOp.SUM)
            (_p_pg_mean_global, _p_kl_mean_global, _p_objective_mean_global,
             _p_raw_seq_sum_global, _p_valid_sequences_global) = map(float, _p_metric.tolist())
            del _p_metric
            _p_loss_sum_global = _p_objective_mean_global
            _p_aux = torch.tensor([_p_reward_std_sum, _p_nonzero_adv_groups],
                                  device=device, dtype=torch.float64)
            if world > 1:
                dist.all_reduce(_p_aux, op=dist.ReduceOp.SUM)
            _p_reward_std_mean, _p_nonzero_adv_group_fraction = map(float, _p_aux.tolist())
            del _p_aux
            # ---- Local metrics: proposer phase ----
            # step 必须是**全局单调递增**的，否则 P/S 两条曲线会叠在同一 x 上。
            # 只有 rank0 真的会发（其它 rank 的 _REPORTER 是 no-op 对象）。
            _REPORTER.log({
                "train/loss": _p_loss_sum_global,
                "train/objective_mean": _p_objective_mean_global,
                "train/raw_sequence_objective_sum": _p_raw_seq_sum_global,
                "train/pg_mean": _p_pg_mean_global,
                "train/kl_mean": _p_kl_mean_global,
                "train/valid_sequences": _p_valid_sequences_global,
                "train/grad_norm": _gn,
                "train/grad_norm_lora_A": _gnA,
                "train/grad_norm_lora_B": _gnB,
                "train/active_adapter_delta": _update_rel_p,
                "proposer/reward_mean": float(np.mean(_rw)) if _rw else 0.0,
                "proposer/reward_spread": max(_rw) - min(_rw) if _rw else 0.0,
                "proposer/group_reward_std_mean": _p_reward_std_mean,
                "proposer/nonzero_adv_group_fraction": _p_nonzero_adv_group_fraction,
                "proposer/accept_rate": len(accepted) / max(len(cand), 1),
                "proposer/pool_size": len(task_pool),
                "proposer/sec_per_step": time.time() - st0,
            }, step=outer * (M_P + M_S) + step)
            p_done = step + 1 == p_steps
            if p_done or (step + 1) % 5 == 0:
                next_outer, next_phase, next_step = ((outer, "solver", 0) if p_done
                                                     else (outer, "proposer", step + 1))
                ez_save(raw, tokenizer, optimizer, run_dir, f"outer{outer}-P{step}", cfg,
                        {"next_outer": next_outer, "next_phase": next_phase,
                         "next_step": next_step}, task_pool, rank, world,
                        uid_counter=uid_counter, dry_run=args.protocol_smoke,
                        protocol_manifest=protocol_manifest)
            if args.protocol_smoke:
                break

        # ================= solver 相位 =================
        model.eval()
        select_adapter(raw, "solver")
        # 任务来源对齐论文 Appendix C：solver 只练**本 outer proposer 相位产出的任务**
        # （论文是每迭代用冻结 proposer 重新采样；跨 outer 抽旧题会让任务分布与
        #   当前 proposer 脱节 —— review #2）。池子仍全量持久化供分析。
        outer_pool = [t for t in task_pool if str(t.meta.get("uid", "")).startswith(f"o{outer}")]
        # ★ "跳不跳"必须是**全局决定**：某个 rank 空池就 continue，而别的 rank 继续
        #   backward —— 空池那个 rank 直接卡死在 NCCL barrier（红线 B）。
        #   池子经 all_gather 后本来就一致，这里再加一道 MIN all_reduce 兜底。
        _nlocal = len(outer_pool)
        if world > 1:
            _t = torch.tensor([_nlocal], device=device)
            dist.all_reduce(_t, op=dist.ReduceOp.MIN)
            _nglob = int(_t.item())
            del _t
        else:
            _nglob = _nlocal
        if _nglob == 0:
            log(f"[solver] 本 outer 任务池为空（rank{rank} 本地 {_nlocal} 条，全局最少 {_nglob} 条），"
                f"所有 rank 一起跳过 solver 相位")
            ez_save(raw, tokenizer, optimizer, run_dir, f"outer{outer}-empty-S", cfg,
                    {"next_outer": outer + 1, "next_phase": "proposer", "next_step": 0},
                    task_pool, rank, world, uid_counter=uid_counter,
                    dry_run=args.protocol_smoke,
                    protocol_manifest=protocol_manifest)
            continue
        s_steps = M_S if not args.protocol_smoke else 1
        s_start = resume_step if outer == start_outer and resume_phase == "solver" else 0
        for step in range(s_start, s_steps):
            st0 = time.time()
            _cuda_sync_and_reset_peak()
            model.eval()
            # ★ vLLM 后端：solver 相位**每步都更新 solver 权重**，所以每步采样前都要刷。
            sync_rollout_adapter("solver")
            # Draw the same-sized, non-overlapping task slice on each rank.
            n_tasks = min(TPS, len(outer_pool))
            chosen = broadcast_pool_indices(len(outer_pool), n_tasks,
                                            outer * max(M_S, 1) + step,
                                            seed + 104729, rank, world, dist)
            batch = [outer_pool[i] for i in chosen]
            # ★ 批量并行：本步**所有任务**的 rollout 合并成一次调用（原逐任务串行，是 TPS>1
            #   时的第二个瓶颈）。按 game_file 轮转分批（同一批内同 game 只出现一次），
            #   env_s.batch = G_local×TPS，末尾不足时用 sub[0] 补齐（补的槽位结果忽略）。
            _res: Dict[str, Any] = {}
            for _sub, _fill in _game_unique_calls(batch, lambda t: t.game_file, TPS):
                _flat = [t for t in _fill for _ in range(G_local)]
                _outs = run_ez_solver_rollouts(env_s, policy, _flat, s_cfg, tokenize=True, log_fn=log)
                for _j, t in enumerate(_sub):
                    _res[t.meta["uid"]] = _outs[_j * G_local:(_j + 1) * G_local]
            solver_groups: List[List[EzSolverTrajectory]] = [_res[t.meta["uid"]] for t in batch]
            _rollout_s = time.time() - st0
            model.train()
            optimizer.zero_grad(set_to_none=True)
            # ★ backward 次数必须**恒等于 len(batch)**（红线 B）。原来那句
            #   `if not grp: continue` 会让某个 rank 少一次 backward，其它 rank
            #   就永久卡在 NCCL barrier 上。空组用"零 loss 但带计算图"的 backward 占位。
            _s_loss_sum = 0.0
            _s_components_local = np.zeros(5, dtype=np.float64)
            _s_reward_std_sum = 0.0
            _s_nonzero_adv_groups = 0.0
            all_roll: List[EzSolverTrajectory] = []
            _tokens_local = 0
            _backward_t0 = time.time()
            for grp in solver_groups:
                full_rewards = gather_group_values([r.traj.reward for r in grp], rank, world, dist)
                full_advs = global_group_advantages(full_rewards, ddof=adv_ddof)
                _group_weight = 1.0 / max(len(solver_groups), 1)
                _s_reward_std_sum += (float(np.std(full_rewards, ddof=0)) * _group_weight
                                      if rank == 0 else 0.0)
                _s_nonzero_adv_groups += (float(any(abs(float(a)) > 1e-12 for a in full_advs)) * _group_weight
                                          if rank == 0 else 0.0)
                for r, a in zip(grp, full_advs[rank::world]):
                    r.traj.advantage = a
                _g_loss, _g_stats, _g_valid = backward_microbatches(
                    raw, model, [r.traj for r in grp], pad_id, device, adapter="solver",
                    clip_eps=clip_eps, micro_batch_size=micro_batch_size, world=world,
                    loss_norm=loss_norm, beta=kl_beta, dist_obj=dist,
                    group_weight=_group_weight)
                _s_loss_sum += _g_loss
                _stats_local = {key: sum(float(row.get(key, 0.0)) for row in _g_stats)
                                for key in ("pg_sequence_sum", "kl_sequence_sum",
                                            "objective_sequence_sum", "valid_sequence_count")}
                _denom_group = max(int(_g_valid), 1)
                _s_components_local[0] += _stats_local["pg_sequence_sum"] / _denom_group * _group_weight
                _s_components_local[1] += _stats_local["kl_sequence_sum"] / _denom_group * _group_weight
                _s_components_local[2] += _stats_local["objective_sequence_sum"] / _denom_group * _group_weight
                _s_components_local[3] += _stats_local["objective_sequence_sum"] * _group_weight
                _s_components_local[4] += _stats_local["valid_sequence_count"]
                all_roll.extend(grp)
                _tokens_local += sum(int(r.traj.n_trainable_tokens) for r in grp)
            _cuda_sync()
            _backward_s = time.time() - _backward_t0
            # ★ S 相位也打一行同款诊断：判读脚本要靠「真实 backward grad_norm」把
            #   「有梯度却没位移」（真断链，FATAL）和「本来就没梯度」（奖励全同分，
            #   正确不动）分开。只有 P 相位有诊断行时，S 相位的 0/504 无法判定。
            _nofS = {id(p): n for n, p in raw.named_parameters()}
            _sync_t0 = time.time()
            _cuda_sync()
            sync_active_adapter_gradients(raw, "solver", world, sync_bucket_mb, dist)
            _cuda_sync()
            _sync_s = time.time() - _sync_t0
            _actS = [p for p in opt_params if p.grad is not None and p.requires_grad]
            _gnS = float(sum(p.grad.detach().float().norm() for p in _actS)) if _actS else 0.0
            _gnSA = float(sum(p.grad.detach().float().norm() for p in _actS
                              if ".lora_A." in _nofS.get(id(p), "")))
            _gnSB = float(sum(p.grad.detach().float().norm() for p in _actS
                              if ".lora_B." in _nofS.get(id(p), "")))
            _step_t0 = time.time()
            torch.nn.utils.clip_grad_norm_(opt_params, max_grad_norm)
            _update_rel_s = update_probe(raw, opt_params, optimizer, f"outer {outer} S{step}")
            if step == 0 or (check_sync_every > 0 and (step + 1) % check_sync_every == 0):
                _sync_max = check_adapter_sync(raw, "solver", world, dist)
                log(f"[adapter-sync] phase=solver step={step} status=PASS max_abs={_sync_max:.3e}")
            optimizer.zero_grad(set_to_none=True)
            _cuda_sync()
            _optimizer_s = time.time() - _step_t0
            rw = [r.traj.reward for r in all_roll]
            _s_metric = torch.tensor(_s_components_local, device=device, dtype=torch.float64)
            if world > 1:
                dist.all_reduce(_s_metric, op=dist.ReduceOp.SUM)
            (_s_pg_mean_global, _s_kl_mean_global, _s_objective_mean_global,
             _s_raw_seq_sum_global, _s_valid_sequences_global) = map(float, _s_metric.tolist())
            del _s_metric
            _s_loss_sum_global = _s_objective_mean_global
            _s_aux = torch.tensor([_s_reward_std_sum, _s_nonzero_adv_groups],
                                  device=device, dtype=torch.float64)
            if world > 1:
                dist.all_reduce(_s_aux, op=dist.ReduceOp.SUM)
            _s_reward_std_mean, _s_nonzero_adv_group_fraction = map(float, _s_aux.tolist())
            del _s_aux
            _s_outcomes = torch.tensor([
                float(len(all_roll)),
                float(sum(1 for r in all_roll if r.verify.final_match)),
                float(sum(1 for r in all_roll if not r.verify.final_match and r.traj.reward > 0)),
                float(sum(1 for r in all_roll if r.traj.stats is not None and r.traj.stats.truncated)),
                float(sum(len(r.traj.segments) for r in all_roll)),
                float(sum(r.traj.stats.n_valid_steps for r in all_roll if r.traj.stats is not None)),
            ], device=device, dtype=torch.float64)
            if world > 1:
                dist.all_reduce(_s_outcomes, op=dist.ReduceOp.SUM)
            (_s_n_global, _s_final_global, _s_partial_global, _s_truncated_global,
             _s_segments_global, _s_valid_actions_global) = map(float, _s_outcomes.tolist())
            del _s_outcomes
            log(f"[outer {outer}][S{step}] rank={rank} global_tasks={len(batch)} local_G={G_local} "
                f"| reward mean={np.mean(rw):.3f} "
                f"| final_match={sum(r.verify.final_match for r in all_roll)}/{len(all_roll)} "
                f"| pool={len(outer_pool)} | {time.time()-st0:.0f}s")
            log(f"[outer {outer}][S{step}] 诊断 真实 backward grad_norm={_gnS:.4e} "
                f"({len(_actS)} 个张量有 grad；A={_gnSA:.4e} B={_gnSB:.4e})"
                f"{'  ← 奖励全同分 ⇒ 优势全 0 ⇒ 梯度精确为 0，参数不动是正确的' if _gnS == 0.0 else ''}")
            _mem_alloc, _mem_reserved = _cuda_peak_gib()
            _token_metric = torch.tensor([_tokens_local], dtype=torch.int64, device=device)
            if world > 1:
                dist.all_reduce(_token_metric, op=dist.ReduceOp.SUM)
            _tokens_global = int(_token_metric.item())
            del _token_metric
            log(f"[telemetry] phase=solver outer={outer} step={step} rollout_s={_rollout_s:.3f} "
                f"backward_s={_backward_s:.3f} sync_s={_sync_s:.3f} optimizer_s={_optimizer_s:.3f} "
                f"generated_tokens_global={_tokens_global} generated_tokens_local={_tokens_local} "
                f"effective_global_batch={len(batch) * G} "
                f"cuda_peak_allocated_gib={_mem_alloc:.3f} cuda_peak_reserved_gib={_mem_reserved:.3f} "
                f"attention={_attention_backend(raw, cfg)}")
            # ---- Local metrics: solver phase ----
            # step 要排在同 outer 的 P 相位**之后**（+M_P），否则两条曲线会撞车。
            _sanity = {
                "train/loss": _s_loss_sum_global,
                "train/objective_mean": _s_objective_mean_global,
                "train/raw_sequence_objective_sum": _s_raw_seq_sum_global,
                "train/pg_mean": _s_pg_mean_global,
                "train/kl_mean": _s_kl_mean_global,
                "train/valid_sequences": _s_valid_sequences_global,
                "train/grad_norm": _gnS,
                "train/grad_norm_lora_A": _gnSA,
                "train/grad_norm_lora_B": _gnSB,
                "train/active_adapter_delta": _update_rel_s,
                "solver/reward_mean": float(np.mean(rw)) if rw else 0.0,
                "solver/final_match_rate": (sum(r.verify.final_match for r in all_roll)
                                            / max(len(all_roll), 1)),
                "solver/group_reward_std_mean": _s_reward_std_mean,
                "solver/nonzero_adv_group_fraction": _s_nonzero_adv_group_fraction,
                "solver/partial_reward_rate": _s_partial_global / max(_s_n_global, 1.0),
                "solver/truncation_rate": _s_truncated_global / max(_s_n_global, 1.0),
                "solver/generated_responses": _s_segments_global,
                "solver/action_tag_valid_actions": _s_valid_actions_global,
                "solver/action_tag_valid_rate": _s_valid_actions_global / max(_s_segments_global, 1.0),
                "solver/response_tokens": _tokens_global,
                "solver/final_match_rate_global": _s_final_global / max(_s_n_global, 1.0),
                "solver/n_rollouts": len(all_roll),
                "solver/sec_per_step": time.time() - st0,
            }
            _REPORTER.log(_sanity, step=outer * (M_P + M_S) + M_P + step)
            s_done = step + 1 == s_steps
            if s_done or (step + 1) % 5 == 0:
                next_outer, next_phase, next_step = ((outer + 1, "proposer", 0) if s_done
                                                     else (outer, "solver", step + 1))
                ez_save(raw, tokenizer, optimizer, run_dir, f"outer{outer}-S{step}", cfg,
                        {"next_outer": next_outer, "next_phase": next_phase,
                         "next_step": next_step}, task_pool, rank, world,
                        uid_counter=uid_counter, dry_run=args.protocol_smoke,
                        protocol_manifest=protocol_manifest)
            if args.protocol_smoke:
                break

        # ★ 只在 rank0 存档：peft 保存的是 adapter（几百 MB），8 个 rank 各存一份
        #   既浪费又会在同一路径上互相踩。（deepspeed 靠不同的 zero stage 存分片，
        #   我们不走那条路。）
    close_all()
    persist_pool()
    # ★ dry-run 只跑 outer 0，但 N 仍是配置里的 4 —— 直接打 N 会让人以为跑了 4 轮
    _n_done = 1 if args.protocol_smoke and N > 0 else N
    log(f"[done] rank={rank} {_n_done} outer loops in {(time.time()-t_start)/60:.1f} min -> {run_dir}")
    if _REPORTER is not None:
        _REPORTER.finish()
    for _fh in (_LOG_FH, _LOG_FH0):
        if _fh is not None:
            _fh.close()
    if world > 1 and dist.is_initialized():
        # ★ barrier 放在 destroy 之前：某个 rank 先 destroy 会让其它 rank 的
        #   收尾 all-reduce（如有）直接报 "process group has been destroyed"。
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
