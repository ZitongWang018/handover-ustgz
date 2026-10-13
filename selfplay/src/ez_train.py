"""Explorer-Zero 交替训练主循环（论文 Algorithm 1 / Appendix C / Table 3）。

    python src/ez_train.py --config configs/ez_qwen3_4b.yaml

结构（对齐论文）：
    for outer in 1..N:                     # N=4 外循环
        proposer 相位 M_P 步：              # ALFWorld M_P=25
            探索 → 提案 → 冻结 solver 估 solve rate（G_P=8）→ Eq.1 奖励
            → 组归一化 GRPO 更新 proposer（同组共享 s0）
        solver 相位 M_S 步：                # ALFWorld M_S=25
            任务池采样 → G=16 rollout（Eq.7 奖励）→ 组归一化 GRPO 更新 solver

⚠️ 与论文的**已知偏差**（复现计划 U5 + 工程取舍，全部可配置）：
* `tasks_per_step` 是全局 task-group 数，`group_size` 是每个全局 GRPO 组的大小；
  8 卡默认 TPS=1、G=16、每 rank 两条轨迹。
* EZ 使用 raw PEFT model 和显式 bucketed LoRA gradient all-reduce，不依赖 DDP
  对动态 adapter `requires_grad` 的注册；主线 DDP 训练路径保持独立。
* KL β 默认为 0.001，通过 `disable_adapter()` 的冻结 base reference 计算。
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
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
import yaml  # noqa: E402

from alfworld_env import ALFWorldBatchEnv, collect_game_files, load_alfworld_config  # noqa: E402
from ez_proposer import ProposerConfig, ProposerRollout, run_proposer_exploration  # noqa: E402
from ez_solver import EzSolverTrajectory, run_ez_solver_rollouts  # noqa: E402
from ez_swanlab import EzSwanLab  # noqa: E402
from ez_state import StateSnapshot  # noqa: E402
from ez_verify import ProposedTask, proposer_reward, validity_filter  # noqa: E402
from ez_dist import (broadcast_pool_indices, check_adapter_sync, gather_group_values,
                     broadcast_adapter_parameters, global_group_advantages, local_group_size, select_adapter,
                     sync_active_adapter_gradients)  # noqa: E402
from rollout import HFPolicy, RolloutConfig  # noqa: E402
from train_grpo import (build_model_and_tokenizer, grpo_loss, group_advantages,  # noqa: E402
                        is_main, load_cfg, log0, make_batch, per_token_logps,
                        set_seed, setup_dist)


_LOG_FH = None    # 本 rank 的日志：train_log.rank{rank}.jsonl
_LOG_FH0 = None   # 仅 rank0：额外的 train_log.jsonl（让现成的判读脚本不用改）
RANK = 0
_REPORTER: Any = None   # SwanLab 上报器（未启用时是 no-op 对象，见 ez_swanlab）


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


def tokenize_proposer_rollout(ro: ProposerRollout, policy, gen_cfg) -> ProposerRollout:
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
    """'参数真的动了吗' probe（主线 train_grpo 同款准入门）。

    ⚠️ 实测教训（job 12989450）：旧实现取"**第一个** numel>=4096 的 LoRA 张量"，
    按 `named_parameters()` 顺序正好命中 `layers.0.self_attn.q_proj.lora_A.<adapter>`。
    而 peft 把 **lora_B 初始化成全 0**：

        y = W x + B(A x) · s      ⇒   ∂L/∂A = Bᵀ(∂L/∂y) xᵀ ≡ 0     （B == 0 时）

    即该张量在第 0 步的梯度是**精确的 0**（不是"小"，是 0）。AdamW 里
    exp_avg = 0 ⇒ Δp = 0 ⇒ 打印 `update_rel=0.000e+00`。于是日志出现
    「真实 backward grad_norm=6.1474e+00（504 个张量有 grad）却 update_rel=0.000e+00」
    这种自相矛盾的指纹 —— **梯度链是通的，是 probe 挑错了张量**。

    新实现（三条都更硬）：
      * 对**所有带 grad 的参数**逐个测位移，报 max，而不是只看某一个固定张量；
      * 报"多少张量真的动了"（n_moved/n_total）—— 这才是"优化器到底动了没"的直接答案；
      * rel 用**有界**形式 ‖Δp‖/(‖p‖+‖Δp‖) ∈ [0,1)：zero-init 的 lora_B 分母不为 0，
        给 1.0 而不是 ∞，避免日志里冒出 1e8 这种没意义的数。
      * 再打 top3 张量名，出问题能直接定位到具体层。
    """
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


def _resume_signature(cfg: dict, world: int) -> dict:
    semantic = json.loads(json.dumps(cfg, sort_keys=True, default=str))
    run = semantic.get("run", {})
    run.pop("output_dir", None)
    run.pop("name", None)
    # These are operational checks/storage knobs, not training trajectory semantics.
    train = semantic.get("train", {})
    train.pop("check_adapter_sync_every", None)
    train.pop("checkpoint_every", None)
    model = semantic.get("model", {})
    if "path" in model:
        model["path"] = os.path.expandvars(str(model["path"]))
    env_cfg = semantic.get("env", {}).get("config_path")
    if env_cfg:
        path = Path(os.path.expandvars(str(env_cfg))).expanduser()
        if not path.is_absolute():
            path = SRC_DIR.parent / path
        if path.is_file():
            semantic["env"]["config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    canonical = json.dumps(semantic, sort_keys=True, separators=(",", ":"), default=str)
    return {"world_size": int(world), "semantic_config_sha256": hashlib.sha256(
        canonical.encode("utf-8")).hexdigest(), "semantic_config": semantic}


def ez_save(model, tokenizer, optimizer, run_dir: Path, tag: str, cfg: dict,
            extra: dict, task_pool: list, rank: int, world: int,
            uid_counter: int = 0, dry_run: bool = False, keep: int = 0) -> None:
    """Write complete resume state, then atomically promote it while retaining old ckpt."""
    import shutil
    ckpt_dir, tmp_dir, backup_dir = (run_dir / "ckpt", run_dir / "ckpt.tmp",
                                     run_dir / "ckpt.previous")
    if rank == 0:
        if not ckpt_dir.exists() and backup_dir.exists():
            os.replace(backup_dir, ckpt_dir)  # recover interruption between renames
        elif ckpt_dir.exists() and backup_dir.exists():
            ckpt_complete = (ckpt_dir / "train_state.json").is_file()
            backup_complete = (backup_dir / "train_state.json").is_file()
            if not ckpt_complete and backup_complete:
                shutil.rmtree(ckpt_dir, ignore_errors=True)
                os.replace(backup_dir, ckpt_dir)
            else:
                shutil.rmtree(backup_dir, ignore_errors=True)
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)
        torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
        with open(tmp_dir / "task_pool.jsonl", "w", encoding="utf-8") as fh:
            for task in task_pool:
                fh.write(json.dumps(task.to_json(), ensure_ascii=False) + "\n")
        state = {"checkpoint_schema_version": 2, "tag": tag, "config": cfg,
                 "resume_signature": _resume_signature(cfg, world),
                 "world_size": world, "dry_run": bool(dry_run), **extra}
        with open(tmp_dir / "train_state.json", "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, default=str)
        (tmp_dir / "rank_state").mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    rng_state = {"python": random.getstate(), "numpy": np.random.get_state(),
                 "torch_cpu": torch.get_rng_state(),
                 "torch_cuda": (torch.cuda.get_rng_state(torch.cuda.current_device())
                                if torch.cuda.is_available() else None),
                 "uid_counter": int(uid_counter)}
    torch.save(rng_state, tmp_dir / "rank_state" / f"rank{rank}.pt")
    if world > 1:
        dist.barrier()
    if rank == 0:
        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)
        if ckpt_dir.exists():
            os.replace(ckpt_dir, backup_dir)
        try:
            os.replace(tmp_dir, ckpt_dir)
        except Exception:
            if backup_dir.exists() and not ckpt_dir.exists():
                os.replace(backup_dir, ckpt_dir)
            raise
        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)
    if world > 1:
        dist.barrier()
    # ★「断点留多个」：keep>0 时把刚写完的最新断点完整复制成一份历史快照，
    #   供中间评测加载 solver adapter、以及从中间状态恢复。keep<=0 保持旧行为（只留最新）。
    #   ⚠️ 复制必须 rank0 做完后**再一起 barrier**：若放在上面的 barrier 之后，
    #   rank0 会落后其它 rank（它们在下一个 outer 的集合通信里干等 → 死锁）。
    if keep and keep > 0 and rank == 0:
        snap_dir = run_dir / "checkpoints"
        snap_dir.mkdir(parents=True, exist_ok=True)
        dest = snap_dir / f"ckpt-{tag}"
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        try:
            shutil.copytree(ckpt_dir, dest)
        except Exception as _e:  # noqa: BLE001
            log(f"[ckpt] 历史断点存档失败 {tag}: {_e}（继续训练，不影响最新断点）")
        snaps = sorted([p for p in snap_dir.iterdir()
                        if p.is_dir() and p.name.startswith("ckpt-")],
                       key=lambda p: p.stat().st_mtime)
        for _old in snaps[:-keep]:
            shutil.rmtree(_old, ignore_errors=True)
            log(f"[ckpt] 清理旧历史断点 {_old.name}（保留最近 {keep} 个）")
    if world > 1 and keep and keep > 0:
        dist.barrier()


def load_ez_resume(ckpt_dir: Path, cfg: dict, world: int, rank: int):
    with open(ckpt_dir / "train_state.json", encoding="utf-8") as fh:
        state = json.load(fh)
    if int(state.get("checkpoint_schema_version", 0)) < 2:
        raise ValueError("legacy checkpoint lacks complete optimizer/RNG/task-pool state; resume refused")
    if state.get("dry_run", False):
        raise ValueError("smoke/dry-run checkpoints cannot be resumed as formal training")
    expected = _resume_signature(cfg, world)
    if state.get("resume_signature") != expected:
        raise ValueError(f"resume config/world mismatch: saved={state.get('resume_signature')} current={expected}")
    pool_path = ckpt_dir / "task_pool.jsonl"
    if not pool_path.is_file():
        raise ValueError("resume checkpoint is missing task_pool.jsonl")
    pool = []
    with open(pool_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                if int(row.get("schema_version", 1)) < 2:
                    raise ValueError("legacy task pool lost raw fact multiplicity/identity; formal resume refused")
                task = ProposedTask.from_json(row)
                for snap in (task.s0, task.ref_final, *(s.ref_snapshot for s in task.subtasks)):
                    if snap.mode == "facts" and snap.facts and not snap.raw_facts:
                        raise ValueError("schema-2 task is missing raw facts; refusing unsafe resume")
                pool.append(task)
    rng_path = ckpt_dir / "rank_state" / f"rank{rank}.pt"
    if not rng_path.is_file():
        raise ValueError(f"resume checkpoint is missing rank-specific RNG state: {rng_path}")
    return state, pool, rng_path


def resolve_resume_directory(path: Path, rank: int) -> Path:
    """Use the last complete directory if a crash interrupted checkpoint promotion."""
    candidates = [path, path.parent / "ckpt.previous"]
    for candidate in candidates:
        required = [candidate / "train_state.json", candidate / "optimizer.pt",
                    candidate / "task_pool.jsonl", candidate / "proposer",
                    candidate / "solver", candidate / "rank_state" / f"rank{rank}.pt"]
        if all(p.exists() for p in required):
            return candidate
    raise ValueError(f"neither {path} nor its previous checkpoint is complete")


def mid_eval_solver(raw, policy, alf_cfg, games, *, max_steps, temperature,
                    max_new_tokens, max_context_tokens, batch_size, seed, log_fn):
    """在 valid_unseen 上评测当前 solver adapter（只读，不改权重、不动优化器）。

    判胜用官方 benchmark_done（ignore_benchmark_done=False），与训练内
    verify.final_match（状态匹配）口径不同 —— 这才是论文/官方 valid_unseen 成功率。
    rollout 复用 eval_ez._rollout_batch（已验证的官方评测链路）。
    """
    from alfworld_env import task_instance_id, task_type_of
    from eval_ez import _rollout_batch
    select_adapter(raw, "solver")
    raw.eval()
    eval_cfg = RolloutConfig(group_size=1, max_steps=max_steps,
                             max_new_tokens=max_new_tokens, temperature=temperature,
                             top_p=0.95, top_k=0, repetition_penalty=1.05,
                             max_context_tokens=max_context_tokens,
                             stop_mode="tool_call", max_batch_size=batch_size)
    won = []
    for start in range(0, len(games), batch_size):
        gb = games[start:start + batch_size]
        records = [{"task_id": task_instance_id(g), "task_type": task_type_of(g),
                    "sample": 0} for g in gb]
        env = ALFWorldBatchEnv(alf_cfg, split="eval_out_of_distribution",
                               batch_size=len(gb), seed=seed + start,
                               with_expert=False, with_facts=False,
                               ignore_benchmark_done=False)
        try:
            won.extend(r["won"] for r in _rollout_batch(
                env, policy, gb, records, eval_cfg, max_steps))
        finally:
            env.close()
    select_adapter(raw, "proposer")
    return (sum(won) / len(won)) if won else 0.0, won


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
    from train_grpo import per_token_logps
    if loss_norm != "seq":
        raise ValueError("EZ microbatch accumulation currently requires train.loss_norm='seq'")
    mb = max(1, int(micro_batch_size))
    local_valid = sum(1 for t in trajectories if any(t.loss_mask[1:]))
    global_valid = torch.tensor([local_valid], dtype=torch.float64, device=device)
    if world > 1:
        dist_obj.all_reduce(global_valid, op=dist_obj.ReduceOp.SUM)
    denom = max(float(global_valid.item()), 1.0)
    loss_sum = 0.0
    stats = []
    for start in range(0, len(trajectories), mb):
        bt = make_batch(trajectories[start:start + mb], pad_id, device)
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


# --------------------------------------------------------------------------- 主循环


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true", help="只跑 1 步 proposer + 1 步 solver 冒烟")
    ap.add_argument("--max-outer", type=int, default=None)
    ap.add_argument("--resume", default=None, help="resume from a complete EZ ckpt directory")
    args = ap.parse_args()
    if args.resume and args.dry_run:
        ap.error("--resume cannot be combined with --dry-run; resume a formal run without changing its schedule")

    install_alfworld_compat()
    cfg = load_cfg(args.config)
    if os.environ.get("EZ_RUN_OUTPUT_DIR"):
        cfg.setdefault("run", {})["output_dir"] = os.environ["EZ_RUN_OUTPUT_DIR"]
    seed = cfg["run"].get("seed", 42)
    # ★ 必须用 setup_dist()（会 init_process_group），不能用 dist_info()：
    #   dist_info() 只读环境变量，从不建进程组 —— 8 进程照跑，但 all_reduce /
    #   broadcast 全部抛 "Default process group has not been initialized"。
    rank, world, local = setup_dist()
    global _RANK
    _RANK = rank
    set_seed(seed, rank)

    resume_ckpt = Path(args.resume).expanduser().resolve() if args.resume else None
    if resume_ckpt is not None:
        resume_ckpt = resolve_resume_directory(resume_ckpt, rank)
        run_dir = resume_ckpt.parent
    elif is_main(rank):
        run_dir = Path(cfg["run"]["output_dir"]) / \
            f"{cfg['run']['name']}-{time.strftime('%m%d-%H%M')}"
    else:
        run_dir = Path(".")          # 占位，马上被广播覆盖
    if world > 1:
        # ★ 各 rank 不能各自取时间命名：跨分钟边界时 rank0 与 rank7 会落到**两个目录**，
        #   表现为"rank0 有日志、rank7 的文件不存在"，很难查。统一由 rank0 决定。
        _rd = [str(run_dir)]
        dist.broadcast_object_list(_rd, src=0)
        run_dir = Path(_rd[0])
    run_dir.mkdir(parents=True, exist_ok=True)
    task_pool_path = run_dir / "task_pool.jsonl"
    resume_info = None
    resume_pool: List[ProposedTask] = []
    resume_rng_path = None
    if resume_ckpt is not None:
        resume_info, resume_pool, resume_rng_path = load_ez_resume(resume_ckpt, cfg, world, rank)
        log(f"[resume] checkpoint={resume_ckpt} next={resume_info.get('next_outer')} "
            f"{resume_info.get('next_phase')}:{resume_info.get('next_step')} pool={len(resume_pool)}")
    # ★ SwanLab 上报在这里起：放后面是因为 `run_dir` 必须已经由 rank0 广播出去了，
    #   否则 8 个 rank 会各自开出 8 个不同名的 run。
    #   ⚠️ 上报失败不影响训练（ez_swanlab 全吞异常），所以直接挂着不管也行。
    global _REPORTER
    _REPORTER = EzSwanLab(cfg, run_dir=run_dir, rank=rank)
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
    N = args.max_outer if args.max_outer is not None else tcfg.get("outer_loops", 4)
    M_P, M_S = tcfg.get("proposer_steps", 25), tcfg.get("solver_steps", 25)
    G = tcfg.get("group_size", 16)           # GRPO 组
    G_P = tcfg.get("solve_rate_rollouts", 8)  # 估 p̂(x) 用
    TPS = int(tcfg.get("tasks_per_step", 1))  # global task groups per step
    G_local = local_group_size(int(G), world)
    w_sub = tcfg.get("w_sub", 0.3)
    w_div = tcfg.get("w_div", 0.3)
    # ★「断点留多个」+「训练中间评测」开关（见 ez_save 的 keep 与 mid_eval_solver）
    checkpoint_keep = int(tcfg.get("checkpoint_keep", 4))
    eval_every_outer = int(tcfg.get("eval_every_outer", 1))
    eval_n_tasks = int(tcfg.get("eval_n_tasks", 16))
    eval_batch = int(tcfg.get("eval_batch", 2))
    eval_max_steps = int(tcfg.get("eval_max_steps", cfg["solver"].get("max_turns", 30)))
    # ★ solve rate 每批并行处理的任务数（TPS>1 时把"逐任务串行"改成批量并行 rollout）
    sr_tasks = max(1, int(tcfg.get("solve_rate_batch_tasks", 4)))

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

    # ★ --dry-run 只把"步数"降到 1 是**不够**的：G=16 × 30 turns 在 A40 debug 分区
    #   （30 min 上限）根本跑不完，会卡到超时。这里把 group / turns / tasks 一并压到
    #   最小，冒烟只验证三件事：链路能通、facts 通道存在、参数真的动了。
    #
    #   ⚠️ turns 不能压到 3！prompt 要求"探索 10-20 个动作后才 finish"，turns=3 时
    #   proposer 必然在 finish 之前就把回合用光 -> accept 恒为 0 -> 任务池空 ->
    #   solver 相位整个跳过 -> 冒烟"看起来通过"但其实一条真链路都没走。
    #   实测证据（job 12989069）：fmt_bad=0（解析已修好）但 accept=0/2、mean_turns=3.0。
    #   12 轮足够让 Qwen3-4B 走到 finish，同时 2 slot × 12 轮在 A40 上约 40s。
    if args.dry_run:
        _smoke_default_g = int(tcfg.get("group_size", 16))
        G = int(os.environ.get("EZ_SMOKE_G", str(_smoke_default_g)))
        G_P = int(os.environ.get("EZ_SMOKE_G_P", str(max(2, world))))
        # ★ 显式默认 1：冒烟必须最小规模。**不能**再用 str(TPS) 兜底 —— 正式配置
        #   TPS 一旦调大（如 8），兜底会把冒烟也一起放大 8 倍，违背本节"压到最小"的初衷。
        TPS = int(os.environ.get("EZ_SMOKE_TPS", "1"))
        G_local = local_group_size(int(G), world)
        M_P = M_S = 1
        _smoke_turns = int(os.environ.get("EZ_SMOKE_TURNS", "12"))
        p_max_turns = s_max_turns = _smoke_turns
        log(f"[dry-run] 缩规模：G={G} G_P={G_P} TPS={TPS} turns={_smoke_turns}")
    else:
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
    # 中间评测用 valid_unseen（官方口径）。排序保证各次评测/各 rank 一致可比。
    eval_game_files = sorted(collect_game_files(alf_cfg, "eval_out_of_distribution"))

    p_cfg = ProposerConfig(
        group_size=G_local, max_turns=p_max_turns,
        max_new_tokens=cfg["proposer"].get("max_new_tokens", 160),
        temperature=cfg["proposer"].get("temperature", 1.0),
        give_admissible=cfg["proposer"].get("give_admissible", False),
        invalid_action_mode=cfg["proposer"].get("invalid_action_mode", "raw"),
        finish_nudge=cfg["proposer"].get("finish_nudge", True),
        generation_max_batch_size=cfg["train"].get("generation_max_batch_size", 2))
    s_cfg = RolloutConfig(
        group_size=1, max_steps=s_max_turns,
        max_new_tokens=cfg["solver"].get("max_new_tokens", 96),
        temperature=cfg["solver"].get("temperature", 1.0),
        max_context_tokens=cfg["solver"].get("max_context_tokens", 40960),
        keep_recent_turns=cfg["solver"].get("keep_recent_turns", 12),
        stop_mode="tool_call",
        max_batch_size=cfg["train"].get("generation_max_batch_size", 2))
    # proposer 训练序列裁剪专用 cfg：必须与生成端 p_cfg 一致，否则训练上下文≠采样上下文
    p_tok_cfg = RolloutConfig(
        group_size=1, max_steps=p_cfg.max_turns,
        max_new_tokens=p_cfg.max_new_tokens, temperature=p_cfg.temperature,
        max_context_tokens=p_cfg.max_context_tokens,
        keep_recent_turns=p_cfg.keep_recent_turns, stop_mode="tool_call")

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
        f"stop_mode=tool_call | 基座冻结={not any(p.requires_grad for n, p in raw.named_parameters() if 'lora_' not in n)} "
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
        optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu", weights_only=False))
        rng_state = torch.load(resume_rng_path, map_location="cpu", weights_only=False)
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch_cpu"])
        if torch.cuda.is_available() and rng_state.get("torch_cuda") is not None:
            torch.cuda.set_rng_state(rng_state["torch_cuda"], torch.cuda.current_device())
        resume_uid_counter = int(rng_state.get("uid_counter", 0))
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

    # ★ 评测链路自测（EZ_EVAL_PRETEST=1 时）：训练前先在 valid_unseen 上评一次，
    #   确保 mid_eval_solver 不炸。评测是本次新增，正式 8 卡作业里第一次触发在
    #   第一个 outer 结束（30-60min 后）——提前 3 分钟验证，把「评测 bug 烧 8 卡 1 小时」
    #   降成「烧 8 卡 3 分钟」。
    if os.environ.get("EZ_EVAL_PRETEST") == "1" and not args.dry_run:
        if world > 1:
            dist.barrier()
        if is_main(rank):
            _pt0 = time.time()
            _pt_n = max(1, min(eval_n_tasks, 4))
            _sr0, _won0 = mid_eval_solver(
                raw, policy, alf_cfg, eval_game_files[:_pt_n],
                max_steps=eval_max_steps,
                temperature=cfg["solver"].get("temperature", 1.0),
                max_new_tokens=cfg["solver"].get("max_new_tokens", 96),
                max_context_tokens=cfg["solver"].get("max_context_tokens", 40960),
                batch_size=eval_batch, seed=seed,
                log_fn=log)
            log(f"[eval-pretest] 评测链路 OK：SR={_sr0:.4f} "
                f"({int(sum(_won0))}/{len(_won0)}，{(time.time() - _pt0) / 60:.1f}min)")
        if world > 1:
            dist.barrier()

    t_start = time.time()
    for outer in range(start_outer, N):
        # ⚠️ 冒烟只跑 1 个 outer 的判据必须放在**循环开头**。
        #    之前写在循环结尾（ez_save 之后），但 solver 空池那条路径会 `continue`
        #    直接跳到下一轮，把 break 绕过去 —— 实测日志里出现了 outer 0/1/2/3 各跑一遍。
        if args.dry_run and outer > 0:
            break
        # ================= proposer 相位 =================
        model.eval()
        p_steps = M_P if not args.dry_run else 1
        p_start = resume_step if outer == start_outer and resume_phase == "proposer" else 0
        p_range = range(p_start, p_steps) if not (outer == start_outer and resume_phase == "solver") else range(0)
        for step in p_range:
            st0 = time.time()
            _cuda_sync_and_reset_peak()
            model.eval()
            select_adapter(raw, "proposer")
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
                    f" 各 rank 的 backward 次数会不一致，DDP 必然死锁。"
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
            _tokens_local = 0
            _backward_t0 = time.time()
            for group_idx, groll in enumerate(groups):
                for c in groll:
                    tokenize_proposer_rollout(c, policy, p_tok_cfg)
                    _tokens_local += int(sum(c.__dict__["loss_mask"][1:]))
                full_rewards = gather_group_values([c.reward for c in groll], rank, world, dist)
                full_advs = global_group_advantages(full_rewards, ddof=adv_ddof)
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
                    group_weight=1.0 / max(len(groups), 1))
                _p_loss_sum += _g_loss
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
            # ★ A/B 分开报：peft 把 lora_B 初始化成全 0 ⇒ ∂L/∂A = Bᵀ(...) ≡ 0
            #   ⇒ **第 0 步 lora_A 的梯度必然是精确的 0**（这是 LoRA 的正常性质，
            #   不是 bug）。旧 probe 恰好挑中 lora_A，才打出 update_rel=0.000e+00 的假警报。
            #   把两条分开打，这个性质就永远看得见。
            _gnA = float(sum(p.grad.detach().float().norm() for p in _act
                             if ".lora_A." in _nof.get(id(p), "")))
            _gnB = float(sum(p.grad.detach().float().norm() for p in _act
                             if ".lora_B." in _nof.get(id(p), "")))
            _step_t0 = time.time()
            torch.nn.utils.clip_grad_norm_(opt_params, max_grad_norm)
            update_probe(raw, opt_params, optimizer, f"outer {outer} P{step}")
            if step == 0 or (check_sync_every > 0 and (step + 1) % check_sync_every == 0):
                _sync_max = check_adapter_sync(raw, "proposer", world, dist)
                log(f"[adapter-sync] phase=proposer step={step} status=PASS max_abs={_sync_max:.3e}")
            optimizer.zero_grad(set_to_none=True)
            _cuda_sync()
            _optimizer_s = time.time() - _step_t0
            # ★ 组奖励必须进日志：只有这样，"update_rel=0" 才能被判成
            #   「奖励同分（=没信号，正常）」还是「奖励有区分度却不动（=梯度链断，必修）」。
            #   实测教训（job 12989143 outer 2）：accept=1/2 却 update_rel=0，
            #   当时日志里连组奖励都没有，完全无法判断 —— 这就是补它的原因。
            _rw = [round(float(c.reward), 3) for c in cand]
            # ★ 拒绝原因分布必须汇总进日志（`EZ_EXPERIMENT_PLAN.md` E0）：
            #   光有 `accept=1/4` 只能知道"合格率低"，看不出**是哪条规则在杀题**。
            #   `ez_proposer` 已把 reason 存在每条 rollout 的 `reject_reason` 上，
            #   这里只差汇总。看分布才能决定该放宽过滤规则（U1/U3）
            #   还是该做 base 模型的格式冷启动（U7）。
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
            _p_metric = torch.tensor([_p_loss_sum], device=device, dtype=torch.float64)
            if world > 1:
                dist.all_reduce(_p_metric, op=dist.ReduceOp.SUM)
            _p_loss_sum_global = float(_p_metric.item())
            del _p_metric
            # ---- SwanLab 上报：proposer 相位 ----
            # step 必须是**全局单调递增**的，否则 P/S 两条曲线会叠在同一 x 上。
            # 只有 rank0 真的会发（其它 rank 的 _REPORTER 是 no-op 对象）。
            _REPORTER.log({
                "train/loss": _p_loss_sum_global,
                "train/grad_norm": _gn,
                "train/grad_norm_lora_A": _gnA,
                "train/grad_norm_lora_B": _gnB,
                "proposer/reward_mean": float(np.mean(_rw)) if _rw else 0.0,
                "proposer/reward_spread": max(_rw) - min(_rw) if _rw else 0.0,
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
                        uid_counter=uid_counter, dry_run=args.dry_run)
            if args.dry_run:
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
                    task_pool, rank, world, uid_counter=uid_counter, dry_run=args.dry_run)
            continue
        s_steps = M_S if not args.dry_run else 1
        s_start = resume_step if outer == start_outer and resume_phase == "solver" else 0
        for step in range(s_start, s_steps):
            st0 = time.time()
            _cuda_sync_and_reset_peak()
            model.eval()
            # ★★ 采样改为**按 rank 分片**，不用 `rng.sample`（多卡下是真 bug，见下）
            #   旧写法 `rng.sample(outer_pool, k)`：rng 是 `random.Random(seed)`，
            #   各 rank 的 seed **完全相同**，而池子经 all_gather 后也完全一致 ——
            #   于是所有 rank 采到**同一批任务**。实测 8 卡 + k=1 时
            #   rank1..rank7 全部采到同一个任务，全局只用到 2 个不同任务
            #   （有效 batch 缩水到 25%，8 卡的算力白扔 6/8）。
            #   改成分片后：跨 rank **天然不重复**，也不再依赖 rng 状态同步。
            #
            # 无放回：有放回会让同 uid 的组被合并、loss 双计（review P1-2）
            # ★ k 必须**各 rank 相同**（=min(TPS//world, 池子大小)）：条数不同则
            #   backward 次数不同，DDP 死锁。池子已跨 rank 一致，所以这个 k 天然一致。
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
            all_roll: List[EzSolverTrajectory] = []
            _tokens_local = 0
            _backward_t0 = time.time()
            for grp in solver_groups:
                full_rewards = gather_group_values([r.traj.reward for r in grp], rank, world, dist)
                full_advs = global_group_advantages(full_rewards, ddof=adv_ddof)
                for r, a in zip(grp, full_advs[rank::world]):
                    r.traj.advantage = a
                _g_loss, _g_stats, _g_valid = backward_microbatches(
                    raw, model, [r.traj for r in grp], pad_id, device, adapter="solver",
                    clip_eps=clip_eps, micro_batch_size=micro_batch_size, world=world,
                    loss_norm=loss_norm, beta=kl_beta, dist_obj=dist,
                    group_weight=1.0 / max(len(solver_groups), 1))
                _s_loss_sum += _g_loss
                all_roll.extend(grp)
                _tokens_local += sum(int(sum(r.traj.loss_mask[1:])) for r in grp)
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
            update_probe(raw, opt_params, optimizer, f"outer {outer} S{step}")
            if step == 0 or (check_sync_every > 0 and (step + 1) % check_sync_every == 0):
                _sync_max = check_adapter_sync(raw, "solver", world, dist)
                log(f"[adapter-sync] phase=solver step={step} status=PASS max_abs={_sync_max:.3e}")
            optimizer.zero_grad(set_to_none=True)
            _cuda_sync()
            _optimizer_s = time.time() - _step_t0
            rw = [r.traj.reward for r in all_roll]
            _s_metric = torch.tensor([_s_loss_sum], device=device, dtype=torch.float64)
            if world > 1:
                dist.all_reduce(_s_metric, op=dist.ReduceOp.SUM)
            _s_loss_sum_global = float(_s_metric.item())
            del _s_metric
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
            # ---- SwanLab 上报：solver 相位 ----
            # step 要排在同 outer 的 P 相位**之后**（+M_P），否则两条曲线会撞车。
            _sanity = {
                "train/loss": _s_loss_sum_global,
                "train/grad_norm": _gnS,
                "train/grad_norm_lora_A": _gnSA,
                "train/grad_norm_lora_B": _gnSB,
                "solver/reward_mean": float(np.mean(rw)) if rw else 0.0,
                "solver/final_match_rate": (sum(r.verify.final_match for r in all_roll)
                                            / max(len(all_roll), 1)),
                "solver/n_rollouts": len(all_roll),
                "solver/sec_per_step": time.time() - st0,
            }
            _REPORTER.log(_sanity, step=outer * (M_P + M_S) + M_P + step)
            # solver 一步都没解出来 —— 这条要能在手机端看到，而不是等回来看日志
            if rw and max(rw) == 0.0:
                _REPORTER.alert(
                    title="solver 奖励全 0",
                    text=f"outer {outer} S{step}：{len(all_roll)} 条 rollout 奖励全为 0，"
                         f"本轮可能发生了 reward collapse，去看 train_log.rank0.jsonl")
            s_done = step + 1 == s_steps
            if s_done or (step + 1) % 5 == 0:
                next_outer, next_phase, next_step = ((outer + 1, "proposer", 0) if s_done
                                                     else (outer, "solver", step + 1))
                ez_save(raw, tokenizer, optimizer, run_dir, f"outer{outer}-S{step}", cfg,
                        {"next_outer": next_outer, "next_phase": next_phase,
                         "next_step": next_step}, task_pool, rank, world,
                        uid_counter=uid_counter, dry_run=args.dry_run,
                        keep=(checkpoint_keep if s_done else 0))
                # ★ 中间评测：每个 outer 完成时，rank0 在 valid_unseen 上评一次 solver，
                #   其余 rank 在 barrier 等待。结果进日志 + SwanLab，用户能实时看进展。
                #   EZ_SMOKE_EVAL=1 时 dry-run 也评测（冒烟顺带验证评测链路，task 数压到 4）。
                _force_eval = bool(int(os.environ.get("EZ_SMOKE_EVAL", "0")))
                if s_done and (not args.dry_run or _force_eval) and eval_every_outer > 0 \
                        and (outer + 1) % eval_every_outer == 0:
                    if world > 1:
                        dist.barrier()
                    if is_main(rank):
                        _n = min(eval_n_tasks, 4) if args.dry_run else eval_n_tasks
                        _eg = eval_game_files[:_n] if _n > 0 else eval_game_files
                        _t0 = time.time()
                        _sr, _won = mid_eval_solver(
                            raw, policy, alf_cfg, _eg,
                            max_steps=eval_max_steps,
                            temperature=cfg["solver"].get("temperature", 1.0),
                            max_new_tokens=cfg["solver"].get("max_new_tokens", 96),
                            max_context_tokens=cfg["solver"].get("max_context_tokens", 40960),
                            batch_size=eval_batch, seed=seed + outer + 1,
                            log_fn=log)
                        log(f"[eval] outer {outer} 完成 valid_unseen SR={_sr:.4f} "
                            f"({int(sum(_won))}/{len(_won)}，{len(_eg)} tasks，"
                            f"{(time.time() - _t0) / 60:.1f}min)")
                        _REPORTER.log({
                            "eval/valid_unseen_sr": _sr,
                            "eval/n_won": int(sum(_won)),
                            "eval/n_tasks": len(_eg),
                        }, step=(outer + 1) * (M_P + M_S))
                    if world > 1:
                        dist.barrier()
            if args.dry_run:
                break

        # ★ 只在 rank0 存档：peft 保存的是 adapter（几百 MB），8 个 rank 各存一份
        #   既浪费又会在同一路径上互相踩。（deepspeed 靠不同的 zero stage 存分片，
        #   我们不走那条路。）
    close_all()
    persist_pool()
    # ★ dry-run 只跑 outer 0，但 N 仍是配置里的 4 —— 直接打 N 会让人以为跑了 4 轮
    _n_done = 1 if (args.dry_run and N > 0) else N
    log(f"[done] rank={rank} {_n_done} outer loops in {(time.time()-t_start)/60:.1f} min -> {run_dir}")
    # ⚠️ finish 必须在**关日志文件之后、destroy_process_group 之前**：
    #    SwanLab 收尾会把缓存的指标 flush 到云端/本地，放在 barrier 后面
    #    可能因为进程组已销毁而卡住（别的 rank 还在等它参与集合通信）。
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
