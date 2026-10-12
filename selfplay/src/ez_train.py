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
* 论文 "batch 16 / rollouts per step 16" 的精确编排没写死；本实现用
  `tasks_per_step` 个任务 × G 条 rollout，默认 2×16=32（单节点 HF generate
  吞吐下的务实值），8 卡 DDP 分片留 TODO。
* KL penalty（β=0.001 对参考策略）：我们的 GRPO 内核是 β=0 无参考模型版，
  KL 项暂以 TODO 标注（LoRA 低 lr 下影响小）。
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alfworld_compat import install as install_alfworld_compat  # noqa: E402

install_alfworld_compat()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import yaml  # noqa: E402

from alfworld_env import ALFWorldBatchEnv, collect_game_files, load_alfworld_config  # noqa: E402
from ez_proposer import ProposerConfig, ProposerRollout, run_proposer_exploration  # noqa: E402
from ez_solver import EzSolverTrajectory, run_ez_solver_rollouts  # noqa: E402
from ez_state import StateSnapshot  # noqa: E402
from ez_verify import ProposedTask, proposer_reward, validity_filter  # noqa: E402
from rollout import HFPolicy, RolloutConfig, build_token_sequence  # noqa: E402
from train_grpo import (build_model_and_tokenizer, grpo_loss, group_advantages,  # noqa: E402
                        is_main, load_cfg, log0, make_batch, per_token_logps,
                        set_seed, setup_dist)


_LOG_FH = None    # 本 rank 的日志：train_log.rank{rank}.jsonl
_LOG_FH0 = None   # 仅 rank0：额外的 train_log.jsonl（让现成的判读脚本不用改）
_RANK = 0


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


# --------------------------------------------------------------------------- LoRA


def build_dual_lora(model, cfg: dict):
    """proposer 和 solver 是**两个独立策略**（论文 §3：同一基座初始化、各自训练）。

    实现：共享基座 + peft 两个 LoRA adapter（"proposer" / "solver"），
    相位切换用 `model.set_adapter(name)`。全参训练需两份完整权重，
    4B fp32 显存放不下 —— 这正是论文用 LoRA 的原因之一。
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise SystemExit("Explorer-Zero 需要双策略 LoRA：pip install peft") from exc
    lc = LoraConfig(r=cfg["model"].get("lora_r", 32),
                    lora_alpha=cfg["model"].get("lora_alpha", 32),
                    lora_dropout=0.0,
                    target_modules=cfg["model"].get("lora_targets", "all-linear"))
    model = get_peft_model(model, lc, adapter_name="proposer")
    model.add_adapter("solver", lc)
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
    # ★ 8 卡 DDP 红线 C：DDP 在**构造那一刻**按"哪些参数 requires_grad"固定注册
    #   all-reduce hook（决定 bucket 成员）。之后即使某参数再变成 requires_grad=True，
    #   它的梯度也**永远不会被同步** —— 而且是静默错，loss 照跑、只有多卡不一致。
    #   所以两个 adapter 必须在 wrap 之前就都是 trainable。
    #   相位隔离不再靠 requires_grad，而靠"没参与前向 → unused → 跳过"
    #   （配 DDP 的 find_unused_parameters=True）。
    n_tr = 0
    for _n, p in model.named_parameters():
        if "lora_" in _n and not p.requires_grad:
            p.requires_grad_(True)
            n_tr += 1
    if n_tr:
        print(f"[ez] 打开 {n_tr} 个 LoRA 张量的 requires_grad（DDP wrap 前必须，红线 C）")
    model.print_trainable_parameters()
    return model


# --------------------------------------------------------------------------- proposer 训练序列


def tokenize_proposer_rollout(ro: ProposerRollout, policy, gen_cfg) -> ProposerRollout:
    """把探索消息历史裁成训练序列（与主线 run_rollouts 同一套 truncate 逻辑）。

    `gen_cfg` 必须用 proposer 专用的裁剪配置（p_tok_cfg）：生成端按同样的窗口截断，
    训练/采样上下文才一致（review P1-1）。
    """
    tr_msgs = policy.truncate_messages(ro.messages, gen_cfg)
    n_asst = sum(1 for m in tr_msgs if m["role"] == "assistant")
    gen_kept = ro.gen_ids[len(ro.gen_ids) - n_asst:] if n_asst else []
    ids, mask, _ = build_token_sequence(tr_msgs, gen_kept, policy)
    ro.__dict__["input_ids"] = ids       # 动态挂字段，避免改 dataclass 影响面
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


def ez_save(model, tokenizer, run_dir: Path, tag: str, cfg: dict, extra: dict) -> None:
    """EZ 存档：peft 模型走自己的 save_pretrained（保存基座引用 + 两个 adapter），
    替代主线 _save_model_manual（那条路是为裸 HF 模型写的）。仍遵守铁律二：只留一个。"""
    import shutil
    ckpt_dir = run_dir / "ckpt"
    tmp_dir = run_dir / "ckpt.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(tmp_dir)          # peft：adapter 权重 + 配置
    tokenizer.save_pretrained(tmp_dir)
    with open(tmp_dir / "train_state.json", "w") as fh:
        json.dump({"tag": tag, "config": cfg, **extra}, fh, indent=2, default=str)
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    tmp_dir.rename(ckpt_dir)


def grad_chain_selfcheck(model, tokenizer, device) -> Dict[str, float]:
    """启动自检：证明**两个 adapter 各自都真的能接到梯度**。

    这条自检刻意不依赖环境、不依赖奖励、不依赖模型能不能完成任务，几秒钟跑完。
    它回答的是一个和"奖励有没有区分度"**完全无关**的问题：

        前向 -> 该 adapter 的 LoRA 参数 -> loss -> 反向 -> p.grad 非空 ？

    为什么必须单独测：双 adapter + `set_adapter` 的隔离语义、peft 的包装、
    gradient_checkpointing、以及（将来到 8 卡时）DDP 注册 hook 的时机，
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
            model.set_adapter(name)
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
    model.set_adapter("proposer")
    return out


def _borrow_batch(all_roll, pad_id: int, device):
    """从已有的 rollout 里借一条造"占位 batch"，用来保持各 rank 的 backward 次数一致。

    DDP 的 all-reduce 是集合通信：某个 rank 少做一次 backward，其它 rank 就会
    永久卡在 NCCL barrier（红线 B）。空组本身不该出现，但**出现了也绝不能 continue** ——
    这里用"零 loss 但保留计算图"的 backward 顶上，次数对齐、梯度为 0、结果不受影响。
    """
    for r in all_roll:
        if getattr(r.traj, "input_ids", None):
            return make_batch([r.traj], pad_id, device)
    return None


# --------------------------------------------------------------------------- 主循环


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true", help="只跑 1 步 proposer + 1 步 solver 冒烟")
    ap.add_argument("--max-outer", type=int, default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    seed = cfg["run"].get("seed", 42)
    # ★ 必须用 setup_dist()（会 init_process_group），不能用 dist_info()：
    #   dist_info() 只读环境变量，从不建进程组 —— 8 进程照跑，但 all_reduce /
    #   broadcast 全部抛 "Default process group has not been initialized"。
    rank, world, local = setup_dist()
    global _RANK
    _RANK = rank
    set_seed(seed, rank)

    if is_main(rank):
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
    global _LOG_FH, _LOG_FH0
    _LOG_FH = open(run_dir / f"train_log.rank{rank}.jsonl", "w")
    _LOG_FH0 = open(run_dir / "train_log.jsonl", "w") if is_main(rank) else None
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
    TPS = tcfg.get("tasks_per_step", 2)       # U5 偏差：每步任务数
    w_sub = tcfg.get("w_sub", 0.3)
    w_div = tcfg.get("w_div", 0.3)

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
        G = int(os.environ.get("EZ_SMOKE_G", "2"))
        G_P = int(os.environ.get("EZ_SMOKE_G_P", "2"))
        TPS = int(os.environ.get("EZ_SMOKE_TPS", "1"))
        M_P = M_S = 1
        _smoke_turns = int(os.environ.get("EZ_SMOKE_TURNS", "12"))
        p_max_turns = s_max_turns = _smoke_turns
        log(f"[dry-run] 缩规模：G={G} G_P={G_P} TPS={TPS} turns={_smoke_turns}")
    else:
        p_max_turns = cfg["proposer"].get("max_turns", 30)
        s_max_turns = cfg["solver"].get("max_turns", 30)

    # ---- ★ 8 卡 DDP 红线 B：每个 rank 的 backward 次数必须**完全相同** ----
    #   DDP 的 all-reduce 是集合通信：rank0 做了 3 次 backward 而 rank7 做了 0 次，
    #   就会永久卡在 NCCL barrier 上（直到 30 分钟 timeout 才 abort，等于白排一次队）。
    #   分片后每 rank 处理 TPS/world 个 game，所以 TPS 必须是 world 的正整数倍。
    #   配置默认 TPS=2 是**单卡**的务实值；8 卡时直接断言会死在起跑线，因此这里
    #   向上取整到 world 的倍数并**大声抱怨** —— 组语义不变，只是全局 batch 变大。
    if world > 1 and (TPS % world or TPS < world):
        TPS_new = max(world, ((TPS + world - 1) // world) * world)
        log(f"[dist] ⚠️ tasks_per_step={TPS} 不能被 world={world} 整除（或小于 world）。"
            f"每 rank 的 backward 次数必须一致，否则 DDP 直接死锁（红线 B）。"
            f"已自动上调为 TPS={TPS_new}（每 rank {TPS_new // world} 个 game）。"
            f"想固定请设 train.tasks_per_step 或 EZ_SMOKE_TPS 为 {world} 的倍数")
        TPS = TPS_new

    # ---- 环境（facts 通道是 EZ 的命根子，必须开）----
    alf_cfg = load_alfworld_config(
        cfg["env"]["config_path"] if os.path.isabs(cfg["env"]["config_path"])
        else str((SRC_DIR.parent / cfg["env"]["config_path"]).resolve()))
    game_files = collect_game_files(alf_cfg, split=cfg["env"].get("split", "train"))
    log(f"[data] {len(game_files)} train games")

    p_cfg = ProposerConfig(
        group_size=G, max_turns=p_max_turns,
        max_new_tokens=cfg["proposer"].get("max_new_tokens", 160),
        temperature=cfg["proposer"].get("temperature", 1.0),
        give_admissible=cfg["proposer"].get("give_admissible", False))
    s_cfg = RolloutConfig(
        group_size=1, max_steps=s_max_turns,
        max_new_tokens=cfg["solver"].get("max_new_tokens", 96),
        temperature=cfg["solver"].get("temperature", 1.0),
        max_context_tokens=cfg["solver"].get("max_context_tokens", 40960),
        keep_recent_turns=cfg["solver"].get("keep_recent_turns", 12),
        stop_mode="tool_call")
    # proposer 训练序列裁剪专用 cfg：必须与生成端 p_cfg 一致，否则训练上下文≠采样上下文
    p_tok_cfg = RolloutConfig(
        group_size=1, max_steps=p_cfg.max_turns,
        max_new_tokens=p_cfg.max_new_tokens, temperature=p_cfg.temperature,
        max_context_tokens=p_cfg.max_context_tokens,
        keep_recent_turns=p_cfg.keep_recent_turns, stop_mode="tool_call")

    env_p = ALFWorldBatchEnv(alf_cfg, split=cfg["env"].get("split", "train"),
                             batch_size=G, max_episode_steps=p_cfg.max_turns,
                             with_facts=True)
    env_s = ALFWorldBatchEnv(alf_cfg, split=cfg["env"].get("split", "train"),
                             batch_size=G, max_episode_steps=s_cfg.max_steps,
                             with_facts=True)
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
                                      batch_size=G_P, max_episode_steps=s_cfg.max_steps,
                                      with_facts=True)
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
    # ★ wrap_ddp=False：DDP 必须等**双 adapter 都 trainable 之后**才包（红线 C）。
    raw, tokenizer = build_model_and_tokenizer(cfg, local, world, wrap_ddp=False)
    raw = build_dual_lora(raw, cfg)

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
                                with_facts=True)
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
    #   在 wrap **之前**跑：DDP 包装后会插 forward hook，自检的裸 forward/backward
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

    # ---- ★ DDP（在双 adapter 都可训练之后才包）----
    #   find_unused_parameters=True 是**必须**的：proposer 相位里 solver adapter
    #   不参与前向，它的 bucket 拿不到梯度；不设 True 会直接报
    #   "Expected to have finished reduction in the prior iteration" 或静默卡住。
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as _DDP
        model = _DDP(raw, device_ids=[local], output_device=local,
                     find_unused_parameters=True)
        log(f"[dist] DDP 已包装：world_size={world} find_unused_parameters=True；"
            f"两个 adapter 合计 {sum(p.numel() for p in _lora_all)/1e6:.1f}M 参数，"
            f"每步 all-reduce 的是 **fp32 梯度** ≈{sum(p.numel() for p in _lora_all)*4/1e9:.2f} GB"
            f"（LoRA 让 DDP 在这里非常划算，不需要 ZeRO/FSDP）")
        # ★ 探针：DDP 到底转不转发 peft/HF 的方法？不同 torch 版本行为不同，
        #   实测（torch 2.x, job 12989565）是**不转发** —— 直接调就 AttributeError。
        #   把它写进日志，以后升级 torch 时一眼能看出行为有没有变。
        log(f"[dist] 属性转发探针：hasattr(DDP,'set_adapter')={hasattr(model, 'set_adapter')} "
            f"hasattr(DDP,'generate')={hasattr(model, 'generate')} "
            f"（False 表示必须用内层 raw 调这些方法；loss 的 forward 仍要用包装后的 model）")
    else:
        model = raw
    # ★★ DDP 的分工（实测踩过一次，见 job 12989565）：
    #   `DistributedDataParallel` **不转发** unknown 属性 —— `model.set_adapter(...)`
    #   会直接抛 `AttributeError: 'DistributedDataParallel' object has no attribute
    #   'set_adapter'`；`generate` 同理。所以：
    #     · **peft / HF 自己的方法**（set_adapter / generate / save_pretrained /
    #       print_trainable_parameters）→ 一律用内层 `raw`
    #     · **只有"要算梯度的那次 forward"** → 必须用包装后的 `model`，
    #       否则绕过 DDP 的 forward hook，梯度永远不会 all-reduce（且不报错）
    #     · `train()/eval()/zero_grad()` 是 nn.Module 自己的方法，用 `model`（包装后）
    #       才对 —— DDP 要靠 train() 重置 require_backward_grad_sync
    #   由于 HFPolicy 内部走 `generate`，它必须持有 `raw`；而 loss 的 forward
    #   由 ez_train 显式把 `model` 传进 `per_token_logps/grpo_loss`，两条路各走各的。
    policy = HFPolicy(raw, tokenizer)

    # ⚠️ 双 adapter 隔离由 peft 的 set_adapter 负责；optimizer 必须覆盖**两个**
    #    adapter 的全部参数（按名字里的 "lora_" 选），否则切到 solver 相位时新参数不在优化器里。
    #    注意这里用 raw.named_parameters()：名字干净（没有 DDP 的 module. 前缀），
    #    而且是**同一批 Parameter 对象**，DDP 的 hook 会把同步后的 grad 写回它们。
    opt_params = [p for n, p in raw.named_parameters() if "lora_" in n]
    if not opt_params:
        raise SystemExit("没有找到任何 LoRA 参数，检查 peft 适配是否正常")
    optimizer = torch.optim.AdamW(opt_params, lr=tcfg.get("lr", 1e-4),
                                  weight_decay=tcfg.get("weight_decay", 0.0))
    clip_eps = tcfg.get("clip_eps", 0.3)
    max_grad_norm = tcfg.get("max_grad_norm", 1.0)
    pad_id = tokenizer.pad_token_id
    # GRPO surrogate 的归约口径：
    #   "seq"   = 论文语义（先每条序列内部按 token 平均，再组间等权）—— 默认
    #   "token" = 旧行为（把组里所有 token 混在一起平均 = 按长度加权，有 length bias）
    # 详见 src/grpo_norm.py 的模块 docstring，以及 scripts/ez_repro_lossnorm.py 的实测。
    loss_norm = tcfg.get("loss_norm", "seq")
    if loss_norm not in ("seq", "token"):
        raise ValueError(f"train.loss_norm 必须是 'seq' 或 'token'，收到 {loss_norm!r}")
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
    rng = random.Random(seed + rank)
    task_pool: List[ProposedTask] = []
    uid_counter = 0

    def persist_pool() -> None:
        """任务池落盘。★ 只 rank0 写：8 个 rank 同时 open(...,"w") 同一文件会互相截断。
        各 rank 的池子在 all_gather 之后本来就是同一份，所以 rank0 写的即全集。"""
        if not is_main(rank):
            return
        with open(task_pool_path, "w") as fh:
            for t in task_pool:
                fh.write(json.dumps(t.to_json(), ensure_ascii=False) + "\n")

    t_start = time.time()
    for outer in range(N):
        # ⚠️ 冒烟只跑 1 个 outer 的判据必须放在**循环开头**。
        #    之前写在循环结尾（ez_save 之后），但 solver 空池那条路径会 `continue`
        #    直接跳到下一轮，把 break 绕过去 —— 实测日志里出现了 outer 0/1/2/3 各跑一遍。
        if args.dry_run and outer > 0:
            break
        # ================= proposer 相位 =================
        model.eval()
        for step in range(M_P if not args.dry_run else 1):
            st0 = time.time()
            raw.set_adapter("proposer")
            # ---- ★ 分片采样（红线 B）：全局 TPS 个 game 由 rank0 抽一次后广播 ----
            #   各 rank 抽各自的会"不重不漏"变成"重且漏"；而 rank0 抽完再切
            #   `games[rank::world]` 才是严格不重不漏。组语义不变 ——
            #   每个 game 的 G 条 rollout 仍在**同一个 rank** 上跑完。
            games: List[str] = []
            if is_main(rank):
                games = [rng.choice(game_files) for _ in range(TPS)]
            if world > 1:
                _go = [games]
                dist.broadcast_object_list(_go, src=0)
                games = _go[0]
            if len(games) % world:
                raise SystemExit(f"[dist] 全局 games={len(games)} 不能被 world={world} 整除")
            local_games = games[rank::world]      # 每 rank 恰好 TPS/world 个
            # 1) 探索 + 提案
            # ★ 组按 game **逐个**收集（而不是事后按 G 切片）：切片一旦错位，
            #   组里会混进不同 game 的 rollout（红线 A：组是算法不可分单元）。
            cand: List[ProposerRollout] = []
            groups: List[List[ProposerRollout]] = []
            for gf in local_games:
                _one = run_proposer_exploration(env_p, policy, gf, p_cfg, log_fn=log)
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
            # 2) 冻结 solver 估 solve rate（G_P 条 rollout，不训练；用 batch=G_P 的 env_sr）
            new_tasks: List[ProposedTask] = []
            for c in accepted:
                t = c.task
                # ★ uid 必须带 rank：每 rank 各自 proposals，不带 rank 会跨 rank 撞 uid，
                #   撞了之后 solver 相位的 `task_id == uid` 分组会把不同任务混成一组。
                t.meta["uid"] = f"o{outer}s{step}-r{rank}-{uid_counter}"
                uid_counter += 1
                raw.set_adapter("solver")
                rollouts = run_ez_solver_rollouts(_ensure_env_sr(), policy, [t] * G_P, s_cfg,
                                                  tokenize=False, log_fn=log)
                raw.set_adapter("proposer")
                p_hat = sum(r.verify.final_match for r in rollouts) / G_P
                r_total, parts = proposer_reward(t, p_hat, w_sub=w_sub, w_div=w_div)
                c.reward = r_total
                c.__dict__["reward_parts"] = parts
                t.meta["solve_rate"] = p_hat
                new_tasks.append(t)
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
            # 3) proposer GRPO 更新（组 = 同一 game 的 G 条探索轨迹；上面已按 game 分好）
            #   ★ `len(groups)` 是**本 rank 的组数**（=TPS/world）。DDP 会对梯度取平均，
            #     全局等价于 1/TPS —— 正确，**不要**再手动乘 world（乘了就变成 1/(TPS/world)）。
            model.train()
            optimizer.zero_grad(set_to_none=True)
            _diag = []
            for groll in groups:
                for c in groll:
                    tokenize_proposer_rollout(c, policy, p_tok_cfg)
                advs = group_advantages([c.reward for c in groll],
                                        [0] * len(groll), use_std=True,
                                        ddof=adv_ddof)
                trajs_like = []
                from rollout import Trajectory as _T
                for c, a in zip(groll, advs):
                    trajs_like.append(_T(task_id="ez_proposer", task_type="proposer",
                                         game_file="", messages=c.messages,
                                         input_ids=c.__dict__["input_ids"],
                                         loss_mask=c.__dict__["loss_mask"],
                                         stats=None, advantage=a))
                bt = make_batch(trajs_like, pad_id, device)
                _diag.append((round(max(advs) - min(advs), 3), int(bt.n_active),
                              [len(c.__dict__["input_ids"]) for c in groll],
                              [int(sum(c.__dict__["loss_mask"])) for c in groll]))
                # ★ old logps 必须脱离计算图：不包 no_grad 的话 new-old 对参数梯度
                #   精确抵消，loss 非零但参数一步不动（review P0-1，与主线 train_grpo 一致）
                with torch.no_grad():
                    old = per_token_logps(model, bt.input_ids, bt.attention_mask)
                loss, _st = grpo_loss(model, bt, old, clip_eps, loss_norm=loss_norm)
                (loss / len(groups)).backward()
            _nof = {id(p): n for n, p in raw.named_parameters()}
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
            torch.nn.utils.clip_grad_norm_(opt_params, max_grad_norm)
            update_probe(raw, opt_params, optimizer, f"outer {outer} P{step}")
            # ★ 组奖励必须进日志：只有这样，"update_rel=0" 才能被判成
            #   「奖励同分（=没信号，正常）」还是「奖励有区分度却不动（=梯度链断，必修）」。
            #   实测教训（job 12989143 outer 2）：accept=1/2 却 update_rel=0，
            #   当时日志里连组奖励都没有，完全无法判断 —— 这就是补它的原因。
            _rw = [round(float(c.reward), 3) for c in cand]
            log(f"[outer {outer}][P{step}] rank={rank} 本地 games={len(local_games)}/{TPS} "
                f"| pool={len(task_pool)} (+{len(new_tasks)}) "
                f"| accept={len(accepted)}/{len(cand)} | rew={_rw} "
                f"| rew_spread={max(_rw) - min(_rw) if _rw else 0:.3f} | {time.time()-st0:.0f}s")
            # 断层定位：grad_norm>0 而 update_rel=0 -> 优化器/精度问题；
            # grad_norm=0 而 adv_spread>0 -> loss 没把梯度传出来（mask/图断）
            log(f"[outer {outer}][P{step}] 诊断 adv_spread/n_active/len_ids/mask_sum="
                f"{_diag} | 真实 backward grad_norm={_gn:.4e} "
                f"({len(_act)} 个张量有 grad；A={_gnA:.4e} B={_gnB:.4e})"
                f"{'  ← A=0 是正常的（B 初始全 0）' if _gnA == 0.0 and _gnB > 0 else ''}")
            if args.dry_run:
                break

        # ================= solver 相位 =================
        model.eval()
        raw.set_adapter("solver")
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
            continue
        for step in range(M_S if not args.dry_run else 1):
            st0 = time.time()
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
            k_local = max(1, min(TPS // max(world, 1), len(outer_pool)))
            # ★ `k_local` 直接决定 backward 次数，次数不同 = NCCL 永久卡死（红线 B）。
            #   上面那句只依赖"池子已跨 rank 一致"这个**注释里的假设** —— 假设破了
            #   就是 30 分钟 timeout 才炸。显式 all_reduce MIN 同步一次，顺带在
            #   假设真破了的时候把不一致打到日志里，而不是让人去猜死锁原因。
            if world > 1:
                _kt = torch.tensor([k_local], device=device)
                dist.all_reduce(_kt, op=dist.ReduceOp.MIN)
                _kmin = int(_kt.item())
                del _kt
                if _kmin != k_local:
                    log(f"[warn] solver k 跨 rank 不一致（rank{rank} 本地 {k_local}，"
                        f"全局最小 {_kmin}）→ 统一取 {_kmin}。"
                        f"这说明 outer_pool 并非真的跨 rank 一致，去查 all_gather")
                k_local = _kmin
            batch = [outer_pool[(rank + i * world) % len(outer_pool)]
                     for i in range(k_local)]
            all_roll: List[EzSolverTrajectory] = []
            for t in batch:
                all_roll.extend(run_ez_solver_rollouts(env_s, policy, [t] * G, s_cfg,
                                                       tokenize=True, log_fn=log))
            model.train()
            optimizer.zero_grad(set_to_none=True)
            # ★ backward 次数必须**恒等于 len(batch)**（红线 B）。原来那句
            #   `if not grp: continue` 会让某个 rank 少一次 backward，其它 rank
            #   就永久卡在 NCCL barrier 上。空组用"零 loss 但带计算图"的 backward 占位。
            _n_bwd = 0
            for t in batch:
                grp = [r for r in all_roll if r.traj.task_id == t.meta["uid"]]
                if not grp:
                    _zbt = _borrow_batch(all_roll, pad_id, device)
                    if _zbt is None:
                        raise SystemExit(
                            "[solver][红线B] 整批 rollout 全空，无法保持各 rank 的 "
                            "backward 次数一致。检查 run_ez_solver_rollouts 是否静默丢样本")
                    _z = per_token_logps(model, _zbt.input_ids, _zbt.attention_mask).sum() * 0.0
                    (_z / len(batch)).backward()
                    _n_bwd += 1
                    continue
                advs = group_advantages([r.traj.reward for r in grp], [0] * len(grp),
                                        use_std=True, ddof=adv_ddof)
                for r, a in zip(grp, advs):
                    r.traj.advantage = a
                bt = make_batch([r.traj for r in grp], pad_id, device)
                with torch.no_grad():
                    old = per_token_logps(model, bt.input_ids, bt.attention_mask)
                loss, stt = grpo_loss(model, bt, old, clip_eps, loss_norm=loss_norm)
                #   ★ 分母是**本 rank 的 batch 条数**（=TPS/world）：DDP 取平均后
                #     全局正好是 1/TPS。不要手动乘 world。
                (loss / len(batch)).backward()
                _n_bwd += 1
            if _n_bwd != len(batch):
                raise SystemExit(f"[solver][红线B] backward 次数 {_n_bwd} != len(batch) {len(batch)}")
            # ★ S 相位也打一行同款诊断：判读脚本要靠「真实 backward grad_norm」把
            #   「有梯度却没位移」（真断链，FATAL）和「本来就没梯度」（奖励全同分，
            #   正确不动）分开。只有 P 相位有诊断行时，S 相位的 0/504 无法判定。
            _nofS = {id(p): n for n, p in raw.named_parameters()}
            _actS = [p for p in opt_params if p.grad is not None and p.requires_grad]
            _gnS = float(sum(p.grad.detach().float().norm() for p in _actS)) if _actS else 0.0
            _gnSA = float(sum(p.grad.detach().float().norm() for p in _actS
                              if ".lora_A." in _nofS.get(id(p), "")))
            _gnSB = float(sum(p.grad.detach().float().norm() for p in _actS
                              if ".lora_B." in _nofS.get(id(p), "")))
            torch.nn.utils.clip_grad_norm_(opt_params, max_grad_norm)
            update_probe(raw, opt_params, optimizer, f"outer {outer} S{step}")
            rw = [r.traj.reward for r in all_roll]
            log(f"[outer {outer}][S{step}] rank={rank} batch={len(batch)}/{TPS} "
                f"| reward mean={np.mean(rw):.3f} "
                f"| final_match={sum(r.verify.final_match for r in all_roll)}/{len(all_roll)} "
                f"| pool={len(outer_pool)} | {time.time()-st0:.0f}s")
            log(f"[outer {outer}][S{step}] 诊断 真实 backward grad_norm={_gnS:.4e} "
                f"({len(_actS)} 个张量有 grad；A={_gnSA:.4e} B={_gnSB:.4e})"
                f"{'  ← 奖励全同分 ⇒ 优势全 0 ⇒ 梯度精确为 0，参数不动是正确的' if _gnS == 0.0 else ''}")
            if args.dry_run:
                break

        # ★ 只在 rank0 存档：peft 保存的是 adapter（几百 MB），8 个 rank 各存一份
        #   既浪费又会在同一路径上互相踩。（deepspeed 靠不同的 zero stage 存分片，
        #   我们不走那条路。）
        if is_main(rank):
            ez_save(raw, tokenizer, run_dir, f"outer{outer+1}", cfg, {"outer": outer + 1})

    close_all()
    persist_pool()
    # ★ dry-run 只跑 outer 0，但 N 仍是配置里的 4 —— 直接打 N 会让人以为跑了 4 轮
    _n_done = 1 if (args.dry_run and N > 0) else N
    log(f"[done] rank={rank} {_n_done} outer loops in {(time.time()-t_start)/60:.1f} min -> {run_dir}")
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
