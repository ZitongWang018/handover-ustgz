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
import yaml  # noqa: E402

from alfworld_env import ALFWorldBatchEnv, collect_game_files, load_alfworld_config  # noqa: E402
from ez_proposer import ProposerConfig, ProposerRollout, run_proposer_exploration  # noqa: E402
from ez_solver import EzSolverTrajectory, run_ez_solver_rollouts  # noqa: E402
from ez_state import StateSnapshot  # noqa: E402
from ez_verify import ProposedTask, proposer_reward, validity_filter  # noqa: E402
from rollout import HFPolicy, RolloutConfig, build_token_sequence  # noqa: E402
from train_grpo import (build_model_and_tokenizer, grpo_loss, group_advantages,  # noqa: E402
                        load_cfg, log0, make_batch, per_token_logps, set_seed,
                        dist_info)


_LOG_FH = None


def log(msg: str) -> None:
    """同时落 stdout 和 train_log.jsonl（review P1-4：句柄不能是死的）。"""
    log0(0, msg)
    if _LOG_FH is not None:
        _LOG_FH.write(json.dumps({"t": time.time(), "msg": msg}, ensure_ascii=False) + "\n")
        _LOG_FH.flush()


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

    LoRA 下取中等大小的 adapter 参数张量测 step 前后相对变化量。
    """
    probe = None
    for p in opt_params:
        if p.numel() >= 4096:
            probe = p
            break
    if probe is None:
        probe = opt_params[len(opt_params) // 2]
    before = probe.detach().clone()
    optimizer.step()
    with torch.no_grad():
        rel = float((probe.detach() - before).norm() / max(probe.detach().norm().item(), 1e-12))
    log(f"[{tag}] update_rel={rel:.3e}（必须为 >>1e-10，否则梯度是死的）")
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


# --------------------------------------------------------------------------- 主循环


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true", help="只跑 1 步 proposer + 1 步 solver 冒烟")
    ap.add_argument("--max-outer", type=int, default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    seed = cfg["run"].get("seed", 42)
    rank, world, local = dist_info()   # 本版 scaffold 按 world==1 设计；DDP 分片 TODO
    set_seed(seed, rank)

    run_dir = Path(cfg["run"]["output_dir"]) / f"{cfg['run']['name']}-{time.strftime('%m%d-%H%M')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    task_pool_path = run_dir / "task_pool.jsonl"
    global _LOG_FH
    _LOG_FH = open(run_dir / "train_log.jsonl", "w")

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
    if args.dry_run:
        G = int(os.environ.get("EZ_SMOKE_G", "2"))
        G_P = int(os.environ.get("EZ_SMOKE_G_P", "2"))
        TPS = int(os.environ.get("EZ_SMOKE_TPS", "1"))
        M_P = M_S = 1
        _smoke_turns = int(os.environ.get("EZ_SMOKE_TURNS", "3"))
        p_max_turns = s_max_turns = _smoke_turns
        log(f"[dry-run] 缩规模：G={G} G_P={G_P} TPS={TPS} turns={_smoke_turns}")
    else:
        p_max_turns = cfg["proposer"].get("max_turns", 30)
        s_max_turns = cfg["solver"].get("max_turns", 30)

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
    env_sr = ALFWorldBatchEnv(alf_cfg, split=cfg["env"].get("split", "train"),
                              batch_size=G_P, max_episode_steps=s_cfg.max_steps,
                              with_facts=True)
    # 异常退出也释放子进程/句柄（review P1-4）
    atexit.register(env_p.close)
    atexit.register(env_s.close)
    atexit.register(env_sr.close)

    # ---- 模型（共享基座 + 双 LoRA adapter：proposer / solver）----
    model, tokenizer = build_model_and_tokenizer(cfg, local, world)
    model = build_dual_lora(model, cfg)
    policy = HFPolicy(model.module if hasattr(model, "module") else model, tokenizer)

    # ---- 启动自检：dtype / adapter / GC / facts（跑几小时前先炸掉所有致命项）----
    _p0 = next(model.parameters())
    _adapters = [n for n, _ in model.named_parameters() if "lora_" in n][:1]
    log(f"[selfcheck] base dtype={_p0.dtype} | lora参数样例={_adapters} | "
        f"gc={getattr(model, 'is_gradient_checkpointing', '?')} | "
        f"stop_mode=tool_call | 基座冻结={not any(p.requires_grad for n, p in model.named_parameters() if 'lora_' not in n)}")
    # facts 预检：EZ 没有 facts 就是废的，早炸比跑一半炸好
    from ez_state import assert_facts_mode as _afm
    from ez_proposer import _slot_snapshot as _snp
    _pre_games = [game_files[0]] * G_P
    _obs, _views, _adm, _used, _ = env_sr.reset(_pre_games)
    _pre = _snp(getattr(env_sr, "last_reset_facts", [None] * G_P)[0], _obs[0])
    _afm(_pre, "startup preflight")
    log(f"[selfcheck] facts 通道 OK（{len(_pre.facts)} 条事实）")

    device = next(model.parameters()).device
    # ⚠️ 双 adapter 下 peft 靠切换 requires_grad 隔离梯度：optimizer 必须覆盖**两个**
    #    adapter 的全部参数（按名字里的 "lora_" 选），否则切到 solver 相位时新参数不在优化器里。
    opt_params = [p for n, p in model.named_parameters() if "lora_" in n]
    if not opt_params:
        raise SystemExit("没有找到任何 LoRA 参数，检查 peft 适配是否正常")
    optimizer = torch.optim.AdamW(opt_params, lr=tcfg.get("lr", 1e-4),
                                  weight_decay=tcfg.get("weight_decay", 0.0))
    clip_eps = tcfg.get("clip_eps", 0.3)
    max_grad_norm = tcfg.get("max_grad_norm", 1.0)
    pad_id = tokenizer.pad_token_id

    rng = random.Random(seed)
    task_pool: List[ProposedTask] = []
    uid_counter = 0

    def persist_pool() -> None:
        with open(task_pool_path, "w") as fh:
            for t in task_pool:
                fh.write(json.dumps(t.to_json(), ensure_ascii=False) + "\n")

    t_start = time.time()
    for outer in range(N):
        # ================= proposer 相位 =================
        model.eval()
        for step in range(M_P if not args.dry_run else 1):
            st0 = time.time()
            model.set_adapter("proposer")
            games = [rng.choice(game_files) for _ in range(TPS)]
            # 1) 探索 + 提案
            cand: List[ProposerRollout] = []
            for gf in games:
                cand.extend(run_proposer_exploration(env_p, policy, gf, p_cfg, log_fn=log))
            accepted = [c for c in cand if c.accepted and c.task is not None]
            # 2) 冻结 solver 估 solve rate（G_P 条 rollout，不训练；用 batch=G_P 的 env_sr）
            new_tasks: List[ProposedTask] = []
            for c in accepted:
                t = c.task
                t.meta["uid"] = f"o{outer}s{step}-{uid_counter}"
                uid_counter += 1
                model.set_adapter("solver")
                rollouts = run_ez_solver_rollouts(env_sr, policy, [t] * G_P, s_cfg,
                                                  tokenize=False, log_fn=log)
                model.set_adapter("proposer")
                p_hat = sum(r.verify.final_match for r in rollouts) / G_P
                r_total, parts = proposer_reward(t, p_hat, w_sub=w_sub, w_div=w_div)
                c.reward = r_total
                c.__dict__["reward_parts"] = parts
                t.meta["solve_rate"] = p_hat
                new_tasks.append(t)
            task_pool.extend(new_tasks)
            persist_pool()
            # 3) proposer GRPO 更新（组 = 同一 game 的 G 条探索轨迹）
            groups: List[List[ProposerRollout]] = []
            gi = 0
            for _gf in games:
                groups.append(cand[gi:gi + G])
                gi += G
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for groll in groups:
                for c in groll:
                    tokenize_proposer_rollout(c, policy, p_tok_cfg)
                advs = group_advantages([c.reward for c in groll],
                                        [0] * len(groll), use_std=True)
                trajs_like = []
                from rollout import Trajectory as _T
                for c, a in zip(groll, advs):
                    trajs_like.append(_T(task_id="ez_proposer", task_type="proposer",
                                         game_file="", messages=c.messages,
                                         input_ids=c.__dict__["input_ids"],
                                         loss_mask=c.__dict__["loss_mask"],
                                         stats=None, advantage=a))
                bt = make_batch(trajs_like, pad_id, device)
                # ★ old logps 必须脱离计算图：不包 no_grad 的话 new-old 对参数梯度
                #   精确抵消，loss 非零但参数一步不动（review P0-1，与主线 train_grpo 一致）
                with torch.no_grad():
                    old = per_token_logps(model, bt.input_ids, bt.attention_mask)
                loss, _st = grpo_loss(model, bt, old, clip_eps)
                (loss / len(groups)).backward()
            torch.nn.utils.clip_grad_norm_(opt_params, max_grad_norm)
            update_probe(model, opt_params, optimizer, f"outer {outer} P{step}")
            log(f"[outer {outer}][P{step}] pool={len(task_pool)} (+{len(new_tasks)}) "
                f"| accept={len(accepted)}/{len(cand)} | {time.time()-st0:.0f}s")
            if args.dry_run:
                break

        # ================= solver 相位 =================
        model.eval()
        model.set_adapter("solver")
        # 任务来源对齐论文 Appendix C：solver 只练**本 outer proposer 相位产出的任务**
        # （论文是每迭代用冻结 proposer 重新采样；跨 outer 抽旧题会让任务分布与
        #   当前 proposer 脱节 —— review #2）。池子仍全量持久化供分析。
        outer_pool = [t for t in task_pool if str(t.meta.get("uid", "")).startswith(f"o{outer}")]
        if not outer_pool:
            log("[solver] 本 outer 任务池为空，跳过 solver 相位")
            continue
        for step in range(M_S if not args.dry_run else 1):
            st0 = time.time()
            model.eval()
            # 无放回采样：有放回会让同 uid 的组被合并、loss 双计（review P1-2）
            batch = rng.sample(outer_pool, k=min(TPS, len(outer_pool)))
            all_roll: List[EzSolverTrajectory] = []
            for t in batch:
                all_roll.extend(run_ez_solver_rollouts(env_s, policy, [t] * G, s_cfg,
                                                       tokenize=True, log_fn=log))
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for t in batch:
                grp = [r for r in all_roll if r.traj.task_id == t.meta["uid"]]
                if not grp:
                    continue
                advs = group_advantages([r.traj.reward for r in grp], [0] * len(grp),
                                        use_std=True)
                for r, a in zip(grp, advs):
                    r.traj.advantage = a
                bt = make_batch([r.traj for r in grp], pad_id, device)
                with torch.no_grad():
                    old = per_token_logps(model, bt.input_ids, bt.attention_mask)
                loss, stt = grpo_loss(model, bt, old, clip_eps)
                (loss / len(batch)).backward()
            torch.nn.utils.clip_grad_norm_(opt_params, max_grad_norm)
            update_probe(model, opt_params, optimizer, f"outer {outer} S{step}")
            rw = [r.traj.reward for r in all_roll]
            log(f"[outer {outer}][S{step}] reward mean={np.mean(rw):.3f} "
                f"| final_match={sum(r.verify.final_match for r in all_roll)}/{len(all_roll)} "
                f"| {time.time()-st0:.0f}s")
            if args.dry_run:
                break

        ez_save(model, tokenizer, run_dir, f"outer{outer+1}", cfg, {"outer": outer + 1})

    env_p.close()
    env_s.close()
    env_sr.close()
    persist_pool()
    log(f"[done] {N} outer loops in {(time.time()-t_start)/60:.1f} min -> {run_dir}")
    _LOG_FH.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
