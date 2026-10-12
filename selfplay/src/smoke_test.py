#!/usr/bin/env python
"""分层冒烟测试：从"环境能不能跑"一路验到"loss 能不能反传"，逐层定位问题。

    python src/smoke_test.py --stage env        # 只验 ALFWorld 环境（最常出问题的一层）
    python src/smoke_test.py --stage rollout    # 环境 + 随机策略跑完整一局
    python src/smoke_test.py --stage rollout --batch 8 --group-size 4   # 压测 slot 打乱 + 组连续性
    python src/smoke_test.py --stage tokens     # 上面 + chat 模板 / loss mask 拼接
    python src/smoke_test.py --stage all        # 上面 + 4B 模型 rollout + 前向/反向（要 GPU）
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alfworld_compat import install as install_alfworld_compat  # noqa: E402

STAGES = ["env", "rollout", "tokens", "all"]


# --------------------------------------------------------------------------- 1


def stage_env(args):
    from alfworld_env import ALFWorldBatchEnv, load_alfworld_config

    print("\n=== [1/4] 环境：构造 + reset + step ===")
    cfg = load_alfworld_config(args.config)
    t0 = time.time()
    env = ALFWorldBatchEnv(cfg, split=args.split, batch_size=args.batch, seed=0)
    print(f"  init_env({args.batch}) 耗时 {time.time() - t0:.1f}s，任务池 {len(env.game_files)}")
    t0 = time.time()
    obs, views, adm, used, _expert = env.reset()
    print(f"  reset() 耗时 {time.time() - t0:.1f}s")
    for i in range(len(obs)):
        print(f"  --- slot {i} --- game={used[i].split('/')[-3]}/{used[i].split('/')[-2]}")
        print(f"      task: {views[i].task_desc[:110]}")
        print(f"      obs : {views[i].env_desc[:110]}")
        print(f"      adm : {len(adm[i])} 条，例如 {adm[i][:3]}")
    t0 = time.time()
    for _ in range(min(5, env.max_episode_steps)):
        env.step(["look"] * len(obs))
    dt = time.time() - t0
    print(f"  5 次 step() 总耗时 {dt:.2f}s（avg {dt / 5:.3f}s/step）")
    return env


# --------------------------------------------------------------------------- 2


def stage_rollout(env, args, policy=None, group_size: int = None):
    from reward import make_reward_cfg
    from rollout import RolloutConfig, run_rollouts

    print("\n=== [2/4] rollout：走完整一局 ===")
    if policy is None:
        from agents import RandomPolicy

        policy = RandomPolicy(seed=0)
    if group_size is None:
        group_size = int(args.group_size)
    if group_size < 1 or env.batch_size % group_size != 0:
        raise SystemExit(f"--batch={env.batch_size} 必须能被 group_size={group_size} 整除")
    # ⚠️ 这里刻意用「同一个 task 重复 group_size 次」来触发 textworld 的 slot 打乱路径，
    #    验证 run_rollouts 的 task-major 重排（group 内必须是同一个 game_file）。
    n_tasks = max(1, env.batch_size // group_size)
    tasks = [
        {"game_file": env.game_files[i % len(env.game_files)], "task_id": f"smoke{i}",
         "task_type": "smoke"}
        for i in range(n_tasks)
    ]
    cfg = RolloutConfig(group_size=group_size, max_steps=args.max_steps, temperature=1.0)
    # `tokenize=True` 需要 policy 能渲染 chat 模板（build_token_sequence 要用到）。非 HF 的
    # 假策略（Random/Echo/Scripted/Oracle）没有 render/tokenize_text，传 True 会
    # AttributeError: 'RandomPolicy' object has no attribute 'render'。
    can_tokenize = hasattr(policy, "render") and hasattr(policy, "tokenize_text")
    trajs = run_rollouts(env, policy, tasks, cfg, make_reward_cfg(None),
                         log_fn=lambda m: print("  " + m), tokenize=can_tokenize)
    for t in trajs[:6]:
        print(f"  {t.task_id}: won={t.stats.won} steps={t.stats.n_steps} reward={t.reward:.4f} "
              f"invalid={t.stats.n_invalid} fmt_bad={t.stats.n_format_violation} "
              f"tokens={t.n_tokens} game={os.path.basename(os.path.dirname(t.game_file))}")
    # 奖励分布诊断：如果所有轨迹同分，GRPO 会拿到全 0 的 advantage（等于白跑）
    rs = [t.reward for t in trajs]
    print(f"  --- 奖励分布 --- min={min(rs):.4f} max={max(rs):.4f} "
          f"mean={sum(rs) / len(rs):.4f} n_distinct={len(set(round(r, 6) for r in rs))}")
    if len(set(round(r, 6) for r in rs)) == 1:
        print("  ⚠️ 所有轨迹奖励完全相同 -> 组内 std=0 -> advantage 全 0 -> 这一步不会有任何梯度！")

    # ---- 组连续性检查（与 train_grpo.py 的断言同源）----
    print(f"  --- group 连续性检查（group_size={group_size}, {len(trajs)} 条轨迹）---")
    for i in range(0, len(trajs), group_size):
        grp = trajs[i:i + group_size]
        ids = {t.task_id for t in grp}
        gfs = {os.path.realpath(t.game_file) for t in grp}
        assert len(gfs) == 1, f"group {i // group_size} 的 game_file 不一致：{sorted(gfs)}"
        print(f"    group {i // group_size}: task_id={sorted(ids)} n={len(grp)} OK")
    order_tasks = [t.task_id for t in trajs]
    assert order_tasks == [f"smoke{i // group_size}" for i in range(len(trajs))], \
        f"返回顺序不是 task-major：{order_tasks}"
    print("  OK：返回顺序是 task-major / 组连续，且每组 game_file 一致")
    return trajs


# --------------------------------------------------------------------------- 3


def stage_tokens(trajs, args):
    print("\n=== [3/4] tokens：chat 模板 + loss mask 拼接 ===")
    from transformers import AutoTokenizer

    from rollout import HFPolicy, build_token_sequence

    model_path = os.path.expandvars(os.path.expanduser(args.model))
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    class _Tok:  # 只借 render/tokenize 两个方法，不需要真模型
        pass

    policy = HFPolicy.__new__(HFPolicy)
    policy.tokenizer = tok
    policy.enable_thinking = False

    for t in trajs[:4]:
        n_asst = sum(1 for m in t.messages if m["role"] == "assistant")
        gen_ids = [[] for _ in range(n_asst)]
        ids, mask, bounds = build_token_sequence(t.messages, gen_ids, policy)
        print(f"  {t.task_id}: turns={n_asst} tokens={len(ids)} mask_sum={sum(mask)} "
              f"boundaries={len(bounds)}")
        assert len(ids) == len(mask), "input_ids 与 loss_mask 长度必须一致"
    print("  OK：模板前缀稳定，token/mask 拼接无异常")


# --------------------------------------------------------------------------- 4


def stage_all(env, args):
    print("\n=== [4/4] 模型：真实 rollout + 前向/反向 ===")
    import torch
    import yaml

    from reward import group_advantages, make_reward_cfg
    from rollout import HFPolicy, RolloutConfig, run_rollouts
    from train_grpo import build_model_and_tokenizer, grpo_loss, make_batch, per_token_logps

    with open(args.train_config) as fh:
        tcfg = yaml.safe_load(fh)
    tcfg = dict(tcfg)
    tcfg["model"] = dict(tcfg["model"])
    tcfg["model"]["path"] = args.model
    tcfg["model"]["attn_implementation"] = args.attn
    tcfg["model"]["gradient_checkpointing"] = True

    model, tok = build_model_and_tokenizer(tcfg, 0, 1)
    policy = HFPolicy(model, tok)

    group_size = args.group_size if args.group_size > 1 else 2
    trajs = stage_rollout(env, args, policy=policy, group_size=group_size)
    gids = [i // group_size for i in range(len(trajs))]
    for t, a in zip(trajs, group_advantages([t.reward for t in trajs], gids)):
        t.advantage = a
    print("  advantages:", [round(t.advantage, 3) for t in trajs])

    device = next(model.parameters()).device
    batch = make_batch(trajs, pad_id=tok.pad_token_id, device=device)
    print(f"  batch: input_ids={tuple(batch.input_ids.shape)} active_tokens={batch.n_active}")
    with torch.no_grad():
        old = per_token_logps(model, batch.input_ids, batch.attention_mask)
    loss, stats = grpo_loss(model, batch, old, clip_eps=0.2)
    print(f"  loss={loss.item():.5f} stats={ {k: round(v, 4) for k, v in stats.items()} }")
    loss.backward()
    gn = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0))
    print(f"  grad_norm={gn:.4f}")
    assert torch.isfinite(loss), "loss 不是有限值"
    print("  OK：forward/backward 通路正常")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="env", choices=STAGES)
    ap.add_argument("--config", default=str(SRC_DIR.parent / "configs" / "base_config.yaml"))
    ap.add_argument("--train-config", default=str(SRC_DIR.parent / "configs" / "train_qwen3_4b.yaml"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=1,
                    help="每个 task 采几条轨迹；>1 会真正压测 textworld 的 slot 打乱路径")
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--model", default=os.path.expanduser("~/thinktool/models/qwen3_4b_inst_2507"))
    ap.add_argument("--attn", default="sdpa")
    args = ap.parse_args()

    install_alfworld_compat(verbose=True)
    env = stage_env(args)
    try:
        if args.stage == "rollout":
            stage_rollout(env, args)
        elif args.stage == "tokens":
            stage_tokens(stage_rollout(env, args), args)
        elif args.stage == "all":
            stage_all(env, args)
        print("\n*** 冒烟测试通过 ***")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
