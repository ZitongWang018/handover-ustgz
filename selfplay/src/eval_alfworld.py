#!/usr/bin/env python
"""ALFWorld 评测：把训练出的 Qwen3-4B（或 DeepSeek 基线 / 随机基线）放到留出集上跑。

用法示例
--------
    # 0. 环境自检（不调模型）
    python src/eval_alfworld.py --agent echo --split eval_out_of_distribution --n 8

    # 0b. 官方 handcoded expert（环境可解性上界参考点，不调模型）
    python src/eval_alfworld.py --agent expert --split eval_out_of_distribution --n 8 --batch 8

    # 1. 随机基线（成功率下限）
    python src/eval_alfworld.py --agent random --n 20 --batch 10

    # 2. DeepSeek-V4-Pro 基线（需要外网，三期不行）
    python src/eval_alfworld.py --agent deepseek --n 20 --batch 5 --concurrency 5

    # 3. 本地 Qwen3-4B（base 或我们训出来的 ckpt）
    python src/eval_alfworld.py --agent hf --model ~/thinktool/models/qwen3_4b_inst_2507 --n 20 --batch 8

输出：`runs/eval/<name>/<split>_<agent>.json`，含总体成功率、逐任务类型成功率、平均步数、非法动作率。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alfworld_compat import install as install_alfworld_compat  # noqa: E402

from alfworld_env import ALFWorldBatchEnv, collect_game_files, load_alfworld_config, task_instance_id, task_type_of  # noqa: E402
from reward import RewardConfig, make_reward_cfg  # noqa: E402
from rollout import HFPolicy, RolloutConfig, run_expert, run_rollouts  # noqa: E402


def build_policy(args, system_prompt: Optional[str] = None):
    if args.agent == "expert":
        # expert 计划来自 env infos，不经过 policy 抽象 —— 无需构造策略对象。
        return None
    if args.agent == "echo":
        from agents import EchoPolicy

        return EchoPolicy()
    if args.agent == "random":
        from agents import RandomPolicy

        return RandomPolicy(seed=args.seed)
    if args.agent == "scripted":
        from agents import ScriptedOraclePolicy

        return ScriptedOraclePolicy()
    if args.agent == "deepseek":
        from agents import ChatAPIPolicy
        from llm_client import ChatClient

        client = ChatClient(model=args.api_model)
        print(f"[eval] API models: {client.list_models()}")
        return ChatAPIPolicy(
            client=client,
            model=args.api_model,
            temperature=args.temperature,
            max_tokens=args.api_max_tokens,
            effort=None if args.effort == "none" else args.effort,
            concurrency=args.concurrency,
            system_prompt=system_prompt,
        )
    if args.agent == "hf":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = os.path.expandvars(os.path.expanduser(args.model))
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation=args.attn
        ).to("cuda" if torch.cuda.is_available() else "cpu").eval()
        return HFPolicy(model, tok, enable_thinking=args.enable_thinking)
    raise SystemExit(f"未知 agent：{args.agent}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="hf", choices=["hf", "deepseek", "random", "echo", "scripted", "expert"])
    ap.add_argument("--model", default=os.path.expanduser("~/thinktool/models/qwen3_4b_inst_2507"))
    ap.add_argument("--config", default=str(SRC_DIR.parent / "configs" / "base_config.yaml"))
    ap.add_argument("--train-config", default=str(SRC_DIR.parent / "configs" / "train_qwen3_4b.yaml"))
    ap.add_argument("--split", default="eval_out_of_distribution",
                    choices=["eval_out_of_distribution", "eval_in_distribution", "train"])
    ap.add_argument("--n", type=int, default=20, help="评测任务数（0=整个 split）")
    ap.add_argument("--batch", type=int, default=5, help="env 批量大小（同时跑几局）")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--attn", default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    # DeepSeek / API 专用
    ap.add_argument("--api-model", default="deepseek-v4-pro")
    ap.add_argument("--api-max-tokens", type=int, default=2048)
    ap.add_argument("--effort", default="max", choices=["low", "high", "max", "none"])
    ap.add_argument("--concurrency", type=int, default=5)
    args = ap.parse_args()

    install_alfworld_compat(verbose=True)

    cfg = load_alfworld_config(args.config)
    tcfg: Dict[str, Any] = {}
    if os.path.isfile(args.train_config):
        import yaml

        with open(args.train_config) as fh:
            tcfg = yaml.safe_load(fh) or {}
    reward_cfg = make_reward_cfg((tcfg or {}).get("reward"))

    games = sorted(collect_game_files(cfg, args.split))
    rng = random.Random(args.seed)
    rng.shuffle(games)
    if args.n > 0:
        games = games[: args.n]
    tasks = [{"game_file": g, "task_id": task_instance_id(g), "task_type": task_type_of(g)} for g in games]
    print(f"[eval] split={args.split} tasks={len(tasks)} batch={args.batch} agent={args.agent}")

    policy = build_policy(args)
    rollout_cfg = RolloutConfig(
        group_size=1,
        max_steps=args.max_steps,
        temperature=args.temperature,
        max_context_tokens=8192,
    )

    results: List[Dict[str, Any]] = []
    t0 = time.time()
    for start in range(0, len(tasks), args.batch):
        chunk = tasks[start:start + args.batch]
        env = ALFWorldBatchEnv(cfg, split=args.split, batch_size=len(chunk),
                               seed=args.seed + start, with_expert=(args.agent == "expert"))
        try:
            if args.agent == "expert":
                trajs = run_expert(env, chunk, args.max_steps, reward_cfg,
                                   log_fn=lambda m: print("  " + m, flush=True))
            else:
                trajs = run_rollouts(env, policy, chunk, rollout_cfg, reward_cfg,
                                     log_fn=lambda m: print("  " + m, flush=True), tokenize=False)
        finally:
            env.close()
        # 对齐守卫：轨迹顺序必须与 chunk 一一对应（防 slot/task 置换回归）。
        if len(trajs) != len(chunk):
            raise RuntimeError(f"trajs={len(trajs)} 与 chunk={len(chunk)} 数量不符")
        for _tr, _tk in zip(trajs, chunk):
            if os.path.realpath(_tr.game_file) != os.path.realpath(_tk["game_file"]):
                raise RuntimeError(
                    f"轨迹与 task 对不上：traj.game_file={_tr.game_file} vs task={_tk['game_file']}"
                )
        for t, task in zip(trajs, chunk):
            results.append({
                "task_id": task["task_id"],
                "task_type": task["task_type"],
                "game_file": task["game_file"],
                "won": bool(t.stats.won),
                "steps": t.stats.n_steps,
                "invalid": t.stats.n_invalid,
                "format_bad": t.stats.n_format_violation,
                "repeats": t.stats.n_repeat,
                "truncated": t.stats.truncated,
                "reward": t.reward,
                "last_observation": t.messages[-1]["content"][:400] if t.messages else "",
                "trajectory_actions": [
                    m["content"].split("Action:")[-1].strip()[:80]
                    for m in t.messages if m["role"] == "assistant"
                ],
            })
        done = len(results)
        print(f"[eval] {done}/{len(tasks)} won={sum(r['won'] for r in results)} "
              f"({sum(r['won'] for r in results) / max(1, done):.1%})  {time.time() - t0:.0f}s", flush=True)

    by_type: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"n": 0, "won": 0})
    for r in results:
        by_type[r["task_type"]]["n"] += 1
        by_type[r["task_type"]]["won"] += int(r["won"])
    if args.agent == "hf":
        used_model = args.model
    elif args.agent == "deepseek":
        used_model = args.api_model
    else:  # echo / random / scripted / expert 都不调模型
        used_model = "-"
    summary = {
        "agent": args.agent,
        "model": used_model,
        "split": args.split,
        "n": len(results),
        "success_rate": sum(r["won"] for r in results) / max(1, len(results)),
        "mean_steps": sum(r["steps"] for r in results) / max(1, len(results)),
        "invalid_action_rate": sum(r["invalid"] for r in results) / max(1, sum(r["steps"] for r in results)),
        "format_violation_rate": sum(r["format_bad"] for r in results) / max(1, sum(r["steps"] for r in results)),
        "by_task_type": {k: {**v, "success_rate": v["won"] / max(1, v["n"])} for k, v in by_type.items()},
        "elapsed_sec": time.time() - t0,
    }
    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    out_dir = Path(args.out_dir) if args.out_dir else SRC_DIR.parent / "runs" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out_path = out_dir / f"{args.split}_{args.agent}{tag}.json"
    with open(out_path, "w") as fh:
        json.dump({"summary": summary, "results": results}, fh, indent=2, ensure_ascii=False)
    print(f"\n[eval] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
