#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""bench_alfworld_thinktool.py -- ALFWorld 文本环境吞吐基准 (thinktool / py3.10)。

测三件事（每个 batch_size 各测一遍）：
  1. init_env(batch_size=N)          —— 注册 + make gym env
  2. reset() 冷启动 & 稳态          —— 用第 1 次 / 后续几次分别统计
  3. step() 稳态                    —— 连续 N_STEPS 次，报 avg / p50 / steps_per_sec

用法：
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate thinktool
    python bench_alfworld_thinktool.py --batches 1 2 4 8 --steps 30 --resets 4 --json /tmp/bench.json

必须在 SLURM 里跑（登录节点只做小测试）：
    srun -p debug -t 10 --cpus-per-task=8 python bench_alfworld_thinktool.py ...
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

REPO = os.path.expanduser(os.environ.get("SELFPLAY_REPO", "~/ztwang/selfplay"))
DEPS = os.path.join(REPO, "third_party", "py310deps")
SRC = os.path.join(REPO, "src")
CFG = os.path.join(REPO, "configs", "base_config.yaml")


def bootstrap():
    for p in (SRC, DEPS):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))
    from alfworld_compat import install
    install(verbose=False)


def q(vals, p):
    vs = sorted(vals)
    if not vs:
        return 0.0
    k = max(0, min(len(vs) - 1, int(round((len(vs) - 1) * p))))
    return vs[k]


def bench_one(AlfredTWEnv, config, n, n_steps, n_resets):
    out = {"batch_size": n}
    env = AlfredTWEnv(config, train_eval="eval_out_of_distribution")
    env.game_files = env.game_files[:max(n, 8)]
    env.num_games = len(env.game_files)

    t0 = time.time()
    tw = env.init_env(batch_size=n)
    out["init_env"] = round(time.time() - t0, 4)

    reset_secs = []
    infos = None
    for i in range(n_resets):
        t0 = time.time()
        _obs, infos = tw.reset()
        dt = time.time() - t0
        reset_secs.append(dt)
        print(f"  [N={n}] reset#{i}: {dt:.3f}s", flush=True)
    out["reset_cold"] = round(reset_secs[0], 4)
    out["reset_warm_list"] = [round(x, 4) for x in reset_secs[1:]]
    out["reset_warm_avg"] = round(statistics.mean(reset_secs[1:]), 4) if len(reset_secs) > 1 else None

    step_secs = []
    n_resets_during = 0
    for i in range(n_steps):
        if all(infos["won"]) or i == 0:
            pass
        acts = [infos["admissible_commands"][b][0] for b in range(n)]
        t0 = time.time()
        _o, _r, done, infos = tw.step(acts)
        step_secs.append(time.time() - t0)
        if all(done):
            t0 = time.time()
            _o, infos = tw.reset()
            reset_secs.append(time.time() - t0)
            n_resets_during += 1
    out["n_resets_during_steps"] = n_resets_during
    out["step_avg"] = round(statistics.mean(step_secs), 4)
    out["step_p50"] = round(q(step_secs, 0.5), 4)
    out["step_p90"] = round(q(step_secs, 0.9), 4)
    out["step_min"] = round(min(step_secs), 4)
    out["step_max"] = round(max(step_secs), 4)
    out["env_steps_per_sec"] = round(n / statistics.mean(step_secs), 2)
    out["wall_steps_per_sec"] = round(len(step_secs) / sum(step_secs), 2)
    try:
        tw.close()
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--resets", type=int, default=4)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    print("host:", os.uname().nodename, "| cpus:", os.cpu_count(), "| python:", sys.version.split()[0])
    bootstrap()
    import yaml
    with open(CFG) as fh:
        config = yaml.safe_load(fh)
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    results = []
    t_construct0 = time.time()
    for n in args.batches:
        print(f"=== batch_size={n} ===", flush=True)
        r = bench_one(AlfredTWEnv, config, n, args.steps, args.resets)
        print("  ->", json.dumps(r, ensure_ascii=False), flush=True)
        results.append(r)

    payload = {
        "host": os.uname().nodename,
        "cpus": os.cpu_count(),
        "python": sys.version,
        "total_wall": round(time.time() - t_construct0, 2),
        "results": results,
    }
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print("[json]", args.json)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
