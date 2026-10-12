#!/usr/bin/env python
"""C：独立复算 run_rollouts 的返回顺序（真跑 ALFWorld 环境，不依赖 smoke_test 的自带断言）。

做四件事：
  1. 独立复现 textworld 的 shuffle：用 numpy `RandomState(seed).shuffle(games)` 预测 used，
     再跟 env.reset() 实际返回的 used 对拍（第三个来源）。
  2. 打印 请求 games vs 实际 used，确认二者不同（否则没触发打乱路径，脚本会换 seed）。
  3. 对每条 Trajectory 用三个独立来源交叉核对：
        (a) `t.stats.task_id`  ==  `task_instance_id(t.game_file)`   （纯路径推导）
        (b) `t.messages` 里 "Your task is to: ..." 的文本
            ==  我**自己 reset 得到**的该 slot 的 view.task_desc（独立于 alignment 代码）
        (c) `t.game_file` 的 basename == env 实际装载的 used[slot]
  4. 确认第 i 组的 g 条轨迹对应同一 task（game_file / task_id / task_desc 三者组内一致），
     并对比"如果照 slot 原序分组会被污染"，证明重排是必需的。

运行（在 CPU 计算节点上，需要 ALFWORLD_DATA / 兼容层）：
    python scripts/verify_rollout_order.py
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from alfworld_compat import install as install_alfworld_compat  # noqa: E402

import numpy as np  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(f"{name} :: {detail}")


def norm_task_desc(text: str) -> str:
    """把 "Your task is to: X" 与裸 "X" 归一化到同一种形式再比较。"""
    text = (text or "").strip()
    low = text.lower()
    if low.startswith("your task is to:"):
        return text[len("your task is to:"):].strip()
    return text


def task_desc_from_messages(messages) -> str:
    for m in messages:
        if m.get("role") == "user":
            first = m["content"].split("\n")[0]
            if first.lower().startswith("your task is to:"):
                return norm_task_desc(first)
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "configs" / "base_config.yaml"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=None, help="不指定则自动挑一个能触发 shuffle 的 seed")
    args = ap.parse_args()

    install_alfworld_compat(verbose=True)

    from agents import RandomPolicy
    from alfworld_env import ALFWorldBatchEnv, load_alfworld_config, task_instance_id, task_type_of
    from reward import make_reward_cfg
    from rollout import RolloutConfig, run_rollouts

    g = args.group_size
    n_tasks = args.batch // g
    if args.batch % g != 0:
        raise SystemExit("batch 必须能被 group_size 整除")
    cfg = load_alfworld_config(args.config)

    # --- 选 N 个 task（稳定挑选）---
    pool = sorted(load_pool(cfg))
    rng = random.Random(2026)
    rng.shuffle(pool)
    chosen = pool[:n_tasks]
    tasks = [{"game_file": gf, "task_id": task_instance_id(gf), "task_type": task_type_of(gf)} for gf in chosen]
    games = [t["game_file"] for t in tasks for _ in range(g)]

    print(f"\n=== 请求 {n_tasks} 个 task × group_size={g} = {len(games)} slots ===")
    for i, t in enumerate(tasks):
        print(f"  task{i}: {t['game_file']}")
        print(f"          task_id={t['task_id']}  type={t['task_type']}")

    # --- 挑 seed：让预测的 shuffle != games ---
    seed = args.seed
    if seed is None:
        for s in range(0, 200):
            pred = list(games)
            np.random.RandomState(s).shuffle(pred)
            if pred != games:
                seed = s
                break
    print(f"\n[seed] 使用 seed={seed}")

    env = ALFWorldBatchEnv(cfg, split="train", batch_size=args.batch, seed=seed)
    try:
        # ---- 1/2. 独立预测 used 并对拍 ----
        pred = list(games)
        np.random.RandomState(seed).shuffle(pred)

        _obs, views, _adm, used1, _e = env.reset(games)
        _obs2, views2, _adm2, used2, _e2 = env.reset(games)

        print("\n=== 1/2. 独立预测 used 与真实 used 对拍 ===")
        print("  请求 games:")
        for i, x in enumerate(games):
            print(f"    slot{i}: {os.path.basename(os.path.dirname(x))}")
        print("  实际 used:")
        for i, x in enumerate(used1):
            print(f"    slot{i}: {os.path.basename(os.path.dirname(x))}")
        check("两次 reset 的 used 完全一致（reset 对同一 seed 确定性）", used1 == used2, f"{used1} vs {used2}")
        check("独立预测的 shuffle 与真实 used 一致", pred == used1,
              f"pred={pred} used={used1}")
        check("used 是 games 的多重集合排列", Counter(used1) == Counter(games))
        check("used != games（确实触发了 textworld 的打乱路径）", used1 != games,
              "used 与 games 相同，换个 seed 再来")

        # 每个 gamefile 对应的真实 task_desc（独立来源：env 自己的 view）
        desc_by_gf = {}
        for s, u in enumerate(used1):
            desc_by_gf.setdefault(os.path.realpath(u), norm_task_desc(views[s].task_desc))

        # ---- 3. 真跑 run_rollouts ----
        print("\n=== 3. run_rollouts（RandomPolicy）===")
        policy = RandomPolicy(seed=0)
        rcfg = RolloutConfig(group_size=g, max_steps=args.max_steps, temperature=1.0)
        trajs = run_rollouts(env, policy, tasks, rcfg, make_reward_cfg(None),
                             log_fn=lambda m: print("  " + m), tokenize=False)

        check("轨迹条数 == batch", len(trajs) == args.batch, f"{len(trajs)}")

        # (a) task_id == task_instance_id(game_file)
        bad_a = [(i, t.task_id, t.game_file) for i, t in enumerate(trajs)
                 if t.task_id != task_instance_id(t.game_file)]
        check("(a) 每条 traj: stats.task_id == task_instance_id(game_file)", not bad_a, f"{bad_a[:3]}")

        # (b) messages 里的 task 文本 == 该 game_file 的真实 task_desc
        bad_b = []
        for i, t in enumerate(trajs):
            got = task_desc_from_messages(t.messages)
            want = desc_by_gf.get(os.path.realpath(t.game_file), "<none>")
            if got != want:
                bad_b.append((i, got, want, os.path.basename(os.path.dirname(t.game_file))))
        check("(b) 每条 traj 的 prompt 任务文本 == 该 game_file 的真实 task_desc", not bad_b, f"{bad_b[:3]}")

        # (c) game_file basename 属该 task
        bad_c = []
        for i, t in enumerate(trajs):
            grp = i // g
            if os.path.realpath(t.game_file) != os.path.realpath(tasks[grp]["game_file"]):
                bad_c.append((i, t.game_file, tasks[grp]["game_file"]))
        check("(c) 每条 traj 的 game_file == 其所在组对应 task 的 game_file", not bad_c, f"{bad_c[:3]}")

        # ---- 4. 组内一致性 + 返回顺序 ----
        print("\n=== 4. 组内一致性 / task-major 顺序 ===")
        for j in range(n_tasks):
            grp = trajs[j * g:(j + 1) * g]
            gfs = {os.path.realpath(t.game_file) for t in grp}
            ids = {t.task_id for t in grp}
            descs = {task_desc_from_messages(t.messages) for t in grp}
            print(f"  group {j}: game_file={len(gfs)}种 task_id={sorted(ids)} task_desc={len(descs)}种")
            check(f"group {j} 内 game_file 唯一", len(gfs) == 1, f"{gfs}")
            check(f"group {j} 内 task_id 唯一", len(ids) == 1, f"{ids}")
            check(f"group {j} 内 task_desc 唯一", len(descs) == 1, f"{descs}")

        order_ids = [t.task_id for t in trajs]
        want_ids = [tasks[i // g]["task_id"] for i in range(len(trajs))]
        check("返回顺序是 task-major（第 i 组 == 第 i 个 task）", order_ids == want_ids,
              f"got={order_ids} want={want_ids}")
        check("第 i 组 == 请求列表第 i 个 task 的 game_file",
              all(os.path.realpath(trajs[j * g].game_file) == os.path.realpath(tasks[j]["game_file"])
                  for j in range(n_tasks)))

        # ---- 反证：如果不重排（按 slot 原序返回）会怎样 ----
        print("\n=== 反证：按 slot 原序返回（未重排）的归因错误 ===")
        n = len(used1)
        misattributed = sum(1 for i in range(n)
                            if os.path.realpath(used1[i]) != os.path.realpath(games[i]))
        print(f"  slot 原序下，{misattributed}/{n} 个 slot 实际装载的 game != 请求列表同位置的 game")
        if g > 1:
            polluted = 0
            for j in range(n_tasks):
                grp = used1[j * g:(j + 1) * g]
                if len({os.path.realpath(x) for x in grp}) != 1:
                    polluted += 1
            print(f"  未重排时按 g={g} 分组的被污染组数：{polluted}/{n_tasks}")
        check("未重排确实会归因错误（证明重排不是多余的）", misattributed > 0,
              "本 seed 下 slot 原序恰与请求一致，换个 seed 再验")
    finally:
        env.close()

    print("\n" + ("C 全部通过 ✅" if not FAILS else f"C 失败 {len(FAILS)} 项 ❌"))
    for f in FAILS:
        print("  - " + f)
    return 1 if FAILS else 0


def load_pool(cfg):
    from alfworld_env import collect_game_files
    return collect_game_files(cfg, "train")


if __name__ == "__main__":
    raise SystemExit(main())
