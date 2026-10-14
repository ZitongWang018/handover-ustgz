#!/usr/bin/env python
"""评测框架的**决定性自检**：用 ALFWorld 官方 expert 轨迹驱动我们自己的评测环境。

为什么需要这个
---------------------------------------------------------------------------
"模型在 valid_unseen 上得 0"有两种完全不同的解释：
  (a) 模型真的弱；
  (b) 我们的评测环境/判胜链路坏了（比如 `won` 恒 False、动作没送进游戏、
      或者任务根本无解）。
光看数字分不出来。**唯一干净的判据是让官方专家来跑**：
ALFWorld 每个 game 文件自带 expert（`with_expert=True` 时 `reset()` 返回
`extra.expert_plan`，即"专家下一步该做什么"）。如果连专家都判不了胜，
那问题一定在评测框架，不在模型。

顺带量两件事：
  * 专家走完一个任务需要多少步（对照我们的 `max_steps` 上限 —— 如果专家都要
    >30 步，那 30 步上限就等于把任务判死，任何策略都不可能赢）；
  * 每一步 `admissible` 是否非空（空列表说明动作空间是坏的）。

用法：
    python scripts/ez_eval_oracle_check.py [split] [max_steps] [n_tasks]
    split 默认 eval_out_of_distribution（= 论文口径的 valid_unseen）
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from alfworld_env import ALFWorldBatchEnv, collect_game_files, load_alfworld_config  # noqa: E402


def main() -> int:
    split = sys.argv[1] if len(sys.argv) > 1 else "eval_out_of_distribution"
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    n_want = int(sys.argv[3]) if len(sys.argv) > 3 else 12

    proj = SRC.parent
    cfg = load_alfworld_config(str(proj / "configs" / "base_config.yaml"))
    games = sorted(collect_game_files(cfg, split))
    n = min(n_want, len(games))
    print("split=%s  任务数=%d  步数上限=%d  抽检=%d" % (split, len(games), cap, n))

    wins = 0
    truncated = 0
    steps_used = []
    empty_adm = 0
    tasks = []
    for i, game in enumerate(games[:n]):
        env = ALFWorldBatchEnv(cfg, split=split, batch_size=1,
                               max_episode_steps=cap, with_expert=True,
                               with_facts=False, ignore_benchmark_done=False)
        try:
            _obs, _views, adm, _used, plan = env.reset([game])
            if not adm or not adm[0]:
                empty_adm += 1
            won, n_step, plan_empty = False, 0, 0
            for _t in range(cap):
                act = plan[0] if (plan and plan[0]) else "look"
                if not (plan and plan[0]):
                    plan_empty += 1
                bs = env.step([act])
                n_step += 1
                plan = getattr(bs, "expert_plan", None)
                if bool(bs.dones[0]):
                    won = bool(bs.won[0]) if bs.won else False
                    break
            else:
                truncated += 1
            steps_used.append(n_step)
            wins += 1 if won else 0
            name = game.replace("\\", "/").split("/")[-3][:44]
            print("  [%2d] %-5s steps=%2d plan_empty=%d  %s"
                  % (i, "WIN" if won else "LOSE", n_step, plan_empty, name))
            tasks.append({"game": game, "won": won, "steps": n_step,
                          "plan_empty": plan_empty})
        finally:
            env.close()

    print("\n=== 专家 Oracle 汇总 ===")
    print("胜 %d/%d   被 %d 步上限截断 %d/%d   初始 admissible 为空 %d/%d"
          % (wins, n, cap, truncated, n, empty_adm, n))
    print("专家步数 min=%d max=%d mean=%.1f" % (
        min(steps_used), max(steps_used), statistics.mean(steps_used)))
    if wins == n:
        print("VERDICT PASS：评测框架能正确判胜 —— 模型得 0 是**模型**的问题，不是评测坏了")
        rc = 0
    else:
        print("VERDICT FAIL：连官方专家都判不了胜，**评测框架有问题**，先修框架再看模型")
        rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
