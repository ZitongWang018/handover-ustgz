#!/usr/bin/env python
"""用 ALFWorld expert 原始命令提供环境侧动作与判胜诊断。

为什么需要这个
---------------------------------------------------------------------------
"模型在 valid_unseen 上得 0"有两种完全不同的解释：
  (a) 模型真的弱；
  (b) 我们的评测环境/判胜链路坏了（比如 `won` 恒 False、动作没送进游戏、
      或者任务根本无解）。
expert rollout 是补充诊断，不能单独归因模型格式、策略或框架问题：
ALFWorld 每个 game 文件自带 expert（`with_expert=True` 时 `reset()` 返回
`extra.expert_plan`，即"专家下一步该做什么"）。如果连专家都判不了胜，
应结合预算、空计划、环境执行状态和逐步 trace 继续分析。

顺带量两件事：
  * 专家走完一个任务需要多少步；超出当前预算只表示专家轨迹被截断，
    不能据此推断其他策略的成功上限；
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
    budget_boundary = 0
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
            won, n_step, plan_empty, terminal = False, 0, 0, False
            for _t in range(cap):
                act = plan[0] if (plan and plan[0]) else "look"
                if not (plan and plan[0]):
                    plan_empty += 1
                bs = env.step([act])
                n_step += 1
                plan = getattr(bs, "expert_plan", None)
                if bool(bs.dones[0]):
                    terminal = True
                    won = bool(bs.won[0]) if bs.won else False
                    break
            if not terminal:
                truncated += 1
            steps_used.append(n_step)
            wins += 1 if won else 0
            name = game.replace("\\", "/").split("/")[-3][:44]
            if won:
                status = "won"
            elif n_step >= cap:
                budget_boundary += 1
                status = "budget_boundary_not_won"
            elif terminal:
                status = "environment_terminal_not_won"
            else:
                status = "budget_truncated"
            print("  [%2d] %-30s steps=%2d plan_empty=%d  %s"
                  % (i, status, n_step, plan_empty, name))
            tasks.append({"game": game, "status": status, "won": won, "steps": n_step,
                          "plan_empty": plan_empty})
        finally:
            env.close()

    print("\n=== Expert environment diagnostics ===")
    print("胜 %d/%d   budget_truncated=%d   budget_boundary_not_won=%d   initial_admissible_empty=%d/%d"
          % (wins, n, truncated, budget_boundary, empty_adm, n))
    if steps_used:
        print("专家步数 min=%d max=%d mean=%.1f" % (
            min(steps_used), max(steps_used), statistics.mean(steps_used)))
    print("Expert raw commands bypass model JSON parsing; inspect task traces before drawing conclusions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
