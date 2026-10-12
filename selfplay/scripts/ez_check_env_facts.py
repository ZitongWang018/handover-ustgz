#!/usr/bin/env python
"""集群首次运行检查：确认 textworld 的 infos 里真的有 facts（EZ 的命根子）。

没有 facts，状态匹配会静默降级成观测哈希模式（ez_state.snapshot_from_infos 的
text_fallback）——那种模式下子任务验证全是"观测变了就算过"，奖励基本是废的。
smoke 之前必须先跑这个脚本。

用法（集群上）：
    python scripts/ez_check_env_facts.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from alfworld_compat import install

install()

import textworld  # noqa: E402


def main() -> int:
    print(f"textworld version: {textworld.__version__}")
    try:
        infos = textworld.EnvInfos(won=True, admissible_commands=True, facts=True, extras=["gamefile"])
        print(f"[ok] EnvInfos 支持 facts 字段: {infos.facts}")
    except (AttributeError, TypeError) as exc:
        print(f"[FATAL] 当前 textworld 不支持 facts: {exc}")
        print("        升级 textworld>=1.5，或整改 alfworld_env._make_env 的 request_infos")
        return 1

    # 真正起一个环境验证 facts 有内容
    from alfworld_env import ALFWorldBatchEnv, collect_game_files, load_alfworld_config
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "base_config.yaml")
    cfg = load_alfworld_config(str(cfg_path))
    games = collect_game_files(cfg, split="eval_in_distribution")[:1]
    if not games:
        print("[FATAL] 没有可用 game 文件，检查 ALFWORLD_DATA")
        return 1
    env = ALFWorldBatchEnv(cfg, split="eval_in_distribution", batch_size=1, with_facts=True)
    try:
        obs, views, adm, used, _ = env.reset(games)
        facts = getattr(env, "last_reset_facts", [None])[0]
        if facts:
            print(f"[ok] reset 后 facts 有内容（{len(facts)} 条），示例：")
            for f in list(facts)[:5]:
                print(f"      {f}")
            return 0
        print("[FATAL] with_facts=True 但 infos 里没有 facts —— 检查 textworld 版本")
        return 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
