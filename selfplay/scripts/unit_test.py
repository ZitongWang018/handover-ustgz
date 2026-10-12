#!/usr/bin/env python
"""纯本机可跑的单元测试（不碰集群、不碰模型）。

    python scripts/unit_test.py

覆盖：动作解析、admissible 对齐、chat 模板无关的 prompt 渲染、奖励与 advantage、
评测策略对 `Admissible Commands` 块的解析、以及 slot/task 对齐（防 textworld shuffle）。
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import agents  # noqa: E402
import prompts as P  # noqa: E402
import reward as R  # noqa: E402
import rollout  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def test_parse_action() -> None:
    print("\n== parse_action ==")
    cases = [
        ("Thought: I should look around.\nAction: look", "look", True),
        ("Thought: go\nAction: go to shelf 1\n", "go to shelf 1", True),
        ("my action: take apple 1 from countertop", "take apple 1 from countertop", True),
        ("Thought: hmm\nlook", "look", False),
        ("**Action:** open fridge 1", "open fridge 1", True),
        ("Thought: x\nAction: `take mug 1 from desk`", "take mug 1 from desk", True),
        ("Action: go to shelf 1.", "go to shelf 1", True),
        ("Thought: x\nAction:", "", True),
    ]
    for text, want, want_marker in cases:
        got = P.parse_action(text)
        check(f"{text!r:52s} -> {got.action!r}", got.action == want, f"expected {want!r}")
        check(f"{'':52s} marker={got.has_marker}", got.has_marker == want_marker,
              f"expected {want_marker}")


def test_match_and_stop() -> None:
    print("\n== match_admissible / stop ==")
    adm = ["go to shelf 1", "look", "take apple 1 from countertop"]
    check("case-insensitive 命中", P.match_admissible("Go To Shelf 1", adm) == "go to shelf 1")
    check("非法动作不命中", P.match_admissible("dance", adm) is None)
    check("空串不命中", P.match_admissible("", adm) is None)
    check("模糊匹配（0.6）", P.match_admissible("take apple from countertop", adm, 0.6) is not None)
    check("未写完不停", not P.stop_text_finished("Thought: a\nAction: look"))
    check("写完就停", P.stop_text_finished("Thought: a\nAction: look\n"))


def test_render() -> None:
    print("\n== 观测渲染 ==")
    txt = P.format_observation("You are in a room.", ["look", "go to shelf 1"], first_turn=True, task_desc="put apple in fridge")
    check("含任务", "Your task is to: put apple in fridge" in txt)
    check("含观测", "Observation: You are in a room." in txt)
    check("含动作表", "- go to shelf 1" in txt)


def test_reward() -> None:
    print("\n== 奖励 ==")
    win = R.TrajectoryStats(won=True, n_steps=10, n_valid_steps=10)
    fail = R.TrajectoryStats(won=False, n_steps=30, n_valid_steps=20, n_invalid=8,
                             n_format_violation=2, n_repeat=5)
    rich = R.make_reward_cfg(None)
    sparse = R.make_reward_cfg({"use_sparse_only": True})
    rw_win, rw_fail = R.compute_reward(win, rich, 30), R.compute_reward(fail, rich, 30)
    sp_win, sp_fail = R.compute_reward(win, sparse, 30), R.compute_reward(fail, sparse, 30)
    check("稀疏：成功=1", abs(sp_win - 1.0) < 1e-9, f"got {sp_win}")
    check("稀疏：失败=0", abs(sp_fail) < 1e-9, f"got {sp_fail}")
    check("丰富：成功 > 1", rw_win > 1.0, f"got {rw_win}")
    check("丰富：失败 != 0（有过程信号）", abs(rw_fail) > 1e-9, f"got {rw_fail}")
    check("裁剪生效", rw_fail <= rich.clip_max and rw_fail >= rich.clip_min)


def test_advantages() -> None:
    print("\n== group advantage ==")
    check("全同分 -> 全 0", R.group_advantages([1, 1, 1, 1], [0, 0, 0, 0]) == [0.0] * 4)
    a = R.group_advantages([1, 0, 1, 0], [0, 0, 0, 0])
    check("1/0 分组 -> ±1", abs(a[0] - 1) < 1e-3 and abs(a[1] + 1) < 1e-3, str(a))
    a2 = R.group_advantages([1, 0, 0.5, 0.5], [0, 0, 1, 1])
    check("退化组 -> 0", abs(a2[2]) < 1e-9 and abs(a2[3]) < 1e-9, str(a2))
    st = R.summarize_grouping([1, 0, 0, 0], [0, 0, 1, 1])
    check("退化组计数", abs(st["group_degenerate_frac"] - 0.5) < 1e-9, str(st))


def test_agents_parse_admissible() -> None:
    print("\n== agents 解析 admissible ==")
    msgs = [{"role": "user", "content": P.format_observation(
        "You see a shelf.", ["look", "go to shelf 1", "take apple 1 from countertop"])}]
    adm = agents.parse_admissible_from_messages(msgs)
    check("解析出 3 条", adm == ["look", "go to shelf 1", "take apple 1 from countertop"], str(adm))
    out = agents.RandomPolicy(seed=0).generate([msgs])[0][0]
    check("随机策略产出可解析", P.parse_action(out).has_marker, out)
    out2 = agents.ScriptedOraclePolicy().generate([msgs])[0][0]
    check("脚本策略优先 take", "take apple 1 from countertop" in out2, out2)


def test_align_slot_tasks() -> None:
    print("\n== _align_slot_tasks（slot/task 对齐）==")
    A, B, C, D = "games/A.tw-pddl", "games/B.tw-pddl", "games/C.tw-pddl", "games/D.tw-pddl"
    # 恒等（未 shuffle）
    check("恒等 g=1", rollout._align_slot_tasks([A, B, C, D], [A, B, C, D], 1) == [0, 1, 2, 3])
    # 置换（textworld 打乱后）
    check("置换 g=1", rollout._align_slot_tasks([A, B, C, D], [C, A, D, B], 1) == [2, 0, 3, 1])
    # 重复 + g=2：旧实现（dict 反查）会因重复 key 压塌而崩，新实现必须返回 [1, 0, 1, 0]
    check("重复 g=2", rollout._align_slot_tasks([A, A, B, B], [B, A, B, A], 2) == [1, 0, 1, 0])
    # 同一 gamefile 出现次数超过请求列表 → 必须抛 RuntimeError（而不是静默错配）
    try:
        rollout._align_slot_tasks([A, B], [A, A], 1)
        check("超次数抛 RuntimeError", False, "未抛错")
    except RuntimeError:
        check("超次数抛 RuntimeError", True)


def main() -> int:
    for fn in (test_parse_action, test_match_and_stop, test_render, test_reward,
               test_advantages, test_agents_parse_admissible, test_align_slot_tasks):
        fn()
    print("\n" + ("全部通过 ✅" if not FAILS else f"失败 {len(FAILS)} 项 ❌: {FAILS}"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
