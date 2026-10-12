#!/usr/bin/env python
"""A：`_align_slot_tasks` 的对抗性属性测试（独立参考实现 + 随机轰炸）。

判据（团队约定）—— 同一 task 的 g 个 slot 可互换，所以**不比 list**，比等价性：
    (1) 计数：Counter(slot_task) == Counter(i // g for i in range(len(games)))
    (2) 一致性：对每个 s，realpath(used[s]) == realpath(games[slot_task[s] * g])
另外用一个**暴力枚举**参考实现（枚举所有合法 slot->task 映射）算出合法解集合，
再断言 `_align_slot_tasks` 的返回值**属于**该集合。

运行：
    python scripts/verify_align_property.py
"""
from __future__ import annotations

import itertools
import os
import random
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from rollout import _align_slot_tasks  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        FAILS.append(f"{name} :: {detail}")
        print(f"  [FAIL] {name}  {detail}")
    else:
        print(f"  [ok]   {name}")


# --------------------------------------------------------------------------- 判据


def rp(x: str) -> str:
    return os.path.realpath(x)


def is_valid_alignment(games, used, g, slot_task) -> tuple[bool, str]:
    """合法判据 (1)+(2)。返回 (ok, reason)。"""
    n = len(games)
    if len(slot_task) != n:
        return False, f"长度 {len(slot_task)} != {n}"
    if len(used) != n:
        return False, f"used 长度 {len(used)} != {n}"
    if Counter(slot_task) != Counter(i // g for i in range(n)):
        return False, f"计数不符 Counter={Counter(slot_task)} 期望={Counter(i // g for i in range(n))}"
    for s in range(n):
        t = slot_task[s]
        if not isinstance(t, int) or t < 0 or t * g >= n:
            return False, f"slot {s} 的 task={t} 越界"
        if rp(used[s]) != rp(games[t * g]):
            return False, f"slot {s}: used={used[s]!r} != games[task{t}*g={t*g}]={games[t*g]!r}"
    return True, ""


# --------------------------------------------------------------------------- 暴力参考实现


def brute_force_valid_set(games, used, g):
    """枚举所有合法 slot->task 映射（按 gamefile 分组独立枚举），返回 set[tuple]。"""
    n = len(games)
    if len(used) != n:
        return set()
    # used 必须是 games 的多重集合
    if Counter(rp(u) for u in used) != Counter(rp(x) for x in games):
        return set()

    slots_by_gf: dict[str, list[int]] = defaultdict(list)
    for s, u in enumerate(used):
        slots_by_gf[rp(u)].append(s)

    # 每个 gamefile 组：候选 task 及其 multiplicity
    options_per_gf: dict[str, list[tuple[int, ...]]] = {}
    for gf, slots in slots_by_gf.items():
        cand: list[int] = []
        for idx, x in enumerate(games):
            if rp(x) == gf:
                cand.append(idx // g)
        # 该组 m 个 slot 上要放 cand 的一个排列（按多重集合去重）
        perms = set(itertools.permutations(cand))
        options_per_gf[gf] = sorted(perms)

    gfs = list(options_per_gf)
    valid: set[tuple[int, ...]] = set()
    for combo in itertools.product(*(options_per_gf[k] for k in gfs)):
        mapping: list[int | None] = [None] * n
        for gf, perm in zip(gfs, combo):
            for s, t in zip(slots_by_gf[gf], perm):
                mapping[s] = t
        valid.add(tuple(mapping))
    return valid


def gen_case(rng: random.Random):
    """games = k 个 task 各重复 g 次（task 间 gamefile 互不相同），used = 随机置换。"""
    k = rng.randint(1, 6)
    g = rng.randint(1, 4)
    gfs = [f"/data/task{i}/game.tw-pddl" for i in range(k)]
    games = [gfs[i] for i in range(k) for _ in range(g)]
    used = list(games)
    rng.shuffle(used)
    return games, used, g


# --------------------------------------------------------------------------- 测试项


def test_random_2000() -> None:
    print("\n== A1. 随机 2000 个 case：等价性判据 ==")
    rng = random.Random(20261012)
    n_case = 0
    n_shuffled = 0          # 真正触发打乱路径的 case 数
    mismatches: list[str] = []
    for _ in range(2000):
        games, used, g = gen_case(rng)
        n_case += 1
        if [rp(u) for u in used] != [rp(x) for x in games]:
            n_shuffled += 1
        try:
            got = _align_slot_tasks(games, used, g)
        except Exception as exc:  # noqa: BLE001
            mismatches.append(f"抛异常 games={games} used={used} g={g}: {exc!r}")
            continue
        ok, reason = is_valid_alignment(games, used, g, got)
        if not ok:
            mismatches.append(f"判据不过 games={games} used={used} g={g} got={got} :: {reason}")
    check(f"跑了 {n_case} 个 case（其中 {n_shuffled} 个真的被打乱）",
          not mismatches, f"{len(mismatches)} 个不符：{mismatches[:5]}")


def test_random_brute_force() -> None:
    print("\n== A1b. 随机 500 个 case：与暴力枚举解集比对 ==")
    rng = random.Random(424242)
    n_brute_checked = 0
    mismatches: list[str] = []
    for _ in range(500):
        games, used, g = gen_case(rng)
        try:
            got = _align_slot_tasks(games, used, g)
        except Exception as exc:  # noqa: BLE001
            mismatches.append(f"抛异常 games={games} used={used} g={g}: {exc!r}")
            continue
        vs = brute_force_valid_set(games, used, g)
        n_brute_checked += 1
        if tuple(got) not in vs:
            mismatches.append(f"不在暴力解集里 games={games} used={used} g={g} got={got} valid={vs}")
    check(f"{n_brute_checked} 个 case 的返回都落在暴力枚举出的合法解集内",
          not mismatches, f"{len(mismatches)} 个不符：{mismatches[:5]}")


def test_brute_force_meta() -> None:
    """先证明暴力参考实现本身是对的：它枚举出的解必须全部满足判据 (1)(2)。"""
    print("\n== A2. 元测试：暴力参考实现自身的正确性 ==")
    rng = random.Random(7)
    bad = []
    total_solutions = 0
    for _ in range(400):
        games, used, g = gen_case(rng)
        vs = brute_force_valid_set(games, used, g)
        if not vs:
            bad.append(f"解集为空（应至少有一个合法解）games={games} used={used} g={g}")
            continue
        for sol in vs:
            total_solutions += 1
            ok, reason = is_valid_alignment(games, used, g, list(sol))
            if not ok:
                bad.append(f"暴力解不满足判据 sol={sol} :: {reason}")
    check(f"暴力解集全部自洽（累计 {total_solutions} 个解）", not bad, f"{bad[:3]}")


def test_multiset_permutation_only() -> None:
    """确认我们生成的 used 确实是 games 的多重集合排列（否则测试无意义）。"""
    print("\n== A3. 生成器自检：used 是 games 的多重集合排列 ==")
    rng = random.Random(99)
    bad = 0
    for _ in range(1000):
        games, used, _g = gen_case(rng)
        if Counter(games) != Counter(used):
            bad += 1
    check("1000 个 case 的 used 都是 games 的排列", bad == 0, f"{bad} 个不是")


def test_explicit_cases() -> None:
    print("\n== A4. 显式 4 用例 ==")
    A, B, C, D = "games/A.tw-pddl", "games/B.tw-pddl", "games/C.tw-pddl", "games/D.tw-pddl"

    got = _align_slot_tasks([A, B, C, D], [A, B, C, D], 1)
    check(f"恒等 g=1 -> {got}", got == [0, 1, 2, 3], f"期望 [0,1,2,3]")

    got = _align_slot_tasks([A, B, C, D], [C, A, D, B], 1)
    check(f"置换 g=1 -> {got}", got == [2, 0, 3, 1], f"期望 [2,0,3,1]")

    games, used, g = [A, A, B, B], [B, A, B, A], 2
    got = _align_slot_tasks(games, used, g)
    ok, reason = is_valid_alignment(games, used, g, got)
    check(f"重复 g=2 -> {got}（等价性）", ok and got == [1, 0, 1, 0], f"{reason}")

    try:
        _align_slot_tasks([A, B], [A, A], 1)
        check("超次数 -> RuntimeError", False, "没抛")
    except RuntimeError as exc:
        check(f"超次数 -> RuntimeError ({str(exc)[:40]}...)", True)
    except Exception as exc:  # noqa: BLE001
        check("超次数 -> RuntimeError", False, f"抛了别的 {exc!r}")


def test_edges() -> None:
    print("\n== A5. 边界 ==")
    A, B, C = "g/A.tw-pddl", "g/B.tw-pddl", "g/C.tw-pddl"

    # 恒等快速路径（g>1）
    games, used, g = [A, A, B, B], [A, A, B, B], 2
    got = _align_slot_tasks(games, used, g)
    check(f"恒等快速路径 g=2 -> {got}", got == [0, 0, 1, 1], "期望 [0,0,1,1]")

    # 单元素
    check("单任务 g=1", _align_slot_tasks([A], [A], 1) == [0], "")
    check("单任务 g=3", _align_slot_tasks([A, A, A], [A, A, A], 3) == [0, 0, 0], "")

    # used 里出现 games 没有的 gamefile -> 必须抛
    try:
        _align_slot_tasks([A, B], [A, C], 1)
        check("used 含未知 gamefile -> RuntimeError", False, "没抛")
    except RuntimeError:
        check("used 含未知 gamefile -> RuntimeError", True)

    # 数量同但内容不匹配
    try:
        _align_slot_tasks([A, B], [B, C], 1)
        check("数量同内容不匹配 -> RuntimeError", False, "没抛")
    except RuntimeError:
        check("数量同内容不匹配 -> RuntimeError", True)

    # 超次数（同 gamefile 出现次数超过请求）
    try:
        _align_slot_tasks([A, B], [A, A], 1)
        check("同 gamefile 超次数 -> RuntimeError", False, "没抛")
    except RuntimeError:
        check("同 gamefile 超次数 -> RuntimeError", True)

    # g=1 全排列（4 元素）
    for perm in itertools.permutations([A, B, C]):
        got = _align_slot_tasks([A, B, C], list(perm), 1)
        ok, reason = is_valid_alignment([A, B, C], list(perm), 1, got)
        if not ok:
            check(f"3! 全排列 {perm}", False, reason)
            return
    check("3! 全排列全部等价合法", True)


def test_path_normalization() -> None:
    """realpath 归一化：带 ./ 或重复斜杠的等价路径必须被视作同一个 gamefile。"""
    print("\n== A6. 路径归一化 ==")
    a1, a2 = "g/./A.tw-pddl", "g//A.tw-pddl"
    b = "g/B.tw-pddl"
    check("identity 快速路径对等价路径成立",
          _align_slot_tasks([a1, b], [a2, b], 1) == [0, 1],
          "带 ./ 与 // 应归一化后相等")
    got = _align_slot_tasks([a1, b], [b, a2], 1)
    check(f"置换 + 路径归一化 -> {got}", got == [1, 0], "期望 [1,0]")


def test_out_of_contract_duplicate_gf_across_tasks() -> None:
    """越界探测（不在 run_rollouts 契约内）：不同 task 共用同一 gamefile。

    run_rollouts 永远生成「同一 task 的 g 个副本 gamefile 相同、task 间不同」，
    所以这个场景不会出现。这里只是记录 `_align_slot_tasks` 在这种输入下的行为，
    不当作 bug（env 只回报 gamefile，本身无法区分两个 task 共用同一局）。
    """
    print("\n== A7. 越界探测（仅记录，不算 bug）==")
    games = ["g/A.tw-pddl", "g/B.tw-pddl", "g/A.tw-pddl", "g/B.tw-pddl"]  # task0=(A,B) task1=(A,B)
    used = ["g/B.tw-pddl", "g/A.tw-pddl", "g/B.tw-pddl", "g/A.tw-pddl"]
    got = _align_slot_tasks(games, list(used), 2)
    ok, reason = is_valid_alignment(games, list(used), 2, got)
    print(f"      games={games}\n      used ={used}\n      got  ={got}  valid={ok} ({reason})")
    print("      说明：同一 gamefile 跨 task 时函数无法区分；run_rollouts 不会产生这种输入。")
    check("越界场景已记录（不判定为 bug）", True)


def main() -> int:
    for fn in (test_random_2000, test_random_brute_force, test_brute_force_meta,
               test_multiset_permutation_only, test_explicit_cases, test_edges,
               test_path_normalization, test_out_of_contract_duplicate_gf_across_tasks):
        try:
            fn()
        except Exception:  # noqa: BLE001
            FAILS.append(f"{fn.__name__} 崩溃")
            print(traceback.format_exc())
    print("\n" + ("A 全部通过 ✅" if not FAILS else f"A 失败 {len(FAILS)} 项 ❌"))
    for f in FAILS:
        print("  - " + f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
