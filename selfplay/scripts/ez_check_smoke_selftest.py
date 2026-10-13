#!/usr/bin/env python
"""本地回归 ez_check_smoke_ok.py 的判读逻辑（不需 GPU / 不需 pyyaml）。

跑法：
    python scripts/ez_check_smoke_selftest.py

这些用例全部来自**真实踩过的坑**（job 12989143 / 12989450）：
  * B  = 旧 probe 挑到 lora_A（B 初始全 0 ⇒ ∂L/∂A ≡ 0）打出的 update_rel=0 假警报
  * C  = 真的梯度链断（奖励有区分度但一个张量都没动）
  * I/J/K = 8 卡分片的三类典型故障（rank 日志缺失 / 某个 rank 没参与梯度同步）
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECKER = HERE / "ez_check_smoke_ok.py"

GRADCHAIN = ("[selfcheck] 梯度链 proposer grad_norm=5.6663e+02 (504 个张量有 grad) | "
             "solver grad_norm=5.6441e+02 (504 个张量有 grad)")
FACTS = "[selfcheck] facts 通道 OK（109 条事实）"


def _dist_line(rank, world):
    return f"[dist] rank={rank}/{world} local_rank={rank} device=cuda:{rank} | run_dir=/tmp/x"


def _p_line(outer=0, step=0, acc=1, tot=2, rews="[0.3, 0.0]", spread="0.300", rank=0):
    return (f"[outer {outer}][P{step}] rank={rank} 本地 games=1/1 | pool=1 (+1) "
            f"| accept={acc}/{tot} | rew={rews} | rew_spread={spread} | 87s")


def _upd_new(outer=0, step=0, rel="1.000e+00", moved=252, tot=504, dmax="1.000e-04"):
    return (f"[outer {outer} P{step}] update_rel={rel}"
            f"（=max ‖Δp‖/(‖p‖+‖Δp‖)；{moved}/{tot} 张量真的动了；max‖Δp‖={dmax}）")


def _upd_old(outer=0, step=0, rel="0.000e+00"):
    """旧 probe 的日志格式（回归用：必须被判成"跑的是旧代码"）。"""
    return f"[outer {outer} P{step}] update_rel={rel}（必须为 >>1e-10，否则梯度是死的）"


def _diag_p(outer=0, step=0, gn="4.3530e+00", n=504, ga="0.0000e+00", gb="4.3530e+00"):
    """P 相位诊断行（真实 backward grad_norm）—— 新版判据的权威依据。"""
    return (f"[outer {outer}][P{step}] 诊断 adv_spread/n_active/len_ids/mask_sum="
            f"[(2.309, 2651, [2532], [716])] | 真实 backward grad_norm={gn} "
            f"({n} 个张量有 grad；A={ga} B={gb})")


def _diag_s(outer=0, step=0, gn="4.3258e+00", n=504, ga="0.0000e+00", gb="4.3258e+00"):
    """S 相位诊断行（2026-10-12 新增：只有 P 有诊断行时 S 的 0/504 无法判定）。"""
    return (f"[outer {outer}][S{step}] 诊断 真实 backward grad_norm={gn} "
            f"({n} 个张量有 grad；A={ga} B={gb})")


SREWARD = "[outer 0 S0] rank=0 batch=1/1 | reward mean=0.250 | final_match=2/8 | pool=4 | 60s"


def run_case(name, msgs, expect_rc, must_contain=(), rank_msgs=None):
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        with open(d / "train_log.jsonl", "w") as fh:
            for m in msgs:
                fh.write(json.dumps({"t": 0, "msg": m}, ensure_ascii=False) + "\n")
        for rk, rmsgs in (rank_msgs or {}).items():
            with open(d / f"train_log.rank{rk}.jsonl", "w") as fh:
                for m in rmsgs:
                    fh.write(json.dumps({"t": 0, "rank": rk, "msg": m},
                                        ensure_ascii=False) + "\n")
        p = subprocess.run([sys.executable, str(CHECKER), "--config", "x.yaml",
                            "--run-dir", str(d)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        out = (p.stdout or "") + (p.stderr or "")
        ok = (p.returncode == expect_rc)
        miss = [s for s in must_contain if s not in out]
        tag = "PASS" if (ok and not miss) else "FAIL"
        print(f"[{tag}] {name}  rc={p.returncode} (期望 {expect_rc})")
        if not ok or miss:
            print("  --- 输出 ---")
            print("  " + out.replace("\n", "\n  "))
            if miss:
                print(f"  缺少期望串: {miss}")
        return tag == "PASS"


def _rank_msgs(rel_by_rank, world=8, n_files=None, gradchain=GRADCHAIN, facts=FACTS,
               gn_by_rank=None, moved=None, accept=None, spread=None,
               accept_by_rank=None):
    """造一套 rank 日志；rel_by_rank={rank: "1.0e+00", ...}。

    `world` 是**日志里写的** world_size，`n_files` 是**实际落盘的 rank 文件数** ——
    两者故意分开：torchrun 没生效时正好是"8 个进程、每个都以为 world=1"。
    `gn_by_rank` 用来单独控制某 rank 的真实 grad_norm（默认 4.3530e+00）。
    `accept_by_rank` 可逐 rank 覆盖 accept（用例 Q 用：只有 rank3 有产出）。
      ⚠️ 别写成 `accept or 1` —— accept=0 会被吞成 1，就造不出"全被拒"的用例了。
    """
    n_files = n_files if n_files is not None else world
    gn_by_rank = gn_by_rank or {}
    out = {}
    for r in range(n_files):
        _gn = gn_by_rank.get(r, "4.3530e+00")
        _acc = accept if accept_by_rank is None else accept_by_rank.get(r, accept)
        _acc = 1 if _acc is None else _acc
        out[r] = [_dist_line(r, world), facts, gradchain,
                  "[adapter-sync] phase=proposer step=0 status=PASS max_abs=0.000e+00",
                  "[adapter-sync] phase=solver step=0 status=PASS max_abs=0.000e+00",
                  _p_line(rank=r, acc=_acc, tot=2,
                          rews=("[0.3, 0.0]" if (spread is None or spread > 0) else "[0.3, 0.3]"),
                          spread=("0.300" if spread is None else f"{spread:.3f}")),
                  _upd_new(rel=rel_by_rank.get(r, "1.000e+00"),
                           moved=(252 if moved is None else moved)),
                  _diag_p(outer=0, step=0, gn=_gn), _diag_s(outer=0, step=0, gn=_gn)]
    return out


def main() -> int:
    res = []
    ALL8 = {r: "1.0e+00" for r in range(8)}
    base = [GRADCHAIN, FACTS, _p_line(), _diag_p(), _upd_new(), SREWARD]

    # A 全绿（新 probe：rel>0 且 252/504 动了）-> 硬通过
    res.append(run_case("A 全绿=硬通过", base, 0, ["结论：**通过**", "252/504"]))

    # B 假警报回归：旧 probe 只挑 lora_A -> rel 精确 0，但 grad_norm 非零
    res.append(run_case(
        "B 旧 probe 假警报(必须报出来)",
        [GRADCHAIN, FACTS, _p_line(), _upd_old(),
         _diag_p(gn="6.1474e+00", ga="0.0000e+00", gb="6.1474e+00")],
        1, ["关2b", "旧代码"]))

    # C 真断链：有梯度(6.1e+00) 但 0/504 动了 -> FATAL
    res.append(run_case(
        "C 真断链(有梯度却 0 位移)",
        [GRADCHAIN, FACTS, _p_line(), _diag_p(),
         _upd_new(rel="0.000e+00", moved=0, tot=504, dmax="0.000e+00")],
        1, ["关2b **真断链**", "**未通过**"]))

    # D 组内全同分 + accept=0 -> 关2b/关4 不该再误报，但 accept=0 仍是 FATAL
    res.append(run_case(
        "D 无信号(全同分)+accept=0",
        [GRADCHAIN, FACTS, _p_line(acc=0, tot=2, rews="[0.0, 0.0]", spread="0.000"),
         _diag_p(gn="0.0000e+00", n=0, ga="0.0000e+00", gb="0.0000e+00"),
         _upd_new(rel="0.000e+00", moved=0, tot=504, dmax="0.000e+00")],
        1, ["未通过", "正确的"]))

    # E 梯度链断（关0）-> FATAL
    res.append(run_case(
        "E 关0 梯度链断",
        ["[selfcheck] 梯度链 proposer grad_norm=0.0 (0 个张量有 grad) | "
         "solver grad_norm=5.6e+02 (504 个张量有 grad)", FACTS, _p_line(), _diag_p(),
         _upd_new()],
        1, ["关0 梯度链断"]))

    # F facts 缺失 -> FATAL
    res.append(run_case("F facts 缺失", [GRADCHAIN, _p_line(), _diag_p(), _upd_new()],
                        1, ["关1"]))

    # G 没有 update_rel 行（旧代码路径）-> FATAL
    res.append(run_case("G 无 update_rel", [GRADCHAIN, FACTS, _p_line(), _diag_p()],
                        1, ["关2"]))

    # H solver 奖励全 0 -> WARN（不致命），proposer 动过 -> 仍硬通过
    res.append(run_case(
        "H solver 全 0 但 proposer 动过",
        [GRADCHAIN, FACTS, _p_line(), _diag_p(),
         _upd_new(rel="1.0e+00", moved=252, tot=504, dmax="1.0e-04"),
         "[outer 0 S0] reward mean=0.000 | final_match=0/8"],
        0, ["结论：**通过**"]))

    # I 8 卡全绿：8 个 rank 日志齐、world=8、量级一致 -> 硬通过
    _rm = _rank_msgs(ALL8)
    res.append(run_case(
        "I 8 卡全绿", _rm[0], 0,
        ["结论：**通过**", "8 个 rank 日志都存在且非空", "world_size=[8]"], rank_msgs=_rm))

    # J 8 卡里 rank7 日志为空 -> FATAL（那个 rank 提早挂了）
    _rm = _rank_msgs(ALL8)
    _rm[7] = []
    res.append(run_case(
        "J 8 卡少一个 rank 的日志",
        _rm[0], 1, ["rank 日志是空的", "train_log.rank7.jsonl"], rank_msgs=_rm))

    # K 8 卡里 rank3 的 update_rel 比别的 rank 小 1000× 且它自己有梯度 -> FATAL
    _rm = _rank_msgs(ALL8)
    _rm[3] = [_dist_line(3, 8), FACTS, GRADCHAIN, _p_line(rank=3),
              _upd_new(rel="1.0e-03"), _diag_p()]
    res.append(run_case(
        "K 8 卡某 rank 梯度没同步",
        _rm[0], 1, ["关4 各 rank 的 max(update_rel)", "1000.0"], rank_msgs=_rm))

    # L 8 个进程都以为自己是 world=1（torchrun 没生效）-> WARN，不致命
    _rm = _rank_msgs(ALL8, world=1, n_files=8)
    res.append(run_case("L world_size=1 警告", _rm[0], 0, ["torchrun 没生效"], rank_msgs=_rm))

    # M 单卡运行：只有 rank0 文件，world_size=1 是**正常的**，不该报 WARN
    #   （job 12989528 实测踩到：单卡也会写 train_log.rank0.jsonl）
    res.append(run_case(
        "M 单卡不误报关4",
        [_dist_line(0, 1), GRADCHAIN, FACTS, _p_line(), _diag_p(), _upd_new(), SREWARD],
        0, ["关4 单卡运行", "结论：**通过**"],
        rank_msgs={0: [_dist_line(0, 1), GRADCHAIN, FACTS, _p_line(), _diag_p(), _upd_new()]}))

    # N ★ 无信号（全同分）但 proposer 有产出 -> 0/504 是**正确的**，只能"条件通过"
    #   （job 12989603 实测：旧判据在这正是打了两条 FATAL）
    res.append(run_case(
        "N 无信号全同分=条件通过(不误报 FATAL)",
        [GRADCHAIN, FACTS, _p_line(acc=1, tot=2, rews="[0.3, 0.3]", spread="0.000"),
         _diag_p(gn="0.0000e+00", n=0, ga="0.0000e+00", gb="0.0000e+00"),
         _upd_new(rel="0.000e+00", moved=0, tot=504, dmax="0.000e+00"), SREWARD],
        0, ["条件通过", "真实 backward grad_norm 全为 0", "正确的"]))

    # O ★ 2 卡：两 rank 一致地"没梯度"（update_rel 全 0、grad_norm 全 0）-> 关4 不报 FATAL
    _rm = _rank_msgs({0: "0.000e+00", 1: "0.000e+00"}, world=2, moved=0, spread=0.0,
                     gn_by_rank={0: "0.0000e+00", 1: "0.0000e+00"})
    res.append(run_case(
        "O 2 卡一致无梯度(不误报关4)",
        _rm[0], 0, ["关4 各 rank 一致无梯度", "条件通过"], rank_msgs=_rm))

    # P ★ 2 卡：某 rank 有梯度却 0 位移 -> 真断链，仍然 FATAL
    _rm = _rank_msgs({0: "0.000e+00", 1: "0.000e+00"}, world=2, moved=0, spread=0.0,
                     gn_by_rank={0: "4.3530e+00", 1: "0.0000e+00"})
    res.append(run_case(
        "P 2 卡某 rank 真断链(有梯度却不动)",
        _rm[0], 1, ["关2b **真断链**", "关4 各 rank 的 max(update_rel)"], rank_msgs=_rm))

    # Q ★ 8 卡：rank0 自己 accept=0/2，但 rank3 有 2/2 -> 全局有产出，不该 FATAL
    #     （job 12989532 实测：8 卡各 rank 只有 1 个 game，rank0 被拒是常态。
    #       旧判据只看 rank0，把一发完全正确的 8 卡作业判成 FAILED。）
    _rm = _rank_msgs({r: "5.0e-01" for r in range(8)},
                     accept=0, accept_by_rank={3: 2})
    res.append(run_case(
        "Q 8 卡 rank0 被拒但 rank3 有产出(不误报)",
        _rm[0], 0, ["关3 proposer 有有效提案", "r3=2/2", "全局 2/16"], rank_msgs=_rm))

    # R ★ 8 卡：所有 rank 都是 accept=0/2 -> 真的全拒，仍然 FATAL
    _rm = _rank_msgs({r: "0.0e+00" for r in range(8)}, accept=0, spread=0.0)
    res.append(run_case(
        "R 8 卡全部被拒(仍 FATAL)",
        _rm[0], 1, ["全部被拒", "全局 0/16"], rank_msgs=_rm))

    print(f"\n{sum(res)}/{len(res)} PASS")
    return 0 if all(res) else 1


if __name__ == "__main__":
    raise SystemExit(main())
