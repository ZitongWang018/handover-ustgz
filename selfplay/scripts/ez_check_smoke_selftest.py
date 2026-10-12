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


def _rank_msgs(rel_by_rank, world=8, gradchain=GRADCHAIN, facts=FACTS):
    """造一套 8 个 rank 的完整日志；rel_by_rank={rank: "1.0e+00", ...}。"""
    out = {}
    for r in range(world):
        out[r] = [_dist_line(r, world), facts, gradchain,
                  _p_line(rank=r), _upd_new(rel=rel_by_rank.get(r, "1.000e+00"))]
    return out


def main() -> int:
    res = []
    base = [GRADCHAIN, FACTS, _p_line(), _upd_new(), SREWARD]

    # A 全绿（新 probe：rel>0 且 252/504 动了）-> 硬通过
    res.append(run_case("A 全绿=硬通过", base, 0, ["结论：**通过**", "252/504"]))

    # B 假警报回归：旧 probe 只挑 lora_A -> rel 精确 0，但 grad_norm 非零
    res.append(run_case(
        "B 旧 probe 假警报(必须报出来)",
        [GRADCHAIN, FACTS, _p_line(), _upd_old(),
         "[outer 0][P0] 诊断 adv_spread/n_active/len_ids/mask_sum="
         "[(2.309, 2651, [2532], [716])] | 真实 backward grad_norm=6.1474e+00 "
         "(504 个张量有 grad；A=0.0000e+00 B=6.1474e+00)  ← A=0 是正常的（B 初始全 0）"],
        1, ["关2b", "旧代码"]))

    # C 真断链：奖励有区分度(0.3) 但 0/504 动了 -> FATAL
    res.append(run_case(
        "C 真断链(moved=0)",
        [GRADCHAIN, FACTS, _p_line(),
         _upd_new(rel="0.000e+00", moved=0, tot=504, dmax="0.000e+00")],
        1, ["**未通过**"]))

    # D 组内全同分 + 0 动 -> 正常，只是"条件通过"
    res.append(run_case(
        "D 无信号(全同分)+accept=0",
        [GRADCHAIN, FACTS, _p_line(acc=0, tot=2, rews="[0.0, 0.0]", spread="0.000"),
         _upd_new(rel="0.000e+00", moved=0, tot=504, dmax="0.000e+00")],
        1, ["未通过"]))   # accept=0 -> 关3 FATAL

    # E 梯度链断（关0）-> FATAL
    res.append(run_case(
        "E 关0 梯度链断",
        ["[selfcheck] 梯度链 proposer grad_norm=0.0 (0 个张量有 grad) | "
         "solver grad_norm=5.6e+02 (504 个张量有 grad)", FACTS, _p_line(), _upd_new()],
        1, ["关0 梯度链断"]))

    # F facts 缺失 -> FATAL
    res.append(run_case("F facts 缺失", [GRADCHAIN, _p_line(), _upd_new()], 1, ["关1"]))

    # G 没有 update_rel 行（旧代码路径）-> FATAL
    res.append(run_case("G 无 update_rel", [GRADCHAIN, FACTS, _p_line()], 1, ["关2"]))

    # H solver 奖励全 0 -> WARN（不致命），proposer 动过 -> 仍硬通过
    res.append(run_case(
        "H solver 全 0 但 proposer 动过",
        [GRADCHAIN, FACTS, _p_line(),
         _upd_new(rel="1.0e+00", moved=252, tot=504, dmax="1.0e-04"),
         "[outer 0 S0] reward mean=0.000 | final_match=0/8"],
        0, ["结论：**通过**"]))

    # I 8 卡全绿：8 个 rank 日志齐、world=8、量级一致 -> 硬通过
    res.append(run_case(
        "I 8 卡全绿",
        _rank_msgs({r: "1.0e+00" for r in range(8)})[0],
        0, ["结论：**通过**", "8 个 rank 日志都存在且非空", "world_size=[8]"],
        rank_msgs=_rank_msgs({r: "1.0e+00" for r in range(8)})))

    # J 8 卡里 rank7 日志为空 -> FATAL（那个 rank 提早挂了）
    _rm = _rank_msgs({r: "1.0e+00" for r in range(8)})
    _rm[7] = []
    res.append(run_case(
        "J 8 卡少一个 rank 的日志",
        _rm[0], 1, ["rank 日志是空的", "train_log.rank7.jsonl"], rank_msgs=_rm))

    # K 8 卡里 rank3 的 update_rel 比别的 rank 小 1000× -> FATAL（没参与梯度同步）
    _rm = _rank_msgs({r: "1.0e+00" for r in range(8)})
    _rm[3] = [_dist_line(3, 8), FACTS, GRADCHAIN, _p_line(rank=3),
              _upd_new(rel="1.0e-03")]
    res.append(run_case(
        "K 8 卡某 rank 梯度没同步",
        _rm[0], 1, ["关4 各 rank 的 max(update_rel)", "1000.0"], rank_msgs=_rm))

    # L 8 卡但 world_size 记成 1 -> WARN（torchrun 没生效），不致命
    _rm = _rank_msgs({r: "1.0e+00" for r in range(8)}, world=1)
    res.append(run_case("L world_size=1 警告", _rm[0], 0, ["torchrun 没生效"], rank_msgs=_rm))

    print(f"\n{sum(res)}/{len(res)} PASS")
    return 0 if all(res) else 1


if __name__ == "__main__":
    raise SystemExit(main())
