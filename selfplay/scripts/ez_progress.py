#!/usr/bin/env python
"""从 train_log.rank0.jsonl 汇总 EZ 训练进度与关键指标。

用法：python scripts/ez_progress.py <run_dir>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def main() -> int:
    run = Path(sys.argv[1]).expanduser()
    log = run / "train_log.rank0.jsonl"
    if not log.is_file():
        print(f"找不到 {log}")
        return 1
    rows = [json.loads(l) for l in log.open() if l.strip()]
    if not rows:
        print("日志为空")
        return 1
    msgs = [r.get("msg", "") for r in rows]
    ts = [r["t"] for r in rows]

    tele = [(t, m) for t, m in zip(ts, msgs) if "telemetry" in m]
    P = [x for x in tele if "phase=proposer" in x[1]]
    S = [x for x in tele if "phase=solver" in x[1]]
    print(f"=== 进度 ===")
    print(f"  proposer 步 = {len(P)}   solver 步 = {len(S)}   总遥测 = {len(tele)}")
    if tele:
        print(f"  墙钟 = {(ts[-1] - ts[0]) / 60:.1f} min")
        d = [tele[i][0] - tele[i - 1][0] for i in range(1, len(tele))]
        if d:
            print(f"  步间隔 均值 {sum(d)/len(d):.1f}s  中位 {sorted(d)[len(d)//2]:.1f}s  最新 {d[-1]:.1f}s")
        m = tele[-1][1]
        for k in ("phase", "outer", "step", "rollout_s", "backward_s",
                  "generated_tokens_global", "cuda_peak_allocated_gib"):
            mm = re.search(k + r"=([0-9.]+)", m)
            if mm:
                print(f"  最新 {k} = {mm.group(1)}")

    print("=== proposer 相位 ===")
    acc1 = sum(1 for m in msgs if re.search(r"accepted", m) and "/2 accepted" in m
               and not m.strip().startswith("["))
    accs = [int(x) for x in re.findall(r"\[proposer\] (\d+)/2 accepted", "\n".join(msgs))]
    if accs:
        print(f"  探索调用 {len(accs)} 次：接受 0 条 {accs.count(0)} 次，"
              f"接受 1 条 {accs.count(1)} 次，接受 2 条 {accs.count(2)} 次")
    turns = [float(x) for x in re.findall(r"mean_turns=([0-9.]+)", "\n".join(msgs))]
    if turns:
        print(f"  探索平均回合数 = {sum(turns)/len(turns):.1f}")
    pools = [int(x) for x in re.findall(r"pool=(\d+)", "\n".join(msgs))]
    if pools:
        print(f"  任务池 = {pools[0]} -> {pools[-1]}")

    print("=== solver 相位 ===")
    srows = []
    for m in msgs:
        mm = re.search(r"\[outer (\d+)\]\[S(\d+)\].*reward mean=([0-9.]+).*final_match=(\d+)/(\d+)", m)
        if mm:
            srows.append((int(mm.group(1)), int(mm.group(2)), float(mm.group(3)),
                          int(mm.group(4)), int(mm.group(5))))
    if srows:
        print(f"  共 {len(srows)} 步；最后一次: outer{srows[-1][0]} S{srows[-1][1]} "
              f"reward={srows[-1][2]:.3f} match={srows[-1][3]}/{srows[-1][4]}")
        k = min(5, len(srows))
        print(f"  前 {k} 步 reward: {[round(r[2],3) for r in srows[:k]]}")
        print(f"  后 {k} 步 reward: {[round(r[2],3) for r in srows[-k:]]}")
        print(f"  前 {k} 步 match : {[f'{r[3]}/{r[4]}' for r in srows[:k]]}")
        print(f"  后 {k} 步 match : {[f'{r[3]}/{r[4]}' for r in srows[-k:]]}")

    print("=== solve rate（proposer 奖励估计）===")
    sr = [float(x) for x in re.findall(r"solve_rate",
                                       "\n".join(msgs))]
    srl = re.findall(r"final_match=(\d+)/(\d+).*?subtask_rate", "\n".join(msgs))
    if srl:
        vals = [int(a) / int(b) for a, b in srl if int(b)]
        if vals:
            print(f"  solve rate 采样 {len(vals)} 组：均值 {sum(vals)/len(vals):.3f} "
                  f"（全 0 的比例 {sum(1 for v in vals if v == 0)/len(vals):.2f}）")

    print("=== 评测 ===")
    ev = re.findall(r"\[eval\] outer (\d+) 完成 valid_unseen SR=([0-9.]+) \((\d+)/(\d+)", "\n".join(msgs))
    print("  ", ev if ev else "（还没有完成的评测）")
    bad = re.findall(r"\[eval\]\[FATAL-BUT-IGNORED\] outer (\d+)([^|]{0,120})", "\n".join(msgs))
    if bad:
        print("   ⚠️ 被吞掉的评测失败:", bad[:3])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
