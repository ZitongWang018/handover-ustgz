#!/usr/bin/env python
"""把 eval_alfworld_prompting.py 的产出换算成**论文口径**（Table 1）。

为什么必须单独一步（2026-10-14）：
    论文 Table 1 报的是 **6 类任务的 macro 平均**（列 Pick/Look/Clean/Heat/Cool/Pick2，
    最后是 "Avg."），**不是 episode 加权**。两者在类别不均衡时差别很大，直接拿
    episode 加权去对表会得出错误结论。

用法：python skillrise/summarize.py skillrise/out/react_eval_in_distribution_*.json
"""
import json
import sys
from collections import Counter
from pathlib import Path

# 论文 Table 1 的列名 → 我们的 task_type 前缀
PAPER_COLS = [("Pick", "pick_and_place_simple"),
              ("Look", "look_at_obj_in_light"),
              ("Clean", "pick_clean_then_place_in_recep"),
              ("Heat", "pick_heat_then_place_in_recep"),
              ("Cool", "pick_cool_then_place_in_recep"),
              ("Pick2", "pick_two_obj_and_place")]


def summarize(path):
    d = json.load(open(path, encoding="utf-8"))
    rows = d["rows"]
    per = Counter()
    for r in rows:
        t = Path(str(r["game"]).replace("\\", "/")).parent.parent.name.split("-")[0]
        per[(t, "n")] += 1
        if r["won"]:
            per[(t, "w")] += 1
    n = len(rows)
    won = sum(1 for r in rows if r["won"])
    macro_vals, missing = [], []
    print(f"=== {Path(path).name}")
    print(f"    style={d['summary']['style']} split={d['summary']['split']} "
          f"episodes={n} won={won} episode_weighted_SR={won/max(1,n)*100:.1f}%")
    print(f"    {'col':6s} {'task_type':32s} {'n':>4s} {'won':>4s} {'SR%':>7s}")
    for col, t in PAPER_COLS:
        nn, ww = per[(t, "n")], per[(t, "w")]
        if nn:
            sr = ww / nn * 100
            macro_vals.append(sr)
            print(f"    {col:6s} {t:32s} {nn:4d} {ww:4d} {sr:7.1f}")
        else:
            missing.append(col)
    print(f"    {'Avg.':6s} {'(6-type macro)':32s} {'':4s} {'':4s} "
          f"{(sum(macro_vals)/len(macro_vals) if macro_vals else 0):7.1f}")
    if missing:
        print(f"    [警告] 缺列: {missing}")
    return {"episodes": n, "won": won, "episode_weighted": won / max(1, n),
            "macro": sum(macro_vals) / len(macro_vals) if macro_vals else 0.0}


def per_seed(path):
    """按 seed 分组算 6 类 macro，并给 mean ± std（多种子实验的主指标）。"""
    import statistics
    d = json.load(open(path, encoding="utf-8"))
    rows = d["rows"]
    if "seed" not in (rows[0] if rows else {}):
        return None
    by = {}
    for r in rows:
        by.setdefault(r["seed"], []).append(r)
    macros, lines = [], []
    for sd in sorted(by):
        per = Counter()
        for r in by[sd]:
            tp = Path(str(r["game"]).replace("\\", "/")).parent.parent.name.split("-")[0]
            per[(tp, "n")] += 1
            if r["won"]:
                per[(tp, "w")] += 1
        vals = [per[(tp, "w")] / per[(tp, "n")] * 100 for _, tp in PAPER_COLS if per[(tp, "n")]]
        m = sum(vals) / len(vals) if vals else 0.0
        macros.append(m)
        lines.append(f"      seed={sd:<3d} macro={m:6.1f}  episodes={len(by[sd])}")
    print("    [逐 seed]")
    print("\n".join(lines))
    mean = statistics.mean(macros)
    std = statistics.stdev(macros) if len(macros) > 1 else 0.0
    print(f"    [多种子主指标] macro = {mean:.1f} ± {std:.1f}  (n_seeds={len(macros)})")
    return mean, std


if __name__ == "__main__":
    out = [summarize(p) for p in sys.argv[1:]]
    for p in sys.argv[1:]:
        per_seed(p)
    if len(out) > 1:
        print("\n=== 论文对照（Table 1, Qwen3-4B, ALFWorld）===")
        print("    Zero-shot 4.4 avg | ReAct 25.0 avg  （Pass@1 %, 6-type macro）")
