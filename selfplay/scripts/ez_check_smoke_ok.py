#!/usr/bin/env python
"""EZ 冒烟产物自检（EZ 版准入门）。

读最新 run 目录里的 `train_log.jsonl`，回答三个问题：

  关1  facts 通道在不在？（EZ 的命根子；不在 = 奖励是废的）
  关2  参数真的动了吗？（update_rel，证明 no_grad 没把梯度链切断）
  关3  奖励退化了吗？（proposer 全被拒 / solver 全 0 分 = 学不了）

判读逻辑（重要，别把两种情况混为一谈）：

  update_rel == 0 **且** 奖励有区分度   → 【致命】梯度链断了（no_grad/断图 bug），必修
  update_rel == 0 **且** 奖励全同分     → 【警告】没有学习信号，不是 bug。
                                        GRPO 组内优势全 0 时梯度天然为 0。
                                        冒烟缩到 turns=3 时常发生，加大 turns 再看。
  update_rel  > 0                      → 梯度链是通的（这一关就过了）

用法（集群上，紧跟 sbatch 之后手动跑也行）：
    python scripts/ez_check_smoke_ok.py --config configs/ez_qwen3_4b.yaml
    python scripts/ez_check_smoke_ok.py --config configs/ez_qwen3_4b.yaml --run-dir <指定目录>
    python scripts/ez_check_smoke_ok.py --config ... --strict   # 警告也当失败
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

UPDATE_RE = re.compile(r"\[(?P<tag>[^\]]+)\]\s+update_rel=(?P<val>[0-9.eE+-]+)")
ACCEPT_RE = re.compile(r"\| accept=(?P<acc>\d+)/(?P<tot>\d+)")
REWARD_RE = re.compile(r"reward mean=(?P<mean>-?[0-9.]+).*?final_match=(?P<fm>\d+)/(?P<n>\d+)")
FACTS_RE = re.compile(r"facts 通道 OK（(?P<n>\d+) 条事实）")

FATAL, WARN, OK = "FATAL", "WARN", "OK"


def load_yaml(path: Path) -> dict:
    try:
        import yaml  # noqa: PLC0415
    except ImportError:
        print("[FATAL] 需要 pyyaml 来解析 config（pip install pyyaml）")
        raise SystemExit(1)
    with open(path) as fh:
        return yaml.safe_load(fh)


def find_run_dir(cfg: dict, cfg_path: Path) -> Path:
    """output_dir/<name>-MMDD-HHMM 里取 mtime 最新的那个。"""
    run = cfg["run"]
    out = Path(run["output_dir"])
    if not out.is_absolute():
        out = (cfg_path.resolve().parent.parent / out).resolve()
    cands = sorted([p for p in out.glob(f"{run['name']}-*") if p.is_dir()],
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        print(f"[FATAL] 在 {out} 下找不到 {run['name']}-* 运行目录")
        raise SystemExit(1)
    return cands[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", default=None, help="不指定则取 output_dir 下最新的")
    ap.add_argument("--strict", action="store_true", help="WARN 也当失败")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)
    run_dir = Path(args.run_dir) if args.run_dir else find_run_dir(cfg, cfg_path)
    log_path = run_dir / "train_log.jsonl"
    print(f"[info] run_dir = {run_dir}")
    if not log_path.exists():
        print(f"[FATAL] 找不到 {log_path} —— 训练可能第一行就崩了，看 .err")
        return 1

    msgs = []
    with open(log_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                msgs.append(json.loads(line)["msg"])
            except (json.JSONDecodeError, KeyError):
                msgs.append(line)

    findings: list[tuple[str, str]] = []

    # ---- 关1：facts 通道 ----
    facts_n = 0
    for m in msgs:
        if (mt := FACTS_RE.search(m)):
            facts_n = max(facts_n, int(mt.group("n")))
    if facts_n > 0:
        findings.append((OK, f"关1 facts 通道存在（{facts_n} 条事实）"))
    else:
        findings.append((FATAL, "关1 日志里没有 'facts 通道 OK' —— "
                                "不是没跑到自检（看 .out），就是 facts 是空的"))

    # ---- 关2：update_rel ----
    ups = [(m.group("tag"), float(m.group("val")))
           for m in (UPDATE_RE.search(x) for x in msgs) if m]
    if not ups:
        findings.append((FATAL, "关2 一条 update_rel 都没有 —— 训练没走到更新那一步"))
    else:
        worst_tag, worst = min(ups, key=lambda kv: kv[1])
        tags = ", ".join(f"{t}={v:.2e}" for t, v in ups)
        findings.append((OK, f"关2 update_rel 共 {len(ups)} 处：{tags}"))

    # ---- 关3：奖励区分度 ----
    accs = [(int(m.group("acc")), int(m.group("tot")))
            for m in (ACCEPT_RE.search(x) for x in msgs) if m]
    rw = [(float(m.group("mean")), int(m.group("fm")), int(m.group("n")))
          for m in (REWARD_RE.search(x) for x in msgs) if m]
    if not accs:
        findings.append((WARN, "关3 没有 proposer accept 记录（可能本 outer 没走到）"))
    elif max(a for a, _ in accs) == 0:
        findings.append((FATAL, f"关3 proposer 全部被拒（{accs}）—— 提案器一条有效任务都没产出"))
    else:
        findings.append((OK, f"关3 proposer 有有效提案：{accs}"))
    reward_degenerate = True
    if rw:
        best = max(m for m, _, _ in rw)
        reward_degenerate = best <= 0.0
        findings.append((OK if best > 0 else WARN,
                         f"关3 solver reward mean={[f'{m:.3f}' for m, _, _ in rw]}"
                         + ("（全 0：组内优势全 0，梯度天然为 0 —— 加大 turns 再看）"
                            if reward_degenerate else "")))
    else:
        findings.append((WARN, "关3 没有 solver reward 记录"))

    # ---- 综合判定 ----
    print("\n===== EZ 冒烟自检 =====")
    for lvl, text in findings:
        print(f"  [{lvl:5s}] {text}")

    fatal = [t for l, t in findings if l == FATAL]
    if fatal:
        print("\n结论：**未通过** —— 见上面 FATAL 项")
        return 1

    zero_up = bool(ups) and min(v for _, v in ups) <= 1e-10
    if zero_up:
        if reward_degenerate:
            print("\n结论：条件通过（梯度链通，但本 outer 无学习信号）。"
                  "\n      → 这是冒烟缩规模 G=2/turns=3 的正常现象，不是 bug。"
                  "\n      → 开正式作业前建议用 EZ_SMOKE_G=4,EZ_SMOKE_TURNS=5 再确认一次。")
            return 0 if not args.strict else 1
        print("\n结论：**未通过** —— 奖励有区分度但 update_rel=0，梯度链是断的"
              "（检查 no_grad / 断图 / lr=0）")
        return 1

    warns = [t for l, t in findings if l == WARN]
    if warns and args.strict:
        print("\n结论：**未通过**（--strict：WARN 也算失败）")
        return 1
    print("\n结论：**通过** ✅ 三关全过，可以开 8 卡正式作业了")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
