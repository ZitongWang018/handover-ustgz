#!/usr/bin/env python
"""训练运行监控器：读 jsonl/history + 扫日志，算各项指标与**异常率**，并给出归因。

设计目的
--------
训练"看起来在跑"不等于"在学"。GRPO 有几种典型的**静默失败**：
    * 组内全同分 -> advantage 全 0 -> loss=0 -> 一步都没更新（日志还是好好的）
    * 模型吐不出合法动作 -> invalid_rate=1.0 -> 奖励恒等于"全程非法"那一档
    * 生成被 gradient_checkpointing 搞坏 -> 输出乱码 -> 上面两条一起出现
这些都不会抛异常，只会让 300 iter 白跑。所以监控必须**主动算异常率**。

异常定义（每条都会给出中文原因 + 排查方向）
--------------------------------------
    A1 invalid_rate > 0.90            模型没吐出可解析动作
    A2 group_degenerate_frac >= 1.0   组内全同分（g 小时可能是样本不足）
    A3 adv_abs_mean < 1e-8            完全没有梯度信号（GRPO 空转）
    A4 loss == 0 且 adv != 0          loss 计算有问题（不该发生）
    A5 reward_max == reward_mean      奖励无区分度
    A6 mean_steps 顶满 max_steps      每局都跑满步数（没提前成功/终止）
    A7 eval/success_rate == 0 且 n>=8 评测全挂

用法
----
    python scripts/monitor_runs.py --config configs/train_qwen3_4b.yaml
    python scripts/monitor_runs.py --jsonl runs/xxx/train_rank0.jsonl --max-steps 50
    python scripts/monitor_runs.py --config configs/train_qwen3_4b.yaml --watch 60   # 每 60s 刷新
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "src"))

# ---------- 异常阈值 ----------
INVALID_RATE_HI = 0.90
ADV_EPS = 1e-8


def _f(d: Dict[str, Any], k: str, default: float = 0.0) -> float:
    v = d.get(k)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_rows(src: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not src.exists():
        return rows
    for line in src.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def pick_jsonl(cfg_path: Optional[Path], jsonl: Optional[Path], rank: int) -> Path:
    if jsonl:
        return jsonl.expanduser().resolve()
    if cfg_path:
        from train_grpo import load_cfg  # 延迟导入
        cfg = load_cfg(str(cfg_path))
        return (cfg_path.resolve().parent.parent / cfg["run"]["output_dir"]
                / cfg["run"]["name"] / f"train_rank{rank}.jsonl").resolve()
    raise SystemExit("必须给 --config 或 --jsonl")


def check_row(r: Dict[str, Any], max_steps: int) -> List[Tuple[str, str, str]]:
    """返回 [(编码, 现象, 排查方向)]。"""
    out: List[Tuple[str, str, str]] = []
    inv = _f(r, "invalid_rate", -1.0)
    if inv >= 0 and inv > INVALID_RATE_HI:
        out.append(("A1", f"非法动作率 {inv:.1%}",
                    "模型没吐出可解析动作 → 查 prompt/解析/max_new_tokens，"
                    "以及生成是否被 gradient_checkpointing 搞坏"))
    if _f(r, "group_degenerate_frac") >= 1.0:
        out.append(("A2", "组内全同分（每个 group std=0）",
                    "g 很小时可能只是样本不足；配合 adv=0 就一定是奖励没区分度"))
    if _f(r, "adv_abs_mean") < ADV_EPS:
        out.append(("A3", "advantage 全为 0",
                    "GRPO 空转，这一步完全没有梯度 → 查 group_size 与奖励区分度"))
    if abs(_f(r, "loss")) < 1e-12 and _f(r, "adv_abs_mean") >= ADV_EPS:
        out.append(("A4", "adv 非 0 但 loss=0", "loss 计算有问题，查 grpo_loss / mask"))
    rm, rx = _f(r, "reward_mean"), _f(r, "reward_max")
    if abs(rx - rm) < 1e-9:
        out.append(("A5", f"奖励无区分度（mean=max={rm:.4f}）",
                    "奖励被 clip 或过程项没归一化 → 查 reward.py 与 clip_min/max"))
    ms = _f(r, "mean_steps", -1.0)
    if max_steps > 0 and ms >= max_steps - 1e-6:
        out.append(("A6", f"每局都跑满 {max_steps} 步", "没有提前成功或终止，任务大概率没解出来"))
    if "eval/success_rate" in r and _f(r, "eval/n") >= 8 and _f(r, "eval/success_rate") <= 0:
        out.append(("A7", "评测成功率为 0", "先看 A1/A6；都不是就是策略本身太弱"))
    # A8：有梯度但参数没动 —— 纯 bf16 + 小 lr 时更新量小于 ulp，被舍入抹掉
    if "update_rel" in r and _f(r, "update_rel") < 1e-9 and _f(r, "adv_abs_mean") >= ADV_EPS:
        out.append(("A8", "参数相对更新量≈0（有梯度但参数没动）",
                    "bf16 参数下 AdamW 更新被 ulp 舍入抹掉 → dtype 改 float32 + bf16 autocast"))
    # A9：奖励符号反了 —— 非法动作反而拿高分，其它指标全部正常
    if _f(r, "invalid_reward_corr") > 0.3:
        out.append(("A9", f"非法动作越多奖励越高 (corr={_f(r, 'invalid_reward_corr'):+.2f})",
                    "penalty 项符号写反（写了 r += 而不是 r -=）→ 查 reward.py 的 compute_reward"))
    return out


def scan_log(path: Path, n: int = 12) -> List[str]:
    """扫日志里的硬错误行。"""
    if not path.exists():
        return []
    pat = re.compile(r"(Traceback|Error|error:|FAILED|CUDA out of memory|OutOfMemory|"
                     r"RuntimeError|ValueError|AssertionError|NCCL|core dumped)",
                     re.IGNORECASE)
    hits: List[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if pat.search(line):
            hits.append(line.strip()[:160])
    return hits[-n:]


def job_state(user: str = "xyu608") -> str:
    try:
        p = subprocess.run(["squeue", "-u", user, "-o", "%.10i %.14j %.14P %.9T %.18S"],
                           capture_output=True, text=True, timeout=20)
        return p.stdout.strip() or "(squeue 无输出 / 未安装 slurm)"
    except Exception as e:  # noqa: BLE001
        return f"(查不了作业状态: {type(e).__name__})"


def render(rows: List[Dict[str, Any]], max_steps: int, tail: int, title: str) -> Dict[str, Any]:
    rows = rows[-tail:]
    if not rows:
        print("（还没有任何训练日志行）")
        return {"n": 0}

    print("=" * 100)
    print(title)
    print("=" * 100)
    print(f"{'iter':<6}{'reward':<9}{'std':<8}{'退化':<7}{'adv':<9}{'loss':<10}"
          f"{'步数':<6}{'非法率':<8}{'更新量':<10}{'奖励相关性':<11}{'异常':<12}")
    print("-" * 110)
    per_code: Dict[str, int] = {}
    n_bad_rows = 0
    for r in rows:
        codes = check_row(r, max_steps)
        if codes:
            n_bad_rows += 1
        for c, _, _ in codes:
            per_code[c] = per_code[c] + 1
        mark = ",".join(sorted({c for c, _, _ in codes})) or "-"
        print(f"{int(_f(r, 'iter')):<6}{_f(r, 'reward_mean'):<9.4f}{_f(r, 'group_std_mean'):<8.4f}"
              f"{_f(r, 'group_degenerate_frac'):<7.2f}{_f(r, 'adv_abs_mean'):<9.5f}"
              f"{_f(r, 'loss'):<10.6f}{_f(r, 'mean_steps', -1):<6.1f}"
              f"{_f(r, 'invalid_rate', -1):<8.3f}{_f(r, 'update_rel', -1):<10.2e}"
              f"{_f(r, 'invalid_reward_corr', 0):<+11.2f}{mark:<12}")
    print("-" * 110)

    n = len(rows)
    rate = n_bad_rows / n
    print(f"\n【异常率】{n_bad_rows}/{n} = {rate:.1%} 的 iter 命中至少一条异常")
    if per_code:
        print("按编码统计（命中 iter 数 / 占比）：")
        for c in sorted(per_code):
            print(f"  {c}: {per_code[c]}/{n} = {per_code[c] / n:.1%}")
    else:
        print("  未发现异常 ✅")

    # 聚合 + 归因
    print("\n【异常归因与排查方向】")
    seen: Dict[str, Tuple[str, str]] = {}
    for r in rows:
        for c, phen, fix in check_row(r, max_steps):
            seen.setdefault(c, (phen, fix))
    if not seen:
        print("  无")
    for c in sorted(seen):
        phen, fix = seen[c]
        print(f"  [{c}] {phen}")
        print(f"        → {fix}")

    # 汇总量
    print("\n【汇总】")
    for label, key in (("reward_mean", "reward_mean"), ("group_std_mean", "group_std_mean"),
                       ("adv_abs_mean", "adv_abs_mean"), ("loss", "loss"),
                       ("mean_steps", "mean_steps"), ("invalid_rate", "invalid_rate")):
        vals = [_f(r, key, float("nan")) for r in rows if key in r]
        vals = [v for v in vals if v == v]  # 去 nan
        if vals:
            print(f"  {label:<16} 均值={sum(vals) / len(vals):.6f}  "
                  f"最小={min(vals):.6f}  最大={max(vals):.6f}")
    for key in ("eval/success_rate", "eval/invalid_action_rate", "eval/mean_steps"):
        vals = [_f(r, key) for r in rows if key in r]
        if vals:
            print(f"  {key:<16} 最近={vals[-1]:.4f}")
    return {"n": n, "anomaly_rate": rate, "codes": per_code}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--jsonl")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=30, help="用于 A6（跑满步数）判定")
    ap.add_argument("--tail", type=int, default=30, help="只看最近 N 行")
    ap.add_argument("--log", nargs="*", default=None, help="额外要扫的日志文件")
    ap.add_argument("--watch", type=int, default=0, help=">0 则每 N 秒刷新一次")
    ap.add_argument("--no-job", action="store_true")
    args = ap.parse_args()

    src = pick_jsonl(Path(args.config) if args.config else None,
                     Path(args.jsonl) if args.jsonl else None, args.rank)
    print(f"[monitor] 数据源: {src}")

    while True:
        rows = load_rows(src)
        title = f"训练监控 — {src.parent.name}  (共 {len(rows)} 行，显示最近 {min(args.tail, len(rows))} 行)  {time.strftime('%H:%M:%S')}"
        render(rows, args.max_steps, args.tail, title)

        if not args.no_job:
            print("\n【作业队列】")
            print(job_state())

        logs = args.log
        if logs is None:
            logs = sorted((PROJ / "logs").glob("*.out"), key=lambda p: p.stat().st_mtime,
                          reverse=True)[:2]
        print("\n【日志硬错误扫描】")
        any_hit = False
        for lp in logs:
            hits = scan_log(Path(lp))
            if hits:
                any_hit = True
                print(f"  {lp.name}:")
                for h in hits:
                    print(f"    {h}")
        if not any_hit:
            print("  未发现 Error/Traceback/OOM 等硬错误 ✅")

        if args.watch <= 0:
            break
        print(f"\n--- {args.watch}s 后刷新 (Ctrl-C 退出) ---")
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
