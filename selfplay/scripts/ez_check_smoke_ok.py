#!/usr/bin/env python
"""EZ 冒烟产物自检（EZ 版准入门）。

读最新 run 目录里的 `train_log.jsonl`（以及多卡时的 `train_log.rank*.jsonl`），
回答五个问题：

  关0  两个 adapter 各自都能接到梯度吗？（`grad_chain_selfcheck`，
       前向→LoRA→loss→反向，**不依赖环境/奖励**，是"链路通不通"的权威判据）
  关1  facts 通道在不在？（EZ 的命根子；不在 = 奖励是废的）
  关2  参数真的动了吗？（update_rel）
  关2b 优化器实际更新了几个张量？（比 update_rel 更直接的判据，专治 probe 挑错张量）
  关3  proposer 有没有产出有效任务、solver 奖励有没有区分度？
  关4  （多卡才有）每个 rank 的日志都在吗？update_rel 在 rank 之间同量级吗？

判读逻辑（★ 关键：这四种现象长得像，但处理方法完全不同）

  update_rel=0 且 该组奖励有区分度(rew_spread>0)   → 【致命】梯度链断了。必修。
  update_rel=0 且 该组奖励全同分(rew_spread=0)     → 【正常】GRPO 组内优势全 0，
                                                    梯度天然为 0，不是 bug。
  proposer accept=0/0                              → 【致命】提案器一条任务都没产出。
  solver reward 全 0                               → 【警告】没学习信号（可能只是模型弱）。

★ 为什么必须看**组奖励的 spread** 而不是笼统的 "reward mean"：
  之前的版本用 solver 的 `reward mean` 当"奖励区分度"信号，于是
  `accept=1/2`（proposer 组内明明有区分度）+ `update_rel=0` 被误判成"正常无信号"，
  恰好把**梯度链断裂**这个最该报错的场景漏掉了（2026-10-12 job 12989143 实测）。

★ 为什么还要关2b（张量计数）：
  旧 probe 只测"第一个 numel>=4096 的 LoRA 张量"，按 named_parameters 顺序恰好命中
  `lora_A` —— peft 把 lora_B 初始化成全 0 ⇒ ∂L/∂A ≡ 0 ⇒ 该张量第 0 步梯度**精确为 0**。
  于是日志同时出现「grad_norm=6.15 非零」和「update_rel=0.000e+00」，看着像梯度链断了，
  其实 probe 挑了个天生零梯度的张量（2026-10-12 job 12989450 定位）。

用法（集群上，紧跟 sbatch 之后手动跑也行）：
    python scripts/ez_check_smoke_ok.py --config configs/ez_qwen3_4b.yaml
    python scripts/ez_check_smoke_ok.py --config configs/ez_qwen3_4b.yaml --run-dir <指定目录>
    python scripts/ez_check_smoke_ok.py --config ... --strict   # 警告也当失败
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

FATAL, WARN, OK = "FATAL", "WARN", "OK"

# [outer 2 P0] update_rel=1.234e-03（...）
UPD_RE = re.compile(r"\[(outer (?P<outer>\d+) (?P<phase>[PS])(?P<step>\d+))\]\s+"
                    r"update_rel=(?P<val>[0-9.eE+-]+)")
# [outer 2 P0] update_rel=1.0e+00（=max ‖Δp‖/(‖p‖+‖Δp‖)；252/504 张量真的动了；max‖Δp‖=1.0e-04）
UPD_MOVED_RE = re.compile(r"\[(outer (?P<outer>\d+) (?P<phase>[PS])(?P<step>\d+))\]\s+"
                          r"update_rel=.*?；\s*(?P<moved>\d+)/(?P<tot>\d+)\s+张量真的动了")
# [outer 2][P0] pool=1 (+1) | accept=1/2 | rew=[0.3, 0.0] | rew_spread=0.300 | 112s
PPLINE_RE = re.compile(r"\[outer (?P<outer>\d+)\]\[P(?P<step>\d+)\].*?"
                       r"accept=(?P<acc>\d+)/(?P<tot>\d+).*?"
                       r"rew=\[(?P<rews>[^\]]*)\].*?rew_spread=(?P<spread>[0-9.]+)")
SREWARD_RE = re.compile(r"reward mean=(?P<mean>-?[0-9.]+).*?final_match=(?P<fm>\d+)/(?P<n>\d+)")
FACTS_RE = re.compile(r"facts 通道 OK（(?P<n>\d+) 条事实）")
GRADCHAIN_RE = re.compile(r"梯度链 proposer grad_norm=(?P<p>[^\s]+).*?solver grad_norm=(?P<s>[^\s]+)")


def _fnum(s):
    try:
        return float(s)
    except ValueError:
        return float("nan")


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
    # yaml 只在"需要自己找 run 目录"时才必需：显式 --run-dir 时不必装 pyyaml
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = find_run_dir(load_yaml(cfg_path), cfg_path)
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

    # ---- 关0：梯度链自检（最权威的一条）----
    gc = None
    for m in msgs:
        if (mt := GRADCHAIN_RE.search(m)):
            gc = (_fnum(mt.group("p")), _fnum(mt.group("s")))
    if gc is None:
        findings.append((WARN, "关0 日志里没有梯度链自检 —— 是不是跑的是旧代码？"))
    elif not (gc[0] > 0) or not (gc[1] > 0):
        findings.append((FATAL, f"关0 梯度链断：proposer grad_norm={gc[0]}, "
                                f"solver grad_norm={gc[1]} —— 某个 adapter 收不到梯度"))
    else:
        findings.append((OK, f"关0 梯度链通：proposer grad_norm={gc[0]:.3e}, "
                             f"solver grad_norm={gc[1]:.3e}"))

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

    # ---- 关2 + 关3（proposer 相位）：update_rel 必须对着**本组奖励 spread** 判 ----
    updates: dict[str, float] = {}
    for m in (UPD_RE.search(x) for x in msgs):
        if m:
            updates[m.group(1)] = _fnum(m.group("val"))
    p_lines: dict[tuple[int, int], tuple[int, int, float]] = {}
    for m in (PPLINE_RE.search(x) for x in msgs):
        if m:
            p_lines[(int(m.group("outer")), int(m.group("step")))] = (
                int(m.group("acc")), int(m.group("tot")), _fnum(m.group("spread")))

    if not updates:
        findings.append((FATAL, "关2 一条 update_rel 都没有 —— 训练没走到更新那一步"))
    else:
        tags = ", ".join(f"{t}={v:.2e}" for t, v in updates.items())
        findings.append((OK, f"关2 update_rel 共 {len(updates)} 处：{tags}"))

    # ---- 关2b：优化器**实际更新了几个张量**（比 update_rel 更直接的判据）----
    #   ★ 实测教训（job 12989450）：旧 probe 只测"第一个 numel>=4096 的 LoRA 张量"，
    #     正好命中 lora_A —— peft 把 lora_B 初始化成全 0 ⇒ ∂L/∂A ≡ 0 ⇒
    #     该张量第 0 步梯度精确为 0 ⇒ update_rel 打印 0.000e+00。
    #     于是「grad_norm=6.15 非零」与「update_rel=0」同时成立，看起来像自相矛盾。
    #     现在 probe 测全部带 grad 的张量并报 N/M，这条判据直接把假警报钉死。
    moved_map: dict[str, tuple[int, int]] = {}
    for m in (UPD_MOVED_RE.search(x) for x in msgs):
        if m:
            moved_map[m.group(1)] = (int(m.group("moved")), int(m.group("tot")))
    if not moved_map:
        findings.append((WARN, "关2b 日志里没有 '张量真的动了' —— 跑的是旧代码？（probe 未升级）"))
    else:
        _b = max(a for a, _ in moved_map.values())
        findings.append((OK if _b > 0 else FATAL,
                         f"关2b 优化器实际更新张量数："
                         + ", ".join(f"{k}={a}/{t}" for k, (a, t) in moved_map.items())
                         + ("（全 0 —— 优化器真的一步没动）" if _b == 0 else "")))

    # 断链判定：P 相位 update_rel≈0 但该组奖励有区分度
    broken = []
    for (outer, step), (acc, tot, spread) in p_lines.items():
        key = f"outer {outer} P{step}"
        rel = updates.get(key)
        if rel is not None and rel <= 1e-10 and spread > 1e-9:
            broken.append((key, acc, tot, spread))
    if broken:
        findings.append((FATAL, "关2 **梯度链断**：以下 P 相位组内奖励有区分度"
                                "（本该产生非零 advantage）但参数一动没动 —— "
                                + "; ".join(f"{k}(accept={a}/{t}, spread={s:.3f})"
                                            for k, a, t, s in broken)))
    elif p_lines:
        no_sig = [f"{o}P{st}(spread={s:.3f})" for (o, st), (_, _, s) in p_lines.items()
                  if s <= 1e-9]
        if no_sig:
            findings.append((WARN, f"关2 这些 P 相位组内奖励全同分（无信号，属正常）：{no_sig}"))

    if not p_lines:
        findings.append((WARN, "关3 没有 proposer 相位记录（可能本 outer 没走到）"))
    else:
        accs = [(a, t) for a, t, _ in p_lines.values()]
        if max(a for a, _ in accs) == 0:
            findings.append((FATAL, f"关3 proposer 全部被拒（{accs}）—— 提案器一条有效任务都没产出。"
                                    f"\n          先看 fmt_bad：>0 是解析/格式问题；=0 但仍 accept=0，"
                                    f"通常是**探索轮数不够**（prompt 要求探索 10-20 个动作后才 "
                                    f"finish，EZ_SMOKE_TURNS 太小就永远走不到 finish）"))
        else:
            findings.append((OK, f"关3 proposer 有有效提案：{accs}"))

    # ---- 关3（solver 相位）----
    rw = [(float(m.group("mean")), int(m.group("fm")), int(m.group("n")))
          for m in (SREWARD_RE.search(x) for x in msgs) if m]
    if rw:
        best = max(m for m, _, _ in rw)
        findings.append((OK if best > 0 else WARN,
                         f"关3 solver reward mean={[f'{m:.3f}' for m, _, _ in rw]}"
                         + ("（全 0：solver 一次都没完成任务，组内优势全 0）"
                            if best <= 0 else "")))
    else:
        findings.append((WARN, "关3 没有 solver reward 记录（本 outer 任务池为空？）"))

    # ---- 关4（仅多卡）：8 个 rank 的日志都在？update_rel 同量级？----
    #   ★ 这一关只在跑过 torchrun 8 进程时才有意义。判据来自 EZ_DDP_PLAN.md §8：
    #     ① 每 rank 的 train_log.rank*.jsonl 存在且非空（缺文件 = 那个 rank 提早挂了）
    #     ② update_rel 在 8 个 rank 之间同量级（差 100× 说明某个 rank 没参与梯度同步）
    #     ③ 日志里的 world_size 真的是 8（torchrun 没生效时**不报错**，只是 8 个进程
    #        各自在单卡上跑 —— 速度没变快，日志却多了 8 倍）
    rank_files = sorted(run_dir.glob("train_log.rank*.jsonl"))
    if rank_files:
        per_rank: dict[int, list[float]] = {}
        worlds: set[int] = set()
        empty: list[str] = []
        for rf in rank_files:
            rk = int(re.search(r"rank(\d+)", rf.name).group(1))
            vals: list[float] = []
            n_lines = 0
            with open(rf) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    n_lines += 1
                    try:
                        m_ = json.loads(line)["msg"]
                    except (json.JSONDecodeError, KeyError):
                        continue
                    if (mw := re.search(r"\[dist\] rank=\d+/(\d+)", m_)):
                        worlds.add(int(mw.group(1)))
                    if (mv := UPD_RE.search(m_)) and mv.group("phase") == "P":
                        vals.append(_fnum(mv.group("val")))
            if n_lines == 0:
                empty.append(rf.name)
            per_rank[rk] = vals
        if empty:
            findings.append((FATAL, f"关4 这些 rank 日志是空的（该 rank 提早挂了）：{empty}"))
        else:
            findings.append((OK, f"关4 {len(rank_files)} 个 rank 日志都存在且非空"))
        if worlds:
            w_max = max(worlds)
            findings.append((OK if w_max > 1 else WARN,
                             f"关4 日志里的 world_size={sorted(worlds)}"
                             + ("" if w_max > 1 else "（=1！torchrun 没生效，8 个进程各跑各的）")))
        peaks = {r: max(v) for r, v in per_rank.items() if v}
        if len(peaks) >= 2:
            lo, hi = min(peaks.values()), max(peaks.values())
            ratio = (hi / lo) if lo > 0 else float("inf")
            findings.append((OK if (lo > 0 and ratio < 100) else FATAL,
                             f"关4 各 rank 的 max(update_rel)="
                             + ", ".join(f"r{r}={v:.2e}" for r, v in sorted(peaks.items()))
                             + f"（最大/最小 = {ratio:.1f}；≥100× 或某 rank 为 0 "
                               f"说明它没参与梯度同步）"))

    # ---- 综合判定 ----
    print("\n===== EZ 冒烟自检 =====")
    for lvl, text in findings:
        print(f"  [{lvl:5s}] {text}")

    if any(l == FATAL for l, _ in findings):
        print("\n结论：**未通过** —— 见上面 FATAL 项")
        return 1

    # 只有"参数真的动过"才算硬通过：update_rel>1e-10 **且** 至少动了一个张量
    moved = [t for t, v in updates.items() if v > 1e-10]
    _nmoved = max((a for a, _ in moved_map.values()), default=0)
    if moved and _nmoved > 0:
        print(f"\n结论：**通过** ✅ 关0/1/2b/3 全过："
              f"{len(moved)} 处 update_rel>0，单步最多更新 {_nmoved} 个张量")
        return 0

    warns = [t for l, t in findings if l == WARN]
    if warns and args.strict:
        print("\n结论：**未通过**（--strict：WARN 也算失败）")
        return 1
    print("\n结论：条件通过 —— 链路通、无致命项，但**没有任何一处 update_rel>0**，"
          "\n      即梯度链尚未被实测验证过（所有组奖励都同分）。"
          "\n      → 开 8 卡前建议先加大规模再跑一次："
          "\n        sbatch --export=ALL,EZ_SMOKE_G=4,EZ_SMOKE_TURNS=16 ... ez_smoke_1gpu.sbatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
