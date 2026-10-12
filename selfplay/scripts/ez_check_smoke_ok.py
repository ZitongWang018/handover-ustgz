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

判读逻辑（★ 关键：这些现象长得像，但处理方法完全不同）

  ★ 判"参数没动"的最终依据是**真实 backward grad_norm**（GN_RE），不是 update_rel：
      grad_norm>0 且 一步没动                → 【致命】梯度算出来却没落到参数上
      grad_norm=0（奖励全同分 ⇒ 优势全 0）    → 【正常】不动是正确的
      某 rank 的 update_rel 与别 rank 差 100×  → 【致命】那个 rank 没参与梯度同步，
                                                  **除非**它的 grad_norm 也是 0（=一致无信号）

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

★ 反过来，关2b 也需要 grad_norm 兜底（2026-10-12 job 12989603 实测）：
  2 卡冒烟本发 `accept=0/2` ⇒ 全部候选同分 ⇒ 组内优势全 0 ⇒ 梯度**精确为 0** ⇒
  `0/504 张量动了` 是**正确**行为，但被旧版判据打成两条 FATAL。
  现在只有"grad_norm 非零却一步没动"才判 FATAL。

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
# [selfcheck] 梯度链 proposer grad_norm=5.6663e+02 (504 个张量有 grad) | solver grad_norm=5.6441e+02 (...)
GRADCHAIN_RE = re.compile(r"\[selfcheck\]\s*梯度链\s*proposer grad_norm=(?P<p>[0-9.eE+-]+)"
                          r".*?solver grad_norm=(?P<s>[0-9.eE+-]+)")
# ★ 这一条是「真实 backward grad_norm」，它是把"0 位移"的两种成因分开的唯一权威依据：
#   注意 P/S 相位的诊断行是**两个独立方括号** `[outer 2][P0] 诊断 ...`，和 update_probe
#   的 `[outer 2 P0] update_rel=...`（一个括号、空格分隔）格式不同，正则不能混用。
#   P 相位： [outer 2][P0] 诊断 adv_spread/... | 真实 backward grad_norm=4.3530e+00 (504 个张量有 grad；A=.. B=..)
#   S 相位： [outer 2][S0] 诊断 真实 backward grad_norm=0.0000e+00 (0 个张量有 grad；A=.. B=..)
GN_RE = re.compile(r"\[outer (?P<outer>\d+)\]\[(?P<phase>[PS])(?P<step>\d+)\]\s*诊断"
                   r".*?真实 backward grad_norm=(?P<gn>[0-9.eE+-]+)")


def _gn_key(m) -> str:
    """把 GN_RE 的匹配归一成与 UPD_RE.group(1) 同构的 key（'outer 2 P0'）。"""
    return f"outer {m.group('outer')} {m.group('phase')}{m.group('step')}"


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

    # ---- ★ 提前读取各 rank 日志并聚合（**只读一次**，关3 与关4 共用）----
    #   原来这段在文件末尾的关4 里，导致关3 的 proposer 判据只能扫到 rank0 的日志。
    #   多卡时每个 rank 只处理 TPS/world 个 game，rank0 自己 accept=0/2 是**正常的**
    #   （别的 rank 接受了任务；日志里 `pool=2 (+2)` 就是 all_gather 后的全局证据）。
    #   per_rank     : rank -> 该 rank 所有 P 相位 update_rel 值
    #   per_rank_gn  : rank -> 该 rank 所有相位的「真实 backward grad_norm」值
    #   per_rank_acc : rank -> 该 rank 所有 proposer 相位的 (accepted, total)
    #   worlds       : 各 rank 日志里写的 world_size
    #   empty        : 内容为空的 rank 日志文件名
    rank_files = sorted(run_dir.glob("train_log.rank*.jsonl"))
    per_rank: dict[int, list[float]] = {}
    per_rank_gn: dict[int, list[float]] = {}
    per_rank_acc: dict[int, list[tuple[int, int]]] = {}
    worlds: set[int] = set()
    empty: list[str] = []
    for rf in rank_files:
        rk = int(re.search(r"rank(\d+)", rf.name).group(1))
        vals: list[float] = []
        gns: list[float] = []
        accs_r: list[tuple[int, int]] = []
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
                if (mg := GN_RE.search(m_)):
                    gns.append(_fnum(mg.group("gn")))
                if (mp := PPLINE_RE.search(m_)):
                    accs_r.append((int(mp.group("acc")), int(mp.group("tot"))))
        if n_lines == 0:
            empty.append(rf.name)
        per_rank[rk] = vals
        per_rank_gn[rk] = gns
        per_rank_acc[rk] = accs_r

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
    #   ★ 但"0 个张量动了"本身还有第二种成因，必须用**真实 backward grad_norm** 分开
    #     （job 12989603 实测）：奖励全同分 ⇒ 组内优势全 0 ⇒ 梯度**精确为 0** ⇒
    #     参数不动是**正确的**，不是 bug。只有"梯度非零却一步没动"才是真断链。
    #     这正是 2 卡那一发被误判成两条 FATAL 的原因（accept=0/2 ⇒ 全同分 ⇒ 0/504）。
    moved_map: dict[str, tuple[int, int]] = {}
    for m in (UPD_MOVED_RE.search(x) for x in msgs):
        if m:
            moved_map[m.group(1)] = (int(m.group("moved")), int(m.group("tot")))
    gn_map: dict[str, float] = {}
    for m in (GN_RE.search(x) for x in msgs):
        if m:
            gn_map[_gn_key(m)] = _fnum(m.group("gn"))
    if not moved_map:
        findings.append((WARN, "关2b 日志里没有 '张量真的动了' —— 跑的是旧代码？（probe 未升级）"))
    else:
        _b = max(a for a, _ in moved_map.values())
        _gn_max = max(gn_map.values()) if gn_map else None
        _mv_txt = ", ".join(f"{k}={a}/{t}" for k, (a, t) in moved_map.items())
        if _b > 0:
            findings.append((OK, f"关2b 优化器实际更新张量数：{_mv_txt}"))
        elif _gn_max is None:
            findings.append((FATAL, f"关2b 一步都没动（{_mv_txt}），且日志里没有"
                                    f"'真实 backward grad_norm' —— 无法区分「真断链」与"
                                    f"「奖励同分」，跑的是旧代码？这一步必修"))
        elif _gn_max == 0.0:
            findings.append((WARN, f"关2b 一步都没动（{_mv_txt}），但各相位真实 backward "
                                   f"grad_norm 全为 0 —— 奖励全同分 ⇒ 优势全 0 ⇒ 参数不动是"
                                   f"**正确的**（无学习信号，非 bug）"))
        else:
            findings.append((FATAL, f"关2b **真断链**：一步都没动（{_mv_txt}），但真实 "
                                    f"backward grad_norm={_gn_max:.3e} 非零 —— 梯度算出来了"
                                    f"却没落到参数上（优化器/lr=0/精度不匹配）"))

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

    #   ★ proposer 判据必须**跨 rank 聚合**：多卡时 TPS 个 game 被切成 world 份，
    #     每个 rank 只有 TPS/world 个，单独看任意一个 rank 的 accept=0 都是正常的。
    #     只有**所有 rank 合计**为 0 才是"提案器一条有效任务都没产出"。
    if per_rank_acc:
        # {rank: (accepted 合计, total 合计)}
        _sum_by_rank = {rk: (sum(a for a, _ in v), sum(t for _, t in v))
                        for rk, v in per_rank_acc.items() if v}
    else:
        # 退化：没有 rank 文件（旧产物）时用主日志 = rank0
        _sum_by_rank = {0: (sum(a for a, _, _ in p_lines.values()),
                            sum(t for _, t, _ in p_lines.values()))}

    if not _sum_by_rank:
        findings.append((WARN, "关3 没有 proposer 相位记录（可能本 outer 没走到）"))
    else:
        g_acc = sum(a for a, _ in _sum_by_rank.values())
        g_tot = sum(t for _, t in _sum_by_rank.values())
        detail = ", ".join(f"r{rk}={a}/{t}" for rk, (a, t) in _sum_by_rank.items())
        if g_acc == 0:
            findings.append((FATAL, f"关3 proposer 全部被拒（{detail}，全局 {g_acc}/{g_tot}）"
                                    f"—— 提案器一条有效任务都没产出。"
                                    f"\n          先看 fmt_bad：>0 是解析/格式问题；=0 但仍 accept=0，"
                                    f"通常是**探索轮数不够**（prompt 要求探索 10-20 个动作后才 "
                                    f"finish，EZ_SMOKE_TURNS 太小就永远走不到 finish）"))
        else:
            findings.append((OK, f"关3 proposer 有有效提案：{detail}（全局 {g_acc}/{g_tot}）"))

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
    #   ★ 这一关只在跑过 torchrun 多进程时才有意义。判据来自 EZ_DDP_PLAN.md §8：
    #     ① 每 rank 的 train_log.rank*.jsonl 存在且非空（缺文件 = 那个 rank 提早挂了）
    #     ② update_rel 在各 rank 之间同量级（差 100× 说明某个 rank 没参与梯度同步）
    #     ③ 日志里的 world_size 真的是 N（torchrun 没生效时**不报错**，只是 N 个进程
    #        各自在单卡上跑 —— 速度没变快，日志却多了 N 倍）
    #   ★ ② 有个陷阱（job 12989603 实测）：本发 accept=0/2 ⇒ 全组奖励同分 ⇒ 各 rank
    #     的 update_rel **一致地**为 0，比值 = inf，被当成"没参与同步"。但各 rank 的
    #     真实 grad_norm 也都是 0，是"一致无信号"而非"不同步"。必须用 GN_RE 分开。
    #   （rank_files / per_rank / per_rank_gn / per_rank_acc / worlds / empty
    #     已在关2 之前提前聚合好，这里直接复用，不再重复读文件）
    if rank_files:
        # ⚠️ 只有**真的跑过多进程**才判关4：单卡运行时也会写 train_log.rank0.jsonl
        #    （每 rank 一份日志是通用设计），此时 world_size=1 是**正常的**，
        #    拿"8 进程"的标准去判单卡会打出假 WARN（job 12989528 实测踩到）。
        multi = len(rank_files) >= 2 or max(worlds or {1}) > 1
        if not multi:
            findings.append((OK, f"关4 单卡运行（world_size={sorted(worlds)}, "
                                 f"{len(rank_files)} 个 rank 日志），跳过 8 卡判据"))
        else:
            if empty:
                findings.append((FATAL, f"关4 这些 rank 日志是空的（该 rank 提早挂了）：{empty}"))
            else:
                findings.append((OK, f"关4 {len(rank_files)} 个 rank 日志都存在且非空"))
            if worlds:
                w_max = max(worlds)
                findings.append((OK if w_max > 1 else WARN,
                                 f"关4 日志里的 world_size={sorted(worlds)}"
                                 + ("" if w_max > 1 else
                                    "（=1！torchrun 没生效，8 个进程各跑各的）")))
            if len(rank_files) < 2 and args.strict:
                findings.append((WARN, f"关4 只找到 {len(rank_files)} 个 rank 日志 —— "
                                       f"以为在跑 8 卡？检查 torchrun --nproc_per_node"))
            peaks = {r: max(v) for r, v in per_rank.items() if v}
            gn_pk = {r: max(v) for r, v in per_rank_gn.items() if v}
            if len(peaks) >= 2:
                lo, hi = min(peaks.values()), max(peaks.values())
                ratio = (hi / lo) if lo > 0 else float("inf")
                _pktxt = ", ".join(f"r{r}={v:.2e}" for r, v in sorted(peaks.items()))
                _allzero = bool(gn_pk) and max(gn_pk.values()) == 0.0
                if lo > 0 and ratio < 100:
                    findings.append((OK, f"关4 各 rank 的 max(update_rel)={_pktxt}"
                                         f"（最大/最小={ratio:.1f}，同量级 ⇒ 各 rank 都参与了"
                                         f"梯度同步）"))
                elif lo == 0 and _allzero:
                    _gn0 = ", ".join(f"r{r}={v:.1e}" for r, v in sorted(gn_pk.items()))
                    findings.append((OK, f"关4 各 rank 一致无梯度：max(update_rel) 全 0 且"
                                         f"各 rank 真实 grad_norm 也全 0（{_gn0}）"
                                         f"—— 奖励全同分，不是梯度同步问题"))
                else:
                    _gn_txt = ", ".join(f"r{r}={v:.1e}" for r, v in sorted(gn_pk.items())) or "无"
                    findings.append((FATAL, f"关4 各 rank 的 max(update_rel)={_pktxt}"
                                            f"（最大/最小={ratio}）与各 rank 真实 grad_norm="
                                            f"[{_gn_txt}] 不一致 —— 某 rank 没参与梯度同步"
                                            f"（或该 rank 的日志本来就是旧的）"))

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
    _gn_any = max(gn_map.values()) if gn_map else None
    if _gn_any == 0.0:
        print("\n结论：条件通过 —— 链路通、无致命项，且各相位**真实 backward grad_norm 全为 0**"
              "\n      （奖励全同分 ⇒ 组内优势全 0 ⇒ 梯度精确为 0 ⇒ 参数不动是正确的）。"
              "\n      即：DDP/梯度链路已验证，但本发没有任何学习信号。"
              "\n      → 要拿到「参数真的动了」的硬证据，加大规模再跑一次："
              "\n        sbatch --export=ALL,EZ_SMOKE_G=4,EZ_SMOKE_TURNS=16 ... ez_smoke_1gpu.sbatch")
        return 0
    print("\n结论：条件通过 —— 链路通、无致命项，但**没有任何一处 update_rel>0**，"
          "\n      即梯度链尚未被实测验证过（所有组奖励都同分）。"
          "\n      → 开 8 卡前建议先加大规模再跑一次："
          "\n        sbatch --export=ALL,EZ_SMOKE_G=4,EZ_SMOKE_TURNS=16 ... ez_smoke_1gpu.sbatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
