#!/usr/bin/env python
"""冒烟训练结果的「准入门」：读 `train_rank{rank}.jsonl`，判断值不值得继续正式训练。

为什么需要这道闸
----------------
8 卡 A800 整机作业在二期要排十几个小时。如果冒烟阶段其实已经暴露了
「模型根本没吐出可解析的动作」或「advantage 恒为 0」这类问题，却照样往下跑
正式训练，就是白白烧掉一整个 allocation（300 iter × 8 卡）。

实测踩过的坑（2026-10-12）：
    reward_mean=1.6 reward_max=1.6 group_std_mean=0.0 group_degenerate_frac=1.0
    adv_abs_mean=0.0 loss=0.0  eval/success_rate=0.0  eval/invalid_action_rate=1.0
—— 每一步动作都不是 admissible 里的合法命令，奖励恒等于「全程非法」那一档，
   组内 std=0 → advantage 全 0 → loss=0。这种状态跑 300 iter 和跑 1 iter 没区别。

判定规则
--------
* FAIL  mean(invalid_rate) > 0.95        —— 模型没吐出任何合法动作，先去修 prompt/解析
* FAIL  max(adv_abs_mean) < 1e-8         —— 每一步都没有梯度信号，GRPO 空转
* FAIL  max(update_rel) < 1e-10          —— **参数一步都没动**（lr 太小 / dtype 舍入）
* FAIL  mean(invalid_reward_corr) > 0.3  —— 奖励符号写反了，模型正在被训成"专出非法动作"
* WARN  mean(group_degenerate_frac)>=1.0 —— 组内全同分（g 小时可能正常，但值得看一眼）

最后两条是 2026-10-12 补的，专门盯「数值全非零、日志全正常、但训练是错的」这类
**静默失败**。它们不会让 loss 变成 0，所以只查 advantage/loss 完全抓不到。

退出码：0 = 放行，1 = 拦下。

用法
----
    python scripts/check_smoke_ok.py --config configs/train_qwen3_4b.yaml --iters 5
    python scripts/check_smoke_ok.py --jsonl runs/qwen3-4b-alfworld-grpo/train_rank0.jsonl --iters 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

INVALID_RATE_FAIL = 0.95      # 非法动作率高于此 -> 模型根本不会输出合法命令
ADV_ABS_MIN = 1e-8            # advantage 绝对值上限低于此 -> 完全无梯度
UPDATE_REL_MIN = 1e-10        # 参数相对更新量低于此 -> optimizer.step() 没真的改到参数
CORR_MAX = 0.3                # "非法动作数" 与 "奖励" 的相关系数上限（必须 <= 0）


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _last_lines(path: Path, n: int) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"[gate] 找不到 {path} —— 冒烟训练根本没写出日志，直接拦下")
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue          # 写到一半被 kill 的残行，忽略
    if not rows:
        raise SystemExit(f"[gate] {path} 里没有任何完整 jsonl 行 —— 直接拦下")
    return rows[-n:] if n > 0 else rows


def _resolve_jsonl(cfg_path: Path, rank: int) -> Path:
    """从训练配置里推出 run_dir，和 `train_grpo.py:main()` 的算法保持一致。"""
    from train_grpo import load_cfg  # noqa: WPS433（延迟导入：需要先插 sys.path）

    cfg = load_cfg(str(cfg_path))
    name = cfg["run"]["name"]
    out_dir = cfg["run"]["output_dir"]
    return (cfg_path.resolve().parent.parent / out_dir / name / f"train_rank{rank}.jsonl").resolve()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="训练 yaml（自动推出 jsonl 路径）")
    ap.add_argument("--jsonl", help="直接指定 jsonl 路径")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--iters", type=int, default=5, help="只看最后 N 行")
    ap.add_argument("--strict", action="store_true",
                    help="把 WARN 也当 FAIL（CI / 无人值守时用）")
    args = ap.parse_args()

    if args.jsonl:
        jsonl = Path(args.jsonl).expanduser().resolve()
    elif args.config:
        jsonl = _resolve_jsonl(Path(args.config).expanduser(), args.rank)
    else:
        ap.error("必须给 --config 或 --jsonl")

    rows = _last_lines(jsonl, args.iters)
    print(f"[gate] 读 {jsonl} 的最后 {len(rows)} 行")

    # `invalid_rate` / `mean_steps` / `format_viol_rate` 是 2026-10-12 才加的诊断量，
    # 老日志里没有。缺了就跳过那一判，并明确提示，避免"因为字段不存在而误放行"。
    has_invalid = "invalid_rate" in rows[0]
    inv = [float(r["invalid_rate"]) for r in rows if "invalid_rate" in r]
    adv = [abs(float(r.get("adv_abs_mean", 0.0))) for r in rows]
    degen = [float(r.get("group_degenerate_frac", 0.0)) for r in rows]
    steps = [float(r.get("mean_steps", 0.0)) for r in rows if "mean_steps" in r]
    rewards = [float(r.get("reward_mean", 0.0)) for r in rows]
    # ⚠️ 只有"确实走过 optimizer.step()"的 iter 才配谈 update_rel。
    #    advantage=0 时代码会主动跳过这一步（DDP 下必须所有 rank 一致），
    #    此时 update_rel=0 是**正确的**，不能当成"参数没动"报警。
    #    `did_step` 没写进日志，用 adv_abs_mean>0 反推（和 train_grpo 的判据一致）。
    stepped = [r for r in rows if abs(float(r.get("adv_abs_mean", 0.0))) > ADV_ABS_MIN]
    upd = [float(r.get("update_rel", 0.0)) for r in stepped if "update_rel" in r]
    corr = [float(r.get("invalid_reward_corr", 0.0)) for r in rows if "invalid_reward_corr" in r]

    print(f"[gate] reward_mean      = {_mean(rewards):.4f}")
    print(f"[gate] adv_abs_mean max = {max(adv):.6f}")
    print(f"[gate] degenerate_frac  = {_mean(degen):.3f}")
    if has_invalid:
        print(f"[gate] invalid_rate     = {_mean(inv):.4f}")
        if steps:
            print(f"[gate] mean_steps       = {_mean(steps):.2f}")
    else:
        print("[gate] ⚠️ 日志里没有 invalid_rate 字段（旧代码写的）—— 无法判断动作合法性")
    if upd:
        print(f"[gate] update_rel max    = {max(upd):.3e}  ({len(upd)}/{len(rows)} iter 真的 step 了)")
    else:
        print("[gate] ⚠️ 没有任何一个 iter 走到 optimizer.step()（advantage 全 0）")
    if corr:
        print(f"[gate] invalid_reward_corr mean = {_mean(corr):.3f}   (必须 <= 0)")

    fails: List[str] = []
    warns: List[str] = []

    if has_invalid and _mean(inv) > INVALID_RATE_FAIL:
        fails.append(
            f"非法动作率 {_mean(inv):.1%} —— 模型每一步都没吐出 admissible 里的合法命令。"
            "这是 prompt / 解析 / max_new_tokens 的问题，不是奖励的问题；"
            "去修 `src/prompts.py` 和 rollout 的 max_new_tokens，别浪费 8 张卡。"
        )
    if max(adv) < ADV_ABS_MIN:
        fails.append(
            "每一步 advantage 全是 0（组内无区分度）—— GRPO 完全没有梯度，"
            "跑正式训练等于空转。检查 group_size、奖励是否有区分度。"
        )
    if _mean(degen) >= 1.0:
        warns.append(
            f"所有 group 都是退化的（组内全同分，共 {len(rows)} iter）。"
            "g 很小时可能只是样本不够，但配合 adv=0 就一定是奖励没区分度。"
        )
    if upd and max(upd) < UPDATE_REL_MIN:
        fails.append(
            f"有 {len(upd)} 个 iter 走了 optimizer.step()，但参数相对变化量 "
            f"max(update_rel)={max(upd):.3e} ≈ 0 —— 参数一步都没动。"
            "典型原因：① 参数 dtype 是 bfloat16 而 lr 只有 1e-6，更新量被 ulp 舍入抹掉；"
            "② lr 量级设错。改法：model.dtype 用 float32（配 bf16 autocast），或调大 lr。"
        )
    if corr and _mean(corr) > CORR_MAX:
        fails.append(
            f"'非法动作数' 与 '奖励' 的相关系数 = {_mean(corr):+.3f}（必须 <= 0）—— "
            "奖励的惩罚项符号写反了：输出越多非法动作反而拿越高分，"
            "模型正在被直接训成『专门输出非法动作』。去查 reward.compute_reward 的 `-=`。"
        )

    for w in warns:
        print(f"[gate] WARN  {w}")

    if fails or (warns and args.strict):
        for f in fails + (warns if args.strict else []):
            print(f"[gate] FAIL  {f}")
        print("[gate] ❌ 不准入 —— 不进入正式训练阶段")
        return 1

    print("[gate] ✅ 准入门通过，可以继续正式训练")
    return 0


if __name__ == "__main__":
    sys.exit(main())
