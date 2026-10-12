#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""最小复现：`grpo_loss` 的归一化口径是 **token 级** 还是 **序列级**。

背景（review 交叉核对时由第二个模型独立指出，我再自己推一遍核实）：

  DeepSeekMath / GRPO 论文公式(4)：
      J = (1/G) Σ_i [ (1/|o_i|) Σ_t ( ... ) ]
      —— 先**每条序列**内部按 token 平均，再**组间等权**平均。

  `train_grpo.grpo_loss` 现在的写法：
      denom = mask.sum()                 # 整个 batch 的参与 token 总数
      loss  = (per_tok * mask).sum() / denom
      —— 把组里 G 条序列的 token **混在一起**平均，等于按长度加权。

为什么这是真问题（不是风格差异）：
  ratio ≡ 1 时 `per_tok = -adv_i` 在**序列内是常数**，于是

      token 级：  loss = - Σ_i |o_i|·adv_i / Σ_i |o_i|      ← 长度加权
      序列级：    loss = - (1/G) Σ_i adv_i                   ← 组内等权

  GRPO 归一化后 Σ_i adv_i = 0，所以序列级下组内"没有一致信号"；
  而 token 级下**长度本身变成了信号**：长序列的 advantage 被放大
  |o_i| / mean(|o|) 倍。这就是 Dr.GRPO 点名的 length bias ——
  模型会去学"写长一点/短一点"，而不是"答对一点"。

本脚本不碰 torch，直接算"每条序列对最终 loss 的贡献权重"来量化。
"""

import random
from typing import List, Tuple


def seq_weights(lengths: List[int]) -> List[float]:
    """序列级（论文）：w_i = 1/G。"""
    g = len(lengths)
    return [1.0 / g] * g


def tok_weights(lengths: List[int]) -> List[float]:
    """token 级（现状）：w_i = |o_i| / Σ|o|。"""
    tot = sum(lengths)
    return [L / tot for L in lengths]


def adv_ddof0(rewards: List[float]) -> List[float]:
    """reward.group_advantages 的现状：np.std 默认 ddof=0。"""
    g = len(rewards)
    mean = sum(rewards) / g
    var = sum((r - mean) ** 2 for r in rewards) / g          # ddof=0
    std = var ** 0.5
    if std < 1e-4:
        return [0.0] * g
    return [(r - mean) / (std + 1e-4) for r in rewards]


def adv_ddof1(rewards: List[float]) -> List[float]:
    """论文/TRL 常用：无偏样本标准差 ddof=1。"""
    g = len(rewards)
    mean = sum(rewards) / g
    var = sum((r - mean) ** 2 for r in rewards) / (g - 1)    # ddof=1
    std = var ** 0.5
    if std < 1e-4:
        return [0.0] * g
    return [(r - mean) / (std + 1e-4) for r in rewards]


def report(tag: str, lengths: List[int], rewards: List[float]) -> None:
    print(f"\n--- {tag} ---")
    print(f"  长度  {lengths}")
    print(f"  奖励  {rewards}")
    a0, a1 = adv_ddof0(rewards), adv_ddof1(rewards)
    print(f"  adv(ddof=0, 现状) {[round(x, 3) for x in a0]}")
    print(f"  adv(ddof=1, 论文) {[round(x, 3) for x in a1]}")
    for nm, adv in (("ddof=0", a0), ("ddof=1", a1)):
        wt, ws = tok_weights(lengths), seq_weights(lengths)
        eff_t = sum(w * a for w, a in zip(wt, adv))
        eff_s = sum(w * a for w, a in zip(ws, adv))
        print(f"  [{nm}] 组内净优势(决定更新方向)  token级={eff_t:+.4f}  序列级={eff_s:+.4f}")
        print(f"          每条序列权重  token级={[round(w,3) for w in wt]}"
              f"  序列级={[round(w,3) for w in ws]}")


def main() -> None:
    print("=" * 74)
    print("grpo_loss 归一化口径对照（token 级 = 现状 / 序列级 = 论文）")
    print("=" * 74)

    # A. 最刺眼的情形：G=2，一条短一条长，一好一坏
    report("A. G=2，长度 20 vs 200，奖励 1 vs 0（一好一坏）",
           [20, 200], [1.0, 0.0])

    # B. G=8，长度有真实差异，奖励 4 好 4 坏
    report("B. G=8，长度 30..300，奖励 4 好 4 坏",
           [30, 45, 60, 90, 120, 180, 240, 300],
           [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    # C. 奖励与长度**正相关**（长回答答对）——token 级会把这个偏置再放大
    report("C. G=8，越长的越答对（长度与奖励正相关）",
           [30, 45, 60, 90, 120, 180, 240, 300],
           [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # D. 奖励与长度**负相关**
    report("D. G=8，越长的越答错",
           [30, 45, 60, 90, 120, 180, 240, 300],
           [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    # E. 随机大样本：统计 token 级相对序列级的"方向偏离"
    print("\n" + "=" * 74)
    print("E. 随机压力测试：G=8，长度 20~300 均匀，奖励 0/1 各半，2000 组")
    print("=" * 74)
    rng = random.Random(0)
    bad = 0
    dev: List[float] = []
    for _ in range(2000):
        lens = [rng.randint(20, 300) for _ in range(8)]
        rew = [1.0] * 4 + [0.0] * 4
        rng.shuffle(rew)
        a = adv_ddof0(rew)
        eff_t = sum(w * x for w, x in zip(tok_weights(lens), a))
        eff_s = sum(w * x for w, x in zip(seq_weights(lens), a))
        dev.append(abs(eff_t - eff_s))
        if (eff_t > 0) != (eff_s > 0) and abs(eff_s) > 1e-9:
            bad += 1
    print(f"  序列级下组内净优势恒为 0（Σadv=0，组内无一致信号）")
    print(f"  token 级下 |净优势| 均值 = {sum(dev)/len(dev):.4f}，"
          f"最大 = {max(dev):.4f}")
    print(f"  方向被长度带偏的组数 = {bad}/2000 = {bad/2000:.1%}")
    print("\n  结论：token 级下即使组内奖励完全对称（4 好 4 坏，序列级净信号为 0），")
    print("        loss 仍会给出**非零的净更新方向**，方向由长度分布决定 ——")
    print("        这就是长度偏置，且 G 越小、长度方差越大越严重。")


if __name__ == "__main__":
    main()
