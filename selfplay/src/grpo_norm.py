"""GRPO surrogate loss 的**归约口径**（token 级 vs 序列级）。

单独拆成一个只依赖 torch 的小文件，是为了让它**可被离线单测直接测到真代码**：

    `train_grpo.py` 顶层要 import transformers / alfworld_compat / alfworld_env /
    rollout，而跑单测的环境（CI、登录节点、沙箱）往往只有 torch。以前这段数学
    埋在 `grpo_loss` 里，离线单测 import 不进来，只能复刻一份实现去测 ——
    复刻的和跑的不是一个东西，改了跑的那份单测照样绿。抽出来就没这个问题。

口径之争（2026-10-12 交叉 review 时由第二个模型独立指出，我再自己推一遍核实）：

    DeepSeekMath / GRPO 论文公式(4)：
        J = (1/G) Σ_i [ (1/|o_i|) Σ_t ( ... ) ]
        —— 先**每条序列**内部按 token 平均，再**组间等权**平均。

    `loss_norm="token"`（旧行为）：
        J = ( Σ_i Σ_t (...) ) / ( Σ_i |o_i| )
        —— 把 G 条序列的 token 混在一起平均 = **按长度加权**。

为什么必须改：ratio ≡ 1 时 `per_tok = -adv_i` 在**序列内是常数**，于是

    token 级： loss = - Σ_i |o_i|·adv_i / Σ_i |o_i|     ← 长度加权
    序列级：   loss = - (1/G) Σ_i adv_i                  ← 组内等权

GRPO 归一化后 Σ_i adv_i = 0，所以序列级下"组内奖励对称"必然给出净信号 0
（这是 GRPO *应有*的性质）；而 token 级下**长度本身变成信号** —— 长序列的
advantage 被放大 |o_i| / mean(|o|) 倍。这就是 Dr.GRPO 点名的 length bias：
模型会去学"写长一点/短一点"，而不是"答对一点"。

实测（scripts/ez_repro_lossnorm.py，G=8、长度 20~300、奖励 4 好 4 坏、2000 组）：
    序列级净信号恒为 0；token 级 |净信号| 均值 0.149、最大 0.614。
极端例子：G=2、长度 20 vs 200、一好一坏 → token 级权重 (0.091, 0.909)，
长序列独占 91%。G 越小、长度方差越大，偏置越离谱。
"""

from __future__ import annotations

import torch

SEQUENCE_LEVEL = "seq"
TOKEN_LEVEL = "token"


def grpo_reduce(per_tok: torch.Tensor,
                mask: torch.Tensor,
                loss_norm: str = SEQUENCE_LEVEL,
                sequence_reduction: str = "mean") -> torch.Tensor:
    """把 per-token surrogate 张量归约成标量 loss。

    参数：
        per_tok:  形状 (n, W-1)，每个位置已算好的 `-min(surr1, surr2)`。
        mask:     同形状的 0/1 float 掩码（`loss_mask[:, 1:]`）。
        loss_norm:`"seq"`（论文语义，默认）或 `"token"`（旧行为，有长度偏置）。

    返回：标量 loss。**保证挂在计算图上**（见下面 mask 全 0 的兜底分支）。
    """
    denom = mask.sum().clamp(min=1.0)
    if sequence_reduction not in ("mean", "sum"):
        raise ValueError("sequence_reduction must be 'mean' or 'sum'")

    if loss_norm == TOKEN_LEVEL:
        if sequence_reduction == "sum":
            raise ValueError("sequence_reduction='sum' is defined only for seq normalization")
        value = (per_tok * mask).sum() / denom
        return value

    if loss_norm != SEQUENCE_LEVEL:
        raise ValueError(
            f"loss_norm 必须是 {SEQUENCE_LEVEL!r} 或 {TOKEN_LEVEL!r}，收到 {loss_norm!r}")

    # ---- 序列级：每条序列先按自己的 token 数平均，再对序列取等权平均 ----
    n_tok = mask.sum(dim=1)                     # (n,)
    seq_sum = (per_tok * mask).sum(dim=1)       # (n,)
    ok = n_tok > 0                              # mask 全 0 的行不能进平均，否则会稀释
    if bool(ok.any()):
        seq_values = seq_sum[ok] / n_tok[ok].clamp(min=1.0)
        return seq_values.sum() if sequence_reduction == "sum" else seq_values.mean()

    # 整批 mask 全 0：不能 `return torch.zeros(())` —— 那样 loss 没有 grad_fn，
    # 外层 `(loss / len(groups)).backward()` 会直接抛错，本 rank 就少一次
    # backward，其它 rank 永久卡在 NCCL barrier（红线 B）。
    # 用"值恒为 0 但仍挂在图上"的表达式占位。
    return (per_tok * mask).sum() / denom * 0.0


def seq_weights(mask: torch.Tensor) -> torch.Tensor:
    """序列级口径下每条序列的权重（= 1/有效序列数），只用于诊断/单测。"""
    ok = mask.sum(dim=1) > 0
    g = int(ok.sum().item())
    if g == 0:
        return torch.zeros_like(mask.sum(dim=1))
    w = torch.zeros_like(mask.sum(dim=1))
    w[ok] = 1.0 / g
    return w


def tok_weights(mask: torch.Tensor) -> torch.Tensor:
    """token 级口径下每条序列的权重（= |o_i| / Σ|o|），只用于诊断/单测。"""
    n_tok = mask.sum(dim=1)
    tot = n_tok.sum().clamp(min=1.0)
    return n_tok / tot
