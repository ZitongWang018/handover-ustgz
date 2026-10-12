"""ALFWorld 上的奖励函数与 GRPO 的 group-normalized advantage。

ALFWorld 的任务奖励是**二值 + 稀疏**（整局结束才给 1 或 0），长时域（最多 50 步）。
纯 0/1 奖励在 GRPO 里早期极容易"组内全 0 → advantage 全 0 → 梯度 0"。
所以这里把奖励拆成可开关的几项，配置里逐项给权重：

    R = success * won
      + invalid_penalty * n_invalid            # 动作不在 admissible 里
      + format_penalty  * n_format_violation   # 没按 "Thought:/Action:" 格式输出
      + repeat_penalty  * n_repeat             # 连续重复同一个动作（死循环）
      + valid_step_bonus * n_valid_steps       # 每步都合法的小步奖励（提供方差的"陪伴"）
      + efficiency_bonus * (1 - steps/max_steps) if won
      + admissible_ratio_bonus * (n_valid_steps / n_steps)

⚠️ 上面所有**逐步骤累加**的项在 `compute_reward` 里都会先 **除以 `max_steps`**。
   否则长局的负分会被 clip 边界吃掉，让所有失败轨迹同分、组内 std=0、advantage 全 0（无梯度）。

**论文式对照**：全部置 0、只留 `success=1.0`，就是最"素"的 ALFWorld GRPO 设定。
配置里 `use_sparse_only: true` 可直接切到这一档。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class RewardConfig:
    # --- 主奖励 ---
    success: float = 1.0
    # --- 过程信号（默认很轻，只为在"全 0 组"里保留一点梯度方差）---
    invalid_action_penalty: float = 0.10
    format_violation_penalty: float = 0.10
    repeat_penalty: float = 0.05
    valid_step_bonus: float = 0.01
    admissible_ratio_bonus: float = 0.0
    efficiency_bonus: float = 0.10
    step_penalty: float = 0.0
    # --- 开关 ---
    use_sparse_only: bool = False
    clip_min: float = -1.0
    clip_max: float = 2.0
    # --- 任务类型加权（可选课程：难任务权重更高）---
    task_type_weights: Dict[str, float] = field(default_factory=dict)

    def resolved(self) -> "RewardConfig":
        """`use_sparse_only=True` 时把所有过程项清零，只留 success（论文式最素设定）。"""
        if not self.use_sparse_only:
            return self
        c = RewardConfig(
            success=self.success,
            invalid_action_penalty=0.0,
            format_violation_penalty=0.0,
            repeat_penalty=0.0,
            valid_step_bonus=0.0,
            admissible_ratio_bonus=0.0,
            efficiency_bonus=0.0,
            step_penalty=0.0,
            use_sparse_only=True,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
            task_type_weights=dict(self.task_type_weights),
        )
        return c

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrajectoryStats:
    """一趟 rollout 的统计量（rollout 阶段就地累计，不需要重放轨迹）。"""

    won: bool = False
    n_steps: int = 0
    n_valid_steps: int = 0
    n_invalid: int = 0
    n_format_violation: int = 0
    n_repeat: int = 0
    truncated: bool = False
    task_type: str = "unknown"
    task_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def compute_reward(stats: TrajectoryStats, cfg: RewardConfig, max_steps: int) -> float:
    """把一趟 rollout 的统计量打成标量奖励。

    ⚠️ 设计要点：所有**逐步骤累加**的过程项都先除以 `max_steps` 归一化。

    不归一化会踩一个很隐蔽的坑：长局（30/50 步）里 `-0.10 × 30 = -3.0` 直接把奖励顶到
    `clip_min`，于是**所有失败轨迹的分数完全相同** → 组内 std = 0 →
    GRPO 的 advantage 全 0 → 完全没有梯度。
    （实测就撞过：`reward_mean=2.000 / group_std=0.000`，训练恒等于空转。）

    归一化后每个分量都有固定预算，`success` 依旧主导：

        成功            ≈ +1.00 ~ +1.10
        全合法但没成功  ≈ +0.01
        全程非法动作    ≈ -0.20
    """
    c = cfg.resolved()
    denom = float(max_steps) if max_steps and max_steps > 0 else 1.0
    # ⚠️⚠️ 符号约定：所有名为 `*_penalty` 的项，配置里填**正数**表示惩罚，因此这里
    #    必须 `r -=`。曾经全部写成 `r +=`，后果是灾难性的且完全静默：
    #        全程非法  -> +0.2483
    #        全合法失败 -> +0.0100
    #    非法动作的奖励**高于**合法动作 -> 模型被直接训成"专门输出非法动作"。
    #    而日志里一切正常（advantage 非零、std 非零、loss 非零），
    #    所有基于数值的异常检测都抓不到它 —— 只有"奖励排序"检查能发现。
    r = c.success * float(stats.won)
    r -= c.invalid_action_penalty * (stats.n_invalid / denom)
    r -= c.format_violation_penalty * (stats.n_format_violation / denom)
    r -= c.repeat_penalty * (stats.n_repeat / denom)
    r += c.valid_step_bonus * (stats.n_valid_steps / denom)
    if c.admissible_ratio_bonus and stats.n_steps:
        r += c.admissible_ratio_bonus * (stats.n_valid_steps / stats.n_steps)
    if stats.won:
        r += c.efficiency_bonus * max(0.0, 1.0 - stats.n_steps / denom)
    r -= c.step_penalty * (stats.n_steps / denom)
    # ⚠️ 课程权重 task_type_weights **不要在这里乘**：同一组所有轨迹乘同一个 w，
    #    组内归一化 (R-mean)/(std+eps) 后 advantage 严格不变（mean/std 同比例缩放），
    #    乘了也白乘、纯属"配置骗人"。真正的课程作用点在 train_grpo 里、归一化**之后**的
    #    advantage 上乘（那里不同 task 的 advantage 是独立的，乘 w 才有效）。
    return float(np.clip(r, c.clip_min, c.clip_max))


def group_advantages(
    rewards: Sequence[float],
    group_ids: Sequence[int],
    eps: float = 1e-4,
    use_std: bool = True,
) -> List[float]:
    """按 group 做 GRPO 标准化：A_i = (R_i - mean_g) / (std_g + eps)。

    `use_std=False` 时只减均值（Dr.GRPO 风格，避免 std 太小时放大噪声）。
    **组内全同分时直接返回 0**，否则会除出一个放大的噪声方向。
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    group_ids = np.asarray(group_ids)
    adv = np.zeros_like(rewards)
    for g in np.unique(group_ids):
        m = group_ids == g
        grp = rewards[m]
        if grp.size <= 1:
            continue
        std = grp.std()
        if std < eps:
            # 组内无区分度：不给梯度（这正是稀疏奖励下的常态，靠过程奖励打破）
            adv[m] = 0.0
            continue
        adv[m] = (grp - grp.mean()) / (std + eps) if use_std else (grp - grp.mean())
    return adv.tolist()


def summarize_grouping(rewards: Sequence[float], group_ids: Sequence[int]) -> Dict[str, float]:
    """给日志用：组内方差情况（判断 current 奖励是否有区分度）。"""
    rewards = np.asarray(rewards, dtype=np.float64)
    group_ids = np.asarray(group_ids)
    stds, means, degenerate = [], [], 0
    for g in np.unique(group_ids):
        grp = rewards[group_ids == g]
        stds.append(float(grp.std()) if grp.size else 0.0)
        means.append(float(grp.mean()) if grp.size else 0.0)
        if grp.size > 1 and grp.std() < 1e-4:
            degenerate += 1
    n_groups = max(1, len(stds))
    return {
        "reward_mean": float(rewards.mean()) if rewards.size else 0.0,
        "reward_max": float(rewards.max()) if rewards.size else 0.0,
        "group_std_mean": float(np.mean(stds)) if stds else 0.0,
        "group_degenerate_frac": degenerate / n_groups,
        "n_groups": float(n_groups),
    }


def make_reward_cfg(d: Optional[dict]) -> RewardConfig:
    d = dict(d or {})
    known = {f for f in RewardConfig().to_dict()}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"未知的奖励配置项：{sorted(unknown)}；可用项：{sorted(known)}")
    return RewardConfig(**d)
