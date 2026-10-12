"""Explorer-Zero 验证工件与奖励（论文 §3.1/§3.2 的 Eq.1-7）。

对应关系
--------
* `ProposedTask`      —— proposer 的产物：(x, s_ref_final, {(g_k, s_ref_k)})
* `validity_filter`   —— §3.2 的三条规则过滤
* `redundant_pair`    —— §3.1 冗余判定：delta 相同且描述 Jaccard > 0.6
* `solver_reward`     —— Eq.5/6/7：成功=1；失败=子任务完成率
* `proposer_reward`   —— Eq.1-4：难度(1-|2p-1|) + w_sub·K + w_div·|动作类型集|
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ez_state import (StateDelta, StateSnapshot, deltas_equal, match_final_state,
                      match_reference, state_delta)

# --------------------------------------------------------------------------- 数据结构


@dataclass
class SubtaskRef:
    """一个声明的子任务：自然语言描述 + 到达该子任务时的参考状态/delta。"""

    description: str
    ref_snapshot: StateSnapshot          # 快照（绝对状态）
    ref_delta: StateDelta                # 相对 s0 的变化（冗余判定与匹配都用它）

    def to_json(self) -> dict:
        return {"description": self.description,
                "ref_snapshot": self.ref_snapshot.to_json(),
                "ref_delta": self.ref_delta.to_json()}

    @staticmethod
    def from_json(d: dict) -> "SubtaskRef":
        return SubtaskRef(description=d["description"],
                          ref_snapshot=StateSnapshot.from_json(d["ref_snapshot"]),
                          ref_delta=StateDelta.from_json(d["ref_delta"]))


@dataclass
class ProposedTask:
    """proposer 一条成功 rollout 的产出（论文 §3.1 的元组）。"""

    task_desc: str                       # x，祈使句
    s0: StateSnapshot                    # 初始状态
    ref_final: StateSnapshot             # s_ref_final
    ref_final_delta: StateDelta
    subtasks: List[SubtaskRef] = field(default_factory=list)
    # —— 溯源/调试 ——
    game_file: str = ""
    proposer_actions: List[str] = field(default_factory=list)
    proposer_action_types: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "task_desc": self.task_desc,
            "s0": self.s0.to_json(),
            "ref_final": self.ref_final.to_json(),
            "ref_final_delta": self.ref_final_delta.to_json(),
            "subtasks": [s.to_json() for s in self.subtasks],
            "game_file": self.game_file,
            "proposer_actions": self.proposer_actions,
            "proposer_action_types": self.proposer_action_types,
            "meta": self.meta,
        }

    @staticmethod
    def from_json(d: dict) -> "ProposedTask":
        return ProposedTask(
            task_desc=d["task_desc"],
            s0=StateSnapshot.from_json(d["s0"]),
            ref_final=StateSnapshot.from_json(d["ref_final"]),
            ref_final_delta=StateDelta.from_json(d["ref_final_delta"]),
            subtasks=[SubtaskRef.from_json(s) for s in d.get("subtasks", [])],
            game_file=d.get("game_file", ""),
            proposer_actions=d.get("proposer_actions", []),
            proposer_action_types=d.get("proposer_action_types", []),
            meta=d.get("meta", {}),
        )


# --------------------------------------------------------------------------- 有效性过滤（§3.2）

_WORD_RE = re.compile(r"[a-z]+")


def _words(text: str) -> set:
    return set(_WORD_RE.findall(text.lower()))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    return inter / max(1, len(a | b))


def redundant_pair(desc_a: str, delta_a: StateDelta, desc_b: str, delta_b: StateDelta,
                   jaccard_threshold: float = 0.6) -> bool:
    """§3.1：delta 相同 **且** 描述词级 Jaccard 超过阈值 → 冗余。"""
    return deltas_equal(delta_a, delta_b) and jaccard(_words(desc_a), _words(desc_b)) > jaccard_threshold


def validity_filter(task: ProposedTask, jaccard_threshold: float = 0.6) -> Tuple[bool, str]:
    """§3.2 三条规则。返回 (接受?, 拒绝原因)。被拒绝的任务不进 solver 训练分布，
    proposer 奖励为 0（Appendix C）。"""
    if not task.task_desc or not task.task_desc.strip():
        return False, "no_finish_task"
    # 规则 0（review P1-2）：终态相对 s0 没有任何变化的"任务"没有信息量——
    # 不拒的话 Eq.5 空 delta 匹配恒真，solver 奖励恒 1、p̂ 恒 1，整条线退化。
    if task.ref_final_delta.is_empty():
        return False, "empty_final_delta"
    for i in range(len(task.subtasks)):
        for j in range(i + 1, len(task.subtasks)):
            a, b = task.subtasks[i], task.subtasks[j]
            if redundant_pair(a.description, a.ref_delta, b.description, b.ref_delta,
                              jaccard_threshold):
                return False, f"redundant_subtask:{i}:{j}"
    # 规则 iii：任务描述里点名的物体/应用必须出现在 proposer 轨迹中。
    # 实现：任务描述里的每个"物体词"至少要出现在某条动作或子任务描述里。
    # ALFWorld 动作形如 "heat apple 1 with microwave 1"，先取动作的物体部分。
    traj_text = " ".join(task.proposer_actions + [s.description for s in task.subtasks]).lower()
    # 从 task_desc 抽候选物体词：去掉停用词后的实词
    stop = {"a", "an", "the", "in", "on", "at", "and", "then", "with", "to", "of",
            "put", "place", "take", "go", "heat", "clean", "cool", "open", "close",
            "look", "use", "some", "it", "into", "from"}
    missing = [w for w in _words(task.task_desc) - stop
               if len(w) >= 4 and w not in traj_text]
    # 抽词是启发式：为避免误杀，只在缺失词>=2 时才拒（单词缺失常是描述措辞差异）
    if len(missing) >= 2:
        return False, f"unsupported_objects:{missing}"
    return True, "ok"


# --------------------------------------------------------------------------- solver 奖励（Eq.5-7）


@dataclass
class SolverRolloutResult:
    """solver 一条轨迹的验证结果。"""

    final_match: bool                     # Eq.5: s_T ≃ s_ref_final
    subtask_hits: List[bool]              # Eq.6 每一项
    n_subtasks: int
    reward: float                         # Eq.7

    @property
    def subtask_rate(self) -> float:
        return sum(self.subtask_hits) / max(1, self.n_subtasks)


def verify_solver_rollout(task: ProposedTask,
                          solver_snapshots: Sequence[StateSnapshot]) -> SolverRolloutResult:
    """对 solver 一条轨迹（沿途状态序列）算 Eq.5/6/7。

    `solver_snapshots[0]` 必须等于任务初始状态 s0（调用方负责 reset 到同一局）。
    """
    s0 = task.s0
    sT = solver_snapshots[-1] if solver_snapshots else s0
    final_match = match_final_state(s0, task.ref_final, sT)
    hits = [match_reference(s0, st.ref_snapshot, solver_snapshots) for st in task.subtasks]
    if final_match:
        reward = 1.0
    else:
        reward = (sum(hits) / max(1, len(task.subtasks))) if task.subtasks else 0.0
    return SolverRolloutResult(final_match=final_match, subtask_hits=hits,
                               n_subtasks=len(task.subtasks), reward=reward)


# --------------------------------------------------------------------------- proposer 奖励（Eq.1-4）


def action_type(action: str) -> str:
    """ALFWorld 动作类型 = 命令动词（论文 §3.1 定义）。"""
    return action.strip().split(" ", 1)[0].lower() if action.strip() else ""


def proposer_reward(task: ProposedTask,
                    solve_rate: float,
                    w_sub: float = 0.3,
                    w_div: float = 0.3,
                    jaccard_threshold: float = 0.6) -> Tuple[float, Dict[str, float]]:
    """Eq.1-4。`solve_rate` 是冻结 solver 的 G_P 条 rollout 的成功率 p̂(x)。

    返回 (奖励, 分解项)。无效提案（validity_filter 不过）由调用方直接给 0。
    """
    r_diff = 1.0 - abs(2.0 * solve_rate - 1.0)          # Eq.2，p̂=0.5 最优
    # Eq.3：只数非冗余子任务
    kept = 0
    for i in range(len(task.subtasks)):
        red = any(redundant_pair(task.subtasks[i].description, task.subtasks[i].ref_delta,
                                 task.subtasks[j].description, task.subtasks[j].ref_delta,
                                 jaccard_threshold)
                  for j in range(len(task.subtasks)) if j != i)
        if not red:
            kept += 1
    r_sub = w_sub * kept
    r_div = w_div * len({action_type(a) for a in task.proposer_action_types or task.proposer_actions})
    total = r_diff + r_sub + r_div
    return total, {"r_difficulty": r_diff, "r_subtask": r_sub, "r_diversity": r_div,
                   "n_nonredundant_subtasks": kept,
                   "n_action_types": len({action_type(a) for a in task.proposer_actions})}
