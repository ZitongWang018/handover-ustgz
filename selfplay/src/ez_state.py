"""Explorer-Zero 状态快照与匹配（论文 §3.1/§3.2/§4.1 的实现）。

论文对 ALFWorld 的验证规则只有一句："compare object types while ignoring instance
indices"（§4.1）。本模块把这句话落成可执行的规格（复现计划文档 U1 项）：

1. **状态 = 谓词事实集合**。优先取 textworld 的 `infos["facts"]`（textworld>=1.5
   支持 `EnvInfos(facts=True)`）；取不到时退回"观测文本签名"模式并大声警告——
   那种模式下匹配退化成字符串比对，只配调试用，不得用于正式实验。
2. **实例归一化**：`apple 1` / `apple 2` → `apple`（object type 级比对）。
3. **delta 语义**：`state_delta(s0, s1)` = (新增事实, 消失事实)。匹配要求候选
   状态的 delta **覆盖**参考 delta（ref.added ⊆ cand.added 且
   ref.removed ⊆ cand.removed），**容忍额外变化**——论文原文
   "s reproduces the task-relevant changes exhibited by s_ref relative to s0"，
   即 solver 多做无关动作不罚（这些动作只影响最终成败，不影响子任务命中）。
4. 观测型子任务（无持久状态变化，论文 §3.1）记 (动作, 返回值) 对，solver 必须
   复现；ALFWorld 场景基本用不到，但结构保留。

⚠️ 集群首次运行时必须用 `python -c` 验证 infos 里确实有 facts（见
scripts/ez_check_env_facts.py），没有就走快照模式整改 request_infos。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# --------------------------------------------------------------------------- 实例归一化

# "apple 1" -> "apple"，"fridge 1" -> "fridge"。ALFWorld 物体名总是 "<type> <n>"。
_INSTANCE_RE = re.compile(r"\b([a-zA-Z_]+)\s+\d+\b")


def normalize_object_name(name: str) -> str:
    """剥离实例编号，保留 object type。"""
    return _INSTANCE_RE.sub(r"\1", name).strip()


def normalize_fact(fact: Any) -> str:
    """把一条 fact（textworld Fact 对象或字符串）归一化成 "pred(arg, arg)" 字符串。

    所有参数都过 `normalize_object_name`，因此 "in(apple 1, fridge 1)" 与
    "in(apple 2, fridge 2)" 归一化后**相同**——这正是论文"忽略实例编号"的要求。
    """
    if not isinstance(fact, str):
        fact = str(fact)
    # fact 形如 "in(apple 1, fridge 1)" 或裸谓词；只对括号内参数做归一化
    m = re.match(r"^([a-zA-Z_][\w-]*)\((.*)\)$", fact.strip())
    if not m:
        return normalize_object_name(fact)
    pred, args = m.group(1), m.group(2)
    norm_args = ", ".join(normalize_object_name(a) for a in args.split(","))
    return f"{pred}({norm_args})"


# --------------------------------------------------------------------------- 状态快照


@dataclass
class StateSnapshot:
    """某一时刻的环境状态。

    `facts`: 归一化后的谓词集合（首选，真值来自引擎状态机）。
    `text_sig`: 观测文本哈希（fallback；facts 不可用时的降级模式）。
    """

    facts: frozenset = frozenset()
    text_sig: str = ""
    mode: str = "facts"           # "facts" | "text_fallback"

    def to_json(self) -> dict:
        return {"facts": sorted(self.facts), "text_sig": self.text_sig, "mode": self.mode}

    @staticmethod
    def from_json(d: dict) -> "StateSnapshot":
        return StateSnapshot(facts=frozenset(d.get("facts", [])),
                             text_sig=d.get("text_sig", ""),
                             mode=d.get("mode", "facts"))


def snapshot_from_infos(infos_slot: Dict[str, Any], obs_text: str = "") -> StateSnapshot:
    """从 textworld 单 slot 的 infos 构造快照。

    `infos_slot` 是 batch infos 按 slot 取出来的 dict（该 slot 的
    admissible_commands/won/facts 等）。facts 可能是 list[str] 或 list[Fact]。
    """
    facts_raw = infos_slot.get("facts") if isinstance(infos_slot, dict) else None
    if facts_raw:
        return StateSnapshot(facts=frozenset(normalize_fact(f) for f in facts_raw),
                             mode="facts")
    sig = hashlib.sha256(obs_text.encode("utf-8", errors="ignore")).hexdigest()[:32]
    return StateSnapshot(facts=frozenset(), text_sig=sig, mode="text_fallback")


def snapshot_from_obs_text(obs_text: str) -> StateSnapshot:
    """纯观测 fallback（无 facts 通道时调试用）。"""
    sig = hashlib.sha256(obs_text.encode("utf-8", errors="ignore")).hexdigest()[:32]
    return StateSnapshot(facts=frozenset(), text_sig=sig, mode="text_fallback")


# --------------------------------------------------------------------------- delta 与匹配


@dataclass
class StateDelta:
    """相对 s0 的变化：新增事实 + 消失事实（均已归一化）。"""

    added: frozenset = frozenset()
    removed: frozenset = frozenset()

    def is_empty(self) -> bool:
        return not self.added and not self.removed

    def to_json(self) -> dict:
        return {"added": sorted(self.added), "removed": sorted(self.removed)}

    @staticmethod
    def from_json(d: dict) -> "StateDelta":
        return StateDelta(added=frozenset(d.get("added", [])),
                          removed=frozenset(d.get("removed", [])))


def state_delta(s0: StateSnapshot, s1: StateSnapshot) -> StateDelta:
    if s0.mode == "text_fallback" or s1.mode == "text_fallback":
        # 降级模式：把"不同"本身编码成 added 里的一条伪事实
        if s0.text_sig == s1.text_sig:
            return StateDelta()
        return StateDelta(added=frozenset([f"__obs_changed__:{s1.text_sig}"]))
    return StateDelta(added=frozenset(s1.facts - s0.facts),
                      removed=frozenset(s0.facts - s1.facts))


def delta_covers(ref: StateDelta, cand: StateDelta) -> bool:
    """候选 delta 是否覆盖参考 delta（多退少补原则：多做的动作不罚，少的必须补）。"""
    return ref.added.issubset(cand.added) and ref.removed.issubset(cand.removed)


def deltas_equal(a: StateDelta, b: StateDelta) -> bool:
    return a.added == b.added and a.removed == b.removed


def match_reference(s0: StateSnapshot,
                    ref_snapshot: StateSnapshot,
                    candidate_snapshots: Sequence[StateSnapshot]) -> bool:
    """Eq.6 的核心：solver 轨迹中**任一历史状态**复现参考状态即算命中。

    `candidate_snapshots` 是 solver 从 s0 出发沿途的每个状态（含 s0 与终态）。
    """
    ref_delta = state_delta(s0, ref_snapshot)
    if ref_delta.is_empty():
        # 参考状态相对 s0 无变化（观测型子任务不该走到这）；保守判 False，
        # 让上层用 observational artifact 处理。
        return False
    for cand in candidate_snapshots:
        if cand.mode != ref_snapshot.mode:
            continue
        if delta_covers(ref_delta, state_delta(s0, cand)):
            return True
    return False


def match_final_state(s0: StateSnapshot, ref_final: StateSnapshot, sT: StateSnapshot) -> bool:
    """Eq.5：终态是否复现最终参考状态（终态也走覆盖语义——论文 Eq.5 是状态比对，
    覆盖语义对 solver 更宽容，与 §3.1 的 delta 描述一致）。

    ⚠️ 空 ref delta（proposer 开局就 finish、全程无状态变化）恒为 True 的退化路径
    在这里直接掐死：空 delta 的"任务"没有任何信息，返回 False（上层过滤也应拒）。
    """
    if sT.mode != ref_final.mode:
        return False
    ref_delta = state_delta(s0, ref_final)
    if ref_delta.is_empty():
        return False
    return delta_covers(ref_delta, state_delta(s0, sT))


def assert_facts_mode(snap: StateSnapshot, where: str) -> None:
    """EZ 路径硬断言：状态快照必须来自 textworld facts。

    facts 通道缺失时快照会静默降级成观测哈希（text_fallback），那种模式下
    match_reference 因 mode 不匹配全部判 False —— solver 奖励恒 0、p̂ 恒 0，
    而日志只会显示 reward mean=0。把静默降级变成显式失败（review P1-3）。
    """
    if snap.mode != "facts":
        raise RuntimeError(
            f"[{where}] 状态快照不是 facts 模式（textworld facts 通道不可用）。"
            "先跑 scripts/ez_check_env_facts.py 确认 facts，EZ 拒绝在降级模式下训练。")
