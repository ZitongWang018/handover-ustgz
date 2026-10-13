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
import ast
import re
from collections import Counter
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
        # TextWorld Proposition keeps predicate and typed arguments separately.
        # Preserve the type when available; stringifying the object loses this.
        pred = getattr(fact, "name", None)
        args = getattr(fact, "arguments", None)
        if pred is not None and args is not None:
            vals = []
            for arg in args:
                name = getattr(arg, "name", str(arg))
                typ = getattr(arg, "type", None)
                typ_name = getattr(typ, "name", typ)
                name = normalize_object_name(str(name))
                vals.append(f"{name}:{typ_name}" if typ_name else name)
            return f"{pred}({', '.join(vals)})"
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

    # tuple retains duplicate normalized facts (e.g. two apples in a room).
    # Older JSON set/list records remain readable.
    facts: tuple = ()
    text_sig: str = ""
    mode: str = "facts"           # "facts" | "text_fallback"
    raw_facts: tuple = ()

    def to_json(self) -> dict:
        return {"facts": sorted(self.facts), "raw_facts": list(self.raw_facts),
                "text_sig": self.text_sig, "mode": self.mode}

    @staticmethod
    def from_json(d: dict) -> "StateSnapshot":
        return StateSnapshot(facts=tuple(d.get("facts", [])),
                             text_sig=d.get("text_sig", ""),
                             mode=d.get("mode", "facts"),
                             raw_facts=tuple(d.get("raw_facts", ())))


def snapshot_from_infos(infos_slot: Dict[str, Any], obs_text: str = "") -> StateSnapshot:
    """从 textworld 单 slot 的 infos 构造快照。

    `infos_slot` 是 batch infos 按 slot 取出来的 dict（该 slot 的
    admissible_commands/won/facts 等）。facts 可能是 list[str] 或 list[Fact]。
    """
    facts_raw = infos_slot.get("facts") if isinstance(infos_slot, dict) else None
    if facts_raw:
        return StateSnapshot(facts=tuple(normalize_fact(f) for f in facts_raw),
                             mode="facts", raw_facts=tuple(_raw_fact(f) for f in facts_raw))
    sig = hashlib.sha256(obs_text.encode("utf-8", errors="ignore")).hexdigest()[:32]
    return StateSnapshot(facts=frozenset(), text_sig=sig, mode="text_fallback")


def snapshot_from_obs_text(obs_text: str) -> StateSnapshot:
    """纯观测 fallback（无 facts 通道时调试用）。"""
    sig = hashlib.sha256(obs_text.encode("utf-8", errors="ignore")).hexdigest()[:32]
    return StateSnapshot(facts=frozenset(), text_sig=sig, mode="text_fallback")


def _raw_fact(fact: Any) -> str:
    """A JSON-safe structural rendering retained for auditing the source facts."""
    pred = getattr(fact, "name", None)
    args = getattr(fact, "arguments", None)
    if pred is not None and args is not None:
        parts = []
        for arg in args:
            typ = getattr(arg, "type", None)
            parts.append({"name": str(getattr(arg, "name", arg)),
                          "type": str(getattr(typ, "name", typ)) if typ else ""})
        return repr({"predicate": str(pred), "arguments": parts})
    return str(fact)


# --------------------------------------------------------------------------- delta 与匹配


@dataclass
class StateDelta:
    """相对 s0 的变化：新增事实 + 消失事实（均已归一化）。"""

    added: tuple = ()
    removed: tuple = ()
    # Exact-instance evidence is retained to avoid composing properties from two
    # same-type objects into one task (e.g. heat apple 1, then move apple 2).
    raw_added: tuple = ()
    raw_removed: tuple = ()

    def is_empty(self) -> bool:
        return not self.added and not self.removed

    def to_json(self) -> dict:
        return {"added": sorted(self.added), "removed": sorted(self.removed),
                "raw_added": sorted(self.raw_added), "raw_removed": sorted(self.raw_removed)}

    @staticmethod
    def from_json(d: dict) -> "StateDelta":
        return StateDelta(added=tuple(d.get("added", [])),
                          removed=tuple(d.get("removed", [])),
                          raw_added=tuple(d.get("raw_added", [])),
                          raw_removed=tuple(d.get("raw_removed", [])))


def _project_facts(facts):
    """Ignore pure player navigation while retaining task-object predicates."""
    out = []
    for fact in facts:
        m = re.match(r"^([a-zA-Z_][\w-]*)\((.*)\)$", fact)
        if m and m.group(1).lower() in {"at", "in", "atlocation"}:
            first = m.group(2).split(",", 1)[0].strip().lower()
            if re.match(r"^(?:player|agent|p)(?:\b|:|\s|$)", first):
                continue
        raw = _parse_raw_fact(fact)
        if raw and raw[0].lower() in {"at", "in", "atlocation"} and raw[1]:
            first = raw[1][0][0].lower()
            if re.match(r"^(?:player|agent|p)(?:\b|:|\s|$)", first):
                continue
        out.append(fact)
    return out


def _task_scope_types(task_desc: Optional[str]) -> Optional[set]:
    """Known ALFWorld entity types named by task text; None means conservative legacy scope."""
    if not task_desc:
        return None
    try:
        from ez_verify import _ALFWORLD_TYPES
    except ImportError:
        return None
    words = re.findall(r"[a-z]+", task_desc.lower())
    return {w[:-1] if w.endswith("s") and w[:-1] in _ALFWORLD_TYPES else w
            for w in words if (w in _ALFWORLD_TYPES or
                               (w.endswith("s") and w[:-1] in _ALFWORLD_TYPES))}


_SPATIAL_PREDICATES = {"in", "inreceptacle", "at", "atlocation", "on", "onreceptacle",
                       "objectatlocation", "receptacleatlocation"}
_CONTAINER_STATE_PREDICATES = {"open", "closed", "isopen", "isclosed", "opened", "not_opened"}


def _scope_fact(fact: Any, target_types: Optional[set], task_desc: str) -> bool:
    """Conservative task-relevance projection for known ALFWorld entities.

    Spatial relations count only when both endpoints are named by the task. This
    avoids turning an incidental source location into a hidden requirement. Container
    state changes count only when that container is explicitly an open/close target.
    Unknown predicates remain in scope when they touch a named task entity.
    """
    if target_types is None or not target_types:
        return True
    parsed = _parse_raw_fact(fact) if isinstance(fact, str) else None
    if parsed is None:
        m = re.match(r"^([a-zA-Z_][\w-]*)\((.*)\)$", str(fact))
        if not m:
            return True
        pred = m.group(1).lower()
        names = [p.strip().split(":", 1)[0] for p in m.group(2).split(",")]
    else:
        pred, args = parsed
        pred = pred.lower()
        names = [a[0].split(":", 1)[0].strip() for a in args]
    try:
        from ez_verify import _ALFWORLD_TYPES
    except ImportError:
        return True
    types = []
    for name in names:
        norm = normalize_object_name(name).lower().replace(" ", "")
        types.append(norm if norm in _ALFWORLD_TYPES else None)
    if pred in _CONTAINER_STATE_PREDICATES:
        explicit_targets = set()
        for match in re.finditer(r"\b(?:open|close)\s+(?:(?:the|a|an)\s+)?([a-z]+)",
                                 task_desc.lower()):
            word = match.group(1)
            explicit_targets.add(word[:-1] if word.endswith("s") and word[:-1] in _ALFWORLD_TYPES
                                 else word)
        return any(t in explicit_targets for t in types if t)
    if pred in _SPATIAL_PREDICATES:
        relevant = [t for t in types if t]
        return bool(len(relevant) >= 2 and all(t in target_types for t in relevant))
    return any(t in target_types for t in types if t)


def _scoped(snapshot: StateSnapshot, task_desc: Optional[str]) -> StateSnapshot:
    targets = _task_scope_types(task_desc)
    if targets is None or not targets:
        return snapshot
    facts = tuple(f for f in snapshot.facts if _scope_fact(f, targets, task_desc or ""))
    raw = tuple(f for f in snapshot.raw_facts if _scope_fact(f, targets, task_desc or ""))
    return StateSnapshot(facts=facts, raw_facts=raw, text_sig=snapshot.text_sig, mode=snapshot.mode)


def _parse_raw_fact(value):
    """Return (predicate, [(name, type), ...]) for stored structural/raw facts."""
    try:
        obj = ast.literal_eval(value) if isinstance(value, str) else value
    except (ValueError, SyntaxError):
        obj = None
    if isinstance(obj, dict) and "predicate" in obj:
        return str(obj["predicate"]), [
            (str(a.get("name", "")), str(a.get("type", "")))
            for a in obj.get("arguments", [])]
    if isinstance(value, str):
        m = re.match(r"^([a-zA-Z_][\w-]*)\((.*)\)$", value.strip())
        if m:
            args = []
            for part in m.group(2).split(","):
                name = part.strip()
                args.append((name, normalize_object_name(name)))
            return m.group(1), args
    return None


def _raw_signature(parsed):
    pred, args = parsed
    # TextWorld's engine types are often only the broad ``o``/``r`` classes.
    # The normalized entity name supplies the ALFWorld object/receptacle type.
    return pred, tuple((normalize_object_name(name).lower(), typ.lower())
                       for name, typ in args)


def _raw_covers(ref: StateDelta, cand: StateDelta) -> bool:
    """Match exact changed predicates under one injective entity-renaming map."""
    if not ref.raw_added and not ref.raw_removed:
        return True  # legacy pool record: normalized-multiset check only
    ref_rows = [("+", x) for x in ref.raw_added] + [("-", x) for x in ref.raw_removed]
    cand_rows = [("+", x) for x in cand.raw_added] + [("-", x) for x in cand.raw_removed]
    parsed_ref = [(sign, raw, _parse_raw_fact(raw)) for sign, raw in ref_rows]
    parsed_cand = [(sign, raw, _parse_raw_fact(raw)) for sign, raw in cand_rows]
    if any(parsed is None for _sign, _raw, parsed in parsed_ref + parsed_cand):
        raise ValueError("unparseable raw TextWorld fact in a schema-2 delta")
    options = []
    for sign, _raw, parsed in parsed_ref:
        sig = _raw_signature(parsed)
        matches = [i for i, (csign, _craw, cparsed) in enumerate(parsed_cand)
                   if csign == sign and _raw_signature(cparsed) == sig]
        if not matches:
            return False
        options.append(matches)
    order = sorted(range(len(options)), key=lambda i: len(options[i]))

    visited = 0
    def search(pos, used, mapping, reverse):
        nonlocal visited
        visited += 1
        if visited > 100_000:
            raise RuntimeError("raw-fact entity matcher exceeded 100000 search states")
        if pos == len(order):
            return True
        ri = order[pos]
        _sign, _raw, rparsed = parsed_ref[ri]
        _rp, rargs = rparsed
        for ci in options[ri]:
            if ci in used:
                continue
            _cs, _cr, cparsed = parsed_cand[ci]
            cargs = cparsed[1]
            newmap, newrev = dict(mapping), dict(reverse)
            valid = True
            for (rname, rtype), (cname, ctype) in zip(rargs, cargs):
                # Keep constants/literals exact; entities use their typed class.
                entity = bool(rtype or re.search(r"\s+\d+$", rname))
                if not entity:
                    if rname != cname:
                        valid = False
                        break
                    continue
                if normalize_object_name(rname).lower() != normalize_object_name(cname).lower():
                    valid = False
                    break
                if rtype and ctype and rtype != ctype:
                    valid = False
                    break
                if rname in newmap and newmap[rname] != cname:
                    valid = False
                    break
                if cname in newrev and newrev[cname] != rname:
                    valid = False
                    break
                newmap[rname], newrev[cname] = cname, rname
            if valid and search(pos + 1, used | {ci}, newmap, newrev):
                return True
        return False

    return search(0, set(), {}, {})


def state_delta(s0: StateSnapshot, s1: StateSnapshot, task_desc: Optional[str] = None) -> StateDelta:
    s0, s1 = _scoped(s0, task_desc), _scoped(s1, task_desc)
    if s0.mode == "text_fallback" or s1.mode == "text_fallback":
        # 降级模式：把"不同"本身编码成 added 里的一条伪事实
        if s0.text_sig == s1.text_sig:
            return StateDelta()
        return StateDelta(added=frozenset([f"__obs_changed__:{s1.text_sig}"]))
    c0, c1 = Counter(_project_facts(s0.facts)), Counter(_project_facts(s1.facts))
    raw0, raw1 = Counter(_project_facts(s0.raw_facts)), Counter(_project_facts(s1.raw_facts))
    return StateDelta(added=tuple((c1 - c0).elements()),
                      removed=tuple((c0 - c1).elements()),
                      raw_added=tuple((raw1 - raw0).elements()),
                      raw_removed=tuple((raw0 - raw1).elements()))


def delta_covers(ref: StateDelta, cand: StateDelta) -> bool:
    """候选 delta 是否覆盖参考 delta（多退少补原则：多做的动作不罚，少的必须补）。"""
    return (not (Counter(ref.added) - Counter(cand.added)) and
            not (Counter(ref.removed) - Counter(cand.removed)) and
            _raw_covers(ref, cand))


def deltas_equal(a: StateDelta, b: StateDelta) -> bool:
    normalized_equal = (Counter(a.added) == Counter(b.added) and
                        Counter(a.removed) == Counter(b.removed))
    if not normalized_equal:
        return False
    # Old artifacts have no instance-level source facts; preserve their historic
    # normalized comparison. New artifacts compare isomorphic typed entity graphs.
    if not (a.raw_added or a.raw_removed) or not (b.raw_added or b.raw_removed):
        return True
    return _raw_covers(a, b) and _raw_covers(b, a)


def match_reference(s0: StateSnapshot,
                    ref_snapshot: StateSnapshot,
                    candidate_snapshots: Sequence[StateSnapshot],
                    task_desc: Optional[str] = None) -> bool:
    """Eq.6 的核心：solver 轨迹中**任一历史状态**复现参考状态即算命中。

    `candidate_snapshots` 是 solver 从 s0 出发沿途的每个状态（含 s0 与终态）。
    """
    ref_delta = state_delta(s0, ref_snapshot, task_desc)
    if ref_delta.is_empty():
        # 参考状态相对 s0 无变化（观测型子任务不该走到这）；保守判 False，
        # 让上层用 observational artifact 处理。
        return False
    for cand in candidate_snapshots:
        if cand.mode != ref_snapshot.mode:
            continue
        if delta_covers(ref_delta, state_delta(s0, cand, task_desc)):
            return True
    return False


def match_final_state(s0: StateSnapshot, ref_final: StateSnapshot, sT: StateSnapshot,
                      task_desc: Optional[str] = None) -> bool:
    """Eq.5：终态是否复现最终参考状态（终态也走覆盖语义——论文 Eq.5 是状态比对，
    覆盖语义对 solver 更宽容，与 §3.1 的 delta 描述一致）。

    ⚠️ 空 ref delta（proposer 开局就 finish、全程无状态变化）恒为 True 的退化路径
    在这里直接掐死：空 delta 的"任务"没有任何信息，返回 False（上层过滤也应拒）。
    """
    if sT.mode != ref_final.mode:
        return False
    ref_delta = state_delta(s0, ref_final, task_desc)
    if ref_delta.is_empty():
        return False
    return delta_covers(ref_delta, state_delta(s0, sT, task_desc))


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
