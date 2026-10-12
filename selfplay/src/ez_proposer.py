"""Explorer-Zero proposer：探索环境、声明子任务、产出 ProposedTask（论文 §3.1/Algorithm 1 左半）。

流程（每个 GRPO 组共享同一初始状态 s0，论文 Appendix D）：
1. reset 到同一 game_file × G 个 slot；
2. 每轮 generate → 解析 <tool_call>：
   - 普通动作 → env.step，观测/快照进消息历史；
   - `sub_finish` → **不执行环境动作**，把当前状态存为子任务参考快照，继续探索；
   - `finish` → 存终态快照，结束该 slot 的探索；
3. 组内 G 条探索轨迹 → G 个候选提案 → validity_filter → 有效提案进任务池。

⚠️ base 模型（Qwen3-4B 非 instruct）遵循格式的能力弱（U7），parse 失败率单独统计。
格式违例的惩罚是**间接**的：违例轨迹大概率走不到 finish → 提案无效 → 奖励 0
（没有独立的 format penalty 项，与论文 Eq.1 一致）。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ez_prompts import append_tool_result, build_proposer_messages
from ez_state import (StateDelta, StateSnapshot, assert_facts_mode, snapshot_from_infos,
                      state_delta)
from ez_verify import (ProposedTask, SubtaskRef, action_type, validity_filter)

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


@dataclass
class ProposerConfig:
    group_size: int = 16            # G（Table 3）
    max_turns: int = 30             # T_P（Table 3）
    max_new_tokens: int = 160       # 补全: 附录未给，探索动作+理由预留比 solver(96) 多
    temperature: float = 1.0
    top_p: float = 0.95
    repetition_penalty: float = 1.05
    max_context_tokens: int = 40960  # Table 3 context length
    keep_recent_turns: int = 40     # 补全: 40960 token 大约能留 40 轮
    give_admissible: bool = False   # U3 决策点：默认不给 admissible_commands
    seed: Optional[int] = None


@dataclass
class ProposerRollout:
    """一条 proposer 探索轨迹的结果（= GRPO 里的一条'轨迹'）。"""

    task: Optional[ProposedTask]     # finish 成功且过滤通过才非 None
    accepted: bool
    reject_reason: str
    messages: List[Dict[str, str]]
    gen_ids: List[List[int]]
    n_turns: int = 0
    n_format_violation: int = 0
    n_invalid_action: int = 0
    # —— GRPO 奖励（外层 ez_train 填入 solve-rate 后算 Eq.1）——
    reward: float = 0.0

    def summary(self) -> Dict[str, Any]:
        return {"accepted": self.accepted, "reason": self.reject_reason,
                "turns": self.n_turns, "fmt_bad": self.n_format_violation,
                "invalid": self.n_invalid_action,
                "task": self.task.task_desc if self.task else ""}


# --------------------------------------------------------------------------- 工具调用解析

_ACTION_TYPE_COMMANDS = {
    "go_to": "go to {receptacle_name}",
    "take": "take {object_name} from {receptacle_name}",
    "put": "put {object_name} in {receptacle_name}",
    "open": "open {receptacle_name}",
    "close": "close {receptacle_name}",
    "look": "look",
    "inventory": "inventory",
    "examine": "examine {object_name}",
    "use": "use {object_name}",
    "heat": "heat {object_name} with {receptacle_name}",
    "clean": "clean {object_name} with {receptacle_name}",
    "cool": "cool {object_name} with {receptacle_name}",
    "slice": "slice {object_name} with {receptacle_name}",
    "move": "move {object_name} to {receptacle_name}",
}


@dataclass
class ParsedToolCall:
    kind: str                                    # "action" | "sub_finish" | "finish" | "malformed"
    action: str = ""                             # kind=action 时的环境命令
    task_description: str = ""                   # kind=sub_finish/finish 时的描述
    raw: str = ""


def parse_tool_call(text: str) -> ParsedToolCall:
    """从模型输出解析 <tool_call> JSON。

    兼容两类形态：附录 G 的 `{"name":"alfworld_tool","arguments":{...}}`
    与裸 `{"action_type": ...}`。
    """
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return ParsedToolCall(kind="malformed", raw=text[-200:])
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError:
        return ParsedToolCall(kind="malformed", raw=m.group(1)[:200])
    args = obj.get("arguments", obj)
    if not isinstance(args, dict):
        return ParsedToolCall(kind="malformed", raw=m.group(1)[:200])
    at = str(args.get("action_type", "")).strip().lower()
    desc = str(args.get("task_description", "")).strip()
    if at == "sub_finish":
        return ParsedToolCall(kind="sub_finish", task_description=desc, raw=m.group(1))
    if at == "finish":
        return ParsedToolCall(kind="finish", task_description=desc, raw=m.group(1))
    tpl = _ACTION_TYPE_COMMANDS.get(at)
    if tpl is None:
        return ParsedToolCall(kind="malformed", raw=m.group(1)[:200])
    try:
        action = tpl.format(**{k: str(v) for k, v in args.items() if k != "action_type"})
    except KeyError:
        return ParsedToolCall(kind="malformed", raw=m.group(1)[:200])
    action = " ".join(action.split())
    if "{" in action or "}" in action:
        return ParsedToolCall(kind="malformed", raw=m.group(1)[:200])
    return ParsedToolCall(kind="action", action=action, raw=m.group(1))


def sanitize_explore_action(action: str, admissible: Sequence[str],
                            mode: str = "best_match") -> Tuple[str, bool]:
    """探索动作送给环境前的清洗。

    与 solver 的 `invalid_action_mode=raw` 不同：proposer 是出题人，动作失败只会
    浪费探索预算，所以默认 best_match 模糊对齐（论文未规定，复现决策点）。
    """
    if not admissible:
        return action, True
    low = action.strip().lower()
    for a in admissible:
        if a.strip().lower() == low:
            return a.strip(), True
    if mode == "best_match":
        from prompts import match_admissible
        hit = match_admissible(action, admissible, fuzzy_threshold=0.6)
        if hit:
            return hit, True
    return action.strip(), False


# --------------------------------------------------------------------------- 探索 loop


def _slot_snapshot(facts_slot: Any, obs_text: str) -> StateSnapshot:
    infos = {"facts": facts_slot} if facts_slot is not None else {}
    return snapshot_from_infos(infos, obs_text)


def run_proposer_exploration(env, policy, game_file: str, cfg: ProposerConfig,
                             log_fn: Optional[Callable[[str], None]] = None) -> List[ProposerRollout]:
    """一个 GRPO 组的 proposer 探索：G 条轨迹共享初始状态 s0。

    `env.batch_size` 必须等于 cfg.group_size。
    返回 G 个 ProposerRollout（顺序 = slot 顺序）。
    """
    g = cfg.group_size
    if env.batch_size != g:
        raise ValueError(f"env.batch_size={env.batch_size} != group_size={g}")

    from rollout import RolloutConfig as _RC  # 借 HFPolicy.generate 的 cfg 形态

    gen_cfg = _RC(group_size=1, max_steps=cfg.max_turns, max_new_tokens=cfg.max_new_tokens,
                  temperature=cfg.temperature, top_p=cfg.top_p,
                  repetition_penalty=cfg.repetition_penalty,
                  max_context_tokens=cfg.max_context_tokens,
                  keep_recent_turns=cfg.keep_recent_turns,
                  stop_mode="tool_call")   # 写完 </tool_call> 即停，别白烧 token

    obs_list, views, admissible, used, _ = env.reset([game_file] * g)
    s0_facts = list(getattr(env, "last_reset_facts", [None] * g))
    s0_snaps = [_slot_snapshot(s0_facts[s], obs_list[s]) for s in range(g)]
    for s in range(g):
        assert_facts_mode(s0_snaps[s], f"proposer reset slot {s}")
    # 每个 slot 的"当前状态"沿轨迹更新（sub_finish/finish 快照取当前值）
    cur_snaps = list(s0_snaps)
    cur_facts = list(s0_facts)

    messages: List[List[Dict[str, str]]] = []
    gen_ids_log: List[List[List[int]]] = [[] for _ in range(g)]
    subtask_bufs: List[List[SubtaskRef]] = [[] for _ in range(g)]
    action_bufs: List[List[str]] = [[] for _ in range(g)]
    active = [True] * g
    stats = [{"fmt": 0, "inv": 0, "turns": 0} for _ in range(g)]

    for s in range(g):
        v = views[s]
        adm = admissible[s] if cfg.give_admissible else ()
        messages.append(build_proposer_messages(v.env_desc or v.raw, cfg.max_turns, adm))

    t0 = time.time()
    for turn in range(cfg.max_turns):
        if not any(active):
            break
        idxs = [s for s in range(g) if active[s]]
        texts, gen_ids, _ = policy.generate([messages[s] for s in idxs], gen_cfg)

        # ---- 解析 + 分类 ----
        env_actions = ["look"] * g
        todo: List[Tuple[int, ParsedToolCall, str]] = []
        for k, s in enumerate(idxs):
            parsed = parse_tool_call(texts[k])
            messages[s].append({"role": "assistant", "content": texts[k].strip()})
            gen_ids_log[s].append(list(gen_ids[k]))
            stats[s]["turns"] += 1

            if parsed.kind == "malformed":
                stats[s]["fmt"] += 1
                todo.append((s, parsed, ""))
            elif parsed.kind in ("sub_finish", "finish"):
                todo.append((s, parsed, ""))
            else:
                executed, exact = sanitize_explore_action(parsed.action, admissible[s])
                if not exact:
                    stats[s]["inv"] += 1
                action_bufs[s].append(executed)
                env_actions[s] = executed
                todo.append((s, parsed, executed))

        # ---- 先执行普通动作（sub_finish/finish 的 slot 发 look 空转）----
        need_step = any(t[1].kind == "action" for t in todo)
        bs = env.step(env_actions) if need_step else None
        if bs is not None:
            cur_facts = list(bs.facts) if bs.facts else cur_facts
            for s in range(g):
                if active[s]:
                    cur_snaps[s] = _slot_snapshot(
                        cur_facts[s] if s < len(cur_facts) else None, bs.obs[s])
                    assert_facts_mode(cur_snaps[s], f"proposer step slot {s}")
            # ★ 官方底层任务被误完成 → episode 提前结束：冻结该 slot。
            #   此刻还没 finish 的提案永远等不到了，直接按无效收尾（review P1-6）。
            for s in range(g):
                if active[s] and bool(bs.dones[s]):
                    stats[s]["_reason"] = "env_done_no_finish"
                    active[s] = False

        # ---- 处理声明 ----
        for s, parsed, executed in todo:
            if parsed.kind == "malformed":
                # 格式违例不杀轨迹：给反馈继续（惩罚走 GRPO 奖励）。
                # 反馈与 env.step 解耦：本轮没有 env step 也要让模型知道错了（review P2-4）。
                messages[s].append({"role": "user", "content":
                    "Your last response was not a valid tool call. Emit exactly one "
                    "<tool_call>{...}</tool_call> block."})
                continue
            if parsed.kind == "sub_finish":
                # 快照当前状态（子任务由**上一步动作**完成，此刻状态即参考状态）
                delta = state_delta(s0_snaps[s], cur_snaps[s])
                if not delta.is_empty():
                    subtask_bufs[s].append(SubtaskRef(
                        description=parsed.task_description,
                        ref_snapshot=cur_snaps[s], ref_delta=delta))
                    # 确认反馈同样与 env.step 解耦
                    append_tool_result(messages[s], "sub_finish",
                                       f"Recorded subtask: {parsed.task_description}",
                                       admissible[s] if cfg.give_admissible else ())
                else:
                    stats[s]["dropped_sub"] = stats[s].get("dropped_sub", 0) + 1
                    messages[s].append({"role": "user", "content":
                        "sub_finish ignored: the world state did not change. Declare a "
                        "subtask only right after a state-changing action."})
                continue
            if parsed.kind == "finish":
                final_snap = cur_snaps[s]
                task = ProposedTask(
                    task_desc=parsed.task_description,
                    s0=s0_snaps[s],
                    ref_final=final_snap,
                    ref_final_delta=state_delta(s0_snaps[s], final_snap),
                    subtasks=subtask_bufs[s],
                    game_file=used[s] if s < len(used) else game_file,
                    proposer_actions=action_bufs[s],
                    proposer_action_types=[action_type(a) for a in action_bufs[s]],
                    meta={"proposer_turns": stats[s]["turns"]},
                )
                ok, reason = validity_filter(task)
                stats[s]["_task"] = task
                stats[s]["_accepted"] = ok
                stats[s]["_reason"] = reason
                active[s] = False
                continue

        # ---- 普通动作的观测反馈 ----
        if bs is not None:
            for s, parsed, executed in todo:
                if parsed.kind == "action" and active[s]:
                    adm = bs.admissible[s] if cfg.give_admissible else ()
                    append_tool_result(messages[s], executed, bs.obs[s], adm)
                    admissible[s] = list(bs.admissible[s])

    # ---- 收尾：没 finish 的轨迹 = 无效提案（论文：无 finish 任务描述直接拒）----
    rollouts: List[ProposerRollout] = []
    for s in range(g):
        task = stats[s].get("_task")
        accepted = bool(stats[s].get("_accepted", False))
        reason = stats[s].get("_reason", "no_finish")
        rollouts.append(ProposerRollout(
            task=task,                     # 拒绝的也保留 task 供分析，但不进池
            accepted=accepted, reject_reason=reason,
            messages=messages[s], gen_ids=gen_ids_log[s],
            n_turns=stats[s]["turns"], n_format_violation=stats[s]["fmt"],
            n_invalid_action=stats[s]["inv"],
        ))

    if log_fn:
        acc = sum(1 for r in rollouts if r.accepted)
        dropped = sum(stats[s].get("dropped_sub", 0) for s in range(g))
        log_fn(f"[proposer] {acc}/{g} accepted | mean_turns="
               f"{np.mean([r.n_turns for r in rollouts]):.1f} | "
               f"fmt_bad={sum(r.n_format_violation for r in rollouts)} | "
               f"dropped_sub={dropped} | {time.time()-t0:.1f}s")
    return rollouts
