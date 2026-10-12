#!/usr/bin/env python
"""Explorer-Zero 离线单测：状态匹配 / 奖励 / 过滤 / 工具解析。

纯离线：不需要 GPU、不需要 ALFWorld 数据、不需要模型。
运行：python scripts/ez_unit_test.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ez_proposer import parse_tool_call                     # noqa: E402
from ez_state import (StateDelta, StateSnapshot, delta_covers,  # noqa: E402
                      match_final_state, match_reference, normalize_fact,
                      snapshot_from_obs_text, state_delta)
from ez_verify import (ProposedTask, SubtaskRef, action_type,  # noqa: E402
                       proposer_reward, redundant_pair, validity_filter,
                       verify_solver_rollout)
from rollout import _ToolCallStopCriteria                   # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [ok] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {extra}")


def snap(facts) -> StateSnapshot:
    return StateSnapshot(facts=frozenset(normalize_fact(f) for f in facts), mode="facts")


print("== 实例归一化 ==")
check("apple 1 -> apple", normalize_fact("in(apple 1, fridge 1)") == "in(apple, fridge)")
check("apple 1 与 apple 2 归一后相同",
      normalize_fact("on(apple 1, table 1)") == normalize_fact("on(apple 2, table 2)"))
check("多参数", normalize_fact("cooked(apple 1)") == "cooked(apple)")

print("== delta 与覆盖匹配 ==")
S0 = snap(["at(agent, kitchen)", "closed(fridge 1)", "in(apple 1, fridge 1)"])
S1 = snap(["at(agent, kitchen)", "open(fridge 1)", "in(apple 1, fridge 1)"])   # 开了冰箱
S2 = snap(["at(agent, kitchen)", "open(fridge 1)", "held(apple 1)"])          # 拿了苹果
S3 = snap(["at(agent, kitchen)", "open(fridge 1)", "held(apple 1)",
           "sliced(apple 1)"])                                                  # 多切了一刀

d_open = state_delta(S0, S1)
check("开冰箱 delta=removed closed", d_open.removed == frozenset({"closed(fridge)"}))
check("开冰箱 delta=added open", d_open.added == frozenset({"open(fridge)"}))
check("覆盖：S2 包含 open 变化", delta_covers(d_open, state_delta(S0, S2)))
check("不覆盖：S1 不含 take 变化",
      not delta_covers(state_delta(S0, S2), d_open))
check("容忍额外变化：S3 覆盖 S2 的 delta",
      delta_covers(state_delta(S0, S2), state_delta(S0, S3)))

print("== Eq.5/6/7 solver 奖励 ==")
# 终态 = 开着冰箱**且拿着苹果**（与 S1 仅开冰箱区分开，否则"部分完成"场景不成立）
ref_final = snap(["at(agent, kitchen)", "open(fridge 1)", "held(apple 1)"])
task = ProposedTask(
    task_desc="Put an apple in the fridge.",
    s0=S0, ref_final=ref_final,
    ref_final_delta=state_delta(S0, ref_final),
    subtasks=[SubtaskRef(description="Open the fridge.",
                         ref_snapshot=S1, ref_delta=state_delta(S0, S1))],
    proposer_actions=["open fridge 1", "take apple 1 from fridge 1"],
    proposer_action_types=["open", "take"],
)

# 全成功：终态命中 + 子任务命中 -> reward=1
full = verify_solver_rollout(task, [S0, S1, ref_final])
check("成功轨迹 reward=1", full.reward == 1.0 and full.final_match)
# 失败但有进展：开了冰箱没放进 -> reward=子任务率=1.0（唯一子任务完成）
partial = verify_solver_rollout(task, [S0, S1])
check("部分完成 reward=子任务率", (not partial.final_match) and partial.reward == 1.0
      and partial.subtask_hits == [True])
# 完全失败
none_ = verify_solver_rollout(task, [S0, S0])
check("完全失败 reward=0", none_.reward == 0.0 and not any(none_.subtask_hits))
# 无序清单：后完成子任务也算（历史状态命中）
reordered = verify_solver_rollout(task, [S0, ref_final, S1])
check("历史状态命中子任务", reordered.subtask_hits == [True])
# 实例替换：solver 用 apple 2（归一化后等价）
S1_b = snap(["at(agent, kitchen)", "open(fridge 1)", "in(apple 2, fridge 2)"])
S2_b = snap(["at(agent, kitchen)", "open(fridge 1)", "in(apple 2, fridge 2)"])
swap = verify_solver_rollout(task, [S0, S1_b, S2_b])
check("实例替换等价（apple 2 也算开冰箱）", swap.subtask_hits == [True])
# 无子任务任务：失败即 0
task_nosub = ProposedTask(task_desc="x", s0=S0, ref_final=ref_final,
                          ref_final_delta=state_delta(S0, ref_final))
check("无子任务失败=0", verify_solver_rollout(task_nosub, [S0, S1]).reward == 0.0)
check("无子任务成功=1", verify_solver_rollout(task_nosub, [S0, ref_final]).reward == 1.0)

print("== 冗余子任务判定（delta 同 + Jaccard>0.6）==")
d_a = state_delta(S0, S1)
check("完全相同描述+同delta -> 冗余",
      redundant_pair("Open the fridge.", d_a, "Open the fridge.", d_a))
check("不同delta 不冗余",
      not redundant_pair("Open the fridge.", d_a, "Open the fridge.", state_delta(S0, S2)))
check("delta 同但描述不同 -> 不冗余",
      not redundant_pair("Open the fridge.", d_a, "Unlock the cabinet.", d_a))

print("== 有效性过滤三规则 ==")
dup = ProposedTask(task_desc="Do two things.", s0=S0, ref_final=S2,
                   ref_final_delta=state_delta(S0, S2),
                   subtasks=[SubtaskRef("Open the fridge.", S1, d_a),
                             SubtaskRef("Open the fridge.", S1, d_a)],
                   proposer_actions=["open fridge 1"])
ok, reason = validity_filter(dup)
check("冗余子任务被拒", not ok and reason.startswith("redundant_subtask"), reason)
empty = ProposedTask(task_desc="", s0=S0, ref_final=S2,
                     ref_final_delta=state_delta(S0, S2))
ok, _ = validity_filter(empty)
check("空任务描述被拒", not ok)
# review P1-2：终态相对 s0 无变化（开局就 finish / 全程只 look）必须被拒，
# 否则 Eq.5 空 delta 匹配恒真 → solver 奖励恒 1、p̂ 恒 1，整条线退化。
nochange = ProposedTask(task_desc="Look around.", s0=S0, ref_final=S0,
                        ref_final_delta=state_delta(S0, S0),
                        proposer_actions=["look", "look"])
ok, reason = validity_filter(nochange)
check("空 delta 任务被拒", not ok and reason == "empty_final_delta", reason)
good = ProposedTask(task_desc="Open the fridge.", s0=S0, ref_final=S1,
                    ref_final_delta=state_delta(S0, S1),
                    proposer_actions=["open fridge 1"])
ok, reason = validity_filter(good)
check("正常任务通过", ok, reason)

print("== proposer 奖励 Eq.1-4 ==")
r, parts = proposer_reward(task, solve_rate=0.5)
check("p̂=0.5 难度=1", abs(parts["r_difficulty"] - 1.0) < 1e-9)
r2, parts2 = proposer_reward(task, solve_rate=0.0)
check("p̂=0 难度=0", abs(parts2["r_difficulty"] - 0.0) < 1e-9)
check("子任务项 = w_sub*K", abs(parts["r_subtask"] - 0.3 * 1) < 1e-9)
check("多样性项 = w_div*|动作类型|", abs(parts["r_diversity"] - 0.3 * 2) < 1e-9)
check("动作类型提取", action_type("open fridge 1") == "open")

print("== 工具调用解析 ==")
p = parse_tool_call('blah <tool_call>{"name":"alfworld_tool","arguments":{"action_type":"go_to","receptacle_name":"kitchen 1"}}</tool_call>')
check("go_to 解析", p.kind == "action" and p.action == "go to kitchen 1", p.raw)
p = parse_tool_call('<tool_call>{"name":"alfworld_tool","arguments":{"action_type":"heat","object_name":"apple 1","receptacle_name":"microwave 1"}}</tool_call>')
check("heat 解析", p.kind == "action" and p.action == "heat apple 1 with microwave 1", p.raw)
p = parse_tool_call('<tool_call>{"name":"alfworld_tool","arguments":{"action_type":"sub_finish","task_description":"Heat an apple."}}</tool_call>')
check("sub_finish 解析", p.kind == "sub_finish" and p.task_description == "Heat an apple.")
p = parse_tool_call('<tool_call>{"name":"alfworld_tool","arguments":{"action_type":"finish","task_description":"Heat an apple and put it on the countertop."}}</tool_call>')
check("finish 解析", p.kind == "finish" and "countertop" in p.task_description)
p = parse_tool_call('<tool_call>{"action_type":"look"}</tool_call>')
check("裸 JSON 兼容", p.kind == "action" and p.action == "look")
p = parse_tool_call("没有任何工具调用")
check("无工具调用 -> malformed", p.kind == "malformed")
p = parse_tool_call('<tool_call>{"name":"alfworld_tool","arguments":{"action_type":"fly"}}</tool_call>')
check("未知动作 -> malformed", p.kind == "malformed")
p = parse_tool_call('<tool_call>{"name":"alfworld_tool","arguments":{"action_type":"take","object_name":"apple 1"}}</tool_call>')
check("缺参数 -> malformed", p.kind == "malformed")

print("== tool_call 停止条件（EZ 专用）==")


class _FakeTok:
    """把 token id 直接当字符解码：id n -> 词表第 n 个字符。"""
    CHARS = ' abcdefghijklmnopqrstuvwxyz<>/{}_-'

    def __init__(self):
        self.chars = self.CHARS

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.chars[i] if 0 <= i < len(self.chars) else "?" for i in ids)


class _FakeTensor:
    """只实现 stopping criteria 用到的 shape / 二维切片。"""
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]) if rows else 0)

    def __getitem__(self, key):
        i, sl = key
        return self.rows[i][sl]


def _encode(tok: _FakeTok, text: str):
    return [tok.chars.index(c) for c in text]


tok = _FakeTok()
crit = _ToolCallStopCriteria(tok, [0], max_new_tokens=200)
partial = _encode(tok, "<tool_call>{a}")
done = _encode(tok, "<tool_call>{a}</tool_call>")
check("未闭合不停", crit(_FakeTensor([partial]), None) is False)
check("闭合后停", crit(_FakeTensor([done]), None) is True)
check("stop_len 记录", crit.stop_len(0, 999) == len(done))
crit2 = _ToolCallStopCriteria(tok, [0], max_new_tokens=200)
check("从没出现标签 -> 永不触发", crit2(_FakeTensor([partial]), None) is False
      and crit2.stop_len(0, 999) == 999)

print("== text fallback 降级模式 ==")
fa = snapshot_from_obs_text("You are in the kitchen.")
fb = snapshot_from_obs_text("You are in the kitchen.")
fc = snapshot_from_obs_text("You are in the bathroom.")
check("fallback 相同观测无变化", state_delta(fa, fb).is_empty())
check("fallback 不同观测有变化", not state_delta(fa, fc).is_empty())
check("fallback 匹配", match_reference(fa, fc, [fa, fc]))

print(f"\n{'全部通过 ✅' if FAIL == 0 else f'失败 {FAIL} 项 ❌'} ({PASS} passed)")
sys.exit(1 if FAIL else 0)
