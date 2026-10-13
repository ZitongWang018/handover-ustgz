"""Explorer-Zero 的 prompt 模板（按论文附录 G 的 Fig.13/14 骨架复刻）。

不确定项声明（复现计划 U2）：论文 prompt 标注 "Details omitted for space"，
本文件在骨架基础上补全了细节，所有**补全决策**用「补全:」注释标出，
与骨架冲突处以骨架为准。
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

# ALFWorld 动作空间的公共说明段（附录 G Fig.13/14 的 "Common actions" 节）
_ACTION_SPACE_DOC = """Common actions. Each example shows only the `arguments` object --
always emit it wrapped as `{"name":"alfworld_tool","arguments":{...}}` inside
<tool_call></tool_call> tags:
- Go To: Move to a location. `{"action_type": "go_to", "receptacle_name": "table 1"}`
- Take: Pick up an object. `{"action_type": "take", "object_name": "apple 2", "receptacle_name": "counter 1"}`
- Look: Look around. `{"action_type": "look"}`
- Inventory: Check your inventory. `{"action_type": "inventory"}`
- Open: Open a receptacle. `{"action_type": "open", "receptacle_name": "fridge 1"}`
- Close: Close a receptacle. `{"action_type": "close", "receptacle_name": "fridge 1"}`
- Move: Move an object. `{"action_type": "move", "object_name": "mug 1", "receptacle_name": "counter 1"}`
- Examine: Examine an object. `{"action_type": "examine", "object_name": "apple 1"}`
- Use: Use an object. `{"action_type": "use", "object_name": "light 1"}`
- Heat: Heat an object. `{"action_type": "heat", "object_name": "potato 2", "receptacle_name": "microwave 1"}`
- Clean: Clean an object. `{"action_type": "clean", "object_name": "mug 1", "receptacle_name": "sinkbasin 1"}`
- Cool: Cool an object. `{"action_type": "cool", "object_name": "apple 1", "receptacle_name": "fridge 1"}`
- Slice: Slice an object. `{"action_type": "slice", "object_name": "apple 1", "receptacle_name": "knife 1"}`"""

PROPOSER_SYSTEM = """You are an AI agent exploring ALFWorld, a text-based household environment.
You are NOT given a task. First declare the task you intend to pursue with `declare_task`.
Your job is to EXPLORE this scene by interacting with it, and DISCOVER an interesting task
that could be done here.
Take a sequence of actions that changes the state of the world ... Aim for about 10 to 20
actions.
IMPORTANT: respond in this format.
- On each exploration step: the <think></think> tags stay empty; write
a brief plain-text reasoning AFTER them about what interesting, state-changing action
to take next, then output your action in <tool_call></tool_call> tags.
- *** ONE ACTION PER RESPONSE -- THEN STOP IMMEDIATELY. ***
  Your response must contain EXACTLY ONE <tool_call></tool_call> block and must end right
  after its closing tag. After emitting it, STOP: do not write a second tool call, do not
  explain what you would do next. The environment replies with the new observation, and only
  then do you choose your next action.
  Example of exactly one well-formed response (nothing before <think>, nothing after the
  closing tag):
  <think></think>The desk may hold a small object, so I will walk over and check it.
  <tool_call>{"name":"alfworld_tool","arguments":{"action_type":"go_to","receptacle_name":"desk 1"}}</tool_call>
""" + _ACTION_SPACE_DOC + """
WHEN YOU ARE DONE: once you have accomplished something interesting and state-changing, END
by emitting a single "finish" tool call stating that task:
`{"name":"alfworld_tool","arguments": {"action_type": "finish", "task_description": "<one
concise sentence>"}}`
Write the task_description as an imperative instruction, never as a report: "Put a pen in the
garbagecan.", never "I put a pen in the garbagecan."
MULTI-STEP TASKS: if the task you are building has more than one sub-step (e.g. FIRST heat an
object, THEN place it somewhere), call "sub_finish" right after EACH completed sub-step
to declare it, and reserve "finish" for the very last sub-step:
`{"name":"alfworld_tool","arguments": {"action_type": "sub_finish", "task_description":
"<that sub-step, as an imperative instruction>"}}`
sub_finish records the sub-step WITHOUT ending the episode -- you keep acting afterward.
Example: heat an apple -> sub_finish("Heat an apple.") -> put it on the countertop ->
finish("Heat some apple and put it on the countertop.")."""

# 补全: 附录 G 的 user 轮骨架只有 "Explore this scene. Take one action per turn. ..."，
# 补全了回合数上限与动作合法性提示（U3 决策点：默认不给 admissible_commands，
# 由 ez_cfg.proposer_give_admissible 开关控制，smoke 时对比两种设定）。
PROPOSER_USER_TEMPLATE = """{observation}
Explore this scene. Take exactly ONE action per turn, then stop and wait for the observation.
When you have reached an interesting state-changing result, stop and call `finish` with a
`task_description` stating that task as an imperative instruction. You have at most {max_turns}
turns.
{admissible_hint}"""

# 补全: admissible 提示块（开启 U3 开关时注入；格式与 format_observation 对齐）
ADMISSIBLE_HINT = "Admissible actions at this step: {admissible}"

# 补全（论文未规定）：**临近回合上限时的收尾提醒**。
#
# 为什么必须有：实测（job 12989069/12989143/12989301）Qwen3-4B-Instruct-2507 会把
# 探索回合跑满也不主动调 `finish` —— mean_turns 恒等于 max_turns、accept=0，
# 于是 proposer 相位一条任务都产不出来、solver 相位整段被跳过，整轮训练白跑。
# 只在**剩余 <= NUDGE_WHEN_LEFT 轮**时注入一次，不影响正常节奏；
# 语义上它只是"提醒该收尾了"，不改变奖励/过滤规则。
FINISH_NUDGE = """You have {remaining} turn(s) left out of {total}.
If the actions above have already produced an interesting, reproducible state change, call
`finish` NOW with the imperative `task_description`. A declared task whose end state has been
reached is worth far more than more exploration steps. If you still need one more sub-step,
do it this turn and call `finish` on the very next turn."""

#: 剩余多少轮时注入 FINISH_NUDGE
NUDGE_WHEN_LEFT = 3

SOLVER_SYSTEM = """You are an AI assistant playing ALFWorld, a text-based household environment.
Your goal is to complete household tasks by interacting with objects.
IMPORTANT: You MUST respond in the following format:
1. The <think></think> tags stay empty; write your brief plain-text plan AFTER them
2. Then, output your action in <tool_call></tool_call> tags
3. You are given the task; do not emit proposer declarations.
*** ONE ACTION PER RESPONSE -- THEN STOP IMMEDIATELY. ***
Emit EXACTLY ONE <tool_call></tool_call> block per response and end your response right after
its closing tag. Do not write a second tool call, do not keep planning further steps.
""" + _ACTION_SPACE_DOC

# 补全: 附录 G Fig.14 的 user 轮含 "# Example Omitted" + "{current_observation}" +
# "Your task is: {task_description}"，照抄骨架，Example 一节留空（骨架注明 Omitted）。
SOLVER_USER_TEMPLATE = """# Your Task
{observation}
Your task is: {task_description}
Explore the environment and complete the task."""


def build_proposer_messages(observation: str, max_turns: int,
                            admissible: Sequence[str] = ()) -> List[Dict[str, str]]:
    adm = ADMISSIBLE_HINT.format(admissible=", ".join(admissible)) if admissible else ""
    return [
        {"role": "system", "content": PROPOSER_SYSTEM},
        {"role": "user", "content": PROPOSER_USER_TEMPLATE.format(
            observation=observation, max_turns=max_turns, admissible_hint=adm)},
    ]


def build_solver_messages(observation: str, task_desc: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SOLVER_SYSTEM},
        {"role": "user", "content": SOLVER_USER_TEMPLATE.format(
            observation=observation, task_description=task_desc)},
    ]


def append_tool_result(messages: List[Dict[str, str]], action_text: str,
                       obs_text: str, admissible: Sequence[str] = ()) -> None:
    """把一步动作+观测追加进对话。

    ⚠️ 用 **user** role 而非 tool role：Qwen3 的 chat template 对"没有 tool_calls
    上下文的孤立 tool 消息"渲染行为未验证（review P1-5），主线也全走 user role。
    """
    adm = ""
    if admissible:
        adm = "\nAdmissible actions at this step: " + ", ".join(admissible)
    messages.append({"role": "user",
                     "content": f"You did: {action_text}\nThe environment responds:\n{obs_text}{adm}"})
