"""Explorer-Zero 的 prompt 模板（按论文附录 G 的 Fig.13/14 骨架复刻）。

不确定项声明（复现计划 U2）：论文 prompt 标注 "Details omitted for space"，
本文件在骨架基础上补全了细节，所有**补全决策**用「补全:」注释标出，
与骨架冲突处以骨架为准。
"""
from __future__ import annotations

import os
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

# 补全: 论文附录 G Fig.14 的 user 轮里 **确实有一个 `# Example Omitted` 节**
# （骨架注明"因版面省略"），说明论文真实 prompt 是**带示例**的，我们照抄骨架时留空了。
# 用环境变量 EZ_SOLVER_FEWSHOT=1 打开 one-shot worked example；**默认关闭**
# （=0 / 未设），保证仍在跑的正式训练的行为与改动前逐字一致。
# 这是仓库自拟诊断示例，不是论文中被省略的作者示例；启用后会改变评测协议。
SOLVER_FEWSHOT_EXAMPLE = """# Example
The environment gave this observation:
    You are in a kitchen. You see a countertop 1, a fridge 1, a microwave 1 and an apple 1.
Your task is: put a hot apple in the microwave.
Your response must look like this:
    <think></think> I need a hot apple. Apples are usually inside the fridge, so I should go there first.
    <tool_call>{"name":"alfworld_tool","arguments":{"action_type":"go_to","receptacle_name":"fridge 1"}}</tool_call>"""

# 兼容旧环境变量的别名。它不保留任何可追溯到作者的示例内容。
SOLVER_FEWSHOT_EXAMPLE_ONSCHEMA = SOLVER_FEWSHOT_EXAMPLE


def _solver_fewshot() -> str:
    """返回要在 `# Your Task` 前插入的示例块（含尾换行分隔），默认（都未设/0）为空串。

    - EZ_SOLVER_FEWSHOT=1        → 仓库自拟合法 JSON 示例
    - EZ_SOLVER_FEWSHOT_SCHEMA=1 → 为兼容旧配置保留的同内容别名
    两者默认都关闭，保证正式训练行为与改动前逐字一致。
    """
    if os.environ.get("EZ_SOLVER_FEWSHOT_SCHEMA", "0") == "1":
        return SOLVER_FEWSHOT_EXAMPLE_ONSCHEMA + "\n"
    if os.environ.get("EZ_SOLVER_FEWSHOT", "0") == "1":
        return SOLVER_FEWSHOT_EXAMPLE + "\n"
    return ""


def build_proposer_messages(observation: str, max_turns: int,
                            admissible: Sequence[str] = ()) -> List[Dict[str, str]]:
    adm = ADMISSIBLE_HINT.format(admissible=", ".join(admissible)) if admissible else ""
    return [
        {"role": "system", "content": PROPOSER_SYSTEM},
        {"role": "user", "content": PROPOSER_USER_TEMPLATE.format(
            observation=observation, max_turns=max_turns, admissible_hint=adm)},
    ]


def build_solver_messages(observation: str, task_desc: str,
                         admissible: Sequence[str] = ()) -> List[Dict[str, str]]:
    initial_admissible = admissible_suffix(admissible)
    return [
        {"role": "system", "content": SOLVER_SYSTEM},
        {"role": "user", "content": _solver_fewshot() + SOLVER_USER_TEMPLATE.format(
            observation=observation, task_description=task_desc) + initial_admissible},
    ]


_RICH_SYSTEM = """You are an agent acting in a text-based household environment (ALFWorld).

How the environment works:
- You can only interact with things at your current location. Use "go to X" to move.
- Objects are inside or on receptacles (countertops, cabinets, drawers, shelves, fridge, ...). To find an object, visit likely receptacles; "open X" closed ones (cabinets, drawers, fridge, microwave, safe) to see inside.
- You can carry only one object at a time. "take OBJ from RECEP" picks it up; "move OBJ to RECEP" puts it down.
- To heat an object: take it, go to the microwave, then "heat OBJ with microwave".
- To cool an object: take it, go to the fridge, then "cool OBJ with fridge".
- To clean an object: take it, go to the sinkbasin, then "clean OBJ with sinkbasin".
- To look at/examine an object under a lamp: take the object, go to where the desklamp is, then "use desklamp".
- For "two" tasks, place the first object, then find and place the second one.
- "Nothing happens." means the action had no effect: do not repeat it, try something else.
- Do not keep revisiting the same places; explore receptacles you have not checked yet."""

_RICH_USER = """Task:
{task}

Step {step} of {max_steps}.

Recent history:
{history}

Actions that had no effect so far (avoid repeating them):
{failed}

Current observation:
{observation}

Available actions:
{actions}

Think about which sub-goal you are on (find the object, take it, transform it, place it) and choose the single best next action to complete the task.
Return ONLY the exact action text from the available actions."""


def solver_style() -> str:
    """`EZ_SOLVER_STYLE=rich` 切到"主流 rich scaffold"；默认 `ours`（我们原来的骨架）。

    为什么要加这一档（2026-10-14 调研实锤）：公开复现里**同一个 Qwen3-4B base**，
    只把 prompt 从 "strict" 换成 "rich"，valid_unseen 就从 **3.5% 提到 10.4%**
    （episode 加权；未加权的 6 类平均是 12.0%）。rich 相对 strict 只多了四样东西：
      ① system 里 10 条环境规则（物体在哪、`open X`、一次只拿一件、
         heat→微波炉 / cool→冰箱 / clean→水槽、`"Nothing happens."` 即无效别重复、别反复回同一地点）
      ② 步数计数器 `Step i of N`
      ③ 失败动作黑名单
      ④ 子目标提示（找物体→拿起→变换→放置）
    而且它**每步把合法动作表编号列出**，要模型 `Return ONLY the exact action text`
    —— 让模型从菜单里**照抄**一条原文，而不是自己构造 JSON。
    这正是我们缺的那 3 倍。
    """
    import os
    return os.environ.get("EZ_SOLVER_STYLE", "ours").strip().lower()


def build_solver_messages_rich(task: str, observation: str, history, admissible,
                               step: int, max_steps: int, failed) -> List[Dict[str, str]]:
    """rich scaffold 的每步重建式 prompt（与公开实现逐字对齐）。

    注意它是**无状态重建**：每步用 (task, observation, history, failed, admissible)
    重新拼整条消息，而不是在对话后面追加。所以调用方必须自己维护 history / failed。
    """
    h = "\n".join(f"> {a}\n{o}" for a, o in history) if history else "(none)"
    acts = "\n".join(f"{i}. {c}" for i, c in enumerate(admissible))
    return [
        {"role": "system", "content": _RICH_SYSTEM},
        {"role": "user", "content": _RICH_USER.format(
            task=task, step=step + 1, max_steps=max_steps, history=h,
            failed=", ".join(failed) if failed else "(none)",
            observation=str(observation).strip(), actions=acts)},
    ]


def admissible_suffix(admissible: Sequence[str]) -> str:
    """把**当前这步的全部合法动作**拼成一小段，附在每回合的 feedback 尾巴上。

    ⚠️ 为什么（2026-10-14 实测诊断）：基座的评测动作分布是
        `look` **60.7%**、`go to X` 33.7%、`take` 1.0%、`heat` 0.7%
       —— 模型几乎不尝试完成任务。注意 `look` 和 `inventory` 是动作表里**唯二不需要
       任何参数**的调用；基座模型 schema 遵循能力弱，就会往"最省事、不用拼实例名"
       那条路上塌。而 ALFWorld 要求实例名**逐字精确**（`desk 1` vs `desk 2`），
       自己拼几乎必错。

    把合法动作表直接给出来，模型就能**照抄一条完整命令**，同时绕开两个坑：
      * 不用猜实例编号；* 不用自己构造 JSON 参数。
    论文没写明给不给（我们复现计划里记为风险 U3），所以用开关控制、默认关闭。
    `EZ_SOLVER_ADMISSIBLE=1` 打开。
    """
    import os
    if os.environ.get("EZ_SOLVER_ADMISSIBLE", "0") != "1" or not admissible:
        return ""
    items = [str(a).strip() for a in admissible if str(a).strip()]
    if not items:
        return ""
    return "\n" + ADMISSIBLE_HINT.format(admissible=", ".join(items))


def resolve_action(text: str, admissible: Sequence[str], *, strict: bool | None = None) -> tuple:
    """把模型输出解析成一条合法动作，返回 `(command, how)`。**永不返回"无动作"**。

    这是为公开 rich-scaffold 适配的多档解析实现，参考了公开 agent 的动作选择逻辑；
    它并非逐字移植，也不把公开结果归因于单一解析器：

        exact → index → cleaned → substring → fuzzy → fallback(永不 no-action)

    这些启发式用于 rich scaffold 每步提供编号菜单时的输出兼容：
    合法动作表编号列出时，输出可能包含序号和额外文字，例如：
        '2. go to the desklamp. use desklamp.'
        '2. go to the sinkbasin, then "clean OBJ from the sinkbasin".'
    这类输出可能在编号后带有额外文字；不同模型与 prompt 下的命中率需由当前 run 实测。

    ★ 每一档都只在 admissible 白名单里取，**不可能**返回非法动作。
    """
    import os
    import re as _re
    if strict is None:
        strict = os.environ.get("EZ_SOLVER_STRICT_RESOLVE", "0") == "1"
    if not admissible or strict:
        return "", "none"
    adm = [str(a).strip() for a in admissible]

    def _normalize(x: str) -> str:
        # Match the public parser's safety-preserving normalization exactly:
        # case folding and whitespace normalization only.
        return " ".join(str(x).lower().split())

    raw = str(text).strip()
    # 1) exact（未清洗前先试）
    if raw in adm:
        return raw, "exact"
    # 去 <think>...</think>
    raw = _re.sub(r"<think>.*?(</think>|$)", "", raw, flags=_re.S)
    lines = [l for l in raw.strip().splitlines() if l.strip()]
    text = lines[0].strip() if lines else ""
    text = text.strip("`*\"' ")
    text = _re.sub(r"^(?:action|next action|answer|command|output)\s*[:\-]\s*", "", text, flags=_re.I)
    text = text.lstrip("> ").strip("`*\"' ").rstrip(".").strip()
    # 2) index：整串就是个编号 → 直接取 admissible[i]
    if text.isdigit() and int(text) < len(adm):
        return adm[int(text)], "index"
    # 3) cleaned：去掉 `N.` 前缀后再试
    text = _re.sub(r"^\d+\s*[.):\-]\s*", "", text).strip("`*\"' ").rstrip(".").strip()
    norm = {_normalize(c): c for c in adm}
    if _normalize(text) in norm:
        return norm[_normalize(text)], "cleaned"
    # 4) substring：带词边界地在**整段原文**里找（去包含关系，避免 look 抢走更长的命令）
    low = _normalize(raw)
    hits = [c for c in adm if _re.search(r"(?<!\w)" + _re.escape(_normalize(c)) + r"(?!\w)", low)]
    hits = [c for c in hits if not any(c != h and _normalize(c) in _normalize(h) for h in hits)]
    if len(hits) == 1:
        return hits[0], "substring"
    # 5) fuzzy：忽略空格后完全相同
    squashed = _normalize(text).replace(" ", "")
    same = [c for k, c in norm.items() if k.replace(" ", "") == squashed]
    if len(same) == 1:
        return same[0], "fuzzy"
    # 5b) fuzzy：唯一前缀补全（模型只写了前半条命令）
    comp = [c for k, c in norm.items() if text and k.startswith(_normalize(text) + " ")]
    if len(comp) == 1:
        return comp[0], "fuzzy"
    # 6) fallback：**永不无动作**（与参考实现一致）
    return ("look" if "look" in adm else adm[0]), "fallback"


def recover_bare_command(text: str, admissible: Sequence[str]) -> str:
    """从"被判 malformed"的输出里抢救出一条**确实可执行**的自然命令。

    ⚠️ 为什么需要（2026-10-14 实测，可能是复现卡住的主因之一）：
        ALFWorld 的**原生动作空间本来就是自然命令字符串**（`"go to desk 1"`），
        但我们要求模型输出 JSON tool-call，而 `parse_tool_call` **只认 JSON**。实测：
            parse_tool_call("go to desk 1")               -> malformed
            parse_tool_call("take apple 1 from fridge 1") -> malformed
            parse_tool_call('{"action_type":"go_to","receptacle_name":"desk 1"}')
                                                          -> action "go to desk 1" ✓
        而 `_rollout_batch` / `run_ez_solver_rollouts` 会把 malformed **静默换成 look**。
        基座模型很可能根本不吐 JSON、直接吐自然命令 ⇒ **完全可用的回答被丢掉**，
        历史诊断样本出现过这种格式差异；当前格式比例应从对应运行的 trace 单独统计。

    ★ 安全性：只在**与当前 admissible 列表能对上**时才采用 ——
      先小写化 + 空白归一，再要求整条命令作为子串出现，取命中的**最长**那条。
      所以本函数**不可能**返回非法动作，最坏情况返回 ""（退回原来的 look）。

    默认关闭（`EZ_SOLVER_BARE_FALLBACK!=1` 时恒返回 ""），保证正式训练行为逐字节不变。
    """
    import os
    if not admissible or os.environ.get("EZ_SOLVER_BARE_FALLBACK", "0") != "1":
        return ""
    norm = " ".join(str(text).lower().split())
    if not norm:
        return ""
    best, best_len = "", 0
    for a in admissible:
        a_str = str(a).strip()
        a_norm = " ".join(a_str.lower().split())
        if a_norm and a_norm in norm and len(a_norm) > best_len:
            best, best_len = a_str, len(a_norm)
    return best


def dump_malformed(path: str, game, turn: int, text: str) -> None:
    """把被判 malformed 的**原始生成文本**逐条落盘（`EZ_DUMP_MALFORMED` 为空时不做）。

    没有这个就没法判断"模型是真的不会"还是"我们的解析器把它的回答扔了"。
    """
    if not path:
        return
    import json as _json
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(_json.dumps({"game": str(game), "turn": int(turn),
                                  "text": text}, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 —— 诊断落盘失败绝不能影响评测
        pass


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


def format_execution_feedback(command: str, observation: str, *, malformed: bool = False,
                              recovered: bool = False, control: bool = False,
                              admissible: Sequence[str] = (),
                              format_error: str = "") -> str:
    """Include the executed command and real environment observation in every feedback."""
    prefix = ""
    if format_error:
        prefix = format_error.rstrip() + "\n"
    elif malformed:
        if recovered:
            prefix = "Your response format was invalid; the recovered command was executed.\n"
        else:
            prefix = "Your response was not a valid tool call; the fallback command was executed.\n"
    elif control:
        prefix = "Declarations are not environment actions; the fallback command was executed.\n"
    text = f"{prefix}You did: {command}\nThe environment responds:\n{observation}"
    hint = admissible_suffix(admissible)
    return text + hint
