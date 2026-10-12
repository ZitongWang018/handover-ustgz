"""ALFWorld ReAct 的 prompt 构造与动作解析。

约定
----
* 每步模型输出**两行**：`Thought: ...` / `Action: ...`，生成在 `Action:` 行结束后立即停止。
* `Action:` 后面必须是 `Admissible Commands` 里的一条**原样**命令（大小写不敏感）。
* 环境观测一律用 `Observation:` 前缀喂回去，保持 ReAct 的轨迹结构。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, List, Optional, Sequence, Tuple

ACTION_RE = re.compile(r"^[\s\*_`#>•\-]*action[\s\*_`#]*[:：][\s\*_`]*([^\n]*)$", re.IGNORECASE | re.MULTILINE)
THOUGHT_RE = re.compile(r"^[\s\*_`#>•\-]*thought[\s\*_`#]*[:：][\s\*_`]*([^\n]*)$", re.IGNORECASE | re.MULTILINE)
# 退路：允许 "my action: xxx" 这种行内出现的情况
ACTION_LOOSE_RE = re.compile(r"action[\s\*_`#]*[:：][\s\*_`]*([^\n]*)", re.IGNORECASE)

SYSTEM_PROMPT = """You are an expert agent solving household tasks in the ALFWorld text environment.

At each turn you receive:
1. The task you must complete.
2. The current observation (what you see / what just happened).
3. A list of Admissible Commands - this is the ONLY action space.

Rules:
- You must output exactly two lines and nothing else.
- Line 1: "Thought: " followed by one short sentence of reasoning.
- Line 2: "Action: " followed by exactly one command copied verbatim from the Admissible Commands list.
- Never invent commands, never output more than one action, never add explanations after the action.

Example of a valid response:
Thought: I should first look around to find the objects mentioned in the task.
Action: look"""


@dataclass
class ParsedAction:
    thought: str
    action: str
    has_marker: bool


def build_system_prompt() -> str:
    return SYSTEM_PROMPT


def format_task_block(task_desc: str) -> str:
    task_desc = task_desc.strip()
    if not task_desc.lower().startswith("your task is"):
        task_desc = f"Your task is to: {task_desc}"
    return task_desc


def format_observation(obs: str, admissible: Sequence[str], first_turn: bool = False, task_desc: str = "") -> str:
    """把一条环境反馈渲染成一个 user turn。"""
    parts: List[str] = []
    if first_turn and task_desc:
        parts.append(format_task_block(task_desc))
    parts.append("Observation: " + obs.strip())
    if admissible:
        parts.append("Admissible Commands:\n" + "\n".join(f"- {c}" for c in admissible))
    return "\n\n".join(parts)


def format_turn_feedback(action: str, obs: str, admissible: Sequence[str], done: bool = False, won: bool = False) -> str:
    """上一轮动作 + 新的观测，构成下一轮的 user turn。"""
    parts = [f"Observation: {obs.strip()}"]
    if done:
        parts.append("Episode finished. " + ("Task completed successfully." if won else "Task failed."))
    if admissible:
        parts.append("Admissible Commands:\n" + "\n".join(f"- {c}" for c in admissible))
    return "\n\n".join(parts)


def parse_action(text: str) -> ParsedAction:
    """从模型输出里抽 (thought, action)。`has_marker` 表示是否出现了 `Action:` 标记。"""
    text = text.strip()
    action = ""
    has_marker = False
    matches = list(ACTION_RE.finditer(text))
    if matches:
        has_marker = True
        action = matches[-1].group(1).strip()
    if not action:
        loose = list(ACTION_LOOSE_RE.finditer(text))   # 容忍 "my action: xxx"
        if loose:
            has_marker = True
            action = loose[-1].group(1).strip()
    if not action:
        # 最后一层退路：取最后一行非 Thought 的行
        for line in reversed([ln.strip() for ln in text.splitlines() if ln.strip()]):
            if not THOUGHT_RE.match(line):
                action = line
                break
    thought_matches = list(THOUGHT_RE.finditer(text))
    thought = thought_matches[-1].group(1).strip() if thought_matches else ""

    action = _strip_wrappers(action)
    return ParsedAction(thought=thought, action=action, has_marker=has_marker)


def _strip_wrappers(action: str) -> str:
    action = action.strip()
    action = re.sub(r"^[\-\*_`#>\u2022\s]+", "", action)       # 去掉列表符号 / markdown 强调符
    action = re.sub(r"[\*_`#\s]+$", "", action)               # 去掉结尾的强调符
    action = re.sub(r"^[`\"'“”‘’](.*)[`\"'“”‘’]$", r"\1", action).strip()  # 去掉包裹的引号
    action = re.sub(r"^(action|command)\s*[:：]\s*", "", action, flags=re.IGNORECASE)
    action = re.sub(r"[.!。]+$", "", action).strip()
    return action


def match_admissible(action: str, admissible: Sequence[str], fuzzy_threshold: float = 0.0) -> Optional[str]:
    """把模型输出的动作对齐到 admissible 列表里的合法命令。

    `fuzzy_threshold` > 0 时允许模糊匹配（只用于分析，正式训练默认关闭，以保持奖励诚实）。
    """
    if not action:
        return None
    cands = list(admissible)
    norm = {c.strip().lower(): c for c in cands}
    if action.strip().lower() in norm:
        return norm[action.strip().lower()]
    if fuzzy_threshold > 0:
        best, best_score = None, 0.0
        for c in cands:
            score = SequenceMatcher(None, action.lower(), c.strip().lower()).ratio()
            if score > best_score:
                best, best_score = c, score
        if best_score >= fuzzy_threshold:
            return best
    return None


def stop_text_finished(generated: str) -> bool:
    """判断生成是否已经可以停：`Action:` 那一行已经写完（后面出现了换行）。

    ⚠️ 为什么**不能**用"Action 行后面是空的"作为判据：
        criterion 每 `check_every=2` 步才检查一次，完全可能落在
        "Thought: ...\nAction: lo" 这种**动作写到一半**的时刻。
        若此时判"写完"，就会把半截动作截下来 -> parse_action 解出残句 ->
        动作非法率直接上去。正确性 > 省那一点算力（单条序列反正有 eos 兜底，
        criterion 只是让整批提前齐停，收益本就有限）。

    ⚠️ 也不能只找第一个 `Action:`：模型可能先写一段再纠正，所以要取**最后一个**。
    """
    m = None
    for m in ACTION_RE.finditer(generated):
        pass
    if m is None:
        return False
    tail = generated[m.end():]
    return "\n" in tail
