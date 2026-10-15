"""Shared raw-text ALFWorld prompting protocol based on the measured SkillRise quick run.

This captures that run's prompt construction, 15-step action/observation history,
action projection, and plain-prompt tokenization. It is an aligned local setting,
not a claim that every SkillRise implementation detail is reproduced.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

ALFWORLD_PLAY_PROMPT = """
You are an expert agent operating in the ALFRED Embodied Environment.
{init_observation}{past_trajectories_reflections}{current_trajectory}

Your admissible actions of the current situation are:\x20
[{admissible_actions}]

Now it's your turn to take an action.

- Your response should first by step-by-step reasoning about the current situation.
- Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

CURRENT_TRAJECTORY = """
Below are the actions you took and the corresponding observations:\x20
{current_trajectory}"""

HISTORY_LIMIT = 15
MAX_PROMPT_CONTEXT = 32768
ACTION_CLOSE = "</action>"


def format_skillrise_history(records: Sequence[tuple[str, str]], history_length: int = HISTORY_LIMIT) -> str:
    """Format executed actions and resulting observations in SkillRise memory style."""
    if history_length < 0:
        raise ValueError("history_length must be nonnegative")
    recent = list(records[-history_length:]) if history_length else []
    start = len(records) - len(recent)
    return "\n".join(
        f"Action {start + j + 1}: {action}\nObservation {start + j + 1}: {observation}"
        for j, (action, observation) in enumerate(recent)
    )


def build_skillrise_prompt(init_observation: str, history: Sequence[tuple[str, str]],
                           admissible: Sequence[str], turn_idx: int,
                           history_length: int = HISTORY_LIMIT) -> str:
    """Build the exact quick-eval raw string prompt for one decision turn."""
    current = "" if turn_idx == 0 else CURRENT_TRAJECTORY.format(
        current_trajectory=format_skillrise_history(history, history_length=history_length))
    adm = ",\n ".join(f"'{action}'" for action in admissible if action != "help")
    body = ALFWORLD_PLAY_PROMPT.format(
        init_observation=init_observation,
        past_trajectories_reflections="",
        current_trajectory=current,
        admissible_actions=adm,
    )
    return body.strip()


def project_skillrise_action(text: str) -> tuple[str, bool]:
    """Project model text using SkillRise play-phase action-tag semantics."""
    lowered = str(text).lower()
    start, end = "<action>", ACTION_CLOSE
    si, ei = lowered.find(start), lowered.find(end)
    if si == -1 or ei == -1:
        return lowered[-30:], False
    return lowered[si + len(start):ei].strip(), True


def encode_raw_prompt(tokenizer: Any, prompt: str) -> list[int]:
    """Match vLLM's plain string prompt path, including tokenizer special tokens."""
    encoded = tokenizer(prompt, add_special_tokens=True)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("expected one prompt from tokenizer")
        ids = ids[0]
    return [int(token) for token in ids]


_TASK_GOAL_LINE = re.compile(r"(?im)^(\s*Your task is(?: to)?:\s*).*$")


def replace_task_goal(raw_observation: str, generated_task: str) -> str:
    """Replace only ALFWorld's displayed goal line; fail closed if it is absent."""
    observation = str(raw_observation)
    task = str(generated_task).strip()
    matches = list(_TASK_GOAL_LINE.finditer(observation))
    if not task or len(matches) != 1:
        raise ValueError(f"expected exactly one official task goal line, found {len(matches)}")
    match = matches[0]
    return observation[:match.start()] + match.group(1) + task + observation[match.end():]


def action_tag_stop_len(tokenizer: Any, response_ids: Sequence[int]) -> int:
    """Return the first response length containing `</action>`, including its tokens."""
    ids = list(response_ids)
    if not ids:
        return 0
    full_text = tokenizer.decode(ids, skip_special_tokens=True)
    if ACTION_CLOSE not in full_text:
        return len(ids)
    for length in range(1, len(ids) + 1):
        if ACTION_CLOSE in tokenizer.decode(ids[:length], skip_special_tokens=True):
            return length
    return len(ids)
