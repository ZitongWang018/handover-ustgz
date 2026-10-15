"""Run EZ's solver in the shared SkillRise-style ALFWorld prompt setting."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ez_proposer import _slot_snapshot
from ez_state import StateSnapshot, assert_facts_mode
from ez_verify import ProposedTask, SolverRolloutResult, verify_solver_rollout
from protocol import (MAX_PROMPT_CONTEXT, build_skillrise_prompt, encode_raw_prompt,
                      project_skillrise_action, replace_task_goal)
from rollout import RolloutConfig, Trajectory, _align_slot_tasks
from reward import TrajectoryStats


@dataclass
class EzSolverTrajectory:
    """A solver trajectory with per-turn exact prompt/response token segments."""

    traj: Trajectory
    verify: SolverRolloutResult
    snapshots: List[StateSnapshot]


def run_ez_solver_rollouts(env, policy, tasks: Sequence[ProposedTask], cfg: RolloutConfig,
                           tokenize: bool = True,
                           log_fn: Optional[Callable[[str], None]] = None) -> List[EzSolverTrajectory]:
    """Roll out proposed tasks; callers pass repeated copies for GRPO groups."""
    n = len(tasks)
    if env.batch_size != n:
        raise ValueError(f"env.batch_size={env.batch_size} != n_tasks={n}")
    if cfg.stop_mode != "action_tag":
        raise ValueError("EZ solver requires explicit stop_mode='action_tag'")
    if int(cfg.max_context_tokens) != MAX_PROMPT_CONTEXT:
        raise ValueError(f"EZ solver requires max_context_tokens={MAX_PROMPT_CONTEXT}")
    seen: Dict[str, str] = {}
    for task in tasks:
        if task.game_file in seen and seen[task.game_file] != task.task_desc:
            raise ValueError("same game_file has different task descriptions; split this solver batch")
        seen[task.game_file] = task.task_desc

    games = [task.game_file for task in tasks]
    _obs, views, admissible, used, _ = env.reset(games)
    slot_task = _align_slot_tasks(games, used, group_size=1)
    facts0 = list(getattr(env, "last_reset_facts", [None] * n))
    snap_hist: List[List[StateSnapshot]] = []
    init_observations: List[str] = []
    histories: List[List[tuple[str, str]]] = [[] for _ in range(n)]
    for slot in range(n):
        task = tasks[slot_task[slot]]
        s0 = _slot_snapshot(facts0[slot], views[slot].raw or _obs[slot])
        assert_facts_mode(s0, f"solver reset slot {slot}")
        from collections import Counter
        if s0.mode != task.s0.mode or Counter(s0.facts) != Counter(task.s0.facts):
            raise RuntimeError("solver reset state differs from ProposedTask.s0; refusing mismatched rollout")
        snap_hist.append([s0])
        # Only the task goal changes; preserve the generated state's room description and facts.
        init_observations.append(replace_task_goal(
            (views[slot].raw or _obs[slot]).strip(), task.task_desc))

    stats = [TrajectoryStats(task_type="ez_generated",
                             task_id=tasks[slot_task[s]].meta.get("uid", str(slot_task[s])))
             for s in range(n)]
    messages: List[List[Dict[str, str]]] = [[] for _ in range(n)]
    segments: List[List[tuple[list[int], list[int]]]] = [[] for _ in range(n)]
    active = [True] * n
    solver_cfg = cfg
    max_steps = min(cfg.max_steps, env.max_episode_steps)
    t0 = time.time()

    for step in range(max_steps):
        if not any(active):
            break
        slots = [s for s in range(n) if active[s]]
        prompt_texts = [build_skillrise_prompt(
            init_observations[s], histories[s], admissible[s], step,
            history_length=solver_cfg.keep_recent_turns) for s in slots]
        prompt_ids = [encode_raw_prompt(policy.tokenizer, prompt) for prompt in prompt_texts]
        for s, ids in zip(slots, prompt_ids):
            if len(ids) + solver_cfg.max_new_tokens > MAX_PROMPT_CONTEXT:
                raise ValueError(
                    f"solver prompt plus generation budget exceeds {MAX_PROMPT_CONTEXT} tokens")
        texts, response_ids, _ = policy.generate_from_token_ids(prompt_ids, solver_cfg)

        step_actions = ["look"] * n
        action_by_slot: Dict[int, tuple[str, bool]] = {}
        for idx, slot in enumerate(slots):
            action, valid_tag = project_skillrise_action(texts[idx])
            executed = action if action else "look"
            step_actions[slot] = executed
            action_by_slot[slot] = (executed, valid_tag)
            if tokenize:
                segments[slot].append((list(prompt_ids[idx]), list(response_ids[idx])))
            # Preserve readable audit messages, while training consumes only exact ID segments.
            messages[slot].append({"role": "user", "content": prompt_texts[idx]})
            messages[slot].append({"role": "assistant", "content": texts[idx]})

        result = env.step(step_actions)
        facts = list(result.facts) if result.facts else [None] * n
        for slot in slots:
            snapshot = _slot_snapshot(facts[slot], result.obs[slot])
            assert_facts_mode(snapshot, f"solver step slot {slot}")
            snap_hist[slot].append(snapshot)
            executed, _valid_tag = action_by_slot[slot]
            histories[slot].append((executed, result.obs[slot]))
            stats[slot].n_steps += 1
            if _valid_tag:
                stats[slot].n_valid_steps += 1
            else:
                stats[slot].n_format_violation += 1
                stats[slot].n_invalid += 1
            if result.admissible:
                admissible[slot] = result.admissible[slot]
            if bool(result.dones[slot]):
                active[slot] = False

    out: List[EzSolverTrajectory] = []
    for slot in range(n):
        task = tasks[slot_task[slot]]
        st = stats[slot]
        if active[slot]:
            st.truncated = True
        verify = verify_solver_rollout(task, snap_hist[slot])
        st.won = verify.final_match
        kept_segments = segments[slot] if tokenize else []
        n_tokens = sum(len(response) for _prompt, response in kept_segments)
        traj = Trajectory(
            task_id=st.task_id,
            task_type=st.task_type,
            game_file=used[slot],
            messages=messages[slot],
            input_ids=[],
            loss_mask=[],
            stats=st,
            n_turns=st.n_steps,
            n_tokens=n_tokens,
            segments=kept_segments,
        )
        traj.reward = verify.reward
        out.append(EzSolverTrajectory(traj=traj, verify=verify, snapshots=snap_hist[slot]))

    if log_fn:
        rewards = [entry.traj.reward for entry in out]
        log_fn(f"[ez-solver] {n} rollouts | reward mean={np.mean(rewards):.3f} | "
               f"final_match={sum(e.verify.final_match for e in out)}/{n} | "
               f"subtask_rate={np.mean([e.verify.subtask_rate for e in out]):.3f} | "
               f"{time.time()-t0:.1f}s")
    return out
