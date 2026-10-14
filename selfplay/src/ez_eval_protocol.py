"""Protocol bookkeeping for EZ evaluations; intentionally free of model/ALFWorld imports."""
from __future__ import annotations

import math
import os
import statistics
from collections import defaultdict, deque
from typing import Mapping, Sequence


def _game_key(path: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(str(path))))


def align_records_to_slots(requested_games: Sequence[str], records: Sequence[dict],
                           used_games: Sequence[str]) -> list[dict]:
    """FIFO-align duplicate samples to the post-shuffle slots and reject bad resets."""
    if len(requested_games) != len(records):
        raise ValueError("requested game/record lengths differ")
    if len(used_games) != len(requested_games):
        raise ValueError("reset returned a different number of game slots")
    queues = defaultdict(deque)
    for game, record in zip(requested_games, records):
        queues[_game_key(game)].append(dict(record))
    aligned = []
    for game in used_games:
        q = queues.get(_game_key(game))
        if not q:
            raise ValueError(f"reset returned unexpected or excess game slot: {game}")
        aligned.append(q.popleft())
    missing = [key for key, q in queues.items() if q]
    if missing:
        raise ValueError(f"reset omitted requested game slots: {missing[:3]}")
    return aligned


def aggregate_complete_evaluation(rows: Sequence[dict], expected_tasks: Mapping[str, str],
                                  samples_per_task: int) -> dict:
    """Validate a complete task×sample grid and compute explicitly named SR statistics."""
    if samples_per_task <= 0:
        raise ValueError("samples_per_task must be positive")
    if not expected_tasks:
        raise ValueError("expected task set is empty")
    expected_keys = {(task_id, sample) for task_id in expected_tasks
                     for sample in range(samples_per_task)}
    seen = {}
    for row in rows:
        task_id = str(row.get("task_id", ""))
        sample = row.get("sample")
        if isinstance(sample, bool) or not isinstance(sample, int):
            raise ValueError("each sample row needs an integer sample index")
        key = (task_id, sample)
        if key not in expected_keys:
            raise ValueError(f"unexpected task/sample row: {key}")
        if key in seen:
            raise ValueError(f"duplicate task/sample row: {key}")
        if str(row.get("task_type")) != str(expected_tasks[task_id]):
            raise ValueError(f"task type mismatch for {task_id}")
        won = row.get("won")
        if isinstance(won, bool):
            seen[key] = won
        elif isinstance(won, int) and won in (0, 1):
            seen[key] = bool(won)
        else:
            raise ValueError(f"sample row {key} needs won as bool or 0/1 integer")
    missing = expected_keys.difference(seen)
    if missing:
        raise ValueError(f"missing task/sample rows: {sorted(missing)[:5]}")

    suite_srs = [sum(seen[(task_id, s)] for task_id in expected_tasks) / len(expected_tasks)
                 for s in range(samples_per_task)]
    episode_sr = sum(seen.values()) / len(seen)
    by_task = defaultdict(list)
    by_type = defaultdict(list)
    for task_id, task_type in expected_tasks.items():
        vals = [float(seen[(task_id, s)]) for s in range(samples_per_task)]
        by_task[task_id] = vals
        by_type[str(task_type)].append(sum(vals) / len(vals))
    type_macro = {k: statistics.mean(v) for k, v in sorted(by_type.items())}
    type_summary = {}
    for task_type, task_means in by_type.items():
        task_ids = [tid for tid, typ in expected_tasks.items() if str(typ) == task_type]
        wins = sum(seen[(task_id, sample)] for task_id in task_ids
                   for sample in range(samples_per_task))
        episodes = len(task_ids) * samples_per_task
        type_summary[task_type] = {
            "task_macro_sr": type_macro[task_type],
            "episode_weighted_sr": wins / episodes,
            "n_tasks": len(task_ids),
            "n_episodes": episodes,
        }
    return {
        "episode_weighted_sr": episode_sr,
        "episode_bernoulli_std": math.sqrt(episode_sr * (1.0 - episode_sr)),
        "sample_suite_success_rates": suite_srs,
        "sample_suite_sr_population_std": statistics.pstdev(suite_srs),
        "sample_suite_sr_sample_std": (statistics.stdev(suite_srs)
                                        if len(suite_srs) > 1 else None),
        "sample_suite_std_definition": "population ddof=0 and sample ddof=1, across the per-sample whole-suite SRs",
        "per_task_sample_success": {
            task_id: {"sample_mean": statistics.mean(vals),
                      "sample_std_population": statistics.pstdev(vals),
                      "n_samples": len(vals)}
            for task_id, vals in by_task.items()},
        "task_type": dict(sorted(type_summary.items())),
        "task_type_macro_sr": statistics.mean(type_macro.values()),
        "n_task_types": len(type_macro),
        "n_tasks": len(expected_tasks),
        "n_episodes": len(seen),
        "samples_per_task": samples_per_task,
    }
