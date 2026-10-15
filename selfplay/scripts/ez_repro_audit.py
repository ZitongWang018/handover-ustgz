#!/usr/bin/env python3
"""Read-only integrity/statistics audit for the archived EZ evaluation evidence.

This utility uses only the Python standard library. It reads local JSON/JSONL
artifacts, never imports the training or environment stack, and never launches
commands or jobs. By default it writes the report to stdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

EXPECTED_SEEDS = (0, 1, 2, 3, 4)
SLOT_COUNT = 128
STYLES = ("react", "zero_shot")
TASK_TYPES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)
DEFAULT_INPUT_REL = Path("handover/skillrise_prompting_20261015/raw")
EVIDENCE_REL = Path("handover/skillrise_prompting_20261015/ez_training_evidence")


class AuditError(ValueError):
    """The supplied artifacts fail a required structural/integrity check."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_json(path: Path, label: str) -> tuple[Any, str]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise AuditError(f"cannot read {label}: {type(exc).__name__}") from None
    try:
        return json.loads(data.decode("utf-8")), _sha256(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"invalid UTF-8 JSON in {label}: {type(exc).__name__}") from None


def _canonical_game(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AuditError(f"{label} must be a non-empty string")
    # Preserve identity while accepting JSON artifacts produced on either OS.
    return PurePosixPath(value.strip().replace("\\", "/")).as_posix()


def _task_type_from_game(game: str) -> str | None:
    for component in PurePosixPath(game).parts:
        for task_type in TASK_TYPES:
            if component.startswith(task_type + "-"):
                return task_type
    return None


def _selection_index(selection: Any) -> tuple[Counter[str], dict[str, str], dict[str, Any]]:
    if not isinstance(selection, dict):
        raise AuditError("selection manifest must be a JSON object")
    slots = selection.get("slots")
    if not isinstance(slots, list) or len(slots) != SLOT_COUNT:
        raise AuditError("selection manifest must contain exactly 128 slots")
    if selection.get("slot_count") != SLOT_COUNT:
        raise AuditError("selection manifest slot_count must equal 128")
    ids: list[int] = []
    expected: Counter[str] = Counter()
    types: dict[str, str] = {}
    for pos, slot in enumerate(slots):
        if not isinstance(slot, dict):
            raise AuditError(f"selection slot {pos} must be an object")
        idx = slot.get("slot")
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise AuditError(f"selection slot {pos} has a non-integer slot id")
        ids.append(idx)
        game = _canonical_game(slot.get("game_file"), f"selection slot {pos} game_file")
        game_id = _canonical_game(slot.get("game_id"), f"selection slot {pos} game_id")
        task_type = slot.get("task_type")
        if task_type not in TASK_TYPES:
            raise AuditError(f"selection slot {pos} has an unknown task_type")
        path_type = _task_type_from_game(game)
        if path_type != task_type:
            raise AuditError(f"selection slot {pos} task_type disagrees with its game path")
        if game in types and types[game] != task_type:
            raise AuditError("one game_file maps to multiple task types in selection")
        if game_id != game:
            # Some producers may use a relative id; require it to identify the
            # same path suffix without exposing any machine-local root.
            if not game.endswith("/" + game_id) and game != game_id:
                raise AuditError(f"selection slot {pos} game_id does not identify game_file")
        types[game] = task_type
        expected[game] += 1
    if ids != list(range(SLOT_COUNT)):
        raise AuditError("selection slots must be uniquely ordered 0..127")
    return expected, types, selection


def _sample_sd(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def _validate_raw_style(
    style: str,
    raw: Any,
    expected_games: Counter[str],
    game_types: dict[str, str],
    expected_split: str,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise AuditError(f"{style} raw result must be a JSON object")
    summary = raw.get("summary")
    rows = raw.get("rows")
    if not isinstance(summary, dict) or not isinstance(rows, list):
        raise AuditError(f"{style} raw result requires object summary and list rows")
    if summary.get("style") != style:
        raise AuditError(f"{style} summary style does not match its filename")
    if summary.get("split") != expected_split:
        raise AuditError(f"{style} summary split does not match selection split")

    expected_counter = Counter(expected_games)
    expected_seed_set = set(EXPECTED_SEEDS)
    rows_by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise AuditError(f"{style} row {index} must be an object")
        seed = row.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise AuditError(f"{style} row {index} seed must be an integer")
        if seed not in expected_seed_set:
            raise AuditError(f"{style} has unexpected seed {seed}")
        if type(row.get("won")) is not bool:
            raise AuditError(f"{style} row {index} won must be a JSON boolean")
        if row.get("style") != style:
            raise AuditError(f"{style} row {index} has a mismatched style")
        if row.get("split") != summary.get("split"):
            raise AuditError(f"{style} row {index} split disagrees with summary")
        sample = row.get("sample")
        if isinstance(sample, bool) or not isinstance(sample, int) or sample != 0:
            raise AuditError(f"{style} row {index} must be single-sample index 0")
        game = _canonical_game(row.get("game"), f"{style} row {index} game")
        if game not in game_types:
            raise AuditError(f"{style} row {index} game is absent from selection")
        if _task_type_from_game(game) != game_types[game]:
            raise AuditError(f"{style} row {index} task type disagrees with selection")
        rows_by_seed[seed].append({"game": game, "won": row["won"]})

    if set(rows_by_seed) != expected_seed_set:
        missing = sorted(expected_seed_set - set(rows_by_seed))
        raise AuditError(f"{style} missing seed groups: {missing}")

    seed_stats: list[dict[str, Any]] = []
    all_won = 0
    repeat_groups = 0
    mixed_repeat_groups = 0
    for seed in EXPECTED_SEEDS:
        seed_rows = rows_by_seed[seed]
        if len(seed_rows) != SLOT_COUNT:
            raise AuditError(f"{style} seed {seed} has {len(seed_rows)} rows; expected 128")
        observed = Counter(row["game"] for row in seed_rows)
        if observed != expected_counter:
            missing = sum((expected_counter - observed).values())
            extra = sum((observed - expected_counter).values())
            raise AuditError(f"{style} seed {seed} game multiset mismatch (missing={missing}, extra={extra})")

        wins_by_type = Counter()
        count_by_type = Counter()
        outcomes_by_game: dict[str, list[bool]] = defaultdict(list)
        seed_wins = 0
        for row in seed_rows:
            task_type = game_types[row["game"]]
            count_by_type[task_type] += 1
            wins_by_type[task_type] += int(row["won"])
            seed_wins += int(row["won"])
            outcomes_by_game[row["game"]].append(row["won"])
        macro = None
        if all(count_by_type[task_type] > 0 for task_type in TASK_TYPES):
            macro = sum(wins_by_type[t] / count_by_type[t] for t in TASK_TYPES) / len(TASK_TYPES)
        for outcomes in outcomes_by_game.values():
            if len(outcomes) > 1:
                repeat_groups += 1
                if any(outcomes) and not all(outcomes):
                    mixed_repeat_groups += 1
        all_won += seed_wins
        seed_stats.append({
            "seed": seed,
            "episodes": len(seed_rows),
            "won": seed_wins,
            "episode_weighted_sr_pct": 100 * seed_wins / SLOT_COUNT,
            "six_type_macro_sr_pct": None if macro is None else 100 * macro,
            "per_type": {
                task_type: {
                    "episodes": count_by_type[task_type],
                    "won": wins_by_type[task_type],
                    "sr_pct": (100 * wins_by_type[task_type] / count_by_type[task_type]
                               if count_by_type[task_type] else None),
                }
                for task_type in TASK_TYPES
            },
        })

    total_expected = SLOT_COUNT * len(EXPECTED_SEEDS)
    if len(rows) != total_expected:
        raise AuditError(f"{style} has {len(rows)} rows; expected exactly {total_expected}")
    expected_per_type_wins = Counter()
    expected_per_type_count = Counter()
    for seed_rows in rows_by_seed.values():
        for row in seed_rows:
            task_type = game_types[row["game"]]
            expected_per_type_count[task_type] += 1
            expected_per_type_wins[task_type] += int(row["won"])
    if summary.get("n_episodes") != total_expected:
        raise AuditError(f"{style} summary n_episodes does not match rows")
    if summary.get("n_won") != all_won:
        raise AuditError(f"{style} summary n_won does not match rows")
    sr = summary.get("sr")
    if isinstance(sr, bool) or not isinstance(sr, (int, float)) or not math.isclose(sr, all_won / total_expected, abs_tol=1e-12):
        raise AuditError(f"{style} summary sr does not match rows")
    reported_type_wins = summary.get("per_type_wins")
    if not isinstance(reported_type_wins, dict):
        raise AuditError(f"{style} summary per_type_wins must be an object")
    if any(task_type not in TASK_TYPES for task_type in reported_type_wins):
        raise AuditError(f"{style} summary per_type_wins contains an unknown task type")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in reported_type_wins.values()):
        raise AuditError(f"{style} summary per_type_wins values must be non-negative integers")
    expected_positive_wins = {t: expected_per_type_wins[t] for t in TASK_TYPES if expected_per_type_wins[t]}
    normalized_reported_wins = {task_type: value for task_type, value in reported_type_wins.items() if value}
    if normalized_reported_wins != expected_positive_wins:
        raise AuditError(f"{style} summary per_type_wins does not match rows")

    weighted_seed_values = [row["episode_weighted_sr_pct"] for row in seed_stats]
    macro_values = [row["six_type_macro_sr_pct"] for row in seed_stats]
    valid_macro = [value for value in macro_values if value is not None]
    return {
        "style": style,
        "complete_grid": True,
        "episodes": len(rows),
        "won": all_won,
        "episode_weighted_sr_pct": 100 * all_won / len(rows),
        "seed_weighted_sr_mean_pct": statistics.mean(weighted_seed_values),
        "seed_weighted_sr_sample_sd_pp_ddof1": _sample_sd(weighted_seed_values),
        "seed_six_type_macro_mean_pct": statistics.mean(valid_macro) if len(valid_macro) == len(EXPECTED_SEEDS) else None,
        "seed_six_type_macro_sample_sd_pp_ddof1": _sample_sd(valid_macro) if len(valid_macro) == len(EXPECTED_SEEDS) else None,
        "macro_missing_seed_count": len(EXPECTED_SEEDS) - len(valid_macro),
        "seeds": seed_stats,
        "same_seed_repeated_game_groups": repeat_groups,
        "same_seed_repeated_game_mixed_outcome_groups": mixed_repeat_groups,
    }


def _training_evidence(repo_root: Path) -> dict[str, Any]:
    folder = repo_root / EVIDENCE_REL
    results = []
    for path in sorted(folder.glob("*.jsonl")) if folder.is_dir() else []:
        try:
            data = path.read_bytes()
            text = data.decode("utf-8", errors="replace")
        except OSError:
            continue
        line_tags: dict[str, list[int]] = defaultdict(list)
        for line_number, line in enumerate(text.splitlines(), 1):
            try:
                record = json.loads(line)
                message = str(record.get("msg", "")) if isinstance(record, dict) else line
            except json.JSONDecodeError:
                message = line
            low = message.lower()
            tags: list[str] = []
            if "[dry-run]" in low or "dry-run" in low:
                tags.append("dry_run_config_line")
            if "[proposer]" in low or "rew=[" in low:
                tags.append("proposer_activity_line")
            if "[ez-solver]" in low or "reward mean=" in low:
                tags.append("solver_reward_line")
            if "update_rel=" in low:
                tags.append("adapter_update_diagnostic_line")
            if "grad_norm=" in low:
                tags.append("gradient_diagnostic_line")
            if "[done]" in low:
                tags.append("completion_marker_line")
            if "traceback" in low or "exception" in low:
                tags.append("error_marker_line")
            if "tasks_per_step=" in low or "自动上调" in message:
                tags.append("configured_batch_warning_line")
            for tag in tags:
                line_tags[tag].append(line_number)
        line_count = len(text.splitlines())
        results.append({
            "file": (EVIDENCE_REL / path.name).as_posix(),
            "sha256": _sha256(data),
            "line_count": line_count,
            "completion_marker_observed": bool(line_tags.get("completion_marker_line")),
            "evidence_line_numbers_by_category": dict(sorted(line_tags.items())),
            "interpretation_limit": "archived line evidence only; it does not identify which run was the user's intended complete run",
        })
    return {
        "files": results,
        "scope_note": "These logs are classified as line evidence only. Logged dry-run settings, rewards, gradients, adapter deltas, and completion markers do not by themselves establish a complete intended training run.",
    }


def audit(repo_root: Path, input_dir: Path | None = None) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    input_dir = Path(input_dir) if input_dir is not None else repo_root / DEFAULT_INPUT_REL
    if not input_dir.is_absolute():
        input_dir = repo_root / input_dir
    selection, selection_hash = _load_json(input_dir / "selection_128.json", "selection_128.json")
    expected_games, game_types, selection = _selection_index(selection)
    expected_split = selection.get("split")
    if expected_split != "eval_in_distribution":
        raise AuditError("archived 128-slot audit requires selection split eval_in_distribution")
    style_data = {}
    hashes = {"selection_128.json": selection_hash}
    for style in STYLES:
        name = f"{style}_128_seeds01234.json"
        raw, digest = _load_json(input_dir / name, name)
        hashes[name] = digest
        style_data[style] = _validate_raw_style(style, raw, expected_games, game_types, expected_split)

    evidence = _training_evidence(repo_root)
    for item in evidence["files"]:
        hashes[item["file"]] = item["sha256"]
    script_hash = _sha256(Path(__file__).read_bytes())
    raw_missing = [
        "raw-level evaluator source commit / exact script hash",
        "direct link from each raw file to the selection-manifest hash",
        "per-row slot id (the archived JSON only permits game-multiset validation)",
        "per-row prompt/template hash and complete prompt text",
        "tokenizer identity/hash and model checkpoint/adapter hash",
        "per-turn generated token ids, observations, executed actions, and verifier trace",
        "effective backend/runtime manifest attached to the raw episode rows",
    ]
    # The Counter values are multiplicities; uniqueness/repeats are computed
    # from the set of game keys, not from slot identity.
    unique_games = len(expected_games)
    repeated_games = sum(1 for count in expected_games.values() if count > 1)
    max_repeat = max(expected_games.values(), default=0)
    return {
        "audit_schema_version": 1,
        "audit_kind": "local_read_only_handover_integrity_and_statistics",
        "input_hashes_sha256": hashes,
        "selection_source_commit": selection.get("source_commit"),
        "selection_profile": selection.get("profile"),
        "evaluation_scope": {
            "styles": list(STYLES),
            "seeds": list(EXPECTED_SEEDS),
            "slots_per_seed": SLOT_COUNT,
            "episodes_per_style": SLOT_COUNT * len(EXPECTED_SEEDS),
            "split": selection.get("split"),
            "selection_multiset_verified_for_every_seed": True,
            "selection_unique_games": unique_games,
            "selection_repeated_games": repeated_games,
            "selection_max_multiplicity": max_repeat,
        },
        "results": style_data,
        "training_evidence": evidence,
        "audit_code_sha256": script_hash,
        "raw_schema_provenance_gaps": raw_missing,
        "limitations": [
            "Raw episodes do not record slot ids, so the audit verifies each seed against the exact selected game multiset rather than a row-to-slot assignment.",
            "Game and task-type consistency can be checked from the selected game path, but absent prompts, traces, model hashes, and evaluator-source hashes prevent proving trajectory or parser correctness.",
            "Mixed wins among repeated same-seed game groups are diagnostic only; they are not evidence that a seed caused the difference.",
            "The SkillRise 128-slot valid-seen prompting diagnostic is not Explorer-Zero's full valid-unseen four-sample evaluation and is not a training result.",
            "The archived training logs are not sufficient to identify a user's intended complete run or establish complete outer-loop/checkpoint history.",
        ],
    }


def _atomic_write(path: Path, payload: str) -> None:
    path = path.expanduser()
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as f:
            temp_name = f.name
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_name, path)
    except OSError as exc:
        if temp_name:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
        raise AuditError(f"cannot atomically write report: {type(exc).__name__}") from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="directory containing selection_128.json and both raw style JSON files")
    parser.add_argument("--output", type=Path, default=None,
                        help="atomically write JSON to this file; default prints JSON to stdout")
    args = parser.parse_args(argv)
    try:
        repo_root = args.repo_root.resolve()
        input_dir = args.input_dir if args.input_dir is not None else repo_root / DEFAULT_INPUT_REL
        if not input_dir.is_absolute():
            input_dir = repo_root / input_dir
        evidence_dir = repo_root / EVIDENCE_REL
        if args.output is not None:
            output_path = args.output.expanduser().resolve()
            protected_dirs = (input_dir.resolve(), evidence_dir.resolve())
            script_path = Path(__file__).resolve()
            if output_path == script_path or any(
                output_path == protected or protected in output_path.parents
                for protected in protected_dirs
            ):
                raise AuditError("--output must not overwrite or be inside audited input directories")
        result = audit(repo_root, input_dir)
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output is None:
            sys.stdout.write(payload)
        else:
            _atomic_write(args.output, payload)
        return 0
    except AuditError as exc:
        print(f"audit failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
