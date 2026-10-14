"""Small, dependency-light pieces of the SkillRise ALFWorld eval protocol.

This is a code-aligned evaluation of the upstream GRPO validation path. It is
not a claim that the paper's undisclosed prompt, split, or exact task list has
been recovered.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

UPSTREAM_COMMIT = "2b38da3ad7b64414305d39609484803b07d3b9b0"
PROFILE = "upstream-grpo-validation"
SPLIT = "eval_in_distribution"
SLOT_COUNT = 128
SLOT_SEED_BASE = 1000
SHARD_COUNT = 8
SLOTS_PER_SHARD = 16
ATTEMPTS = 3
MAX_TURNS_PER_ATTEMPT = 10
ENVIRONMENT_STEP_BUDGET = 50
MAX_PROMPT_TOKENS = 4096
MAX_RESPONSE_TOKENS = 1024
TASK_TYPES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)
UPSTREAM_AGENT_SAMPLING = {
    "n": 1,
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": -1,
    "max_tokens": MAX_RESPONSE_TOKENS,
    "min_tokens": 5,
    "repetition_penalty": 1.0,
    "ignore_eos": False,
    "detokenize": False,
    # The active SPMD SamplingParams constructor sets this even though this
    # validation path does not consume the returned log probabilities.
    "logprobs": 0,
}


class ProtocolError(ValueError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")).encode("utf-8"))


def game_id(game_file: str | Path, data_root: str | Path | None = None) -> str:
    path = Path(game_file)
    if data_root is not None:
        try:
            return path.resolve().relative_to(Path(data_root).resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def _first_reset_game(pool: Sequence[str], worker_seed: int,
                      shuffle: Callable[[list[str], int], None] | None = None) -> str:
    """Reproduce TextWorld's seeded shuffled-cycle first item.

    TextWorld 1.6.2's reset iterator shuffles a fresh copy of its game list
    using the worker seed, then returns the first entry. A callback is exposed
    so the implementation can be checked against the installed TextWorld
    source in an integration test without making this module depend on it.
    """
    if not pool:
        raise ProtocolError("Cannot choose a worker's initial game from an empty pool")
    shuffled = list(pool)
    if shuffle is None:
        import numpy as np
        np.random.RandomState(int(worker_seed)).shuffle(shuffled)
    else:
        shuffle(shuffled, int(worker_seed))
    return shuffled[0]


def make_selection_manifest(
    pool: Sequence[str],
    task_type_for: Callable[[str], str],
    *,
    pool_order_sha256: str,
    data_config_sha256: str,
    source_manifest_sha256: str,
    split: str = SPLIT,
    slot_count: int = SLOT_COUNT,
    seed_base: int = SLOT_SEED_BASE,
) -> dict[str, Any]:
    if split != SPLIT:
        raise ProtocolError(f"Profile {PROFILE} requires split={SPLIT!r}, got {split!r}")
    if not pool:
        raise ProtocolError("ALFWorld task pool is empty")
    slots = []
    for slot in range(slot_count):
        seed = seed_base + slot
        selected = _first_reset_game(pool, seed)
        slots.append({
            "slot": slot,
            "worker_seed": seed,
            "game_file": selected,
            "game_id": game_id(selected),
            "task_type": task_type_for(selected),
        })
    return {
        "schema_version": 1,
        "profile": PROFILE,
        "source_commit": UPSTREAM_COMMIT,
        "split": split,
        "selection_semantics": "first game from each worker-seeded TextWorld shuffled cycle over the complete unsorted ALFWorld pool; duplicates retained",
        "slot_count": slot_count,
        "worker_seed_base": seed_base,
        "pool_count": len(pool),
        "pool_order_sha256": pool_order_sha256,
        "data_config_sha256": data_config_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "slots": slots,
    }


def validate_selection_manifest(manifest: Mapping[str, Any], expected_slots: int = SLOT_COUNT) -> None:
    if manifest.get("profile") != PROFILE or manifest.get("source_commit") != UPSTREAM_COMMIT:
        raise ProtocolError("Selection manifest is from an incompatible profile or upstream commit")
    if manifest.get("split") != SPLIT:
        raise ProtocolError(f"Selection manifest must use split {SPLIT}")
    slots = manifest.get("slots")
    if not isinstance(slots, list) or len(slots) != expected_slots:
        raise ProtocolError(f"Expected {expected_slots} selected slots")
    if manifest.get("slot_count") != expected_slots or manifest.get("worker_seed_base") != SLOT_SEED_BASE:
        raise ProtocolError("Selection manifest slot count or seed base is incompatible")
    for key in ("pool_order_sha256", "data_config_sha256", "source_manifest_sha256"):
        digest = manifest.get(key)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ProtocolError(f"Selection manifest has an invalid {key}")
    for i, row in enumerate(slots):
        slot, seed = row.get("slot"), row.get("worker_seed")
        if (not isinstance(slot, int) or isinstance(slot, bool) or slot != i
                or not isinstance(seed, int) or isinstance(seed, bool) or seed != SLOT_SEED_BASE + i):
            raise ProtocolError(f"Invalid slot or worker seed at position {i}")
        if not row.get("game_file") or not row.get("game_id"):
            raise ProtocolError(f"Missing game identity for slot {i}")
        if row.get("task_type") not in TASK_TYPES:
            raise ProtocolError(f"Unsupported or missing task type in slot {i}")


def shard_slots(manifest: Mapping[str, Any], rank: int, world_size: int = SHARD_COUNT) -> list[dict[str, Any]]:
    validate_selection_manifest(manifest)
    if world_size != SHARD_COUNT or not 0 <= rank < world_size:
        raise ProtocolError(f"This profile requires {SHARD_COUNT} ranks")
    start = rank * SLOTS_PER_SHARD
    return list(manifest["slots"][start:start + SLOTS_PER_SHARD])


def tokenize_upstream_prompt(tokenizer: Any, prompt: str,
                             max_tokens: int = MAX_PROMPT_TOKENS) -> tuple[str, list[int]]:
    """Apply the active SkillRise Qwen chat path and reject overlong prompts."""
    chat = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    token_ids = tokenizer.encode(rendered, add_special_tokens=False)
    if len(token_ids) > max_tokens:
        raise ProtocolError(
            f"Prompt has {len(token_ids)} tokens, over the upstream {max_tokens}-token limit; no truncation is allowed"
        )
    return rendered, list(token_ids)


def decode_vllm_token_ids(output: Any, tokenizer: Any) -> tuple[str, list[int]]:
    """Decode the generated IDs; intentionally ignore vLLM's `text` field."""
    ids = list(getattr(output, "token_ids", ()))
    return tokenizer.decode(ids, skip_special_tokens=True), ids


def upstream_engine_kwargs() -> dict[str, Any]:
    """vLLM SPMD engine settings from the GRPO shell plus active config defaults."""
    return {
        "dtype": "bfloat16", "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.5, "max_model_len": 5120,
        "max_num_seqs": 512, "enable_chunked_prefill": True,
        "max_num_batched_tokens": 8192, "enable_prefix_caching": True,
        "enforce_eager": False, "seed": 0,
    }


def verify_preflight(result: Mapping[str, Any], tag_rate_warning_below: float = 0.2) -> dict[str, Any]:
    """Check that GPU preflight exercised generation, parser, and the real environment."""
    rows = result.get("rows")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ProtocolError("Preflight must contain exactly two slot rows")
    parity = result.get("generation_parity")
    if not isinstance(parity, Mapping) or parity.get("equivalent") is not True:
        raise ProtocolError("Preflight is missing a successful source-parameter generation parity check")
    traces = [event for row in rows for attempt in row.get("attempts", [])
              for event in attempt.get("trace", [])]
    if not traces:
        raise ProtocolError("Preflight produced no generation/environment trace")
    if any(not event.get("prompt") or "raw_output" not in event or "observation" not in event
           or "observation_before" not in event
           or "executed_action" not in event for event in traces):
        raise ProtocolError("Preflight trace is missing prompt, generation, action, or observation evidence")
    tagged = sum(bool(event.get("valid_action_tag")) for event in traces)
    admissible = sum(bool(event.get("projected_is_admissible")) for event in traces)
    rate = tagged / len(traces)
    return {
        "events": len(traces),
        "generation_parity": "passed",
        "valid_tag_steps": tagged,
        "tag_rate": rate,
        "admissible_projected_steps": admissible,
        "diagnosis": "low_tag_rate_review" if rate < tag_rate_warning_below else "trace_complete",
        "warning_threshold": tag_rate_warning_below,
    }


def aggregate_rows(rows: Sequence[Mapping[str, Any]], expected_slots: int = SLOT_COUNT) -> dict[str, Any]:
    """Calculate attempt-0 Pass@1 and cumulative retry metrics without OR naming ambiguity."""
    if len(rows) != expected_slots:
        raise ProtocolError(f"Expected {expected_slots} slot rows, got {len(rows)}")
    by_slot = {}
    for row in rows:
        slot = row.get("slot")
        if not isinstance(slot, int) or isinstance(slot, bool) or not 0 <= slot < expected_slots:
            raise ProtocolError(f"Invalid slot id: {slot!r}")
        if slot in by_slot:
            raise ProtocolError(f"Duplicate result for slot {slot}")
        if not isinstance(row.get("attempts"), list) or len(row["attempts"]) != ATTEMPTS:
            raise ProtocolError(f"Slot {slot} must contain {ATTEMPTS} attempt records")
        for attempt_idx, attempt in enumerate(row["attempts"]):
            if attempt.get("attempt") != attempt_idx or not isinstance(attempt.get("won"), bool):
                raise ProtocolError(f"Invalid attempt result in slot {slot}")
        by_slot[slot] = row
    if set(by_slot) != set(range(expected_slots)):
        raise ProtocolError("Result slots are not a complete 0..N-1 grid")

    def metric(attempt_count: int) -> tuple[float, dict[str, float], dict[str, int]]:
        wins = [any(a["won"] for a in by_slot[s]["attempts"][:attempt_count])
                for s in range(expected_slots)]
        counts: dict[str, int] = defaultdict(int)
        type_wins: dict[str, int] = defaultdict(int)
        for s, won in enumerate(wins):
            typ = by_slot[s]["task_type"]
            counts[typ] += 1
            type_wins[typ] += int(won)
        per_type = {t: type_wins[t] / counts[t] for t in sorted(counts)}
        return sum(wins) / expected_slots, per_type, dict(counts)

    metrics = {}
    for count in range(1, ATTEMPTS + 1):
        weighted, per_type, counts = metric(count)
        macro_types = [per_type[t] for t in TASK_TYPES if t in per_type]
        missing_types = [t for t in TASK_TYPES if t not in per_type]
        metrics[f"pass_at_{count}"] = {
            "episode_weighted_sr": weighted,
            "six_type_macro_sr": sum(macro_types) / len(TASK_TYPES) if not missing_types else None,
            "available_type_macro_sr": sum(macro_types) / len(macro_types) if macro_types else None,
            "missing_task_types": missing_types,
            "per_type_sr": per_type,
            "per_type_slot_counts": counts,
            "label": "Pass@1" if count == 1 else f"cumulative Pass@{count} (code-path diagnostic)",
        }
    return metrics


def validate_shard_grid(shards: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any], samples: int = 1) -> list[dict[str, Any]]:
    """Strict merge validation: exactly eight compatible shards and every slot once."""
    validate_selection_manifest(manifest)
    if len(shards) != SHARD_COUNT:
        raise ProtocolError(f"Expected {SHARD_COUNT} shard files, got {len(shards)}")
    merged = []
    seen: set[int] = set()
    expected_manifest_hash = sha256_json(manifest)
    common = None
    for rank, shard in enumerate(shards):
        if shard.get("schema_version") != 1 or shard.get("profile") != PROFILE:
            raise ProtocolError(f"Shard {rank} uses an incompatible schema/profile")
        if shard.get("rank") != rank or shard.get("world_size") != SHARD_COUNT:
            raise ProtocolError(f"Shard identity mismatch at rank {rank}")
        if shard.get("selection_manifest_sha256") != expected_manifest_hash:
            raise ProtocolError(f"Selection manifest mismatch in rank {rank}")
        if shard.get("samples") != samples:
            raise ProtocolError(f"Sample count mismatch in rank {rank}")
        if not isinstance(shard.get("run_id"), str) or not shard["run_id"].strip():
            raise ProtocolError(f"Rank {rank} is missing its unique run_id")
        provenance = shard.get("prompt_provenance")
        if (not isinstance(provenance, Mapping) or provenance.get("paper_exact") is not False
                or provenance.get("kind") not in ("upstream prompt", "user-approved derived approximation")):
            raise ProtocolError(f"Rank {rank} is missing explicit non-paper-exact prompt provenance")
        if not isinstance(shard.get("glibc_compat"), Mapping) or not shard["glibc_compat"].get("mode"):
            raise ProtocolError(f"Rank {rank} is missing runtime glibc compatibility provenance")
        comparable = (shard.get("run_id"), shard.get("style"), shard.get("base_model"),
                      shard.get("model_identity"), shard.get("decode"), shard.get("engine"),
                      shard.get("tokenizer"), shard.get("software"), shard.get("upstream_sources"),
                      shard.get("vllm_attention_backend"), shard.get("prompt_provenance"),
                      shard.get("glibc_compat"))
        if common is None:
            common = comparable
        elif comparable != common:
            raise ProtocolError("Ranks used incompatible run, style, model, tokenizer, source, software, or decode parameters")
        rows = shard.get("rows")
        want = shard_slots(manifest, rank)
        if not isinstance(rows, list) or len(rows) != len(want):
            raise ProtocolError(f"Rank {rank} must report {len(want)} slot rows")
        got = [row.get("slot") for row in rows]
        want_ids = [slot["slot"] for slot in want]
        if got != want_ids:
            raise ProtocolError(f"Rank {rank} slot rows must be in declared contiguous order")
        for row, expected in zip(rows, want):
            slot = row["slot"]
            if (row.get("worker_seed") != expected["worker_seed"]
                    or row.get("game_file") != expected["game_file"]
                    or row.get("game_id") != expected["game_id"]
                    or row.get("task_type") != expected["task_type"]
                    or row.get("sample") != 0):
                raise ProtocolError(f"Row identity does not match selection manifest for slot {slot}")
            if slot in seen:
                raise ProtocolError(f"Duplicate slot {slot} across shards")
            seen.add(slot)
        merged.extend(rows)
    if seen != set(range(SLOT_COUNT)):
        raise ProtocolError("Merged shard set does not cover all 128 slots")
    return sorted(merged, key=lambda r: r["slot"])
