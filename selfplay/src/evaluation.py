"""CPU-only contracts and result checks for the unified SkillRise/ReAct evaluation."""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

SCHEMA = "skillrise-ez-eval-v1"
SPLITS = {"valid_seen": "eval_in_distribution"}
SEEDS = (0, 1, 2, 3, 4)
TASK_TYPES = (
    "pick_and_place_simple", "look_at_obj_in_light",
    "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep", "pick_two_obj_and_place",
)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def request_seed(seed: int, task_id: str, turn: int) -> int:
    raw = f"{int(seed)}\0{task_id}\0turn\0{int(turn)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def align_task_slots(tasks: list[dict], used_games: list[str]) -> list[dict]:
    """Reorder task metadata to the environment's returned slot order, preserving duplicates."""
    if len(tasks) != len(used_games):
        raise ValueError("requested task count and reset slot count differ")
    buckets = {}
    for task in tasks:
        buckets.setdefault(task["game_file"], []).append(task)
    aligned = []
    for game in used_games:
        queue = buckets.get(game)
        if not queue:
            raise ValueError(f"environment returned an unexpected or excess task slot: {game}")
        aligned.append(queue.pop(0))
    if any(queue for queue in buckets.values()):
        raise ValueError("environment omitted one or more assigned task slots")
    return aligned


def discover_suite(config_path: str, split: str = "valid_seen", shards: int = 8) -> dict:
    if split not in SPLITS or shards < 1:
        raise ValueError("only valid_seen and a positive shard count are supported")
    from alfworld_compat import find_compat_dir, install
    compat_dir = find_compat_dir()
    if compat_dir:
        install(verbose=False)
        compat = {"mode": "local", "directory": str(Path(compat_dir).resolve()),
                  "downward_sha256": sha256_file(Path(compat_dir) / "libdownward.so"),
                  "shim_sha256": sha256_file(Path(compat_dir) / "libglibc_single_threaded_shim.so")}
    else:
        try:
            import fast_downward
            lib = fast_downward.load_lib(); fast_downward.close_lib(lib)
            compat = {"mode": "native"}
        except Exception as exc:
            raise ValueError("ALFWorld planner is unavailable; no compatibility shim will be installed") from exc

    from alfworld_env import collect_game_files, load_alfworld_config, task_instance_id, task_type_of
    cfg_path = Path(config_path).expanduser().resolve(strict=True)
    cfg = load_alfworld_config(str(cfg_path))
    split_name = SPLITS[split]
    # Retain the environment's published pool order. A complete task ID is required exactly once.
    games = [str(Path(p).resolve(strict=True)) for p in collect_game_files(cfg, split_name)]
    ids = [task_instance_id(p) for p in games]
    if not games or len(ids) != len(set(ids)):
        raise ValueError("valid_seen task pool is empty or has duplicate task IDs")
    tasks = [{"task_id": tid, "task_type": task_type_of(path), "game_file": path,
              "game_sha256": sha256_file(path)} for tid, path in zip(ids, games)]
    import inspect
    import textworld
    from alfworld_env import ALFWorldBatchEnv
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
    from checkpoint import environment_file_identity
    env_sources = {
        "alfworld_env": sha256_file(Path(inspect.getfile(ALFWorldBatchEnv)).resolve()),
        "alfred_tw_env": sha256_file(Path(inspect.getfile(AlfredTWEnv)).resolve()),
        "compatibility": compat,
        "textworld_version": str(getattr(textworld, "__version__", "unknown")),
        "environment_files": environment_file_identity(cfg, cfg_path, [Path(p) for p in games]),
    }
    config_hash = sha256_file(cfg_path)
    pool_hash = canonical_hash({"split": split, "config_sha256": config_hash,
        "environment_sources": env_sources,
        "tasks": [{k: t[k] for k in ("task_id", "task_type", "game_sha256")} for t in tasks]})
    n_shards = min(shards, len(tasks)); q, rem = divmod(len(tasks), n_shards)
    assignments, offset = [], 0
    for rank in range(n_shards):
        size = q + (rank < rem)
        subset = tasks[offset:offset + size]
        assignments.append({"rank": rank, "task_ids": [t["task_id"] for t in subset]})
        offset += size
    return {"schema": SCHEMA, "split": split, "alfworld_split": split_name,
        "config_path": str(cfg_path), "config_sha256": config_hash,
        "environment_sources": env_sources, "pool_sha256": pool_hash,
        "tasks": tasks, "shards": assignments}


def validate_suite(suite: dict) -> None:
    if suite.get("schema") != SCHEMA or suite.get("split") not in SPLITS:
        raise ValueError("unsupported task-pool manifest")
    tasks = suite.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("task pool is empty")
    ids = [t.get("task_id") for t in tasks]
    if any(not isinstance(x, str) or not x for x in ids) or len(ids) != len(set(ids)):
        raise ValueError("task IDs are missing or duplicated")
    for task in tasks:
        path = Path(task["game_file"])
        if not path.is_file() or sha256_file(path) != task.get("game_sha256"):
            raise ValueError(f"task file is missing or changed: {task.get('task_id')}")
    expected_pool = canonical_hash({"split": suite["split"], "config_sha256": suite["config_sha256"],
        "environment_sources": suite["environment_sources"],
        "tasks": [{k: t[k] for k in ("task_id", "task_type", "game_sha256")} for t in tasks]})
    if expected_pool != suite.get("pool_sha256"):
        raise ValueError("task pool manifest hash is invalid")
    config_path = Path(suite.get("config_path", ""))
    if not config_path.is_file() or sha256_file(config_path) != suite.get("config_sha256"):
        raise ValueError("ALFWorld configuration changed or is missing")
    flat = [tid for shard in suite.get("shards", []) for tid in shard.get("task_ids", [])]
    if flat != ids or len(flat) != len(set(flat)):
        raise ValueError("rank assignments do not partition the full task pool")


def summarize(rows: list[dict], expected_tasks: list[dict], seeds: tuple[int, ...] = SEEDS,
              *, require_six_types: bool = True) -> dict:
    task_by_id = {t["task_id"]: t for t in expected_tasks}
    expected_keys = {(tid, seed) for tid in task_by_id for seed in seeds}
    seen = {}
    for row in rows:
        seed = row.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("episode seed must be an integer")
        key = (row.get("task_id"), seed)
        if key not in expected_keys or key in seen:
            raise ValueError(f"unexpected or duplicate task/seed row: {key}")
        if row.get("task_type") != task_by_id[key[0]]["task_type"] or not isinstance(row.get("won"), bool):
            raise ValueError(f"invalid task type or won value for {key}")
        if isinstance(row.get("environment_reset_seed"), bool) or row.get("environment_reset_seed") != key[1]:
            raise ValueError(f"environment reset seed mismatch for {key}")
        raw = dict(row); row_digest = raw.pop("record_sha256", None)
        if row_digest != canonical_hash(raw):
            raise ValueError(f"episode record checksum mismatch for {key}")
        if row.get("steps") != len(row.get("trace", [])) or not 1 <= row["steps"] <= 30:
            raise ValueError(f"episode trace length mismatch for {key}")
        trace = row["trace"]
        if row.get("won") != trace[-1].get("won"):
            raise ValueError(f"episode won flag differs from terminal trace for {key}")
        if not trace[-1].get("done") and row["steps"] != 30:
            raise ValueError(f"episode ended before done or the turn limit for {key}")
        from protocol import project_skillrise_action
        for turn, item in enumerate(row.get("trace", [])):
            if (item.get("turn") != turn or not isinstance(item.get("raw_prompt"), str) or
                    not isinstance(item.get("prompt_token_ids"), list) or
                    not isinstance(item.get("output_token_ids"), list) or
                    not isinstance(item.get("raw_output"), str) or
                    not isinstance(item.get("executed_action"), str) or
                    not isinstance(item.get("observation"), str) or
                    not isinstance(item.get("won"), bool) or not isinstance(item.get("done"), bool) or
                    not isinstance(item.get("truncated"), bool) or
                    isinstance(item.get("stop_len"), bool) or item.get("stop_len") != len(item.get("output_token_ids", [])) or
                    isinstance(item.get("request_seed"), bool) or not isinstance(item.get("request_seed"), int) or
                    item.get("request_seed") != request_seed(key[1], key[0], turn)):
                raise ValueError(f"malformed rollout evidence for {key} at turn {turn}")
            for field in ("prompt_token_ids", "output_token_ids"):
                if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in item[field]):
                    raise ValueError(f"invalid token evidence in {key} turn {turn}")
            projection, tag_valid = project_skillrise_action(item.get("raw_output", ""))
            if (projection != item.get("parser_projection") or bool(tag_valid) != item.get("action_tag_valid") or
                    item.get("executed_action") != (projection if projection else "look")):
                raise ValueError(f"parser/action evidence mismatch in {key} turn {turn}")
            if item["done"] and turn != len(trace) - 1:
                raise ValueError(f"rollout continued after terminal done in {key}")
        seen[key] = row["won"]
    if set(seen) != expected_keys:
        raise ValueError(f"task/seed grid incomplete: missing {len(expected_keys - set(seen))} rows")
    by_seed = {seed: sum(seen[(tid, seed)] for tid in task_by_id) / len(task_by_id) for seed in seeds}
    by_type = {}
    for task_type in TASK_TYPES:
        ids = [tid for tid, task in task_by_id.items() if task["task_type"] == task_type]
        if not ids and require_six_types:
            raise ValueError(f"full valid_seen pool is missing task type {task_type}")
        if not ids:
            continue
        by_type[task_type] = {seed: sum(seen[(tid, seed)] for tid in ids) / len(ids) for seed in seeds}
    episode_sr = sum(seen.values()) / len(seen)
    type_means = {kind: statistics.mean(vals.values()) for kind, vals in by_type.items()}
    all_types = len(by_type) == len(TASK_TYPES)
    type_macro_by_seed = ({seed: statistics.mean(by_type[kind][seed] for kind in TASK_TYPES) for seed in seeds}
                          if all_types else None)
    macro_values = list(type_macro_by_seed.values()) if type_macro_by_seed is not None else []
    return {"episode_weighted_sr": episode_sr,
        "per_seed_episode_sr": {str(k): v for k, v in by_seed.items()},
        "per_seed_task_type_macro_sr": ({str(k): v for k, v in type_macro_by_seed.items()}
                                         if type_macro_by_seed is not None else None),
        "episode_sr_mean_across_seeds": statistics.mean(by_seed.values()),
        "episode_sr_sd_across_seeds": statistics.stdev(by_seed.values()),
        "six_type_macro_sr": statistics.mean(type_means.values()) if all_types else None,
        "six_type_macro_mean_across_seeds": statistics.mean(macro_values) if all_types else None,
        "six_type_macro_sd_across_seeds": statistics.stdev(macro_values) if all_types else None,
        "per_task_type_macro_sr": type_means,
        "per_task_type_seed_sr": {k: {str(s): v for s, v in vals.items()} for k, vals in by_type.items()},
        "n_tasks": len(task_by_id), "n_episodes": len(rows), "seed_count": len(seeds)}


def validate_shard(contract: dict, result: dict, rank: int) -> None:
    shard = contract["suite"]["shards"][rank]
    if result.get("schema") != SCHEMA or result.get("contract_sha256") != contract["contract_sha256"]:
        raise ValueError("shard schema or immutable contract mismatch")
    if result.get("run_id") != contract.get("run_id"):
        raise ValueError("shard belongs to a different prepared run")
    if result.get("rank") != rank or result.get("task_ids") != shard["task_ids"]:
        raise ValueError("shard rank/task assignment mismatch")
    if result.get("protocol_signature") != contract["protocol_signature"]:
        raise ValueError("shard protocol identity mismatch")
    if result.get("arm_identity") != contract.get("arm_identity"):
        raise ValueError("shard base/checkpoint identity mismatch")
    if result.get("base_inventory") != contract.get("base_inventory"):
        raise ValueError("shard base inventory mismatch")
    signature = contract["protocol_signature"]
    if result.get("software") != signature.get("software") or result.get("runtime_source_sha256") != signature.get("runtime_source_sha256"):
        raise ValueError("shard software/source identity mismatch")
    expected = [t for t in contract["suite"]["tasks"] if t["task_id"] in set(shard["task_ids"])]
    expected_by_id = {t["task_id"]: t for t in expected}
    for row in result.get("rows", []):
        task = expected_by_id.get(row.get("task_id"))
        if task is None or any(row.get(k) != task.get(k) for k in ("task_type", "game_file", "game_sha256")):
            raise ValueError("shard episode is outside its prepared task assignment")
    summarize(result.get("rows", []), expected, require_six_types=False)


def merge(contract: dict, shards: list[dict]) -> dict:
    raw = dict(contract); digest = raw.pop("contract_sha256", None)
    if digest != canonical_hash(raw):
        raise ValueError("immutable run contract hash is invalid")
    validate_suite(contract["suite"])
    if len(shards) != len(contract["suite"]["shards"]):
        raise ValueError("missing rank shard")
    all_rows = []
    for rank, result in enumerate(shards):
        validate_shard(contract, result, rank)
        all_rows.extend(result["rows"])
    summary = summarize(all_rows, contract["suite"]["tasks"])
    return {"schema": SCHEMA, "run_id": contract["run_id"], "contract_sha256": digest,
        "protocol_signature": contract["protocol_signature"],
        "arm_identity": contract["arm_identity"], "base_inventory": contract["base_inventory"],
        "software": contract["protocol_signature"]["software"],
        "runtime_source_sha256": contract["protocol_signature"]["runtime_source_sha256"], "summary": summary,
        "rows": all_rows, "expected_task_ids": [t["task_id"] for t in contract["suite"]["tasks"]]}


def compare(base: dict, trained: dict) -> dict:
    if base.get("schema") != SCHEMA or trained.get("schema") != SCHEMA:
        raise ValueError("comparison inputs are not unified evaluator results")
    if base.get("protocol_signature") != trained.get("protocol_signature"):
        raise ValueError("baseline and trained solver do not share the same evaluation protocol")
    signature = base["protocol_signature"]
    required = ("pool_sha256", "split", "config_sha256", "config", "solver_settings",
        "prompt_source_sha256", "tokenizer_files_sha256", "raw_prompt_encoding", "action_parser",
        "stop", "seed_policy", "seeds", "max_turns", "max_context_tokens", "max_new_tokens",
        "temperature", "top_p", "top_k", "min_p", "repetition_penalty", "backend", "dtype",
        "base_inventory_sha256", "alfworld_data_root", "software", "runtime_source_sha256")
    if any(key not in signature or signature[key] is None for key in required):
        raise ValueError("paired results lack required common protocol identity fields")
    if base.get("arm_identity", {}).get("kind") != "base-only" or trained.get("arm_identity", {}).get("kind") != "checkpoint":
        raise ValueError("comparison requires a base-only run and a sealed checkpoint run")
    if (base["arm_identity"].get("base_inventory_sha256") != signature["base_inventory_sha256"] or
            trained["arm_identity"].get("base_inventory_sha256") != signature["base_inventory_sha256"]):
        raise ValueError("paired arms do not use the exact prepared base model")
    if base.get("expected_task_ids") != trained.get("expected_task_ids"):
        raise ValueError("baseline and trained task pools differ")
    def index(result):
        keyed = {}
        for row in result.get("rows", []):
            seed, tid = row.get("seed"), row.get("task_id")
            if isinstance(seed, bool) or not isinstance(seed, int) or not isinstance(tid, str):
                raise ValueError("paired result has malformed task/seed row")
            key = (tid, seed)
            if key in keyed:
                raise ValueError("duplicate paired task/seed records")
            raw = dict(row); row_digest = raw.pop("record_sha256", None)
            if row_digest != canonical_hash(raw) or not isinstance(row.get("won"), bool):
                raise ValueError("paired result has invalid or corrupted episode record")
            keyed[key] = row
        expected = {(tid, seed) for tid in result["expected_task_ids"] for seed in SEEDS}
        if not result.get("expected_task_ids") or set(keyed) != expected:
            raise ValueError("paired result task/seed grid is incomplete or unexpected")
        return keyed
    a, b = index(base), index(trained)
    if a.keys() != b.keys():
        raise ValueError("baseline and trained task/seed grids differ")
    delta = {str(seed): statistics.mean(int(b[(tid, seed)]["won"]) - int(a[(tid, seed)]["won"])
            for tid in base["expected_task_ids"]) for seed in SEEDS}
    per_seed_macro_delta = {str(seed): (trained["summary"]["per_seed_task_type_macro_sr"][str(seed)] -
            base["summary"]["per_seed_task_type_macro_sr"][str(seed)])
            for seed in SEEDS}
    if any(not isinstance(result.get("contract_sha256"), str) or len(result["contract_sha256"]) != 64
           for result in (base, trained)):
        raise ValueError("paired result lacks prepared contract identity")
    return {"schema": SCHEMA, "baseline_contract_sha256": base["contract_sha256"],
        "trained_contract_sha256": trained["contract_sha256"],
        "paired_episode_sr_delta_by_seed": delta,
        "paired_episode_sr_delta_mean": statistics.mean(delta.values()),
        "paired_episode_sr_delta_sd": statistics.stdev(delta.values()),
        "paired_six_type_macro_delta_by_seed": per_seed_macro_delta,
        "paired_six_type_macro_delta_mean": statistics.mean(per_seed_macro_delta.values()),
        "paired_six_type_macro_delta_sd": statistics.stdev(per_seed_macro_delta.values()),
        "baseline_summary": base["summary"], "trained_summary": trained["summary"],
        "claim_scope": "local matched comparison; not an official paper reproduction"}
