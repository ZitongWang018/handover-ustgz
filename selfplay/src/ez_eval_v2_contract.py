"""Strict, CPU-only contracts for Explorer-Zero evaluation v2."""
from __future__ import annotations

import hashlib
import json
import os
import statistics
from pathlib import Path

SCHEMA = "ez-eval-v2"
SPLITS = {"valid_unseen": "eval_out_of_distribution", "valid_seen": "eval_in_distribution"}
SAMPLES = 4
SHARDS = 8


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                  ensure_ascii=False).encode("utf-8")).hexdigest()


def ensure_environment_compat() -> dict:
    """Use the existing local shim when present, otherwise prove native planner works."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from alfworld_compat import find_compat_dir, install
    compat = find_compat_dir()
    if compat:
        install(verbose=False)
        return {"mode": "local_compat", "directory": str(Path(compat).resolve()),
                "libdownward_sha256": sha256_file(Path(compat) / "libdownward.so"),
                "shim_sha256": sha256_file(Path(compat) / "libglibc_single_threaded_shim.so")}
    try:
        import fast_downward
        lib = fast_downward.load_lib()
        fast_downward.close_lib(lib)
        return {"mode": "native", "fast_downward": "load_lib_success"}
    except Exception as exc:
        raise ValueError("native fast_downward unavailable and no existing ALFWorld compatibility shim found") from exc


def discover_suite(config_path: str, split: str, *, shards: int = SHARDS) -> dict:
    """Discover the installed ALFWorld pool; this imports no model/GPU modules."""
    if split not in SPLITS:
        raise ValueError(f"unsupported suite split: {split}")
    if shards < 1:
        raise ValueError("shards must be positive")
    compat = ensure_environment_compat()
    # Resolve local config without permitting ALFWorld to install/download anything.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from alfworld_env import collect_game_files, load_alfworld_config, task_instance_id, task_type_of
    cfg_path = Path(config_path).expanduser().resolve(strict=True)
    cfg = load_alfworld_config(str(cfg_path))
    games = sorted(str(Path(p).resolve(strict=True)) for p in collect_game_files(cfg, SPLITS[split]))
    if not games or any(not Path(p).is_file() for p in games):
        raise ValueError("discovered ALFWorld suite is empty or contains missing files")
    ids = [task_instance_id(p) for p in games]
    if len(ids) != len(set(ids)):
        raise ValueError("discovered task IDs are not unique")
    tasks = [{"task_id": tid, "task_type": task_type_of(path), "game_file": path,
              "game_sha256": sha256_file(path)} for tid, path in zip(ids, games)]
    cfg_hash = sha256_file(cfg_path)
    import inspect
    from alfworld_env import ALFWorldBatchEnv
    env_sources = {"alfworld_env.py": sha256_file(Path(inspect.getfile(ALFWorldBatchEnv)).resolve()),
                   "compatibility": compat}
    from ez_train_v2_contract import environment_file_identity
    env_sources["environment_files"] = environment_file_identity(
        cfg, cfg_path, [Path(p) for p in games])
    try:
        from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
        env_sources["alfred_tw_env.py"] = sha256_file(Path(inspect.getfile(AlfredTWEnv)).resolve())
    except (ImportError, OSError):
        raise ValueError("installed ALFWorld environment source is unavailable")
    try:
        import textworld
        env_sources["textworld_version"] = str(getattr(textworld, "__version__", "unknown"))
    except ImportError:
        raise ValueError("TextWorld is not installed")
    pool_hash = canonical_hash({"split": split, "config_sha256": cfg_hash,
                                "environment_sources": env_sources,
                                "tasks": [{k: t[k] for k in ("task_id", "task_type", "game_sha256")}
                                          for t in tasks]})
    # Balanced contiguous split; when |pool| < n_shards only non-empty shards exist.
    n = min(shards, len(tasks)); q, r = divmod(len(tasks), n)
    specs, offset = [], 0
    for i in range(n):
        size = q + (1 if i < r else 0)
        subset = tasks[offset:offset + size]
        specs.append({"rank": i, "offset": offset, "task_ids": [t["task_id"] for t in subset],
                      "task_hash": canonical_hash(subset)})
        offset += size
    return {"schema": SCHEMA, "split": split, "alfworld_split": SPLITS[split],
            "config_path": str(cfg_path), "config_sha256": cfg_hash,
            "environment_sources": env_sources,
            "pool_sha256": pool_hash, "tasks": tasks, "n_tasks": len(tasks),
            "n_task_types": len({t["task_type"] for t in tasks}),
            "requested_shards": shards, "shards": specs,
            "sample_seed_policy": "sha256(run_seed,task_id,sample_index) -> uint32",
            "samples_per_task": SAMPLES}


def validate_suite(suite: dict, *, config_path: str | None = None) -> None:
    if suite.get("schema") != SCHEMA or suite.get("split") not in SPLITS:
        raise ValueError("unsupported suite schema or split")
    if suite.get("alfworld_split") != SPLITS[suite["split"]]:
        raise ValueError("suite logical split does not match ALFWorld runtime split")
    tasks = suite.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("suite task pool is empty")
    if suite.get("n_tasks") != len(tasks):
        raise ValueError("suite task count mismatch")
    if config_path is not None and sha256_file(Path(config_path).resolve(strict=True)) != suite.get("config_sha256"):
        raise ValueError("ALFWorld config hash differs from prepared suite")
    seen = set()
    for t in tasks:
        p = Path(t["game_file"])
        if t["task_id"] in seen or not p.is_file() or sha256_file(p) != t.get("game_sha256"):
            raise ValueError(f"duplicate, missing, or changed suite task: {t.get('task_id')}")
        seen.add(t["task_id"])
    expected = canonical_hash({"split": suite["split"], "config_sha256": suite["config_sha256"],
                               "environment_sources": suite.get("environment_sources"),
                               "tasks": [{k: t[k] for k in ("task_id", "task_type", "game_sha256")}
                                         for t in tasks]})
    if expected != suite.get("pool_sha256"):
        raise ValueError("suite content hash mismatch")
    flat = [tid for s in suite.get("shards", []) for tid in s.get("task_ids", [])]
    if flat != [t["task_id"] for t in tasks] or len(flat) != len(set(flat)):
        raise ValueError("suite shards do not exactly partition the full pool")
    for shard in suite.get("shards", []):
        selected = set(shard.get("task_ids", []))
        subset = [t for t in tasks if t["task_id"] in selected]
        if canonical_hash(subset) != shard.get("task_hash"):
            raise ValueError("suite shard content hash mismatch")


def sample_seed(run_seed: int, task_id: str, sample_index: int) -> int:
    if not 0 <= sample_index < SAMPLES:
        raise ValueError("sample index outside complete four-sample grid")
    raw = f"{int(run_seed)}\0{task_id}\0{sample_index}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def request_seed(sample_base_seed: int, turn_index: int) -> int:
    raw = f"{int(sample_base_seed)}\0turn\0{int(turn_index)}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def task_identity(path: str) -> dict:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from alfworld_env import task_instance_id, task_type_of
    return {"task_id": task_instance_id(path), "task_type": task_type_of(path),
            "game_sha256": sha256_file(path)}


def strict_merge(expected: dict, shard_results: list[dict]) -> dict:
    """Merge explicitly ordered shard documents against a prepared suite."""
    suite = expected["suite"]
    contract_hash = expected.get("contract_sha256")
    check_contract = dict(expected); check_contract.pop("contract_sha256", None)
    if contract_hash != canonical_hash(check_contract):
        raise ValueError("prepared run contract hash is invalid")
    validate_suite(expected["suite"])
    specs = expected.get("expected_shards")
    if not isinstance(specs, list) or len(specs) != len(shard_results) or len(specs) != len(suite["shards"]) or not specs:
        raise ValueError("expected shard list and supplied shard results must be the same nonzero length")
    if len({str(Path(s.get("path", "")).resolve()) for s in specs}) != len(specs):
        raise ValueError("prepared run contract repeats a shard result path")
    if any(spec.get("rank") != suite_spec.get("rank") or spec.get("task_ids") != suite_spec.get("task_ids")
           for spec, suite_spec in zip(specs, suite["shards"])):
        raise ValueError("prepared run shard assignment differs from the immutable suite")
    common = expected.get("protocol_signature")
    try:
        _validate_pair_signature(common)
    except ValueError as exc:
        raise ValueError("merge contract lacks a complete prepared protocol signature") from exc
    if (common["pool_sha256"] != suite["pool_sha256"] or common["split"] != suite["split"] or
            common["samples_per_task"] != SAMPLES or common["base_inventory_sha256"] !=
            expected.get("arm_identity", {}).get("base_inventory_sha256")):
        raise ValueError("merge contract lacks a complete prepared protocol signature")
    all_rows, seen = [], set()
    expected_by_id = {t["task_id"]: t for t in expected["suite"]["tasks"]}
    for spec, data in zip(specs, shard_results):
        if data.get("schema") != SCHEMA or data.get("run_id") != expected.get("run_id"):
            raise ValueError("stale/wrong-run shard or result schema")
        if data.get("contract_sha256") != contract_hash:
            raise ValueError("shard was not produced from the prepared run contract")
        if data.get("pool_sha256") != expected["suite"]["pool_sha256"] or data.get("split") != expected["suite"]["split"]:
            raise ValueError("cross-suite or cross-split shard")
        if data.get("protocol_signature") != common:
            raise ValueError("incompatible protocol signature")
        if data.get("arm_identity") != expected.get("arm_identity"):
            raise ValueError("mixed base/adapter identity")
        for evidence_key in ("base_inventory", "software", "runtime_source_sha256"):
            if data.get(evidence_key) != expected.get(evidence_key):
                raise ValueError(f"inconsistent shard evidence: {evidence_key}")
        ids = spec.get("task_ids", [])
        if data.get("rank") != spec.get("rank") or data.get("task_ids") != ids:
            raise ValueError("shard task/rank declaration mismatch")
        rows = data.get("sample_rows")
        if not isinstance(rows, list) or len(rows) != len(ids) * SAMPLES:
            raise ValueError("shard sample grid incomplete")
        local = set()
        for row in rows:
            tid, si = row.get("task_id"), row.get("sample_index")
            task = expected_by_id.get(tid)
            if task is None or tid not in ids or row.get("task_type") != task["task_type"] or row.get("game_file") != task["game_file"] or row.get("game_sha256") != task["game_sha256"]:
                raise ValueError("unexpected, stale, or corrupted sample row")
            key = (tid, si)
            if isinstance(si, bool) or not isinstance(si, int) or not 0 <= si < SAMPLES or key in local or key in seen:
                raise ValueError("duplicate/invalid task sample")
            if not isinstance(row.get("won"), bool) or isinstance(row.get("seed"), bool) or not isinstance(row.get("seed"), int):
                raise ValueError("sample row lacks typed won/seed")
            if row["seed"] != sample_seed(common["run_seed"], tid, si):
                raise ValueError("sample seed does not match declared task/sample seed policy")
            raw_row = dict(row); row_hash = raw_row.pop("record_sha256", None)
            if row_hash != canonical_hash(raw_row):
                raise ValueError("sample row content hash is missing or corrupted")
            trace = row.get("trace")
            steps = row.get("steps")
            if (isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0 or
                    steps > common["max_steps"] or not isinstance(trace, list) or len(trace) != steps):
                raise ValueError("sample trace does not match executed step count")
            actions = row.get("actions")
            if not isinstance(actions, list) or len(actions) != steps:
                raise ValueError("sample action list does not match executed step count")
            if not isinstance(row.get("initial_prompt_token_ids"), list) or not all(
                    isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in row["initial_prompt_token_ids"]):
                raise ValueError("sample row lacks its initial rendered prompt token IDs")
            previous_prompt, previous_output = None, None
            for turn, item in enumerate(trace):
                required = ("step", "prompt_token_ids", "output_token_ids", "raw_output",
                            "parser_projection", "executed_command", "observation", "won", "done",
                            "admissible_actions", "finish_reason", "truncated", "stop_token_count", "request_seed",
                            "resolver_mode")
                if any(k not in item for k in required) or item["step"] != turn:
                    raise ValueError("sample trace is missing prompt/output/parser/action/stop evidence")
                if not isinstance(item["won"], bool) or not isinstance(item["done"], bool) or not isinstance(item["truncated"], bool):
                    raise ValueError("sample trace has invalid terminal/truncation flags")
                if not isinstance(item["prompt_token_ids"], list) or not isinstance(item["output_token_ids"], list):
                    raise ValueError("sample trace token ID fields are malformed")
                for name in ("prompt_token_ids", "output_token_ids"):
                    if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in item[name]):
                        raise ValueError("sample trace token IDs must be nonnegative integers")
                if turn == 0 and item["prompt_token_ids"] != row["initial_prompt_token_ids"]:
                    raise ValueError("first trace prompt differs from recorded initial prompt")
                if turn and item["prompt_token_ids"][:len(previous_prompt) + len(previous_output)] != previous_prompt + previous_output:
                    raise ValueError("later trace prompt does not preserve previous prompt/output token prefix")
                previous_prompt, previous_output = item["prompt_token_ids"], item["output_token_ids"]
                if item["executed_command"] != actions[turn]:
                    raise ValueError("sample action list differs from executed trace command")
                projection = item["parser_projection"]
                if not isinstance(projection, dict):
                    raise ValueError("sample parser projection must be an object")
                if projection.get("kind") == "action" and projection.get("action") != item["executed_command"]:
                    raise ValueError("valid parser projection differs from executed command")
                if projection.get("kind") in ("malformed", "control") and item["resolver_mode"] == "v2_recovery_disabled" and item["executed_command"] != "look":
                    raise ValueError("v2 malformed/control output must project to look")
                if item["done"] and turn != steps - 1:
                    raise ValueError("trace contains actions after terminal environment state")
                if not isinstance(item["request_seed"], int) or isinstance(item["request_seed"], bool):
                    raise ValueError("sample trace request seed is malformed")
                if item["request_seed"] != request_seed(row["seed"], turn):
                    raise ValueError("sample trace request seed differs from declared turn seed")
            if bool(row["won"]) != bool(trace[-1]["won"]):
                raise ValueError("sample win flag differs from final trace terminal state")
            if not trace[-1]["done"] and steps != common["max_steps"]:
                raise ValueError("nonterminal sample ended before declared step budget")
            local.add(key); seen.add(key)
        if local != {(tid, s) for tid in ids for s in range(SAMPLES)}:
            raise ValueError("shard does not cover exact task×sample grid")
        all_rows.extend(rows)
    wanted = {(t["task_id"], s) for t in expected["suite"]["tasks"] for s in range(SAMPLES)}
    if seen != wanted:
        raise ValueError("merged result has missing or extra samples")
    types = sorted({t["task_type"] for t in expected["suite"]["tasks"]})
    wins = [r["won"] for r in all_rows]
    suite_srs = [sum(r["won"] for r in all_rows if r["sample_index"] == i) / len(expected_by_id) for i in range(SAMPLES)]
    by_type = {}
    for typ in types:
        ids = [tid for tid, t in expected_by_id.items() if t["task_type"] == typ]
        by_type[typ] = sum(r["won"] for r in all_rows if r["task_id"] in ids) / (len(ids) * SAMPLES)
    by_type_full = {typ: by_type.get(typ) for typ in ("pick_and_place_simple", "look_at_obj_in_light", "pick_two_obj_and_place", "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep")}
    return {"schema": SCHEMA, "run_id": expected["run_id"], "pool_sha256": expected["suite"]["pool_sha256"],
            "split": expected["suite"]["split"], "protocol_signature": common,
            "expected_task_ids": [t["task_id"] for t in expected["suite"]["tasks"]],
            "arm_identity": expected["arm_identity"], "summary": {
                "episode_weighted_sr": sum(wins) / len(wins), "sample_suite_sr": suite_srs,
                "sample_suite_sd_ddof0": statistics.pstdev(suite_srs),
                "sample_suite_sd_ddof1": statistics.stdev(suite_srs),
                "task_type_sr": by_type_full,
                "six_type_macro_sr": (sum(by_type_full.values()) / 6 if all(v is not None for v in by_type_full.values()) else None),
                "n_tasks": len(expected_by_id), "n_episodes": len(wins), "samples_per_task": SAMPLES},
            "sample_rows": all_rows}


PAIR_FIELDS = ("pool_sha256", "split", "samples_per_task", "protocol", "prompt_sha256", "tokenizer_sha256", "scaffold_sha256", "chat_template_sha256", "parser", "menu", "stop", "decode", "max_context_tokens", "max_new_tokens", "max_steps", "seed_policy", "run_seed", "backend", "base_inventory_sha256", "attention_backend", "dtype", "software", "runtime_source_sha256", "resolver", "fewshot", "parser_strict")


def _validate_pair_signature(signature: dict) -> None:
    if not isinstance(signature, dict) or any(k not in signature for k in PAIR_FIELDS):
        raise ValueError("protocol signature lacks required evaluation fields")
    str_fields = ("pool_sha256", "split", "protocol", "prompt_sha256", "tokenizer_sha256",
        "scaffold_sha256", "parser", "stop", "seed_policy", "backend", "base_inventory_sha256",
        "attention_backend", "dtype", "resolver")
    if any(not isinstance(signature[k], str) or not signature[k] for k in str_fields):
        raise ValueError("protocol signature contains missing/non-string identity fields")
    if signature["chat_template_sha256"] is not None and not isinstance(signature["chat_template_sha256"], str):
        raise ValueError("protocol chat template hash must be a string or null")
    if any(isinstance(signature[k], bool) or not isinstance(signature[k], int) for k in
           ("samples_per_task", "max_context_tokens", "max_new_tokens", "max_steps", "run_seed")):
        raise ValueError("protocol signature contains invalid integer fields")
    if any(not isinstance(signature[k], bool) for k in ("menu", "fewshot", "parser_strict")):
        raise ValueError("protocol signature contains invalid boolean fields")
    decode = signature["decode"]
    if not isinstance(decode, dict) or any(k not in decode for k in ("temperature", "top_p", "top_k")):
        raise ValueError("protocol signature lacks complete decoding settings")
    if any(not isinstance(signature[k], dict) for k in ("software", "runtime_source_sha256")):
        raise ValueError("protocol software/source evidence must be objects")


def compare_pair(base: dict, adapter: dict) -> dict:
    if base.get("schema") != SCHEMA or adapter.get("schema") != SCHEMA:
        raise ValueError("paired inputs must be merged ez-eval-v2 results")
    a, b = base.get("protocol_signature", {}), adapter.get("protocol_signature", {})
    _validate_pair_signature(a); _validate_pair_signature(b)
    diff = [k for k in PAIR_FIELDS if a.get(k) != b.get(k)]
    if diff:
        raise ValueError("paired protocol mismatch: " + ", ".join(diff))
    if base.get("arm_identity", {}).get("kind") != "base-only" or adapter.get("arm_identity", {}).get("kind") != "adapter":
        raise ValueError("pair requires one base-only and one adapter result")
    for result, signature in ((base, a), (adapter, b)):
        if result["arm_identity"].get("base_inventory_sha256") != signature.get("base_inventory_sha256"):
            raise ValueError("arm identity base digest differs from protocol signature")
    if base.get("expected_task_ids") != adapter.get("expected_task_ids"):
        raise ValueError("paired result full task manifests differ")
    def index(result):
        rows = result.get("sample_rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError("paired result has no sample rows")
        out = {}
        types, hashes = {}, {}
        for row in rows:
            tid, si = row.get("task_id"), row.get("sample_index")
            key = (tid, si)
            if key in out or not isinstance(tid, str) or isinstance(si, bool) or not isinstance(si, int) or si not in range(SAMPLES):
                raise ValueError("paired result contains duplicate/invalid task sample")
            if not isinstance(row.get("won"), bool) or isinstance(row.get("seed"), bool) or not isinstance(row.get("seed"), int):
                raise ValueError("paired result has invalid won/seed fields")
            if row["seed"] != sample_seed(result["protocol_signature"]["run_seed"], tid, si):
                raise ValueError("paired result sample seed does not match protocol")
            raw = dict(row); digest = raw.pop("record_sha256", None)
            if digest != canonical_hash(raw):
                raise ValueError("paired result row is corrupted")
            if tid in types and (types[tid] != row.get("task_type") or hashes[tid] != row.get("game_sha256")):
                raise ValueError("paired result task type/content hash changes within task")
            types[tid], hashes[tid] = row.get("task_type"), row.get("game_sha256")
            out[key] = row
        if set(out) != {(tid, si) for tid in types for si in range(SAMPLES)}:
            raise ValueError("paired result is not a complete four-sample grid")
        expected_ids = result.get("expected_task_ids")
        if (not isinstance(expected_ids, list) or not expected_ids or
                len(expected_ids) != len(set(expected_ids)) or
                set(types) != set(expected_ids)):
            raise ValueError("paired result rows do not match its full expected task manifest")
        summary = result.get("summary", {})
        if summary.get("n_tasks") != len(types) or summary.get("n_episodes") != len(out):
            raise ValueError("paired result summary counts do not match rows")
        if summary.get("episode_weighted_sr") != statistics.mean(int(r["won"]) for r in out.values()):
            raise ValueError("paired result summary success rate does not match rows")
        return out
    ra = index(base)
    rb = index(adapter)
    if set(ra) != set(rb):
        raise ValueError("paired result task/sample grids differ")
    if any(ra[k].get("seed") != rb[k].get("seed") or ra[k].get("game_sha256") != rb[k].get("game_sha256") or
           ra[k].get("task_type") != rb[k].get("task_type") or ra[k].get("game_file") != rb[k].get("game_file") for k in ra):
        raise ValueError("paired sample seeds/task identities differ")
    d = [int(rb[k]["won"]) - int(ra[k]["won"]) for k in ra]
    return {"paired_episode_success_difference": statistics.mean(d), "n_pairs": len(d),
            "interpretation": "paired local comparison; not a paper-exact reproduction"}
