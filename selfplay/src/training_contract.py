"""Bind training to a previously prepared, immutable baseline evaluation contract."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import eval_runner
import evaluation


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read baseline evaluation contract: {type(exc).__name__}") from None
    if not isinstance(value, dict):
        raise ValueError("baseline evaluation contract must be a JSON object")
    return value


def validate_baseline_contract(path: str | Path, *, config_path: str | Path,
                               base_inventory: dict[str, dict[str, Any]],
                               data_root: str | Path,
                               solver_config: dict[str, Any]) -> dict[str, Any]:
    """Validate a base-only baseline contract against the current frozen runtime.

    The caller supplies the already-computed training base inventory so this function
    never rehashes model files. Returns a JSON-safe copy suitable for embedding in the
    training manifest, along with the contract hash and shared identity fields.
    """
    contract_path = Path(path).expanduser().resolve(strict=True)
    contract = _read_json(contract_path)
    if contract.get("schema") != evaluation.SCHEMA:
        raise ValueError("unsupported baseline evaluation contract schema")
    digest = contract.get("contract_sha256")
    payload = dict(contract)
    payload.pop("contract_sha256", None)
    if not isinstance(digest, str) or digest != evaluation.canonical_hash(payload):
        raise ValueError("baseline evaluation contract checksum is invalid")
    if not isinstance(contract.get("run_id"), str) or not contract["run_id"]:
        raise ValueError("baseline contract has no run identity")
    if contract.get("arm_identity") != {
            "kind": "base-only",
            "base_inventory_sha256": evaluation.canonical_hash(base_inventory)}:
        raise ValueError("baseline contract must identify this exact base-only model")

    cfg_path = Path(config_path).expanduser().resolve(strict=True)
    if Path(contract.get("config_path", "")).expanduser().resolve() != cfg_path:
        raise ValueError("baseline contract config path differs from training config")
    config = eval_runner._load_yaml(cfg_path)
    signature = contract.get("protocol_signature")
    if not isinstance(signature, dict):
        raise ValueError("baseline protocol signature is malformed")
    if signature.get("config") != config or signature.get("config_sha256") != evaluation.sha256_file(cfg_path):
        raise ValueError("baseline config content or hash differs from the current config")
    if contract.get("base_inventory") != {"files": [
            {"path": name, **record} for name, record in sorted(base_inventory.items())]}:
        raise ValueError("baseline model file inventory differs from training base")
    if signature.get("base_inventory_sha256") != evaluation.canonical_hash(base_inventory):
        raise ValueError("baseline model inventory checksum differs from training base")
    if signature.get("solver_settings") != eval_runner._config_settings(config):
        raise ValueError("baseline solver/evaluation settings are not the approved shared protocol")
    if config.get("solver") != solver_config:
        raise ValueError("baseline solver configuration differs from training solver")
    if signature.get("runtime_source_sha256") != eval_runner._source_identity():
        raise ValueError("baseline evaluator source identity differs from the frozen local evaluator")
    if signature.get("software") != eval_runner._versions():
        raise ValueError("baseline software versions differ from the current evaluation runtime")
    if signature.get("prompt_source_sha256") != evaluation.sha256_file(eval_runner.SRC / "protocol.py"):
        raise ValueError("baseline SkillRise prompt/action protocol source has changed")

    tokenizer_files = {name: {"path": name, **record} for name, record in sorted(base_inventory.items())
                       if name.startswith("tokenizer") or name in
                       ("vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json")}
    if signature.get("tokenizer_files_sha256") != evaluation.canonical_hash(tokenizer_files):
        raise ValueError("baseline tokenizer identity differs from training base")
    from protocol import prompt_encoding
    prompt_format = config.get("solver", {}).get("prompt_format", "raw")
    if signature.get("prompt_format", "raw") != prompt_format:
        raise ValueError("baseline prompt format differs from training config")
    if signature.get("raw_prompt_encoding") != prompt_encoding(prompt_format):
        raise ValueError("baseline tokenizer prompt encoding is not the shared protocol")
    chat_digest = signature.get("chat_template_sha256")
    if prompt_format == "chat_template" and (not isinstance(chat_digest, str) or
            len(chat_digest) != 64 or any(ch not in "0123456789abcdef" for ch in chat_digest)):
        raise ValueError("baseline chat template digest is missing or malformed")
    if prompt_format == "raw" and chat_digest is not None:
        raise ValueError("raw baseline unexpectedly declares a chat template")
    if not isinstance(signature.get("tokenizer_class"), str) or not signature["tokenizer_class"]:
        raise ValueError("baseline tokenizer class identity is missing")

    expected_mirrors = {
        "style": "react", "action_parser": "SkillRise play projection",
        "stop": "first decoded </action>, included",
        "seed_policy": "environment reset receives run seed; sha256(seed,task_id,turn) per generation request",
        "max_turns": 30, "max_context_tokens": 32768, "max_new_tokens": 1024,
        "temperature": 0.7, "top_p": 0.95, "top_k": -1, "min_p": 0.0,
        "repetition_penalty": 1.0, "dtype": "bfloat16",
    }
    expected_mirrors["backend"] = eval_runner._config_settings(config)["backend"]
    settings = eval_runner._config_settings(config)
    expected_mirrors["batch_size"] = settings["batch_size"]
    expected_mirrors["gpu_memory_utilization"] = settings["gpu_memory_utilization"]
    for field, expected_value in expected_mirrors.items():
        if signature.get(field) != expected_value:
            raise ValueError(f"baseline protocol signature has inconsistent {field}")

    requested_data = Path(data_root).expanduser().resolve(strict=True)
    if Path(contract.get("alfworld_data_root", "")).expanduser().resolve() != requested_data:
        raise ValueError("baseline ALFWorld data root differs from training data root")
    if signature.get("alfworld_data_root") != str(requested_data):
        raise ValueError("baseline signature ALFWorld data root differs from training")
    eval_cfg_path = eval_runner._alfworld_config_path(cfg_path, config)
    if (Path(contract.get("alfworld_config_path", "")).resolve() != eval_cfg_path or
            signature.get("alfworld_config_path") != str(eval_cfg_path)):
        raise ValueError("baseline ALFWorld YAML identity differs from current config")
    suite = contract.get("suite")
    evaluation.validate_suite(suite)
    if suite.get("split") != "valid_seen" or signature.get("split") != "valid_seen":
        raise ValueError("baseline suite must be the full valid_seen pool")
    if (suite.get("config_path") != str(eval_cfg_path) or
            suite.get("config_sha256") != evaluation.sha256_file(eval_cfg_path)):
        raise ValueError("baseline suite ALFWorld YAML identity is inconsistent")
    if signature.get("pool_sha256") != suite.get("pool_sha256"):
        raise ValueError("baseline suite and signature task-pool identities differ")
    if len(contract.get("suite", {}).get("tasks", [])) != len({
            t.get("task_id") for t in contract.get("suite", {}).get("tasks", [])}):
        raise ValueError("baseline suite contains duplicate task IDs")
    if signature.get("seeds") != list(evaluation.SEEDS):
        raise ValueError("baseline must use all five fixed evaluation seeds")
    expected = contract.get("expected_shards")
    shards = suite.get("shards", [])
    if (not isinstance(expected, list) or len(expected) != len(shards) or any(
            item.get("rank") != shard.get("rank") or
            item.get("task_ids") != shard.get("task_ids")
            for item, shard in zip(expected, shards))):
        raise ValueError("baseline expected shards differ from the complete suite")

    os.environ["ALFWORLD_DATA"] = str(requested_data)
    fresh_suite = evaluation.discover_suite(str(eval_cfg_path), "valid_seen", settings["shards"])
    evaluation.validate_suite(fresh_suite)
    if fresh_suite != suite:
        raise ValueError("installed full valid_seen suite differs from the baseline contract")

    common = {
        "config_path": str(cfg_path), "config_sha256": signature["config_sha256"],
        "base_inventory_sha256": signature["base_inventory_sha256"],
        "solver_config": copy.deepcopy(solver_config),
        "runtime_source_sha256": copy.deepcopy(signature["runtime_source_sha256"]),
        "software": copy.deepcopy(signature["software"]),
        "alfworld_data_root": str(requested_data), "alfworld_config_path": str(eval_cfg_path),
        "pool_sha256": suite["pool_sha256"], "split": "valid_seen",
        "seeds": list(evaluation.SEEDS), "tokenizer_files_sha256": signature["tokenizer_files_sha256"],
        "tokenizer_class": signature["tokenizer_class"],
    }
    return {"contract": copy.deepcopy(contract), "contract_sha256": digest,
            "common_identity": common}
