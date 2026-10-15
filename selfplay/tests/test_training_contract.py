"""Focused CPU checks for binding training to a prepared base evaluation contract."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
import sys
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import eval_runner
import evaluation
from training_contract import validate_baseline_contract


def _config(alfworld_yaml):
    return {
        "env": {"config_path": str(alfworld_yaml)},
        "solver": {"style": "react", "max_turns": 30, "max_context_tokens": 32768,
            "keep_recent_turns": 15, "max_new_tokens": 1024, "temperature": 0.7,
            "top_p": 0.95, "top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0,
            "stop_mode": "action_tag", "menu_enabled": True},
        "eval": {"seeds": [0, 1, 2, 3, 4], "split": "valid_seen", "batch_size": 32,
            "shards": 8, "backend": "vllm", "gpu_memory_utilization": 0.85},
    }


def _fixture(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[2]
    config_path = tmp_path / "config.yaml"
    alfworld_yaml = repo / "selfplay" / "alfworld.yaml"
    config = _config(alfworld_yaml)
    monkeypatch.setattr(eval_runner, "_load_yaml", lambda p: json.loads(Path(p).read_text(encoding="utf-8")))
    config_path.write_text(json.dumps(config), encoding="utf-8")
    data_root = tmp_path / "alfworld-data"
    data_root.mkdir()
    base_inventory = {
        "config.json": {"size": 4, "sha256": "a" * 64, "symlink": False},
        "model.safetensors": {"size": 8, "sha256": "b" * 64, "symlink": False},
        "tokenizer.json": {"size": 2, "sha256": "c" * 64, "symlink": False},
    }
    kinds = ["pick_and_place_simple", "look_at_obj_in_light",
             "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep",
             "pick_cool_then_place_in_recep", "pick_two_obj_and_place"]
    tasks = []
    for i, kind in enumerate(kinds):
        game = tmp_path / f"game-{i}.json"
        game.write_text(f"game {i}", encoding="utf-8")
        tasks.append({"task_id": f"task-{i}", "task_type": kind,
                      "game_file": str(game.resolve()), "game_sha256": evaluation.sha256_file(game)})
    yaml_hash = evaluation.sha256_file(alfworld_yaml)
    env_sources = {"fake": "identity"}
    pool_hash = evaluation.canonical_hash({"split": "valid_seen", "config_sha256": yaml_hash,
        "environment_sources": env_sources,
        "tasks": [{k: t[k] for k in ("task_id", "task_type", "game_sha256")} for t in tasks]})
    suite = {"schema": evaluation.SCHEMA, "split": "valid_seen", "alfworld_split": "eval_in_distribution",
             "config_path": str(alfworld_yaml.resolve()), "config_sha256": yaml_hash,
             "environment_sources": env_sources, "pool_sha256": pool_hash, "tasks": tasks,
             "shards": [{"rank": i, "task_ids": [t["task_id"]]} for i, t in enumerate(tasks)]}
    fresh_suite = copy.deepcopy(suite)
    monkeypatch.setattr(evaluation, "discover_suite", lambda *args: copy.deepcopy(fresh_suite))
    serial_inventory = {"files": [{"path": name, **record}
                                   for name, record in sorted(base_inventory.items())]}
    base_hash = evaluation.canonical_hash(base_inventory)
    signature = {
        "pool_sha256": suite["pool_sha256"], "split": "valid_seen",
        "style": "react",
        "config_sha256": evaluation.sha256_file(config_path), "config": config,
        "solver_settings": eval_runner._config_settings(config),
        "prompt_source_sha256": evaluation.sha256_file(SRC / "protocol.py"),
        "tokenizer_files_sha256": evaluation.canonical_hash({
            "tokenizer.json": {"path": "tokenizer.json", **base_inventory["tokenizer.json"]}}),
        "tokenizer_class": "FakeTokenizer", "raw_prompt_encoding": "tokenizer.encode(add_special_tokens=True)",
        "action_parser": "SkillRise play projection", "stop": "first decoded </action>, included",
        "seed_policy": "environment reset receives run seed; sha256(seed,task_id,turn) per generation request",
        "seeds": list(evaluation.SEEDS), "max_turns": 30,
        "alfworld_data_root": str(data_root.resolve()),
        "alfworld_config_path": str(alfworld_yaml.resolve()),
        "max_context_tokens": 32768, "max_new_tokens": 1024, "temperature": 0.7, "top_p": 0.95,
        "top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0,
        "backend": "vllm", "dtype": "bfloat16", "base_inventory_sha256": base_hash,
        "batch_size": 32, "gpu_memory_utilization": 0.85,
        "software": eval_runner._versions(), "runtime_source_sha256": eval_runner._source_identity(),
    }
    contract = {"schema": evaluation.SCHEMA, "run_id": "fixture", "suite": suite,
        "config_path": str(config_path.resolve()), "alfworld_data_root": str(data_root.resolve()),
        "alfworld_config_path": str(alfworld_yaml.resolve()),
        "base_path": str(tmp_path / "base"), "base_inventory": serial_inventory,
        "arm_identity": {"kind": "base-only", "base_inventory_sha256": base_hash},
        "protocol_signature": signature,
        "expected_shards": [{"rank": sh["rank"], "task_ids": sh["task_ids"], "path": str(tmp_path / f"r{sh['rank']}.json")}
                            for sh in suite["shards"]]}
    _resign(contract)
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    return path, config_path, base_inventory, data_root, config["solver"], contract


def _resign(contract):
    contract["contract_sha256"] = evaluation.canonical_hash(
        {k: v for k, v in contract.items() if k != "contract_sha256"})


def _validate(fixture):
    path, config_path, inventory, data_root, solver, _ = fixture
    return validate_baseline_contract(path, config_path=config_path, base_inventory=inventory,
                                      data_root=data_root, solver_config=solver)


def test_accepts_valid_baseline_and_returns_full_binding(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    result = _validate(fixture)
    assert result["contract"] == fixture[-1]
    assert result["contract_sha256"] == fixture[-1]["contract_sha256"]
    assert result["common_identity"]["pool_sha256"] == fixture[-1]["suite"]["pool_sha256"]
    assert result["common_identity"]["seeds"] == [0, 1, 2, 3, 4]


def test_eval_prepare_contract_round_trips_into_training_validation(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    fixture = _fixture(tmp_path, monkeypatch)
    path, config_path, inventory, data_root, solver, expected = fixture
    base = tmp_path / "base"
    base.mkdir()
    base_inventory = {"files": [{"path": name, **record} for name, record in sorted(inventory.items())]}
    monkeypatch.setattr(eval_runner, "_base_identity", lambda _: (base, base_inventory,
        evaluation.canonical_hash(inventory)))
    monkeypatch.setattr(eval_runner, "discover_suite", lambda *args: copy.deepcopy(expected["suite"]))
    class FakeTokenizer:
        pass
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda *args, **kwargs: FakeTokenizer())
    contract_path = tmp_path / "prepared-by-evaluator.json"
    results_dir = tmp_path / "results"
    args = type("Args", (), {"config": str(config_path), "alfworld_data": str(data_root),
        "base": str(base), "checkpoint": None, "base_only": True,
        "run_id": "prepared", "results_dir": str(results_dir), "output": str(contract_path)})()
    assert eval_runner.prepare(args) == 0
    result = validate_baseline_contract(contract_path, config_path=config_path,
        base_inventory=inventory, data_root=data_root, solver_config=solver)
    assert result["contract"]["arm_identity"]["kind"] == "base-only"


@pytest.mark.parametrize("mutate", [
    lambda c: c["protocol_signature"].__setitem__("config_sha256", "0" * 64),
    lambda c: c["protocol_signature"].__setitem__("runtime_source_sha256", {"x": "changed"}),
    lambda c: c["protocol_signature"].__setitem__("base_inventory_sha256", "0" * 64),
    lambda c: c["protocol_signature"].__setitem__("alfworld_data_root", "/elsewhere"),
    lambda c: c["protocol_signature"].__setitem__("solver_settings", {"style": "other"}),
    lambda c: c["protocol_signature"].__setitem__("seeds", [0, 1, 2, 3]),
    lambda c: c["protocol_signature"].__setitem__("max_context_tokens", 40960),
    lambda c: c["protocol_signature"].__setitem__("action_parser", "other"),
    lambda c: c["protocol_signature"].__setitem__("batch_size", 1),
    lambda c: c["protocol_signature"].__setitem__("gpu_memory_utilization", 0.1),
    lambda c: c["suite"].__setitem__("split", "train"),
    lambda c: c["arm_identity"].__setitem__("kind", "checkpoint"),
])
def test_rejects_resigned_semantic_drift(tmp_path, monkeypatch, mutate):
    fixture = _fixture(tmp_path, monkeypatch)
    path, _, _, _, _, contract = fixture
    mutate(contract)
    _resign(contract)
    path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError):
        _validate(fixture)


def test_rejects_bad_outer_checksum(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    path, *_ = fixture
    contract = json.loads(path.read_text(encoding="utf-8"))
    contract["contract_sha256"] = "0" * 64
    path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        _validate(fixture)


def test_rejects_resigned_suite_with_shard_count_out_of_config(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    path, *_rest, contract = fixture
    tasks = contract["suite"]["tasks"]
    contract["suite"]["shards"] = [
        {"rank": i, "task_ids": [t["task_id"] for t in tasks[2 * i:2 * i + 2]]}
        for i in range(3)]
    contract["expected_shards"] = [
        {"rank": i, "task_ids": shard["task_ids"], "path": str(path.parent / f"changed-{i}.json")}
        for i, shard in enumerate(contract["suite"]["shards"])]
    _resign(contract)
    path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="shards|suite"):
        _validate(fixture)


def test_rejects_model_data_and_solver_identity_drift(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    path, _, inventory, data_root, solver, contract = fixture
    changed_inventory = dict(inventory)
    changed_inventory["model.safetensors"] = {**inventory["model.safetensors"], "sha256": "e" * 64}
    with pytest.raises(ValueError):
        validate_baseline_contract(path, config_path=fixture[1], base_inventory=changed_inventory,
                                   data_root=data_root, solver_config=solver)
    other_data = data_root.parent / "other"
    other_data.mkdir()
    with pytest.raises(ValueError, match="data root"):
        validate_baseline_contract(path, config_path=fixture[1], base_inventory=inventory,
                                   data_root=other_data, solver_config=solver)
    wrong_solver = {**solver, "temperature": 0.9}
    with pytest.raises(ValueError):
        validate_baseline_contract(path, config_path=fixture[1], base_inventory=inventory,
                                   data_root=data_root, solver_config=wrong_solver)


def test_rejects_changed_config_file_even_if_contract_is_resigned(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    config_path = fixture[1]
    changed = json.loads(config_path.read_text(encoding="utf-8"))
    changed["changed"] = True
    config_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="config"):
        _validate(fixture)


def test_rejects_changed_current_source_identity(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(eval_runner, "_source_identity", lambda: {"eval_runner.py": "changed"})
    with pytest.raises(ValueError, match="source"):
        _validate(fixture)
