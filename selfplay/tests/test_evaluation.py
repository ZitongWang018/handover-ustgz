from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sys
import types
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from evaluation import (SCHEMA, SEEDS, align_task_slots, canonical_hash, merge,
                        request_seed, sha256_file, validate_suite)
from protocol import build_skillrise_prompt, encode_raw_prompt


class TinyTokenizer:
    def __call__(self, text, add_special_tokens=True):
        # Deterministic stand-in to prove the protocol receives the raw prompt.
        return {"input_ids": [ord(c) for c in text] + ([1] if add_special_tokens else [])}
    def decode(self, ids, skip_special_tokens=False):
        tokens = list(ids)
        if skip_special_tokens and tokens and tokens[-1] == 1:
            tokens = tokens[:-1]
        return "".join(chr(int(token)) for token in tokens)


def _fixture(tmp_path):
    config = tmp_path / "alfworld.yaml"
    config.write_text("env: test\n", encoding="utf-8")
    tasks = []
    types = list(("pick_and_place_simple", "look_at_obj_in_light",
        "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep", "pick_two_obj_and_place"))
    for i, task_type in enumerate(types):
        game = tmp_path / f"game{i}.tw-pddl"
        game.write_text(f"task {i}", encoding="utf-8")
        tasks.append({"task_id": f"task{i}", "task_type": task_type,
            "game_file": str(game), "game_sha256": sha256_file(game)})
    from evaluation import canonical_hash
    cfg_hash = sha256_file(config)
    env_sources = {"test": "source"}
    pool_hash = canonical_hash({"split": "valid_seen", "config_sha256": cfg_hash,
        "environment_sources": env_sources,
        "tasks": [{k: t[k] for k in ("task_id", "task_type", "game_sha256")} for t in tasks]})
    suite = {"schema": SCHEMA, "split": "valid_seen", "alfworld_split": "eval_in_distribution",
        "config_path": str(config), "config_sha256": cfg_hash, "environment_sources": env_sources,
        "pool_sha256": pool_hash, "tasks": tasks,
        "shards": [{"rank": 0, "task_ids": [x["task_id"] for x in tasks[:3]]},
                   {"rank": 1, "task_ids": [x["task_id"] for x in tasks[3:]]}]}
    base_inv = {"files": []}
    arm = {"kind": "base-only", "base_inventory_sha256": "b" * 64}
    signature = {"pool_sha256": pool_hash, "split": "valid_seen", "config_sha256": "c" * 64,
        "config": {}, "solver_settings": {}, "prompt_source_sha256": "d" * 64,
        "tokenizer_files_sha256": "e" * 64, "raw_prompt_encoding": "raw",
        "action_parser": "SkillRise play projection", "stop": "action tag",
        "seed_policy": "deterministic", "seeds": list(SEEDS), "max_turns": 30,
        "max_context_tokens": 32768, "max_new_tokens": 1024, "temperature": .7,
        "top_p": .95, "top_k": -1, "min_p": 0., "repetition_penalty": 1.,
        "backend": "vllm", "dtype": "bfloat16", "base_inventory_sha256": "b" * 64,
        "alfworld_data_root": str(tmp_path),
        "software": {"python": "test"}, "runtime_source_sha256": {"test": "f" * 64}}
    contract = {"schema": SCHEMA, "run_id": "run", "suite": suite,
        "arm_identity": arm, "base_inventory": base_inv, "protocol_signature": signature,
        "expected_shards": []}
    contract["expected_shards"] = [{"rank": sh["rank"], "task_ids": sh["task_ids"], "path": str(tmp_path / f"rank{sh['rank']}.json")} for sh in suite["shards"]]
    contract["contract_sha256"] = canonical_hash(contract)
    shards = []
    for rank, shard in enumerate(suite["shards"]):
        selected = [t for t in tasks if t["task_id"] in shard["task_ids"]]
        rows = []
        for task in selected:
            for seed in SEEDS:
                row = {"task_id": task["task_id"], "task_type": task["task_type"],
                    "game_file": task["game_file"], "game_sha256": task["game_sha256"],
                    "seed": seed, "environment_reset_seed": seed, "won": False,
                    "steps": 1, "trace": [{"turn": 0, "raw_prompt": "raw", "prompt_token_ids": [1],
                        "output_token_ids": [2], "raw_output": "<action>look</action>",
                        "parser_projection": "look", "action_tag_valid": True,
                        "executed_action": "look", "request_seed": request_seed(seed, task["task_id"], 0),
                        "observation": "observation", "won": False, "done": True,
                        "truncated": False, "stop_len": 1}]}
                row["record_sha256"] = canonical_hash(row)
                rows.append(row)
        shards.append({"schema": SCHEMA, "run_id": contract["run_id"], "contract_sha256": contract["contract_sha256"],
            "rank": rank, "task_ids": shard["task_ids"], "protocol_signature": signature,
            "arm_identity": arm, "base_inventory": base_inv, "software": signature["software"],
            "runtime_source_sha256": signature["runtime_source_sha256"], "rows": rows})
    return contract, shards


def test_full_grid_merges_and_compares_paired_rows(tmp_path):
    contract, shards = _fixture(tmp_path)
    result = merge(contract, shards)
    assert result["summary"]["n_episodes"] == 30
    trained_contract, trained_shards = copy.deepcopy(contract), copy.deepcopy(shards)
    trained_contract["arm_identity"] = {"kind": "checkpoint", "base_inventory_sha256": "b" * 64}
    trained_contract["contract_sha256"] = canonical_hash({k: v for k, v in trained_contract.items() if k != "contract_sha256"})
    for shard in trained_shards:
        shard["contract_sha256"] = trained_contract["contract_sha256"]
        shard["arm_identity"] = trained_contract["arm_identity"]
    trained = merge(trained_contract, trained_shards)
    from evaluation import compare
    assert compare(result, trained)["paired_episode_sr_delta_mean"] == 0


def test_reordered_environment_slots_keep_task_metadata_aligned():
    a = {"task_id": "a", "game_file": "a.tw-pddl"}
    b = {"task_id": "b", "game_file": "b.tw-pddl"}
    assert [x["task_id"] for x in align_task_slots([a, b], ["b.tw-pddl", "a.tw-pddl"])] == ["b", "a"]
    with pytest.raises(ValueError):
        align_task_slots([a, b], ["b.tw-pddl", "missing.tw-pddl"])


def test_missing_duplicate_tampered_and_wrong_protocol_are_rejected(tmp_path):
    contract, shards = _fixture(tmp_path)
    with pytest.raises(ValueError): merge(contract, shards[:1])
    broken = copy.deepcopy(shards)
    broken[0]["rows"].append(copy.deepcopy(broken[0]["rows"][0]))
    with pytest.raises(ValueError): merge(contract, broken)
    broken = copy.deepcopy(shards)
    broken[0]["rows"][0]["won"] = True
    with pytest.raises(ValueError): merge(contract, broken)
    wrong = copy.deepcopy(contract)
    wrong["protocol_signature"]["temperature"] = .8
    wrong["contract_sha256"] = canonical_hash({k: v for k, v in wrong.items() if k != "contract_sha256"})
    with pytest.raises(ValueError): merge(wrong, shards)


def test_suite_task_reordering_invalidates_manifest(tmp_path):
    contract, _ = _fixture(tmp_path)
    suite = copy.deepcopy(contract["suite"])
    suite["tasks"].reverse()
    with pytest.raises(ValueError): validate_suite(suite)


def test_second_turn_prompt_uses_shared_action_observation_history():
    tok = TinyTokenizer()
    first = build_skillrise_prompt("initial", [], ["look"], 0)
    second = build_skillrise_prompt("initial", [("look", "a clean observation")], ["take apple"], 1)
    assert "Action 1: look\nObservation 1: a clean observation" in second
    assert encode_raw_prompt(tok, first)[-1] == 1
    assert encode_raw_prompt(tok, second)[-1] == 1
    assert "Action 1" not in first


def test_prepare_run_merge_cpu_integration_with_reordered_slots(tmp_path, monkeypatch):
    import eval_runner

    types_by_id = ["pick_and_place_simple", "look_at_obj_in_light",
        "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep", "pick_two_obj_and_place"]
    tasks = []
    for i, kind in enumerate(types_by_id):
        game = tmp_path / f"task{i}.tw-pddl"
        game.write_text(f"game {i}", encoding="utf-8")
        tasks.append({"task_id": f"task{i}", "task_type": kind,
            "game_file": str(game), "game_sha256": sha256_file(game)})
    alf_cfg = tmp_path / "alfworld.yaml"
    alf_cfg.write_text("placeholder: true\n", encoding="utf-8")
    solver_cfg = tmp_path / "config.yaml"
    config = {"env": {"config_path": "alfworld.yaml"},
        "solver": {"style": "react", "max_turns": 30, "max_context_tokens": 32768,
        "keep_recent_turns": 15, "max_new_tokens": 1024, "temperature": .7, "top_p": .95,
        "top_k": -1, "min_p": 0., "repetition_penalty": 1., "stop_mode": "action_tag",
        "menu_enabled": True}, "eval": {"split": "valid_seen", "seeds": list(SEEDS),
        "batch_size": 3, "shards": 2, "backend": "vllm", "gpu_memory_utilization": .85}}
    solver_cfg.write_text(json.dumps(config), encoding="utf-8")
    cfg_hash = sha256_file(alf_cfg)
    env_sources = {"test": "fixed"}
    pool_hash = canonical_hash({"split": "valid_seen", "config_sha256": cfg_hash,
        "environment_sources": env_sources,
        "tasks": [{k: t[k] for k in ("task_id", "task_type", "game_sha256")} for t in tasks]})
    suite = {"schema": SCHEMA, "split": "valid_seen", "alfworld_split": "eval_in_distribution",
        "config_path": str(alf_cfg), "config_sha256": cfg_hash, "environment_sources": env_sources,
        "pool_sha256": pool_hash, "tasks": tasks,
        "shards": [{"rank": 0, "task_ids": [x["task_id"] for x in tasks[:3]]},
                   {"rank": 1, "task_ids": [x["task_id"] for x in tasks[3:]]}]}
    def fake_discover(*_args, **_kwargs):
        assert os.environ.get("ALFWORLD_DATA") == str(tmp_path)
        return suite
    monkeypatch.setattr(eval_runner, "discover_suite", fake_discover)
    monkeypatch.setattr(eval_runner, "_versions", lambda: {"python": "cpu-mock"})

    base = tmp_path / "base"
    base.mkdir()
    weight = base / "weights.bin"
    weight.write_bytes(b"base weights")
    base_record = {"weights.bin": {"size": weight.stat().st_size, "sha256": sha256_file(weight)}}
    import checkpoint
    monkeypatch.setattr(checkpoint, "base_file_inventory", lambda _p: base_record)
    monkeypatch.setattr(checkpoint, "base_inventory_sha256", lambda _p: "b" * 64)

    transformer_stub = types.ModuleType("transformers")
    transformer_stub.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: TinyTokenizer())
    monkeypatch.setitem(sys.modules, "transformers", transformer_stub)

    result_dir = tmp_path / "rank_results"
    result_dir.mkdir()
    contract_path = tmp_path / "contract.json"
    monkeypatch.setenv("ALFWORLD_DATA", str(tmp_path))
    args = types.SimpleNamespace(config=str(solver_cfg), alfworld_data=str(tmp_path),
        base=str(base), base_only=True, checkpoint=None, run_id="integration",
        results_dir=str(result_dir), output=str(contract_path))
    assert eval_runner.prepare(args) == 0
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    class FakePolicy:
        def generate_from_token_ids(self, prompt_ids, _cfg, request_seeds=None):
            ids_out = []
            self.last_finish_reasons = ["stop"] * len(prompt_ids)
            for ids, _seed in zip(prompt_ids, request_seeds):
                prompt = tokenizer.decode(ids)
                tid = re.search(r"task[0-5]", prompt).group(0)
                action = f"finish {tid}" if "Action 1:" in prompt else f"go {tid}"
                ids_out.append([ord(c) for c in f"<action>{action}</action>"])
            return ["" for _ in ids_out], ids_out, [len(x) for x in ids_out]

    tokenizer = TinyTokenizer()
    monkeypatch.setattr(eval_runner, "_load_policy", lambda _contract, _tokenizer: FakePolicy())
    compat = types.ModuleType("alfworld_compat")
    compat.find_compat_dir = lambda: None
    compat.install = lambda **_kwargs: None
    monkeypatch.setitem(sys.modules, "alfworld_compat", compat)

    class FakeEnv:
        def __init__(self, _cfg, *, batch_size, **_kwargs):
            self.n = batch_size
            self.current = []
            self.initial_actions = {}
            self.turn = 0
        def reset(self, games, seed=None):
            self.seed = seed
            self.current = list(reversed(games))
            views = [types.SimpleNamespace(raw=f"Your task is to: task{int(Path(g).stem[-1])}.") for g in self.current]
            admissible = [[f"go task{Path(g).stem[-1]}", f"finish task{Path(g).stem[-1]}"] for g in self.current]
            return [v.raw for v in views], views, admissible, self.current, [None] * self.n
        def step(self, actions):
            obs, won, done, admissible = [], [], [], []
            for game, action in zip(self.current, actions):
                tid = f"task{Path(game).stem[-1]}"
                if self.turn == 0:
                    self.initial_actions[tid] = action
                    obs.append(f"first-result {tid}"); won.append(False); done.append(False)
                else:
                    obs.append(f"second-result {tid}")
                    won.append(tid == "task0" and self.seed == 0 and self.initial_actions[tid] == f"go {tid}" and action == f"finish {tid}")
                    done.append(True)
                admissible.append([f"go {tid}", f"finish {tid}"])
            self.turn += 1
            return types.SimpleNamespace(obs=obs, won=won, dones=done, admissible=admissible)
        def close(self): pass

    env_module = types.ModuleType("alfworld_env")
    env_module.ALFWorldBatchEnv = FakeEnv
    env_module.load_alfworld_config = lambda _p: {}
    monkeypatch.setitem(sys.modules, "alfworld_env", env_module)
    rollout_stub = types.ModuleType("rollout")
    rollout_stub.RolloutConfig = lambda **kwargs: types.SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "rollout", rollout_stub)

    outputs = []
    for rank in range(2):
        output = Path(contract["expected_shards"][rank]["path"])
        if rank == 0:
            with pytest.raises(ValueError, match="output path"):
                eval_runner.run(types.SimpleNamespace(contract=str(contract_path), rank=rank,
                    output=str(tmp_path / "unprepared-rank.json")))
        assert eval_runner.run(types.SimpleNamespace(contract=str(contract_path), rank=rank, output=str(output))) == 0
        outputs.append(json.loads(output.read_text(encoding="utf-8")))
    merged = merge(contract, outputs)
    assert merged["summary"]["n_episodes"] == len(tasks) * len(SEEDS)
    assert merged["summary"]["six_type_macro_sr"] is not None
    rows = {(row["task_id"], row["seed"]): row for row in merged["rows"]}
    for task in tasks:
        for seed in SEEDS:
            row = rows[(task["task_id"], seed)]
            assert row["steps"] == 2
            assert row["trace"][0]["executed_action"] == f"go {task['task_id']}"
            assert row["trace"][1]["executed_action"] == f"finish {task['task_id']}"
            assert f"Your task is to: {task['task_id']}." in row["trace"][0]["raw_prompt"]
            assert f"Action 1: go {task['task_id']}\nObservation 1: first-result {task['task_id']}" in row["trace"][1]["raw_prompt"]
            assert row["trace"][0]["request_seed"] == request_seed(seed, task["task_id"], 0)
            assert row["trace"][1]["request_seed"] == request_seed(seed, task["task_id"], 1)
    assert rows[("task0", 0)]["won"] is True
    assert rows[("task0", 1)]["won"] is False

    drifted = tmp_path / "config_drift.yaml"
    changed = copy.deepcopy(config); changed["solver"]["temperature"] = .8
    drifted.write_text(json.dumps(changed), encoding="utf-8")
    bad_contract = copy.deepcopy(contract)
    bad_contract["config_path"] = str(drifted)
    bad_contract["expected_shards"][0]["path"] = str(tmp_path / "bad-rank0.json")
    bad_contract["contract_sha256"] = canonical_hash({k: v for k, v in bad_contract.items() if k != "contract_sha256"})
    bad_path = tmp_path / "drifted_contract.json"
    bad_path.write_text(json.dumps(bad_contract), encoding="utf-8")
    with pytest.raises(ValueError, match="approved ReAct"):
        eval_runner.run(types.SimpleNamespace(contract=str(bad_path), rank=0,
            output=str(tmp_path / "bad-rank0.json")))
