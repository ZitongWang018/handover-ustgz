"""Focused tests for the accelerated evaluator plumbing without starting vLLM."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "selfplay" / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "selfplay"))
import eval_runner
import evaluate_run


def _config(*, batch_size=4, shards=8):
    return {
        "solver": {"style": "react", "max_turns": 30, "max_context_tokens": 32768,
            "keep_recent_turns": 15, "max_new_tokens": 1024, "temperature": 0.7,
            "top_p": 0.95, "top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0,
            "stop_mode": "action_tag", "menu_enabled": True, "prompt_format": "chat_template"},
        "eval": {"split": "valid_seen", "seeds": [0, 1, 2, 3, 4],
            "batch_size": batch_size, "shards": shards, "backend": "vllm",
            "gpu_memory_utilization": 0.85},
    }


def _install_vllm(monkeypatch):
    built = []

    class FakePolicy:
        def __init__(self, engine, tokenizer):
            self.engine = engine
            self.tokenizer = tokenizer
            self.set_lora_calls = []

        def set_lora(self, *args):
            self.set_lora_calls.append(args)

    def build(path, **kwargs):
        built.append((str(path), kwargs))
        return object()

    module = types.ModuleType("ez_vllm_policy")
    module.VLLMPolicy = FakePolicy
    module.build_vllm_engine = build
    monkeypatch.setitem(sys.modules, "ez_vllm_policy", module)
    return built


def _install_merged_model(monkeypatch, *, exported="c" * 64, validated="c" * 64):
    calls = {"export": [], "validate": []}
    module = types.ModuleType("merged_model")

    def export(base, solver, output, *, checkpoint_identity):
        calls["export"].append((base, solver, output, checkpoint_identity))
        Path(output).mkdir(parents=True, exist_ok=True)
        return {"manifest_sha256": exported}

    def validate(path, *, base_digest, arm):
        calls["validate"].append((Path(path), base_digest, arm))
        return {"manifest_sha256": validated}

    module.export_solver_model = export
    module.validate_merged_model = validate
    monkeypatch.setitem(sys.modules, "merged_model", module)
    return calls


def _vllm_contract(*, arm=None, merged=None):
    contract = {
        "base_path": "base-model",
        "protocol_signature": {"backend": "vllm", "gpu_memory_utilization": 0.85,
            "max_context_tokens": 32768, "batch_size": 4,
            "base_inventory_sha256": "b" * 64},
        "arm_identity": arm or {"kind": "base-only"},
    }
    if merged is not None:
        contract["merged_model"] = merged
    return contract


def test_vllm_base_and_merged_engines_disable_lora_without_set_lora(tmp_path, monkeypatch):
    built = _install_vllm(monkeypatch)
    base = eval_runner._load_policy(_vllm_contract(), object())
    assert built[0][0] == "base-model"
    assert built[0][1]["enable_lora"] is False
    assert base.set_lora_calls == []

    merged_dir = tmp_path / "merged"
    merged_dir.mkdir()
    _install_merged_model(monkeypatch)
    arm = {"kind": "checkpoint", "solver_path": str(tmp_path / "solver")}
    contract = _vllm_contract(arm=arm, merged={
        "path": str(merged_dir), "manifest_sha256": "c" * 64})
    policy = eval_runner._load_policy(contract, object())
    assert built[-1][0] == str(merged_dir.resolve())
    assert built[-1][1]["enable_lora"] is False
    assert policy.set_lora_calls == []
    assert contract["arm_identity"] == arm
    assert "merged_model" not in contract["arm_identity"]


def test_corrupt_merged_export_is_rejected_before_engine(tmp_path, monkeypatch):
    built = _install_vllm(monkeypatch)
    merged_dir = tmp_path / "merged"
    merged_dir.mkdir()
    calls = _install_merged_model(monkeypatch, validated="d" * 64)
    contract = _vllm_contract(
        arm={"kind": "checkpoint", "solver_path": str(tmp_path / "solver")},
        merged={"path": str(merged_dir), "manifest_sha256": "c" * 64})
    with pytest.raises(ValueError, match="manifest"):
        eval_runner._load_policy(contract, object())
    assert calls["validate"]
    assert built == []


def test_unmerged_qwen3_moe_checkpoint_fails_before_engine(tmp_path, monkeypatch):
    built = _install_vllm(monkeypatch)
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"model_type": "qwen3_moe"}), encoding="utf-8")
    contract = _vllm_contract(
        arm={"kind": "checkpoint", "solver_path": str(tmp_path / "solver")})
    contract["base_path"] = str(base)
    with pytest.raises(ValueError, match="merged solver model"):
        eval_runner._load_policy(contract, object())
    assert built == []


def test_legacy_unmerged_adapter_still_uses_lora(tmp_path, monkeypatch):
    built = _install_vllm(monkeypatch)
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"model_type": "qwen2"}), encoding="utf-8")
    contract = _vllm_contract(
        arm={"kind": "checkpoint", "solver_path": str(tmp_path / "solver")})
    contract["base_path"] = str(base)
    policy = eval_runner._load_policy(contract, object())
    assert built[-1][1]["enable_lora"] is True
    assert policy.set_lora_calls == [("solver", str(tmp_path / "solver"))]


@pytest.mark.parametrize("batch_size", [0, 9, True, 4.0, "4"])
def test_eval_batch_size_must_be_a_sane_integer(batch_size):
    with pytest.raises(ValueError, match="batch size"):
        eval_runner._config_settings(_config(batch_size=batch_size))


@pytest.mark.parametrize("shards", [0, 9])
def test_eval_shards_are_bounded(shards):
    with pytest.raises(ValueError, match="shard"):
        eval_runner._config_settings(_config(shards=shards))


def test_evaluate_run_exports_merged_solver_with_original_base_arm(tmp_path, monkeypatch):
    config = _config()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    base = tmp_path / "base"
    base.mkdir()
    checkpoint = tmp_path / "checkpoint"
    solver = checkpoint / "solver"
    solver.mkdir(parents=True)
    outer = tmp_path / "outer"
    outer.mkdir()
    identity = {"prompt_format": "chat_template"}
    signature = {"backend": "vllm", "base_inventory_sha256": "b" * 64,
        "solver_settings": eval_runner._config_settings(config)}
    contract = {"config_path": str(config_path), "base_path": str(base),
        "protocol_signature": signature}
    snapshot = {"outer": 1, "path": checkpoint, "checkpoint_manifest_sha256": "d" * 64}
    arm = {"kind": "checkpoint", "path": str(checkpoint), "solver_path": str(solver),
        "manifest_sha256": "d" * 64, "base_inventory_sha256": "b" * 64}
    seen = {}
    monkeypatch.setattr(eval_runner, "_base_identity",
        lambda _path: (base.resolve(), {"files": []}, "b" * 64))
    class FakeTokenizer:
        pass
    transformer_module = types.ModuleType("transformers")
    transformer_module.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: FakeTokenizer())
    monkeypatch.setitem(sys.modules, "transformers", transformer_module)
    import protocol
    monkeypatch.setattr(protocol, "solver_prompt_identity",
        lambda _tokenizer, prompt_format: {"prompt_format": prompt_format})
    def fake_arm(args, base_digest, solver_config, *, prompt_identity=None,
                 tokenizer_class=None):
        seen.update(checkpoint=args.checkpoint, base_digest=base_digest,
                    solver_config=solver_config, prompt_identity=prompt_identity,
                    tokenizer_class=tokenizer_class)
        return arm
    monkeypatch.setattr(eval_runner, "_arm", fake_arm)
    calls = _install_merged_model(monkeypatch)
    merged_dir, manifest = evaluate_run._export_merged_solver(outer, snapshot, contract)
    assert merged_dir == outer / "merged_solver"
    assert manifest["manifest_sha256"] == "c" * 64
    assert len(calls["export"]) == 1
    base_arg, solver_arg, output_arg, checkpoint_identity = calls["export"][0]
    assert base_arg == str(base.resolve())
    assert solver_arg == str(solver)
    assert output_arg == str(outer / "merged_solver")
    assert checkpoint_identity is arm
    assert seen["checkpoint"] == str(checkpoint)
    assert seen["prompt_identity"] == identity
    assert seen["tokenizer_class"] == FakeTokenizer.__qualname__


def test_rank_telemetry_reports_stages_and_effective_rate():
    telemetry = eval_runner._telemetry_record(rank=2, setup_seconds=1.0,
        generation_seconds=2.0, environment_seconds=1.5, total_seconds=5.0,
        generated_token_count=10, request_count=4, episode_count=5)
    assert telemetry["rank"] == 2
    assert telemetry["timings_seconds"] == {"setup": 1.0, "generation": 2.0,
        "environment": 1.5, "total": 5.0}
    assert telemetry["generated_token_count"] == 10
    assert telemetry["request_count"] == 4
    assert telemetry["effective_episodes_per_minute"] == 60.0


def test_actual_vllm_policy_batches_chat_ids_with_independent_seeds(monkeypatch):
    from ez_vllm_policy import VLLMPolicy
    from rollout import RolloutConfig
    from protocol import encode_solver_prompt
    from test_evaluation import TinyTokenizer
    tokenizer = TinyTokenizer()
    tokenizer.eos_token_id = 0
    tokenizer.pad_token_id = 0
    captured = {}
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(
        SamplingParams=lambda **kwargs: types.SimpleNamespace(**kwargs)))
    class Engine:
        def generate(self, prompts, *, sampling_params, lora_request, use_tqdm):
            captured.update(prompts=prompts, params=sampling_params, adapter=lora_request)
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(
                token_ids=[ord(c) for c in "<action>look</action>"], finish_reason="stop")])
                for _ in prompts]
    prompts = [encode_solver_prompt(tokenizer, text, "chat_template") for text in ("first", "second")]
    policy = VLLMPolicy(Engine(), tokenizer)
    cfg = RolloutConfig(temperature=0.7, top_p=0.95, top_k=-1, min_p=0.0,
        repetition_penalty=1.0, max_new_tokens=1024, max_context_tokens=32768,
        stop_mode="action_tag", prompt_format="chat_template")
    texts, output_ids, stops = policy.generate_from_token_ids(prompts, cfg, request_seeds=[12, 34])
    assert captured["prompts"] == [{"prompt_token_ids": ids} for ids in prompts]
    assert captured["adapter"] is None
    assert [p.seed for p in captured["params"]] == [12, 34]
    for params in captured["params"]:
        assert (params.n, params.temperature, params.top_p, params.top_k) == (1, 0.7, 0.95, -1)
        assert (params.min_p, params.repetition_penalty, params.max_tokens) == (0.0, 1.0, 1024)
        assert params.stop == ["</action>"] and params.include_stop_str_in_output is True
    assert texts == ["<action>look</action>"] * 2
    assert stops == [len(ids) for ids in output_ids]
