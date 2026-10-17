"""Verify the explicit 30B training profile against an unchanged baseline."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train


def test_30b_manifest_binds_matched_hf_chat_baseline_without_changing_protocol(tmp_path, monkeypatch):
    import test_training_contract as fixtures
    import eval_runner
    import evaluation
    from checkpoint import base_file_inventory

    for name in ("ALFWORLD_DATA", "WORLD_SIZE", "RANK", "LOCAL_RANK", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    contract_path, config_path, _, data, _, baseline = fixtures._fixture(tmp_path, monkeypatch)
    config = train._load_config(train.SELFPLAY_ROOT / "config.yaml")
    config["env"]["config_path"] = str(train.SELFPLAY_ROOT / "alfworld.yaml")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    base = tmp_path / "model"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"model_type": "qwen3_moe",
        "max_position_embeddings": 262144}), encoding="utf-8")
    (base / "pytorch_model.bin").write_bytes(b"model inventory fixture, never loaded")
    (base / "tokenizer.json").write_text("{}", encoding="utf-8")
    inventory = base_file_inventory(base)
    digest = evaluation.canonical_hash(inventory)
    signature = baseline["protocol_signature"]
    signature.update(config=copy.deepcopy(config),
        config_sha256=evaluation.sha256_file(config_path),
        solver_settings=eval_runner._config_settings(config), base_inventory_sha256=digest)
    from protocol import solver_prompt_identity
    from test_evaluation import TinyTokenizer
    signature.update(solver_prompt_identity(TinyTokenizer(), config["solver"]["prompt_format"]))
    for key in ("backend", "batch_size", "gpu_memory_utilization"):
        signature[key] = config["eval"][key]
    tokens = {name: {"path": name, **record} for name, record in inventory.items()
              if name.startswith("tokenizer")}
    signature["tokenizer_files_sha256"] = evaluation.canonical_hash(tokens)
    baseline["base_path"] = str(base)
    baseline["base_inventory"] = {"files": [{"path": name, **record}
        for name, record in sorted(inventory.items())]}
    baseline["arm_identity"]["base_inventory_sha256"] = digest
    fixtures._resign(baseline)
    contract_path.write_text(json.dumps(baseline), encoding="utf-8")
    before = contract_path.read_bytes(), config_path.read_bytes()
    monkeypatch.setattr(train, "_resolve_compat", lambda: {"mode": "native", "path": None})
    monkeypatch.setattr(train, "_pool_manifest", lambda *args: {"split": "train", "count": 1})
    resolved, manifest = train.resolve_training_manifest(config_path, base, data, "hf", 8,
        baseline_contract_path=contract_path, memory_profile="qwen3_moe_cpu")
    assert resolved["runtime"]["backend"] == "hf"
    assert resolved["runtime"]["optimizer_state_device"] == "cpu"
    assert resolved["runtime"]["activation_offload"] is True
    assert resolved["train"]["generation_max_batch_size"] == 1
    assert resolved["solver"] == config["solver"]
    assert resolved["proposer"]["max_context_tokens"] == 40960
    for name, value in config["train"].items():
        if name != "generation_max_batch_size":
            assert resolved["train"][name] == value
    assert manifest["baseline_binding"]["contract"] == baseline
    assert manifest["contract_sha256"] == train.contract_signature(manifest)
    assert (contract_path.read_bytes(), config_path.read_bytes()) == before
    with pytest.raises(train.ContractError, match="qwen3_moe_cpu"):
        train.resolve_training_manifest(config_path, base, data, "vllm", 8,
            baseline_contract_path=contract_path, memory_profile="qwen3_moe_cpu")
    with pytest.raises(train.ContractError, match="explicit"):
        train.resolve_training_manifest(config_path, base, data, "hf", 8,
            baseline_contract_path=contract_path)


def test_bounded_update_probe_performs_exactly_one_real_update():
    import torch
    from cpu_adamw import CPUStateAdamW
    from ez_train import update_probe

    model = torch.nn.Linear(4, 3)
    optimizer = CPUStateAdamW(model.parameters(), lr=0.01)
    model(torch.ones(2, 4)).sum().backward()
    before = model.weight.detach().clone()
    result = update_probe(model, list(model.parameters()), optimizer, "test", bounded_cpu=True)
    assert result > 0 and not torch.equal(model.weight, before)
    assert all(float(state["step"]) == 1 for state in optimizer.state.values())


@pytest.mark.parametrize("profile", ["default", "qwen3_moe_cpu"])
def test_gpu_loader_streams_only_the_explicit_memory_profile(tmp_path, monkeypatch, profile):
    import torch
    import grpo
    from types import SimpleNamespace

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    calls = []
    class FakeModel:
        config = SimpleNamespace(use_cache=True)
        def gradient_checkpointing_enable(self, **kwargs): pass
        def train(self): return self
        def to(self, device): return self
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(grpo.AutoTokenizer, "from_pretrained", lambda *a, **k: SimpleNamespace())
    def load(*args, **kwargs):
        calls.append(kwargs)
        return FakeModel()
    monkeypatch.setattr(grpo.AutoModelForCausalLM, "from_pretrained", load)
    grpo.build_model_and_tokenizer({"model": {"path": str(tmp_path),
        "dtype": "bfloat16", "attn_implementation": "sdpa"},
        "runtime": {"memory_profile": profile}}, 0, 8)
    if profile == "qwen3_moe_cpu":
        assert calls[0]["device_map"] == {"": torch.device("cuda", 0)}
        assert calls[0]["low_cpu_mem_usage"] is True
    else:
        assert "device_map" not in calls[0] and "low_cpu_mem_usage" not in calls[0]
