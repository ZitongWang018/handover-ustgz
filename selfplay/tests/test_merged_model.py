"""Merged solver exports must be complete, safe, and solver-only."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
SELFPLAY = TESTS.parent
sys.path[:0] = [str(SELFPLAY), str(SELFPLAY / "src"), str(TESTS)]

import eval_runner
import merge_solver
from evaluation import canonical_hash
from test_chat_protocol import local_tokenizer

def _files(root):
    root = Path(root)
    return [{"path": path.relative_to(root).as_posix(), "size": path.stat().st_size,
             "sha256": eval_runner.sha256_file(path)}
            for path in sorted(root.rglob("*")) if path.is_file()]

def _arm(base, solver, checkpoint):
    _, _, digest = eval_runner._base_identity(str(base))
    checkpoint.mkdir(exist_ok=True)
    manifest = {"fixture": str(checkpoint)}
    (checkpoint / "training_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for name in ("COMPLETE.json", "checkpoint_inventory.json"):
        (checkpoint / name).write_text("{}", encoding="utf-8")
    return {"kind": "checkpoint", "path": str(checkpoint),
            "manifest_sha256": canonical_hash({"fixture": str(checkpoint)}),
            "solver_path": str(Path(solver).resolve()), "solver_files": _files(solver),
            "base_inventory_sha256": digest}

class _Tokenizer:
    def save_pretrained(self, path):
        (Path(path) / "tokenizer.json").write_text("{}", encoding="utf-8")

class _Peft:
    def __init__(self, calls, mutate):
        self.calls, self.mutate = calls, mutate
    def merge_and_unload(self, **kwargs):
        self.calls["merge"] = kwargs
        return self
    def eval(self):
        return self
    def to(self, **kwargs):
        self.calls["to"] = kwargs
        return self
    def state_dict(self):
        import torch
        return {"weight": torch.ones(2, dtype=torch.bfloat16)}
    def save_pretrained(self, path, **kwargs):
        self.calls["save"] = kwargs
        if self.mutate:
            self.mutate()
        path = Path(path)
        (path / "config.json").write_text('{"model_type": "fixture"}', encoding="utf-8")
        from safetensors.torch import save_file
        save_file(self.state_dict(), path / "model.safetensors")

def _fixtures(root, monkeypatch, mutate=None):
    import peft
    import torch
    import transformers
    root.mkdir(parents=True, exist_ok=True)
    base, solver = root / "base", root / "solver"
    base.mkdir()
    solver.mkdir()
    (base / "config.json").write_text('{"model_type": "fixture"}', encoding="utf-8")
    (base / "model.safetensors").write_bytes(b"base")
    (solver / "adapter_config.json").write_text("{}", encoding="utf-8")
    (solver / "adapter_model.safetensors").write_bytes(b"adapter")
    calls = {}
    def tokenizer_load(path, **kwargs):
        calls["tokenizer"] = (path, kwargs)
        return _Tokenizer()
    def model_load(path, **kwargs):
        calls["model"] = (path, kwargs)
        return _Peft(calls, None)
    def adapter_load(model, path, **kwargs):
        calls["adapter"] = (model, path, kwargs)
        return _Peft(calls, mutate)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer_load)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", model_load)
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", adapter_load)
    return base, solver, _arm(base, solver, root / "checkpoint"), calls, torch

def test_mock_export_validate_and_rejections(tmp_path, monkeypatch):
    base, solver, arm, calls, torch = _fixtures(tmp_path, monkeypatch)
    out = tmp_path / "output"
    manifest = merge_solver.export_solver_model(base, solver, out, checkpoint_identity=arm)
    assert merge_solver.validate_merged_model(
        out, base_digest=arm["base_inventory_sha256"], arm=arm) == manifest
    assert calls["model"][1]["torch_dtype"] is torch.bfloat16
    assert calls["model"][1]["device_map"] == {"": "cpu"}
    assert calls["model"][1]["low_cpu_mem_usage"] is True
    assert calls["model"][1]["local_files_only"] is True
    assert calls["tokenizer"][1]["local_files_only"] is True
    assert calls["adapter"][1] == str(solver.resolve())
    assert calls["merge"] == {"safe_merge": True}
    assert calls["save"] == {"safe_serialization": True, "max_shard_size": "5GB"}
    bad = copy.deepcopy(arm)
    bad["manifest_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="different checkpoint"):
        merge_solver.validate_merged_model(out, base_digest=arm["base_inventory_sha256"], arm=bad)
    with pytest.raises(ValueError, match="different base"):
        merge_solver.validate_merged_model(out, base_digest="0" * 64, arm=arm)
    config, weights = out / "config.json", out / "model.safetensors"
    config_bytes, weight_bytes = config.read_bytes(), weights.read_bytes()
    config.unlink()
    with pytest.raises(ValueError, match="missing"):
        merge_solver.validate_merged_model(out, base_digest=arm["base_inventory_sha256"], arm=arm)
    config.write_bytes(config_bytes)
    weights.write_bytes(weight_bytes + b"tamper")
    with pytest.raises(ValueError, match="missing, extra, or tampered"):
        merge_solver.validate_merged_model(out, base_digest=arm["base_inventory_sha256"], arm=arm)
    weights.write_bytes(weight_bytes)
    (out / "extra.bin").write_bytes(b"extra")
    with pytest.raises(ValueError, match="missing, extra, or tampered"):
        merge_solver.validate_merged_model(out, base_digest=arm["base_inventory_sha256"], arm=arm)
    (out / "extra.bin").unlink()
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        merge_solver.export_solver_model(base, solver, existing, checkpoint_identity=arm)
    assert not any(existing.iterdir())

def test_mock_source_mutation_aborts_export(tmp_path, monkeypatch):
    root = tmp_path / "mutated"
    def mutate():
        (root / "solver" / "adapter_model.safetensors").write_bytes(b"changed")
    base, solver, arm, _, _ = _fixtures(root, monkeypatch, mutate=mutate)
    out = tmp_path / "mutated-output"
    with pytest.raises(ValueError, match="changed during merge"):
        merge_solver.export_solver_model(base, solver, out, checkpoint_identity=arm)
    assert out.is_dir()
    with pytest.raises(ValueError, match="manifest is missing"):
        merge_solver.validate_merged_model(out, base_digest=arm["base_inventory_sha256"], arm=arm)


@pytest.mark.parametrize("damage", ["float32", "missing_parameter", "wrong_shape", "changed_seal"])
def test_bad_tensor_layout_or_changed_seal_never_gets_completion_marker(tmp_path, monkeypatch, damage):
    base, solver, arm, _, torch = _fixtures(tmp_path, monkeypatch)
    original = _Peft.save_pretrained
    def damaged_save(model, path, **kwargs):
        original(model, path, **kwargs)
        if damage == "changed_seal":
            (Path(arm["path"]) / "COMPLETE.json").write_text('{"changed": true}', encoding="utf-8")
        else:
            from safetensors.torch import save_file
            key = "other" if damage == "missing_parameter" else "weight"
            dtype = torch.float32 if damage == "float32" else torch.bfloat16
            shape = 3 if damage == "wrong_shape" else 2
            save_file({key: torch.ones(shape, dtype=dtype)}, Path(path) / "model.safetensors")
    monkeypatch.setattr(_Peft, "save_pretrained", damaged_save)
    out = tmp_path / "bad-output"
    with pytest.raises(ValueError, match="BF16|names/shapes|changed during merge"):
        merge_solver.export_solver_model(base, solver, out, checkpoint_identity=arm)
    assert out.is_dir() and not (out / "merged_model.json").exists()


def test_refuses_export_inside_source_before_writing(tmp_path, monkeypatch):
    base, solver, arm, calls, _ = _fixtures(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="overlap"):
        merge_solver.export_solver_model(base, solver, base / "derived", checkpoint_identity=arm)
    assert calls == {} and not (base / "derived").exists()

def _tokenizer32():
    tokenizer = local_tokenizer()
    tokenizer.add_tokens([f"token-{index}" for index in range(7, 32)])
    assert len(tokenizer) == 32
    return tokenizer

def test_real_tiny_qwen3moe_merges_only_solver(tmp_path, monkeypatch):
    import peft
    import torch
    import transformers
    from ez_train import build_dual_lora
    from ez_vllm_policy import export_lora_adapter
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    torch.manual_seed(29)
    config = Qwen3MoeConfig(vocab_size=32, hidden_size=8, intermediate_size=16,
        moe_intermediate_size=8, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=4, num_experts=2, num_experts_per_tok=1,
        max_position_embeddings=64)
    config.torch_dtype = torch.bfloat16
    base_model = Qwen3MoeForCausalLM(config).to(dtype=torch.bfloat16).eval()
    base_dir = tmp_path / "real-base"
    base_model.save_pretrained(base_dir, safe_serialization=True)
    _tokenizer32().save_pretrained(base_dir)
    base_files = _files(base_dir)
    ids = torch.tensor([[1, 4, 3]])
    with torch.no_grad():
        base_logits = base_model(input_ids=ids).logits.float()
    model = build_dual_lora(base_model, {"model": {
        "lora_r": 2, "lora_alpha": 2, "lora_targets": "all-linear"}})
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B.solver" in name:
                parameter.fill_(0.15)
            elif "lora_B.proposer" in name:
                parameter.fill_(-0.2)
    model.eval()
    with torch.no_grad():
        model.set_adapter("solver")
        solver_logits = model(input_ids=ids).logits.float()
        model.set_adapter("proposer")
        proposer_logits = model(input_ids=ids).logits.float()
    solver_dir = Path(export_lora_adapter(model, "solver", str(tmp_path / "solver-adapter")))
    proposer_dir = Path(export_lora_adapter(model, "proposer", str(tmp_path / "proposer-adapter")))
    solver_files, proposer_files = _files(solver_dir), _files(proposer_dir)
    arm = _arm(base_dir, solver_dir, tmp_path / "real-checkpoint")
    out = tmp_path / "real-merged"
    merge_solver.export_solver_model(base_dir, solver_dir, out, checkpoint_identity=arm)
    merge_solver.validate_merged_model(out, base_digest=arm["base_inventory_sha256"], arm=arm)
    merged = transformers.AutoModelForCausalLM.from_pretrained(
        out, torch_dtype="auto", trust_remote_code=True, local_files_only=True).eval()
    assert not isinstance(merged, peft.PeftModel)
    assert next(merged.parameters()).dtype is torch.bfloat16
    with torch.no_grad():
        merged_logits = merged(input_ids=ids).logits.float()
    torch.testing.assert_close(merged_logits, solver_logits, atol=0.02, rtol=0.02)
    assert not torch.allclose(merged_logits, base_logits, atol=1e-4, rtol=1e-4)
    assert not torch.allclose(merged_logits, proposer_logits, atol=1e-4, rtol=1e-4)
    assert _files(base_dir) == base_files
    assert _files(solver_dir) == solver_files
    assert _files(proposer_dir) == proposer_files
