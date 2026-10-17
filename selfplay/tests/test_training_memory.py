"""Unit tests for training_memory resource profiles and accounting."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from training_memory import (
    activation_offload_context,
    apply_training_memory_profile,
    build_memory_footprint_report,
)


_QWEN_MODEL_CONFIG = {"model_type": "qwen3_moe", "num_experts": 128}


def _valid_qwen_cfg():
    return {
        "run": {"name": "explorer-zero", "seed": 42, "output_dir": "runs"},
        "model": {
            "path": "/models/qwen3-30b-a3b",
            "dtype": "bfloat16",
            "attn_implementation": "sdpa",
            "gradient_checkpointing": True,
            "lora": True,
            "lora_r": 32,
            "lora_alpha": 32,
            "lora_targets": "all-linear",
        },
        "env": {"config_path": "alfworld.yaml", "split": "train"},
        "proposer": {
            "max_turns": 30,
            "max_new_tokens": 1024,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": -1,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "max_context_tokens": 32768,
            "keep_recent_turns": 15,
            "seed": None,
            "give_admissible": False,
            "finish_nudge": False,
        },
        "solver": {
            "style": "react",
            "max_turns": 30,
            "max_new_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": -1,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "max_context_tokens": 32768,
            "keep_recent_turns": 15,
            "seed": None,
            "stop_mode": "action_tag",
            "menu_enabled": True,
        },
        "train": {
            "outer_loops": 4,
            "group_size": 16,
            "lr": 1.0e-4,
            "clip_eps": 0.3,
            "loss_norm": "seq",
            "micro_batch_size": 1,
            "sync_bucket_mb": 25,
            "generation_max_batch_size": 2,
        },
        "runtime": {
            "world_size": 8,
            "backend": "hf",
            "vllm": {"max_model_len": 32768},
        },
        "eval": {
            "backend": "vllm",
            "gpu_memory_utilization": 0.85,
            "split": "valid_seen",
        },
        "baseline": {"backend": "hf", "checkpoint": "base"},
    }


def _assert_qwen_profile_rejected(cfg, *, model_config=None, backend="hf"):
    before = copy.deepcopy(cfg)
    with pytest.raises(ValueError):
        apply_training_memory_profile(
            cfg, _QWEN_MODEL_CONFIG if model_config is None else model_config,
            "qwen3_moe_cpu", backend,
        )
    assert cfg == before


def test_qwen_profile_changes_only_disclosed_resource_fields():
    cfg = _valid_qwen_cfg()
    model_config = copy.deepcopy(_QWEN_MODEL_CONFIG)
    expected = copy.deepcopy(cfg)
    expected["train"]["generation_max_batch_size"] = 1
    expected["runtime"]["memory_profile"] = "qwen3_moe_cpu"
    expected["runtime"]["optimizer_state_device"] = "cpu"
    expected["runtime"]["activation_offload"] = True

    returned = apply_training_memory_profile(
        cfg, model_config, "qwen3_moe_cpu", "hf"
    )

    assert returned == "hf"
    assert cfg == expected
    assert cfg["model"]["path"] == "/models/qwen3-30b-a3b"
    assert cfg["model"]["lora_r"] == 32
    assert cfg["model"]["lora_alpha"] == 32
    assert cfg["eval"] == _valid_qwen_cfg()["eval"]
    assert cfg["baseline"] == _valid_qwen_cfg()["baseline"]
    assert model_config == _QWEN_MODEL_CONFIG


def test_default_profile_is_a_noop_and_returns_backend():
    cfg = _valid_qwen_cfg()
    before = copy.deepcopy(cfg)

    returned = apply_training_memory_profile(
        cfg, {"model_type": "ignored"}, "default", "vllm"
    )

    assert returned == "vllm"
    assert cfg == before


def test_qwen_profile_rejects_vllm_backend():
    _assert_qwen_profile_rejected(_valid_qwen_cfg(), backend="vllm")


def test_qwen_profile_rejects_wrong_model_type():
    _assert_qwen_profile_rejected(
        _valid_qwen_cfg(),
        model_config={"model_type": "qwen2_moe"},
    )


def test_qwen_profile_rejects_wrong_lora_targets():
    cfg = _valid_qwen_cfg()
    cfg["model"]["lora_targets"] = "q_proj"
    _assert_qwen_profile_rejected(cfg)


def test_qwen_profile_rejects_non_bf16_dtype():
    cfg = _valid_qwen_cfg()
    cfg["model"]["dtype"] = "float16"
    _assert_qwen_profile_rejected(cfg)


def test_qwen_profile_rejects_wrong_world_size():
    cfg = _valid_qwen_cfg()
    cfg["runtime"]["world_size"] = 4
    _assert_qwen_profile_rejected(cfg)


def test_qwen_profile_rejects_disabled_lora():
    cfg = _valid_qwen_cfg()
    cfg["model"]["lora"] = False
    _assert_qwen_profile_rejected(cfg)


def test_qwen_profile_rejects_disabled_gradient_checkpointing():
    cfg = _valid_qwen_cfg()
    cfg["model"]["gradient_checkpointing"] = False
    _assert_qwen_profile_rejected(cfg)


def test_unknown_profile_is_rejected():
    cfg = _valid_qwen_cfg()
    before = copy.deepcopy(cfg)
    with pytest.raises(ValueError):
        apply_training_memory_profile(cfg, _QWEN_MODEL_CONFIG, "not-a-profile", "hf")
    assert cfg == before


def _gradient_snapshot(offload_enabled):
    torch.manual_seed(7)
    model = nn.Sequential(
        nn.Linear(4, 5),
        nn.Tanh(),
        nn.Linear(5, 2),
    )
    inputs = torch.linspace(-1.0, 1.0, 32).reshape(8, 4)
    with activation_offload_context(offload_enabled):
        loss = model(inputs).square().mean()
    loss.backward()
    return [param.grad.detach().clone() for param in model.parameters()]


def test_activation_offload_context_preserves_cpu_gradients(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    baseline = _gradient_snapshot(False)
    offloaded = _gradient_snapshot(True)

    assert len(baseline) == len(offloaded)
    for expected, actual in zip(baseline, offloaded):
        torch.testing.assert_close(actual, expected)


class _Weight(nn.Module):
    def __init__(self, shape, dtype, requires_grad=True):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=requires_grad)


class _TinyDualAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = _Weight((5, 7), torch.bfloat16, requires_grad=False)
        self.lora_A = nn.ModuleDict({
            "proposer": _Weight((2, 7), torch.float32),
            "solver": _Weight((3, 7), torch.float32),
        })
        self.lora_B = nn.ModuleDict({
            "proposer": _Weight((5, 2), torch.float32),
            "solver": _Weight((5, 3), torch.float32),
        })


def test_report_uses_actual_parameter_bytes(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    report = build_memory_footprint_report(
        _TinyDualAdapter(), profile="qwen3_moe_cpu"
    )

    assert report["frozen_base_bf16_bytes"] == 5 * 7 * 2
    assert report["proposer_fp32_bytes"] == (2 * 7 + 5 * 2) * 4
    assert report["solver_fp32_bytes"] == (3 * 7 + 5 * 3) * 4
    assert report["active_adapter_gradient_bytes"] == 36 * 4
    assert report["adamw_cpu_moments_bytes"] == 2 * (24 + 36) * 4
    assert report["largest_parameter_cpu_update_temp_estimate_bytes"] == 21 * 16
    assert report["param_counts"] == {
        "frozen_base": 35,
        "proposer": 24,
        "solver": 36,
        "adapter_total": 60,
        "model_total": 95,
    }
    assert report["param_dtypes"] == {
        "frozen_base": ["bfloat16"],
        "proposer": ["float32"],
        "solver": ["float32"],
    }
    assert report["cuda_available"] is False
    assert report["gpu_total_bytes"] is None
    assert "activation" in " ".join(report["not_predicted"]).lower()
    assert "kv cache" in " ".join(report["not_predicted"]).lower()
    assert "buckets" in " ".join(report["not_predicted"]).lower()
    assert "does not establish" in report["memory_fit_note"]
    json.dumps(report, sort_keys=True)
