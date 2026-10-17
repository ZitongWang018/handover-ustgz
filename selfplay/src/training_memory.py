"""Memory-profile helpers for Explorer-Zero training.

The footprint report accounts only for parameter storage and named optimizer
state. It deliberately does not model peak activation, KV-cache, or
communication-buffer memory.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import Optional

import torch

_PROFILE_DEFAULT = "default"
_PROFILE_QWEN3_MOE_CPU = "qwen3_moe_cpu"
_KNOWN_PROFILES = (_PROFILE_DEFAULT, _PROFILE_QWEN3_MOE_CPU)
_LORA_ADAPTER_MARKERS = frozenset(
    {
        "lora_A",
        "lora_B",
        "lora_embedding_A",
        "lora_embedding_B",
        "lora_magnitude_vector",
    }
)


def _require_section(cfg: dict, key: str) -> dict:
    if not isinstance(cfg, dict):
        raise ValueError("training config must be a dict")
    section = cfg.get(key)
    if not isinstance(section, dict):
        raise ValueError(f"training config section {key!r} must be a dict")
    return section


def apply_training_memory_profile(
    cfg: dict, model_config: dict, profile: str, backend: str
) -> str:
    """Apply one declared resource profile without changing unrelated config.

    The ``default`` profile is a strict no-op. The ``qwen3_moe_cpu`` profile
    requires an already-compatible HF/Qwen3-MoE training setup and only records
    the CPU/offload resources plus the disclosed generation-batch reduction.
    """
    if not isinstance(profile, str):
        raise ValueError(f"unknown training memory profile: {profile!r}")
    profile_name = profile.strip().lower()
    if profile_name == _PROFILE_DEFAULT:
        return backend
    if profile_name != _PROFILE_QWEN3_MOE_CPU:
        raise ValueError(
            f"unknown training memory profile {profile!r}; expected one of "
            f"{_KNOWN_PROFILES}"
        )
    if not isinstance(backend, str) or backend.strip().lower() != "hf":
        raise ValueError(
            "qwen3_moe_cpu requires backend='hf'; refusing backend "
            f"{backend!r} (no fallback is performed)"
        )
    if (
        not isinstance(model_config, Mapping)
        or model_config.get("model_type") != "qwen3_moe"
    ):
        raise ValueError("qwen3_moe_cpu requires model_type='qwen3_moe'")

    model = _require_section(cfg, "model")
    train = _require_section(cfg, "train")
    runtime = _require_section(cfg, "runtime")
    if model.get("dtype") != "bfloat16":
        raise ValueError("qwen3_moe_cpu requires model.dtype='bfloat16'")
    if model.get("lora") is not True:
        raise ValueError("qwen3_moe_cpu requires model.lora=True")
    if model.get("lora_targets") != "all-linear":
        raise ValueError("qwen3_moe_cpu requires model.lora_targets='all-linear'")
    if model.get("gradient_checkpointing") is not True:
        raise ValueError(
            "qwen3_moe_cpu requires model.gradient_checkpointing=True"
        )
    if runtime.get("world_size") != 8:
        raise ValueError("qwen3_moe_cpu requires runtime.world_size=8")

    # These are the only mutations; eval/baseline and sampling stay untouched.
    train["generation_max_batch_size"] = 1
    runtime["memory_profile"] = _PROFILE_QWEN3_MOE_CPU
    runtime["optimizer_state_device"] = "cpu"
    runtime["activation_offload"] = True
    return "hf"


def activation_offload_context(enabled: bool):
    """Return the configured autograd CPU-save hook or a no-op context."""
    if not enabled:
        return nullcontext()
    return torch.autograd.graph.save_on_cpu(
        pin_memory=torch.cuda.is_available()
    )


def _classify_adapter_parameter(name: str):
    parts = name.split(".")
    for index, part in enumerate(parts):
        if part in _LORA_ADAPTER_MARKERS:
            next_part = parts[index + 1] if index + 1 < len(parts) else None
            if next_part in {"weight", "bias"}:
                return None, True
            return next_part, True
    return None, any(part.startswith("lora_") for part in parts)


def _parameter_bytes(params) -> int:
    return int(sum(int(p.numel()) * int(p.element_size()) for p in params))


def _dtype_labels(params):
    return sorted({str(p.dtype).removeprefix("torch.") for p in params})


def build_memory_footprint_report(model, profile: Optional[str] = None) -> dict:
    """Return a JSON-serializable accounting report for post-PEFT weights.

    The largest-update estimate allows four FP32 scratch buffers per element
    of the largest adapter parameter. It excludes persistent AdamW moments,
    activations, KV cache, synchronization buckets, and allocator overhead.
    """
    buckets = {"frozen_base": [], "proposer": [], "solver": []}
    for name, param in model.named_parameters():
        adapter, is_lora = _classify_adapter_parameter(name)
        if is_lora:
            if adapter not in {"proposer", "solver"}:
                raise ValueError(
                    f"cannot classify PEFT parameter {name!r} by adapter"
                )
            buckets[adapter].append(param)
        else:
            buckets["frozen_base"].append(param)

    proposer_bytes = _parameter_bytes(buckets["proposer"])
    solver_bytes = _parameter_bytes(buckets["solver"])
    frozen_bytes = _parameter_bytes(buckets["frozen_base"])
    all_adapter_params = buckets["proposer"] + buckets["solver"]
    largest_adapter_numel = max(
        (int(param.numel()) for param in all_adapter_params), default=0
    )
    cuda_available = bool(torch.cuda.is_available())
    gpu_total_bytes = None
    if cuda_available:
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        gpu_total_bytes = int(properties.total_memory)

    proposer_count = sum(int(param.numel()) for param in buckets["proposer"])
    solver_count = sum(int(param.numel()) for param in buckets["solver"])
    frozen_count = sum(int(param.numel()) for param in buckets["frozen_base"])
    return {
        "profile": profile,
        "frozen_base_bf16_bytes": frozen_bytes,
        "proposer_fp32_bytes": proposer_bytes,
        "solver_fp32_bytes": solver_bytes,
        "active_adapter_gradient_bytes": max(proposer_bytes, solver_bytes),
        "adamw_cpu_moments_bytes": 2 * (proposer_bytes + solver_bytes),
        "largest_parameter_cpu_update_temp_estimate_bytes": 16 * largest_adapter_numel,
        "param_counts": {
            "frozen_base": frozen_count,
            "proposer": proposer_count,
            "solver": solver_count,
            "adapter_total": proposer_count + solver_count,
            "model_total": frozen_count + proposer_count + solver_count,
        },
        "param_dtypes": {
            "frozen_base": _dtype_labels(buckets["frozen_base"]),
            "proposer": _dtype_labels(buckets["proposer"]),
            "solver": _dtype_labels(buckets["solver"]),
        },
        "cuda_available": cuda_available,
        "gpu_total_bytes": gpu_total_bytes,
        "not_predicted": [
            "activation memory (including checkpoint recomputation peaks)",
            "KV cache",
            "gradient/adapter synchronization buckets",
            "allocator fragmentation and runtime overhead",
            "optimizer temporaries beyond the largest-parameter estimate",
        ],
        "memory_fit_note": (
            "Accounting only; this report does not establish that training "
            "fits in CPU or GPU memory."
        ),
    }


__all__ = [
    "activation_offload_context",
    "apply_training_memory_profile",
    "build_memory_footprint_report",
]
