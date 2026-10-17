"""Training-only isolation for the single-GPU vLLM rollout engine.

This module deliberately leaves the evaluation factory and training process group
alone. The vLLM EngineCore is constructed while torchrun metadata is absent, so its
spawned process cannot join the trainer's distributed world by inheriting env vars.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import json
import logging
import os
import sys
import traceback
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_EXACT_TRAINING_ENV = {
    "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
    "ROLE_RANK", "ROLE_WORLD_SIZE", "ROLE_NAME", "MASTER_ADDR", "MASTER_PORT",
    "VLLM_PORT",
}


def _is_training_env_name(name: str) -> bool:
    return (name in _EXACT_TRAINING_ENV or name.startswith("ROLE_")
            or name.startswith("TORCHELASTIC_") or name.startswith("VLLM_DP_"))


def clean_training_environment(env: Mapping[str, str]) -> dict[str, str]:
    """Return a copy without torchrun/elastic or vLLM data-parallel launch metadata.

    Device masks, NCCL/GLOO network selection, OMP settings, caches, model
    configuration, and ordinary vLLM settings are retained unchanged.
    """
    return {name: value for name, value in env.items()
            if not _is_training_env_name(name)}


def _write_diagnostic(diagnostics_dir: str | os.PathLike[str], rank: int,
                      record: dict[str, Any]) -> Path:
    directory = Path(diagnostics_dir)
    directory.mkdir(parents=True, exist_ok=True)
    # Exclusive creation ensures retries never overwrite earlier evidence.
    while True:
        path = directory / f"training_vllm_rank{rank}_{uuid.uuid4().hex}.json"
        try:
            with path.open("x", encoding="utf-8") as fh:
                record["diagnostic_path"] = str(path)
                json.dump(record, fh, indent=2, sort_keys=True)
                fh.write("\n")
            return path
        except FileExistsError:
            continue


def _supports_vllm_env(vllm_envs: Any, name: str) -> bool:
    registry = getattr(vllm_envs, "environment_variables", None)
    if isinstance(registry, Mapping):
        return name in registry
    # Some older builds expose env settings through module attributes instead.
    return hasattr(vllm_envs, name)


def _require_engine_capabilities(vllm_envs: Any) -> Any:
    """Require the API pieces needed for isolated V1 single-device execution."""
    try:
        from vllm import LLM
        from vllm.engine.arg_utils import EngineArgs
    except Exception as exc:
        raise RuntimeError(
            "Training vLLM startup requires an importable vLLM package exposing "
            "vllm.LLM and vllm.engine.arg_utils.EngineArgs; install a compatible "
            "V1-enabled vLLM build in the training environment.") from exc

    try:
        engine_params = set(inspect.signature(EngineArgs).parameters)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Cannot inspect vLLM EngineArgs; refusing to guess whether TP/PP/DP "
            "isolation is supported. Use a vLLM build with inspectable EngineArgs.") from exc
    required = {"tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
                "distributed_executor_backend"}
    missing = sorted(required - engine_params)
    if missing:
        raise RuntimeError(
            "Installed vLLM cannot explicitly isolate its executor: EngineArgs is "
            f"missing {missing}. Required settings are TP=PP=DP=1 and executor='uni'.")
    if not _supports_vllm_env(vllm_envs, "VLLM_ENABLE_V1_MULTIPROCESSING"):
        raise RuntimeError(
            "Installed vLLM does not expose VLLM_ENABLE_V1_MULTIPROCESSING; "
            "cannot guarantee a spawned V1 EngineCore subprocess. Upgrade to a "
            "V1 build that supports this setting.")

    # Older versions need this selector. Newer versions may have removed it;
    # those must resolve the public LLM engine path directly to V1.
    has_v1_selector = _supports_vllm_env(vllm_envs, "VLLM_USE_V1")
    if not has_v1_selector:
        try:
            llm_engine = importlib.import_module("vllm.engine.llm_engine").LLMEngine
        except Exception as exc:
            raise RuntimeError(
                "This vLLM build has no VLLM_USE_V1 selector and its default LLM "
                "engine cannot be confirmed as V1.") from exc
        if not str(getattr(llm_engine, "__module__", "")).startswith("vllm.v1."):
            raise RuntimeError(
                "This vLLM build has no VLLM_USE_V1 selector and its default LLM "
                "engine is not confirmed as V1. Refusing a V0 or in-process fallback.")
    return LLM


def build_training_vllm_engine(base_path: str, *, rank: int,
                               diagnostics_dir: str | os.PathLike[str],
                               **engine_kwargs: Any):
    """Build the training rollout engine in an isolated V1 single-GPU runtime.

    Sampling, seed, model length, batching, and memory-budget kwargs pass through
    unchanged. The only added topology kwargs are TP=PP=DP=1 and ``uni`` executor.
    """
    original_env = dict(os.environ)
    removed = sorted(name for name in original_env if _is_training_env_name(name))
    cleaned = clean_training_environment(original_env)
    # Remove stale selectors before probing the installed vLLM API. They are set
    # only if that build declares support for them.
    cleaned.pop("VLLM_USE_V1", None)
    engine_kwargs = dict(engine_kwargs)
    topology = {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "distributed_executor_backend": "uni",
    }
    for key, expected in topology.items():
        if key in engine_kwargs and engine_kwargs[key] != expected:
            raise ValueError(f"training vLLM requires {key}={expected}; received {engine_kwargs[key]!r}")

    record: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rank": int(rank),
        "base_path": str(base_path),
        "vllm_version": None,
        "vllm_distribution_version": None,
        "vllm_module_version": None,
        "vllm_module_file": None,
        "device_mask": original_env.get("CUDA_VISIBLE_DEVICES"),
        "topology": {**topology, "v1_engine": True, "v1_multiprocessing": True,
                     "worker_multiprocessing_method": "spawn"},
        "removed_env_var_names": removed,
        "status": "starting",
        "exception_traceback": None,
    }

    engine = None
    failure: BaseException | None = None
    try:
        device_mask = original_env.get("CUDA_VISIBLE_DEVICES")
        devices = [] if device_mask is None else [part.strip() for part in device_mask.split(",")]
        if len(devices) != 1 or not devices[0] or devices[0] == "-1":
            raise RuntimeError(
                "Training vLLM isolation requires CUDA_VISIBLE_DEVICES to name exactly "
                "one visible device. Start training through run_train.sh so each rank "
                "receives a single-GPU mask; this factory will not rewrite the device mask.")
        os.environ.clear()
        os.environ.update(cleaned)
        os.environ["VLLM_HOST_IP"] = "127.0.0.1"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"

        # Import vLLM only after torchrun and DP variables have been removed.
        vllm = importlib.import_module("vllm")
        record["vllm_module_version"] = getattr(vllm, "__version__", None)
        record["vllm_module_file"] = getattr(vllm, "__file__", None)
        try:
            record["vllm_distribution_version"] = importlib.metadata.version("vllm")
        except importlib.metadata.PackageNotFoundError:
            record["vllm_distribution_version"] = None
        record["vllm_version"] = (record["vllm_module_version"]
                                   or record["vllm_distribution_version"] or "unknown")
        vllm_envs = importlib.import_module("vllm.envs")
        if _supports_vllm_env(vllm_envs, "VLLM_USE_V1"):
            os.environ["VLLM_USE_V1"] = "1"
            if getattr(vllm_envs, "VLLM_USE_V1", True) is False:
                raise RuntimeError(
                    "vLLM imported V1 selector state before isolation; restart the training "
                    "process so VLLM_USE_V1=1 takes effect before importing its engine modules.")
        if getattr(vllm_envs, "VLLM_ENABLE_V1_MULTIPROCESSING", True) is False:
            raise RuntimeError(
                "vLLM imported multiprocessing state before isolation; restart the training "
                "process so the V1 EngineCore spawn setting takes effect.")

        LLM = _require_engine_capabilities(vllm_envs)
        kwargs = {
            "model": base_path,
            "tokenizer": base_path,
            "dtype": "bfloat16",
            "enable_lora": True,
            "max_loras": 2,
            "max_cpu_loras": 4,
            "disable_log_stats": True,
            **engine_kwargs,
            **topology,
        }
        budget_keys = (
            "gpu_memory_utilization", "max_model_len", "max_num_seqs",
            "max_num_batched_tokens", "max_new_tokens", "max_loras", "max_cpu_loras",
        )
        startup_record = {
            "rank": int(rank),
            "device_mask": device_mask,
            "vllm_version": record["vllm_version"],
            "vllm_module_file": record["vllm_module_file"],
            "topology": record["topology"],
            "generation_budgets": {key: kwargs[key] for key in budget_keys if key in kwargs},
            "diagnostics_dir": str(diagnostics_dir),
        }
        print("TRAINING_VLLM_STARTUP " + json.dumps(startup_record, sort_keys=True), flush=True)
        engine = LLM(**kwargs)
    except BaseException as exc:
        failure = exc
        record["status"] = "failed"
        record["exception_traceback"] = traceback.format_exc()
    finally:
        # Restore even mutations made by import or LLM construction in the parent.
        os.environ.clear()
        os.environ.update(original_env)

    try:
        if failure is None:
            record["status"] = "success"
        diagnostic_path = _write_diagnostic(diagnostics_dir, int(rank), record)
        logging.getLogger(__name__).info("Training vLLM diagnostic: %s", diagnostic_path)
    except Exception as diag_exc:
        if failure is not None:
            failure.args = (*failure.args, f"Could not write vLLM diagnostic: {diag_exc!r}")
        else:
            raise RuntimeError(f"vLLM engine started, but diagnostic writing failed: {diag_exc}") from diag_exc
    if failure is not None:
        raise RuntimeError(
            "Training vLLM startup failed. See the per-rank diagnostic JSON for the "
            "complete traceback; no alternate backend was selected.") from failure
    return engine
