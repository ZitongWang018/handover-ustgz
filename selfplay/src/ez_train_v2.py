"""Strict fresh/resume entry point for the version-3 EZ training contract.

This module resolves local model/data/source identity before importing the
training runtime or constructing a model. It then delegates the already-audited
GRPO/state/verifier implementation to ez_train.py's explicit protocol-v3 path.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import sys
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parent
SELFPLAY_ROOT = SRC_DIR.parent
REPO_ROOT = SELFPLAY_ROOT.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ez_train_v2_contract import (  # noqa: E402
    ContractError,
    base_file_inventory,
    canonical_json_bytes,
    contract_signature,
    environment_file_identity,
    sha256_bytes,
    sha256_file,
    validate_checkpoint_v3,
)


SOURCE_PATHS = (
    "selfplay/src/ez_train.py",
    "selfplay/src/ez_train_v2.py",
    "selfplay/src/ez_train_v2_contract.py",
    "selfplay/src/ez_dist.py",
    "selfplay/src/ez_proposer.py",
    "selfplay/src/ez_solver.py",
    "selfplay/src/ez_state.py",
    "selfplay/src/ez_verify.py",
    "selfplay/src/ez_prompts.py",
    "selfplay/src/rollout.py",
    "selfplay/src/train_grpo.py",
    "selfplay/src/grpo_norm.py",
    "selfplay/src/reward.py",
    "selfplay/src/prompts.py",
    "selfplay/src/ez_swanlab.py",
    "selfplay/src/alfworld_env.py",
    "selfplay/src/alfworld_compat.py",
    "selfplay/src/ez_vllm_policy.py",
    "selfplay/configs/ez_repro_v2.yaml",
    "selfplay/scripts/run_ez_train_v2_8gpu.sh",
)
LEGACY_SEMANTIC_ENV = (
    "EZ_RUN_OUTPUT_DIR", "ZTWANG_ROLLOUT_BACKEND", "ZTWANG_VLLM_GMEM",
    "ZTWANG_VLLM_MAX_SEQS", "ZTWANG_VLLM_BATCHED", "EZ_EVAL_PRETEST",
    "EZ_SMOKE_EVAL", "EZ_SMOKE_G", "EZ_SMOKE_G_P", "EZ_SMOKE_TPS",
    "EZ_SMOKE_TURNS",
)
PROMPT_ENV = (
    "EZ_SOLVER_STYLE", "EZ_SOLVER_ADMISSIBLE", "EZ_SOLVER_FEWSHOT_SCHEMA",
    "EZ_SOLVER_FEWSHOT", "EZ_SOLVER_STRICT_RESOLVE", "EZ_SOLVER_BARE_FALLBACK",
    "EZ_PARSER_STRICT",
)
TOKENIZER_BASENAMES = {
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.json", "vocab.txt", "merges.txt",
    "tokenizer.model", "spiece.model", "chat_template.jinja",
}


def _json_load(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot parse {Path(path).name}: {type(exc).__name__}") from None


def _sha_obj(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _write_exact(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ContractError(f"existing run artifact differs: {path.name}")
        return
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("xb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(temp, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ContractError(f"concurrent rank wrote different {path.name}") from None
        temp.unlink(missing_ok=True)
    except OSError as exc:
        temp.unlink(missing_ok=True)
        raise ContractError(f"cannot write {path.name}: {type(exc).__name__}") from None


def _load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        raise ContractError("PyYAML must already be installed in the selected local environment") from None
    try:
        with Path(path).open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as exc:
        raise ContractError(f"cannot load config: {type(exc).__name__}") from None
    if not isinstance(cfg, dict):
        raise ContractError("training config must be a YAML mapping")
    return cfg


def _source_inventory() -> dict[str, dict[str, Any]]:
    records = {}
    for rel in SOURCE_PATHS:
        path = REPO_ROOT / rel
        if not path.is_file():
            raise ContractError(f"required training source is missing: {rel}")
        raw = path.read_bytes()
        records[rel] = {"size": len(raw), "sha256": sha256_bytes(raw)}
    return records


def _package_versions() -> dict[str, str | None]:
    names = ("torch", "transformers", "peft", "vllm", "alfworld", "textworld",
             "numpy", "PyYAML", "swanlab")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions}


def _resolve_path(value: str | Path, relative_to: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(value))))
    if "$" in str(path):
        raise ContractError(f"unexpanded environment variable in path: {value}")
    return path.resolve() if path.is_absolute() else (relative_to / path).resolve()


def _check_environment(cfg: dict[str, Any], base: Path, data_root: Path,
                       backend: str, world_size: int) -> dict[str, Any]:
    conflicts = [name for name in LEGACY_SEMANTIC_ENV if os.environ.get(name) not in (None, "")]
    if conflicts:
        raise ContractError("legacy environment overrides are forbidden in v2: " + ", ".join(conflicts))
    expected_prompt = {
        "EZ_SOLVER_STYLE": str(cfg["solver"]["style"]),
        "EZ_SOLVER_ADMISSIBLE": "1" if cfg["solver"]["menu_enabled"] else "0",
        "EZ_SOLVER_FEWSHOT_SCHEMA": "1" if cfg["solver"]["fewshot"] else "0",
        "EZ_SOLVER_FEWSHOT": "1" if cfg["solver"]["fewshot"] else "0",
        "EZ_SOLVER_STRICT_RESOLVE": "1" if cfg["solver"]["parser_strict"] else "0",
        "EZ_SOLVER_BARE_FALLBACK": "1" if cfg["solver"]["fallback_mode"] == "bare" else "0",
        "EZ_PARSER_STRICT": "1" if cfg["solver"]["parser_strict"] else "0",
    }
    for name, wanted in expected_prompt.items():
        current = os.environ.get(name)
        if current is not None and current != wanted:
            raise ContractError(f"{name} conflicts with resolved v2 config (wanted {wanted})")
    for name, wanted in (("ZTWANG_MODEL_PATH", str(base)), ("ALFWORLD_DATA", str(data_root))):
        current = os.environ.get(name)
        if current and Path(current).expanduser().resolve() != Path(wanted).resolve():
            raise ContractError(f"{name} conflicts with explicit CLI path")
    inherited_backend = os.environ.get("ZTWANG_ROLLOUT_BACKEND")
    if inherited_backend and inherited_backend.lower() != backend:
        raise ContractError("ZTWANG_ROLLOUT_BACKEND conflicts with required --backend")
    inherited_world = os.environ.get("WORLD_SIZE")
    if inherited_world and int(inherited_world) != world_size:
        raise ContractError("torchrun WORLD_SIZE conflicts with runtime.world_size")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not 0 <= rank < world_size:
        raise ContractError("torchrun RANK is outside the configured world")
    mask = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = [item.strip() for item in mask.split(",") if item.strip()] if mask else []
    if mask and len(devices) == 1 and world_size > 1:
        if local_rank != 0:
            raise ContractError("single-visible-GPU-per-rank launch requires LOCAL_RANK=0")
        mapping = {"launch_mode": "one_visible_gpu_per_rank", "rank": rank,
                   "physical_device_mask": devices[0]}
    elif mask and len(devices) == world_size:
        if len(set(devices)) != len(devices):
            raise ContractError("CUDA_VISIBLE_DEVICES contains a repeated physical device")
        if not 0 <= local_rank < world_size:
            raise ContractError("LOCAL_RANK is outside CUDA_VISIBLE_DEVICES")
        mapping = {"launch_mode": "shared_full_mask", "rank": rank,
                   "local_rank": local_rank, "visible_devices": devices}
    elif mask:
        raise ContractError(f"CUDA_VISIBLE_DEVICES exposes {len(devices)} devices; expected one per rank or {world_size}")
    else:
        mapping = {"launch_mode": "inherited_device_visibility", "rank": rank,
                   "local_rank": local_rank, "visible_devices": None}
    return {
        "resolved_prompt_environment": expected_prompt,
        "offline_mode": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                         "HF_DATASETS_OFFLINE": "1"},
        "device_environment": {
            "device_mapping": mapping,
            "omp_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_threads": os.environ.get("MKL_NUM_THREADS"),
        },
    }


def _resolve_compat() -> dict[str, Any]:
    from alfworld_compat import find_compat_dir

    explicit = os.environ.get("ALFWORLD_GLIBC_COMPAT_DIR")
    found = find_compat_dir()
    if explicit and (not found or Path(found).resolve() != Path(explicit).resolve()):
        raise ContractError("ALFWORLD_GLIBC_COMPAT_DIR is invalid or not the selected compat directory")
    require = os.environ.get("SKR_REQUIRE_GLIBC_COMPAT") == "1"
    if require and not found:
        raise ContractError("SKR_REQUIRE_GLIBC_COMPAT=1 but no local compatibility library exists")
    info: dict[str, Any] = {"mode": "patched" if found else "native", "path": found,
                            "require_flag": require, "files": {}}
    if found:
        for name in ("libdownward.so", "libglibc_single_threaded_shim.so"):
            path = Path(found) / name
            info["files"][name] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
        from alfworld_compat import install
        install(verbose=False)
    return info


def _pool_manifest(data_root: Path, env_config: Path, split: str) -> dict[str, Any]:
    from alfworld_env import collect_game_files, load_alfworld_config

    os.environ["ALFWORLD_DATA"] = str(data_root)
    cfg = load_alfworld_config(str(env_config))
    games = collect_game_files(cfg, split=split)
    if not games:
        raise ContractError(f"ALFWorld training pool is empty for split={split}")
    environment_files = environment_file_identity(cfg, env_config, [Path(game) for game in games])
    trajectory_by_game = {
        row["task_path"]: row["trajectory"]
        for row in environment_files["task_trajectories"]
    }
    entries = []
    for game in games:
        path = Path(game).resolve()
        try:
            rel = path.relative_to(data_root).as_posix()
        except ValueError:
            raise ContractError("a collected game file resolves outside ALFWORLD_DATA") from None
        if not path.is_file():
            raise ContractError(f"collected task file is missing: {rel}")
        row = {"path": rel, "size": path.stat().st_size, "sha256": sha256_file(path)}
        row["trajectory"] = trajectory_by_game[str(path)]
        entries.append(row)
    return {
        "split": split,
        "count": len(entries),
        "ordering": "exact order returned by installed ALFWorld collect_game_files; duplicates retained",
        "pool_order_sha256": _sha_obj(entries),
        "entries": entries,
        "logic_files": environment_files["logic_files"],
    }


def resolve_training_manifest(config_path: Path, base_path: Path, data_root: Path,
                              backend: str, world_size: int, *, smoke: bool = False
                              ) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = _load_config(config_path)
    if cfg.get("protocol_version") != 3 or cfg.get("protocol_name") != "explorer_zero_training_v2":
        raise ContractError("new training entry accepts only protocol_version=3 explorer_zero_training_v2 config")
    if backend not in {"hf", "vllm"}:
        raise ContractError("--backend must be explicitly hf or vllm")
    base = Path(base_path).expanduser().resolve()
    data = Path(data_root).expanduser().resolve()
    if not base.is_dir() or not (base / "config.json").is_file():
        raise ContractError("--base must be an existing local model directory with config.json")
    if not data.is_dir():
        raise ContractError("--alfworld-data must be an existing local data directory")
    if isinstance(world_size, bool) or world_size != int(cfg.get("runtime", {}).get("world_size", -1)):
        raise ContractError("requested world size differs from explicit runtime.world_size")
    _check_environment(cfg, base, data, backend, world_size)
    cfg.setdefault("model", {})["path"] = str(base)
    cfg.setdefault("env", {})["data_root"] = str(data)
    cfg["runtime"]["backend"] = backend
    cfg["runtime"]["world_size"] = int(world_size)
    if smoke:
        cfg["train"]["outer_loops"] = 1
        cfg["train"]["proposer_steps"] = 1
        cfg["train"]["solver_steps"] = 1
    if cfg["env"].get("split") != "train":
        raise ContractError("v2 training pool must be the explicit train split; eval split is forbidden")
    if cfg["train"].get("eval_every_outer") != 0 or cfg["train"].get("eval_n_tasks") != 0:
        raise ContractError("v2 fresh training forbids valid_unseen mid-eval/pretest")
    if cfg["solver"].get("style") != "ours" or cfg["solver"].get("rich") or cfg["solver"].get("fewshot"):
        raise ContractError("v2 training freezes the paper-approx JSON scaffold with no rich/fewshot")
    if cfg["solver"].get("menu_enabled") or cfg["solver"].get("fallback_mode") != "raw":
        raise ContractError("v2 training freezes menu-off and raw action fallback")
    if cfg["proposer"].get("max_new_tokens", 0) < 1 or cfg["solver"].get("max_new_tokens", 0) < 1:
        raise ContractError("generation limits must be explicit positive integers")
    if cfg["train"].get("group_size") != 16 or cfg["train"].get("tasks_per_step") != 1 \
            or cfg["train"].get("solve_rate_rollouts") != 8:
        raise ContractError("v2 protocol requires global G=16, TPS=1, GP=8")
    vllm_cfg = cfg["runtime"].get("vllm", {})
    if backend == "vllm":
        if int(vllm_cfg.get("max_model_len", -1)) != int(cfg["solver"]["max_context_tokens"]):
            raise ContractError("vLLM max_model_len must equal the resolved solver context limit")
        if int(vllm_cfg.get("max_lora_rank", -1)) != int(cfg["model"]["lora_r"]):
            raise ContractError("vLLM max_lora_rank must equal the resolved adapter rank")
        if isinstance(vllm_cfg.get("seed"), bool) or not isinstance(vllm_cfg.get("seed"), int) or vllm_cfg["seed"] < 0:
            raise ContractError("vLLM engine seed must be an explicit non-negative integer")
    env_config = _resolve_path(cfg["env"]["config_path"], SELFPLAY_ROOT)
    if not env_config.is_file():
        raise ContractError("configured ALFWorld environment YAML is missing")
    compat = _resolve_compat()
    pool = _pool_manifest(data, env_config, str(cfg["env"]["split"]))
    base_files = base_file_inventory(base)
    tokenizer_files = {name: record for name, record in base_files.items()
                       if Path(name).name in TOKENIZER_BASENAMES
                       or Path(name).name in {"config.json", "generation_config.json"}}
    if not any(Path(name).name in TOKENIZER_BASENAMES for name in tokenizer_files):
        raise ContractError("local base has no recognizable tokenizer files")
    model_config = _json_load(base / "config.json")
    gen_config_path = base / "generation_config.json"
    gen_config = _json_load(gen_config_path) if gen_config_path.is_file() else None
    versions = _package_versions()
    source_files = _source_inventory()
    config_resolved = json.loads(json.dumps(cfg, sort_keys=True))
    config_resolved["env"]["config_path"] = str(env_config)
    config_resolved["run"]["output_dir"] = str(_resolve_path(cfg["run"]["output_dir"], SELFPLAY_ROOT))
    config_resolved["model"]["path"] = str(base)
    config_resolved["env"]["data_root"] = str(data)
    env_override = _check_environment(cfg, base, data, backend, world_size)
    manifest = {
        "schema_version": 3,
        "protocol_name": "explorer_zero_training_v2",
        "paper_status": "paper_approx_with_explicit_undisclosed_parameter_choices",
        "checkpoint_eligibility": "smoke_only" if smoke else "formal_training",
        "resolved_config": config_resolved,
        "identity": {
            "base_path": str(base),
            "base_inventory_sha256": _sha_obj(base_files),
            "base_file_inventory": base_files,
            "metadata_hints": {
                "config_model_type": model_config.get("model_type"),
                "config_name_or_path": model_config.get("_name_or_path"),
                "config_eos_token_id": model_config.get("eos_token_id"),
                "config_max_position_embeddings": model_config.get("max_position_embeddings"),
                "generation_config": gen_config,
                "weights_verified_by_content_hash": True,
                "model_semantics_verified_by_hash": False,
            },
            "tokenizer_inventory_sha256": _sha_obj(tokenizer_files),
            "tokenizer_file_inventory": tokenizer_files,
        },
        "environment": {
            "alfworld_config_path": str(env_config.relative_to(REPO_ROOT).as_posix()),
            "alfworld_config_sha256": sha256_file(env_config),
            "data_root": str(data),
            "train_pool": pool,
            "compatibility": compat,
            "resolved_overrides": env_override,
        },
        "source_files": source_files,
        "source_inventory_sha256": _sha_obj(source_files),
        "software": versions,
        "backend": {
            "selected": backend,
            "attention_requested": cfg["model"].get("attn_implementation"),
            "dtype": cfg["model"].get("dtype"),
            "generation": {key: cfg[key] for key in ("proposer", "solver")},
            "vllm_engine": cfg["runtime"].get("vllm", {}),
            "vllm_seed_policy": ("engine_seed_base_plus_global_rank"
                                 if backend == "vllm" else None),
            "stateful_resume_non_bitwise": True,
        },
        "effective_prompt_flags": {
            "style": cfg["solver"]["style"],
            "menu_enabled": bool(cfg["solver"]["menu_enabled"]),
            "fewshot": bool(cfg["solver"]["fewshot"]),
            "rich": bool(cfg["solver"]["rich"]),
            "parser_strict": bool(cfg["solver"]["parser_strict"]),
            "fallback_mode": cfg["solver"]["fallback_mode"],
            "proposer_menu_enabled": bool(cfg["proposer"]["give_admissible"]),
        },
        "run_metadata": {
            "resolved_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "device_environment": env_override.pop("device_environment", {}),
        },
    }
    manifest["contract_sha256"] = contract_signature(manifest)
    return cfg, manifest


def _assert_run_dir(config: dict[str, Any], run_id: str, resume: Path | None) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", run_id):
        raise ContractError("--run-id must contain 1-64 portable filename characters")
    if resume is not None:
        resolved = Path(resume).expanduser().resolve()
        if resolved.name not in {"ckpt", "ckpt.previous"}:
            raise ContractError("--resume must point to a v3 run's ckpt directory")
        return resolved.parent
    output_root = Path(config["run"]["output_dir"])
    if not output_root.is_absolute():
        output_root = (SELFPLAY_ROOT / output_root).resolve()
    run_dir = output_root / f"{config['run']['name']}-{run_id}"
    return run_dir


def _dist_preflight(manifest: dict[str, Any], run_dir: Path, run_id: str,
                    resume: Path | None) -> tuple[dict[str, Any], Path | None, Path]:
    """Initialize the process group before model load and agree on shared artifacts."""
    import torch.distributed as dist
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world != int(manifest["resolved_config"]["runtime"]["world_size"]):
        raise ContractError(f"WORLD_SIZE={world} does not match v2 contract")
    if world > 1 and not dist.is_initialized():
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
            pg_backend = "nccl"
        else:
            # CPU/Gloo is useful for preflight and checkpoint-contract tests; the
            # actual EZ model loader still requires the configured model device.
            pg_backend = "gloo"
        dist.init_process_group(backend=pg_backend, rank=rank, world_size=world)
    signatures: list[Any] = [None] * world
    if world > 1:
        dist.all_gather_object(signatures, manifest["contract_sha256"])
    else:
        signatures = [manifest["contract_sha256"]]
    if any(item != manifest["contract_sha256"] for item in signatures):
        raise ContractError("rank-local resolved training contracts differ")

    shared: list[Any] = [None]
    if rank == 0:
        try:
            if resume is not None:
                candidates = [resume]
                if resume.name == "ckpt":
                    candidates.append(resume.parent / "ckpt.previous")
                info = None
                selected = None
                errors = []
                for candidate in candidates:
                    try:
                        info = validate_checkpoint_v3(candidate, expected_contract=manifest,
                                                       purpose="resume")
                        selected = candidate.resolve()
                        break
                    except (ContractError, OSError) as exc:
                        errors.append(f"{candidate.name}: {exc}")
                if info is None or selected is None:
                    raise ContractError("no complete matching v3 resume checkpoint: " + "; ".join(errors))
                ckpt_manifest = info["manifest"]
                if info["state"].get("run_id") != run_id:
                    raise ContractError("resume run_id differs from sealed checkpoint state")
                if contract_signature(ckpt_manifest) != manifest["contract_sha256"]:
                    raise ContractError("resume checkpoint contract differs from current v2 config")
                shared[0] = {"ok": True, "run_dir": str(selected.parent),
                             "manifest": ckpt_manifest, "selected_checkpoint": str(selected)}
            else:
                if run_dir.exists():
                    raise ContractError(f"fresh run directory already exists; refusing overwrite: {run_dir}")
                run_dir.mkdir(parents=True, exist_ok=False)
                shared[0] = {"ok": True, "run_dir": str(run_dir), "manifest": manifest,
                             "selected_checkpoint": None}
        except Exception as exc:  # broadcast errors so other ranks never wait at a later collective
            shared[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if world > 1:
        dist.broadcast_object_list(shared, src=0)
    result = shared[0]
    if not result or not result.get("ok"):
        raise ContractError(result.get("error", "rank zero v2 preflight failed") if result else "v2 preflight failed")
    run_dir = Path(result["run_dir"])
    manifest = result["manifest"]
    write_status: list[Any] = [None]
    if rank == 0:
        try:
            config_path = run_dir / "resolved_config.yaml"
            manifest_path = run_dir / "training_manifest.json"
            import yaml
            cfg_payload = yaml.safe_dump(manifest["resolved_config"], sort_keys=True,
                                         allow_unicode=True).encode("utf-8")
            manifest_payload = (json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                                           indent=2, allow_nan=False) + "\n").encode("utf-8")
            _write_exact(config_path, cfg_payload)
            _write_exact(manifest_path, manifest_payload)
            if resume is None:
                _write_exact(run_dir / "run_protocol.json", canonical_json_bytes({
                    "schema_version": 3, "contract_sha256": manifest["contract_sha256"],
                    "run_id": run_id, "protocol": "explorer_zero_training_v2"}) + b"\n")
            write_status[0] = {"ok": True}
        except Exception as exc:  # broadcast before any rank reaches a barrier
            write_status[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if world > 1:
        dist.broadcast_object_list(write_status, src=0)
    if not write_status[0] or not write_status[0].get("ok"):
        raise ContractError(write_status[0].get("error", "rank zero could not write v2 run artifacts"))
    if world > 1:
        dist.barrier()
    if resume is not None:
        # Preserve the original sealed manifest, including its non-semantic run metadata.
        return manifest, Path(result["selected_checkpoint"]), run_dir
    return manifest, None, run_dir


def _agree_preflight(manifest: dict[str, Any] | None, error: str | None,
                     expected_world: int, run_id: str,
                     resume: Path | None) -> list[dict[str, Any]]:
    """Propagate local manifest errors before one rank can enter model loading."""
    import torch.distributed as dist
    env_world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if env_world > 1 and not dist.is_initialized():
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
            pg_backend = "nccl"
        else:
            pg_backend = "gloo"
        dist.init_process_group(backend=pg_backend, rank=rank, world_size=env_world)
    if env_world != expected_world:
        error = error or f"WORLD_SIZE={env_world} does not match expected {expected_world}"
    payload = {"error": error, "contract_sha256": (manifest or {}).get("contract_sha256"),
               "run_id": run_id,
               "resume": str(Path(resume).resolve()) if resume is not None else None,
               "device_environment": ((manifest or {}).get("run_metadata", {})
                                      .get("device_environment", {}))}
    gathered: list[Any] = [None] * env_world
    if env_world > 1 and dist.is_initialized():
        dist.all_gather_object(gathered, payload)
    else:
        gathered = [payload]
    errors = [f"rank {i}: {item['error']}" for i, item in enumerate(gathered) if item and item.get("error")]
    signatures = {item.get("contract_sha256") for item in gathered if item}
    if errors:
        raise ContractError("distributed v2 preflight rejected: " + "; ".join(errors))
    if len(signatures) != 1 or None in signatures:
        raise ContractError("rank-local v2 training contracts differ or are unavailable")
    if len({(item.get("run_id"), item.get("resume")) for item in gathered if item}) != 1:
        raise ContractError("ranks supplied different run ids or resume checkpoints")
    return [dict(item.get("device_environment") or {}) for item in gathered]


def _install_resolved_prompt_environment(cfg: dict[str, Any]) -> None:
    solver = cfg["solver"]
    values = {
        "EZ_SOLVER_STYLE": str(solver["style"]),
        "EZ_SOLVER_ADMISSIBLE": "1" if solver["menu_enabled"] else "0",
        "EZ_SOLVER_FEWSHOT_SCHEMA": "1" if solver["fewshot"] else "0",
        "EZ_SOLVER_FEWSHOT": "1" if solver["fewshot"] else "0",
        "EZ_SOLVER_STRICT_RESOLVE": "1" if solver["parser_strict"] else "0",
        "EZ_SOLVER_BARE_FALLBACK": "1" if solver["fallback_mode"] == "bare" else "0",
        "EZ_PARSER_STRICT": "1" if solver["parser_strict"] else "0",
        "EZ_EVAL_PRETEST": "0",
        "EZ_SMOKE_EVAL": "0",
        "ALFWORLD_DATA": str(cfg["env"]["data_root"]),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    for name, value in values.items():
        os.environ[name] = value
    for name in LEGACY_SEMANTIC_ENV:
        os.environ.pop(name, None)


def _strict_model_loader():
    """Build HF model/tokenizer from the explicit local base with Hub/code disabled."""
    def build(cfg, local_rank, world, wrap_ddp=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if wrap_ddp:
            raise ContractError("v2 EZ model must remain raw; manual adapter sync is used")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        model_cfg = cfg["model"]
        path = str(Path(model_cfg["path"]).resolve())
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                     "float32": torch.float32}
        if model_cfg.get("dtype") not in dtype_map:
            raise ContractError("unsupported explicit model dtype")
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True,
                                                  trust_remote_code=False)
        tokenizer.padding_side = "right"
        model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, trust_remote_code=False,
            torch_dtype=dtype_map[model_cfg["dtype"]],
            attn_implementation=model_cfg["attn_implementation"],
        )
        model.config.use_cache = False
        if model_cfg.get("gradient_checkpointing"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        if torch.cuda.is_available():
            model = model.to(torch.device("cuda", local_rank))
        return model, tokenizer
    return build


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SELFPLAY_ROOT / "configs/ez_repro_v2.yaml")
    parser.add_argument("--base", type=Path, required=True, help="existing local model directory; no Hub fallback")
    parser.add_argument("--alfworld-data", type=Path, required=True, help="existing local ALFWorld data root")
    parser.add_argument("--backend", choices=("hf", "vllm"), required=True)
    parser.add_argument("--run-id", required=True, help="shared id passed identically to every torchrun rank")
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="one proposer + one solver step at G16/TPS1; sealed but ineligible for resume/evaluation")
    args = parser.parse_args(argv)
    try:
        raw_cfg = None
        cfg = None
        manifest = None
        local_error = None
        try:
            raw_cfg = _load_config(args.config)
            expected_world = (args.world_size if args.world_size is not None
                              else int(raw_cfg["runtime"]["world_size"]))
            cfg, manifest = resolve_training_manifest(
                args.config.resolve(), args.base, args.alfworld_data, args.backend,
                expected_world, smoke=args.smoke)
            if args.smoke and args.resume is not None:
                raise ContractError("--smoke cannot resume a checkpoint")
        except Exception as exc:  # join distributed preflight and broadcast a uniform failure
            local_error = f"{type(exc).__name__}: {exc}"
            expected_world = args.world_size or int(os.environ.get("WORLD_SIZE", "1"))
        rank_device_environment = _agree_preflight(
            manifest, local_error, expected_world, args.run_id, args.resume)
        assert cfg is not None and manifest is not None
        manifest.setdefault("run_metadata", {})[
            "rank_device_environment"] = rank_device_environment
        run_dir = _assert_run_dir(cfg, args.run_id, args.resume)
        manifest, selected_resume, run_dir = _dist_preflight(manifest, run_dir, args.run_id, args.resume)
        config_path = run_dir / "resolved_config.yaml"
        manifest_path = run_dir / "training_manifest.json"
        if selected_resume is not None:
            # For resume, reuse the exact sealed config/manifest already validated above.
            config_path = run_dir / "resolved_config.yaml"
            manifest_path = run_dir / "training_manifest.json"
        cfg = manifest["resolved_config"]
        _install_resolved_prompt_environment(cfg)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        import ez_train
        ez_train.build_model_and_tokenizer = _strict_model_loader()
        argv_old = ["--config", str(config_path), "--run-id", args.run_id,
                    "--protocol-manifest", str(manifest_path)]
        if selected_resume is not None:
            argv_old.extend(["--resume", str(selected_resume)])
        if args.smoke:
            argv_old.append("--protocol-smoke")
        return ez_train.main(argv_old)
    except (ContractError, ValueError, OSError) as exc:
        print(f"[ez-train-v2] preflight failed: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
