"""Resolve a local training run contract, then start Explorer-Zero training."""
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

SELFPLAY_ROOT = Path(__file__).resolve().parent
SRC_DIR = SELFPLAY_ROOT / "src"
REPO_ROOT = SELFPLAY_ROOT.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from checkpoint import (  # noqa: E402
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
    "selfplay/train.py",
    "selfplay/config.yaml",
    "selfplay/alfworld.yaml",
    "selfplay/scripts/run_train.sh",
    "selfplay/src/checkpoint.py",
    "selfplay/src/training_contract.py",
    "selfplay/src/training_runtime.py",
    "selfplay/src/training_snapshots.py",
    "selfplay/evaluate_run.py",
    "selfplay/scripts/run_experiment.sh",
    "selfplay/src/ez_train.py",
    "selfplay/src/grpo.py",
    "selfplay/src/grpo_norm.py",
    "selfplay/src/ez_dist.py",
    "selfplay/src/ez_proposer.py",
    "selfplay/src/ez_prompts.py",
    "selfplay/src/ez_solver.py",
    "selfplay/src/ez_state.py",
    "selfplay/src/ez_verify.py",
    "selfplay/src/protocol.py",
    "selfplay/src/rollout.py",
    "selfplay/src/reward.py",
    "selfplay/src/alfworld_env.py",
    "selfplay/src/alfworld_compat.py",
    "selfplay/src/ez_vllm_policy.py",
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
             "numpy", "PyYAML")
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
    for name, wanted in (("ALFWORLD_DATA", str(data_root)),):
        current = os.environ.get(name)
        if current and Path(current).expanduser().resolve() != Path(wanted).resolve():
            raise ContractError(f"{name} conflicts with explicit CLI path")
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


def _validate_solver_sampling(solver_cfg: dict[str, Any]) -> None:
    expected = {
        "max_new_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    }
    if any(solver_cfg.get(key) != value for key, value in expected.items()):
        raise ContractError("solver sampling must match the shared SkillRise generation settings")


def resolve_training_manifest(config_path: Path, base_path: Path, data_root: Path,
                              backend: str, world_size: int, *, baseline_contract_path: Path,
                              smoke: bool = False
                              ) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = _load_config(config_path)
    if cfg.get("protocol_version") != 3 or cfg.get("protocol_name") != "explorer_zero_skillrise":
        raise ContractError("training accepts only the shared SkillRise protocol config")
    if backend not in {"hf", "vllm"}:
        raise ContractError("--backend must be explicitly hf or vllm")
    base = Path(base_path).expanduser().resolve()
    data = Path(data_root).expanduser().resolve()
    if not base.is_dir() or not (base / "config.json").is_file():
        raise ContractError("--base must be an existing local model directory with config.json")
    if not data.is_dir():
        raise ContractError("--alfworld-data must be an existing local data directory")
    if isinstance(world_size, bool) or world_size < 1:
        raise ContractError("world size must be a positive integer")
    if not smoke and world_size != 8:
        raise ContractError("formal Explorer-Zero training requires the paper's 8-GPU world")
    if not smoke and world_size != int(cfg.get("runtime", {}).get("world_size", -1)):
        raise ContractError("formal world size differs from the frozen training config")
    if int(cfg.get("runtime", {}).get("world_size", -1)) != 8:
        raise ContractError("frozen training config must declare the paper's 8-GPU world")
    if 16 % world_size:
        raise ContractError("world size must evenly shard the global GRPO group of 16")
    _check_environment(cfg, base, data, backend, world_size)
    expected_train = {
        "outer_loops": 4, "proposer_steps": 25, "solver_steps": 25,
        "group_size": 16, "tasks_per_step": 1, "solve_rate_rollouts": 8,
        "lr": 1e-4, "clip_eps": 0.3, "kl_beta": 0.001,
        "w_sub": 0.3, "w_div": 0.3,
    }
    for key, wanted in expected_train.items():
        if cfg.get("train", {}).get(key) != wanted:
            raise ContractError(f"paper-aligned training requires train.{key}={wanted!r}")
    if cfg.get("model", {}).get("lora_r") != 32 or cfg.get("model", {}).get("lora_alpha") != 32:
        raise ContractError("paper-aligned training requires LoRA rank=32 and alpha=32")
    if cfg.get("solver", {}).get("max_turns") != 30 or cfg.get("proposer", {}).get("max_turns") != 30:
        raise ContractError("ALFWorld Table 3 requires 30 proposer and solver turns")
    if cfg.get("proposer", {}).get("max_context_tokens") != 32768:
        raise ContractError("frozen config proposer context must be the legacy 32768-token value")
    if int(cfg.get("runtime", {}).get("vllm", {}).get("max_model_len", -1)) != 32768:
        raise ContractError("frozen config vLLM max_model_len must be 32768 before training override")
    cfg.setdefault("model", {})["path"] = str(base)
    cfg.setdefault("env", {})["data_root"] = str(data)
    cfg["runtime"]["backend"] = backend
    cfg["runtime"]["world_size"] = int(world_size)
    # Paper Table 3 uses 40,960 context. Keep the shared ReAct solver context
    # at 32,768 and widen only the training proposer / generation engine.
    cfg["proposer"]["max_context_tokens"] = 40960
    cfg["runtime"].setdefault("vllm", {})["max_model_len"] = 40960
    if smoke:
        cfg["train"]["outer_loops"] = 1
        cfg["train"]["proposer_steps"] = 1
        cfg["train"]["solver_steps"] = 1
    if cfg["env"].get("split") != "train":
        raise ContractError("training pool must be the explicit train split")
    solver_cfg = cfg["solver"]
    if solver_cfg.get("style") != "react" or solver_cfg.get("menu_enabled") is not True:
        raise ContractError("solver must use the shared SkillRise React prompt and legal-action menu")
    if solver_cfg.get("stop_mode") != "action_tag" or solver_cfg.get("max_context_tokens") != 32768:
        raise ContractError("solver must use action_tag stopping and the shared 32768-token context")
    if solver_cfg.get("keep_recent_turns") != 15 or solver_cfg.get("max_turns") != 30:
        raise ContractError("solver must use the shared 15-turn history and 30-turn budget")
    _validate_solver_sampling(solver_cfg)
    if cfg["proposer"].get("max_context_tokens") != 40960:
        raise ContractError("training proposer must use the paper's 40960-token context")
    if cfg["proposer"].get("max_new_tokens", 0) < 1 or cfg["solver"].get("max_new_tokens", 0) < 1:
        raise ContractError("generation limits must be explicit positive integers")
    if cfg["train"].get("group_size") != 16 or cfg["train"].get("tasks_per_step") != 1 \
            or cfg["train"].get("solve_rate_rollouts") != 8:
        raise ContractError("shared setting requires global G=16, TPS=1, GP=8")
    vllm_cfg = cfg["runtime"].get("vllm", {})
    if backend == "vllm":
        if int(vllm_cfg.get("max_model_len", -1)) != max(
                int(cfg["proposer"]["max_context_tokens"]),
                int(cfg["solver"]["max_context_tokens"])):
            raise ContractError("vLLM max_model_len must cover the 40960 proposer context")
        if int(vllm_cfg.get("max_lora_rank", -1)) < int(cfg["model"]["lora_r"]):
            raise ContractError("vLLM max_lora_rank must cover the configured adapter rank")
        if isinstance(vllm_cfg.get("seed"), bool) or not isinstance(vllm_cfg.get("seed"), int) or vllm_cfg["seed"] < 0:
            raise ContractError("vLLM engine seed must be an explicit non-negative integer")
    env_config = _resolve_path(cfg["env"]["config_path"], SELFPLAY_ROOT)
    if not env_config.is_file():
        raise ContractError("configured ALFWorld environment YAML is missing")
    compat = _resolve_compat()
    pool = _pool_manifest(data, env_config, str(cfg["env"]["split"]))
    base_files = base_file_inventory(base)
    try:
        from training_contract import validate_baseline_contract
        baseline_binding = validate_baseline_contract(
            baseline_contract_path, config_path=config_path.resolve(),
            base_inventory=base_files, data_root=data,
            solver_config=cfg["solver"])
    except (ValueError, OSError) as exc:
        raise ContractError(f"baseline contract validation failed: {exc}") from None
    tokenizer_files = {name: record for name, record in base_files.items()
                       if Path(name).name in TOKENIZER_BASENAMES
                       or Path(name).name in {"config.json", "generation_config.json"}}
    if not any(Path(name).name in TOKENIZER_BASENAMES for name in tokenizer_files):
        raise ContractError("local base has no recognizable tokenizer files")
    model_config = _json_load(base / "config.json")
    max_positions = model_config.get("max_position_embeddings")
    rope_scaling = model_config.get("rope_scaling")
    if (isinstance(max_positions, int) and not isinstance(max_positions, bool)
            and max_positions < 40960 and not (isinstance(rope_scaling, dict) and rope_scaling)):
        raise ContractError(
            "base model declares fewer than 40960 positions without an explicit rope_scaling config")
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
        "protocol_name": "explorer_zero_skillrise",
        "setting": "skillrise_react",
        "comparison_scope": "local Explorer-Zero method under the shared solver and evaluation protocol",
        "checkpoint_eligibility": "smoke_only" if smoke else "formal_training",
        "resolved_config": config_resolved,
        "baseline_binding": baseline_binding,
        "effective_overrides": {
            "proposer.max_context_tokens": {
                "value": 40960, "source": "Explorer-Zero paper Table 3; proposer training only"},
            "runtime.vllm.max_model_len": {
                "value": 40960, "source": "covers paper proposer context; solver remains 32768"},
            "runtime.world_size": {
                "value": int(world_size), "source": "formal=8 paper GPUs; smoke may use a legal shard count"},
        },
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
                "base_identity_scope": "local checkpoint bytes; official model origin is unverified",
                "max_position_embeddings": max_positions,
                "rope_scaling": rope_scaling,
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
        "effective_prompts": {
            "solver_style": cfg["solver"]["style"],
            "solver_menu": bool(cfg["solver"]["menu_enabled"]),
            "solver_stop_mode": cfg["solver"]["stop_mode"],
            "proposer_format": "internal_json_finish_sub_finish",
            "proposer_menu": bool(cfg["proposer"]["give_admissible"]),
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
        raise ContractError(f"WORLD_SIZE={world} does not match resolved contract")
    if world > 1 and not dist.is_initialized():
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
            pg_backend = "nccl"
        else:
            # CPU/Gloo is useful for preflight and checkpoint-contract tests; the
            # actual EZ model loader still requires the configured model device.
            pg_backend = "gloo"
        dist.init_process_group(backend=pg_backend, rank=rank, world_size=world,
                                timeout=_dt.timedelta(minutes=30))
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
                    raise ContractError("resume checkpoint contract differs from current config")
                from training_snapshots import ensure_resume_snapshots
                ensure_resume_snapshots(selected.parent, selected, ckpt_manifest, checkpoint_info=info)
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
        raise ContractError(result.get("error", "rank zero preflight failed") if result else "preflight failed")
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
                    "run_id": run_id, "protocol": "explorer_zero_skillrise"}) + b"\n")
            write_status[0] = {"ok": True}
        except Exception as exc:  # broadcast before any rank reaches a barrier
            write_status[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if world > 1:
        dist.broadcast_object_list(write_status, src=0)
    if not write_status[0] or not write_status[0].get("ok"):
        raise ContractError(write_status[0].get("error", "rank zero could not write run artifacts"))
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
        dist.init_process_group(backend=pg_backend, rank=rank, world_size=env_world,
                                timeout=_dt.timedelta(minutes=30))
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
        raise ContractError("distributed preflight rejected: " + "; ".join(errors))
    if len(signatures) != 1 or None in signatures:
        raise ContractError("rank-local training contracts differ or are unavailable")
    if len({(item.get("run_id"), item.get("resume")) for item in gathered if item}) != 1:
        raise ContractError("ranks supplied different run ids or resume checkpoints")
    return [dict(item.get("device_environment") or {}) for item in gathered]


def _install_runtime_environment(cfg: dict[str, Any]) -> None:
    os.environ["ALFWORLD_DATA"] = str(cfg["env"]["data_root"])
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SELFPLAY_ROOT / "config.yaml")
    parser.add_argument("--base", type=Path, required=True, help="existing local model directory; no Hub fallback")
    parser.add_argument("--alfworld-data", type=Path, required=True, help="existing local ALFWorld data root")
    parser.add_argument("--baseline-contract", type=Path, required=True,
                        help="existing base-only evaluation contract for the frozen shared protocol")
    parser.add_argument("--backend", choices=("hf", "vllm"), default=None)
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
            backend = args.backend or str(raw_cfg["runtime"]["backend"])
            expected_world = (args.world_size if args.world_size is not None
                              else int(raw_cfg["runtime"]["world_size"]))
            cfg, manifest = resolve_training_manifest(
                args.config.resolve(), args.base, args.alfworld_data, backend,
                expected_world, baseline_contract_path=args.baseline_contract.resolve(),
                smoke=args.smoke)
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
        _install_runtime_environment(cfg)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        import ez_train
        argv_old = ["--config", str(config_path), "--run-id", args.run_id,
                    "--protocol-manifest", str(manifest_path)]
        if selected_resume is not None:
            argv_old.extend(["--resume", str(selected_resume)])
        if args.smoke:
            argv_old.append("--protocol-smoke")
        return ez_train.main(argv_old)
    except (ContractError, ValueError, OSError) as exc:
        print(f"[train] preflight failed: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
