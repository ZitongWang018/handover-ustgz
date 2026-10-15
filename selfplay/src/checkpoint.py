"""Version-3 Explorer-Zero training contract and checkpoint integrity helpers.

This module intentionally uses only the Python standard library. Evaluation
tools can validate checkpoint bytes without importing torch or training code.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 3
INVENTORY_NAME = "checkpoint_inventory.json"
COMPLETE_NAME = "COMPLETE.json"


class ContractError(ValueError):
    """A resolved protocol or checkpoint does not satisfy the v3 contract."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def base_file_inventory(model_dir: Path) -> dict[str, dict[str, Any]]:
    """Return a portable identity map for every regular file in a local base dir."""
    root = Path(model_dir).resolve()
    if not root.is_dir():
        raise ContractError("base model must be an existing local directory")
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() and path.is_dir():
            raise ContractError(f"base model contains a directory symlink: {path.relative_to(root).as_posix()}")
        if path.is_symlink() and not path.is_file():
            raise ContractError(f"base model contains a broken or non-file symlink: {path.relative_to(root).as_posix()}")
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            try:
                target = path.resolve(strict=True)
            except OSError:
                raise ContractError(f"base model contains a broken file symlink: {rel}") from None
            if not target.is_file():
                raise ContractError(f"base model file does not resolve to a regular file: {rel}")
            records[rel] = {"size": target.stat().st_size, "sha256": sha256_file(target),
                            "symlink": bool(path.is_symlink())}
    if "config.json" not in records:
        raise ContractError("base model directory is missing config.json")
    if not any(name.endswith((".safetensors", ".bin", ".pt")) for name in records):
        raise ContractError("base model directory has no recognized weight content")
    for rel in records:
        if not rel.endswith((".safetensors.index.json", "pytorch_model.bin.index.json")):
            continue
        try:
            index = json.loads((root / rel).read_text(encoding="utf-8"))
            weight_map = index["weight_map"]
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("empty weight_map")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            raise ContractError(f"base model weight index is invalid: {rel}") from None
        for filename in set(weight_map.values()):
            if not isinstance(filename, str) or Path(filename).is_absolute() or \
                    ".." in Path(filename.replace("\\", "/")).parts or \
                    not filename.endswith((".safetensors", ".bin")):
                raise ContractError(f"base model weight index contains unsafe shard path: {rel}")
            if filename.replace("\\", "/") not in records:
                raise ContractError(f"base model weight index references missing shard: {filename}")
    return records


def base_inventory_sha256(model_dir: Path) -> str:
    return sha256_bytes(canonical_json_bytes(base_file_inventory(model_dir)))


def environment_file_identity(config: Mapping[str, Any], config_path: Path,
                              task_files: list[Path] | tuple[Path, ...]) -> dict[str, Any]:
    """Hash the configured ALFWorld logic files and each task's trajectory sibling.

    `config` is the expanded mapping consumed by ALFWorld. Missing trajectory files
    are represented explicitly because some ALFWorld releases omit them.
    """
    def record(path_value: str | Path, label: str) -> dict[str, Any]:
        path = Path(os.path.expandvars(os.path.expanduser(str(path_value)))).resolve()
        if not path.is_file():
            raise ContractError(f"configured {label} file is missing: {path.name}")
        return {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}

    logic_records = {}
    logic = config.get("logic", {})
    if isinstance(logic, Mapping):
        for name in ("domain", "grammar"):
            value = logic.get(name)
            if not isinstance(value, str) or not value:
                raise ContractError(f"ALFWorld logic.{name} path is not explicit")
            logic_records[name] = record(value, f"logic.{name}")
    task_records = []
    for task in task_files:
        task_path = Path(task).resolve()
        trajectory_path = task_path.parent / "traj_data.json"
        task_records.append({
            "task_path": str(task_path),
            "trajectory": (record(trajectory_path, "task trajectory")
                           if trajectory_path.is_file() else None),
        })
    config_file = Path(config_path).resolve()
    if not config_file.is_file():
        raise ContractError("ALFWorld config file is missing")
    return {
        "config": {"path": str(config_file), "size": config_file.stat().st_size,
                   "sha256": sha256_file(config_file)},
        "logic_files": logic_records,
        "task_trajectories": task_records,
    }


def contract_signature(manifest: Mapping[str, Any]) -> str:
    clean = dict(manifest)
    clean.pop("contract_sha256", None)
    clean.pop("run_metadata", None)
    return sha256_bytes(canonical_json_bytes(clean))


def _atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         indent=2, allow_nan=False) + "\n"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as fh:
            temp_path = Path(fh.name)
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_path, path)
    except OSError as exc:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise ContractError(f"cannot atomically write {path.name}: {type(exc).__name__}") from None


def checkpoint_inventory(root: Path) -> dict[str, Any]:
    """Hash every regular checkpoint file except the inventory/seal themselves."""
    root = Path(root).resolve()
    records: dict[str, dict[str, Any]] = {}
    excluded = {INVENTORY_NAME, COMPLETE_NAME}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ContractError(f"checkpoint contains a symlink: {path.relative_to(root).as_posix()}")
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel in excluded:
            continue
        records[rel] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    return {"schema_version": SCHEMA_VERSION, "files": records}


def seal_checkpoint(root: Path, contract_sha256: str) -> dict[str, Any]:
    """Write inventory and completion marker after all rank-local files exist."""
    root = Path(root).resolve()
    inventory = checkpoint_inventory(root)
    inventory_path = root / INVENTORY_NAME
    _atomic_json(inventory_path, inventory)
    marker = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "contract_sha256": str(contract_sha256),
        "inventory_sha256": sha256_file(inventory_path),
    }
    _atomic_json(root / COMPLETE_NAME, marker)
    return inventory


def validate_sealed_integrity(path: Path) -> dict[str, Any]:
    """Check v3 bytes without granting resume/evaluation eligibility (smoke-safe)."""
    root = Path(path).resolve()
    required = ("train_state.json", "training_manifest.json", INVENTORY_NAME, COMPLETE_NAME)
    if not root.is_dir() or any(not (root / rel).is_file() for rel in required):
        raise ContractError("checkpoint is not a fully sealed v3 directory")
    try:
        state = json.loads((root / "train_state.json").read_text(encoding="utf-8"))
        manifest = json.loads((root / "training_manifest.json").read_text(encoding="utf-8"))
        inventory = json.loads((root / INVENTORY_NAME).read_text(encoding="utf-8"))
        marker = json.loads((root / COMPLETE_NAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"checkpoint metadata is unreadable: {type(exc).__name__}") from None
    signature = manifest.get("contract_sha256")
    if manifest.get("schema_version") != SCHEMA_VERSION or \
            signature != contract_signature(manifest) or state.get("contract_sha256") != signature:
        raise ContractError("sealed checkpoint manifest/state contract is invalid")
    if marker.get("schema_version") != SCHEMA_VERSION or marker.get("status") != "complete" or \
            marker.get("contract_sha256") != signature or marker.get("inventory_sha256") != sha256_file(root / INVENTORY_NAME):
        raise ContractError("sealed checkpoint completion marker is invalid")
    actual = checkpoint_inventory(root)
    if inventory.get("schema_version") != SCHEMA_VERSION or actual != inventory:
        raise ContractError("sealed checkpoint inventory does not match checkpoint bytes")
    return {"state": state, "manifest": manifest, "inventory": inventory,
            "solver_path": str((root / "solver").resolve())}


def _validate_position(state: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    pos = state.get("next_position")
    if not isinstance(pos, dict):
        raise ContractError("v3 train_state is missing next_position")
    outer = pos.get("outer")
    phase = pos.get("phase")
    step = pos.get("step")
    aliases = (state.get("next_outer"), state.get("next_phase"), state.get("next_step"))
    if aliases != (outer, phase, step):
        raise ContractError("next_position disagrees with next_outer/next_phase/next_step aliases")
    train = manifest.get("resolved_config", {}).get("train", {})
    n_outer = train.get("outer_loops")
    limits = {"proposer": train.get("proposer_steps"), "solver": train.get("solver_steps")}
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (outer, step, n_outer)):
        raise ContractError("v3 next_position values must be integers")
    if phase not in limits or isinstance(limits.get(phase), bool) or not isinstance(limits.get(phase), int):
        raise ContractError("v3 next_position phase/step budget is invalid")
    if not 0 <= outer <= n_outer or step < 0:
        raise ContractError("v3 next_position is outside configured outer range")
    if outer == n_outer:
        if phase != "proposer" or step != 0:
            raise ContractError("completed training must resume at outer=N, proposer:0")
    elif step >= limits[phase]:
        raise ContractError("v3 next_position must point to an uncompleted step")


def validate_checkpoint_v3(
    path: Path,
    expected_contract: Mapping[str, Any] | str | None = None,
    purpose: str | None = None,
) -> dict[str, Any]:
    """Validate a sealed v3 checkpoint and return state/manifest/inventory.

    `purpose` must be explicitly `resume` or `evaluation`. Evaluation verifies
    solver adapter bytes and contract identity but never deserializes optimizer
    state. Resume additionally requires every rank RNG and optimizer/pool files.
    """
    if purpose not in {"resume", "evaluation"}:
        raise ValueError("purpose must be explicitly 'resume' or 'evaluation'")
    root = Path(path).resolve()
    if not root.is_dir():
        raise ContractError("checkpoint directory does not exist")
    required = ["train_state.json", "training_manifest.json", INVENTORY_NAME, COMPLETE_NAME,
                "solver/adapter_config.json"]
    if purpose == "resume":
        required.extend(["optimizer.pt", "task_pool.jsonl", "proposer/adapter_config.json"])
    for rel in required:
        if not (root / rel).is_file():
            raise ContractError(f"checkpoint is missing required {rel}")
    try:
        state = json.loads((root / "train_state.json").read_text(encoding="utf-8"))
        manifest = json.loads((root / "training_manifest.json").read_text(encoding="utf-8"))
        inventory = json.loads((root / INVENTORY_NAME).read_text(encoding="utf-8"))
        marker = json.loads((root / COMPLETE_NAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"checkpoint metadata is unreadable: {type(exc).__name__}") from None
    if state.get("checkpoint_schema_version") != SCHEMA_VERSION:
        raise ContractError("only checkpoint schema v3 is accepted")
    if manifest.get("schema_version") != SCHEMA_VERSION or \
            manifest.get("protocol_name") != "explorer_zero_skillrise":
        raise ContractError("training manifest is not a v3 Explorer-Zero contract")
    if purpose in {"resume", "evaluation"} and state.get("dry_run") is not False:
        raise ContractError("dry-run or non-formal checkpoint cannot be resumed/evaluated")
    signature = manifest.get("contract_sha256")
    if not isinstance(signature, str) or signature != contract_signature(manifest):
        raise ContractError("training manifest contract hash is invalid")
    if state.get("contract_sha256") != signature or marker.get("contract_sha256") != signature:
        raise ContractError("train_state/complete marker contract does not match manifest")
    if marker.get("schema_version") != SCHEMA_VERSION or marker.get("status") != "complete":
        raise ContractError("checkpoint lacks a valid completion marker")
    if inventory.get("schema_version") != SCHEMA_VERSION or not isinstance(inventory.get("files"), dict):
        raise ContractError("checkpoint inventory schema is invalid")
    inv_path = root / INVENTORY_NAME
    if marker.get("inventory_sha256") != sha256_file(inv_path):
        raise ContractError("checkpoint inventory checksum does not match completion marker")
    actual = checkpoint_inventory(root)
    expected_files = inventory["files"]
    if set(actual["files"]) != set(expected_files):
        raise ContractError("checkpoint file inventory is incomplete or contains unsealed files")
    for rel, record in expected_files.items():
        if actual["files"].get(rel) != record:
            raise ContractError(f"checkpoint file hash mismatch: {rel}")
    if expected_contract is not None:
        if purpose == "evaluation" and isinstance(expected_contract, Mapping) \
                and "base_inventory_sha256" in expected_contract:
            got_base = manifest.get("identity", {}).get("base_inventory_sha256")
            if got_base != expected_contract["base_inventory_sha256"]:
                raise ContractError("checkpoint base model identity mismatch")
        else:
            expected_signature = (expected_contract if isinstance(expected_contract, str)
                                  else expected_contract.get("contract_sha256")
                                  or contract_signature(expected_contract))
            if signature != expected_signature:
                raise ContractError("checkpoint resolved training contract mismatch")
    _validate_position(state, manifest)
    if purpose == "resume":
        for adapter in ("proposer", "solver"):
            if not any(rel.startswith(f"{adapter}/") and rel.endswith((".safetensors", ".bin"))
                       for rel in inventory["files"]):
                raise ContractError(f"checkpoint {adapter} adapter has no weight file")
        world = manifest.get("resolved_config", {}).get("runtime", {}).get("world_size")
        if isinstance(world, bool) or not isinstance(world, int) or world < 1:
            raise ContractError("checkpoint resolved world size is invalid")
        if state.get("world_size") != world:
            raise ContractError("checkpoint world size disagrees with resolved contract")
        if not isinstance(manifest.get("contract_sha256"), str):
            raise ContractError("checkpoint manifest contract schema is invalid")
        rank_dir = root / "rank_state"
        ranks = [rank_dir / f"rank{rank}.pt" for rank in range(int(world))]
        if any(not item.is_file() for item in ranks):
            raise ContractError("checkpoint is missing one or more per-rank RNG files")
    if purpose == "evaluation":
        for rel in inventory["files"]:
            if rel.startswith("solver/") and rel.endswith((".safetensors", ".bin")):
                pass
        if not any(rel.startswith("solver/") and rel.endswith((".safetensors", ".bin"))
                   for rel in inventory["files"]):
            raise ContractError("checkpoint solver adapter has no weight file")
    return {"state": state, "manifest": manifest, "inventory": inventory,
            "solver_path": str((root / "solver").resolve())}
