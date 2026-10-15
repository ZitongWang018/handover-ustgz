"""Immutable, sealed snapshots of completed formal training outers."""
from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

from checkpoint import (COMPLETE_NAME, INVENTORY_NAME, ContractError,
                        canonical_json_bytes, sha256_file, validate_checkpoint_v3)

INDEX_NAME = "index.json"
SNAPSHOT_SCHEMA = 1


def _safe_child(parent: Path, child: Path) -> Path:
    root = parent.resolve()
    if child.is_symlink():
        raise ContractError(f"snapshot path cannot be a symlink: {child}")
    candidate = child.resolve(strict=False)
    lexical = Path(os.path.abspath(child))
    if candidate != lexical:
        raise ContractError(f"snapshot path resolves through a symlink or directory alias: {child}")
    if candidate == root or root not in candidate.parents:
        raise ContractError(f"snapshot path escapes its parent: {child}")
    return candidate


def _check_layout(run_dir: Path) -> tuple[Path, Path]:
    run = Path(run_dir).resolve()
    checkpoints = run / "checkpoints"
    if checkpoints.is_symlink():
        raise ContractError("checkpoints directory cannot be a symlink")
    checkpoints.mkdir(parents=True, exist_ok=True)
    if checkpoints.resolve() != _safe_child(run, checkpoints):
        raise ContractError("checkpoints directory escapes run directory")
    return run, checkpoints


def _load_index(path: Path, run_id: str, contract_sha256: str, n_outer: int) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": SNAPSHOT_SCHEMA, "run_id": run_id,
                "contract_sha256": contract_sha256, "snapshots": []}
    if path.is_symlink() or not path.is_file():
        raise ContractError("snapshot index must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"snapshot index is unreadable: {type(exc).__name__}") from None
    if (value.get("schema_version") != SNAPSHOT_SCHEMA or value.get("run_id") != run_id
            or value.get("contract_sha256") != contract_sha256
            or not isinstance(value.get("snapshots"), list)):
        raise ContractError("snapshot index identity/schema mismatch")
    records = value["snapshots"]
    if len(records) > n_outer:
        raise ContractError("snapshot index contains too many outer records")
    for i, row in enumerate(records, 1):
        expected_path = f"checkpoints/outer_{i:02d}"
        if (not isinstance(row, dict) or row.get("outer") != i
                or row.get("path") != expected_path
                or row.get("contract_sha256") != contract_sha256
                or row.get("next_position") != {"outer": i, "phase": "proposer", "step": 0}
                or not isinstance(row.get("complete_marker_sha256"), str)
                or not isinstance(row.get("inventory_sha256"), str)):
            raise ContractError(f"snapshot index is not a valid contiguous prefix at outer {i}")
    return value


def _atomic_index(path: Path, value: dict[str, Any]) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    temp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temp.open("xb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        if path.exists():
            # Replace only after the caller has validated and merged the existing index.
            os.replace(temp, path)
        else:
            os.rename(temp, path)
    except Exception:
        # Index temporary files are retained as crash diagnostics, like partial snapshots.
        raise


def _row_for(path: Path, run_dir: Path, outer: int, expected_contract: Mapping[str, Any]) -> dict[str, Any]:
    info = validate_checkpoint_v3(path, expected_contract=expected_contract, purpose="resume")
    state = info["state"]
    position = state.get("next_position")
    if position != {"outer": outer, "phase": "proposer", "step": 0}:
        raise ContractError(f"checkpoint is not the sealed completion boundary for outer {outer}")
    if state.get("run_id") != run_dir.name:
        manifest_name = str(info["manifest"].get("resolved_config", {}).get("run", {}).get("name", ""))
        expected_id = run_dir.name.removeprefix(manifest_name + "-") if manifest_name else run_dir.name
        if state.get("run_id") != expected_id:
            raise ContractError("checkpoint run_id does not match snapshot run directory")
    marker = path / COMPLETE_NAME
    inventory = path / INVENTORY_NAME
    return {
        "outer": outer,
        "path": f"checkpoints/outer_{outer:02d}",
        "contract_sha256": info["manifest"]["contract_sha256"],
        "complete_marker_sha256": sha256_file(marker),
        "inventory_sha256": sha256_file(inventory),
        "next_position": position,
    }


def _validate_archive(path: Path, row: Mapping[str, Any], run_dir: Path,
                      expected_contract: Mapping[str, Any]) -> None:
    if path.is_symlink():
        raise ContractError(f"snapshot archive cannot be a symlink: {path}")
    if path.resolve(strict=False) != Path(os.path.abspath(path)):
        raise ContractError(f"snapshot archive resolves through a symlink or directory alias: {path}")
    info = validate_checkpoint_v3(path, expected_contract=expected_contract, purpose="evaluation")
    if info["state"].get("next_position") != row["next_position"]:
        raise ContractError(f"archive cursor mismatch at outer {row['outer']}")
    if (sha256_file(path / COMPLETE_NAME) != row["complete_marker_sha256"]
            or sha256_file(path / INVENTORY_NAME) != row["inventory_sha256"]):
        raise ContractError(f"archive seal hash mismatch at outer {row['outer']}")


def publish_outer_snapshot(run_dir: Path, outer: int, expected_contract: Mapping[str, Any],
                           checkpoint_path: Path | None = None) -> dict[str, Any]:
    """Publish one completed outer by copy, retaining partials and refusing overwrites."""
    run_dir, checkpoints = _check_layout(run_dir)
    if isinstance(outer, bool) or not isinstance(outer, int) or outer < 1:
        raise ContractError("outer snapshot number must be a positive integer")
    signature = str(expected_contract.get("contract_sha256", ""))
    if not signature:
        raise ContractError("snapshot contract has no contract_sha256")
    train = expected_contract.get("resolved_config", {}).get("train", {})
    n_outer = train.get("outer_loops")
    if isinstance(n_outer, bool) or not isinstance(n_outer, int) or outer > n_outer:
        raise ContractError("outer snapshot number is outside the training contract")
    manifest_run = str(expected_contract.get("resolved_config", {}).get("run", {}).get("name", ""))
    run_id = run_dir.name.removeprefix(manifest_run + "-") if manifest_run else run_dir.name
    index_path = checkpoints / INDEX_NAME
    index = _load_index(index_path, run_id, signature, n_outer)
    for prior in index["snapshots"]:
        prior_path = run_dir / prior["path"]
        if prior_path.is_symlink():
            raise ContractError(f"indexed snapshot is a symlink: {prior_path}")
        _validate_archive(prior_path, prior, run_dir, expected_contract)
    source = Path(checkpoint_path) if checkpoint_path is not None else run_dir / "ckpt"
    if source.is_symlink():
        raise ContractError("mutable checkpoint cannot be a symlink")
    source = source.resolve()
    if source.parent != run_dir:
        raise ContractError("mutable checkpoint must be directly inside the run directory")
    row = _row_for(source, run_dir, outer, expected_contract)
    dest = _safe_child(checkpoints, checkpoints / f"outer_{outer:02d}")
    while len(index["snapshots"]) < outer - 1:
        raise ContractError(f"cannot publish outer {outer}; prior snapshot prefix is incomplete")
    if len(index["snapshots"]) >= outer:
        existing = index["snapshots"][outer - 1]
        if existing != row:
            raise ContractError(f"outer {outer} already has a different index record")
        _validate_archive(dest, existing, run_dir, expected_contract)
        return existing
    if dest.exists() or dest.is_symlink():
        # A rename may have succeeded before a crash prevented the index update.
        if not dest.is_symlink():
            try:
                _validate_archive(dest, row, run_dir, expected_contract)
            except (ContractError, OSError):
                raise ContractError(f"refusing to replace existing snapshot destination: {dest}") from None
            if dest != _safe_child(checkpoints, dest):
                raise ContractError("archive destination escaped checkpoint directory")
            index["snapshots"].append(row)
            _atomic_index(index_path, index)
            return row
        raise ContractError(f"refusing to replace existing snapshot destination: {dest}")
    temp = checkpoints / f".outer_{outer:02d}.tmp-{uuid.uuid4().hex}"
    if temp.exists() or temp.is_symlink():
        raise ContractError(f"unique snapshot temporary path unexpectedly exists: {temp}")
    try:
        shutil.copytree(source, temp, copy_function=shutil.copy2)
        copied = _row_for(temp, run_dir, outer, expected_contract)
        if copied != row:
            raise ContractError("copied snapshot differs from sealed mutable checkpoint")
        _validate_archive(temp, row, run_dir, expected_contract)
        # Atomic publication: no existing archive is removed or overwritten.
        os.rename(temp, dest)
    except Exception as exc:
        raise ContractError(f"snapshot publication failed; diagnostic partial retained at {temp}: {type(exc).__name__}: {exc}") from None
    _validate_archive(dest, row, run_dir, expected_contract)
    index["snapshots"].append(row)
    _atomic_index(index_path, index)
    return row


def ensure_resume_snapshots(run_dir: Path, checkpoint_path: Path,
                            expected_contract: Mapping[str, Any],
                            checkpoint_info: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Validate completed prefix and only backfill the exact current P0 boundary."""
    run_dir, checkpoints = _check_layout(run_dir)
    info = checkpoint_info or validate_checkpoint_v3(
        checkpoint_path, expected_contract=expected_contract, purpose="resume")
    state = info["state"]
    position = state["next_position"]
    cursor = int(position["outer"])
    train = expected_contract.get("resolved_config", {}).get("train", {})
    n_outer = int(train.get("outer_loops", 0))
    signature = str(expected_contract.get("contract_sha256", ""))
    manifest_run = str(expected_contract.get("resolved_config", {}).get("run", {}).get("name", ""))
    run_id = run_dir.name.removeprefix(manifest_run + "-") if manifest_run else run_dir.name
    index_path = checkpoints / INDEX_NAME
    index = _load_index(index_path, run_id, signature, n_outer)
    for row in index["snapshots"]:
        archive = run_dir / row["path"]
        if archive.is_symlink():
            raise ContractError(f"indexed snapshot is a symlink: {archive}")
        archive = _safe_child(run_dir, archive)
        _validate_archive(archive, row, run_dir, expected_contract)
    boundary = position == {"outer": cursor, "phase": "proposer", "step": 0}
    required = cursor - 1 if boundary and cursor > 0 else cursor
    if len(index["snapshots"]) < required:
        raise ContractError(f"resume requires completed snapshots 1..{required}; found {len(index['snapshots'])}")
    if boundary and cursor > 0 and len(index["snapshots"]) == cursor - 1:
        publish_outer_snapshot(run_dir, cursor, expected_contract, checkpoint_path=checkpoint_path)
        index = _load_index(index_path, run_id, signature, n_outer)
    return index["snapshots"]


def publish_outer_snapshot_distributed(run_dir: Path, outer: int,
                                       expected_contract: Mapping[str, Any], rank: int,
                                       world: int, checkpoint_path: Path | None = None) -> dict[str, Any]:
    """Run publication on rank zero and make every rank observe the same result."""
    packet: list[Any] = [None]
    if rank == 0:
        try:
            packet[0] = {"ok": True, "row": publish_outer_snapshot(
                run_dir, outer, expected_contract, checkpoint_path=checkpoint_path)}
        except Exception as exc:
            packet[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if world > 1:
        import torch.distributed as dist
        dist.broadcast_object_list(packet, src=0)
    result = packet[0]
    if not result or not result.get("ok"):
        raise ContractError(result.get("error", "rank-zero snapshot publication failed") if result else
                            "snapshot publication failed")
    return result["row"]
