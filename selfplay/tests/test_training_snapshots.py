"""Boundary, integrity, and recovery tests for formal outer snapshots."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from checkpoint import ContractError, contract_signature, seal_checkpoint, validate_checkpoint_v3
from training_snapshots import (INDEX_NAME, ensure_resume_snapshots,
                                publish_outer_snapshot)


def _manifest(tmp_path: Path) -> dict:
    value = {
        "schema_version": 3,
        "protocol_name": "explorer_zero_skillrise",
        "resolved_config": {
            "run": {"name": "snaprun", "output_dir": str(tmp_path)},
            "train": {"outer_loops": 4, "proposer_steps": 2, "solver_steps": 2},
            "runtime": {"world_size": 1},
        },
        "identity": {"base_inventory_sha256": "b" * 64},
    }
    value["contract_sha256"] = contract_signature(value)
    return value


def _checkpoint(run_dir: Path, manifest: dict, outer: int, *, phase="proposer", step=0,
                sealed=True) -> Path:
    path = run_dir / "ckpt"
    path.mkdir(parents=True, exist_ok=True)
    for adapter in ("proposer", "solver"):
        folder = path / adapter
        folder.mkdir(exist_ok=True)
        (folder / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        (folder / "adapter_model.safetensors").write_bytes(f"{adapter}-{outer}-{step}".encode())
    (path / "optimizer.pt").write_bytes(b"optimizer")
    (path / "task_pool.jsonl").write_text("", encoding="utf-8")
    (path / "training_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (path / "rank_state").mkdir(exist_ok=True)
    (path / "rank_state" / "rank0.pt").write_bytes(b"rng")
    state = {
        "checkpoint_schema_version": 3,
        "contract_sha256": manifest["contract_sha256"],
        "world_size": 1,
        "dry_run": False,
        "next_position": {"outer": outer, "phase": phase, "step": step},
        "next_outer": outer,
        "next_phase": phase,
        "next_step": step,
        "run_id": run_dir.name.removeprefix("snaprun-"),
    }
    (path / "train_state.json").write_text(json.dumps(state), encoding="utf-8")
    if sealed:
        seal_checkpoint(path, manifest["contract_sha256"])
    return path


def test_publish_four_boundaries_and_resume_prefix(tmp_path):
    manifest = _manifest(tmp_path)
    run = tmp_path / "snaprun-r1"
    run.mkdir()
    for outer in range(1, 5):
        cp = _checkpoint(run, manifest, outer)
        row = publish_outer_snapshot(run, outer, manifest)
        assert row["outer"] == outer
        archived = run / "checkpoints" / f"outer_{outer:02d}"
        validate_checkpoint_v3(archived, manifest, purpose="evaluation")
        # The archive is a byte copy, independent of later mutable checkpoint changes.
        before = (archived / "solver" / "adapter_model.safetensors").read_bytes()
        (cp / "solver" / "adapter_model.safetensors").write_bytes(b"later mutable bytes")
        assert (archived / "solver" / "adapter_model.safetensors").read_bytes() == before
        # Restore a matching sealed checkpoint for the next outer publication.
        _checkpoint(run, manifest, outer)
    index = json.loads((run / "checkpoints" / INDEX_NAME).read_text(encoding="utf-8"))
    assert [row["outer"] for row in index["snapshots"]] == [1, 2, 3, 4]
    assert [row["path"] for row in index["snapshots"]] == [
        f"checkpoints/outer_{i:02d}" for i in range(1, 5)]
    assert len({row["inventory_sha256"] for row in index["snapshots"]}) == 4
    snapshots = ensure_resume_snapshots(run, run / "ckpt", manifest)
    assert len(snapshots) == 4
    index["snapshots"][2]["path"] = "checkpoints/outer_04"
    (run / "checkpoints" / INDEX_NAME).write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ContractError, match="contiguous prefix"):
        ensure_resume_snapshots(run, run / "ckpt", manifest)


def test_publish_requires_sealed_exact_outer_boundary(tmp_path):
    manifest = _manifest(tmp_path)
    run = tmp_path / "snaprun-r2"
    run.mkdir()
    _checkpoint(run, manifest, 1, step=1)
    with pytest.raises(ContractError, match="completion boundary"):
        publish_outer_snapshot(run, 1, manifest)
    unsealed_run = tmp_path / "snaprun-unsealed"
    unsealed_run.mkdir()
    _checkpoint(unsealed_run, manifest, 1, sealed=False)
    with pytest.raises(ContractError, match="required"):
        publish_outer_snapshot(unsealed_run, 1, manifest)
    _checkpoint(run, manifest, 2)
    with pytest.raises(ContractError, match="boundary"):
        publish_outer_snapshot(run, 1, manifest)
    other_manifest = _manifest(tmp_path)
    other_manifest["identity"]["base_inventory_sha256"] = "c" * 64
    other_manifest["contract_sha256"] = contract_signature(other_manifest)
    _checkpoint(run, other_manifest, 1)
    with pytest.raises(ContractError, match="contract mismatch"):
        publish_outer_snapshot(run, 1, manifest)


def test_idempotent_publication_and_conflicting_destination_is_preserved(tmp_path):
    manifest = _manifest(tmp_path)
    run = tmp_path / "snaprun-r3"
    run.mkdir()
    _checkpoint(run, manifest, 1)
    first = publish_outer_snapshot(run, 1, manifest)
    assert publish_outer_snapshot(run, 1, manifest) == first

    other = tmp_path / "snaprun-r4"
    other.mkdir()
    _checkpoint(other, manifest, 1)
    foreign = other / "checkpoints"
    foreign.mkdir()
    dest = foreign / "outer_01"
    dest.mkdir()
    sentinel = dest / "keep.txt"
    sentinel.write_text("do not erase", encoding="utf-8")
    with pytest.raises(ContractError, match="existing snapshot destination"):
        publish_outer_snapshot(other, 1, manifest)
    assert sentinel.read_text(encoding="utf-8") == "do not erase"


def test_missing_or_tampered_indexed_prefix_is_rejected(tmp_path):
    manifest = _manifest(tmp_path)
    run = tmp_path / "snaprun-r7"
    run.mkdir()
    _checkpoint(run, manifest, 1)
    publish_outer_snapshot(run, 1, manifest)
    _checkpoint(run, manifest, 2)
    # A later publication must validate the complete indexed prefix first.
    archived = run / "checkpoints" / "outer_01"
    (archived / "solver" / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ContractError, match="inventory|hash"):
        publish_outer_snapshot(run, 2, manifest)


def test_destination_symlink_is_rejected(tmp_path):
    manifest = _manifest(tmp_path)
    other = tmp_path / "snaprun-r8"
    other.mkdir()
    _checkpoint(other, manifest, 1)
    checkpoints = other / "checkpoints"
    checkpoints.mkdir()
    target = checkpoints / "target"
    target.mkdir()
    try:
        (checkpoints / "outer_01").symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")
    with pytest.raises(ContractError, match="symlink"):
        publish_outer_snapshot(other, 1, manifest)


def test_directory_alias_is_rejected_without_symlink_flag(tmp_path, monkeypatch):
    import training_snapshots as snapshots

    root = tmp_path / "root"
    root.mkdir()
    child = root / "outer_01"
    child.mkdir()
    redirected = root / "redirected"
    redirected.mkdir()
    original_resolve = Path.resolve

    def aliased_resolve(path, *args, **kwargs):
        if path == child:
            return redirected
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", aliased_resolve)
    with pytest.raises(ContractError, match="directory alias"):
        snapshots._safe_child(root, child)


def test_resume_only_backfills_exact_p0_boundary_and_rejects_missing_prefix(tmp_path):
    manifest = _manifest(tmp_path)
    run = tmp_path / "snaprun-r5"
    run.mkdir()
    _checkpoint(run, manifest, 2, phase="solver", step=1)
    with pytest.raises(ContractError, match="completed snapshots"):
        ensure_resume_snapshots(run, run / "ckpt", manifest)
    _checkpoint(run, manifest, 1)
    ensure_resume_snapshots(run, run / "ckpt", manifest)
    assert (run / "checkpoints" / "outer_01").is_dir()
    assert not (run / "checkpoints" / "outer_02").exists()


def test_partial_copy_failure_is_retained_and_reported(tmp_path, monkeypatch):
    import training_snapshots as snapshots

    manifest = _manifest(tmp_path)
    run = tmp_path / "snaprun-r6"
    run.mkdir()
    _checkpoint(run, manifest, 1)
    original = snapshots.shutil.copy2
    def fail_once(source, destination, *args, **kwargs):
        Path(destination).write_bytes(b"partial")
        raise OSError("synthetic copy interruption")
    monkeypatch.setattr(snapshots.shutil, "copy2", fail_once)
    with pytest.raises(ContractError, match="diagnostic partial retained") as exc:
        publish_outer_snapshot(run, 1, manifest)
    partial_text = str(exc.value).split("diagnostic partial retained at ", 1)[1]
    partial = Path(partial_text.rsplit(": ", 2)[0])
    assert partial.exists() and list(partial.iterdir())
    monkeypatch.setattr(snapshots.shutil, "copy2", original)
