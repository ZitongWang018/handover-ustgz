"""Offline solver-only BF16 export with checked source and output identities.

The sealed checkpoint remains the scientific arm. This directory is a derived
inference artifact, never a replacement for a resumable training checkpoint.
"""
from __future__ import annotations

import copy
import json
import stat
from importlib.metadata import version
from pathlib import Path

from checkpoint import base_file_inventory
from evaluation import canonical_hash, sha256_file

MANIFEST_NAME = "merged_model.json"
SCHEMA = 1


def _is_link(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        return getattr(path.lstat(), "st_reparse_tag", 0) == getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", 0xA0000003)
    except FileNotFoundError:
        return False


def _directory_path(value) -> Path:
    """Check spelling before resolve, including symlinked parent directories."""
    raw = Path(value).expanduser().absolute()
    for part in (raw, *raw.parents):
        if _is_link(part):
            raise ValueError(f"merged/solver directory path must not contain symlinks: {part}")
    return raw.resolve()


def _records(root: Path, *, completion_manifest: bool = False) -> list[dict]:
    if not root.is_dir():
        raise ValueError(f"model directory is missing: {root}")
    records = []
    for path in sorted(root.rglob("*")):
        if _is_link(path):
            raise ValueError(f"model export/solver must not contain symlinks: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"model directory contains a non-regular file: {path}")
        relative = path.relative_to(root).as_posix()
        if completion_manifest and relative == MANIFEST_NAME:
            continue
        records.append({"path": relative, "size": path.stat().st_size,
                        "sha256": sha256_file(path)})
    return records


def _check_identity(base_digest: str, arm: dict) -> tuple[Path, list[dict]]:
    if not isinstance(arm, dict) or arm.get("kind") != "checkpoint":
        raise ValueError("checkpoint identity must describe a checkpoint arm")
    required = ("path", "manifest_sha256", "solver_path", "solver_files",
                "base_inventory_sha256")
    if any(key not in arm for key in required):
        raise ValueError("checkpoint identity is incomplete")
    if arm["base_inventory_sha256"] != base_digest:
        raise ValueError("checkpoint identity was produced for a different base model")
    solver = _directory_path(arm["solver_path"])
    expected = arm["solver_files"]
    if not isinstance(expected, list) or _records(solver) != expected:
        raise ValueError("solver adapter files differ from checkpoint identity")
    if not (solver / "adapter_config.json").is_file():
        raise ValueError("solver adapter config is missing")
    if not any(item["path"].endswith((".safetensors", ".bin")) for item in expected):
        raise ValueError("solver adapter weights are missing")
    return solver, expected


def _checkpoint_seals(arm: dict) -> dict:
    root = _directory_path(arm["path"])
    records = {}
    for name in ("training_manifest.json", "COMPLETE.json", "checkpoint_inventory.json"):
        path = root / name
        if _is_link(path) or not path.is_file():
            raise ValueError(f"checkpoint seal is missing or linked: {name}")
        records[name] = sha256_file(path)
    manifest = json.loads((root / "training_manifest.json").read_text(encoding="utf-8"))
    if canonical_hash(manifest) != arm["manifest_sha256"]:
        raise ValueError("checkpoint manifest differs from the validated arm")
    return records


def _model_layout(model) -> tuple[dict, list[list[str]]]:
    state = model.state_dict()
    layout = {name: list(tensor.shape) for name, tensor in state.items()}
    shared = {}
    for name, tensor in state.items():
        if tensor.numel():
            key = (tensor.data_ptr(), tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()))
            shared.setdefault(key, []).append(name)
    return layout, [names for names in shared.values() if len(names) > 1]


def _check_tensors(root: Path, records: list[dict], expected: dict, aliases: list) -> None:
    from safetensors import safe_open
    actual = {}
    owners = {}
    for record in records:
        if not record["path"].endswith(".safetensors"):
            continue
        with safe_open(str(root / record["path"]), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in actual:
                    raise ValueError(f"duplicate merged tensor: {name}")
                tensor = handle.get_slice(name)
                if tensor.get_dtype().startswith(("F", "BF")) and tensor.get_dtype() != "BF16":
                    raise ValueError(f"merged floating tensor is not BF16: {name}")
                actual[name] = tensor.get_shape()
                owners[name] = record["path"]
    index = root / "model.safetensors.index.json"
    if index.is_file() and json.loads(index.read_text(encoding="utf-8")).get("weight_map") != owners:
        raise ValueError("merged tensor shard index disagrees with saved tensor locations")
    if not isinstance(expected, dict) or not expected or not isinstance(aliases, list):
        raise ValueError("merged model lacks its complete source tensor layout")
    if set(actual) - set(expected) or any(actual[name] != expected[name] for name in actual):
        raise ValueError("merged model tensor names/shapes differ from the base")
    missing = set(expected) - set(actual)
    for group in aliases:
        if not isinstance(group, list) or len(group) < 2 or not set(group) <= set(expected):
            raise ValueError("invalid shared tensor aliases")
        if any(name in actual for name in group):
            missing.difference_update(group)
    if missing:
        raise ValueError(f"merged model tensors are missing: {sorted(missing)[:5]}")


def _check_full_model(root: Path, records: list[dict]) -> None:
    if any(Path(item["path"]).name.startswith(("adapter_config", "adapter_model")) for item in records):
        raise ValueError("merged full model must not contain residual adapter files")
    # Validate the layout using the already-hashed inventory; do not reread 60GB.
    paths = {item["path"] for item in records}
    if "config.json" not in paths or not any(name.endswith((".safetensors", ".bin")) for name in paths):
        raise ValueError("merged model config or full weights are missing")
    if not isinstance(json.loads((root / "config.json").read_text(encoding="utf-8")), dict):
        raise ValueError("merged model config must be a mapping")
    for name in paths:
        if not name.endswith((".safetensors.index.json", "pytorch_model.bin.index.json")):
            continue
        index = json.loads((root / name).read_text(encoding="utf-8"))
        mapping = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("merged model shard index is invalid")
        for shard in mapping.values():
            if (not isinstance(shard, str) or Path(shard).is_absolute() or
                    ".." in Path(shard.replace("\\", "/")).parts or
                    shard not in paths or not shard.endswith((".safetensors", ".bin"))):
                raise ValueError("merged model shard index references a missing or unsafe shard")


def validate_merged_model(output_path, *, base_digest, arm) -> dict:
    out = _directory_path(output_path)
    manifest_path = out / MANIFEST_NAME
    if manifest_path.is_symlink():
        raise ValueError("merged model manifest must not be a symlink")
    if not manifest_path.is_file():
        raise ValueError("merged model completion manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("merged model manifest must be a mapping")
    payload = dict(manifest)
    digest = payload.pop("manifest_sha256", None)
    if digest != canonical_hash(payload):
        raise ValueError("merged model manifest checksum is invalid")
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported merged model manifest schema")
    if not isinstance(arm, dict) or arm.get("kind") != "checkpoint":
        raise ValueError("merged model requires a checkpoint arm")
    if manifest.get("checkpoint_identity") != arm:
        raise ValueError("merged model was built from a different checkpoint")
    if _checkpoint_seals(arm) != manifest.get("checkpoint_seals"):
        raise ValueError("source checkpoint seals changed after export")
    if (manifest.get("base_inventory_sha256") != base_digest or
            arm.get("base_inventory_sha256") != base_digest):
        raise ValueError("merged model was built from a different base model")
    if manifest.get("dtype") != "bfloat16" or manifest.get("safe_merge") is not True:
        raise ValueError("merged model does not record a safe bfloat16 merge")
    software = manifest.get("software")
    if not isinstance(software, dict) or set(software) != {"torch", "transformers", "peft"}:
        raise ValueError("merged model lacks its export software identity")
    if any(software[name] != version(name) for name in software):
        raise ValueError("merged export software differs from evaluation runtime")
    actual = _records(out, completion_manifest=True)
    if actual != manifest.get("files"):
        raise ValueError("merged model files are missing, extra, or tampered")
    _check_full_model(out, actual)
    _check_tensors(out, actual, manifest.get("tensor_shapes"), manifest.get("shared_tensors"))
    return manifest


def export_solver_model(base_path, solver_path, output_path, *, checkpoint_identity) -> dict:
    """Merge one validated solver adapter. Failure leaves an incomplete directory."""
    arm = copy.deepcopy(checkpoint_identity)
    out = _directory_path(output_path)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite merged model: {out}")
    base = Path(base_path).expanduser().resolve(strict=True)
    base_inventory = base_file_inventory(base)
    base_digest = canonical_hash(base_inventory)
    solver, solver_files = _check_identity(base_digest, arm)
    seals = _checkpoint_seals(arm)
    if solver != _directory_path(solver_path):
        raise ValueError("solver path differs from checkpoint identity")
    for source in (base, solver, Path(arm["path"]).expanduser().resolve(strict=True)):
        if out == source or out.is_relative_to(source) or source.is_relative_to(out):
            raise ValueError("merged output must not overlap any source model/checkpoint")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(base), trust_remote_code=True,
                                              local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(base), torch_dtype=torch.bfloat16, device_map={"": "cpu"},
        low_cpu_mem_usage=True, trust_remote_code=True, local_files_only=True)
    tensor_shapes, shared_tensors = _model_layout(model)
    model = PeftModel.from_pretrained(model, str(solver), is_trainable=False,
                                      local_files_only=True)
    merged = model.merge_and_unload(safe_merge=True).to(dtype=torch.bfloat16).eval()
    if _model_layout(merged)[0] != tensor_shapes:
        raise ValueError("solver merge lost or added model parameters/buffers")
    merged.save_pretrained(str(out), safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(str(out))
    if (base_file_inventory(base) != base_inventory or _records(solver) != solver_files or
            _checkpoint_seals(arm) != seals):
        raise ValueError("source model files changed during merge")
    files = _records(out)
    _check_full_model(out, files)
    _check_tensors(out, files, tensor_shapes, shared_tensors)
    manifest = {"schema": SCHEMA, "checkpoint_identity": arm,
                "base_inventory_sha256": base_digest, "dtype": "bfloat16",
                "safe_merge": True, "files": files, "checkpoint_seals": seals,
                "tensor_shapes": tensor_shapes, "shared_tensors": shared_tensors,
                "software": {name: version(name) for name in ("torch", "transformers", "peft")}}
    manifest["manifest_sha256"] = canonical_hash(manifest)
    # The completion marker is written only after both source and output checks.
    with (out / MANIFEST_NAME).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest
