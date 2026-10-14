"""Dependency-light, read-only metadata inventory for a local model directory.

Metadata and model-card strings are provenance hints only. This module never loads
model tensors or infers which weights a directory contains.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

_METADATA_FILES = (
    "config.json", "generation_config.json", "tokenizer_config.json",
    "README.md", "model.safetensors.index.json", "pytorch_model.bin.index.json",
)
_MAX_METADATA_BYTES = 8 * 1024 * 1024
_HASH_BLOCK = 8 * 1024 * 1024


def _sha256_bounded(path: Path, limit: int | None = None) -> tuple[str, int, bool]:
    digest = hashlib.sha256()
    read = 0
    truncated = False
    with path.open("rb") as stream:
        while True:
            want = _HASH_BLOCK if limit is None else min(_HASH_BLOCK, limit - read)
            if want <= 0:
                truncated = stream.read(1) != b""
                break
            block = stream.read(want)
            if not block:
                break
            digest.update(block)
            read += len(block)
    return digest.hexdigest(), read, truncated


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as stream:
            raw = stream.read(_MAX_METADATA_BYTES + 1)
        if len(raw) > _MAX_METADATA_BYTES:
            return None
        value = json.loads(raw.decode("utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _read_readme_hints(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            raw = stream.read(_MAX_METADATA_BYTES)
    except OSError:
        return {}
    text = raw.decode("utf-8", errors="replace")
    frontmatter = ""
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for end, line in enumerate(lines[1:], start=1):
            if line.strip() in ("---", "..."):
                frontmatter = "\n".join(lines[1:end])
                break
    hints: dict[str, Any] = {}
    heading = re.search(r"(?m)^#\s+(.+?)\s*$", text)
    if heading:
        hints["declared_model_heading"] = heading.group(1).strip()
    for key in ("base_model", "license_link"):
        match = re.search(rf"(?im)^[ \t]*{re.escape(key)}[ \t]*:[ \t]*([^\r\n]*)",
                          frontmatter)
        if not match:
            continue
        raw_value = match.group(1).strip().strip("'\"")
        if key == "base_model":
            # Handle HF YAML inline lists, scalar values, and indented block lists.
            if raw_value:
                values = re.findall(r"['\"]([^'\"]+)['\"]|([^,\[\]\s]+)", raw_value)
                hints["ancestor_base_models"] = [a or b for a, b in values if a or b]
            else:
                tail = frontmatter[match.end():]
                items = []
                for line in tail.splitlines():
                    item = re.match(r"^[ \t]*-[ \t]+([^\r\n#]+)", line)
                    if item:
                        items.append(item.group(1).strip().strip("'\""))
                        continue
                    if line.strip() and not re.match(r"^[ \t]*-", line):
                        break
                if items:
                    hints["ancestor_base_models"] = items
        else:
            hints[key] = raw_value
    return hints


def _snapshot_commit(path: Path) -> str | None:
    parts = path.parts
    try:
        index = parts.index("snapshots")
    except ValueError:
        return None
    return parts[index + 1] if index + 1 < len(parts) else None


def inspect_model_identity(model_path: str | os.PathLike[str], *,
                           hash_weights: bool = False) -> dict[str, Any]:
    """Describe local metadata and file inventory without importing ML libraries.

    ``hash_weights`` opts into a full sequential SHA256 read of each discovered
    weight file. By default tensor files are stat'ed only (name and byte size).
    """
    requested = os.fspath(model_path)
    expanded = Path(os.path.expandvars(os.path.expanduser(requested)))
    if not expanded.is_dir():
        return {
            "status": "metadata_unavailable",
            "requested_path": requested,
            "reason": "path is not an existing local directory; no network lookup attempted",
            "weights_verified": False,
        }

    root = expanded.resolve()
    metadata: dict[str, Any] = {}
    hashes: dict[str, Any] = {}
    for name in _METADATA_FILES:
        path = root / name
        if not path.is_file():
            continue
        sha, read_bytes, truncated = _sha256_bounded(path, _MAX_METADATA_BYTES)
        hashes[name] = {"sha256_prefix": sha, "bytes_hashed": read_bytes,
                        "truncated": truncated, "file_bytes": path.stat().st_size}
        if name.endswith(".json"):
            metadata[name] = _read_json(path)
    config = metadata.get("config.json") or {}
    gen = metadata.get("generation_config.json") or {}
    tok = metadata.get("tokenizer_config.json") or {}
    readme = _read_readme_hints(root / "README.md") if (root / "README.md").is_file() else {}

    weight_names: set[str] = set()
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = metadata.get(index_name) or {}
        weight_map = index.get("weight_map")
        if isinstance(weight_map, dict):
            weight_names.update(name for name in weight_map.values()
                                if isinstance(name, str) and Path(name).name == name
                                and "/" not in name and "\\" not in name
                                and name not in (".", "..")
                                and name.endswith((".safetensors", ".bin")))
    if not weight_names:
        weight_names.update(p.name for pattern in ("*.safetensors", "pytorch_model*.bin")
                            for p in root.glob(pattern) if p.is_file())

    weights = []
    for name in sorted(weight_names):
        path = root / name
        if not path.is_file():
            weights.append({"filename": name, "present": False})
            continue
        entry: dict[str, Any] = {"filename": name, "present": True,
                                 "bytes": path.stat().st_size}
        if hash_weights:
            entry["sha256"] = _sha256_bounded(path)[0]
        weights.append(entry)

    config_hint = config.get("_name_or_path") if isinstance(config, dict) else None
    provenance = {"readme": readme, "config_name_or_path_hint": config_hint}
    heading = readme.get("declared_model_heading")
    config_text = str(config_hint).strip() if config_hint else ""
    recognized_repo_hint = (config_text if re.fullmatch(r"[^/\\\s]+/[^/\\\s]+", config_text)
                            else None)
    config_repo_hint = recognized_repo_hint.split("/")[-1] if recognized_repo_hint else None
    conflict = bool(heading and config_repo_hint and
                    str(heading).casefold() != config_repo_hint.casefold())
    tokenizer_metadata = {}
    for key in ("tokenizer_class", "model_max_length", "eos_token_id", "bos_token_id",
                "pad_token_id", "chat_template"):
        if key in tok:
            value = tok[key]
            if key == "chat_template":
                value = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
            tokenizer_metadata[key] = value

    return {
        "status": "weights_unverified",
        "requested_path": requested,
        "local_path": str(root),
        "snapshot_commit_hint": _snapshot_commit(root),
        "provenance_hints": provenance,
        "provenance_declaration_conflict": conflict,
        "provenance_declaration_ambiguous": bool(heading and config_hint and
                                                  recognized_repo_hint is None),
        "provenance_is_not_weight_proof": True,
        "architecture": config.get("architectures") if isinstance(config, dict) else None,
        "model_type": config.get("model_type") if isinstance(config, dict) else None,
        "config_dtype": config.get("torch_dtype") if isinstance(config, dict) else None,
        "generation_config_fields": {key: gen[key] for key in
                                     ("eos_token_id", "pad_token_id", "bos_token_id")
                                     if key in gen},
        "config_token_boundary_fields": {key: config[key] for key in
                                          ("eos_token_id", "max_position_embeddings")
                                          if isinstance(config, dict) and key in config},
        "tokenizer_fields": tokenizer_metadata,
        "metadata_sha256": hashes,
        "weights": weights,
        "weights_hash_requested": bool(hash_weights),
        "weights_verified": False,
    }
