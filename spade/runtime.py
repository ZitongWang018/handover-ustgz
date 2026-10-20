"""Read-only provenance and filesystem checks for the standalone tau2 runner."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

BASE = Path(__file__).resolve().parent
PINS = {"upstream": "ebd40ec872fc5630cac299cb5e38e7c89743bef5",
        "tau2-bench": "a2c024725189473d2d7cea3a5cfdbcc67478e41f"}
MODEL_NAME = "qwen3-30b-a3b-instruct-2507"
DOMAINS = {"retail": 114, "telecom": 114, "airline": 50}


def bounded(value, root: Path, *, exists=False) -> Path:
    path = Path(value).expanduser().resolve(strict=exists)
    if path != root and not path.is_relative_to(root):
        raise ValueError(f"path is outside the authorized ztwang root: {path}")
    return path


def tree_files(directory: Path, root: Path):
    for current, directories, names in os.walk(directory, followlinks=False):
        for name in directories:
            child = Path(current) / name
            bounded(child, root, exists=True)
            tag = getattr(child.lstat(), "st_reparse_tag", 0)
            if child.is_symlink() or tag == getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", 0xA0000003):
                raise ValueError(f"directory links are not accepted: {child}")
        for name in names:
            path = Path(current) / name
            target = bounded(path, root, exists=True)
            if not target.is_file():
                raise ValueError(f"not a regular file: {path}")
            yield path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(directory: Path, root: Path) -> list[dict]:
    return [{"path": path.relative_to(directory).as_posix(), "size": path.stat().st_size,
             "sha256": sha256(path)} for path in sorted(tree_files(directory, root))]


def verify_pins() -> dict:
    for name, expected in PINS.items():
        directory = BASE / name
        actual = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "--no-optional-locks", "-C", str(directory), "status", "--porcelain"], text=True)
        if actual != expected or dirty:
            raise ValueError(f"{name} is not the clean pinned checkout {expected}")
    return dict(PINS)


def endpoint_url(value: str) -> str:
    parts = urlsplit(value)
    if (parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or
            parts.password or parts.query or parts.fragment or parts.path.rstrip("/") not in {"", "/v1"}):
        raise ValueError("endpoint must be an HTTP(S) origin, optionally ending in /v1")
    if parts.port is not None and not 1 <= parts.port <= 65535:
        raise ValueError("invalid endpoint port")
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


def check_model(model: Path, root: Path) -> dict:
    files = {p.relative_to(model).as_posix(): p for p in tree_files(model, root)}
    if any(Path(name).name.startswith("adapter_") for name in files):
        raise ValueError("a full original HF checkpoint is required, not an EZ adapter")
    if "config.json" not in files or "tokenizer_config.json" not in files:
        raise ValueError("model config and tokenizer config are required")
    cfg = json.loads(files["config.json"].read_text(encoding="utf-8"))
    expected = {"model_type": "qwen3_moe", "num_hidden_layers": 48, "hidden_size": 2048,
                "num_attention_heads": 32, "num_key_value_heads": 4, "num_experts": 128,
                "num_experts_per_tok": 8, "vocab_size": 151936, "rope_theta": 10000000,
                "head_dim": 128, "moe_intermediate_size": 768}
    if any(cfg.get(key) != val for key, val in expected.items()) or cfg.get("quantization_config"):
        raise ValueError("model does not match the unquantized Qwen3-30B-A3B-Instruct-2507 architecture")
    tokenizer = json.loads(files["tokenizer_config.json"].read_text(encoding="utf-8"))
    if not tokenizer.get("chat_template"):
        raise ValueError("the original model chat template is missing")
    if not any(name.endswith((".safetensors", ".bin")) for name in files):
        raise ValueError("full HF weight files are missing")
    for name, path in files.items():
        if not name.endswith((".safetensors.index.json", "pytorch_model.bin.index.json")):
            continue
        mapping = json.loads(path.read_text(encoding="utf-8")).get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("invalid weight shard index")
        if any(not isinstance(shard, str) or shard not in files or
               not shard.endswith((".safetensors", ".bin")) for shard in mapping.values()):
            raise ValueError("weight shard index references unsafe or missing files")
    return {"architecture": expected, "model_config_sha256": sha256(files["config.json"]),
            "tokenizer_config_sha256": sha256(files["tokenizer_config.json"]),
            "identity_limit": "architecture/template checked; remote endpoint name cannot prove loaded weights"}


def task_manifest(root: Path) -> dict:
    result = {}
    for domain, count in DOMAINS.items():
        directory = bounded(BASE / "tau2-bench/data/tau2/domains" / domain, root, exists=True)
        # Check every file before reading tasks/splits, including linked data files.
        paths = list(tree_files(directory, root))
        split = json.loads((directory / "split_tasks.json").read_text(encoding="utf-8"))["base"]
        tasks = json.loads((directory / "tasks.json").read_text(encoding="utf-8"))
        ids = [str(task["id"]) for task in tasks if str(task["id"]) in set(map(str, split))]
        if (len(split) != count or len(ids) != count or len(set(ids)) != count or
                set(ids) != set(map(str, split))):
            raise ValueError(f"{domain}/base task set differs from the pinned {count} tasks")
        result[domain] = {"task_ids": ids, "trials": 4, "files": [
            {"path": p.relative_to(directory).as_posix(), "sha256": sha256(p)} for p in sorted(paths)]}
    return result


def cache_environment(output: Path) -> dict:
    values = {}
    for name in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_MODULES_CACHE",
                 "TRANSFORMERS_CACHE", "TORCH_HOME", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
                 "CUDA_CACHE_PATH", "XDG_CACHE_HOME", "TIKTOKEN_CACHE_DIR", "WANDB_DIR",
                 "TMPDIR", "TMP", "TEMP"):
        path = output / "cache" / name.lower()
        path.mkdir(parents=True, exist_ok=False)
        values[name] = str(path)
    values.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHON_DOTENV_DISABLED="1",
                  PYTHONDONTWRITEBYTECODE="1",
                  PYTHONUTF8="1", LITELLM_LOCAL_MODEL_COST_MAP="True",
                  TAU2_DATA_DIR=str(BASE / "tau2-bench/data"))
    return values
