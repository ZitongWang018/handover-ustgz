#!/usr/bin/env python3
"""Evaluate all sealed outer-loop solver snapshots after training has exited."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "selfplay" / "src"
sys.path.insert(0, str(SRC))

from checkpoint import (ContractError, base_file_inventory, contract_signature,
                        validate_checkpoint_v3)
from evaluation import SCHEMA as EVAL_SCHEMA, canonical_hash, compare, sha256_file, summarize, validate_suite
import eval_runner


class RunEvaluationError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunEvaluationError(f"cannot read {path.name}: {type(exc).__name__}") from None
    if not isinstance(value, dict):
        raise RunEvaluationError(f"{path.name} must contain a JSON object")
    return value


def _write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    except FileExistsError:
        raise RunEvaluationError(f"refusing to overwrite existing artifact: {path}") from None


def _manifest(run_dir: Path) -> tuple[dict, str, dict]:
    manifest = _read_json(run_dir / "training_manifest.json")
    digest = manifest.get("contract_sha256")
    if not isinstance(digest, str) or digest != contract_signature(manifest):
        raise RunEvaluationError("training manifest contract hash is invalid")
    binding = manifest.get("baseline_binding")
    if not isinstance(binding, dict) or not isinstance(binding.get("contract"), dict):
        raise RunEvaluationError("training manifest lacks its complete baseline contract binding")
    contract = binding["contract"]
    bound_sha = binding.get("contract_sha256")
    raw = dict(contract)
    contract_sha = raw.pop("contract_sha256", None)
    if (not isinstance(bound_sha, str) or contract_sha != bound_sha or
            canonical_hash(raw) != bound_sha):
        raise RunEvaluationError("embedded baseline contract checksum is invalid")
    if binding.get("common_identity", {}).get("base_inventory_sha256") != \
            contract.get("protocol_signature", {}).get("base_inventory_sha256"):
        raise RunEvaluationError("baseline binding common base identity is inconsistent")
    _verify_current_baseline_identity(contract, binding)
    protocol = _read_json(run_dir / "run_protocol.json")
    if protocol.get("contract_sha256") != digest or not protocol.get("run_id"):
        raise RunEvaluationError("run protocol identity differs from training manifest")
    resolved = manifest.get("resolved_config", {})
    if resolved.get("train", {}).get("outer_loops") != 4:
        raise RunEvaluationError("four completed outer loops are required")
    if manifest.get("checkpoint_eligibility") != "formal_training":
        raise RunEvaluationError("smoke or ineligible training runs cannot be evaluated")
    return manifest, str(protocol["run_id"]), contract


def _verify_current_baseline_identity(contract: dict, binding: dict) -> None:
    """Fail before any GPU evaluation if the bound baseline no longer matches this checkout."""
    config_path = Path(contract.get("config_path", "")).expanduser().resolve(strict=True)
    config = eval_runner._load_yaml(config_path)
    signature = contract.get("protocol_signature", {})
    settings = eval_runner._config_settings(config)
    if signature.get("config") != config or signature.get("config_sha256") != sha256_file(config_path):
        raise RunEvaluationError("current evaluation config differs from the bound baseline")
    if signature.get("solver_settings") != settings:
        raise RunEvaluationError("current solver/evaluation settings differ from the bound baseline")
    if signature.get("runtime_source_sha256") != eval_runner._source_identity():
        raise RunEvaluationError("current evaluation source differs from the bound baseline")
    if signature.get("software") != eval_runner._versions():
        raise RunEvaluationError("current evaluation software versions differ from the bound baseline")
    if signature.get("prompt_source_sha256") != sha256_file(SRC / "protocol.py"):
        raise RunEvaluationError("current ReAct prompt/action protocol differs from the bound baseline")
    base_path, base_inventory, base_digest = eval_runner._base_identity(contract["base_path"])
    if (base_inventory != contract.get("base_inventory") or
            base_digest != signature.get("base_inventory_sha256")):
        raise RunEvaluationError("current base model files differ from the bound baseline")
    tokenizer_files = {item["path"]: item for item in base_inventory["files"]
                       if item["path"].startswith("tokenizer") or item["path"] in
                       ("vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json")}
    if signature.get("tokenizer_files_sha256") != canonical_hash(tokenizer_files):
        raise RunEvaluationError("current tokenizer files differ from the bound baseline")
    from transformers import AutoTokenizer
    from protocol import solver_prompt_identity
    tok = AutoTokenizer.from_pretrained(str(base_path), local_files_only=True, trust_remote_code=True)
    if type(tok).__qualname__ != signature.get("tokenizer_class"):
        raise RunEvaluationError("current tokenizer class differs from baseline")
    identity_now = solver_prompt_identity(tok, settings.get("prompt_format", "raw"))
    for key, value in identity_now.items():
        legacy = "raw" if key == "prompt_format" and identity_now["prompt_format"] == "raw" else None
        if signature.get(key, legacy) != value:
            raise RunEvaluationError(f"current tokenizer/chat template differs from baseline: {key}")
    data_root = Path(contract.get("alfworld_data_root", "")).expanduser().resolve(strict=True)
    if not data_root.is_dir() or signature.get("alfworld_data_root") != str(data_root):
        raise RunEvaluationError("current ALFWorld data root differs from the bound baseline")
    alfworld_yaml = eval_runner._alfworld_config_path(config_path, config)
    if (str(alfworld_yaml) != contract.get("alfworld_config_path") or
            signature.get("alfworld_config_path") != str(alfworld_yaml)):
        raise RunEvaluationError("current ALFWorld YAML path differs from the bound baseline")
    os.environ["ALFWORLD_DATA"] = str(data_root)
    suite = contract.get("suite")
    validate_suite(suite)
    fresh = eval_runner.discover_suite(str(alfworld_yaml), "valid_seen", settings["shards"])
    validate_suite(fresh)
    if fresh != suite or signature.get("pool_sha256") != suite.get("pool_sha256"):
        raise RunEvaluationError("current complete valid_seen pool differs from the bound baseline")
    identity = binding.get("common_identity", {})
    if (identity.get("config_sha256") != signature.get("config_sha256") or
            identity.get("base_inventory_sha256") != base_digest or
            identity.get("pool_sha256") != suite.get("pool_sha256") or
            identity.get("tokenizer_files_sha256") != signature.get("tokenizer_files_sha256")):
        raise RunEvaluationError("baseline common identity record differs from its contract")


def _validate_baseline_merged(path: Path, contract: dict) -> dict:
    merged = _read_json(path)
    suite = contract.get("suite")
    validate_suite(suite)
    digest = contract.get("contract_sha256")
    if merged.get("schema") != EVAL_SCHEMA or merged.get("contract_sha256") != digest:
        raise RunEvaluationError("baseline merged result is not the output of the embedded baseline contract")
    expected_ids = [task["task_id"] for task in suite["tasks"]]
    if merged.get("run_id") != contract.get("run_id") or merged.get("expected_task_ids") != expected_ids:
        raise RunEvaluationError("baseline merged run/task identity differs from its contract")
    for key in ("protocol_signature", "arm_identity", "base_inventory"):
        if merged.get(key) != contract.get(key):
            raise RunEvaluationError(f"baseline merged {key} differs from its contract")
    if merged.get("arm_identity", {}).get("kind") != "base-only":
        raise RunEvaluationError("baseline results must be from the base-only arm")
    signature = contract.get("protocol_signature", {})
    for key in ("software", "runtime_source_sha256"):
        if merged.get(key) != signature.get(key):
            raise RunEvaluationError(f"baseline merged {key} differs from its contract")
    rows = merged.get("rows")
    if not isinstance(rows, list):
        raise RunEvaluationError("baseline merged rows are missing")
    task_map = {task["task_id"]: task for task in suite["tasks"]}
    for row in rows:
        task = task_map.get(row.get("task_id"))
        if task is None or any(row.get(key) != task.get(key) for key in ("task_type", "game_file", "game_sha256")):
            raise RunEvaluationError("baseline episode rows disagree with the bound task pool")
    try:
        computed_summary = summarize(rows, suite["tasks"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RunEvaluationError(f"baseline task/seed grid or trace validation failed: {exc}") from None
    if merged.get("summary") != computed_summary:
        raise RunEvaluationError("baseline merged summary differs from its episode rows")
    return merged


def validate_baseline_files(contract_path: str | Path, merged_path: str | Path) -> tuple[dict, dict]:
    """Validate existing immutable base contract/results before a new training run."""
    contract = _read_json(Path(contract_path).expanduser().resolve(strict=True))
    payload = dict(contract)
    digest = payload.pop("contract_sha256", None)
    if contract.get("schema") != EVAL_SCHEMA or digest != canonical_hash(payload):
        raise RunEvaluationError("baseline evaluation contract schema/checksum is invalid")
    if contract.get("arm_identity", {}).get("kind") != "base-only":
        raise RunEvaluationError("baseline evaluation contract must be base-only")
    return contract, _validate_baseline_merged(Path(merged_path).expanduser().resolve(strict=True), contract)


def _validate_index(run_dir: Path, run_id: str, manifest: dict, manifest_sha: str) -> list[dict]:
    checkpoints_root = run_dir / "checkpoints"
    if checkpoints_root.is_symlink():
        raise RunEvaluationError("checkpoint directory must not be a symlink")
    index_path = checkpoints_root / "index.json"
    if index_path.is_symlink():
        raise RunEvaluationError("checkpoint index must not be a symlink")
    index = _read_json(index_path)
    if (index.get("schema_version") != 1 or index.get("run_id") != run_id or
            index.get("contract_sha256") != manifest_sha):
        raise RunEvaluationError("checkpoint index identity/schema differs from this training run")
    entries = index.get("snapshots")
    if not isinstance(entries, list) or len(entries) != 4:
        raise RunEvaluationError("checkpoint index must contain exactly four outer snapshots")
    expected_manifest_sha = manifest_sha
    validated = []
    seen = set()
    for expected_outer, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict) or entry.get("outer") != expected_outer:
            raise RunEvaluationError("checkpoint index outer snapshots must be ordered 1 through 4")
        relative = entry.get("path")
        if relative != f"checkpoints/outer_{expected_outer:02d}" or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise RunEvaluationError(f"outer {expected_outer} checkpoint path is invalid")
        if expected_outer in seen:
            raise RunEvaluationError("checkpoint index contains duplicate outer snapshots")
        seen.add(expected_outer)
        raw_checkpoint_path = run_dir / relative
        if raw_checkpoint_path.is_symlink():
            raise RunEvaluationError("indexed checkpoint path must not be a symlink")
        for component in (raw_checkpoint_path / "COMPLETE.json",
                          raw_checkpoint_path / "checkpoint_inventory.json"):
            if component.is_symlink():
                raise RunEvaluationError("checkpoint seal files must not be symlinks")
        checkpoint_path = raw_checkpoint_path.resolve(strict=True)
        if not checkpoint_path.is_relative_to(run_dir.resolve()):
            raise RunEvaluationError("indexed checkpoint path escapes the run directory")
        try:
            checkpoint = validate_checkpoint_v3(checkpoint_path, expected_contract=expected_manifest_sha,
                                                purpose="evaluation")
        except (ContractError, OSError, ValueError) as exc:
            raise RunEvaluationError(f"outer {expected_outer} checkpoint seal validation failed: {exc}") from None
        state = checkpoint["state"]
        expected_position = {"outer": expected_outer, "phase": "proposer", "step": 0}
        if state.get("next_position") != expected_position:
            raise RunEvaluationError(f"outer {expected_outer} is not a completed solver checkpoint")
        if state.get("run_id") != run_id or checkpoint["manifest"] != manifest:
            raise RunEvaluationError(f"outer {expected_outer} checkpoint manifest/run identity differs")
        marker_path = checkpoint_path / "COMPLETE.json"
        inventory_path = checkpoint_path / "checkpoint_inventory.json"
        marker = _read_json(marker_path)
        if (entry.get("contract_sha256") != expected_manifest_sha or
                entry.get("complete_marker_sha256") != sha256_file(marker_path) or
                entry.get("inventory_sha256") != marker.get("inventory_sha256") or
                entry.get("inventory_sha256") != sha256_file(inventory_path) or
                entry.get("next_position") != expected_position):
            raise RunEvaluationError(f"outer {expected_outer} index seal identity differs from checkpoint")
        validated.append({"outer": expected_outer, "path": checkpoint_path,
                          "checkpoint_manifest": checkpoint["manifest"],
                          "checkpoint_manifest_sha256": canonical_hash(checkpoint["manifest"]),
                          "complete_marker_sha256": entry["complete_marker_sha256"],
                          "inventory_sha256": entry["inventory_sha256"],
                          "next_position": expected_position})
    return validated


def _validate_run_merged(merged_path: Path, eval_contract_path: Path, baseline: dict,
                         snapshot: dict) -> tuple[dict, dict, dict]:
    eval_contract = _read_json(eval_contract_path)
    raw = dict(eval_contract)
    digest = raw.pop("contract_sha256", None)
    if digest != canonical_hash(raw):
        raise RunEvaluationError(f"outer {snapshot['outer']} evaluation contract checksum is invalid")
    validate_suite(eval_contract.get("suite"))
    if eval_contract.get("protocol_signature") != baseline["protocol_signature"]:
        raise RunEvaluationError(f"outer {snapshot['outer']} evaluation protocol differs from baseline")
    arm = eval_contract.get("arm_identity", {})
    if (arm.get("kind") != "checkpoint" or
            Path(arm.get("path", "")).resolve() != snapshot["path"] or
            arm.get("manifest_sha256") != snapshot["checkpoint_manifest_sha256"]):
        raise RunEvaluationError(f"outer {snapshot['outer']} evaluation contract names the wrong checkpoint")
    merged = _read_json(merged_path)
    _validate_merged_against_contract(merged, eval_contract)
    comparison = compare(baseline, merged)
    comparison_path = merged_path.parent / "comparison.json"
    if comparison_path.exists():
        stored_comparison = _read_json(comparison_path)
        if stored_comparison != comparison:
            raise RunEvaluationError(f"outer {snapshot['outer']} comparison differs from raw merged rows")
    return eval_contract, merged, comparison


def _validate_merged_against_contract(merged: dict, contract: dict) -> None:
    suite = contract["suite"]
    if (merged.get("schema") != EVAL_SCHEMA or
            merged.get("contract_sha256") != contract.get("contract_sha256") or
            merged.get("run_id") != contract.get("run_id")):
        raise RunEvaluationError("merged evaluation result identity differs from its contract")
    for key in ("protocol_signature", "arm_identity", "base_inventory"):
        if merged.get(key) != contract.get(key):
            raise RunEvaluationError(f"merged evaluation {key} differs from its contract")
    if merged.get("expected_task_ids") != [task["task_id"] for task in suite["tasks"]]:
        raise RunEvaluationError("merged evaluation task IDs differ from its contract")
    if merged.get("software") != contract["protocol_signature"].get("software") or \
            merged.get("runtime_source_sha256") != contract["protocol_signature"].get("runtime_source_sha256"):
        raise RunEvaluationError("merged evaluation source/software identity differs from its contract")
    rows = merged.get("rows")
    if not isinstance(rows, list):
        raise RunEvaluationError("merged evaluation rows are missing")
    tasks = {task["task_id"]: task for task in suite["tasks"]}
    for row in rows:
        task = tasks.get(row.get("task_id"))
        if task is None or any(row.get(key) != task.get(key) for key in ("task_type", "game_file", "game_sha256")):
            raise RunEvaluationError("merged episode rows disagree with their task pool")
    try:
        computed = summarize(rows, suite["tasks"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RunEvaluationError(f"merged task/seed grid or trace validation failed: {exc}") from None
    if merged.get("summary") != computed:
        raise RunEvaluationError("merged summary differs from raw episode rows")


def _clean_eval_environment(python: str) -> dict[str, str]:
    helper = importlib.import_module("training_runtime")
    clean = helper.clean_training_environment(dict(os.environ))
    if not isinstance(clean, dict):
        raise RunEvaluationError("clean_training_environment must return an environment mapping")
    original_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    if original_cuda is None:
        clean.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        clean["CUDA_VISIBLE_DEVICES"] = original_cuda
    clean["SKR_PYTHON"] = python
    return clean


def _run_command(argv: list[str], cwd: Path, env: dict[str, str], log_dir: Path, label: str) -> None:
    stdout_path, stderr_path = log_dir / f"{label}.stdout.log", log_dir / f"{label}.stderr.log"
    command_path = log_dir / "commands.jsonl"
    try:
        stdout_stream = stdout_path.open("x", encoding="utf-8", newline="\n", buffering=1)
        stderr_stream = stderr_path.open("x", encoding="utf-8", newline="\n", buffering=1)
    except FileExistsError:
        if 'stdout_stream' in locals():
            stdout_stream.close()
        raise RunEvaluationError(f"refusing to overwrite command log for {label}") from None
    try:
        with stdout_stream, stderr_stream, command_path.open("a", encoding="utf-8", newline="\n") as records:
            started = {"status": "started", "argv": argv, "cwd": str(cwd),
                       "stdout": stdout_path.name, "stderr": stderr_path.name}
            records.write(json.dumps(started, ensure_ascii=False, sort_keys=True) + "\n")
            records.flush()
            try:
                result = subprocess.run(argv, cwd=str(cwd), env=env, stdout=stdout_stream,
                                        stderr=stderr_stream, text=True, check=False)
                returncode, error = result.returncode, None
            except OSError as exc:
                returncode, error = None, f"{type(exc).__name__}: {exc}"
                stderr_stream.write(error + "\n")
            ended = {"status": "finished", "argv": argv, "cwd": str(cwd),
                     "returncode": returncode, "stdout": stdout_path.name,
                     "stderr": stderr_path.name, "error": error}
            records.write(json.dumps(ended, ensure_ascii=False, sort_keys=True) + "\n")
    except FileExistsError:
        stdout_stream.close()
        stderr_stream.close()
        raise RunEvaluationError(f"refusing to overwrite command metadata {command_path}") from None
    if returncode != 0:
        raise RunEvaluationError(f"command {label} failed with exit code {returncode}; see {stderr_path}")


def _outer_row(snapshot: dict, eval_contract: dict, merged: dict, comparison: dict) -> dict:
    return {"outer": snapshot["outer"], "checkpoint": str(snapshot["path"]),
        "checkpoint_manifest_sha256": snapshot["checkpoint_manifest_sha256"],
        "inventory_sha256": snapshot["inventory_sha256"],
        "eval_contract_sha256": eval_contract["contract_sha256"],
        "trained_summary": merged["summary"], "comparison": comparison}


def _write_summary(output_dir: Path, run_id: str, baseline: dict, baseline_path: Path,
                   index_path: Path, rows: list[dict]) -> None:
    payload = {"schema": "explorer-zero-run-evaluation-v1", "run_id": run_id,
        "baseline_contract_sha256": baseline["contract_sha256"],
        "baseline_results_sha256": sha256_file(baseline_path),
        "checkpoint_index_sha256": sha256_file(index_path), "outers": rows}
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        if _read_json(summary_path) != payload:
            raise RunEvaluationError("existing run summary differs from validated outer results")
    else:
        _write_json_new(summary_path, payload)
    csv_path = output_dir / "summary.csv"
    buffer = io.StringIO(newline="")
    fields = ["outer", "checkpoint_manifest_sha256", "eval_contract_sha256",
              "baseline_episode_weighted_sr", "trained_episode_weighted_sr",
              "episode_sr_delta_mean", "six_type_macro_delta_mean"]
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({"outer": row["outer"],
            "checkpoint_manifest_sha256": row["checkpoint_manifest_sha256"],
            "eval_contract_sha256": row["eval_contract_sha256"],
            "baseline_episode_weighted_sr": row["comparison"]["baseline_summary"]["episode_weighted_sr"],
            "trained_episode_weighted_sr": row["comparison"]["trained_summary"]["episode_weighted_sr"],
            "episode_sr_delta_mean": row["comparison"]["paired_episode_sr_delta_mean"],
            "six_type_macro_delta_mean": row["comparison"]["paired_six_type_macro_delta_mean"]})
    csv_payload = buffer.getvalue()
    if csv_path.exists():
        if csv_path.read_text(encoding="utf-8") != csv_payload:
            raise RunEvaluationError("existing summary table differs from validated outer results")
    else:
        with csv_path.open("x", encoding="utf-8", newline="") as stream:
            stream.write(csv_payload)


def evaluate_run(run_dir: str | Path, baseline_results: str | Path,
                 results_dir: str | Path | None = None, *, python: str | None = None) -> dict:
    run_path = Path(run_dir).expanduser().resolve(strict=True)
    baseline_path = Path(baseline_results).expanduser().resolve(strict=True)
    if not run_path.is_dir() or not baseline_path.is_file():
        raise RunEvaluationError("run directory or baseline merged result does not exist")
    manifest, run_id, baseline_contract = _manifest(run_path)
    baseline = _validate_baseline_merged(baseline_path, baseline_contract)
    index_path = run_path / "checkpoints" / "index.json"
    snapshots = _validate_index(run_path, run_id, manifest, manifest["contract_sha256"])
    output_dir = (Path(results_dir).expanduser().resolve() if results_dir is not None
                  else run_path / "evaluation")
    if output_dir == run_path or not output_dir.is_relative_to(run_path):
        raise RunEvaluationError("results directory must be an independent child of the run directory")
    checkpoints_path = (run_path / "checkpoints").resolve()
    if output_dir == checkpoints_path or output_dir.is_relative_to(checkpoints_path):
        raise RunEvaluationError("results directory cannot be inside the checkpoints tree")
    if baseline_path.parent == output_dir or baseline_path.parent.is_relative_to(output_dir) or \
            output_dir.is_relative_to(baseline_path.parent):
        raise RunEvaluationError("results directory cannot overlap the baseline results directory")
    if output_dir.exists():
        existing = list(output_dir.iterdir()) if output_dir.is_dir() else [output_dir]
        if existing:
            expected_dirs = [output_dir / f"outer_{n:02d}" for n in range(1, 5)]
            if any(not p.is_dir() for p in expected_dirs):
                raise RunEvaluationError("partial evaluation output exists; choose a new --results-dir")
            rows = []
            for snapshot, outer_dir in zip(snapshots, expected_dirs):
                contract_path = outer_dir / "contract.json"
                merged_path = outer_dir / "contract.merged.json"
                comparison_path = outer_dir / "comparison.json"
                if not all(p.is_file() for p in (contract_path, merged_path, comparison_path)):
                    raise RunEvaluationError("partial evaluation output exists; choose a new --results-dir")
                eval_contract, merged, comparison = _validate_run_merged(merged_path, contract_path,
                                                                          baseline, snapshot)
                rows.append(_outer_row(snapshot, eval_contract, merged, comparison))
            _write_summary(output_dir, run_id, baseline, baseline_path, index_path, rows)
            return {"run_id": run_id, "output_dir": str(output_dir), "outers": rows, "skipped": True}
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_python = python or os.environ.get("SKR_PYTHON") or sys.executable
    env = _clean_eval_environment(selected_python)
    eval_script = ROOT / "selfplay" / "evaluate.py"
    run_eval_script = ROOT / "selfplay" / "scripts" / "run_eval.sh"
    rows = []
    for snapshot in snapshots:
        outer_dir = output_dir / f"outer_{snapshot['outer']:02d}"
        if outer_dir.exists():
            raise RunEvaluationError(f"partial evaluation output exists; choose a new --results-dir: {outer_dir}")
        outer_dir.mkdir(parents=False)
        prepared_contract = outer_dir / "contract.json"
        args = [selected_python, str(eval_script), "prepare", "--config", baseline_contract["config_path"],
            "--alfworld-data", baseline_contract["alfworld_data_root"], "--base", baseline_contract["base_path"],
            "--checkpoint", str(snapshot["path"]), "--results-dir", str(outer_dir), "--output", str(prepared_contract)]
        _run_command(args, ROOT, env, outer_dir, "prepare")
        eval_contract = _read_json(prepared_contract)
        contract_payload = dict(eval_contract)
        contract_digest = contract_payload.pop("contract_sha256", None)
        if contract_digest != canonical_hash(contract_payload):
            raise RunEvaluationError(f"outer {snapshot['outer']} prepared contract checksum is invalid")
        if (eval_contract.get("protocol_signature") != baseline_contract.get("protocol_signature") or
                eval_contract.get("arm_identity", {}).get("kind") != "checkpoint" or
                Path(eval_contract["arm_identity"].get("path", "")).resolve() != snapshot["path"] or
                eval_contract["arm_identity"].get("manifest_sha256") != snapshot["checkpoint_manifest_sha256"]):
            raise RunEvaluationError(f"outer {snapshot['outer']} prepare emitted a mismatched evaluation contract")
        _run_command(["bash", str(run_eval_script), str(prepared_contract)], ROOT, env, outer_dir, "run_eval")
        merged_path = outer_dir / "contract.merged.json"
        merged = _read_json(merged_path)
        _validate_merged_against_contract(merged, eval_contract)
        comparison = compare(baseline, merged)
        comparison_path = outer_dir / "comparison.json"
        _run_command([selected_python, str(eval_script), "compare", "--baseline", str(baseline_path),
                      "--trained", str(merged_path), "--output", str(comparison_path)],
                     ROOT, env, outer_dir, "compare")
        if _read_json(comparison_path) != comparison:
            raise RunEvaluationError(f"outer {snapshot['outer']} compare command differs from validated rows")
        rows.append(_outer_row(snapshot, eval_contract, merged, comparison))
    _write_summary(output_dir, run_id, baseline, baseline_path, index_path, rows)
    return {"run_id": run_id, "output_dir": str(output_dir), "outers": rows, "skipped": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument("--results-dir")
    parser.add_argument("--python", default=None)
    args = parser.parse_args(argv)
    try:
        result = evaluate_run(args.run_dir, args.baseline_results, args.results_dir, python=args.python)
    except (RunEvaluationError, ContractError, OSError, ValueError) as exc:
        print(f"evaluate_run: {exc}", file=sys.stderr)
        return 1
    print(f"validated {len(result['outers'])} outer snapshots at {result['output_dir']} (skipped={result['skipped']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
