"""CPU-only regressions for model metadata inventory and matched control plans."""
from __future__ import annotations

import builtins
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.append(str(ROOT / "scripts"))

from ez_model_identity import inspect_model_identity
from ez_eval_controls import generate_plan
import ez_eval_controls


def _model_dir(path: Path, *, config_name="/models/qwen3_4b") -> Path:
    path.mkdir()
    (path / "config.json").write_text(json.dumps({
        "_name_or_path": config_name,
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "torch_dtype": "bfloat16",
        "eos_token_id": 151645,
        "max_position_embeddings": 40960,
    }), encoding="utf-8")
    (path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": 151645}), encoding="utf-8")
    (path / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "Qwen3Tokenizer",
        "chat_template": "template",
    }), encoding="utf-8")
    (path / "README.md").write_text(
        "---\nlicense_link: https://huggingface.co/Qwen/Qwen3-4B\n"
        "base_model:\n- Qwen/Qwen3-4B-Base\n---\n# Qwen3-4B\n",
        encoding="utf-8")
    (path / "model-00001.safetensors").write_bytes(b"small mock tensor bytes")
    (path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"x": "model-00001.safetensors",
                       "unsafe": "../outside.safetensors"}
    }), encoding="utf-8")
    return path


def test_identity_is_stdlib_metadata_only_and_does_not_claim_base_weights(tmp_path,
                                                                           monkeypatch):
    model = _model_dir(tmp_path / "model")
    original_import = builtins.__import__

    def no_ml_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "safetensors"}:
            raise AssertionError(f"unexpected ML import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_ml_import)
    original_open = Path.open

    def no_weight_read(self, *args, **kwargs):
        if self.name.endswith(".safetensors"):
            raise AssertionError("default identity must not read model weights")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_weight_read)
    identity = inspect_model_identity(model)
    assert identity["status"] == "weights_unverified"
    assert identity["weights_verified"] is False
    assert identity["provenance_is_not_weight_proof"] is True
    assert identity["provenance_hints"]["readme"]["declared_model_heading"] == "Qwen3-4B"
    assert identity["provenance_hints"]["readme"]["ancestor_base_models"] == [
        "Qwen/Qwen3-4B-Base"]
    assert identity["provenance_hints"]["readme"]["license_link"].endswith("Qwen3-4B")
    assert identity["provenance_declaration_ambiguous"] is True
    assert identity["provenance_declaration_conflict"] is False
    assert identity["config_token_boundary_fields"] == {
        "eos_token_id": 151645, "max_position_embeddings": 40960}
    assert [w["filename"] for w in identity["weights"]] == ["model-00001.safetensors"]
    assert "sha256" not in identity["weights"][0]
    assert "Qwen3-4B-Base" not in json.dumps(identity.get("readme", {}))


def test_identity_reports_missing_paths_and_only_flags_explicit_repo_conflict(tmp_path):
    missing = inspect_model_identity("Qwen/Qwen3-4B")
    assert missing["status"] == "metadata_unavailable"
    assert "no network lookup" in missing["reason"]
    model = _model_dir(tmp_path / "conflict", config_name="org/different-model")
    identity = inspect_model_identity(model)
    assert identity["provenance_declaration_conflict"] is True
    assert identity["status"] == "weights_unverified"


def test_optional_weight_hash_reads_only_basename_weight_files(tmp_path):
    model = _model_dir(tmp_path / "model")
    weight = model / "model-00001.safetensors"
    identity = inspect_model_identity(model, hash_weights=True)
    assert identity["weights_hash_requested"] is True
    assert identity["weights"][0]["sha256"] == hashlib.sha256(weight.read_bytes()).hexdigest()
    # Untrusted index paths cannot make the inspector read outside its model directory.
    assert len(identity["weights"]) == 1


def test_identity_skips_oversized_json_and_eval_handles_metadata_permission_error(
        tmp_path, monkeypatch):
    model = _model_dir(tmp_path / "large")
    (model / "config.json").write_bytes(b"{" + b" " * (8 * 1024 * 1024 + 10))
    identity = inspect_model_identity(model)
    assert identity["architecture"] is None
    assert identity["metadata_sha256"]["config.json"]["truncated"] is True
    import eval_ez
    monkeypatch.setattr(eval_ez, "inspect_model_identity",
                        lambda path: (_ for _ in ()).throw(PermissionError("denied")))
    result = eval_ez._safe_model_identity("restricted-model")
    assert result["status"] == "metadata_unavailable"
    assert result["reason"] == "local metadata I/O failed: PermissionError"


def test_control_plan_is_manual_matched_and_quotes_paths(tmp_path):
    out = tmp_path / "plan path $dollar"
    plan = generate_plan(base="/models/base model/$x", data="/data/ALFWorld set",
                         out_dir=str(out), ckpt="/checkpoints/solver ckpt")
    assert not (out / "plan.json").exists()  # pure plan builder; CLI writes artifacts
    assert plan["execution_policy"].startswith("manual_only")
    assert plan["expected_task_count"] == 134 and plan["expected_shards"] == 8
    assert len(plan["jobs"]) == 8
    by_key = {(job["arm"], job["model"]): job for job in plan["jobs"]}
    a = by_key[("paper_no_menu", "base")]
    b = by_key[("json_menu", "base")]
    c = by_key[("json_menu_50", "base")]
    d = by_key[("public_rich", "base")]
    assert a["expected_protocol_effective"] == "paper-approx"
    assert b["expected_protocol_effective"] == "custom"
    assert c["expected_protocol_effective"] == "custom"
    assert d["expected_protocol_effective"] == "public-rich"
    assert {**a["settings"], "menu": "on"} == b["settings"]
    assert {**b["settings"], "steps": 50} == c["settings"]
    assert d["settings"]["top_p"] == 0.95
    for first, second, allowed in ((a, b, {"EZ_ACTION_MENU", "EZ_TAG"}),
                                   (b, c, {"EZ_MAX_STEPS", "EZ_TAG"})):
        changed = {k for k in first["environment"]
                   if first["environment"][k] != second["environment"][k]}
        assert changed == allowed
    base_job, solver_job = by_key[("json_menu", "base")], by_key[("json_menu", "solver")]
    assert base_job["settings"] == solver_job["settings"]
    assert base_job["environment"]["ZTWANG_MODEL_PATH"] == solver_job["environment"]["ZTWANG_MODEL_PATH"]
    differing = {k for k in base_job["environment"]
                 if base_job["environment"][k] != solver_job["environment"][k]}
    assert differing == {"EZ_BASE_ONLY", "EZ_CKPT", "EZ_TAG"}
    argv = shlex.split(a["shell"])
    assert argv == a["argv"]
    assert "ZTWANG_MODEL_PATH=/models/base model/$x" in argv
    assert "ALFWORLD_DATA=/data/ALFWorld set" in argv
    assert all(name in argv for name in ("-u", "EZ_SOLVER_STYLE", "EZ_SOLVER_FEWSHOT",
                                         "EZ_SOLVER_STRICT_RESOLVE", "EZ_MAX_NEW"))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        generate_plan(base="base", data="data", out_dir=str(out))


def test_control_plan_cli_writes_only_reviewable_manual_artifacts(tmp_path, monkeypatch,
                                                                  capsys):
    out = tmp_path / "new-plan"
    monkeypatch.setattr(sys, "argv", ["ez_eval_controls.py", "--base", "base model",
        "--alfworld-data", "alfworld data", "--out-dir", str(out),
        "--arms", "paper_no_menu", "json_menu"])
    ez_eval_controls.main()
    assert sorted(path.name for path in out.iterdir()) == ["plan.json", "run.sh"]
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    shell = (out / "run.sh").read_text(encoding="utf-8")
    assert plan["arms"] == ["paper_no_menu", "json_menu"]
    assert "manual_only" in plan["execution_policy"]
    assert "sbatch" in shell
    assert shlex.quote("ZTWANG_MODEL_PATH=base model") in shell
    assert shlex.quote("ALFWORLD_DATA=alfworld data") in shell
    assert "set -euo pipefail" in shell
    assert "wrote manual-only plan files" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        ez_eval_controls.main()
    assert exc.value.code == 2


def test_control_plan_script_process_writes_artifacts_and_rejects_overwrite(tmp_path):
    out = tmp_path / "process-plan"
    script = ROOT / "scripts" / "ez_eval_controls.py"
    command = [sys.executable, str(script), "--base", "base path",
               "--alfworld-data", "alfworld path", "--out-dir", str(out),
               "--arms", "public_rich"]
    first = subprocess.run(command, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    assert (out / "plan.json").is_file() and (out / "run.sh").is_file()
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    assert plan["jobs"][0]["settings"]["protocol"] == "public-rich"
    assert plan["jobs"][0]["expected_episode_count"] == 134
    assert plan["jobs"][0]["expected_shards"] == 8
    # Only the Python plan generator was launched; run.sh stays inert for manual review.
    second = subprocess.run(command, capture_output=True, text=True, check=False)
    assert second.returncode == 2
    assert "refusing to overwrite existing output directory" in second.stderr
