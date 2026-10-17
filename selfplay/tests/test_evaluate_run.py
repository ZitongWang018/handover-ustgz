from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "selfplay" / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "selfplay"))
import evaluate_run
from evaluation import canonical_hash, compare, merge


def _baseline(tmp_path, chat=False):
    spec = importlib.util.spec_from_file_location("evaluation_fixture_local",
        Path(__file__).with_name("test_evaluation.py"))
    fixture_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture_module)
    contract, shards = fixture_module._fixture(tmp_path)
    if chat:
        from protocol import solver_prompt_identity
        signature = contract["protocol_signature"]
        signature.update(solver_prompt_identity(fixture_module.TinyTokenizer(), "chat_template"))
        signature["backend"] = "hf"
        signature["solver_settings"] = {"prompt_format": "chat_template"}
    contract["run_id"] = "base-run"
    config_path = tmp_path / "eval-config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    base_path = tmp_path / "base-model"
    base_path.mkdir()
    contract["config_path"] = str(config_path)
    contract["base_path"] = str(base_path)
    contract["alfworld_data_root"] = str(tmp_path)
    # Rows/shards are valid and don't bind run_id; sign the copied base contract.
    contract["contract_sha256"] = canonical_hash({k: v for k, v in contract.items() if k != "contract_sha256"})
    for shard in shards:
        shard["run_id"] = "base-run"
        shard["contract_sha256"] = contract["contract_sha256"]
    return contract, shards, merge(contract, shards)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _install_fake_pipeline(monkeypatch, tmp_path, *, fail_at=None, real_snapshots=False, chat=False):
    contract, shards, baseline = _baseline(tmp_path, chat=chat)
    baseline_path = tmp_path / "base" / "contract.merged.json"
    _write(baseline_path, baseline)
    run_dir = tmp_path / ("snaprun-train-run" if real_snapshots else "run")
    run_dir.mkdir()
    (run_dir / "checkpoints").mkdir()
    snapshots = []
    if real_snapshots:
        from checkpoint import contract_signature
        from training_snapshots import publish_outer_snapshot
        spec = importlib.util.spec_from_file_location("snap_fixture_local",
            Path(__file__).with_name("test_training_snapshots.py"))
        snap_fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(snap_fixture)
        manifest = snap_fixture._manifest(tmp_path)
        manifest["checkpoint_eligibility"] = "formal_training"
        manifest["resolved_config"]["train"]["outer_loops"] = 4
        manifest["baseline_binding"] = {"contract": contract,
            "contract_sha256": contract["contract_sha256"],
            "common_identity": {"base_inventory_sha256":
                contract["protocol_signature"]["base_inventory_sha256"]}}
        manifest["contract_sha256"] = contract_signature(manifest)
        _write(run_dir / "training_manifest.json", manifest)
        _write(run_dir / "run_protocol.json", {"contract_sha256": manifest["contract_sha256"],
                                                 "run_id": "train-run"})
        for outer in range(1, 5):
            snap_fixture._checkpoint(run_dir, manifest, outer)
            publish_outer_snapshot(run_dir, outer, manifest)
        index = json.loads((run_dir / "checkpoints" / "index.json").read_text())
        for entry in index["snapshots"]:
            snapshots.append({"outer": entry["outer"], "path": run_dir / entry["path"],
                "checkpoint_manifest_sha256": canonical_hash(manifest),
                "inventory_sha256": entry["inventory_sha256"]})
        monkeypatch.setattr(evaluate_run, "_verify_current_baseline_identity", lambda *_args: None)
    else:
        manifest = {"contract_sha256": "a" * 64}
        (run_dir / "checkpoints" / "index.json").write_text("{}", encoding="utf-8")
        for outer in range(1, 5):
            checkpoint = run_dir / "checkpoints" / f"outer_{outer:02d}"
            checkpoint.mkdir()
            snapshots.append({"outer": outer, "path": checkpoint,
                "checkpoint_manifest_sha256": f"{outer:064x}", "inventory_sha256": f"{outer:064x}"})
        monkeypatch.setattr(evaluate_run, "_manifest", lambda _path: (manifest, "train-run", contract))
        monkeypatch.setattr(evaluate_run, "_validate_index", lambda *_args: snapshots)
    calls = []

    class Result:
        returncode = 0

    def fake_run(argv, *, cwd, env, stdout, stderr, text, check):
        calls.append((Path(argv[1]).name if len(argv) > 1 else argv[0], list(argv), dict(env)))
        is_prepare = "prepare" in argv
        is_eval = "run_eval.sh" in argv[1]
        is_compare = "compare" in argv
        outer = int(Path(argv[-1] if is_prepare else argv[-1]).parent.name[-2:])
        if fail_at == (outer, "run_eval") and is_eval:
            stderr.write("synthetic rollout failure\n")
            return type("Failed", (), {"returncode": 17})()
        if is_prepare:
            output = Path(argv[argv.index("--output") + 1])
            eval_contract = copy.deepcopy(contract)
            eval_contract["arm_identity"] = {"kind": "checkpoint",
                "path": str(snapshots[outer - 1]["path"]),
                "manifest_sha256": snapshots[outer - 1]["checkpoint_manifest_sha256"],
                "base_inventory_sha256": contract["arm_identity"]["base_inventory_sha256"]}
            eval_contract["contract_sha256"] = canonical_hash({k: v for k, v in eval_contract.items()
                                                                 if k != "contract_sha256"})
            _write(output, eval_contract)
        elif is_eval:
            eval_contract = json.loads((Path(argv[-1]).parent / "contract.json").read_text())
            eval_shards = copy.deepcopy(shards)
            for shard in eval_shards:
                shard.update(run_id=eval_contract["run_id"],
                    contract_sha256=eval_contract["contract_sha256"],
                    arm_identity=eval_contract["arm_identity"])
            # merge requires contract and each rank record to carry the same eval identity.
            merged = merge(eval_contract, eval_shards)
            _write(Path(argv[-1]).parent / "contract.merged.json", merged)
        elif is_compare:
            base_rows = json.loads(Path(argv[argv.index("--baseline") + 1]).read_text())
            trained_rows = json.loads(Path(argv[argv.index("--trained") + 1]).read_text())
            _write(Path(argv[argv.index("--output") + 1]), compare(base_rows, trained_rows))
        stdout.write("ok\n")
        return Result()

    monkeypatch.setattr(evaluate_run.subprocess, "run", fake_run)
    return run_dir, baseline_path, calls


def test_runs_four_snapshots_serially_with_selected_python_and_skips_complete(tmp_path, monkeypatch):
    run_dir, baseline_path, calls = _install_fake_pipeline(monkeypatch, tmp_path)
    first = evaluate_run.evaluate_run(run_dir, baseline_path, python="/python/selected")
    assert first["skipped"] is False
    assert [name for name, _, _ in calls] == ["evaluate.py", "run_eval.sh", "evaluate.py"] * 4
    assert all(argv[0] == "/python/selected" for name, argv, _ in calls if name == "evaluate.py")
    assert all(env["SKR_PYTHON"] == "/python/selected" for _, _, env in calls)
    assert [row["outer"] for row in first["outers"]] == [1, 2, 3, 4]
    calls.clear()
    second = evaluate_run.evaluate_run(run_dir, baseline_path, python="/python/selected")
    assert second["skipped"] is True
    assert calls == []


def test_refuses_partial_outputs_and_bad_baseline_before_subprocess(tmp_path, monkeypatch):
    run_dir, baseline_path, calls = _install_fake_pipeline(monkeypatch, tmp_path)
    partial = run_dir / "evaluation"
    partial.mkdir()
    (partial / "outer_01").mkdir()
    with pytest.raises(evaluate_run.RunEvaluationError, match="partial evaluation"):
        evaluate_run.evaluate_run(run_dir, baseline_path)
    assert calls == []

    baseline = json.loads(baseline_path.read_text())
    baseline["rows"].pop()
    _write(baseline_path, baseline)
    with pytest.raises(evaluate_run.RunEvaluationError, match="grid|summary"):
        evaluate_run.evaluate_run(run_dir, baseline_path, results_dir=run_dir / "evaluation_retry")
    assert calls == []


def test_command_failure_keeps_live_logs_and_stops_later_outers(tmp_path, monkeypatch):
    run_dir, baseline_path, calls = _install_fake_pipeline(monkeypatch, tmp_path, fail_at=(2, "run_eval"))
    with pytest.raises(evaluate_run.RunEvaluationError, match="exit code 17"):
        evaluate_run.evaluate_run(run_dir, baseline_path)
    assert len(calls) == 5
    assert not (run_dir / "evaluation" / "outer_03").exists()
    stderr = run_dir / "evaluation" / "outer_02" / "run_eval.stderr.log"
    assert "synthetic rollout failure" in stderr.read_text(encoding="utf-8")
    records = (run_dir / "evaluation" / "outer_02" / "commands.jsonl").read_text(encoding="utf-8")
    assert '"status": "started"' in records and '"returncode": 17' in records


def test_command_log_collision_rejected_before_child_start(tmp_path, monkeypatch):
    logdir = tmp_path / "logs"
    logdir.mkdir()
    (logdir / "prepare.stdout.log").write_text("keep", encoding="utf-8")
    invoked = []
    monkeypatch.setattr(evaluate_run.subprocess, "run", lambda *a, **k: invoked.append(True))
    with pytest.raises(evaluate_run.RunEvaluationError, match="overwrite command log"):
        evaluate_run._run_command(["python", "x"], tmp_path, {}, logdir, "prepare")
    assert invoked == []
    assert (logdir / "prepare.stdout.log").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("chat", [False, True])
def test_real_manifest_baseline_and_four_sealed_snapshots_reach_orchestration(tmp_path, monkeypatch, chat):
    run_dir, baseline_path, calls = _install_fake_pipeline(monkeypatch, tmp_path, real_snapshots=True, chat=chat)
    result = evaluate_run.evaluate_run(run_dir, baseline_path)
    assert [row["outer"] for row in result["outers"]] == [1, 2, 3, 4]
    assert len(calls) == 12
    if chat:
        baseline = json.loads(baseline_path.read_text())
        trained = json.loads((run_dir / "evaluation" / "outer_01" / "contract.merged.json").read_text())
        assert baseline["protocol_signature"]["backend"] == "hf"
        for arm in (baseline, trained):
            arm["protocol_signature"].pop("chat_template_sha256")
        with pytest.raises(ValueError, match="chat template identity"):
            compare(baseline, trained)


@pytest.mark.parametrize("damage", ["missing_fourth", "wrong_cursor", "bad_marker"])
def test_real_snapshot_index_damage_fails_before_any_child(tmp_path, monkeypatch, damage):
    run_dir, baseline_path, calls = _install_fake_pipeline(monkeypatch, tmp_path, real_snapshots=True)
    index_path = run_dir / "checkpoints" / "index.json"
    index = json.loads(index_path.read_text())
    if damage == "missing_fourth":
        index["snapshots"].pop()
        _write(index_path, index)
    elif damage == "wrong_cursor":
        index["snapshots"][1]["next_position"] = {"outer": 2, "phase": "solver", "step": 0}
        _write(index_path, index)
    else:
        index["snapshots"][2]["complete_marker_sha256"] = "f" * 64
        _write(index_path, index)
    with pytest.raises(evaluate_run.RunEvaluationError):
        evaluate_run.evaluate_run(run_dir, baseline_path)
    assert calls == []
