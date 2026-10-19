"""Exercise the actual orchestration shell with CPU stubs, never GPU jobs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

_GIT_BASH = Path("C:/Program Files/Git/bin/bash.exe")
BASH = str(_GIT_BASH) if _GIT_BASH.is_file() else shutil.which("bash")
pytestmark = pytest.mark.skipif(not BASH, reason="bash is required for launcher wiring check")


@pytest.mark.parametrize("scenario", ["success", "vllm_baseline", "wrong_baseline", "train_failure"])
def test_30b_experiment_forwards_profile_and_stops_on_failure(tmp_path, scenario):
    root = tmp_path / "repo"
    scripts = root / "selfplay" / "scripts"
    scripts.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "scripts" / "run_experiment.sh"
    shutil.copyfile(source, scripts / "run_experiment.sh")
    (root / "selfplay" / "config.yaml").write_text(
        'run:\n  name: explorer-zero\n  output_dir: runs\n', encoding="utf-8")
    (scripts / "run_train.sh").write_text('''#!/usr/bin/env bash
"$SKR_PYTHON" - "$@" <<'PY'
import json,os,pathlib,sys
pathlib.Path("train-args.json").write_text(json.dumps(sys.argv[1:]))
raise SystemExit(7 if os.environ.get("FAIL_TRAIN") else 0)
PY
''', encoding="utf-8", newline="\n")
    (root / "selfplay" / "evaluate_run.py").write_text('''import json,pathlib,sys
def validate_baseline_files(a,b):
    return json.loads(pathlib.Path(a).read_text()),json.loads(pathlib.Path(b).read_text())
if __name__ == "__main__":
    pathlib.Path("eval-args.json").write_text(json.dumps(sys.argv[1:]))
''', encoding="utf-8")
    baseline = root / "contract.json"
    baseline.write_text(json.dumps({"protocol_signature": {
        "backend": "vllm" if scenario == "vllm_baseline" else "hf",
        "solver_settings": {"prompt_format": "raw" if scenario == "wrong_baseline" else "chat_template"}}}), encoding="utf-8")
    merged = root / "merged.json"
    merged.write_text("{}", encoding="utf-8")
    run = root / "selfplay" / "runs" / "explorer-zero-check"
    env = dict(os.environ, SKR_PYTHON=Path(sys.executable).as_posix(), WORLD_SIZE="1", SKR_WORKER_MODE="0")
    env.pop("FAIL_TRAIN", None)
    if scenario == "train_failure": env["FAIL_TRAIN"] = "1"
    args = [BASH, str(scripts / "run_experiment.sh"), "--run-id", "check",
        "--run-dir", str(run), "--base", str(root), "--alfworld-data", str(root),
        "--baseline-contract", str(baseline), "--baseline-results", str(merged),
        "--memory-profile", "qwen3_moe_cpu", "--backend", "hf"]
    result = subprocess.run(args, cwd=root, env=env, capture_output=True, text=True, timeout=40)
    if scenario == "wrong_baseline":
        assert result.returncode != 0 and "chat_template" in result.stderr
        assert not (root / "train-args.json").exists()
    else:
        forwarded = json.loads((root / "train-args.json").read_text())
        assert forwarded[forwarded.index("--memory-profile") + 1] == "qwen3_moe_cpu"
        assert forwarded[forwarded.index("--backend") + 1] == "hf"
        assert (root / "eval-args.json").exists() == (scenario in {"success", "vllm_baseline"})
        assert result.returncode == (7 if scenario == "train_failure" else 0), result.stderr
