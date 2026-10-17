#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

usage() {
  echo "usage: run_experiment.sh --run-id ID --run-dir PATH --base PATH --alfworld-data PATH --baseline-contract PATH --baseline-results PATH [--backend vllm|hf] [--memory-profile default|qwen3_moe_cpu] [--resume CKPT] [--results-dir PATH]" >&2
}

run_id=""; run_dir=""; base=""; alfworld_data=""; baseline_contract=""
baseline_results=""; backend="hf"; memory_profile="default"; resume=""; results_dir=""
while (($#)); do
  case "$1" in
    --run-id) run_id="${2:?missing --run-id value}"; shift 2 ;;
    --run-dir) run_dir="${2:?missing --run-dir value}"; shift 2 ;;
    --base) base="${2:?missing --base value}"; shift 2 ;;
    --alfworld-data) alfworld_data="${2:?missing --alfworld-data value}"; shift 2 ;;
    --baseline-contract) baseline_contract="${2:?missing --baseline-contract value}"; shift 2 ;;
    --baseline-results) baseline_results="${2:?missing --baseline-results value}"; shift 2 ;;
    --backend) backend="${2:?missing --backend value}"; shift 2 ;;
    --memory-profile) memory_profile="${2:?missing --memory-profile value}"; shift 2 ;;
    --resume) resume="${2:?missing --resume value}"; shift 2 ;;
    --results-dir) results_dir="${2:?missing --results-dir value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$run_id" || -z "$run_dir" || -z "$base" || -z "$alfworld_data" ||
      -z "$baseline_contract" || -z "$baseline_results" ]]; then
  usage
  exit 2
fi
if [[ "$backend" != "vllm" && "$backend" != "hf" ]]; then
  echo "--backend must be vllm or hf" >&2
  exit 2
fi
if [[ "$memory_profile" != "default" && "$memory_profile" != "qwen3_moe_cpu" ]]; then
  echo "unknown --memory-profile" >&2
  exit 2
fi
if [[ "$memory_profile" == "qwen3_moe_cpu" && "$backend" != "hf" ]]; then
  echo "qwen3_moe_cpu requires --backend hf" >&2
  exit 2
fi
if [[ "${WORLD_SIZE:-1}" != "1" || "${SKR_WORKER_MODE:-0}" == "1" ]]; then
  echo "run_experiment.sh must be launched from the outer scheduler shell, not a torchrun rank or worker" >&2
  exit 2
fi
if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]]; then
  echo "invalid --run-id" >&2
  exit 2
fi

python_bin="${SKR_PYTHON:-$(command -v python)}"
if [[ ! -x "$python_bin" ]]; then
  echo "selected SKR_PYTHON is not executable: $python_bin" >&2
  exit 2
fi
export SKR_PYTHON="$python_bin"
export SKR_PROJECT_ROOT="$PROJECT_ROOT"

# Validate all paths and the run-id-derived training directory before GPU training.
"$python_bin" - "$PROJECT_ROOT" "$run_id" "$run_dir" "$resume" "$results_dir" "$baseline_contract" "$baseline_results" <<'PY'
import sys
from pathlib import Path
import yaml

root = Path(sys.argv[1]).resolve(strict=True)
run_id = sys.argv[2]
provided_run = Path(sys.argv[3]).expanduser().resolve()
resume_arg = sys.argv[4]
results_arg = sys.argv[5]
contract_path = Path(sys.argv[6]).expanduser().resolve(strict=True)
baseline_path = Path(sys.argv[7]).expanduser().resolve(strict=True)
if not contract_path.is_file() or contract_path.stat().st_size == 0:
    raise SystemExit("baseline contract is missing or empty")
if not baseline_path.is_file() or baseline_path.stat().st_size == 0:
    raise SystemExit("baseline merged result is missing or empty")
config = yaml.safe_load((root / "selfplay" / "config.yaml").read_text(encoding="utf-8"))
output_root = Path(config["run"]["output_dir"])
if not output_root.is_absolute():
    output_root = (root / "selfplay" / output_root).resolve()
expected_run = (output_root / f"{config['run']['name']}-{run_id}").resolve()
if provided_run != expected_run:
    raise SystemExit(f"--run-dir must match resolved training run path: {expected_run}")
if resume_arg:
    resume = Path(resume_arg).expanduser().resolve(strict=True)
    if resume.name not in {"ckpt", "ckpt.previous"} or resume.parent != expected_run:
        raise SystemExit("--resume must be ckpt or ckpt.previous inside the selected run directory")
else:
    if expected_run.exists():
        raise SystemExit("run directory already exists; refusing to overwrite or resume implicitly")
results = (Path(results_arg).expanduser().resolve() if results_arg else expected_run / "evaluation")
if results == expected_run or not results.is_relative_to(expected_run):
    raise SystemExit("--results-dir must be an independent subdirectory inside --run-dir")
checkpoints = (expected_run / "checkpoints").resolve()
if results == checkpoints or results.is_relative_to(checkpoints):
    raise SystemExit("--results-dir cannot be inside the checkpoints tree")
if results.exists() and (not results.is_dir() or any(results.iterdir())):
    raise SystemExit("evaluation output already exists and is nonempty; select a fresh --results-dir")
PY

# Validate the existing baseline pair before starting the expensive training run.
"$python_bin" - "$PROJECT_ROOT" "$baseline_contract" "$baseline_results" "$memory_profile" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / "selfplay"))
from evaluate_run import validate_baseline_files
contract, _merged = validate_baseline_files(sys.argv[2], sys.argv[3])
if sys.argv[4] == "qwen3_moe_cpu":
    signature = contract.get("protocol_signature", {})
    if signature.get("backend") != "hf" or signature.get("solver_settings", {}).get("prompt_format") != "chat_template":
        raise SystemExit("30B requires a new matched HF/chat_template baseline from the current config")
PY

train_args=(--base "$base" --alfworld-data "$alfworld_data" \
  --baseline-contract "$baseline_contract" --backend "$backend" --run-id "$run_id" \
  --memory-profile "$memory_profile")
if [[ -n "$resume" ]]; then
  train_args+=(--resume "$resume")
fi
bash "$SCRIPT_DIR/run_train.sh" "${train_args[@]}"

eval_args=(--run-dir "$run_dir" --baseline-results "$baseline_results" --python "$python_bin")
if [[ -n "$results_dir" ]]; then
  eval_args+=(--results-dir "$results_dir")
fi
"$python_bin" "$PROJECT_ROOT/selfplay/evaluate_run.py" "${eval_args[@]}"
