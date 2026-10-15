#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: run_eval.sh CONTRACT.json" >&2
  exit 2
fi
CONTRACT=$(realpath "$1")
PROJECT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${SKR_PYTHON:-python}

mapfile -t SHARDS < <("$PYTHON" - "$CONTRACT" <<'PY'
import json, sys
contract = json.load(open(sys.argv[1], encoding="utf-8"))
for shard in contract["expected_shards"]:
    print(f'{shard["rank"]}\t{shard["path"]}')
PY
)
N=${#SHARDS[@]}
if [[ "$N" -lt 1 || "$N" -gt 8 ]]; then
  echo "prepared contract must contain 1 to 8 nonempty ranks; found $N" >&2
  exit 2
fi

if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
  for gpu in "${GPUS[@]}"; do
    if [[ -z "$gpu" || "$gpu" == "-1" ]]; then
      echo "CUDA_VISIBLE_DEVICES is empty or disables GPUs" >&2
      exit 2
    fi
  done
else
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
if [[ "${#GPUS[@]}" -lt "$N" ]]; then
  echo "contract needs $N GPUs but only ${#GPUS[@]} are available" >&2
  exit 2
fi
declare -A SEEN_GPU=()
for ((i=0; i<${#GPUS[@]}; i++)); do
  gpu=${GPUS[$i]}
  if [[ -z "$gpu" || "$gpu" == "-1" || -n "${SEEN_GPU[$gpu]:-}" ]]; then
    echo "GPU assignment contains an empty, disabled, or duplicate device" >&2
    exit 2
  fi
  SEEN_GPU[$gpu]=1
done

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
pids=()
cleanup() {
  local pid
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM
failed=0
for entry in "${SHARDS[@]}"; do
  IFS=$'\t' read -r rank output <<< "$entry"
  mkdir -p "$(dirname "$output")"
  CUDA_VISIBLE_DEVICES="${GPUS[$rank]}" "$PYTHON" "$PROJECT/evaluate.py" run \
    --contract "$CONTRACT" --rank "$rank" --output "$output" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
pids=()
if [[ "$failed" -ne 0 ]]; then
  echo "one or more evaluator ranks failed; merge was skipped" >&2
  exit 1
fi
"$PYTHON" "$PROJECT/evaluate.py" merge --contract "$CONTRACT" \
  --output "${CONTRACT%.json}.merged.json"
