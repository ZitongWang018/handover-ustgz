#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
PROJECT_ROOT="${SKR_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_ROOT"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

if [[ "${SKR_WORKER_MODE:-0}" == "1" ]]; then
  if [[ "${1:-}" != "--worker" ]]; then
    echo "worker invocation is missing --worker" >&2
    exit 2
  fi
  shift
  IFS=',' read -r -a devices <<< "${SKR_PARENT_CUDA_MASK:?missing assigned GPU mask}"
  local_rank="${LOCAL_RANK:?torchrun did not set LOCAL_RANK}"
  if [[ "${#devices[@]}" -ne 8 || "$local_rank" -lt 0 || "$local_rank" -ge 8 ]]; then
    echo "expected exactly 8 assigned GPUs and LOCAL_RANK in [0,7]" >&2
    exit 2
  fi
  selected="${devices[$local_rank]}"
  if [[ -z "$selected" ]]; then
    echo "empty physical GPU mapping for rank $local_rank" >&2
    exit 2
  fi
  export CUDA_VISIBLE_DEVICES="$selected"
  export LOCAL_RANK=0
  python_bin="${SKR_PYTHON:-$(command -v python)}"
  exec "$python_bin" selfplay/src/ez_train_v2.py "$@"
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  parent_mask="$CUDA_VISIBLE_DEVICES"
else
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "set CUDA_VISIBLE_DEVICES to eight GPUs or provide nvidia-smi" >&2
    exit 2
  fi
  parent_mask="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
fi
IFS=',' read -r -a assigned <<< "$parent_mask"
if [[ "${#assigned[@]}" -ne 8 ]]; then
  echo "eight-GPU training requires exactly 8 assigned devices; got ${#assigned[@]}" >&2
  exit 2
fi
declare -A seen=()
for device in "${assigned[@]}"; do
  if [[ -z "$device" || -n "${seen[$device]:-}" ]]; then
    echo "GPU mask contains an empty or repeated device: $parent_mask" >&2
    exit 2
  fi
  seen[$device]=1
done

export SKR_PARENT_CUDA_MASK="$parent_mask"
export SKR_WORKER_MODE=1
python_bin="${SKR_PYTHON:-$(command -v python)}"
exec "$python_bin" -m torch.distributed.run --standalone --nnodes=1 \
  --nproc_per_node=8 --max_restarts=0 --no_python "$SCRIPT_PATH" --worker "$@"
