#!/usr/bin/env bash
# Portable 8-GPU launcher. It uses already-installed software, model, and data.
set -euo pipefail

PROJ="${SKR_PROJECT_ROOT:-}"
if [[ -z "$PROJ" || ! -d "$PROJ/skillrise" ]]; then
  echo "Set SKR_PROJECT_ROOT to the repository root containing skillrise/." >&2
  exit 2
fi
cd "$PROJ"

if [[ -n "${SKR_CONDA_BASE:-}" ]]; then
  if [[ -z "${SKR_CONDA_ENV:-}" || ! -f "$SKR_CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    echo "SKR_CONDA_BASE requires SKR_CONDA_ENV and conda.sh under that base." >&2
    exit 2
  fi
  source "$SKR_CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$SKR_CONDA_ENV"
fi
PYTHON="${SKR_PYTHON:-python}"
command -v "$PYTHON" >/dev/null 2>&1 || { echo "Python not found: $PYTHON" >&2; exit 2; }
if [[ -z "${ALFWORLD_DATA:-}" || ! -d "$ALFWORLD_DATA" ]]; then
  echo "Set ALFWORLD_DATA to the existing ALFWorld dataset directory." >&2
  exit 2
fi
if [[ -z "${ZTWANG_MODEL_PATH:-}" || ! -d "$ZTWANG_MODEL_PATH" ]]; then
  echo "Set ZTWANG_MODEL_PATH to the existing Qwen3-4B model directory." >&2
  exit 2
fi

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export HF_HUB_OFFLINE=1
# ★ 不要把 py310deps 放进 PYTHONPATH：它含 numpy 源码目录，会遮蔽 site-packages
#   的 numpy 并抛 "should not try to import numpy from its source directory"。
#   改由 skillrise/_deps/sitecustomize.py 在 site-packages 就位后 append 到 sys.path 末尾。
export SKR_PY310DEPS="$PROJ/selfplay/third_party/py310deps"
export PYTHONPATH="$PROJ/selfplay/src:$PROJ/skillrise/_deps:${PYTHONPATH:-}"

# Respect a scheduler/container-assigned device mask. Each worker sees exactly
# one entry from that mask, while rank-to-device ordering stays deterministic.
GPU_MASK="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_DEVICES <<< "$GPU_MASK"
if [[ ${#GPU_DEVICES[@]} -ne 8 ]]; then
  echo "Expected exactly eight comma-separated CUDA devices, got: $GPU_MASK" >&2
  exit 2
fi
for device in "${GPU_DEVICES[@]}"; do
  if [[ -z "$device" ]]; then
    echo "CUDA_VISIBLE_DEVICES contains an empty device entry: $GPU_MASK" >&2
    exit 2
  fi
done

STYLE="${EZ_STYLE:-react}"
case "$STYLE" in react|zero_shot) ;; *) echo "Unsupported EZ_STYLE=$STYLE" >&2; exit 2 ;; esac
SELECTION="${EZ_SELECTION_MANIFEST:-}"
if [[ -z "$SELECTION" || ! -f "$SELECTION" ]]; then
  echo "Set EZ_SELECTION_MANIFEST to a CPU-prepared fixed 128-slot manifest." >&2
  exit 2
fi
RUN_ID="${EZ_RUN_ID:-${SLURM_JOB_ID:-local}-$(date +%Y%m%dT%H%M%S)-${STYLE}}"
OUT="${EZ_OUT_DIR:-skillrise/out/${RUN_ID}}"
if [[ -e "$OUT" ]]; then echo "Refusing existing output directory: $OUT" >&2; exit 2; fi
mkdir -p "$OUT" logs
echo "run_id=$RUN_ID profile=upstream-grpo-validation style=$STYLE model=$ZTWANG_MODEL_PATH split=valid_seen slots=128 samples=1"
echo "selection=$SELECTION output=$OUT"
nvidia-smi -L

# The diagnostic has a fresh engine so its seeded draws cannot advance a formal rank.
CUDA_VISIBLE_DEVICES="${GPU_DEVICES[0]}" "$PYTHON" skillrise/eval_alfworld_prompting.py \
  --style "$STYLE" --base "$ZTWANG_MODEL_PATH" --selection-manifest "$SELECTION" \
  --out-dir "$OUT" --run-id "$RUN_ID" --rank 0 --world-size 8 --samples 1 --preflight
"$PYTHON" skillrise/eval_alfworld_prompting.py --verify-preflight "$OUT/preflight.json"

pids=()
for rank in 0 1 2 3 4 5 6 7; do
  (
    export CUDA_VISIBLE_DEVICES="${GPU_DEVICES[$rank]}"
    "$PYTHON" skillrise/eval_alfworld_prompting.py \
      --style "$STYLE" --base "$ZTWANG_MODEL_PATH" --selection-manifest "$SELECTION" \
      --out-dir "$OUT" --run-id "$RUN_ID" --rank "$rank" --world-size 8 --samples 1 \
      >"logs/${RUN_ID}-rank${rank}.log" 2>&1
  ) &
  pids+=("$!")
done

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "rank $i exited 0"
  else
    rc=$?
    echo "rank $i failed with exit=$rc" >&2
    status=1
  fi
done
if [[ $status -ne 0 ]]; then echo "At least one rank failed; refusing merge." >&2; exit 1; fi

"$PYTHON" skillrise/eval_alfworld_prompting.py --merge \
  --selection-manifest "$SELECTION" --out-dir "$OUT" --samples 1
echo "merged=$OUT/merged.json"
