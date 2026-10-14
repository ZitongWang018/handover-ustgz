#!/bin/bash
# Full valid_unseen evaluation with eight explicitly declared, uniquely named shards.
# No cluster job is started by editing or validating this script.
# Usage: sbatch --export=ALL,ZTWANG_MODEL_PATH=/path/to/model,EZ_CKPT=/path/to/ckpt scripts/ez_eval_full_8gpu.sbatch
# Base control: add EZ_BASE_ONLY=1; no checkpoint is needed in that mode.
#SBATCH --job-name=ez-evalfull
#SBATCH --account=root
#SBATCH --partition=i64m1tga800u8ka
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=600G
#SBATCH --time=0-02:00:00
#SBATCH --output=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.out
#SBATCH --error=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.err

set -euo pipefail
PROJ="$HOME/ztwang/selfplay"
cd "$PROJ"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thinktool

export PYTHONPATH="$PROJ/src:$PROJ/third_party/py310deps:${PYTHONPATH:-}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
: "${ZTWANG_MODEL_PATH:?Set ZTWANG_MODEL_PATH explicitly}"

BASE_ONLY="${EZ_BASE_ONLY:-0}"
if [ "$BASE_ONLY" != "1" ]; then
  : "${EZ_CKPT:?Set EZ_CKPT to a checkpoint directory containing solver/}"
  if [ ! -d "$EZ_CKPT/solver" ] && [ -d "$EZ_CKPT/ckpt/solver" ]; then
    EZ_CKPT="$EZ_CKPT/ckpt"
  fi
  [ -d "$EZ_CKPT/solver" ] || { echo "FATAL: $EZ_CKPT has no solver/"; exit 2; }
fi

TASKS=134
SHARDS="${EZ_SHARDS:-8}"
[ "$SHARDS" = "8" ] || { echo "FATAL: full benchmark merge currently requires EZ_SHARDS=8"; exit 2; }
PROTOCOL="${EZ_PROTOCOL:-paper-approx}"
if [ -n "${EZ_SAMPLES:-}" ]; then
  SAMPLES="$EZ_SAMPLES"
elif [ "$PROTOCOL" = "public-rich" ]; then
  SAMPLES=1
else
  SAMPLES=4
fi
case "$SAMPLES" in ''|*[!0-9]*) echo "FATAL: EZ_SAMPLES must be a positive integer"; exit 2;; esac
[ "$SAMPLES" -gt 0 ] || { echo "FATAL: EZ_SAMPLES must be positive"; exit 2; }
PER=$(( (TASKS + SHARDS - 1) / SHARDS ))
BACKEND="${EZ_BACKEND:-hf}"
BATCH="${EZ_BATCH:-2}"
RUN_ID="${SLURM_JOB_ID:-manual-$(date +%s)-$$}"
TAG="${EZ_TAG:-${PROTOCOL}-$(basename "${EZ_CKPT:-base-only}")-job${RUN_ID}}"
OUTDIR="eval_out/full_${TAG}"
mkdir -p "$OUTDIR" logs
[ ! -e "$OUTDIR/expected_shards.json" ] || { echo "FATAL: refusing existing run directory $OUTDIR"; exit 2; }

python3 - "$OUTDIR/expected_shards.json" "$OUTDIR" "$TASKS" "$SAMPLES" "$SHARDS" "$PER" "$RUN_ID" <<'PY'
import json, sys
manifest, outdir, tasks, samples, shards, per, run_id = sys.argv[1:]
tasks, samples, shards, per = map(int, (tasks, samples, shards, per))
items = []
for rank in range(shards):
    offset = rank * per
    count = max(0, min(per, tasks - offset))
    items.append({"rank": rank, "offset": offset, "n": count,
                  "path": f"{outdir}/shard_{rank:02d}.json"})
with open(manifest, "w", encoding="utf-8") as fh:
    json.dump({"run_id": run_id, "required_task_count": tasks, "samples_per_task": samples,
               "required_task_type_count": 6,
               "expected_shards": items}, fh, indent=2)
PY

PIDS=()
for k in $(seq 0 $((SHARDS - 1))); do
  OFF=$((k * PER))
  COUNT=$(( TASKS - OFF ))
  [ "$COUNT" -gt "$PER" ] && COUNT="$PER"
  [ "$COUNT" -lt 0 ] && COUNT=0
  OUT="$OUTDIR/shard_$(printf '%02d' "$k").json"
  EXTRA=()
  DECODE=()
  if [ "$BASE_ONLY" = "1" ]; then EXTRA+=(--base-only); else EXTRA+=(--ckpt "$EZ_CKPT"); fi
  [ -n "${EZ_MAX_STEPS:-}" ] && DECODE+=(--max-steps "$EZ_MAX_STEPS")
  if [ -n "${EZ_MAX_NEW_TOKENS:-}" ]; then
    DECODE+=(--max-new-tokens "$EZ_MAX_NEW_TOKENS")
  elif [ -n "${EZ_MAX_NEW:-}" ]; then
    DECODE+=(--max-new-tokens "$EZ_MAX_NEW")
  fi
  [ -n "${EZ_MAX_CONTEXT_TOKENS:-}" ] && DECODE+=(--max-context-tokens "$EZ_MAX_CONTEXT_TOKENS")
  [ -n "${EZ_TEMPERATURE:-}" ] && DECODE+=(--temperature "$EZ_TEMPERATURE")
  [ -n "${EZ_TOP_P:-}" ] && DECODE+=(--top-p "$EZ_TOP_P")
  [ -n "${EZ_TOP_K:-}" ] && DECODE+=(--top-k "$EZ_TOP_K")
  [ -n "${EZ_REPETITION_PENALTY:-}" ] && DECODE+=(--repetition-penalty "$EZ_REPETITION_PENALTY")
  CUDA_VISIBLE_DEVICES="$k" python src/eval_ez.py \
    --protocol "$PROTOCOL" "${EXTRA[@]}" --backend "$BACKEND" \
    --base "$ZTWANG_MODEL_PATH" --select head --offset "$OFF" --n "$COUNT" \
    --samples "$SAMPLES" --batch "$BATCH" --seed "${EZ_SEED:-42}" \
    --run-id "$RUN_ID" "${DECODE[@]}" --out "$OUT" > "$OUTDIR/shard_$(printf '%02d' "$k").log" 2>&1 &
  PIDS+=("$!")
done

FAILED=0
for k in $(seq 0 $((SHARDS - 1))); do
  if wait "${PIDS[$k]}"; then
    echo "[eval-shard] rank=$k exit=0"
  else
    rc=$?
    echo "[eval-shard] rank=$k exit=$rc"
    FAILED=1
  fi
done
[ "$FAILED" = "0" ] || { echo "FATAL: one or more evaluation shards failed"; exit 1; }

python src/ez_eval_merge.py --expected-manifest "$OUTDIR/expected_shards.json" \
  --output "$OUTDIR/merged.json"
echo "[eval-full] result=$OUTDIR/merged.json"
