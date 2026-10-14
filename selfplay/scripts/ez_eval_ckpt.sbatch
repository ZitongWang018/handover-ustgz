#!/bin/bash
# 独立评测：加载任意 EZ 断点，在官方 valid_unseen 上算 SR（不改权重、不训练）。
#
# 用途：
#   1) 补训练期间漏掉的评测点（如 outer0 崩在评测那一步）
#   2) 随时对任意 checkpoints/ckpt-* 或 ckpt/ 做独立评测
#
# 用法：
#   sbatch --export=ALL,ZTWANG_MODEL_PATH=...,EZ_CKPT=<ckpt 目录> scripts/ez_eval_ckpt.sbatch
#   可选：EZ_PROTOCOL=public-rich EZ_N=16 EZ_SAMPLES=1 EZ_SELECT=stratify EZ_BACKENDS="vllm hf"
#
# 默认 stratify：子集覆盖各任务类型。需要与训练中 mid_eval_solver 对齐时，显式 EZ_SELECT=head。
#SBATCH --job-name=ez-evalckpt
#SBATCH --account=root
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
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
export ZTWANG_MODEL_PATH="${ZTWANG_MODEL_PATH:-$HOME/thinktool/models/qwen3_4b}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
mkdir -p logs eval_out

BASE_ONLY="${EZ_BASE_ONLY:-0}"
if [ "$BASE_ONLY" != "1" ]; then
  : "${EZ_CKPT:?Set EZ_CKPT to a checkpoint dir containing solver/}"
  if [ ! -d "$EZ_CKPT/solver" ] && [ -d "$EZ_CKPT/ckpt/solver" ]; then
    EZ_CKPT="$EZ_CKPT/ckpt"
  fi
  [ -d "$EZ_CKPT/solver" ] || { echo "FATAL: $EZ_CKPT 下没有 solver/"; exit 2; }
fi

N="${EZ_N:-16}"
PROTOCOL="${EZ_PROTOCOL:-paper-approx}"
SAMPLES="${EZ_SAMPLES:-$([ "$PROTOCOL" = public-rich ] && echo 1 || echo 4)}"
SELECT="${EZ_SELECT:-stratify}"
BATCH="${EZ_BATCH:-$N}"
BACKENDS="${EZ_BACKENDS:-vllm hf}"
TAG="${EZ_TAG:-$(basename "${EZ_CKPT:-base-only}")-job${SLURM_JOB_ID:-manual}}"
export TAG
BASE_SUFFIX=""
[ "$BASE_ONLY" = "1" ] && BASE_SUFFIX="-baseonly"
echo "job=${SLURM_JOB_ID:-manual} protocol=$PROTOCOL base_only=$BASE_ONLY n=$N samples=$SAMPLES select=$SELECT batch=$BATCH backends=[$BACKENDS]"
nvidia-smi -L

for B in $BACKENDS; do
  OUT="eval_out/${TAG}-${PROTOCOL}-${B}-${SELECT}-n${N}s${SAMPLES}${BASE_SUFFIX}.json"
  echo "########## protocol=$PROTOCOL backend=$B -> $OUT (serial on visible GPU 0) ##########"
  EXTRA=(); [ "$BASE_ONLY" = "1" ] && EXTRA+=(--base-only) || EXTRA+=(--ckpt "$EZ_CKPT")
  CUDA_VISIBLE_DEVICES=0 python src/eval_ez.py \
      --protocol "$PROTOCOL" --backend "$B" "${EXTRA[@]}" \
      --base "$ZTWANG_MODEL_PATH" \
      --n "$N" --samples "$SAMPLES" --select "$SELECT" --batch "$BATCH" \
      --seed 42 --out "$OUT" \
      > "logs/eval_${TAG}_${B}.log" 2>&1
  echo "[$B] exit=$?"
done

echo "=================== SUMMARY ==================="
for B in $BACKENDS; do
  OUT="eval_out/${TAG}-${PROTOCOL}-${B}-${SELECT}-n${N}s${SAMPLES}${BASE_SUFFIX}.json"
  python3 - "$OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
s = d["summary"]
print(f"{sys.argv[1]} episode_weighted_SR={s['episode_weighted_sr']:.4f} "
      f"sample_suite_SR={s['sample_suite_success_rates']} "
      f"population_SD={s['sample_suite_sr_population_std']:.4f}")
PY
done
