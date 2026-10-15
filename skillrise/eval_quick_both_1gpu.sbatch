#!/bin/bash
# 合并版：一个作业里**顺序**跑两种评测集（128 槽 → 全 140），各自 5 个种子。
# 目的：把节点需求从 4 个降到 2 个（评测只用 1 张卡，但 8ka 分区只收 8 卡作业）。
#   sbatch --export=ALL,EZ_STYLE=react skillrise/eval_quick_both.sbatch
#SBATCH --job-name=skr-b1g
#SBATCH --account=root
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=0-03:00:00
#SBATCH --output=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.out
#SBATCH --error=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.err

set -uo pipefail
PROJ="$HOME/ztwang"; cd "$PROJ"
PY="$HOME/miniconda3/envs/thinktool/bin/python"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
export ZTWANG_MODEL_PATH="${ZTWANG_MODEL_PATH:-$HOME/thinktool/models/qwen3_4b_inst_2507}"
export PYTHONPATH="$PROJ/selfplay/src:$PROJ/selfplay/third_party/py310deps"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
mkdir -p logs skillrise/out
STYLE="${EZ_STYLE:-react}"
SEEDS="${EZ_SEEDS:-0,1,2,3,4}"
SEL="$PROJ/skillrise/out/selection_128.json"
TAG="$(date +%m%d_%H%M%S)"
echo "job=$SLURM_JOB_ID style=$STYLE seeds=$SEEDS py=$PY model=$ZTWANG_MODEL_PATH"

COMMON=(--style "$STYLE" --base "$ZTWANG_MODEL_PATH" --seed-list "$SEEDS"
        --samples 1 --batch 32 --max-steps 30 --max-new-tokens 1024
        --temperature 0.7 --top-p 0.95)

echo "=== [1/2] 128 槽（清单） ==="
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "$PY" skillrise/eval_quick.py \
  "${COMMON[@]}" --selection-manifest "$SEL" \
  --out "skillrise/out/quick_${STYLE}_128_seeds01234_${TAG}.json"
echo "[both] 128 阶段 exit=$?"

echo "=== [2/2] 全 140（不排序、不加清单） ==="
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "$PY" skillrise/eval_quick.py \
  "${COMMON[@]}" --n 0 \
  --out "skillrise/out/quick_${STYLE}_140_seeds01234_${TAG}.json"
echo "[both] 140 阶段 exit=$?"
echo "[both] done tag=$TAG"
