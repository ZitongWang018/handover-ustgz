#!/bin/bash
# ============================================================================
# ALFWorld 评测作业 —— 二期（需要外网的 API 基线 / 单卡 HF 评测都走这里）
# ----------------------------------------------------------------------------
# 用法（用环境变量选 agent，`sbatch --export=ALL,AGENT=...`）：
#   AGENT=echo    sbatch --export=ALL,AGENT=echo    --job-name=sf-eval scripts/eval.sbatch
#   AGENT=random  sbatch --export=ALL,AGENT=random  --job-name=sf-eval scripts/eval.sbatch
#   AGENT=scripted sbatch --export=ALL,AGENT=scripted --job-name=sf-eval scripts/eval.sbatch
#   AGENT=deepseek sbatch --export=ALL,AGENT=deepseek,N=20 --job-name=sf-eval-ds scripts/eval.sbatch
#   AGENT=hf      sbatch --export=ALL,AGENT=hf,MODEL=$HOME/ztwang/selfplay/runs/.../ckpt --job-name=sf-eval-hf scripts/eval.sbatch
#
# 说明：这个作业只需要 1 张卡（hf）或 0 张卡（API/random/scripted）。
#      二分区的普通 GPU 分区不允许 8 卡，但 1 卡没问题。
# ============================================================================
#SBATCH --job-name=sf-eval
#SBATCH --partition=i64m1tga800u
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.out
#SBATCH --error=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.err

set -euo pipefail

PROJ=$HOME/ztwang/selfplay
cd "$PROJ"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thinktool

export PYTHONPATH="$PROJ/src:$PROJ/third_party/py310deps:${PYTHONPATH:-}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
export ALFWORLD_GLIBC_COMPAT_DIR="$PROJ/third_party/glibc232"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

AGENT=${AGENT:-echo}
SPLIT=${SPLIT:-eval_out_of_distribution}
N=${N:-20}
BATCH=${BATCH:-7}
MAX_STEPS=${MAX_STEPS:-30}
MODEL=${MODEL:-$HOME/thinktool/models/qwen3_4b_inst_2507}
TAG=${TAG:-}

echo "=== job=$SLURM_JOB_ID node=$(hostname) agent=$AGENT split=$SPLIT n=$N ==="
nvidia-smi -L | head -1

python src/eval_alfworld.py \
    --agent "$AGENT" \
    --model "$MODEL" \
    --split "$SPLIT" \
    --n "$N" \
    --batch "$BATCH" \
    --max-steps "$MAX_STEPS" \
    --tag "$TAG"

echo "=== done: $(date) ==="
