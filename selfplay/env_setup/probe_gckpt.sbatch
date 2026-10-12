#!/bin/bash
# ============================================================================
# 2×2×2 组合实验：定位生成乱码的开关
#   sbatch ~/ztwang/selfplay/env_setup/probe_gckpt.sbatch
# ============================================================================
#SBATCH --job-name=probe-gckpt
#SBATCH --partition=debug
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=00:25:00
#SBATCH --output=/hpc2hdd/home/xyu608/ztwang/selfplay/logs/probe-gckpt-%j.out
#SBATCH --error=/hpc2hdd/home/xyu608/ztwang/selfplay/logs/probe-gckpt-%j.err

set -uo pipefail

PROJ=$HOME/ztwang/selfplay
cd "$PROJ"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thinktool

export PYTHONPATH="$PROJ/src:$PROJ/third_party/py310deps:${PYTHONPATH:-}"
export ZTWANG_MODEL_PATH="${ZTWANG_MODEL_PATH:-$HOME/thinktool/models/qwen3_4b_inst_2507}"
export ALFWORLD_GLIBC_COMPAT_DIR="$PROJ/third_party/glibc232"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

echo "=== job=$SLURM_JOB_ID node=$(hostname) start=$(date) ==="
nvidia-smi -L
python env_setup/probe_gckpt.py
echo "=== exit=$? end=$(date) ==="
