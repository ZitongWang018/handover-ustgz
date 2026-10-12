#!/bin/bash
# ============================================================================
# 健全性探针：区分「模型坏了」还是「生成配置坏了」
#   sbatch ~/ztwang/selfplay/env_setup/probe_sanity.sbatch
# ============================================================================
#SBATCH --job-name=probe-sanity
#SBATCH --partition=debug
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=00:25:00
#SBATCH --output=/hpc2hdd/home/xyu608/ztwang/selfplay/logs/probe-sanity-%j.out
#SBATCH --error=/hpc2hdd/home/xyu608/ztwang/selfplay/logs/probe-sanity-%j.err

set -uo pipefail

PROJ=$HOME/ztwang/selfplay
cd "$PROJ"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thinktool

export PYTHONPATH="$PROJ/src:$PROJ/third_party/py310deps:${PYTHONPATH:-}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
export ZTWANG_MODEL_PATH="${ZTWANG_MODEL_PATH:-$HOME/thinktool/models/qwen3_4b_inst_2507}"
export ALFWORLD_GLIBC_COMPAT_DIR="$PROJ/third_party/glibc232"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

echo "=== job=$SLURM_JOB_ID node=$(hostname) start=$(date) ==="
nvidia-smi -L
python -V

python env_setup/probe_sanity.py
echo "=== exit=$? end=$(date) ==="
