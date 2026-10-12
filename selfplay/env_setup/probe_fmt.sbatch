#!/bin/bash
# ============================================================================
# 格式探针：看 Qwen3-4B-Instruct-2507 在 ALFWorld 上到底吐了什么
#
# 提交：
#   sbatch ~/ztwang/selfplay/env_setup/probe_fmt.sbatch
#
# 用 emergency_gpu（1 张 A800），实测立即起算。不要用 long_gpu（要排 3 天）。
# ============================================================================
#SBATCH --job-name=probe-fmt
#SBATCH --partition=emergency_gpu
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=100G
#SBATCH --time=1:00:00
#SBATCH --output=/hpc2hdd/home/xyu608/ztwang/selfplay/logs/probe-fmt-%j.out
#SBATCH --error=/hpc2hdd/home/xyu608/ztwang/selfplay/logs/probe-fmt-%j.err

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

python env_setup/probe_policy_fmt.py
rc=$?

echo "=== probe exit=$rc end=$(date) ==="
