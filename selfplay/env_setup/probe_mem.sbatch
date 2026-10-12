#!/bin/bash
# ============================================================================
# 显存 + 参数更新有效性探针（fp32 vs bf16）
#   ⚠️ 必须 A800 80G；A40 48G 装不下 fp32 的 4B 模型
#   sbatch --partition=emergency_gpu ~/ztwang/selfplay/env_setup/probe_mem.sbatch
# ============================================================================
#SBATCH --job-name=probe-mem
#SBATCH --partition=emergency_gpu
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=200G
#SBATCH --time=00:40:00
#SBATCH --output=/hpc2hdd/home/xyu608/ztwang/selfplay/logs/probe-mem-%j.out
#SBATCH --error=/hpc2hdd/home/xyu608/ztwang/selfplay/logs/probe-mem-%j.err

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
nvidia-smi --query-gpu=memory.total --format=csv
python -V

python env_setup/probe_mem.py
echo "=== exit=$? end=$(date) ==="
