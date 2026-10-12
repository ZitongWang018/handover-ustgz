#!/bin/bash
# ============================================================================
# EZ 通用诊断作业：在 debug 分区（A40，几分钟起算）上跑任意自检/诊断脚本。
#
# 用法：
#   sbatch -p debug -t 00:30:00 --gres=gpu:1 \
#       --export=ALL,TARGET=src/ez_debug_proposer.py,ARGS="--config configs/ez_qwen3_4b.yaml" \
#       ~/ztwang/selfplay/scripts/ez_debug_proposer.sbatch
#
# TARGET 相对 $PROJ；默认就是 proposer 诊断。
# ============================================================================
#SBATCH --job-name=ez-dbg
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.out
#SBATCH --error=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.err

set -euo pipefail

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
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TARGET=${TARGET:-src/ez_debug_proposer.py}
ARGS=${ARGS:---config configs/ez_qwen3_4b.yaml}

echo "=== job=$SLURM_JOB_ID node=$(hostname) date=$(date) ==="
echo "TARGET=$TARGET ARGS=$ARGS"
nvidia-smi -L

# shellcheck disable=SC2086
python "$TARGET" $ARGS

echo "=== done: $(date) ==="
