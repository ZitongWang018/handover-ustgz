#!/bin/bash
# Formal Explorer-Zero job for the eight-GPU A800 partition.
# Invoke with an explicit base path, e.g.:
# sbatch --export=ALL,ZTWANG_MODEL_PATH=/path/to/qwen3-4b-base selfplay/scripts/ez_train_a800x8.sbatch
# Checkpoints and run logs share the configured run directory; on hpc2 this is the
# user's jhspoolers SSD as required for optimizer checkpoints.
#SBATCH --job-name=ez-qwen3
#SBATCH --account=root
#SBATCH --partition=i64m1tga800u8ka
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --time=7-00:00:00
#SBATCH --output=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.out
#SBATCH --error=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.err

set -euo pipefail
: "${ZTWANG_MODEL_PATH:?Set ZTWANG_MODEL_PATH explicitly to the intended Qwen3-4B base checkpoint}"
PROJ="$HOME/ztwang/selfplay"
cd "$PROJ"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${ZTWANG_ENV:-thinktool}"

export PYTHONPATH="$PROJ/src:$PROJ/third_party/py310deps:${PYTHONPATH:-}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
export EZ_RUN_OUTPUT_DIR="${EZ_RUN_OUTPUT_DIR:-$HOME/jhspoolers/ez_runs}"
export EZ_SWANLAB_MODE="${EZ_SWANLAB_MODE:-auto}"
export EZ_SWANLAB_PROJECT="${EZ_SWANLAB_PROJECT:-self-play}"
export EZ_SWANLAB_WORKSPACE="${EZ_SWANLAB_WORKSPACE:-ZitongWang}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN

mkdir -p "$EZ_RUN_OUTPUT_DIR" "$HOME/ztwang/logs"
echo "job=${SLURM_JOB_ID:-local} host=$(hostname) base=$ZTWANG_MODEL_PATH"
echo "global TPS=$(python -c 'import yaml; print(yaml.safe_load(open("configs/ez_qwen3_4b.yaml"))["train"]["tasks_per_step"])') G=$(python -c 'import yaml; print(yaml.safe_load(open("configs/ez_qwen3_4b.yaml"))["train"]["group_size"])')"
nvidia-smi -L
python scripts/ez_unit_test.py
python scripts/ez_check_smoke_selftest.py
python scripts/ez_check_env_facts.py
export MASTER_ADDR="$(hostname)"
export MASTER_PORT="$((29500 + ${SLURM_JOB_ID:-12345} % 1000))"
torchrun --nproc_per_node=8 --nnodes=1 --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" src/ez_train.py --config configs/ez_qwen3_4b.yaml
