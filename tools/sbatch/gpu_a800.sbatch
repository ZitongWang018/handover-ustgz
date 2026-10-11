#!/bin/bash
# ============================================================================
# A800 单卡 / 多卡作业模板（非整机）  —  HKUST-GZ HPC 二期
# 分区: i64m1tga800u (7天, 11台) / long_gpu (14天) / emergency_gpu (14天)
# 提交: sbatch ~/ztwang/tools/sbatch/gpu_a800.sbatch
# ============================================================================
#SBATCH --job-name=mytrain
#SBATCH --partition=i64m1tga800u
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a800:1              # 非 8 卡都可以；=8 必须换 8ka 队列
#SBATCH --mem=200G
#SBATCH --time=8:00:00                 # 短时限利于 backfill 插队
#SBATCH --output=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.out
#SBATCH --error=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.err

set -euo pipefail

module load cuda/12.8
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thinktool

cd ~/ztwang/selfplay

echo "=== job=$SLURM_JOB_ID node=$(hostname) date=$(date) ==="
nvidia-smi -L
python -V

python train.py

echo "=== done: $(date) ==="
