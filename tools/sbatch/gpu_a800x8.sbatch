#!/bin/bash
# ============================================================================
# A800 × 8 整机作业模板  —  HKUST-GZ HPC 二期
# ----------------------------------------------------------------------------
# ⚠️ 关键规则: 8 卡作业【只能】提交到 i64m1tga800u8ka (A800) 或 i64m1tga40u8ka (A40)
#    提交到 i64m1tga800u 等普通分区会被 Slurm 直接拒绝:
#    "8-GPU jobs are ONLY allowed on partitions 'i64m1tga800u8ka' and 'i64m1tga40u8ka'"
#
# 提交:   sbatch ~/ztwang/tools/sbatch/gpu_a800x8.sbatch
# 预测起算: sbatch --test-only -p i64m1tga800u8ka --gres=gpu:a800:8 --time=1:00:00 --wrap true
# 查看:   squeue -u xyu608
# 取消:   scancel <JOBID>
# ============================================================================
#SBATCH --job-name=selfplay8g
#SBATCH --partition=i64m1tga800u8ka
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:a800:8
#SBATCH --mem=800G
#SBATCH --time=1-00:00:00
#SBATCH --output=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.out
#SBATCH --error=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.err

set -euo pipefail

# ---------- 环境 ----------
module load cuda/12.8
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thinktool          # 改成: ReVision / longlive2 / aflow / ...

# ---------- 代码目录 ----------
cd ~/ztwang/selfplay

# ---------- 环境自检（写进日志，方便回溯） ----------
echo "=== job=$SLURM_JOB_ID node=$(hostname) date=$(date) ==="
echo "GPUS: $(nvidia-smi -L | wc -l)"
nvidia-smi -L
python -V

# ---------- 8 卡分布式训练示例 ----------
# torchrun 多卡 / deepspeed / 自定义启动器按需替换
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=8
export OMP_NUM_THREADS=8

# torchrun --nproc_per_node=8 --nnodes=1 train.py --config configs/xxx.yaml

echo "=== done: $(date) ==="
