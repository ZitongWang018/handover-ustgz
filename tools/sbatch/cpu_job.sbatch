#!/bin/bash
# ============================================================================
# CPU 多核作业模板  —  HKUST-GZ HPC 二期
# 分区: i64m512u (64c/512G, 107台) / a128m512u (128c/512G, 16台)
#       i96m3tu (96c/3T 大内存) / long_cpu (14天)
# 提交: sbatch ~/ztwang/tools/sbatch/cpu_job.sbatch
# 注意: 节点独占申请，--mem 不得超过单节点内存
# ============================================================================
#SBATCH --job-name=mycpu
#SBATCH --partition=a128m512u          # 当前较有空闲的 CPU 队列
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=4:00:00
#SBATCH --output=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.out
#SBATCH --error=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ReVision

cd ~/ztwang/selfplay

echo "=== job=$SLURM_JOB_ID node=$(hostname) cpus=$SLURM_CPUS_ON_NODE date=$(date) ==="
python run.py
echo "=== done: $(date) ==="
