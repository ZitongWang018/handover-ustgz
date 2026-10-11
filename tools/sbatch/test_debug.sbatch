#!/bin/bash
# ============================================================================
# 30 秒自检作业 —— 用 debug 分区（限时 30 分钟）验证环境是否正常
# 提交: sbatch ~/ztwang/tools/sbatch/test_debug.sbatch
# 结果: cat ~/ztwang/logs/test_debug-<JOBID>.out
# ============================================================================
#SBATCH --job-name=test_debug
#SBATCH --partition=debug
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.out
#SBATCH --error=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.err

echo "job_id  = $SLURM_JOB_ID"
echo "node    = $(hostname)"
echo "date    = $(date)"
echo "cpus    = $SLURM_CPUS_ON_NODE"
echo "mem     = $SLURM_MEM_PER_NODE MB"
echo "workdir = $(pwd)"
echo "home    = $HOME"

echo "--- Slurm 命令 ---"
export PATH=/opt/slurm/bin:$PATH
sinfo -V

echo "--- SSD 写入测试 (~/jhspoolers) ---"
touch ~/jhspoolers/.write_test_$SLURM_JOB_ID && echo "SSD 可写 OK" && rm -f ~/jhspoolers/.write_test_$SLURM_JOB_ID

echo "--- conda ---"
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda env list | head -15

echo "--- python ---"
python -V 2>&1 || echo "系统 python 不可用，需 conda activate"

echo "--- GPU (debug CPU 节点无 GPU 属正常) ---"
nvidia-smi -L 2>&1 | head -3

echo "--- 本地模型 ---"
ls ~/thinktool/models 2>&1 | head -10

echo "=== OK ==="
