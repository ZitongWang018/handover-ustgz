#!/bin/bash
# ============================================================================
# H100 × 8 整机作业模板  —  HKUST-GZ HPC 三期 (ACD)
# ----------------------------------------------------------------------------
# 三期是独立集群: 登录节点 hpc3login，集群名 chess，Slurm 在 /bin（无需设 PATH）
#   分区别:  acd_u(默认,免费共享) | acd_ue(独占) | emergency_acd(紧急) | debug(30分钟)
#   时限:    7 天 (debug 30 分钟)
#   配额:    CPU 128 核 / GPU 16 张 / 单作业≤7天；MaxJobsPerUser=20
#
# ⚠️ 三期【无法访问外网】，git/pip/conda 联网操作都会失败。
#    代码请从本机推送:  python hpc_tools/sync_ztwang.py --host hpc3
#
# 提交:   sbatch ~/ztwang/tools/sbatch/gpu_h100x8.sbatch
# 预测起算: sbatch --test-only -p acd_u --gres=gpu:8 --time=1:00:00 --wrap true
# 注意:   三期 PrivateData=jobs，squeue 看不到别人的作业，判断拥堵看节点状态
# ============================================================================
#SBATCH --job-name=sp3-h100x8
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=1500G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/user/xyu608/ztwang/logs/%x-%j.out
#SBATCH --error=/data/user/xyu608/ztwang/logs/%x-%j.err

set -euo pipefail

# ---------- 环境 ----------
# 三期 conda 是 /share/anaconda3，登录即用；环境示例: pytorch271 / vllm / ReVision
source /share/anaconda3/etc/profile.d/conda.sh
conda activate pytorch271          # 按需替换

# ---------- 代码目录（铁律一：一切都在 ztwang 里） ----------
cd ~/ztwang/selfplay

# ---------- 运行目录（铁律二：断点只留一个） ----------
RUN_DIR=~/ztwang/runs/$SLURM_JOB_ID
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR"

# ---------- 自检 ----------
echo "=== job=$SLURM_JOB_ID node=$(hostname) date=$(date) ==="
nvidia-smi -L
echo "GPU count: $(nvidia-smi -L | wc -l)"
python -V

# ---------- 8 卡分布式训练 ----------
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=8
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0        # 三期 400Gb/s RoCE v2，实际网卡名以 ibdev2netdev 为准

# ⚠️ 断点只保留一个：--save_total_limit 1 或脚本内「删旧写新」
# torchrun --nproc_per_node=8 --nnodes=1 train.py \
#     --config configs/xxx.yaml \
#     --output_dir "$RUN_DIR" \
#     --save_steps 500 --save_total_limit 1 --save_only_model

echo "=== done: $(date) ==="
