#!/bin/bash
# ============================================================================
# ALFWorld × Qwen3-4B GRPO 训练 —— 三期 ACD H100 × 8 整机
# ----------------------------------------------------------------------------
# ⚠️ 三期【完全无外网】：不能 git clone / pip install。
#    代码与依赖靠 本机 -> 二期(有网) -> 三期 的方式搬：
#        python hpc_tools/sync_ztwang.py --host hpc3
#    依赖 wheel 先在二期 `pip download` 到 selfplay/third_party/wheels/，
#    再用 `pip install --no-index --find-links ... --target ...` 离线装。
# 提交:   sbatch ~/ztwang/selfplay/scripts/train_h100x8.sbatch
# 预测:   sbatch --test-only -p acd_u --gres=gpu:8 --time=1:00:00 --wrap true
# 注意:   PrivateData=jobs，squeue 看不到别人的作业，看 sinfo 节点状态
# ============================================================================
#SBATCH --job-name=sf-grpo-h100
#SBATCH --partition=acd_u
#SBATCH --account=d_tonggehuang_team
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=1500G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/user/xyu608/ztwang/logs/%x-%j.out
#SBATCH --error=/data/user/xyu608/ztwang/logs/%x-%j.err

set -euo pipefail

PROJ=$HOME/ztwang/selfplay
cd "$PROJ"

# ---------- 环境 ----------
# 三期没有 thinktool 环境。两条路：
#   A) 离线 clone/搬迁一个 py3.11 训练环境 + 我们自己的 --target 依赖（见 docs/SELFPLAY.md）
#   B) 用平台自带 /share/anaconda3/envs/vllm017 (py3.11, torch2.10+cu128) + 我们 --target 的 trl/textworld
source /share/anaconda3/etc/profile.d/conda.sh
conda activate "${ZTWANG_ENV:-vllm017}"

export PYTHONPATH="$PROJ/src:$PROJ/third_party/py311deps:${PYTHONPATH:-}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$PROJ/data/alfworld}"
export ZTWANG_MODEL_PATH="${ZTWANG_MODEL_PATH:-$PROJ/data/models/qwen3_4b_inst_2507}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export ALFWORLD_GLIBC_COMPAT_DIR="$PROJ/third_party/glibc232"

echo "=== job=$SLURM_JOB_ID node=$(hostname) date=$(date) ==="
echo "model=$ZTWANG_MODEL_PATH"
nvidia-smi -L
python -V
df -h "$HOME" | tail -1

if [[ "${SMOKE:-1}" == "1" ]]; then
  echo "--- smoke: env ---"
  python src/smoke_test.py --stage env --batch 2
fi

# ---------- 训练 ----------
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29511
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN

torchrun --nproc_per_node=8 --nnodes=1 --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    src/train_grpo.py --config configs/train_qwen3_4b.yaml

echo "=== done: $(date) ==="
