#!/bin/bash
# ============================================================================
# 单卡端到端冒烟（fp32 + adafactor）—— 8 卡整机要排十几个小时，
# 所以先用一张能**立刻拿到**的卡把整条链路跑通，别等到 18:03 才发现崩在第一行。
#
# 用法（分区/时限在命令行给，脚本里不写死）：
#   # A40 48G，常年有空卡，几分钟内起算
#   sbatch -p debug -t 00:30:00 --gres=gpu:1 \
#       ~/ztwang/selfplay/scripts/train_smoke_1gpu.sbatch
#
#   # A800 80G，想顺便试 AdamW（改 CFG 里的 optimizer）时用它
#   sbatch -p emergency_gpu -t 02:00:00 --gres=gpu:1 \
#       --export=ALL,CFG=configs/train_smoke_fp32.yaml \
#       ~/ztwang/selfplay/scripts/train_smoke_1gpu.sbatch
#
# 判读要点（看 .out 末尾的 metrics 行）：
#   invalid_rate  ≈ 0     否则 GC/generate 又出问题了
#   adv_abs_mean  > 0     否则奖励没区分度，学不了
#   update_rel    > 0     否则参数没动（fp32 下还等于 0 就是 lr 量级问题）
# ============================================================================
#SBATCH --job-name=sf-fp32-smoke
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

# ---------- 环境 ----------
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thinktool

export PYTHONPATH="$PROJ/src:$PROJ/third_party/py310deps:${PYTHONPATH:-}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
export ZTWANG_MODEL_PATH="${ZTWANG_MODEL_PATH:-$HOME/thinktool/models/qwen3_4b_inst_2507}"
export ALFWORLD_GLIBC_COMPAT_DIR="$PROJ/third_party/glibc232"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CFG=${CFG:-configs/train_smoke_fp32.yaml}
ITERS=${ITERS:-3}

echo "=== job=$SLURM_JOB_ID node=$(hostname) date=$(date) ==="
echo "CFG=$CFG ITERS=$ITERS"
echo "model=$ZTWANG_MODEL_PATH"
nvidia-smi -L
nvidia-smi --query-gpu=memory.total --format=csv
python -V

# ---------- 阶段 0：预检（env + rollout）----------
echo "--- [0/2] env ---"
python src/smoke_test.py --stage env --batch 2 --max-steps 3
echo "--- [0/2] rollout（看奖励分布：全同分就说明这一步白跑）---"
python src/smoke_test.py --stage rollout --batch 4 --group-size 2 --max-steps 4

# ---------- 阶段 1：单卡 fp32 训练 ----------
echo "--- [1/2] 单卡 fp32 训练：${ITERS} iter ---"
export MASTER_ADDR=$(hostname)
export NCCL_DEBUG=WARN
torchrun --nproc_per_node=1 --nnodes=1 \
    --master_addr="$MASTER_ADDR" --master_port=29521 \
    src/train_grpo.py --config "$CFG" --max-iters "$ITERS"

echo "--- [1/2] 训练跑完，显存峰值见下 ---"
nvidia-smi --query-gpu=memory.peak --format=csv 2>/dev/null || true

# ---------- 阶段 1.5：准入门 ----------
if [[ "${GATE:-1}" == "1" ]]; then
  echo "--- [1.5/2] 冒烟自检 ---"
  if ! python scripts/check_smoke_ok.py --config "$CFG" --iters "$ITERS"; then
    echo "!!! 冒烟自检未通过：invalid_rate 或 advantage 异常，别往下跑 !!!"
    exit 1
  fi
else
  echo "--- [1.5/2] GATE=0，跳过自检 ---"
fi

echo "=== done: $(date) ==="
