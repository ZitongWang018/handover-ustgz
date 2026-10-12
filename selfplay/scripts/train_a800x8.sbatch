#!/bin/bash
# ============================================================================
# ALFWorld × Qwen3-4B GRPO 训练 —— 二期 A800 × 8 整机
# ----------------------------------------------------------------------------
# ⚠️ 二期 8 卡作业【只能】提交到 i64m1tga800u8ka / i64m1tga40u8ka
#
# 提交（默认奖励配置 = 带过程项）：
#   sbatch ~/ztwang/selfplay/scripts/train_a800x8.sbatch
# 跑稀疏消融：
#   CFG=configs/train_qwen3_4b_sparse.yaml sbatch --export=ALL,CFG=configs/train_qwen3_4b_sparse.yaml \
#       --job-name=sf-grpo-sparse scripts/train_a800x8.sbatch
#
# 作业结构（同一个 allocation 里两步走，避免"验证挂掉还要再排 16h"）：
#   阶段 0  预检：env + 一次 group_size=4 的 rollout（对齐/奖励分布自检）
#   阶段 1  5 iter 的 DDP 冒烟（8 rank）—— 挂掉就立刻退出，不浪费时间
#   阶段 2  正式训练（total_iters 全长），中途按 save_every 存断点（只保留最新一个）
#
# 预测起算：sbatch --test-only -p i64m1tga800u8ka --account=root --gres=gpu:a800:8 --time=1:00:00 --wrap true
# ============================================================================
#SBATCH --job-name=sf-grpo-q3-4b
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

PROJ=$HOME/ztwang/selfplay
cd "$PROJ"

# ---------- 环境 ----------
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thinktool                      # py3.10：torch/transformers/trl/peft/accelerate

# py310deps：用 `pip install --target` 装进自己目录的 textworld/alfworld
export PYTHONPATH="$PROJ/src:$PROJ/third_party/py310deps:${PYTHONPATH:-}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
export ZTWANG_MODEL_PATH="${ZTWANG_MODEL_PATH:-$HOME/thinktool/models/qwen3_4b_inst_2507}"
export ALFWORLD_GLIBC_COMPAT_DIR="$PROJ/third_party/glibc232"
# libdownward.so 需要 conda 里的 libstdc++（系统那份缺 GLIBCXX_3.4.29）
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# ⚠️ OMP_NUM_THREADS：8 个 rank 各起 48 个 OMP 线程会直接把 64 核机器压垮，
#    线程互相抢核会让每一步都慢好几倍。64 / 8 = 8，留点余量取 6。
export OMP_NUM_THREADS=6
# ⚠️ expandable_segments：常规 allocator 对"大块显存反复申请/释放"（rollout 的
#    KV cache、训练的 logits）碎片化很严重，经常明明还有 10G 空闲却 OOM。
#    这个开关让段可以伸缩，几乎不影响速度，却能救回好几个 G。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ⚠️ NCCL 默认只在下一個集合通信时才报异步错误，结果是一个 rank 早就挂了、
#    其它 rank 干等到作业超时。打开它才能**当场**失败当场暴露。
export NCCL_ASYNC_ERROR_HANDLING=1

CFG=${CFG:-configs/train_qwen3_4b.yaml}
SMOKE=${SMOKE:-1}
WARMUP_ITERS=${WARMUP_ITERS:-5}

echo "=== job=$SLURM_JOB_ID node=$(hostname) date=$(date) ==="
echo "CFG=$CFG  WARMUP_ITERS=$WARMUP_ITERS"
echo "model=$ZTWANG_MODEL_PATH"
echo "ALFWORLD_DATA=$ALFWORLD_DATA"
nvidia-smi -L
python -V

# ---------- 阶段 0：预检 ----------
if [[ "$SMOKE" == "1" ]]; then
  echo "--- [0/2] smoke: env ---"
  python src/smoke_test.py --stage env --batch 2 --max-steps 3
  echo "--- [0/2] smoke: rollout（会顺带打奖励分布，reward 全同分就说明这步白跑）---"
  python src/smoke_test.py --stage rollout --batch 8 --group-size 4 --max-steps 4
fi

export MASTER_ADDR=$(hostname)
export NCCL_DEBUG=WARN
# ⚠️ master_port 不能用固定值：同一台机器上如果有别的作业也占了 29511/29513，
#    torchrun 会报 "Address already in use" 然后 8 个 rank 一起起不来 ——
#    而你已经排了十几个小时。用 jobid 派生一个基本不会撞的端口。
BASE_PORT=$(( 20000 + (SLURM_JOB_ID % 20000) ))

# ---------- 阶段 1：DDP 冒烟（8 rank，5 iter） ----------
echo "--- [1/2] 8-rank DDP 冒烟：${WARMUP_ITERS} iter ---"
torchrun --nproc_per_node=8 --nnodes=1 \
    --master_addr="$MASTER_ADDR" --master_port=$((BASE_PORT)) \
    src/train_grpo.py --config "$CFG" --max-iters "$WARMUP_ITERS"
echo "--- [1/2] 冒烟跑完 ---"

# ---------- 阶段 1.5：准入门 ----------
# ⚠️ 这道闸必须有。8 卡整机要排十几个小时，如果冒烟已经暴露
#    「invalid_rate≈1.0（模型没吐出合法动作）」或「advantage 恒为 0」，
#    却照样往下跑 300 iter，就是把一整个 allocation 扔进水里。
#    实测撞过一次：reward_mean=1.6 / group_std=0 / adv=0 / loss=0 / eval 成功率 0。
if [[ "${GATE:-1}" == "1" ]]; then
  echo "--- [1.5/2] 冒烟自检（决定是否进入正式训练）---"
  if ! python scripts/check_smoke_ok.py --config "$CFG" --iters "$WARMUP_ITERS"; then
    echo "!!! 冒烟自检未通过 —— 停在阶段 1，不启动正式训练，避免白烧 8 张 A800 !!!"
    echo "!!! 排查方向：invalid_rate≈1 -> prompt/解析/max_new_tokens；adv≈0 -> group_size 或奖励区分度 !!!"
    exit 1
  fi
else
  echo "--- [1.5/2] GATE=0，跳过自检（调试用，正式跑别开）---"
fi

# ---------- 阶段 2：正式训练 ----------
echo "--- [2/2] 正式训练：$CFG ---"
torchrun --nproc_per_node=8 --nnodes=1 \
    --master_addr="$MASTER_ADDR" --master_port=$((BASE_PORT + 1)) \
    src/train_grpo.py --config "$CFG"

echo "=== done: $(date) ==="
