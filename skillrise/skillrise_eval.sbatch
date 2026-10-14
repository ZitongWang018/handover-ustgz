#!/bin/bash
# SkillRise ALFWorld prompting 评测（zero-shot / ReAct）——纯推理，不训练，用我们自己的 vLLM。
#
# 用法：
#   sbatch --export=ALL,EZ_STYLE=react,EZ_SPLIT=eval_in_distribution,EZ_N=8 \
#          skillrise/skillrise_eval.sbatch
#   EZ_MAX_NEW 默认 1024 —— 对齐上游 data.max_response_length=1024（原来 512 会截断）
# 全量：EZ_N=0
#
# 为什么固定 8 卡分区：1 卡作业会掉进共享账号 root 的长队（reason=Priority），
# 而 8 卡分区经常有空闲节点；本评测只用第 0 张卡。
#SBATCH --job-name=skr-eval
#SBATCH --account=root
#SBATCH --partition=i64m1tga800u8ka
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=0-02:00:00
#SBATCH --output=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.out
#SBATCH --error=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.err

set -uo pipefail
PROJ="$HOME/ztwang"
cd "$PROJ"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thinktool

export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
export ZTWANG_MODEL_PATH="${ZTWANG_MODEL_PATH:-$HOME/thinktool/models/qwen3_4b}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export PYTHONPATH="$PROJ/selfplay/src:$PROJ/selfplay/third_party/py310deps:${PYTHONPATH:-}"
mkdir -p logs skillrise/out

STYLE="${EZ_STYLE:-react}"
SPLIT="${EZ_SPLIT:-eval_in_distribution}"
N="${EZ_N:-128}"   # 对齐上游 data.val_batch_size=128
OUT="skillrise/out/${STYLE}_${SPLIT}_n${N}_$(date +%m%d_%H%M%S).json"

echo "job=${SLURM_JOB_ID} style=$STYLE split=$SPLIT n=$N model=$ZTWANG_MODEL_PATH"
nvidia-smi -L | head -2

# 只用第 0 张卡（评测是单进程单卡推理）
CUDA_VISIBLE_DEVICES=0 python skillrise/eval_alfworld_prompting.py \
  --style "$STYLE" --split "$SPLIT" --n "$N" \
  --base "$ZTWANG_MODEL_PATH" --samples "${EZ_SAMPLES:-1}" --batch "${EZ_BATCH:-32}" \
  --max-steps "${EZ_MAX_STEPS:-30}" --max-new-tokens "${EZ_MAX_NEW:-1024}" \
  --temperature "${EZ_TEMPERATURE:-0.7}" --top-p "${EZ_TOP_P:-0.95}" \
  --out "$OUT"
rc=$?
echo "[skillrise-eval] exit=$rc out=$OUT"
exit $rc
