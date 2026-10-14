#!/bin/bash
# 诊断：模型在评测 prompt 上的回答被判成 action / malformed / finish 各多少。
# eval_ez._rollout_batch 把 malformed 静默转成 "look"，所以"90% look"可能是假象。
#
# 同时对比 max_new_tokens=96（我们当前配置）vs 512：
#   若 96 那组 malformed 高、512 那组显著低 => 是**生成被截断**，不是模型不会用格式。
#SBATCH --job-name=ez-diag
#SBATCH --account=root
#SBATCH --partition=i64m1tga800u8ka
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=0-00:50:00
#SBATCH --output=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.out
#SBATCH --error=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.err

set -uo pipefail
PROJ="$HOME/ztwang/selfplay"
cd "$PROJ"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thinktool
export PYTHONPATH="$PROJ/src:$PROJ/third_party/py310deps:${PYTHONPATH:-}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
export ZTWANG_MODEL_PATH="${ZTWANG_MODEL_PATH:-$HOME/thinktool/models/qwen3_4b}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export ZTWANG_VLLM_GMEM=0.85
mkdir -p logs

CK="${EZ_CKPT:-}"
run () {  # run <gpu> <tag> <backend> <ckpt> <n> <maxnew>
  local gpu="$1"; local tag="$2"; shift 2
  ( CUDA_VISIBLE_DEVICES=$gpu python scripts/ez_diag_setup.py --gpu "$@" \
      > "logs/diag_${tag}.log" 2>&1 ; echo "[$tag] exit=$?" ) &
}

echo "job=${SLURM_JOB_ID} ckpt=[$CK]"
nvidia-smi -L
run 0 base_n96   vllm ""        8 96
run 1 base_n512  vllm ""        8 512
run 2 adap_n96   vllm "$CK"     8 96
run 3 adap_n512  vllm "$CK"     8 512
wait

for t in base_n96 base_n512 adap_n96 adap_n512; do
  echo "########## $t ##########"
  grep -aE "解析类型|分布:|生成长度|malformed|\[0\] kind|\[1\] kind" "logs/diag_${t}.log" | head -8
  echo "--- 原始输出样本 ---"
  sed -n "/原始输出样本/,\$p" "logs/diag_${t}.log" | head -12
done
