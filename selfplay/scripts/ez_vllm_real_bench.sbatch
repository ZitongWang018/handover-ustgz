#!/bin/bash
# P1 验收门 #3：真实 ALFWorld 环境上的 HF vs vLLM 对比（proposer 探索 + solve rate 内循环）。
#SBATCH --job-name=ez-vllmreal
#SBATCH --account=root
#SBATCH --partition=i64m1tga800u8ka
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --time=0-01:30:00
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
mkdir -p logs bench_out
echo "job=${SLURM_JOB_ID} host=$(hostname) base=$ZTWANG_MODEL_PATH"
nvidia-smi -L

launch () {
  local gpu="$1"; local tag="$2"; shift 2
  ( CUDA_VISIBLE_DEVICES="$gpu" python scripts/ez_vllm_real_bench.py "$@" \
      --json-out "bench_out/real_${tag}.json" > "logs/real_${tag}.log" 2>&1 ; \
    echo "[gpu$gpu $tag] exit=$?" ) &
}

launch 0 hf   --mode hf
launch 1 vllm --mode vllm
wait

echo "=================== SUMMARY ==================="
cat bench_out/real_*.json 2>/dev/null
echo "=================== LOG TAILS ==================="
for f in logs/real_*.log; do echo "--- $f ---"; grep "\[real\]" "$f" | tail -8; done
