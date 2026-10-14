#!/bin/bash
# P1 实测：同一份权重/同一批 prompt，HF 采样 vs vLLM 采样。
# 单节点 8 卡，每个配置独占一张卡并行跑（互不干扰）。
#SBATCH --job-name=ez-vllmbench
#SBATCH --account=root
#SBATCH --partition=i64m1tga800u8ka
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
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
mkdir -p logs bench_out
echo "job=${SLURM_JOB_ID} host=$(hostname) base=$ZTWANG_MODEL_PATH"
nvidia-smi -L

P=python
OPT="--slots ${SLOTS:-16} --max-new ${MAXNEW:-96} --prompt-tokens ${PROMPT:-2500} --repeat ${REPEAT:-3}"

launch () {  # launch <gpu> <tag> <extra args...>
  local gpu="$1"; local tag="$2"; shift 2
  ( CUDA_VISIBLE_DEVICES="$gpu" $P scripts/ez_vllm_bench.py $OPT "$@" \
      --json-out "bench_out/${tag}.json" > "logs/bench_${tag}.log" 2>&1 ; \
    echo "[gpu$gpu $tag] exit=$?" ) &
}

launch 0 hf_s16_n96   --mode hf
launch 1 vl_s16_n96   --mode vllm
launch 2 hf_s16_n160  --mode hf   --max-new 160
launch 3 vl_s16_n160  --mode vllm --max-new 160
launch 4 hf_s16_p6000 --mode hf   --prompt-tokens 6000
launch 5 vl_s16_p6000 --mode vllm --prompt-tokens 6000
wait

echo "=================== SUMMARY ==================="
cat bench_out/*.json
echo "=================== LOG TAILS ==================="
for f in logs/bench_*.log; do echo "--- $f ---"; tail -6 "$f"; done
