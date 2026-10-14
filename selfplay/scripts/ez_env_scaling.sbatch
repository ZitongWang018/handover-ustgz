#!/bin/bash
# P3 前置对照：环境步进在"单进程"与"8 进程并发（= 生产里 8 个 rank 的真实负载）"
# 下的耗时倍数。决定 vLLM 之后剩余时间能不能压进 10h 的关键数字。
#SBATCH --job-name=ez-envscale
#SBATCH --account=root
#SBATCH --partition=i64m1tga40u
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=0-01:00:00
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
echo "job=${SLURM_JOB_ID} host=$(hostname) cpus=$(nproc)"

echo "########## A) 单进程（无争抢） ##########"
python scripts/ez_vllm_real_bench.py --mode envonly --json-out bench_out/env_1proc.json \
  2>&1 | grep -E "^\[real\]"

echo "########## B) 8 进程并发（≈ 8 rank 生产负载） ##########"
for i in 0 1 2 3 4 5 6 7; do
  ( python scripts/ez_vllm_real_bench.py --mode envonly \
      --json-out "bench_out/env_8p_${i}.json" > "logs/env8p_${i}.log" 2>&1 ) &
done
wait

echo "=================== SUMMARY ==================="
python - <<'EOF'
import json, glob, statistics as st
one = json.load(open("bench_out/env_1proc.json"))[0]
print("A) 单进程 :", json.dumps(one, ensure_ascii=False))
xs = []
for f in sorted(glob.glob("bench_out/env_8p_*.json")):
    d = json.load(open(f))[0]
    xs.append(d)
    print("B)", f.split("/")[-1], json.dumps(d, ensure_ascii=False))
if xs:
    m = st.mean(x["wall_s"] for x in xs)
    print(f"B) 8 进程均值 wall_s = {m:.1f}（单进程 {one['wall_s']:.1f}）"
          f" -> 超订倍数 = {m / max(one['wall_s'], 1e-9):.2f}x")
EOF
