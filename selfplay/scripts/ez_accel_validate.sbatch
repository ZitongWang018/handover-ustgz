#!/bin/bash
# 加速验证总作业：一个 8 卡节点 / 64 核，分三段串行跑，避免 QOS CPU 配额被打满。
#   A) 环境步进对照：单进程 vs 8 进程并发（≈ 生产 8 rank 的真实环境负载）
#   B) vLLM 多 rank 设备隔离探针（4 rank torchrun）
#   C) 真实 ALFWorld 环境上的 vLLM 相位耗时复测
#SBATCH --job-name=ez-accelval
#SBATCH --account=root
#SBATCH --partition=i64m1tga800u8ka
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --time=0-02:00:00
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
nvidia-smi -L

echo "###################### A) 环境步进对照 ######################"
echo "----- A1) 单进程 -----"
python scripts/ez_vllm_real_bench.py --mode envonly --json-out bench_out/env_1proc.json 2>&1 | grep -E "^\[real\]"
echo "----- A2) 8 进程并发 -----"
for i in 0 1 2 3 4 5 6 7; do
  ( python scripts/ez_vllm_real_bench.py --mode envonly \
      --json-out "bench_out/env_8p_${i}.json" > "logs/env8p_${i}.log" 2>&1 ) &
done
wait
python - <<'EOF'
import json, glob, statistics as st
one = json.load(open("bench_out/env_1proc.json"))[0]
print("[A1] 单进程 :", json.dumps(one, ensure_ascii=False))
xs = [json.load(open(f))[0] for f in sorted(glob.glob("bench_out/env_8p_*.json"))]
for f, d in zip(sorted(glob.glob("bench_out/env_8p_*.json")), xs):
    print("[A2]", f.split("/")[-1], json.dumps(d, ensure_ascii=False))
if xs:
    m = st.mean(x["wall_s"] for x in xs)
    print(f"[A-VERDICT] 8 进程均值 wall_s={m:.1f} vs 单进程 {one['wall_s']:.1f} "
          f"-> 超订倍数 {m/max(one['wall_s'],1e-9):.2f}x")
EOF

echo "###################### B) vLLM 多 rank 设备隔离 ######################"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --nnodes=1 \
  --master_addr=127.0.0.1 --master_port=$((29700 + ${SLURM_JOB_ID:-1} % 200)) \
  scripts/ez_vllm_device_probe.py 2>&1 | grep -E "^\[rank|^\[nvidia-smi\]|^\[verdict\]|P2-GATE|^[0-9]+, "
echo "device-probe exit=$?"

echo "###################### C) vLLM 真实相位复测 ######################"
CUDA_VISIBLE_DEVICES=0 python scripts/ez_vllm_real_bench.py --mode vllm \
  --json-out bench_out/real_vllm2.json 2>&1 | grep -E "^\[real\]"

echo "=================== DONE ==================="
cat bench_out/env_1proc.json bench_out/real_vllm2.json 2>/dev/null
