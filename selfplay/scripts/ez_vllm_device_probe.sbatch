#!/bin/bash
# P2 前置检查：8 个进程各自「只可见一张卡」，验证 vLLM 引擎是否各就各位。
# 同时验证正式训练要用的**启动器**（bash 循环代替 torchrun）。
#SBATCH --job-name=ez-vllmdev
#SBATCH --account=root
#SBATCH --partition=i64m1tga800u8ka
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=0-00:40:00
#SBATCH --output=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.out
#SBATCH --error=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.err

set -uo pipefail
PROJ="$HOME/ztwang/selfplay"
cd "$PROJ"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thinktool
export PYTHONPATH="$PROJ/src:$PROJ/third_party/py310deps:${PYTHONPATH:-}"
export ZTWANG_MODEL_PATH="${ZTWANG_MODEL_PATH:-$HOME/thinktool/models/qwen3_4b}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2

echo "=== 启动前 ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

for i in 0 1 2 3 4 5 6 7; do
  ( CUDA_VISIBLE_DEVICES=$i PROBE_TAG=$i python scripts/ez_vllm_device_probe.py \
      > "logs/devprobe_${i}.log" 2>&1 ; echo "probe$i exit=$?" ) &
done
wait

echo "=== 各探针自述 ==="
grep -h "\[probe" logs/devprobe_*.log

echo "=== 引擎驻留后 ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
python - <<'EOF'
import subprocess
smi = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout
used = [int(l.split(",")[1].strip().split()[0]) for l in smi.strip().splitlines()]
busy = [i for i, u in enumerate(used) if u > 1500]
print(f"[verdict] 显存>1.5G 的卡 = {busy}（期望 [0..7] 全覆盖）")
print("P2-GATE-DEVICE " + ("PASS" if len(busy) == 8 else "FAIL"))
EOF
