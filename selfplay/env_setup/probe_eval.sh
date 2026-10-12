#!/bin/bash
# 最小可行性探针：在计算节点上验证「thinktool(py3.10) + third_party/py310deps + glibc232 兼容层」
# 这条链路能否真正 reset/step ALFWorld，并跑通一次 scripted 基线评测。
#
# 用法：
#   srun -p i64m512u --account=root -c 8 --mem=32G -t 00:20:00 bash ~/ztwang/selfplay/env_setup/probe_eval.sh
set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thinktool

PROJ=$HOME/ztwang/selfplay
cd "$PROJ"

export PYTHONPATH="$PROJ/src:$PROJ/third_party/py310deps"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
export ALFWORLD_GLIBC_COMPAT_DIR="$PROJ/third_party/glibc232"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "=== node=$(hostname) date=$(date) ==="
python -V
echo "PYTHONPATH=$PYTHONPATH"
echo "ALFWORLD_DATA=$ALFWORLD_DATA"

echo "=== 1) smoke: env ==="
python src/smoke_test.py --stage env --batch 2 --max-steps 4

echo "=== 2) eval: scripted, n=2 ==="
python src/eval_alfworld.py \
    --agent scripted \
    --split eval_out_of_distribution \
    --n 2 --batch 2 --max-steps 10 --tag probe

echo "=== 3) eval: echo, n=2 ==="
python src/eval_alfworld.py \
    --agent echo \
    --split eval_out_of_distribution \
    --n 2 --batch 2 --max-steps 10 --tag probe

echo "=== ALL PROBES DONE: $(date) ==="
