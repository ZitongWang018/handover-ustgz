#!/bin/bash
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate thinktool
PROJ=$HOME/ztwang/selfplay; cd "$PROJ"
export PYTHONPATH="$PROJ/src:$PROJ/third_party/py310deps"
export ALFWORLD_DATA="$HOME/.cache/alfworld"
export ALFWORLD_GLIBC_COMPAT_DIR="$PROJ/third_party/glibc232"
export PYTHONUNBUFFERED=1
echo "=== node=$(hostname) date=$(date) ==="
# 非 expert 路径回归：run_rollouts 仍能解 5 元组 reset，且不含 expert 字段
python src/eval_alfworld.py --agent echo --split eval_out_of_distribution \
    --n 4 --batch 4 --max-steps 4 --tag noexpert
python src/eval_alfworld.py --agent random --split eval_out_of_distribution \
    --n 8 --batch 8 --max-steps 10 --tag noexpert
echo "=== DONE $(date) ==="
