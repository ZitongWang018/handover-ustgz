#!/bin/bash
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate thinktool
PROJ=$HOME/ztwang/selfplay; cd "$PROJ"
export PYTHONPATH="$PROJ/src:$PROJ/third_party/py310deps"
export ALFWORLD_DATA="$HOME/.cache/alfworld"
export ALFWORLD_GLIBC_COMPAT_DIR="$PROJ/third_party/glibc232"
export PYTHONUNBUFFERED=1
echo "=== node=$(hostname) date=$(date) ==="
python src/eval_alfworld.py --agent expert --split eval_out_of_distribution \
    --n 8 --batch 8 --max-steps 50 --tag expert
python src/eval_alfworld.py --agent expert --split eval_in_distribution \
    --n 8 --batch 8 --max-steps 50 --tag expert
echo "=== DONE $(date) ==="
