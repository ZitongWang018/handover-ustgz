#!/bin/bash
# 步骤3b：dry-run（不更新参数）。torchrun 路径。
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate thinktool
PROJ=$HOME/ztwang/selfplay; cd "$PROJ"
export PYTHONPATH="$PROJ/src:$PROJ/third_party/py310deps"
export ALFWORLD_DATA="$HOME/.cache/alfworld"
export ALFWORLD_GLIBC_COMPAT_DIR="$PROJ/third_party/glibc232"
export PYTHONUNBUFFERED=1; export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export ZTWANG_MODEL_PATH="$HOME/thinktool/models/qwen3_4b_inst_2507"
echo "=== node=$(hostname) date=$(date) ==="
echo "############ torchrun --dry-run --max-iters 1 ############"
torchrun --nproc_per_node=1 --master_port=29513 src/train_grpo.py --config configs/train_smoke.yaml --dry-run --max-iters 1
echo "=== DONE dryrun $(date) ==="
