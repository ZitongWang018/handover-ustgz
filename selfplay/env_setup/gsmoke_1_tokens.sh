#!/bin/bash
# 步骤1：tokens 层（不用 GPU，但 stage 会构造 ALFWorld env，所以仍走 srun 计算节点）
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
python src/smoke_test.py --stage tokens --batch 4 --max-steps 8 --split train
echo "=== DONE tokens $(date) ==="
