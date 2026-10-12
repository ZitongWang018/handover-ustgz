#!/bin/bash
# 步骤3：真正的训练循环（单卡，4 iter）。顺带验证断点"只留一个"。
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate thinktool
PROJ=$HOME/ztwang/selfplay; cd "$PROJ"
export PYTHONPATH="$PROJ/src:$PROJ/third_party/py310deps"
export ALFWORLD_DATA="$HOME/.cache/alfworld"
export ALFWORLD_GLIBC_COMPAT_DIR="$PROJ/third_party/glibc232"
export PYTHONUNBUFFERED=1; export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export ZTWANG_MODEL_PATH="$HOME/thinktool/models/qwen3_4b_inst_2507"
RUN=$PROJ/runs/qwen3-4b-alfworld-grpo
echo "=== node=$(hostname) date=$(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || true

rm -f "$RUN/train_rank0.jsonl" "$RUN/history.json"
echo "--- runs dir BEFORE ---"; ls -la "$RUN/" 2>&1 || true

echo; echo "############ torchrun 4 iter (train_smoke.yaml) ############"
torchrun --nproc_per_node=1 --master_port=29511 src/train_grpo.py --config configs/train_smoke.yaml

echo; echo "--- runs dir AFTER ---"; ls -la "$RUN/"
echo; echo "--- ckpt* 目录（应只有 1 个 ckpt，无 ckpt-2/ckpt-4/ckpt.tmp）---"
ls -ld "$RUN"/ckpt* 2>&1
echo; echo "--- ckpt 内容 ---"; ls -la "$RUN/ckpt/" 2>&1
echo; echo "--- train_state.json ---"; cat "$RUN/ckpt/train_state.json" 2>&1
echo; echo "--- history.json ---"; cat "$RUN/history.json" 2>&1
echo; echo "--- train_rank0.jsonl (逐 iter) ---"; cat "$RUN/train_rank0.jsonl" 2>&1
echo "=== DONE train $(date) ==="
