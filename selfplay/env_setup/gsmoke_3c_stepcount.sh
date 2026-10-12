#!/bin/bash
# 步骤3c：dry-run 是否更新参数的"铁证"——在进程内数 optimizer.step() 调用次数。
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate thinktool
PROJ=$HOME/ztwang/selfplay; cd "$PROJ"; export PROJ
export PYTHONPATH="$PROJ/src:$PROJ/third_party/py310deps"
export ALFWORLD_DATA="$HOME/.cache/alfworld"
export ALFWORLD_GLIBC_COMPAT_DIR="$PROJ/third_party/glibc232"
export PYTHONUNBUFFERED=1; export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export ZTWANG_MODEL_PATH="$HOME/thinktool/models/qwen3_4b_inst_2507"
echo "=== node=$(hostname) date=$(date) ==="
python - <<'PY'
import sys, torch
sys.path.insert(0, "src")
import train_grpo

calls = {"n": 0}
orig = torch.optim.AdamW.step
def patched(self, *a, **k):
    calls["n"] += 1
    return orig(self, *a, **k)

def run(argv):
    calls["n"] = 0
    old = sys.argv
    sys.argv = ["train_grpo.py"] + argv
    try:
        train_grpo.main()
    finally:
        sys.argv = old
    return calls["n"]

torch.optim.AdamW.step = patched
try:
    n_dry = run(["--config", "configs/train_smoke.yaml", "--dry-run", "--max-iters", "1"])
    print(f"[stepcount] DRY-RUN   optimizer.step() 调用次数 = {n_dry}")
    n_real = run(["--config", "configs/train_smoke.yaml", "--max-iters", "1"])
    print(f"[stepcount] REAL      optimizer.step() 调用次数 = {n_real}")
    print(f"[stepcount] 结论: dry-run {'未' if n_dry == 0 else '【仍】'}更新参数；real {'有' if n_real > 0 else '无'}更新")
finally:
    torch.optim.AdamW.step = orig
PY
echo "=== DONE stepcount $(date) ==="
