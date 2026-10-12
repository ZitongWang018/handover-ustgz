#!/bin/bash
# 步骤3（运行时绕过版）：真正跑 train_grpo.py 的循环，验证
#   (A) 4 iter 训练能跑完 + 每 iter 日志 + 断点"只留一个" + save_every=2 生效
#   (B) dry-run 不更新参数（数 optimizer.step() 调用次数）
# 绕过方式（**不改 src**）：
#   - Bug #4：把 build_model_and_tokenizer 包一层，world==1 时手动 .to(cuda)
#   - Bug #3：给 model.generate 包一层，把 stopping_criteria 包成 list
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
nvidia-smi --query-gpu=name,memory.total --format=csv || true
rm -f "$RUN/train_rank0.jsonl" "$RUN/history.json"
echo "--- runs dir BEFORE ---"; ls -la "$RUN/" 2>&1 || true

echo; echo "############ (A) 真实训练 4 iter ############"
python - <<'PY'
import sys, torch
sys.path.insert(0, "src")
import train_grpo

_orig_build = train_grpo.build_model_and_tokenizer
def _build(cfg, local_rank, world):
    m, tok = _orig_build(cfg, local_rank, world)
    core = m.module if hasattr(m, "module") else m
    if not hasattr(m, "module"):          # Bug #4: world==1 时 src 不会搬 GPU
        torch.cuda.set_device(0)
        m = m.to(torch.device("cuda"))
        core = m
        print("[patch] 手动 .to(cuda)")
    g = core.generate                     # Bug #3: stopping_criteria 要 list
    def _gen(*a, **kw):
        sc = kw.get("stopping_criteria", None)
        if sc is not None and not isinstance(sc, (list, tuple)):
            kw["stopping_criteria"] = [sc]
        return g(*a, **kw)
    core.generate = _gen
    return m, tok
train_grpo.build_model_and_tokenizer = _build

sys.argv = ["train_grpo.py", "--config", "configs/train_smoke.yaml"]
raise SystemExit(train_grpo.main())
PY

echo; echo "--- runs dir AFTER ---"; ls -la "$RUN/"
echo; echo "--- ckpt* 目录（应只有 1 个 ckpt，无 ckpt-2/ckpt-4/ckpt.tmp）---"
ls -ld "$RUN"/ckpt* 2>&1
echo; echo "--- ckpt 内容 ---"; ls -la "$RUN/ckpt/" 2>&1
echo; echo "--- train_state.json ---"; cat "$RUN/ckpt/train_state.json" 2>&1
echo; echo "--- history.json ---"; cat "$RUN/history.json" 2>&1
echo; echo "--- train_rank0.jsonl ---"; cat "$RUN/train_rank0.jsonl" 2>&1

echo; echo "############ (B) dry-run 是否更新参数（进程内数 optimizer.step 次数）############"
python - <<'PY'
import sys, torch
sys.path.insert(0, "src")
import train_grpo

_orig_build = train_grpo.build_model_and_tokenizer
def _build(cfg, local_rank, world):
    m, tok = _orig_build(cfg, local_rank, world)
    core = m.module if hasattr(m, "module") else m
    if not hasattr(m, "module"):
        torch.cuda.set_device(0)
        m = m.to(torch.device("cuda")); core = m
    g = core.generate
    def _gen(*a, **kw):
        sc = kw.get("stopping_criteria", None)
        if sc is not None and not isinstance(sc, (list, tuple)):
            kw["stopping_criteria"] = [sc]
        return g(*a, **kw)
    core.generate = _gen
    return m, tok
train_grpo.build_model_and_tokenizer = _build

calls = {"n": 0}
_orig_step = torch.optim.AdamW.step
def _step(self, *a, **k):
    calls["n"] += 1
    return _orig_step(self, *a, **k)

def run(argv):
    calls["n"] = 0
    old = sys.argv
    sys.argv = ["train_grpo.py", "--config", "configs/train_smoke.yaml"] + argv
    try:
        train_grpo.main()
    finally:
        sys.argv = old
    return calls["n"]

torch.optim.AdamW.step = _step
try:
    nd = run(["--dry-run", "--max-iters", "1"])
    print(f"[stepcount] DRY-RUN  optimizer.step() 调用次数 = {nd}")
    nr = run(["--max-iters", "1"])
    print(f"[stepcount] REAL     optimizer.step() 调用次数 = {nr}")
    print(f"[stepcount] 结论：dry-run {'未' if nd == 0 else '【仍】'}更新参数；real {'有' if nr > 0 else '无'}更新")
finally:
    torch.optim.AdamW.step = _orig_step
PY
echo "=== DONE train-patched $(date) ==="
