#!/bin/bash
# 步骤3补充：只做 dry-run vs real 的"是否更新参数"对比。
# 目的：gsmoke_11 的 [stepcount] 输出被 env 退出时的 BrokenPipeError 噪声淹没。
# 手段（**不改 src**）：
#   - Bug #4：build_model_and_tokenizer 包一层，world==1 手动 .to(cuda)
#   - Bug #3：core.generate 包一层，stopping_criteria 包成 list
# 额外：对少量参数做 before/after 指纹（比只数 step 更硬），在 build 时取 before。
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate thinktool
PROJ=$HOME/ztwang/selfplay; cd "$PROJ"
export PYTHONPATH="$PROJ/src:$PROJ/third_party/py310deps"
export ALFWORLD_DATA="$HOME/.cache/alfworld"
export ALFWORLD_GLIBC_COMPAT_DIR="$PROJ/third_party/glibc232"
export PYTHONUNBUFFERED=1; export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export ZTWANG_MODEL_PATH="$HOME/thinktool/models/qwen3_4b_inst_2507"
RES=$PROJ/runs/stepcount_result.txt
echo "=== node=$(hostname) date=$(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv || true

python - > "$RES" 2> "$RES.err" <<'PY'
import sys, torch
sys.path.insert(0, "src")
import train_grpo

STATE = {"params": [], "fp_build": None}

def fp():
    # 轻量指纹：前 8 个参数各取前 2048 个元素的 float64 和
    s = 0.0
    for p in STATE["params"][:8]:
        s += float(p.detach().reshape(-1)[:2048].float().sum().item())
    return s

_orig_build = train_grpo.build_model_and_tokenizer
def _build(cfg, local_rank, world):
    m, tok = _orig_build(cfg, local_rank, world)
    core = m.module if hasattr(m, "module") else m
    if not hasattr(m, "module"):                 # Bug #4
        torch.cuda.set_device(0)
        m = m.to(torch.device("cuda")); core = m
    g = core.generate                            # Bug #3
    def _gen(*a, **kw):
        sc = kw.get("stopping_criteria", None)
        if sc is not None and not isinstance(sc, (list, tuple)):
            kw["stopping_criteria"] = [sc]
        return g(*a, **kw)
    core.generate = _gen
    STATE["params"] = list(core.parameters())
    STATE["fp_build"] = fp()                     # before 指纹（build 完成时）
    return m, tok
train_grpo.build_model_and_tokenizer = _build

calls = {"step": 0, "zero": 0, "backward": 0}
_orig_step = torch.optim.AdamW.step
_orig_zero = torch.optim.AdamW.zero_grad
_orig_bwd  = torch.Tensor.backward
torch.optim.AdamW.step = lambda self, *a, **k: (calls.__setitem__("step", calls["step"]+1), _orig_step(self, *a, **k))[1]
torch.optim.AdamW.zero_grad = lambda self, *a, **k: (calls.__setitem__("zero", calls["zero"]+1), _orig_zero(self, *a, **k))[1]
torch.Tensor.backward = lambda self, *a, **k: (calls.__setitem__("backward", calls["backward"]+1), _orig_bwd(self, *a, **k))[1]

def run(argv, tag):
    calls["step"] = calls["zero"] = calls["backward"] = 0
    old = sys.argv
    sys.argv = ["train_grpo.py", "--config", "configs/train_smoke.yaml"] + argv
    try:
        train_grpo.main()
    finally:
        sys.argv = old
    fp_after = fp()
    changed = STATE["fp_build"] is not None and abs(fp_after - STATE["fp_build"]) > 1e-9
    print(f"[{tag}] optimizer.step={calls['step']}  zero_grad={calls['zero']}  backward={calls['backward']}")
    print(f"[{tag}] param_fp build={STATE['fp_build']!r}  after={fp_after!r}  changed={changed}")
    STATE["fp_build"] = None
    return calls["step"], changed

nd, cd = run(["--dry-run", "--max-iters", "1"], "DRY-RUN")
nr, cr = run(["--max-iters", "1"], "REAL")
print(f"[结论] DRY-RUN: step={nd}, backward={0 if nd==0 else '?'}, 参数被改动={cd}")
print(f"[结论] REAL   : step={nr}, 参数被改动={cr}")
print(f"[结论] dry-run {'不更新参数 ✓' if (nd == 0 and not cd) else '异常：疑似更新了参数 ✗'}")
PY

echo "--- stepcount_result.txt ---"
cat "$RES"
echo "--- stderr 头部40行（BrokenPipe 噪声在尾部，忽略）---"
head -40 "$RES.err" 2>&1
echo "=== DONE stepcount $(date) ==="
