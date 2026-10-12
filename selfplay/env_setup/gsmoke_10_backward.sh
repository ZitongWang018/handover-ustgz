#!/bin/bash
# 极简 backward 定论测试（**不构造 ALFWorld 环境**，只加载模型，1~2 分钟）：
#   用两条**不同文本**的合成轨迹 + 人为 advantage，验证 grpo_loss / backward / 剪裁通路正常、grad_norm>0。
#   顺带给出 loss 的解析期望值作对照（ratio=1 时 loss=-mean(adv)）。
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate thinktool
PROJ=$HOME/ztwang/selfplay; cd "$PROJ"
export PYTHONPATH="$PROJ/src:$PROJ/third_party/py310deps"
export ALFWORLD_DATA="$HOME/.cache/alfworld"
export ALFWORLD_GLIBC_COMPAT_DIR="$PROJ/third_party/glibc232"
export PYTHONUNBUFFERED=1; export TOKENIZERS_PARALLELISM=false
export ZTWANG_MODEL_PATH="$HOME/thinktool/models/qwen3_4b_inst_2507"
export GSMOKE_ATTN="${GSMOKE_ATTN:-flash_attention_2}"
echo "=== node=$(hostname) date=$(date) attn=$GSMOKE_ATTN ==="
python - <<'PY'
import os, math, torch, yaml
from alfworld_compat import install as install_compat
install_compat(verbose=False)
from reward import TrajectoryStats
from rollout import Trajectory
from train_grpo import build_model_and_tokenizer, make_batch, per_token_logps, grpo_loss

def patch_generate(model):
    orig = model.generate
    def gen(*a, **kw):
        sc = kw.get("stopping_criteria", None)
        if sc is not None and not isinstance(sc, (list, tuple)):
            kw["stopping_criteria"] = [sc]
        return orig(*a, **kw)
    model.generate = gen
    return model

cfg = yaml.safe_load(open("configs/train_qwen3_4b.yaml"))
cfg["model"] = dict(cfg["model"])
cfg["model"]["path"] = os.environ["ZTWANG_MODEL_PATH"]
cfg["model"]["attn_implementation"] = os.environ.get("GSMOKE_ATTN", "flash_attention_2")
cfg["model"]["gradient_checkpointing"] = True

torch.cuda.set_device(0)
torch.cuda.reset_peak_memory_stats()
model, tok = build_model_and_tokenizer(cfg, 0, 1)
model = model.to(torch.device("cuda")); torch.cuda.synchronize()
patch_generate(model)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
device = next(model.parameters()).device
trainable = [p for p in model.parameters() if p.requires_grad]

TEXTS = [
    "Thought: I will look around the room to find the object.\nAction: look",
    "Thought: Perhaps the mug is inside the cabinet, let me go there.\nAction: go to cabinet 1",
]
def mk(advs):
    tr = []
    for i, (txt, adv) in enumerate(zip(TEXTS, advs)):
        ids = tok(txt, add_special_tokens=False)["input_ids"]
        mask = [0] * 5 + [1] * (len(ids) - 5)
        tr.append(Trajectory(task_id=f"s{i}", task_type="s", game_file="s", messages=[],
                             input_ids=ids, loss_mask=mask, stats=TrajectoryStats(), advantage=adv))
    return tr

for tag, advs in [("case1 adv=[+1,+1] 期望 loss≈-1", [1.0, 1.0]),
                  ("case2 adv=[+1,-1] 期望 loss≈ 0", [1.0, -1.0])]:
    tr = mk(advs)
    bt = make_batch(tr, pad_id=tok.pad_token_id, device=device)
    with torch.no_grad():
        old = per_token_logps(model, bt.input_ids, bt.attention_mask)
    loss, st = grpo_loss(model, bt, old, clip_eps=0.2)
    loss.backward()
    gn = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
    print(f"[{tag}] seq={tuple(bt.input_ids.shape)} active={bt.n_active} "
          f"loss={loss.item():.5f} finite={math.isfinite(loss.item())} grad_norm={gn:.6f} "
          f">0且有限={(math.isfinite(gn) and gn > 0)} stats={ {k: round(v,4) for k,v in st.items()} }")
    model.zero_grad(set_to_none=True)

print(f"[mem] PEAK(alloc)={torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print("*** 若两行均为 finite、grad_norm>0，则 loss/backward/剪裁通路本身正常 ***")
PY
echo "=== DONE backward $(date) ==="
