#!/bin/bash
# 步骤2：单卡 GPU --stage all。先试 flash_attention_2，失败退 sdpa。
# 附一个计时/显存探针（只用 gsmoke 脚本内联 python，不改 src/）。
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
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || true

echo; echo "############ [A] smoke_test.py --stage all (--attn flash_attention_2) ############"
ATTN=flash_attention_2
if ! python src/smoke_test.py --stage all --batch 4 --group-size 2 --max-steps 8 --attn "$ATTN"; then
  echo "!!! --attn flash_attention_2 失败，退回 sdpa 重试 !!!"
  ATTN=sdpa
  python src/smoke_test.py --stage all --batch 4 --group-size 2 --max-steps 8 --attn "$ATTN"
fi
echo "ATTN_USED=$ATTN"

echo; echo "############ [B] 计时/显存探针 (batch=4 group=2 max_steps=8) ############"
ATTN_USED="$ATTN" python - <<'PY'
import os, time, random
import torch, yaml, numpy as np

from alfworld_compat import install as install_compat
install_compat(verbose=False)
from alfworld_env import ALFWorldBatchEnv, load_alfworld_config
from reward import make_reward_cfg, group_advantages
from rollout import HFPolicy, RolloutConfig, run_rollouts
from train_grpo import build_model_and_tokenizer, make_batch, per_token_logps, grpo_loss

attn = os.environ.get("ATTN_USED", "flash_attention_2")
print(f"[probe] attn={attn} torch={torch.__version__} cuda={torch.version.cuda}")

tcfg = yaml.safe_load(open("configs/train_qwen3_4b.yaml"))
tcfg["model"] = dict(tcfg["model"])
tcfg["model"]["path"] = os.environ["ZTWANG_MODEL_PATH"]
tcfg["model"]["attn_implementation"] = attn
tcfg["model"]["gradient_checkpointing"] = True

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
model, tok = build_model_and_tokenizer(tcfg, 0, 1)
t_model = time.time() - t0
device = next(model.parameters()).device
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
print(f"[probe] model load        : {t_model:6.1f}s  params={sum(p.numel() for p in model.parameters())/1e9:.2f}B  dev={device}")

alf = load_alfworld_config("configs/base_config.yaml")
BATCH, G, NSTEPS, MNT = 4, 2, 8, 96
n_tasks = BATCH // G
env = ALFWorldBatchEnv(alf, split="train", batch_size=BATCH, seed=0)
tasks = [{"game_file": env.game_files[i % len(env.game_files)], "task_id": f"probe{i}", "task_type": "probe"}
         for i in range(n_tasks)]
policy = HFPolicy(model, tok)
rcfg = RolloutConfig(group_size=G, max_steps=NSTEPS, max_new_tokens=MNT, temperature=1.0)
rew = make_reward_cfg(None)

t0 = time.time()
trajs = run_rollouts(env, policy, tasks, rcfg, rew, log_fn=lambda m: print("   " + m))
t_roll = time.time() - t0
for t, a in zip(trajs, group_advantages([t.reward for t in trajs], [i // G for i in range(len(trajs))])):
    t.advantage = a

bt = make_batch(trajs, pad_id=tok.pad_token_id, device=device)
t0 = time.time()
with torch.no_grad():
    old = per_token_logps(model, bt.input_ids, bt.attention_mask)
torch.cuda.synchronize(); t_fwd = time.time() - t0
t0 = time.time()
loss, st = grpo_loss(model, bt, old, clip_eps=0.2)
torch.cuda.synchronize(); t_loss = time.time() - t0
t0 = time.time()
loss.backward()
torch.cuda.synchronize(); t_bwd = time.time() - t0
gn = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0))
peak = torch.cuda.max_memory_allocated() / 1e9

print(f"[probe] batch shape      : {tuple(bt.input_ids.shape)}  active_tokens={bt.n_active}")
print(f"[probe] loss             : {loss.item():.5f}  finite={torch.isfinite(loss).item()}")
print(f"[probe] grad_norm        : {gn:.4f}  finite={np.isfinite(gn) and gn > 0}")
print(f"[probe] rollout          : {t_roll:6.1f}s")
print(f"[probe] old-logp fwd     : {t_fwd:6.2f}s")
print(f"[probe] loss fwd         : {t_loss:6.2f}s")
print(f"[probe] backward         : {t_bwd:6.2f}s")
print(f"[probe] mean n_turns     : {np.mean([t.n_turns for t in trajs]):.2f}  mean n_tokens={int(np.mean([t.n_tokens for t in trajs]))}")
print(f"[probe] PEAK mem alloc   : {peak:.2f} GB")
print(f"[probe] reserved         : {torch.cuda.max_memory_reserved()/1e9:.2f} GB")
env.close()
PY
echo "=== DONE stage-all $(date) ==="
