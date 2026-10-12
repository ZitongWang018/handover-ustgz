#!/bin/bash
# 步骤4：吞吐与显存测量。用 train_qwen3_4b.yaml 的默认规模：
#   n_tasks_per_iter=2, group_size=4, max_episode_steps=30, max_new_tokens=96, 跑 2 iter。
# 只测量，不改 src/（内联 python 复刻 train_grpo 的采样/训练块）。
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate thinktool
PROJ=$HOME/ztwang/selfplay; cd "$PROJ"
export PYTHONPATH="$PROJ/src:$PROJ/third_party/py310deps"
export ALFWORLD_DATA="$HOME/.cache/alfworld"
export ALFWORLD_GLIBC_COMPAT_DIR="$PROJ/third_party/glibc232"
export PYTHONUNBUFFERED=1; export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export ZTWANG_MODEL_PATH="$HOME/thinktool/models/qwen3_4b_inst_2507"
export GSMOKE_ATTN="${GSMOKE_ATTN:-flash_attention_2}"
echo "=== node=$(hostname) date=$(date) attn=$GSMOKE_ATTN ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || true
python - <<'PY'
import os, time, math
import torch, yaml, numpy as np
from alfworld_compat import install as install_compat
install_compat(verbose=False)
from alfworld_env import ALFWorldBatchEnv, load_alfworld_config
from reward import make_reward_cfg, group_advantages, summarize_grouping
from rollout import HFPolicy, RolloutConfig, run_rollouts
from train_grpo import build_model_and_tokenizer, make_batch, per_token_logps, grpo_loss

cfg = yaml.safe_load(open("configs/train_qwen3_4b.yaml"))
cfg["model"] = dict(cfg["model"])
cfg["model"]["path"] = os.environ["ZTWANG_MODEL_PATH"]
cfg["model"]["attn_implementation"] = os.environ.get("GSMOKE_ATTN", "flash_attention_2")
cfg["model"]["gradient_checkpointing"] = True

NT = cfg["env"]["n_tasks_per_iter"]      # 2
G = cfg["env"]["group_size"]             # 4
NSTEPS = cfg["env"]["max_episode_steps"] # 30
MNT = cfg["rollout"]["max_new_tokens"]   # 96
MICRO = cfg["train"].get("micro_batch_size", 2)
ITERS = 2
n_slots = NT * G
print(f"[tp] NT={NT} G={G} slots={n_slots} max_steps={NSTEPS} max_new_tokens={MNT} micro={MICRO}")

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
model, tok = build_model_and_tokenizer(cfg, 0, 1)
print(f"[tp] model load {time.time()-t0:.1f}s")
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
device = next(model.parameters()).device
print(f"[tp] attn_impl(model.config) = {getattr(model.config, '_attn_implementation', '?')}")

alf = load_alfworld_config("configs/base_config.yaml")
env = ALFWorldBatchEnv(alf, split="train", batch_size=n_slots, seed=42)
policy = HFPolicy(model, tok)
rcfg = RolloutConfig(**{**cfg["rollout"], "max_steps": NSTEPS, "group_size": G})
rew = make_reward_cfg(cfg["reward"])
trainable = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(trainable, lr=cfg["train"]["lr"])
import random
pool = env.game_files

rows = []
for it in range(ITERS):
    rng = random.Random(1000 + it)
    tasks = [{"game_file": pool[rng.randrange(len(pool))], "task_id": f"t{it}_{j}", "task_type": "probe"}
             for j in range(NT)]
    t0 = time.time()
    trajs = run_rollouts(env, policy, tasks, rcfg, rew, log_fn=lambda m: print("   " + m))
    t_roll = time.time() - t0
    advs = group_advantages([t.reward for t in trajs], [i // G for i in range(len(trajs))])
    for t, a in zip(trajs, advs):
        t.advantage = a
    gstats = summarize_grouping([t.reward for t in trajs], [i // G for i in range(len(trajs))])

    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    order = list(range(len(trajs)))
    n_micro = math.ceil(len(order) / MICRO)
    batches = []
    with torch.no_grad():
        for k in range(n_micro):
            chunk = [trajs[i] for i in order[k * MICRO:(k + 1) * MICRO]]
            bt = make_batch(chunk, pad_id=tok.pad_token_id, device=device)
            old = per_token_logps(model, bt.input_ids, bt.attention_mask)
            batches.append((bt, old))
    total_active = sum(bt.n_active for bt, _ in batches) or 1
    loss_val = 0.0
    for bt, old in batches:
        loss, st = grpo_loss(model, bt, old, clip_eps=0.2)
        (loss * (bt.n_active / total_active)).backward()
        loss_val += float(loss.item()) * (bt.n_active / total_active)
    gn = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
    opt.step()
    torch.cuda.synchronize()
    t_fb = time.time() - t0

    nturn = np.mean([t.n_turns for t in trajs]); ntok = int(np.mean([t.n_tokens for t in trajs]))
    rows.append((t_roll, t_fb, loss_val, gn, gstats["reward_mean"], nturn, ntok))
    print(f"[tp] iter{it}: rollout={t_roll:.1f}s fwd-bwd={t_fb:.1f}s loss={loss_val:.4f} "
          f"grad_norm={gn:.3f} reward_mean={gstats['reward_mean']:.3f} mean_turns={nturn:.1f} mean_tokens={ntok}")

peak = torch.cuda.max_memory_allocated() / 1e9
print(f"[tp] ==== SUMMARY ====")
print(f"[tp] iter wall (rollout+fwd-bwd): " + ", ".join(f"{r[0]+r[1]:.1f}s" for r in rows))
print(f"[tp] rollout sec : " + ", ".join(f"{r[0]:.1f}" for r in rows))
print(f"[tp] fwd-bwd sec : " + ", ".join(f"{r[1]:.1f}" for r in rows))
print(f"[tp] PEAK mem alloc : {peak:.2f} GB   reserved={torch.cuda.max_memory_reserved()/1e9:.2f} GB")
print(f"[tp] mean_turns : " + ", ".join(f"{r[5]:.1f}" for r in rows))
print(f"[tp] mean_tokens: " + ", ".join(str(r[6]) for r in rows))
env.close()
PY
echo "=== DONE throughput $(date) ==="
