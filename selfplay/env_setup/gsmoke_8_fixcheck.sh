#!/bin/bash
# 步骤2+4 补测（运行时绕过 bug #3/#4）：
#  (A) 用"合成 advantage"的极小 batch 证明 loss/backward 通路本身正常、grad_norm>0
#      —— 排除"组内奖励退化 -> advantage=0 -> grad=0"这个干扰因素。
#  (B) 吞吐/显存复测：NT=2 G=4 steps=30 new=96，micro=1（上一轮 micro=2 在 16k token 轨迹上 OOM）。
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate thinktool
PROJ=$HOME/ztwang/selfplay; cd "$PROJ"
export PYTHONPATH="$PROJ/src:$PROJ/third_party/py310deps"
export ALFWORLD_DATA="$HOME/.cache/alfworld"
export ALFWORLD_GLIBC_COMPAT_DIR="$PROJ/third_party/glibc232"
export PYTHONUNBUFFERED=1; export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ZTWANG_MODEL_PATH="$HOME/thinktool/models/qwen3_4b_inst_2507"
export GSMOKE_ATTN="${GSMOKE_ATTN:-flash_attention_2}"
echo "=== node=$(hostname) date=$(date) attn=$GSMOKE_ATTN ==="
python - <<'PY'
import os, time, math, random
import torch, yaml, numpy as np
from alfworld_compat import install as install_compat
install_compat(verbose=False)
from alfworld_env import ALFWorldBatchEnv, load_alfworld_config
from reward import make_reward_cfg, group_advantages, summarize_grouping, TrajectoryStats
from rollout import HFPolicy, RolloutConfig, Trajectory, run_rollouts
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

torch.cuda.reset_peak_memory_stats()
torch.cuda.set_device(0)
model, tok = build_model_and_tokenizer(cfg, 0, 1)
model = model.to(torch.device("cuda")); torch.cuda.synchronize()
patch_generate(model)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
device = next(model.parameters()).device
trainable = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(trainable, lr=cfg["train"]["lr"])
print(f"[env] device={device} attn={getattr(model.config,'_attn_implementation','?')}")

# ---------------- (A) 反向通路冒烟：合成 advantage ----------------
print("\n########## (A) backward 冒烟：合成 advantage=[1.0,-1.0] ##########")
tr = []
for i, adv in enumerate([1.0, -1.0]):
    ids = tok("Thought: look around the room.\nAction: look", add_special_tokens=False)["input_ids"]
    mask = [0] * 6 + [1] * (len(ids) - 6)
    tr.append(Trajectory(task_id=f"s{i}", task_type="s", game_file="s", messages=[],
                         input_ids=ids, loss_mask=mask, stats=TrajectoryStats(), advantage=adv))
bt = make_batch(tr, tok.pad_token_id, device)
with torch.no_grad():
    old = per_token_logps(model, bt.input_ids, bt.attention_mask)
loss, st = grpo_loss(model, bt, old, clip_eps=0.2)
loss.backward()
gn = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
print(f"[A] seq={tuple(bt.input_ids.shape)} active={bt.n_active} loss={loss.item():.5f} finite={math.isfinite(loss.item())}")
print(f"[A] grad_norm={gn:.4f}  >0且有限 = {(math.isfinite(gn) and gn > 0)}")
print(f"[A] stats={ {k: round(v,4) for k,v in st.items()} }")
opt.zero_grad(set_to_none=True)
peakA = torch.cuda.max_memory_allocated() / 1e9
print(f"[A] PEAK(alloc)={peakA:.2f} GB")

# ---------------- (B) 吞吐/显存：micro=1 ----------------
NT = cfg["env"]["n_tasks_per_iter"]; G = cfg["env"]["group_size"]
NSTEPS = cfg["env"]["max_episode_steps"]; MNT = cfg["rollout"]["max_new_tokens"]
MICRO = 1
n_slots = NT * G
alf = load_alfworld_config("configs/base_config.yaml")
print(f"\n########## (B) 吞吐 NT={NT} G={G} slots={n_slots} steps={NSTEPS} new={MNT} micro={MICRO} ##########")
t0 = time.time()
env = ALFWorldBatchEnv(alf, split="train", batch_size=n_slots, seed=42)
print(f"[B] env(batch{n_slots}) 构造 {time.time()-t0:.1f}s")
policy = HFPolicy(model, tok)
rew = make_reward_cfg(cfg["reward"])
rcfg = RolloutConfig(max_new_tokens=MNT, max_steps=NSTEPS, group_size=G,
                     temperature=cfg["rollout"]["temperature"], top_p=cfg["rollout"]["top_p"],
                     repetition_penalty=cfg["rollout"]["repetition_penalty"])

rows = []
for it in range(2):
    rng = random.Random(2000 + it)
    tasks = [{"game_file": env.game_files[rng.randrange(len(env.game_files))], "task_id": f"c{it}_{j}", "task_type": "c"}
             for j in range(NT)]
    t0 = time.time()
    trajs = run_rollouts(env, policy, tasks, rcfg, rew, log_fn=lambda m: print("     " + m))
    t_roll = time.time() - t0
    gids = [i // G for i in range(len(trajs))]
    for t, a in zip(trajs, group_advantages([t.reward for t in trajs], gids)):
        t.advantage = a
    gs = summarize_grouping([t.reward for t in trajs], gids)
    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    batches = []
    with torch.no_grad():
        for k in range(math.ceil(len(trajs) / MICRO)):
            chunk = trajs[k * MICRO:(k + 1) * MICRO]
            b = make_batch(chunk, pad_id=tok.pad_token_id, device=device)
            batches.append((b, per_token_logps(model, b.input_ids, b.attention_mask)))
    tot = sum(b.n_active for b, _ in batches) or 1
    lv = 0.0
    for b, o in batches:
        l, _ = grpo_loss(model, b, o, clip_eps=0.2)
        (l * (b.n_active / tot)).backward()
        lv += float(l.item()) * (b.n_active / tot)
    gn = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
    opt.step(); torch.cuda.synchronize()
    t_fb = time.time() - t0
    del batches
    peak_now = torch.cuda.max_memory_allocated() / 1e9
    rows.append((t_roll, t_fb, lv, gn, gs["reward_mean"], gs["group_std_mean"]))
    print(f"[B] iter{it}: rollout={t_roll:.1f}s fwd+bwd={t_fb:.1f}s loss={lv:.4f} grad_norm={gn:.3f} "
          f"reward_mean={gs['reward_mean']:.3f} std={gs['group_std_mean']:.3f} "
          f"turns={np.mean([t.n_turns for t in trajs]):.1f} toks={int(np.mean([t.n_tokens for t in trajs]))} "
          f"maxlen={max(t.n_tokens for t in trajs)} PEAK={peak_now:.1f}GB")
print(f"[B] rollout: " + ", ".join(f"{r[0]:.1f}" for r in rows) + "  fwd+bwd: " + ", ".join(f"{r[1]:.1f}" for r in rows))
print(f"[B] PEAK(alloc)={torch.cuda.max_memory_allocated()/1e9:.2f} GB reserved={torch.cuda.max_memory_reserved()/1e9:.2f} GB")
env.close()
print("\n*** 补测完成（bug #3/#4 运行时绕过）***")
PY
echo "=== DONE fixcheck $(date) ==="
