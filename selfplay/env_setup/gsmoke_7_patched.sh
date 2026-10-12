#!/bin/bash
# 步骤2+4（运行时绕过版）：src bug #3 让 HFPolicy.generate 必崩
#   （transformers 4.57 要求 stopping_criteria 是 list，代码传的是单个对象）。
# 这里**不改 src**，只在进程内给 model.generate 打个补丁把 stopping_criteria 包成 list，
# 从而先把「GPU 前反向通路是否通 + loss/grad_norm/峰值显存/各段耗时 + 吞吐」测出来。
# 注意：数值来自"等价补丁"，仅用于定参；正式结论仍以修好 src 后的运行为准。
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
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python - <<'PY'
import os, time, math
import torch, yaml, numpy as np
from alfworld_compat import install as install_compat
install_compat(verbose=False)
from alfworld_env import ALFWorldBatchEnv, load_alfworld_config
from reward import make_reward_cfg, group_advantages, summarize_grouping
from rollout import HFPolicy, RolloutConfig, run_rollouts
from train_grpo import build_model_and_tokenizer, make_batch, per_token_logps, grpo_loss


def patch_generate(model):
    """运行时绕过 src bug #3：把 stopping_criteria 包成 list。"""
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
# ---- 运行时绕过 src bug #4：world==1 时 build_model_and_tokenizer 从不 .to(cuda) ----
torch.cuda.set_device(0)
t0 = time.time()
model, tok = build_model_and_tokenizer(cfg, 0, 1)
t_model = time.time() - t0
print(f"[env] 模型 from_pretrained {t_model:.1f}s  当前设备={next(model.parameters()).device}（world==1 时 src 不会搬到 GPU）")
t0 = time.time()
model = model.to(torch.device("cuda"))
torch.cuda.synchronize()
print(f"[env] 手动 .to(cuda) {time.time()-t0:.1f}s")
patch_generate(model)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
device = next(model.parameters()).device
attn_used = getattr(model.config, "_attn_implementation", "?")
print(f"[env] 设备={device}  attn_impl={attn_used}  params={sum(p.numel() for p in model.parameters())/1e9:.2f}B")

alf = load_alfworld_config("configs/base_config.yaml")
policy = HFPolicy(model, tok)
rew = make_reward_cfg(cfg["reward"])
trainable = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(trainable, lr=cfg["train"]["lr"])


def run_one_iter(env, tasks, group_size, max_steps, max_new_tokens, micro, lr):
    rcfg = RolloutConfig(max_new_tokens=max_new_tokens, max_steps=max_steps, group_size=group_size,
                         temperature=cfg["rollout"]["temperature"], top_p=cfg["rollout"]["top_p"],
                         repetition_penalty=cfg["rollout"]["repetition_penalty"])
    t0 = time.time()
    trajs = run_rollouts(env, policy, tasks, rcfg, rew, log_fn=lambda m: print("     " + m))
    t_roll = time.time() - t0
    gids = [i // group_size for i in range(len(trajs))]
    for t, a in zip(trajs, group_advantages([t.reward for t in trajs], gids)):
        t.advantage = a
    gstats = summarize_grouping([t.reward for t in trajs], gids)

    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    order = list(range(len(trajs)))
    n_micro = math.ceil(len(order) / micro)
    batches = []
    with torch.no_grad():
        for k in range(n_micro):
            chunk = [trajs[i] for i in order[k * micro:(k + 1) * micro]]
            bt = make_batch(chunk, pad_id=tok.pad_token_id, device=device)
            batches.append((bt, per_token_logps(model, bt.input_ids, bt.attention_mask)))
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
    del batches
    return dict(t_roll=t_roll, t_fb=t_fb, loss=loss_val, gn=gn,
                reward_mean=gstats["reward_mean"], group_std=gstats["group_std_mean"],
                degen=gstats["group_degenerate_frac"],
                turns=float(np.mean([t.n_turns for t in trajs])),
                toks=int(np.mean([t.n_tokens for t in trajs])))


# ============ (A) stage-all 等价：batch=4 group=2 steps=8 ============
print("\n########## (A) stage-all 等价 batch=4 group=2 max_steps=8 max_new=96 ##########")
t0 = time.time()
env4 = ALFWorldBatchEnv(alf, split="train", batch_size=4, seed=0)
print(f"[A] env(batch4) 构造 {time.time()-t0:.1f}s  池={len(env4.game_files)}  max_steps={env4.max_episode_steps}")
tasks4 = [{"game_file": env4.game_files[i % len(env4.game_files)], "task_id": f"a{i}", "task_type": "a"} for i in range(2)]
rA = run_one_iter(env4, tasks4, 2, 8, 96, 2, cfg["train"]["lr"])
print(f"[A] loss={rA['loss']:.5f} finite={math.isfinite(rA['loss'])} grad_norm={rA['gn']:.4f} "
      f"finite&>0={(math.isfinite(rA['gn']) and rA['gn'] > 0)}")
print(f"[A] rollout={rA['t_roll']:.1f}s fwd+bwd={rA['t_fb']:.1f}s turns={rA['turns']:.1f} toks={rA['toks']}")
peakA = torch.cuda.max_memory_allocated() / 1e9
print(f"[A] PEAK mem(alloc)={peakA:.2f} GB   reserved={torch.cuda.max_memory_reserved()/1e9:.2f} GB")
env4.close()

# ============ (B) 吞吐测量：n_tasks=2 group=4 steps=30 max_new=96 跑 2 iter ============
NT = cfg["env"]["n_tasks_per_iter"]; G = cfg["env"]["group_size"]
NSTEPS = cfg["env"]["max_episode_steps"]; MNT = cfg["rollout"]["max_new_tokens"]
MICRO = cfg["train"].get("micro_batch_size", 2)
n_slots = NT * G
print(f"\n########## (B) 吞吐 NT={NT} G={G} slots={n_slots} max_steps={NSTEPS} max_new={MNT} micro={MICRO} ##########")
t0 = time.time()
envB = ALFWorldBatchEnv(alf, split="train", batch_size=n_slots, seed=42)
print(f"[B] env(batch{n_slots}) 构造 {time.time()-t0:.1f}s")
import random
rows = []
for it in range(2):
    rng = random.Random(1000 + it)
    tasks = [{"game_file": envB.game_files[rng.randrange(len(envB.game_files))], "task_id": f"b{it}_{j}", "task_type": "b"}
             for j in range(NT)]
    r = run_one_iter(envB, tasks, G, NSTEPS, MNT, MICRO, cfg["train"]["lr"])
    rows.append(r)
    print(f"[B] iter{it}: rollout={r['t_roll']:.1f}s fwd+bwd={r['t_fb']:.1f}s loss={r['loss']:.4f} "
          f"grad_norm={r['gn']:.3f} reward_mean={r['reward_mean']:.3f} group_std={r['group_std']:.3f} "
          f"degen={r['degen']:.2f} turns={r['turns']:.1f} toks={r['toks']}")
peakB = torch.cuda.max_memory_allocated() / 1e9
print(f"[B] iter wall: " + ", ".join(f"{r['t_roll']+r['t_fb']:.1f}s" for r in rows))
print(f"[B] rollout: " + ", ".join(f"{r['t_roll']:.1f}" for r in rows) + "  fwd+bwd: " + ", ".join(f"{r['t_fb']:.1f}" for r in rows))
print(f"[B] PEAK mem(alloc)={peakB:.2f} GB  reserved={torch.cuda.max_memory_reserved()/1e9:.2f} GB")
print(f"[B] mean_turns: " + ", ".join(f"{r['turns']:.1f}" for r in rows) + "  mean_tokens: " + ", ".join(str(r['toks']) for r in rows))
envB.close()
print("\n*** 运行时绕过探针完成（src bug #3 未修）***")
PY
echo "=== DONE patched-probe $(date) ==="
