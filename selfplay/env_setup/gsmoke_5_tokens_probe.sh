#!/bin/bash
# 步骤1（等价探针）：不依赖 GPU / 不依赖 RandomPolicy，直接验证 chat 模板渲染 + loss mask 拼接。
# 背景：官方 `smoke_test.py --stage tokens` 会崩 —— stage_rollout 用 RandomPolicy（没有
#       render/tokenize_text），而 run_rollouts 默认 tokenize=True，在 build_token_sequence 里
#       AttributeError。这里用 stage_tokens 同款的"假 HFPolicy（只挂 tokenizer）"独立复核这一步。
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate thinktool
PROJ=$HOME/ztwang/selfplay; cd "$PROJ"
export PYTHONPATH="$PROJ/src:$PROJ/third_party/py310deps"
export PYTHONUNBUFFERED=1; export TOKENIZERS_PARALLELISM=false
export ZTWANG_MODEL_PATH="$HOME/thinktool/models/qwen3_4b_inst_2507"
echo "=== node=$(hostname) date=$(date) ==="
python - <<'PY'
import os
from transformers import AutoTokenizer
from rollout import HFPolicy, build_token_sequence
from prompts import build_system_prompt, format_observation, format_turn_feedback

tok = AutoTokenizer.from_pretrained(os.environ["ZTWANG_MODEL_PATH"], trust_remote_code=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
print(f"[tok] vocab={tok.vocab_size} pad={tok.pad_token!r} eos={tok.eos_token!r}")

policy = HFPolicy.__new__(HFPolicy)     # 只借 render/tokenize_text，不加载模型
policy.tokenizer = tok
policy.enable_thinking = False

msgs = [
    {"role": "system", "content": build_system_prompt()},
    {"role": "user", "content": format_observation(
        "You are in the middle of a room. Looking quickly around you, you see a bathtubbasin 1, a cart 1.",
        ["go to bathtubbasin 1", "go to cart 1", "look"], first_turn=True,
        task_desc="put some toiletpaper on toiletpaperhanger.")},
    {"role": "assistant", "content": "Thought: I should find the hanger.\nAction: go to cart 1"},
    {"role": "user", "content": format_turn_feedback(
        "go to cart 1", "On the cart 1, you see a toiletpaper 1, a soapbar 1.",
        ["take toiletpaper 1 from cart 1", "look"], done=False, won=False)},
    {"role": "assistant", "content": "Thought: take the toilet paper.\nAction: take toiletpaper 1 from cart 1"},
]

full = policy.render(msgs, add_generation_prompt=False)
print(f"[tmpl] render 长度 {len(full)} 字符；含 im_start={('<|im_start|>' in full)} im_end={('<|im_end|>' in full)}")

# 前缀稳定性（build_token_sequence 的立身之本）：逐步渲染的 token id 必须是前一步的前缀
prev, ok = [], True
for i in range(1, len(msgs) + 1):
    cur = policy.tokenize_text(policy.render(msgs[:i], add_generation_prompt=False))
    if cur[:len(prev)] != prev:
        ok = False
        print(f"[tmpl] !!! 前缀不稳定 at msg {i}")
        break
    prev = cur
print(f"[tmpl] 逐步渲染前缀稳定 = {ok}")

# assistant 的真实生成 id（模拟模型产出：直接 tokenize 文本）
gen_ids = [policy.tokenize_text(m["content"]) for m in msgs if m["role"] == "assistant"]
ids, mask, bounds = build_token_sequence(msgs, gen_ids, policy)

assert len(ids) == len(mask), "input_ids 与 loss_mask 长度必须一致"
n_asst = sum(1 for m in msgs if m["role"] == "assistant")
print(f"[mask] n_tokens={len(ids)} mask_sum={sum(mask)} boundaries={len(bounds)} (assistant 轮数={n_asst})")
print(f"[mask] 生成 token 数={[len(g) for g in gen_ids]}  boundaries={bounds}")
assert len(bounds) == n_asst, "turn_boundaries 数量应等于 assistant 轮数"
assert sum(mask) >= sum(len(g) for g in gen_ids), "mask 至少应覆盖全部生成 token"

# 被 mask 的 token（应=两条 assistant 内容+模板尾标记）与未 mask 的（应=system/user 骨架）
masked = tok.decode([t for t, m in zip(ids, mask) if m == 1], skip_special_tokens=False)
unmasked = tok.decode([t for t, m in zip(ids, mask) if m == 0], skip_special_tokens=True)
print("[mask] === 被训练(mask=1) 的片段 ===")
print(repr(masked))
print("[mask] === 不训练(mask=0) 的片段(截断) ===")
print(repr(unmasked[:400]))
assert "Thought: I should find the hanger." in masked
assert "take toiletpaper 1 from cart 1" in masked
assert "Observation:" in unmasked and "Observation:" not in masked, "训练片段里不该出现环境观测"
print("[mask] OK：mask=1 只覆盖 assistant 产出，环境观测/system 全为 0")
print("*** tokens 层（chat 模板 + loss mask）验证通过 ***")
PY
echo "=== DONE tokens-probe $(date) ==="
