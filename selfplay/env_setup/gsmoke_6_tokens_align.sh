#!/bin/bash
# 步骤1b：定位 build_token_sequence 的 token 对齐是否错位（会污染训练目标）。
# 手法：用"真实生成口径"的 gen_ids（= 在生成 prompt 之后紧接着的 token），
#       再走一遍 build_token_sequence，看 mask=1 的片段是否 == 内容 + <|im_end|>\n。
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
policy = HFPolicy.__new__(HFPolicy); policy.tokenizer = tok; policy.enable_thinking = False

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
asst_idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]

# --- 1) 确认 assistant 头的 token 数 H（add_generation_prompt 会插入 <|im_start|>assistant\n）---
for i in asst_idx:
    a = len(policy.tokenize_text(policy.render(msgs[:i], add_generation_prompt=False)))
    b = len(policy.tokenize_text(policy.render(msgs[:i], add_generation_prompt=True)))
    hdr = policy.render(msgs[:i], add_generation_prompt=True)[len(policy.render(msgs[:i], add_generation_prompt=False)):]
    print(f"[H] assistant@{i}: header_tokens H={b - a}  header_str={hdr!r}")

# --- 2) 造"真实生成口径"的 gen_ids：生成 prompt 之后紧跟的 token ---
gen_real = []
for i in asst_idx:
    gp = policy.render(msgs[:i], add_generation_prompt=True)
    with_c = gp + msgs[i]["content"]
    gen_real.append(policy.tokenize_text(with_c)[len(policy.tokenize_text(gp)):])
print(f"[gen] 真实口径 gen 长度={[len(g) for g in gen_real]}")

ids, mask, bounds = build_token_sequence(msgs, gen_real, policy)
masked = tok.decode([t for t, m in zip(ids, mask) if m == 1], skip_special_tokens=False)
expected = "".join(msgs[i]["content"] + "<|im_end|>\n" for i in asst_idx)
print("[out] mask=1 片段 :", repr(masked))
print("[out] 期望片段    :", repr(expected))
print(f"[out] 完全一致    = {masked == expected}")

# --- 3) 直接看：把 prev_len 换成"含 assistant 头"的量，是否就对了 ---
#     复刻 build_token_sequence 的 tail 计算，两种口径对比
prev_len = len(policy.tokenize_text(policy.render(msgs[:asst_idx[0]], add_generation_prompt=False)))
after = policy.tokenize_text(policy.render(msgs[:asst_idx[0] + 1], add_generation_prompt=False))
tail_bug = after[prev_len + len(gen_real[0]):]
tail_fix = after[prev_len + (len(policy.tokenize_text(policy.render(msgs[:asst_idx[0]], add_generation_prompt=True))) - prev_len) + len(gen_real[0]):]
print(f"[tail] 现口径 tail = {tok.decode(tail_bug, skip_special_tokens=False)!r}")
print(f"[tail] 修正 tail   = {tok.decode(tail_fix, skip_special_tokens=False)!r}")
PY
echo "=== DONE tokens-align $(date) ==="
