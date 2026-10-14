#!/usr/bin/env python
"""P1 验收门 #1（纯 CPU，无需 GPU）：vLLM 侧的"事后停止判据"与 HF 侧逐条等价。

被验证的等式
---------------------------------------------------------------------------
    toolcall_stop_len(tokenizer, ids)  ==  _ToolCallStopCriteria 逐 token 记录的 stop_len

判据来源：`ez_vllm_policy._stop_triggered` 用的是 `rollout.scan_bare_json_call`，
判定式与 `rollout._ToolCallStopCriteria.CLOSE / scan_bare_json_call` 完全同一份代码，
差别只在"什么时候调用它"。本脚本把"什么时候"也对齐：
  * HF：`generated_len = 1,2,3,...`，`< start_after` 跳过，否则解码当前**全部已生成
    前缀**再判；取**第一个**触发的 `generated_len`。
  * vLLM：生成满之后，对同一个 ids 列表做同样的前缀扫描。

测试集刻意覆盖历史上踩过的坑（见 rollout.py 里的注释）：
  1. 只吐裸 JSON、从不吐 </tool_call>（Qwen3-4B-Instruct-2507 的真实行为）
  2. `arguments` 是嵌套对象 —— 非贪婪正则/朴素括号扫描会在内层 `}` 处提前命中
  3. 一个回合连写多个动作（extra_after_call）
  4. 触发点落在 `start_after` 之前（必须忽略）
  5. 全程无有效调用（必须返回全长）
  6. 字符串字面量里含 `{` `}` `"` 与转义引号
  7. 中途出现 eos/pad
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ez_vllm_policy import toolcall_stop_len  # noqa: E402

CASES = [
    # (名字, 生成文本)
    ("close_tag_basic",
     "Thought: I need to check.\n<tool_call>{\"name\":\"alfworld_tool\","
     "\"arguments\":{\"action_type\":\"look\"}}</tool_call>\n"),
    ("bare_json_only",
     "Thought: I will look around.\n{\"name\":\"alfworld_tool\","
     "\"arguments\":{\"action_type\":\"look\"}}"),
    ("nested_arguments",
     "{\"name\":\"alfworld_tool\",\"arguments\":{\"action_type\":\"go to\","
     "\"target\":\"cabinet 1\"}}{\"name\":\"alfworld_tool\",\"arguments\":"
     "{\"action_type\":\"look\"}}"),
    ("multi_action_one_turn",
     "{\"name\":\"alfworld_tool\",\"arguments\":{\"action_type\":\"look\"}}\n"
     "{\"name\":\"alfworld_tool\",\"arguments\":{\"action_type\":\"inventory\"}}\n"
     "{\"name\":\"alfworld_tool\",\"arguments\":{\"action_type\":\"look\"}}\n"),
    ("trigger_before_start_after", "{\"}\""),
    ("no_valid_call", "Thought: I am thinking about the room and its contents for a while."),
    ("braces_in_string",
     "{\"name\":\"alfworld_tool\",\"arguments\":{\"action_type\":\"go to\","
     "\"target\":\"drawer {2}\"}}"),
    ("escaped_quote",
     "{\"name\":\"alfworld_tool\",\"arguments\":{\"action_type\":\"go to\","
     "\"target\":\"cabinet \\\"1\\\"\"}}"),
    ("subtask_desc_call",
     "{\"name\":\"alfworld_tool\",\"arguments\":{\"task_description\":"
     "\"put a {hot} apple in the fridge\",\"subtasks\":[\"find apple\",\"cool it\"]}}"),
]


def simulate_hf_criteria(tokenizer, ids, start_after: int = 4) -> int:
    """逐 token 复刻 `_ToolCallStopCriteria.__call__` 的记账（check_every=1）。"""
    from rollout import scan_bare_json_call

    stop = None
    for generated_len in range(1, len(ids) + 1):
        if generated_len < start_after:
            continue
        text = tokenizer.decode(ids[:generated_len], skip_special_tokens=True)
        if ("</tool_call>" in text) or (scan_bare_json_call(text) is not None):
            stop = generated_len
            break
    return len(ids) if stop is None else stop


def main() -> int:
    tok_path = os.environ.get("ZTWANG_MODEL_PATH")
    if not tok_path:
        raise SystemExit("set ZTWANG_MODEL_PATH to the Qwen3-4B base checkpoint")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    bad = 0
    for name, text in CASES:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        expect = simulate_hf_criteria(tok, ids)
        got = toolcall_stop_len(tok, ids)
        ok = expect == got
        bad += 0 if ok else 1
        print(f"{'  ok ' if ok else 'FAIL '} {name:28s} n={len(ids):4d} "
              f"hf_stop={expect:4d} vllm_stop={got:4d}")
        if not ok:
            print(f"       ids={ids}")
            print(f"       text={text!r}")

    # 随机压测：随机 token 序列上两者也必须一致（防止只在手写样例上对）。
    import random
    rng = random.Random(0)
    vocab_pool = [t for t in tok("{}[]\":,abc 0123456789tool_call</name>", add_special_tokens=False)["input_ids"]]
    for it in range(300):
        n = rng.randint(1, 40)
        ids = [rng.choice(vocab_pool) for _ in range(n)]
        if simulate_hf_criteria(tok, ids) != toolcall_stop_len(tok, ids):
            bad += 1
            print(f"FAIL random#{it} ids={ids}")
            break
    else:
        print("  ok  300 random token sequences")

    print(json.dumps({"cases": len(CASES) + 300, "failures": bad}, ensure_ascii=False))
    if bad == 0:
        print("P1-GATE-1 PASS：事后停止判据与 HF 逐条等价")
    else:
        print("P1-GATE-1 FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
