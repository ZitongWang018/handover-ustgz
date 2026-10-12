#!/usr/bin/env python
"""2×2×2 全组合实验：精确定位"生成变乱码"到底是哪个开关干的。

背景
----
probe_policy_fmt.py（用 train_grpo.build_model_and_tokenizer 的配置）吐的是：
    'Thoughts = 1999999999999999...'   /  'Thoughts = $pessoa = 34月结元光...'
probe_sanity.py（干净配置）吐的是：
    'Thought: I need to look around ... \nAction: look'      ← 完美
权重 NaN=0 Inf=0，模型本身是好的。所以差异一定在这三个开关上：
    F1 model.train() vs model.eval()
    F2 use_cache True vs False
    F3 gradient_checkpointing 开 vs 关
本脚本把 8 种组合全跑一遍，看哪种组合开始变乱码。

用法
----
    python env_setup/probe_gckpt.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ / "src"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from prompts import match_admissible, parse_action  # noqa: E402

MODEL = os.environ.get("ZTWANG_MODEL_PATH", str(Path.home() / "thinktool/models/qwen3_4b_inst_2507"))
ADMISSIBLE = ["go to cabinet 1", "go to cabinet 2", "go to countertop 1", "go to garbagecan 1",
              "go to sinkbasin 1", "go to toilet 1", "go to toiletpaperhanger 1", "go to towelholder 1",
              "inventory", "look"]
SYSTEM = ("You are an expert agent solving household tasks in the ALFWorld text environment.\n\n"
          "At each turn you receive:\n1. The task you must complete.\n"
          "2. The current observation (what you see / what just happened).\n"
          "3. A list of Admissible Commands - this is the ONLY action space.\n\n"
          "Rules:\n- You must output exactly two lines and nothing else.\n"
          "- Line 1: \"Thought: \" followed by one short sentence of reasoning.\n"
          "- Line 2: \"Action: \" followed by exactly one command copied verbatim from the Admissible Commands list.\n"
          "- Never invent commands, never output more than one action, never add explanations after the action.\n\n"
          "Example of a valid response:\n"
          "Thought: I should first look around to find the objects mentioned in the task.\nAction: look")
USER = ("Your task is to: put some toiletpaper on toiletpaperhanger.\n\n"
        "Observation: You are in the middle of a room. Looking quickly around you, you see a cabinet 4, "
        "a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 1, "
        "a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.\n\n"
        "Admissible Commands:\n" + "\n".join(f"- {c}" for c in ADMISSIBLE))


def main() -> int:
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    print(f"[gckpt] prompt {len(enc['input_ids'][0])} tokens")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    print("[gckpt] 基础模型加载完成（默认 eval / use_cache=True / 无 GC）\n")

    results = []
    header = f"{'#':<3}{'train':<7}{'cache':<7}{'GC':<5}{'状态':<6}{'n_tok':<7}{'Action 解析':<28}输出前 70 字符"
    print(header)
    print("-" * len(header))

    n = 0
    for train_mode in (False, True):
        for use_cache in (True, False):
            for gc in (False, True):
                n += 1
                # --- 设置三个开关 ---
                if gc:
                    model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False})
                else:
                    model.gradient_checkpointing_disable()
                model.config.use_cache = use_cache
                if train_mode:
                    model.train()
                else:
                    model.eval()

                # 注意：transformers 的 generate() 在 is_gradient_checkpointing 时
                # 会**强制**把 use_cache 改成 False —— 这正是真实训练配置的处境。
                forced = gc and use_cache
                try:
                    with torch.no_grad():
                        o = model.generate(
                            **enc, do_sample=False, max_new_tokens=128,
                            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                            use_cache=use_cache)
                    ids = o[0, enc["input_ids"].shape[1]:].tolist()
                    while ids and ids[-1] == tok.eos_token_id:
                        ids.pop()
                    text = tok.decode(ids, skip_special_tokens=True)
                except Exception as e:  # noqa: BLE001
                    text = f"<EXCEPTION {type(e).__name__}: {e}>"
                    ids = []

                p = parse_action(text)
                m = match_admissible(p.action, ADMISSIBLE)
                verdict = (" OK " if (p.has_marker and m) else "乱码") if ids else "异常"
                marker = f"->{m}" if m else (p.action[:20] or "(空)")
                flag = f"{'train' if train_mode else 'eval ':<6}{'T' if use_cache else 'F':<6}" \
                       f"{'T' if gc else 'F':<5}{'强制F' if forced else '    ':<5}"
                print(f"{n:<3}{flag}{len(ids):<7}{marker:<28}{text[:70]!r}")
                results.append({"n": n, "train": train_mode, "use_cache": use_cache, "gc": gc,
                                "forced_no_cache": forced, "text": text, "n_tok": len(ids),
                                "action": p.action, "matched": m, "ok": bool(p.has_marker and m)})
                sys.stdout.flush()

    print("\n" + "=" * 74)
    print("结论")
    print("=" * 74)
    good = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    print(f"正常 {len(good)}/8   乱码 {len(bad)}/8")
    for r in good:
        print(f"  ✅ #{r['n']} train={r['train']} cache={r['use_cache']} GC={r['gc']}")
    for r in bad:
        print(f"  ❌ #{r['n']} train={r['train']} cache={r['use_cache']} GC={r['gc']}"
              f"{' (GC 强制关了 cache)' if r['forced_no_cache'] else ''}")
    if good and bad:
        print("\n把所有'正常'组合取交集，剩下的就是必须保持的设置：")
        for key, label in (("train", "model.train()"), ("use_cache", "use_cache=True"),
                           ("gc", "gradient_checkpointing")):
            vals = {r[key] for r in good}
            if len(vals) == 1:
                v = vals.pop()
                print(f"  → {label} 必须 = {v}")

    import json
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "probe_gckpt.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
