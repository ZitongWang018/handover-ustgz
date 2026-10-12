#!/usr/bin/env python
"""最小健全性探针：区分「模型坏了」还是「我们的生成配置坏了」。

背景
----
格式探针（probe_policy_fmt.py）实测到模型吐的是纯乱码：
    A1 greedy  -> 'Thoughts = 1999999999999999...'   256 token 截断，无 Action: 行
    A2 T=1.0   -> 'Thoughts = $pessoa = 34月结元光...'  同样无 Action: 行
`19999999...` 这种重复数字**非常像噪声**，不像"没听懂 prompt"。
所以先别急着改 prompt —— 先确认权重本身是不是好的。

本脚本按「从最底层往上」的顺序查，任何一步炸就停在那一步：
    L0  safetensors 原始文件的张量统计（不经过 transformers）
    L1  from_pretrained 之后的权重统计（NaN / Inf / 全零？）
    L2  lm_head / embed_tokens 是否正常（tie_word_embeddings=true，只存一份）
    L3  **干净配置**生成（use_cache=True、不开 gradient checkpointing）
        —— 与 probe_policy_fmt 的配置（gradient_checkpointing=true -> use_cache=False）对比
    L4  三个难度递增的 prompt：2+2 / 复述 / ALFWorld 真实 prompt

用法
----
    python env_setup/probe_sanity.py
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

MODEL = os.environ.get("ZTWANG_MODEL_PATH", str(Path.home() / "thinktool/models/qwen3_4b_inst_2507"))


def hr(t: str) -> None:
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


def stats(t: torch.Tensor, name: str) -> None:
    tf = t.detach().float()
    nan = int(torch.isnan(tf).sum())
    inf = int(torch.isinf(tf).sum())
    print(f"  {name:<44} shape={tuple(t.shape)} dtype={t.dtype} "
          f"mean={tf.mean():+.6f} std={tf.std():.6f} "
          f"min={tf.min():+.5f} max={tf.max():+.5f} NaN={nan} Inf={inf} "
          f"{'  <-- 全零!' if tf.abs().max() == 0 else ''}")


def main() -> int:
    print(f"[sanity] model = {MODEL}")

    # ---------------- L0: 直接读 safetensors 文件 ----------------
    hr("L0  safetensors 原始张量统计（不经过 transformers）")
    try:
        from safetensors import safe_open
        idx_path = Path(MODEL) / "model.safetensors.index.json"
        shards = None
        if idx_path.exists():
            import json
            wmap = json.loads(idx_path.read_text())["weight_map"]
            shards = sorted({v for v in wmap.values()})
        else:
            shards = [p.name for p in sorted(Path(MODEL).glob("*.safetensors"))]
        print(f"  shards = {shards}")
        for sh in shards[:2]:
            with safe_open(str(Path(MODEL) / sh), framework="pt") as f:
                keys = list(f.keys())
                print(f"  -- {sh} ({len(keys)} tensors) 前 3 个：")
                for k in keys[:3]:
                    stats(f.get_tensor(k), k)
                # 明确找一下 embedding
                emb = [k for k in keys if "embed_tokens" in k]
                for k in emb[:1]:
                    stats(f.get_tensor(k), k)
    except Exception as e:  # noqa: BLE001
        print(f"  [L0] 失败：{type(e).__name__}: {e}")

    # ---------------- L1: from_pretrained ----------------
    hr("L1  加载（干净配置：不 gradient_checkpointing，use_cache 保持 True）")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.cuda().eval()
    print(f"  device={next(model.parameters()).device}  dtype={next(model.parameters()).dtype}")
    print(f"  config: {model.config.model_type} hidden={model.config.hidden_size} "
          f"layers={model.config.num_hidden_layers} vocab={model.config.vocab_size} "
          f"tie={getattr(model.config, 'tie_word_embeddings', None)}")

    hr("L2  加载后的权重统计")
    named = list(model.named_parameters())
    print(f"  共 {len(named)} 个参数张量，抽查：")
    for n, p in named[:2]:
        stats(p, n)
    for n, p in named[len(named) // 2: len(named) // 2 + 1]:
        stats(p, n)
    for n, p in named[-2:]:
        stats(p, n)
    for want in ["model.embed_tokens.weight", "lm_head.weight"]:
        hit = [p for n, p in named if n == want]
        if hit:
            stats(hit[0], want)
    tot_nan = sum(int(torch.isnan(p.detach().float()).sum()) for _, p in named)
    tot_inf = sum(int(torch.isinf(p.detach().float()).sum()) for _, p in named)
    print(f"  >>> 全部参数 NaN 总数={tot_nan}  Inf 总数={tot_inf}")

    # ---------------- L3/L4: 干净生成 ----------------
    def gen(prompt: str, max_new: int = 64, tag: str = "") -> str:
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            o = model.generate(**enc, do_sample=False, max_new_tokens=max_new,
                               pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        ids = o[0, enc["input_ids"].shape[1]:].tolist()
        while ids and ids[-1] == tok.eos_token_id:
            ids.pop()
        text = tok.decode(ids, skip_special_tokens=True)
        print(f"\n  [{tag}] n_tok={len(ids)}")
        print(f"    {text!r}")
        return text

    hr("L3  干净配置生成（use_cache=True，无 gradient checkpointing）")
    gen("Question: What is 2+2? Answer with a single number.", 32, "L3a 英语算术")

    msgs = [{"role": "user", "content": "What is 2+2? Answer with a single number."}]
    tpl = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print(f"\n  chat 模板渲染结果：{tpl!r}")
    gen(tpl, 32, "L3b 走 chat 模板")

    hr("L4  ALFWorld 真实 prompt（同一份，与格式探针一致）")
    sys.path.insert(0, str(PROJ / "src"))
    from prompts import build_system_prompt  # noqa: E402
    sysmod = sys.modules.get("prompts")
    alf_msgs = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": "Your task is to: put some toiletpaper on toiletpaperhanger.\n\n"
                                    "Observation: You are in the middle of a room.\n\n"
                                    "Admissible Commands:\n- go to cabinet 1\n- go to countertop 1\n"
                                    "- inventory\n- look"},
    ]
    tpl2 = tok.apply_chat_template(alf_msgs, tokenize=False, add_generation_prompt=True)
    gen(tpl2, 128, "L4 ALFWorld")

    hr("结论提示")
    print("  判断顺序：")
    print("  1) L0/L2 出现全零或 NaN/Inf -> 模型文件坏了，换模型，改 prompt 没用")
    print("  2) L3a/L3b 也乱码          -> 模型/tokenizer 不匹配，或权重坏")
    print("  3) L3 正常、L4 乱码         -> 才是 prompt / 解析 / 生成参数的问题")
    print("  4) L4 正常（探针里乱码）    -> 是 probe_policy_fmt 里 gradient_checkpointing")
    print("                              导致 use_cache=False 的问题，改训练配置即可")
    return 0


if __name__ == "__main__":
    sys.exit(main())
