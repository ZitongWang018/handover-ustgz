#!/usr/bin/env python
"""P1 验收门 #2（需要 1 张 GPU）：同一份权重、同一批 prompt，HF 采样 vs vLLM 采样实测。

两种模式，故意跑成两个独立进程（互不污染显存/未初始化 CUDA 上下文）：
    --mode hf     走 rollout.HFPolicy（现状）
    --mode vllm   走 ez_vllm_policy.VLLMPolicy（新）

被测对象是**采样器本身**，prompt 由 `rollout.HFPolicy.initial_prompt_ids` 生成
（两个模式共用同一份代码 ⇒ 逐字节相同的输入），因此耗时差异 100% 来自采样实现。

指标：
    * wall_s / total_gen_tokens / tok_per_s —— 主指标
    * 生成长度直方图 —— 确认"事后截断"与 HF 的"边生成边判停"分布一致
    * well_formed 比例 —— 确认两边的样本性质可比（不是靠生成垃圾换速度）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rollout import HFPolicy, RolloutConfig  # noqa: E402


def build_cfg(max_new: int, max_batch: int) -> RolloutConfig:
    """完全对齐 ez_train.py 里的 s_cfg（solver 相位）。"""
    return RolloutConfig(
        group_size=1, max_steps=30, max_new_tokens=max_new,
        temperature=1.0, top_p=0.95, top_k=-1, min_p=0.0, repetition_penalty=1.05,
        max_context_tokens=40960, keep_recent_turns=12,
        stop_mode="tool_call", max_batch_size=max_batch,
    )


def synthetic_prompt_ids(policy: HFPolicy, n: int, target_tokens: int,
                         cfg: RolloutConfig):
    """造 n 条长度可控的 solver prompt（长度贴近真实：观测 + 若干轮工具结果）。"""
    from ez_prompts import build_solver_messages

    filler = ("The room is a standard ALFWorld room. You see a desk, a shelf, "
              "a cabinet, a drawer, a fridge, a table, a sink, a garbage can, "
              "a microwave, a countertop, a coffee machine, a toaster, an apple, "
              "a mug, a plate, a knife, a pot, a pan, and a spatula. ")
    ids_list = []
    for i in range(n):
        msg = build_solver_messages(filler, f"task #{i}: inspect the scene")
        base = policy.initial_prompt_ids(msg, cfg)
        while len(base) < target_tokens:
            msg = msg + [{"role": "user", "content": filler}]
            base = policy.render(msg)
            base = policy.tokenize_text(base)
        ids_list.append(base[:target_tokens] if len(base) > target_tokens else base)
    return ids_list


def summarize(tag: str, wall: float, gen: list, texts: list) -> dict:
    from ez_proposer import parse_tool_call
    from ez_solver import EzSolverTrajectory  # noqa: F401  (仅确认 import 面稳定)

    def _wellformed(t):
        try:
            return parse_tool_call(t).kind == "action"
        except Exception:
            return False

    n_tok = sum(len(g) for g in gen)
    lens = sorted(len(g) for g in gen)
    ok = sum(1 for t in texts if _wellformed(t))
    rec = {
        "mode": tag, "wall_s": round(wall, 2), "n_seqs": len(gen),
        "gen_tokens": n_tok, "tok_per_s": round(n_tok / max(wall, 1e-9), 1),
        "len_min": lens[0] if lens else 0, "len_p50": lens[len(lens) // 2] if lens else 0,
        "len_max": lens[-1] if lens else 0,
        "well_formed": f"{ok}/{len(texts)}",
    }
    print("[bench] " + json.dumps(rec, ensure_ascii=False))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hf", "vllm", "both"], required=True)
    ap.add_argument("--base", default=os.environ.get("ZTWANG_MODEL_PATH"))
    ap.add_argument("--slots", type=int, default=16, help="每次生成的序列数（=每 rank slot 数）")
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--prompt-tokens", type=int, default=2500)
    ap.add_argument("--repeat", type=int, default=3, help="重复轮数（RL 是逐回合反复调用的）")
    ap.add_argument("--gpu-mem", type=float, default=0.55)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()
    if not args.base:
        raise SystemExit("set ZTWANG_MODEL_PATH or --base")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    probe = HFPolicy(None, tok)          # 只用它的分词面
    cfg0 = build_cfg(args.max_new, 0)
    ids_list = synthetic_prompt_ids(probe, args.slots, args.prompt_tokens, cfg0)
    print(f"[bench] slots={args.slots} prompt_tokens~{sum(len(x) for x in ids_list)//len(ids_list)} "
          f"max_new={args.max_new} repeat={args.repeat}")

    results = []
    if args.mode in ("hf", "both"):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            args.base, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
            device_map="cuda:0", trust_remote_code=True)
        lc = LoraConfig(r=32, lora_alpha=32, lora_dropout=0.0, bias="none",
                        target_modules="all-linear", task_type="CAUSAL_LM")
        model = get_peft_model(model, lc, adapter_name="solver")
        model.add_adapter("proposer", lc)
        model.set_adapter("solver")
        model.eval()
        for mb in (0, 8):
            pol = HFPolicy(model, tok)
            cfg = build_cfg(args.max_new, mb)
            t0 = time.time()
            texts, gen, _ = [], [], []
            for _ in range(args.repeat):
                texts, gen, _ = pol.generate_from_token_ids(ids_list, cfg)
            results.append(summarize(f"hf_maxbatch{mb}", time.time() - t0, gen, texts))
        del model
        torch.cuda.empty_cache()

    if args.mode in ("vllm", "both"):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        from ez_vllm_policy import VLLMPolicy, build_vllm_engine, export_lora_adapter

        # 导出与训练侧同构的 adapter（零初始化 = 恒等，等价于"尚未学到东西"的初始状态）
        cpu = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16,
                                                  trust_remote_code=True)
        lc = LoraConfig(r=32, lora_alpha=32, lora_dropout=0.0, bias="none",
                        target_modules="all-linear", task_type="CAUSAL_LM")
        cpu = get_peft_model(cpu, lc, adapter_name="solver")
        adir = export_lora_adapter(cpu, "solver",
                                   os.path.abspath("vllm_lora_bench/solver"))
        print(f"[bench] exported adapter -> {adir}")
        del cpu
        import gc
        gc.collect()

        eng = build_vllm_engine(args.base, gpu_memory_utilization=args.gpu_mem)
        pol = VLLMPolicy(eng, tok, log_fn=lambda m: print(m))
        pol.set_lora("solver", adir)
        cfg = build_cfg(args.max_new, 0)
        t0 = time.time()
        texts, gen, _ = [], [], []
        for _ in range(args.repeat):
            texts, gen, _ = pol.generate_from_token_ids(ids_list, cfg)
        results.append(summarize("vllm", time.time() - t0, gen, texts))

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
