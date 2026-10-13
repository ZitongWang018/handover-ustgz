"""vLLM PoC（P0）：加载 Qwen3-4B，跑一批 **token-in/token-out** 生成，量吞吐与显存。

为什么用 token-in/token-out：EZ 的训练序列要求 assistant 段用模型**真实生成的 token id**
（见 rollout.build_token_sequence），重新套 chat template 会改变训练输入。所以 vLLM 后端
必须支持 prompt_token_ids 进、token_ids 出。

用法（在二期 8 卡节点上、用独立 venv）：
    ~/ztwang/vllm_venv/bin/python scripts/ez_vllm_poc.py \
        --model ~/thinktool/models/qwen3_4b --n 32 --max-new-tokens 96

对照：HF 版单步的实测数字（TPS=1，30 turns）在 docs/EZ_VLLM_ACCEL_PLAN.md §0。
"""
from __future__ import annotations

import argparse
import time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF base model path（qwen3_4b）")
    ap.add_argument("--n", type=int, default=32, help="并发序列数（≥ 一次 rollout 的 slot 数）")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--tp", type=int, default=1, help="tensor_parallel_size（单卡=1）")
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--prefix-caching", action="store_true", default=True)
    ap.add_argument("--wo-prefix-caching", dest="prefix_caching", action="store_false")
    ap.add_argument("--dump", default=None, help="把结果 json 写到该路径")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        enable_prefix_caching=args.prefix_caching,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        trust_remote_code=True,
    )
    load_s = time.time() - t0
    print(f"[load] {load_s:.1f}s  prefix_caching={args.prefix_caching} tp={args.tp}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # 模拟多轮 rollout 的共享前缀：同一段 system+user 骨架，尾部各自不同
    shared = tok("You are an AI assistant playing ALFWorld, a text-based household "
                 "environment.\n" * 8)["input_ids"]
    prompts = []
    for i in range(args.n):
        tail = tok(f"\nSession {i}. React with one action.")["input_ids"]
        prompts.append({"prompt_token_ids": list(shared) + list(tail)})

    sp = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=args.max_new_tokens)

    # 冷启动一次（编译 / 抓取 kernel），再计时
    llm.generate(prompts[: min(2, args.n)], sp)
    torch.cuda.synchronize()

    t1 = time.time()
    outs = llm.generate(prompts, sp)
    torch.cuda.synchronize()
    dt = time.time() - t1
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30

    print(f"[gen] n={args.n} max_new={args.max_new_tokens} time={dt:.2f}s "
          f"tot_tokens={ntok} throughput={ntok / dt:.0f} tok/s", flush=True)
    print(f"[gen] per-seq latency={dt / args.n * 1000:.0f} ms", flush=True)
    print(f"[mem] peak_allocated={peak:.2f} GiB", flush=True)
    print("[sample]", repr(tok.decode(outs[0].outputs[0].token_ids)[:200]), flush=True)

    if args.dump:
        import json
        with open(args.dump, "w") as fh:
            json.dump({"n": args.n, "max_new": args.max_new_tokens, "load_s": load_s,
                       "gen_s": dt, "tot_tokens": ntok, "tok_per_s": ntok / dt,
                       "peak_gib": peak, "prefix_caching": args.prefix_caching}, fh, indent=2)
        print(f"[dump] {args.dump}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
