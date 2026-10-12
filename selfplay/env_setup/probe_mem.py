#!/usr/bin/env python
"""显存 + 参数更新有效性探针（必须在 A800 80G 上跑，A40 48G 装不下）。

要回答两个直接关系到"8 卡作业会不会白跑"的问题：

Q1  fp32 参数（修 bf16 舍入问题所必需）在单卡 80G 上放得下吗？
    静态: params 16.08G + grads 16.08G + AdamW(fp32 m,v) 32.16G = 64.32G
    激活: GC + bf16 autocast + W=8192 B=1
          logits bf16 2.49G + log_softmax 分块 fp32 反向 4.98G + 层输入 1.5G
    估算合计 ~75G / 80G —— 很紧，必须实测。

Q2  fp32 下 AdamW(lr=1e-6) 的更新还被舍入抹掉吗？
    对比 bf16 跑一遍，看 `update_rel` 是不是 0。

用法
----
    python env_setup/probe_mem.py            # 两种 dtype 都跑
    python env_setup/probe_mem.py --only fp32
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ / "src"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from train_grpo import _autocast_bf16  # noqa: E402

MODEL = os.environ.get("ZTWANG_MODEL_PATH", str(Path.home() / "thinktool/models/qwen3_4b_inst_2507"))
W = int(os.environ.get("PROBE_W", "8192"))      # 序列长度（= max_context_tokens 上限）
B = int(os.environ.get("PROBE_B", "1"))         # micro_batch_size
LR = float(os.environ.get("PROBE_LR", "1e-6"))
CHUNK = 512


def hr(t: str) -> None:
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def gb(n: float) -> str:
    return f"{n / 1024 ** 3:.2f}G"


def run_one(dtype_name: str) -> dict:
    dt = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype_name]
    hr(f"=== dtype={dtype_name} ===")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    res: dict = {"dtype": dtype_name}

    base = torch.cuda.memory_allocated()
    print(f"起始占用            : {gb(base)}")

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=dt, attn_implementation="sdpa").cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    n_param = sum(p.numel() for p in model.parameters())
    print(f"参数量              : {n_param / 1e9:.3f}B")
    print(f"加载模型后          : {gb(torch.cuda.memory_allocated())}  "
          f"(理论 params {gb(n_param * dt.itemsize)})")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    print(f"建 AdamW 后         : {gb(torch.cuda.memory_allocated())}  (state 还没分配)")

    # ---- 构造一批假数据，形状与真实 micro batch 一致 ----
    input_ids = torch.randint(0, 100000, (B, W), device="cuda")
    attn = torch.ones_like(input_ids)
    print(f"构造输入 B={B} W={W} 后: {gb(torch.cuda.memory_allocated())}")

    with _autocast_bf16(model):
        out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    logits = out.logits[:, :-1, :]
    print(f"forward 后          : {gb(torch.cuda.memory_allocated())}  "
          f"logits={tuple(logits.shape)} {logits.dtype}")

    # 复刻 per_token_logps 的分块 log-softmax
    targets = input_ids[:, 1:]
    parts = []
    for s in range(0, logits.size(1), CHUNK):
        lg = logits[:, s:s + CHUNK, :].float()
        tgt = targets[:, s:s + CHUNK]
        parts.append(lg.log_softmax(dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1))
    lp = torch.cat(parts, dim=1)
    print(f"log_softmax 后      : {gb(torch.cuda.memory_allocated())}")

    loss = -lp.mean()
    loss.backward()
    print(f"backward 后         : {gb(torch.cuda.memory_allocated())}")

    probe = [p for p in model.parameters() if p.requires_grad]
    probe = probe[len(probe) // 2]
    before = probe.detach().clone()
    optimizer.step()
    with torch.no_grad():
        delta = float((probe.detach() - before).norm().item())
        pnorm = float(probe.detach().norm().item())
    update_rel = delta / max(pnorm, 1e-12)

    print(f"optimizer.step 后   : {gb(torch.cuda.memory_allocated())}")
    peak = torch.cuda.max_memory_allocated()
    total = torch.cuda.get_device_properties(0).total_memory
    print(f"【峰值显存】         : {gb(peak)} / 卡总量 {gb(total)}  "
          f"({peak / total:.1%})")
    print(f"【参数更新】         : |Δ|={delta:.6e}  |w|={pnorm:.6e}  "
          f"update_rel={update_rel:.3e}")
    print(f"                      -> {'✅ 参数真的动了' if update_rel > 0 else '❌ 参数没动（更新被舍入抹掉）'}")

    res.update({"n_param_B": n_param / 1e9, "peak_gb": peak / 1024 ** 3,
                "total_gb": total / 1024 ** 3, "peak_pct": peak / total,
                "update_rel": update_rel, "delta": delta, "pnorm": pnorm})

    del model, optimizer, out, logits, lp, loss, input_ids, attn, before
    gc.collect()
    torch.cuda.empty_cache()
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["fp32", "bf16"], default=None)
    args = ap.parse_args()

    print(f"[mem] 设备: {torch.cuda.get_device_name(0)}")
    print(f"[mem] 模型: {MODEL}")
    print(f"[mem] 形状: B={B} W={W}  分块={CHUNK}  lr={LR}")

    results = []
    for d in (["fp32", "bf16"] if args.only is None else [args.only]):
        try:
            results.append(run_one(d))
        except torch.cuda.OutOfMemoryError as e:
            print(f"\n❌ dtype={d} 显存不足: {str(e)[:300]}")
            torch.cuda.empty_cache()
            gc.collect()
            results.append({"dtype": d, "oom": True})

    hr("最终结论")
    print(f"{'dtype':<8}{'峰值':<12}{'占比':<10}{'update_rel':<16}{'判定':<20}")
    for r in results:
        if r.get("oom"):
            print(f"{r['dtype']:<8}{'OOM':<12}{'-':<10}{'-':<16}{'❌ 放不下':<20}")
            continue
        ok = r["update_rel"] > 0
        print(f"{r['dtype']:<8}{gb(r['peak_gb']):<12}{r['peak_pct']:.1%:<10}"
              f"{r['update_rel']:.3e}    {'✅ 可更新' if ok else '❌ 更新被抹掉':<20}")
    fp32 = next((r for r in results if r.get("dtype") == "fp32"), None)
    if fp32 and not fp32.get("oom"):
        verdict = ("✅ fp32 放得下，正式训练用 fp32"
                   if fp32["peak_pct"] < 0.92 else
                   f"⚠️ fp32 峰值已到 {fp32['peak_pct']:.1%}，余量太小，"
                   "建议减小 max_context_tokens / keep_recent_turns，或换 Adafactor/8-bit 优化器")
        print(f"\n{verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
