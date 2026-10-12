#!/usr/bin/env python
"""诊断：proposer 为什么 0/N accepted（smoke 实测 fmt_bad=6 = 6 次生成全部解析失败）。

冒烟日志证据：
    [proposer] 0/2 accepted | mean_turns=3.0 | fmt_bad=6 | dropped_sub=0 | 19.4s
6 次 generate（G=2 × turns=3）**没有一次**产出 <tool_call>{...}</tool_call>。

本脚本把三件事一次打出来，用来区分三种可能：
  A. prompt 本身有问题（打印渲染后的 prompt 与 env_desc）
  B. 生成质量/停止条件问题（打印原始输出 + repr）
  C. **GC 兜底在 peft 包装下失效** —— rollout.py 明确记录过
     "GC 开着时 Qwen3 的 generate 输出乱码（'Thoughts = 19999...'）"，
     而 GC 的判断/关闭走的是 model.is_gradient_checkpointing /
     model.gradient_checkpointing_disable()，EZ 的 model 被 peft 包了一层，
     这两个调用是否还落在真正的 HF 模型上必须实测。

用法（集群上，1 张卡）：
    python src/ez_debug_proposer.py --config configs/ez_qwen3_4b.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alfworld_compat import install as install_alfworld_compat  # noqa: E402

install_alfworld_compat()

import torch  # noqa: E402

from alfworld_env import ALFWorldBatchEnv, collect_game_files, load_alfworld_config  # noqa: E402
from ez_prompts import build_proposer_messages  # noqa: E402
from ez_proposer import parse_tool_call  # noqa: E402
from rollout import HFPolicy, RolloutConfig, gc_set  # noqa: E402
from train_grpo import build_model_and_tokenizer, load_cfg  # noqa: E402


def hr(t: str) -> None:
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


def gen_once(policy, messages, *, max_new_tokens, stop_mode, label):
    cfg = RolloutConfig(group_size=1, max_steps=4, max_new_tokens=max_new_tokens,
                        temperature=1.0, top_p=0.95, repetition_penalty=1.05,
                        max_context_tokens=40960, keep_recent_turns=40,
                        stop_mode=stop_mode)
    texts, gen_ids, stop = policy.generate([messages], cfg)
    raw = texts[0]
    parsed = parse_tool_call(raw)
    hr(f"[{label}] max_new_tokens={max_new_tokens} stop_mode={stop_mode!r}")
    print(f"生成 token 数 = {len(gen_ids[0])}  stop_len = {stop}")
    print(f"parse 结果 = {parsed.kind!r}  action={parsed.action!r} "
          f"desc={parsed.task_description!r}")
    print("--- raw 原文 ---")
    print(raw)
    print("--- repr（看有没有乱码/异常字符）---")
    print(repr(raw[:1200]))
    return parsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--game-idx", type=int, default=0)
    ap.add_argument("--no-lora", action="store_true",
                    help="跳过 build_dual_lora（排除 peft 包装的影响）")
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    # ---------- 环境 ----------
    alf_cfg = load_alfworld_config(
        cfg["env"]["config_path"] if os.path.isabs(cfg["env"]["config_path"])
        else str((SRC_DIR.parent / cfg["env"]["config_path"]).resolve()))
    games = collect_game_files(alf_cfg, split=cfg["env"].get("split", "train"))
    game = sorted(games)[args.game_idx]
    hr("[env] 环境")
    print(f"train games = {len(games)}\ngame = {game}")
    env = ALFWorldBatchEnv(alf_cfg, split=cfg["env"].get("split", "train"),
                           batch_size=1, max_episode_steps=4, with_facts=True)
    obs, views, adm, used, _ = env.reset([game])
    v = views[0]
    print(f"\n--- env_desc（送进 prompt 的场景描述）---\n{(v.env_desc or v.raw)[:2000]}")
    print(f"\n--- admissible（前 10 条）---\n{list(adm[0])[:10]}")

    # ---------- 模型 ----------
    hr("[model] 加载")
    model, tokenizer = build_model_and_tokenizer(cfg, 0, 1)
    print(f"裸 HF 模型 dtype = {next(model.parameters()).dtype}")
    print(f"裸 HF 模型 is_gradient_checkpointing = "
          f"{getattr(model, 'is_gradient_checkpointing', '?')}")
    if not args.no_lora:
        from ez_train import build_dual_lora
        model = build_dual_lora(model, cfg)
        model.set_adapter("proposer")
        inner = model
        for attr in ("base_model", "model"):
            if hasattr(inner, attr):
                inner = getattr(inner, attr)
        print(f"\npeft 包装后：")
        print(f"  model.__class__ = {type(model).__name__}")
        print(f"  model.is_gradient_checkpointing = "
              f"{getattr(model, 'is_gradient_checkpointing', '?')}")
        print(f"  最内层 {type(inner).__name__}.is_gradient_checkpointing = "
              f"{getattr(inner, 'is_gradient_checkpointing', '?')}")

    # ★ 关键检查：rollout 的 GC 兜底能不能真的关掉 GC？
    hr("[GC] gc_set(model, False) 是否真的把 GC 关掉了？（peft 包装下的失效点）")
    print(f"  调用前 model.is_gradient_checkpointing      = {getattr(model, 'is_gradient_checkpointing', '?')}")
    moved = gc_set(model, False)
    print(f"  gc_set 返回（是否真的动手）= {moved}")
    print(f"  调用后 model.is_gradient_checkpointing      = {getattr(model, 'is_gradient_checkpointing', '?')}")
    if not args.no_lora:
        inner2 = model
        for attr in ("base_model", "model"):
            if hasattr(inner2, attr):
                inner2 = getattr(inner2, attr)
        print(f"  调用后最内层 .is_gradient_checkpointing     = {getattr(inner2, 'is_gradient_checkpointing', '?')}")

    policy = HFPolicy(model, tokenizer)

    # ---------- prompt ----------
    messages = build_proposer_messages(v.env_desc or v.raw, cfg["proposer"].get("max_turns", 30), ())
    hr("[prompt] 渲染后的完整 prompt（前 3000 字符）")
    print(policy.render(messages, add_generation_prompt=True)[:3000])

    def gc_state(tag):
        print(f"  [{tag}] model.is_gradient_checkpointing = "
              f"{getattr(model, 'is_gradient_checkpointing', '?')}")

    def innermost():
        m = model
        for attr in ("base_model", "model"):
            if hasattr(m, attr):
                m = getattr(m, attr)
        return m

    # ---------- 生成对比 ----------
    # ⚠️ 顺序很重要：A 必须在 GC 的**初始状态**下跑，否则复现不出线上现象。
    p_max = cfg["proposer"].get("max_new_tokens", 160)
    hr("[A] 现场配置（GC 初始状态 → generate 内部自己走 gc_set 兜底）")
    gc_state("A 前")
    torch.manual_seed(0)
    gen_once(policy, messages, max_new_tokens=p_max, stop_mode="tool_call",
             label="A 现场配置")
    gc_state("A 后（如果这里还是 True，说明 rollout 的 GC 兜底在 peft 下失效了）")

    print(f"\n  手动 gc_set(model, False) 是否动手 = {gc_set(model, False)}")
    gc_state("手动关后")
    if not args.no_lora:
        print(f"  最内层 {type(innermost()).__name__}.is_gradient_checkpointing = "
              f"{getattr(innermost(), 'is_gradient_checkpointing', '?')}")

    hr("[B] 放宽到 512 token（看是不是被 160 截断）")
    torch.manual_seed(0)
    gen_once(policy, messages, max_new_tokens=512, stop_mode="tool_call",
             label="B 512 token")

    hr("[C] 512 token + 不用 tool_call 停止条件")
    torch.manual_seed(0)
    gen_once(policy, messages, max_new_tokens=512, stop_mode="action_line",
             label="C action_line")

    hr("[D] 强制在最内层关 GC 后重跑 512 token（直接验证 GC 假设）")
    for m in (innermost(), model, model.base_model if hasattr(model, "base_model") else model):
        if hasattr(m, "gradient_checkpointing_disable"):
            m.gradient_checkpointing_disable()
    print(f"  最内层 GC = {getattr(innermost(), 'is_gradient_checkpointing', '?')}")
    torch.manual_seed(0)
    gen_once(policy, messages, max_new_tokens=512, stop_mode="tool_call",
             label="D 最内层关 GC")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
