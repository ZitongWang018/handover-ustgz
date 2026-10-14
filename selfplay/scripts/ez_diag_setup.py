#!/usr/bin/env python
"""EZ 评测"为什么是 0"的现场诊断。

背景：论文摘要原文说基座 Qwen3-4B 在 ALFWorld 上 **0.17**，我们实测 **0.00**（0/32）。
32 连败在 p=0.17 下的概率只有 0.8^32 ≈ 0.26% —— **统计上不相容**，
说明某处一阶设置和论文不一样。本脚本把最可能的几处一次性查清。

CPU 部分（不需要 GPU）：
  1) 基座到底是 base 还是 instruct（看 model card / config / 是否有 SFT 痕迹）
  2) chat template 渲染出来的真实 prompt 长什么样（Qwen3 base 也带模板，
     但渲染细节决定模型能不能看懂）
  3) ALFWorld 官方配置里的步数上限
  4) 评测游戏若含 expert 计划，专家需要多少步（对照我们 30 步的上限）

GPU 部分（--gpu，需要 1 张卡）：
  5) **最关键的诊断**：模型在评测 prompt 上的回答，`parse_tool_call` 判成
     action / malformed / finish 各占多少？
     eval_ez._rollout_batch 会把 malformed **静默转成 "look"**，
     所以"90% 都是 look"既可能是模型真在空转，也可能是它的输出根本没被解析成功。
     这两种情况的修法完全不同，必须先分清。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


def cpu_part():
    import os

    from transformers import AutoTokenizer

    base = os.path.expandvars(os.path.expanduser(
        os.environ.get("ZTWANG_MODEL_PATH", "")))
    if not base:
        raise SystemExit("set ZTWANG_MODEL_PATH")

    print("=== 1) 基座身份 ===")
    cfg = json.load(open(Path(base) / "config.json"))
    for k in ("_name_or_path", "model_type", "architectures", "torch_dtype"):
        print(f"  {k}: {cfg.get(k)}")
    rd = Path(base) / "README.md"
    if rd.is_file():
        head = rd.read_text(encoding="utf-8", errors="replace")[:400]
        tag = "INSTRUCT" if "instruct" in head.lower() else "base(未见 instruct 字样)"
        print(f"  README 首段判定: {tag}")
        print("  README 片段:", " ".join(head.split())[:220])
    gc = Path(base) / "generation_config.json"
    if gc.is_file():
        print("  generation_config:", json.dumps(json.load(open(gc)))[:200])

    print("=== 2) chat template 渲染实况 ===")
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    ct = getattr(tok, "chat_template", None)
    print("  chat_template 长度:", len(ct) if ct else None,
          "| 含 enable_thinking:", ("enable_thinking" in (ct or "")))
    from ez_prompts import build_solver_messages
    msgs = build_solver_messages("You are in a room. There is a desk.",
                                 "look at the alarm clock under the desklamp")
    for think in (False, True):
        try:
            s = tok.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True,
                                        enable_thinking=think)
        except TypeError:
            s = tok.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True)
        print(f"  --- enable_thinking={think} 渲染结果（前 500 字，原样）---")
        print(s[:500])
    ids = tok(s, add_special_tokens=False)["input_ids"]
    print("  渲染后 token 数:", len(ids))

    print("=== 3) ALFWorld 配置里的步数上限 ===")
    import yaml
    for p in (Path(base).parent.parent / "ztwang" / "selfplay" / "configs"
              / "base_config.yaml",
              SRC.parent / "configs" / "base_config.yaml"):
        if p.is_file():
            y = yaml.safe_load(p.read_text())
            print(f"  {p.name}: max_episode_steps={y.get('env', {}).get('max_episode_steps')}")
            break
    print("  （脚本里的评测上限来自 configs/ez_qwen3_4b.yaml 的 env.max_episode_steps"
          " 与 solver.max_turns）")


def gpu_part(backend: str, ckpt: str | None, n_prompts: int, max_new: int):
    import os

    from alfworld_env import ALFWorldBatchEnv, collect_game_files, load_alfworld_config
    from ez_prompts import build_solver_messages
    from ez_proposer import parse_tool_call
    from rollout import RolloutConfig

    base = os.path.expandvars(os.path.expanduser(os.environ["ZTWANG_MODEL_PATH"]))
    cfg = load_alfworld_config(str(SRC.parent / "configs" / "base_config.yaml"))
    games = sorted(collect_game_files(cfg, "eval_out_of_distribution"))[:n_prompts]
    env = ALFWorldBatchEnv(cfg, split="eval_out_of_distribution",
                           batch_size=len(games), max_episode_steps=30,
                           with_facts=False, ignore_benchmark_done=False)
    obs, views, _adm, _used, _p = env.reset(games)
    env.close()
    prompts = [build_solver_messages(views[i].env_desc or views[i].raw,
                                     views[i].task_desc or "")
               for i in range(len(games))]

    rcfg = RolloutConfig(group_size=1, max_steps=30, max_new_tokens=max_new,
                         temperature=1.0, top_p=0.95, top_k=-1, min_p=0.0,
                         repetition_penalty=1.05, max_context_tokens=40960,
                         stop_mode="tool_call", max_batch_size=0)

    if backend == "hf":
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from rollout import HFPolicy
        tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        m = AutoModelForCausalLM.from_pretrained(
            base, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
            trust_remote_code=True).to("cuda").eval()
        if ckpt:
            m = PeftModel.from_pretrained(m, str(Path(ckpt) / "solver"),
                                          adapter_name="solver").eval()
            m.set_adapter("solver")
        policy = HFPolicy(m, tok)
    else:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ez_vllm_policy import VLLMPolicy, build_vllm_engine, export_lora_adapter
        tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        adir = None
        if ckpt:
            cpu = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16,
                                                      trust_remote_code=True)
            cpu = PeftModel.from_pretrained(cpu, str(Path(ckpt) / "solver"),
                                            adapter_name="solver")
            adir = export_lora_adapter(cpu, "solver", "/tmp/diag_lora")
            del cpu
        eng = build_vllm_engine(base, gpu_memory_utilization=0.85, max_model_len=40960)
        policy = VLLMPolicy(eng, tok, log_fn=lambda m: print(m, flush=True))
        if adir:
            policy.set_lora("solver", adir)

    texts, gen, _ = policy.generate_from_token_ids(
        [policy.initial_prompt_ids(m, rcfg) for m in prompts], rcfg)

    kinds, lens = Counter(), []
    for t, g in zip(texts, gen):
        k = parse_tool_call(t).kind
        kinds[k] += 1
        lens.append(len(g))
    print(f"\n=== 5) 解析类型分布（backend={backend} ckpt={ckpt} max_new={max_new}）===")
    print("  分布:", dict(kinds))
    lens.sort()
    print(f"  生成长度: min={lens[0]} p50={lens[len(lens)//2]} max={lens[-1]} "
          f"命中上限({max_new})的比例={sum(1 for x in lens if x >= max_new)/len(lens):.2f}")
    n_bad = kinds.get("malformed", 0)
    print(f"  ⚠️ malformed 会被 eval 静默当成 'look'：{n_bad}/{len(texts)}")
    print("\n--- 原始输出样本（前 3 条，不做任何转述）---")
    for i in range(min(3, len(texts))):
        print(f"  [{i}] kind={parse_tool_call(texts[i]).kind} len={len(gen[i])}")
        print("      " + repr(texts[i][:400]))


def main() -> int:
    if "--gpu" in sys.argv:
        i = sys.argv.index("--gpu")
        backend = sys.argv[i + 1] if len(sys.argv) > i + 1 else "vllm"
        ckpt = sys.argv[i + 2] if len(sys.argv) > i + 2 else ""
        n = int(sys.argv[i + 3]) if len(sys.argv) > i + 3 else 8
        mx = int(sys.argv[i + 4]) if len(sys.argv) > i + 4 else 96
        gpu_part(backend, ckpt or None, n, mx)
    else:
        cpu_part()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
