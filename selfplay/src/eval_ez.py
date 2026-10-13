#!/usr/bin/env python
"""Official ALFWorld valid_unseen evaluation for an EZ solver adapter or base control."""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from alfworld_env import (ALFWorldBatchEnv, collect_game_files, load_alfworld_config,
                          task_instance_id, task_type_of)
from ez_proposer import parse_tool_call
from ez_prompts import build_solver_messages
from rollout import HFPolicy, RolloutConfig


def _pkg_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _load_policy(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = os.path.expandvars(os.path.expanduser(args.base))
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=dtype, trust_remote_code=True,
        attn_implementation=args.attn).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    if not args.base_only:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(Path(args.ckpt) / "solver"),
                                          adapter_name="solver", is_trainable=False).eval()
        model.set_adapter("solver")
    return HFPolicy(model, tok, enable_thinking=False)


def _rollout_batch(env, policy, game_files, task_records, cfg, max_steps):
    observations, views, _admissible, used, _ = env.reset(game_files)
    if len(used) != len(game_files):
        raise RuntimeError("ALFWorld reset returned a different number of evaluation slots")
    for i, (actual, expected) in enumerate(zip(used, game_files)):
        if os.path.realpath(actual) != os.path.realpath(expected):
            raise RuntimeError(f"eval slot {i} mapped to unexpected game {actual!r}")
    prompts = [build_solver_messages(views[i].env_desc or views[i].raw,
                                     views[i].task_desc or "") for i in range(len(used))]
    buffers = [policy.initial_prompt_ids(m, cfg) for m in prompts]
    active = [True] * len(used)
    won = [False] * len(used)
    steps = [0] * len(used)
    actions = [[] for _ in used]
    for _ in range(max_steps):
        if not any(active):
            break
        idxs = [i for i, yes in enumerate(active) if yes]
        texts, generated, _ = policy.generate_from_token_ids([buffers[i] for i in idxs], cfg)
        commands = ["look"] * len(used)
        parsed_by_slot = [None] * len(used)
        for j, slot in enumerate(idxs):
            parsed = parse_tool_call(texts[j])
            command = parsed.action if parsed.kind == "action" else "look"
            parsed_by_slot[slot] = parsed
            commands[slot] = command
            actions[slot].append(command)
            steps[slot] += 1
            buffers[slot].extend(generated[j])
        result = env.step(commands)
        for slot in idxs:
            won[slot] = bool(result.won[slot]) if result.won else False
            if result.dones[slot]:
                active[slot] = False
                continue
            parsed = parsed_by_slot[slot]
            if parsed.kind == "malformed":
                feedback = ("Your last response was not a valid tool call. Emit exactly one "
                            "<tool_call>{...}</tool_call> block.")
            elif parsed.kind in ("declare", "sub_finish", "finish"):
                feedback = "Only environment actions are valid for you. Keep working on the task."
            else:
                feedback = f"You did: {commands[slot]}\nThe environment responds:\n{result.obs[slot]}"
            suffix = (policy.assistant_end_ids() + policy.append_feedback_ids(feedback) +
                      policy.next_generation_prefix_ids(feedback))
            buffers[slot].extend(suffix)
    return [{"task_id": task_records[i]["task_id"], "task_type": task_records[i]["task_type"],
             "game_file": game_files[i], "sample": task_records[i]["sample"],
             "won": won[i], "steps": steps[i], "actions": actions[i],
             "max_steps_reached": bool(active[i])} for i in range(len(used))]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="explicit HF base model path")
    ap.add_argument("--ckpt", default=None, help="EZ checkpoint directory containing solver/")
    ap.add_argument("--base-only", action="store_true", help="evaluate explicit base without adapter")
    ap.add_argument("--config", default=str(SRC.parent / "configs" / "base_config.yaml"))
    ap.add_argument("--n", type=int, default=0, help="valid_unseen games; 0 evaluates the full split")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--max-context-tokens", type=int, default=40960)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    ap.add_argument("--attn", choices=["flash_attention_2", "sdpa", "eager"], default="flash_attention_2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.base_only and not args.ckpt:
        ap.error("--ckpt is required unless --base-only is set")
    if args.samples <= 0 or args.batch <= 0 or args.max_steps <= 0:
        ap.error("samples, batch, and max-steps must be positive")
    if args.max_context_tokens <= args.max_new_tokens or args.top_k < 0:
        ap.error("max-context-tokens must exceed max-new-tokens and top-k must be nonnegative")

    from alfworld_compat import install as install_alfworld_compat
    install_alfworld_compat(verbose=True)
    random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except ImportError:
        pass
    env_cfg = load_alfworld_config(args.config)
    games = sorted(collect_game_files(env_cfg, "eval_out_of_distribution"))
    random.Random(args.seed).shuffle(games)
    if args.n > 0:
        games = games[:args.n]
    if not games:
        raise RuntimeError("valid_unseen split is empty after applying --n; refusing to report an empty evaluation")
    policy = _load_policy(args)
    rollout_cfg = RolloutConfig(group_size=1, max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
        top_k=args.top_k, repetition_penalty=args.repetition_penalty,
        max_context_tokens=args.max_context_tokens,
        stop_mode="tool_call", max_batch_size=2)
    samples = []
    started = time.time()
    eval_rows = [(g, sample) for g in games for sample in range(args.samples)]
    for start in range(0, len(eval_rows), args.batch):
        rows = eval_rows[start:start + args.batch]
        game_batch = [g for g, _sample in rows]
        records = [{"task_id": task_instance_id(g), "task_type": task_type_of(g),
                    "sample": sample} for g, sample in rows]
        env = ALFWorldBatchEnv(env_cfg, split="eval_out_of_distribution",
                               batch_size=len(rows), seed=args.seed + start,
                               with_expert=False, with_facts=False,
                               ignore_benchmark_done=False)
        try:
            samples.extend(_rollout_batch(env, policy, game_batch, records, rollout_cfg,
                                          args.max_steps))
        finally:
            env.close()
        if start == 0 or (start // args.batch + 1) % 25 == 0:
            print(f"[eval-ez] {min(start + len(rows), len(eval_rows))}/{len(eval_rows)} samples",
                  flush=True)

    by_type = defaultdict(list)
    by_task = defaultdict(list)
    for row in samples:
        by_type[row["task_type"]].append(float(row["won"]))
        by_task[row["task_id"]].append(float(row["won"]))
    report = {
        "protocol": "official_alfworld_valid_unseen_benchmark_done",
        "base": os.path.expandvars(os.path.expanduser(args.base)),
        "checkpoint": None if args.base_only else str(Path(args.ckpt).resolve()),
        "base_only": args.base_only,
        "split": "eval_out_of_distribution", "n_tasks": len(games),
        "samples_per_task": args.samples, "sample_count": len(samples),
        "sample_level_success_mean": statistics.mean([r["won"] for r in samples]) if samples else 0.0,
        "sample_level_success_std": statistics.pstdev([r["won"] for r in samples]) if samples else 0.0,
        "task_type": {k: {"mean": statistics.mean(v), "std": statistics.pstdev(v), "n": len(v)}
                      for k, v in sorted(by_type.items())},
        "per_task_sample_success": {k: {"mean": statistics.mean(v),
                                        "std": statistics.pstdev(v), "n": len(v)}
                                    for k, v in by_task.items()},
        "seed": args.seed,
        "decode": {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
                   "repetition_penalty": args.repetition_penalty,
                   "max_new_tokens": args.max_new_tokens, "max_steps": args.max_steps,
                   "max_context_tokens": args.max_context_tokens},
        "metadata": {"alfworld": _pkg_version("alfworld"), "textworld": _pkg_version("textworld"),
                     "torch": _pkg_version("torch"), "transformers": _pkg_version("transformers"),
                     "peft": _pkg_version("peft"), "elapsed_seconds": time.time() - started},
        "samples": samples,
    }
    out = Path(args.out) if args.out else Path("runs/eval_ez") / (
        "base-only.json" if args.base_only else "solver-adapter.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval-ez] sample SR={report['sample_level_success_mean']:.4f} "
          f"sample SD={report['sample_level_success_std']:.4f}; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
