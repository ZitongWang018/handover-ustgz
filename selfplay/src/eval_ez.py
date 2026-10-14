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
from ez_prompts import (build_solver_messages, dump_malformed,
                        recover_bare_command)
from rollout import HFPolicy, RolloutConfig


def _pkg_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _load_policy(args):
    import torch
    from transformers import AutoTokenizer
    base = os.path.expandvars(os.path.expanduser(args.base))
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    if args.backend == "vllm":
        # vLLM 只认 PEFT 目录，所以先用 CPU 上的 PEFT 模型把 solver adapter 导出来。
        # ★ 走这里而不是 HF，是因为本脚本要支持"全量 valid_unseen（134 任务 × 4 采样
        #   = 536 条轨迹 × 30 回合）"这种规模 —— HF 逐 token 循环要跑好几小时，
        #   vLLM 连续批处理快一个数量级。两条路径的采样参数语义由
        #   `ez_vllm_policy.normalize_sampling` 统一，不会出现"评测脚本和训练脚本
        #   对同一个 top_k 理解不同"的情况。
        import gc
        import tempfile

        from ez_vllm_policy import VLLMPolicy, build_vllm_engine, export_lora_adapter

        adapter_dir = None
        if not args.base_only:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM

            cpu = AutoModelForCausalLM.from_pretrained(
                base, dtype=torch.bfloat16, trust_remote_code=True)
            cpu = PeftModel.from_pretrained(cpu, str(Path(args.ckpt) / "solver"),
                                            adapter_name="solver")
            adapter_dir = export_lora_adapter(cpu, "solver",
                                              tempfile.mkdtemp(prefix="eval_lora_"))
            print(f"[eval-ez] exported solver adapter -> {adapter_dir}", flush=True)
            del cpu
            gc.collect()

        eng = build_vllm_engine(
            base,
            gpu_memory_utilization=float(os.environ.get("ZTWANG_VLLM_GMEM", "0.85")),
            max_model_len=int(args.max_context_tokens),
            max_num_seqs=int(os.environ.get("ZTWANG_VLLM_MAX_SEQS", "64")),
        )
        policy = VLLMPolicy(eng, tok, log_fn=lambda m: print(m, flush=True))
        if adapter_dir:
            policy.set_lora("solver", adapter_dir)
        return policy

    from transformers import AutoModelForCausalLM
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


def stratified_games(games, n: int):
    """按 `task_type` **轮转**取样，保证取到的 n 个任务覆盖所有类型。

    为什么必须有这个（2026-10-14 实测发现，比 top_k 那个崩溃更隐蔽）：
        valid_unseen 的 134 个任务在 `sorted()` 之后是**按类型聚簇**的
        （目录名以类型开头），而 `look_at_obj_in_light` 因为 `l < p ` 排在最前。
        实测：**前 16 个任务 100% 是 look_at_obj_in_light**。
        训练中 `mid_eval_solver` 用的正是 `eval_game_files[:16]` ——
        于是"中间评测"测的是 6 类任务里的 1 类（18/134），曲线严重失真。
        n>=len(games) 时直接返回全量（等价于全量评测）。
    """
    from collections import defaultdict
    if not games:
        return []
    if n <= 0:
        # ★★ `n<=0` 必须表示"**全量，但按类型轮转重排**"，不能原样返回！
        #    踩过的坑：旧写法 `if n <= 0 or n >= len(games): return list(games)`
        #    让 `stratified_games(games, 0)` 变成恒等函数，调用方再 `[:n]` 切片时
        #    拿到的还是**未重排**的头 n 个 —— 实测头 16 个全是 look_at_obj_in_light，
        #    于是"分层评测"实际没分层（tag 写着 stratify，测的却是单一类）。
        n = len(games)
    if n >= len(games):
        n = len(games)
    buckets = defaultdict(list)
    for g in games:
        buckets[task_type_of(g)].append(g)
    order = sorted(buckets)
    out = []
    depth = 0
    deepest = max(len(v) for v in buckets.values())
    while len(out) < n and depth <= deepest:
        for k in order:
            if len(out) >= n:
                break
            if depth < len(buckets[k]):
                out.append(buckets[k][depth])
        depth += 1
    return out


def _rollout_batch(env, policy, game_files, task_records, cfg, max_steps):
    observations, views, admissible, used, _ = env.reset(game_files)
    if len(used) != len(game_files):
        raise RuntimeError("ALFWorld reset returned a different number of evaluation slots")
    # ★ textworld 批量环境会 shuffle gamefiles（textworld_batch.py 的 rng.shuffle），
    #   `used`（实际装载）与 `game_files`（请求）顺序不一致是**正常**的，不能按序比对。
    #   views 与 used 是同一次 shuffle 后的对齐顺序，所以用 `used` 现场重算
    #   task_id/task_type，才是 slot 的真实对应关系。
    records = [{"task_id": task_instance_id(g), "task_type": task_type_of(g),
                "sample": (task_records[0]["sample"] if task_records else 0)}
               for g in used]
    prompts = [build_solver_messages(views[i].env_desc or views[i].raw,
                                     views[i].task_desc or "") for i in range(len(used))]
    buffers = [policy.initial_prompt_ids(m, cfg) for m in prompts]
    active = [True] * len(used)
    won = [False] * len(used)
    steps = [0] * len(used)
    actions = [[] for _ in used]
    # ★ `admissible`（reset 的第 3 个返回值）原来被丢掉了。它是"当前这步环境允许的
    #   全部命令"，是 `recover_bare_command` 唯一的安全依据（只从这张表里挑），
    #   所以必须接住并且**逐步更新**。
    adm = list(admissible) if admissible else [[] for _ in used]
    # ★ 逐回合记录解析结果：`parse_tool_call` 判成 malformed 时，下面会**静默**
    #   把动作换成 "look"。不统计就会把"输出没解析成功"误读成"模型一直在 look"。
    kind_counts = [{"action": 0, "malformed": 0, "control": 0, "recovered": 0}
                   for _ in used]
    dump_path = os.environ.get("EZ_DUMP_MALFORMED", "")
    for _step in range(max_steps):
        if not any(active):
            break
        idxs = [i for i, yes in enumerate(active) if yes]
        texts, generated, _ = policy.generate_from_token_ids([buffers[i] for i in idxs], cfg)
        commands = ["look"] * len(used)
        parsed_by_slot = [None] * len(used)
        for j, slot in enumerate(idxs):
            parsed = parse_tool_call(texts[j])
            command = parsed.action if parsed.kind == "action" else "look"
            if parsed.kind == "malformed":
                kind_counts[slot]["malformed"] += 1
                # 抢救：模型很可能吐的是 ALFWorld 原生自然命令而不是我们的 JSON
                _rec = recover_bare_command(texts[j], adm[slot])
                if _rec:
                    command = _rec
                    kind_counts[slot]["recovered"] += 1
                dump_malformed(dump_path, used[slot], _step, texts[j])
            elif parsed.kind == "action":
                kind_counts[slot]["action"] += 1
            else:
                kind_counts[slot]["control"] += 1
            parsed_by_slot[slot] = parsed
            commands[slot] = command
            actions[slot].append(command)
            steps[slot] += 1
            buffers[slot].extend(generated[j])
        result = env.step(commands)
        for slot in idxs:
            won[slot] = bool(result.won[slot]) if result.won else False
            if result.admissible:
                adm[slot] = result.admissible[slot]
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
    return [{"task_id": records[i]["task_id"], "task_type": records[i]["task_type"],
             "game_file": used[i], "sample": records[i]["sample"],
             "won": won[i], "steps": steps[i], "actions": actions[i],
             "kinds": kind_counts[i],
             "max_steps_reached": bool(active[i])} for i in range(len(used))]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=os.environ.get("ZTWANG_MODEL_PATH", ""),
                    help="explicit HF base model path（默认取 $ZTWANG_MODEL_PATH）")
    ap.add_argument("--ckpt", default=None, help="EZ checkpoint directory containing solver/")
    ap.add_argument("--base-only", action="store_true", help="evaluate explicit base without adapter")
    # ★★ 采样后端：hf 与训练默认一致；vllm 快一个数量级，做全量 valid_unseen 评测时建议开。
    #   两条路径的采样参数由 `ez_vllm_policy.normalize_sampling` 归一，语义一致。
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--config", default=str(SRC.parent / "configs" / "base_config.yaml"))
    ap.add_argument("--n", type=int, default=0, help="valid_unseen games; 0 evaluates the full split")
    # ★ 任务子集怎么选，决定这个数字能不能和别的评测点比：
    #   "shuffle" = 固定 seed 随机抽 n 个（本脚本的历史行为，适合独立评测）
    #   "head"    = 取排序后的前 n 个 —— **与训练中 mid_eval_solver 完全同一批任务**
    #               （ez_train 里就是 `eval_game_files[:_n]`，无 shuffle）。
    #               想补训练期间漏掉的某个 outer 的评测点，必须用 head 才可比。
    # ★ 任务子集怎么选，决定这个数字能不能和别的评测点比：
    #   "stratify" = 按 task_type 轮转，**覆盖全部 6 类**（默认，强烈推荐）
    #   "head"     = 排序后取前 n 个 —— ⚠️ 实测前 16 个全是 look_at_obj_in_light，
    #                只有 1 类任务！保留它只为"分片跑全量"（配合 --offset 可以无缝拼回全集）
    #   "shuffle"  = 固定 seed 随机抽 n 个（本脚本最老的行为）
    ap.add_argument("--select", choices=["stratify", "shuffle", "head"], default="stratify")
    # ★ 分片用：`--offset k --n m` 取 [k, k+m)。配合每卡一个 shard 就能把全量
    #   valid_unseen（134 任务 × 4 采样 = 536 条轨迹）压到一次作业里跑完。
    ap.add_argument("--offset", type=int, default=0,
                    help="从排序后的第 offset 个任务开始取（与 --select head 配合做分片）")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--max-context-tokens", type=int, default=40960)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    # ★★ 默认值必须是 -1（禁用），不能是 0。历史上这里是 0：HF 的
    #   `if cfg.top_k and cfg.top_k > 0` 让 0 = "不启用" 所以无害，但 vLLM 把 0 当
    #   非法值直接抛 `ValueError: top_k must be -1 (disable), or at least 1, got 0.`
    #   —— 同一个坑在训练里已经炸过一次（job 13026569 整个作业报废）。
    #   现在两条后端都由 `normalize_sampling` 兜底，但默认值本身也要给对。
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    ap.add_argument("--attn", choices=["flash_attention_2", "sdpa", "eager"], default="flash_attention_2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.base:
        ap.error("--base 未给且 $ZTWANG_MODEL_PATH 为空")
    if not args.base_only and not args.ckpt:
        ap.error("--ckpt is required unless --base-only is set")
    if args.samples <= 0 or args.batch <= 0 or args.max_steps <= 0:
        ap.error("samples, batch, and max-steps must be positive")
    # 只拒绝**真正非法**的 top-k：-1 = 禁用（vLLM 口径），>=1 = 启用。
    # 0 按"禁用"处理（HF 的历史语义），不再像旧版那样把它当合法默认值、
    # 反而把 -1 当非法（那正好写反了）。
    if args.max_context_tokens <= args.max_new_tokens or args.top_k < -1:
        ap.error("max-context-tokens must exceed max-new-tokens, "
                 "and top-k must be -1 (disable) or >= 1")

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
    if args.select == "shuffle":
        random.Random(args.seed).shuffle(games)
    elif args.select == "stratify":
        # 先重排成"按类型轮转"的全序，再切 offset/n —— 这样分片也照样覆盖所有类型
        games = stratified_games(games, 0)
    if args.offset:
        games = games[args.offset:]
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
    if args.backend == "vllm":
        from ez_vllm_policy import normalize_sampling
        print(f"[eval-ez] backend=vllm 实际采样参数={normalize_sampling(rollout_cfg)}", flush=True)
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
        "select": args.select, "offset": args.offset,
        "samples_per_task": args.samples, "sample_count": len(samples),
        "sample_level_success_mean": statistics.mean([r["won"] for r in samples]) if samples else 0.0,
        "sample_level_success_std": statistics.pstdev([r["won"] for r in samples]) if samples else 0.0,
        "task_type": {k: {"mean": statistics.mean(v), "std": statistics.pstdev(v), "n": len(v)}
                      for k, v in sorted(by_type.items())},
        "per_task_sample_success": {k: {"mean": statistics.mean(v),
                                        "std": statistics.pstdev(v), "n": len(v)}
                                    for k, v in by_task.items()},
        "seed": args.seed, "backend": args.backend,
        "decode": {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
                   "repetition_penalty": args.repetition_penalty,
                   "max_new_tokens": args.max_new_tokens, "max_steps": args.max_steps,
                   "max_context_tokens": args.max_context_tokens},
        # ★ 把**后端实际收到**的参数也记下来（归一化之后），否则事后看 JSON 无法判断
        #   `top_k=0` 到底被当成了"禁用"还是"报错"。
        "decode_effective": (normalize_sampling(rollout_cfg)
                             if args.backend == "vllm" else None),
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
