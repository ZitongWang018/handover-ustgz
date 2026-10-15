#!/usr/bin/env python
"""Official ALFWorld valid_unseen evaluation for an EZ solver adapter or base control."""
from __future__ import annotations

import argparse
import hashlib
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
from ez_prompts import (build_solver_messages,
                        build_solver_messages_rich, dump_malformed,
                        format_execution_feedback, recover_bare_command,
                        resolve_action, solver_style)
from ez_eval_protocol import (aggregate_complete_evaluation,
                              align_records_to_slots)
from ez_model_identity import inspect_model_identity
from rollout import HFPolicy, RolloutConfig


def _pkg_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _safe_model_identity(path):
    try:
        return inspect_model_identity(path)
    except OSError as exc:
        return {"status": "metadata_unavailable", "requested_path": str(path),
                "reason": f"local metadata I/O failed: {type(exc).__name__}",
                "weights_verified": False}


def _manifest(args, policy, *, effective_profile, style, decode, overrides,
              menu_enabled, menu_requested, menu_source, legacy_menu_value):
    import ez_prompts
    if style == "rich":
        prompt_parts = [ez_prompts._RICH_SYSTEM, ez_prompts._RICH_USER]
    else:
        prompt_parts = [ez_prompts.SOLVER_SYSTEM, ez_prompts.SOLVER_USER_TEMPLATE,
                        ez_prompts._solver_fewshot()]
        if menu_enabled:
            prompt_parts.append(ez_prompts.ADMISSIBLE_HINT)
    prompt_parts.extend(["prompt_style=" + style,
                         "action_menu_enabled=" + str(bool(menu_enabled))])
    prompt_hash = hashlib.sha256("\n\0\n".join(prompt_parts).encode("utf-8")).hexdigest()
    chat_template = getattr(getattr(policy, "tokenizer", None), "chat_template", None)
    chat_hash = (hashlib.sha256(str(chat_template).encode("utf-8")).hexdigest()
                 if chat_template is not None else None)
    env_keys = ("ALFWORLD_DATA", "ZTWANG_MODEL_PATH", "EZ_SOLVER_STYLE",
                "EZ_SOLVER_ADMISSIBLE", "EZ_SOLVER_BARE_FALLBACK",
                "EZ_SOLVER_STRICT_RESOLVE", "EZ_SOLVER_FEWSHOT",
                "EZ_SOLVER_FEWSHOT_SCHEMA", "EZ_PARSER_STRICT")
    model = getattr(policy, "model", None)
    config = getattr(model, "config", None)
    actual_attn = getattr(config, "_attn_implementation", None)
    return {
        "protocol_requested": args.protocol,
        "protocol_effective": effective_profile,
        "faithfulness": ("paper approximation; Figure 14 says Example Omitted and the paper's sampling details are incomplete"
                         if effective_profile == "paper-approx" else
                         "public rich zero-shot control; not the Explorer-Zero paper protocol"
                         if effective_profile == "public-rich" else "custom evaluation protocol"),
        "prompt_style": style,
        "prompt_sha256": prompt_hash,
        "tokenizer_chat_template_sha256": chat_hash,
        "enable_thinking": False,
        "tokenizer": getattr(getattr(policy, "tokenizer", None), "name_or_path", None),
        "base_model": os.path.expandvars(os.path.expanduser(args.base)),
        "model_identity": _safe_model_identity(args.base),
        "checkpoint": None if args.base_only else str(Path(args.ckpt).resolve()),
        "adapter": None if args.base_only else "solver",
        "base_only": args.base_only,
        "backend": args.backend,
        "parser": {"type": ("public_admissible_action_text" if style == "rich"
                              else "ez_json_tool_call"),
                   "strict": (False if style == "rich" else
                              os.environ.get("EZ_PARSER_STRICT", "0") == "1"),
                   "strict_is_diagnostic_only": style == "rich"},
        "resolver": {"type": "public_whitespace_normalization_plus_unique_resolver",
                     "strict": (False if style == "rich" else
                                os.environ.get("EZ_SOLVER_STRICT_RESOLVE", "0") == "1"),
                     "ignored_environment_flags": (["EZ_PARSER_STRICT",
                                                     "EZ_SOLVER_ADMISSIBLE",
                                                     "EZ_SOLVER_STRICT_RESOLVE"]
                                                    if style == "rich" else []),
                     "menu_enabled": bool(menu_enabled),
                     "menu_requested": menu_requested,
                     "menu_resolution_source": menu_source,
                     "legacy_environment_value": legacy_menu_value,
                     "menu_scope": ("numbered actions rebuilt in every prompt" if style == "rich"
                                    else "initial prompt and execution feedback"
                                    if menu_enabled else "none"),
                     "recovery_menu_used_separately_from_visible_menu": True,
                     "natural_command_recovery": True},
        "fewshot": {"enabled": (style != "rich" and any(
                                     os.environ.get(k, "0") == "1"
                                     for k in ("EZ_SOLVER_FEWSHOT", "EZ_SOLVER_FEWSHOT_SCHEMA"))),
                    "requested_flags": {k: os.environ.get(k, "0") for k in
                                        ("EZ_SOLVER_FEWSHOT", "EZ_SOLVER_FEWSHOT_SCHEMA")},
                    "schema_variant": os.environ.get("EZ_SOLVER_FEWSHOT_SCHEMA", "0") == "1"},
        "ignored_environment_flags": (["EZ_SOLVER_FEWSHOT", "EZ_SOLVER_FEWSHOT_SCHEMA",
                                       "EZ_SOLVER_ADMISSIBLE", "EZ_PARSER_STRICT",
                                       "EZ_SOLVER_STRICT_RESOLVE"]
                                      if style == "rich" else []),
        "decode": decode,
        "samples_per_task": args.samples,
        "split": "eval_out_of_distribution",
        "cli_overrides": overrides,
        "seed": args.seed,
        "environment": {k: os.environ.get(k) for k in env_keys},
        "software": {name: _pkg_version(name) for name in
                     ("alfworld", "textworld", "torch", "transformers", "peft", "vllm")},
        "model_attention_backend": ("vllm" if args.backend == "vllm"
                                     else actual_attn or args.attn),
        "schema_version": 2,
        "evaluation_run_id": args.run_id,
    }


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
            seed=int(args.seed),
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


def _rollout_batch_rich(env, policy, used, views, records, adm0, cfg, max_steps):
    """`EZ_SOLVER_STYLE=rich` 的评测循环：每步**重建**整条 prompt（对齐公开 rich scaffold）。

    与默认路径的区别：
      * prompt 每步用 (task, 当前观测, 最近 5 条 (action,obs), 无效动作黑名单, admissible)
        重新拼（无状态），而不是在对话后面追加 token；
      * 合法动作表**编号列出**，要求模型 `Return ONLY the exact action text`；
      * step0 的观测用**原始 reset obs**（含横幅+房间描述+`Your task is to:` 原句），
        后续 step 用动作反馈 —— 与参考实现一致。
    """
    n = len(used)
    tasks = [(views[i].task_desc or "").strip() for i in range(n)]
    cur_obs = [(views[i].raw or "").strip() for i in range(n)]
    adm = [list(adm0[i]) if adm0 else [] for i in range(n)]
    hist = [[] for _ in range(n)]
    failed = [[] for _ in range(n)]
    active = [True] * n
    won = [False] * n
    steps = [0] * n
    actions = [[] for _ in range(n)]
    traces = [[] for _ in range(n)]
    kind_counts = [{"action": 0, "malformed": 0, "control": 0, "recovered": 0,
                    "resolver_modes": {}, "invalid_action": 0} for _ in range(n)]
    dump_path = os.environ.get("EZ_DUMP_MALFORMED", "")

    for _step in range(max_steps):
        if not any(active):
            break
        idxs = [i for i, yes in enumerate(active) if yes]
        msgs = [build_solver_messages_rich(tasks[i], cur_obs[i], hist[i][-5:], adm[i],
                                           _step, max_steps, failed[i][-10:])
                for i in idxs]
        buffers = [policy.initial_prompt_ids(m, cfg) for m in msgs]
        texts, _gen, _ = policy.generate_from_token_ids(buffers, cfg)
        commands = ["look"] * n
        parsed_by_slot = [None] * n
        recovered_by_slot = [False] * n
        resolver_by_slot = ["not_used"] * n
        for j, slot in enumerate(idxs):
            parsed = parse_tool_call(texts[j])  # diagnostic only; public rich selects action text
            # The public rich profile always uses its menu parser semantics; do not
            # let an ambient EZ-specific strict flag change the executed command.
            _rec, _how = resolve_action(texts[j], adm[slot], strict=False)
            resolver_by_slot[slot] = _how
            command = (_rec if _how != "none" else
                       ("look" if "look" in adm[slot] else (adm[slot][0] if adm[slot] else "look")))
            modes = kind_counts[slot]["resolver_modes"]
            modes[_how] = modes.get(_how, 0) + 1
            if _how == "fallback":
                kind_counts[slot]["malformed"] += 1
            else:
                kind_counts[slot]["action"] += 1
                kind_counts[slot]["menu_resolved"] = kind_counts[slot].get("menu_resolved", 0) + 1
                recovered_by_slot[slot] = _how not in ("exact", "none")
                if recovered_by_slot[slot]:
                    kind_counts[slot]["recovered"] += 1
            if parsed.kind == "action":
                kind_counts[slot]["json_diagnostic_action_ignored"] = (
                    kind_counts[slot].get("json_diagnostic_action_ignored", 0) + 1)
            elif parsed.kind in ("declare", "sub_finish", "finish"):
                kind_counts[slot]["control"] += 1
            if adm[slot] and command not in adm[slot]:
                kind_counts[slot]["invalid_action"] += 1
            parsed_by_slot[slot] = parsed
            commands[slot] = command
            actions[slot].append(command)
            steps[slot] += 1
        result = env.step(commands)
        for slot in idxs:
            won[slot] = bool(result.won[slot]) if result.won else False
            if result.admissible:
                adm[slot] = result.admissible[slot]
            _o = result.obs[slot]
            hist[slot].append((commands[slot], _o))
            traces[slot].append({"step": steps[slot] - 1, "raw_output": texts[idxs.index(slot)],
                                 "parse_kind": ("menu_fallback" if resolver_by_slot[slot] == "fallback"
                                                else "menu_action"),
                                 "json_parse_kind": parsed_by_slot[slot].kind,
                                 "resolver_mode": resolver_by_slot[slot],
                                 "executed_command": commands[slot],
                                 "observation": _o})
            if "Nothing happens" in (_o or ""):
                failed[slot].append(commands[slot])
            # Public rich prompts carry the action in history and use the raw
            # environment observation as the current observation.
            cur_obs[slot] = _o
            if result.dones[slot]:
                active[slot] = False
    return [{"task_id": records[i]["task_id"], "task_type": records[i]["task_type"],
             "game_file": used[i], "sample": records[i]["sample"],
             "won": won[i], "steps": steps[i], "actions": actions[i],
             "trace": traces[i],
             "kinds": kind_counts[i],
             "max_steps_reached": steps[i] >= max_steps,
             "budget_boundary_not_won": bool(not won[i] and steps[i] >= max_steps)}
            for i in range(n)]


def _rollout_batch(env, policy, game_files, task_records, cfg, max_steps, *,
                   style=None, menu_enabled=None, capture_v2_trace=False):
    observations, views, admissible, used, _ = env.reset(game_files)
    if len(used) != len(game_files):
        raise RuntimeError("ALFWorld reset returned a different number of evaluation slots")
    # ★ textworld 批量环境会 shuffle gamefiles（textworld_batch.py 的 rng.shuffle），
    #   `used`（实际装载）与 `game_files`（请求）顺序不一致是**正常**的，不能按序比对。
    #   views 与 used 是同一次 shuffle 后的对齐顺序，所以用 `used` 现场重算
    #   task_id/task_type，才是 slot 的真实对应关系。
    records = align_records_to_slots(game_files, task_records, used)
    style = style or solver_style()
    if style == "rich":
        return _rollout_batch_rich(env, policy, used, views, records, admissible,
                                   cfg, max_steps)
    resolved_menu = (os.environ.get("EZ_SOLVER_ADMISSIBLE", "0") == "1"
                     if menu_enabled is None else bool(menu_enabled))
    adm = list(admissible) if admissible else [[] for _ in used]
    prompts = [build_solver_messages(views[i].env_desc or views[i].raw,
                                     views[i].task_desc or "", adm[i],
                                     menu_enabled=resolved_menu) for i in range(len(used))]
    buffers = [policy.initial_prompt_ids(m, cfg) for m in prompts]
    initial_prompt_ids = [list(x) for x in buffers]
    active = [True] * len(used)
    won = [False] * len(used)
    steps = [0] * len(used)
    actions = [[] for _ in used]
    traces = [[] for _ in used]
    # ★ `admissible`（reset 的第 3 个返回值）原来被丢掉了。它是"当前这步环境允许的
    #   全部命令"，是 `recover_bare_command` 唯一的安全依据（只从这张表里挑），
    #   所以必须接住并且**逐步更新**。
    # ★ 逐回合记录解析结果：`parse_tool_call` 判成 malformed 时，下面会**静默**
    #   把动作换成 "look"。不统计就会把"输出没解析成功"误读成"模型一直在 look"。
    kind_counts = [{"action": 0, "malformed": 0, "control": 0, "recovered": 0,
                    "resolver_modes": {}, "invalid_action": 0} for _ in used]
    dump_path = os.environ.get("EZ_DUMP_MALFORMED", "")
    for _step in range(max_steps):
        if not any(active):
            break
        idxs = [i for i, yes in enumerate(active) if yes]
        step_prompts = [list(buffers[i]) for i in idxs]
        request_seeds = [int(records[i].get("sample_seed", 0)) for i in idxs] if capture_v2_trace else None
        if request_seeds is not None:
            from ez_eval_v2_contract import request_seed
            request_seeds = [request_seed(seed, steps[slot])
                             for seed, slot in zip(request_seeds, idxs)]
            texts, generated, stop_lens = policy.generate_from_token_ids(
                step_prompts, cfg, request_seeds=request_seeds)
        else:
            texts, generated, stop_lens = policy.generate_from_token_ids(step_prompts, cfg)
        step_admissible = [list(x) for x in adm]
        commands = ["look"] * len(used)
        parsed_by_slot = [None] * len(used)
        recovered_by_slot = [False] * len(used)
        resolver_by_slot = ["not_used"] * len(used)
        for j, slot in enumerate(idxs):
            parsed = parse_tool_call(texts[j])
            command = parsed.action if parsed.kind == "action" else "look"
            if parsed.kind == "malformed":
                kind_counts[slot]["malformed"] += 1
                # 抢救：模型很可能吐的是 ALFWorld 原生自然命令而不是我们的 JSON
                _rec, _how = ((None, "v2_recovery_disabled") if capture_v2_trace
                              else resolve_action(texts[j], adm[slot]))
                resolver_by_slot[slot] = _how
                if _rec and _how not in ("none", "fallback"):
                    command = _rec
                    recovered_by_slot[slot] = True
                    kind_counts[slot]["recovered"] += 1
                    modes = kind_counts[slot]["resolver_modes"]
                    modes[_how] = modes.get(_how, 0) + 1
                dump_malformed(dump_path, used[slot], _step, texts[j])
            elif parsed.kind == "action":
                kind_counts[slot]["action"] += 1
            else:
                kind_counts[slot]["control"] += 1
            parsed_by_slot[slot] = parsed
            commands[slot] = command
            # ★ 无效动作率：ALFWorld 要求**实例名精确**（`desk 1` vs `desk 2`），
            #   而 solver 走的是 `invalid_action_mode=raw`（不做模糊对齐就发给环境）。
            #   如果模型吐的 `go to cabinet 5` 在场上根本不存在，环境只会回一句
            #   "You can't see any cabinet 5"，这一步就纯浪费 —— 表现成"走多少步都不赢"。
            #   这里用当步的 admissible 直接量出来，把"步数不够"和"动作全废"分开。
            if adm[slot] and command not in adm[slot]:
                kind_counts[slot]["invalid_action"] = kind_counts[slot].get(
                    "invalid_action", 0) + 1
            actions[slot].append(command)
            steps[slot] += 1
            buffers[slot].extend(generated[j])
        result = env.step(commands)
        for slot in idxs:
            won[slot] = bool(result.won[slot]) if result.won else False
            if result.admissible:
                adm[slot] = result.admissible[slot]
            j = idxs.index(slot)
            traces[slot].append({"step": steps[slot] - 1, "raw_output": texts[j],
                                 "parse_kind": parsed_by_slot[slot].kind,
                                 "parser_projection": {"kind": parsed_by_slot[slot].kind,
                                     "action": parsed_by_slot[slot].action,
                                     "task_description": parsed_by_slot[slot].task_description},
                                 "resolver_mode": resolver_by_slot[slot],
                                 "executed_command": commands[slot],
                                 "observation": result.obs[slot],
                                 "won": won[slot], "done": bool(result.dones[slot]),
                                 "initial_observation": views[slot].env_desc or views[slot].raw,
                                 "task_description": views[slot].task_desc or "",
                                 "admissible_menu_enabled": resolved_menu,
                                 "visible_menu_count": (len(step_admissible[slot])
                                                        if resolved_menu else 0),
                                 "resolver_consulted_action_list": (
                                     parsed_by_slot[slot].kind == "malformed" and not capture_v2_trace),
                                 "resolver_action_list_available": bool(step_admissible[slot]),
                                 "admissible_actions": step_admissible[slot],
                                 **({"prompt_token_ids": step_prompts[j],
                                     "output_token_ids": list(generated[j]),
                                     "request_seed": request_seeds[j],
                                     "stop_token_count": int(stop_lens[j]),
                                     "finish_reason": (getattr(policy, "last_finish_reasons", [None] * len(idxs))[j]
                                                       if getattr(policy, "last_finish_reasons", None) else
                                                       "max_new_tokens" if len(generated[j]) >= cfg.max_new_tokens else "stop_or_eos"),
                                     "truncated": len(generated[j]) >= cfg.max_new_tokens}
                                    if capture_v2_trace else {})})
            if result.dones[slot]:
                active[slot] = False
                continue
            parsed = parsed_by_slot[slot]
            feedback = format_execution_feedback(
                commands[slot], result.obs[slot], malformed=parsed.kind == "malformed",
                recovered=recovered_by_slot[slot],
                control=parsed.kind not in ("action", "malformed"),
                admissible=adm[slot], menu_enabled=resolved_menu)
            suffix = (policy.assistant_end_ids() + policy.append_feedback_ids(feedback) +
                      policy.next_generation_prefix_ids(feedback))
            buffers[slot].extend(suffix)
    return [{"task_id": records[i]["task_id"], "task_type": records[i]["task_type"],
             "game_file": used[i], "sample": records[i]["sample"],
             **({"sample_seed": records[i]["sample_seed"],
                 "initial_prompt_token_ids": initial_prompt_ids[i],
                 "chat_template": getattr(getattr(policy, "tokenizer", None), "chat_template", None)}
                if capture_v2_trace else {}),
             "won": won[i], "steps": steps[i], "actions": actions[i],
             "trace": traces[i],
             "kinds": kind_counts[i],
             "max_steps_reached": steps[i] >= max_steps,
             "budget_boundary_not_won": bool(not won[i] and steps[i] >= max_steps)}
            for i in range(len(used))]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", choices=["paper-approx", "public-rich", "custom"],
                    default="paper-approx",
                    help="paper approximation or separate public rich zero-shot control")
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
    #   "head"    = 取排序后的前 n 个；全量 8 卡脚本按 offset 用它做不重叠分片。
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
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--max-context-tokens", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    # ★★ 默认值必须是 -1（禁用），不能是 0。历史上这里是 0：HF 的
    #   `if cfg.top_k and cfg.top_k > 0` 让 0 = "不启用" 所以无害，但 vLLM 把 0 当
    #   非法值直接抛 `ValueError: top_k must be -1 (disable), or at least 1, got 0.`
    #   —— 同一个坑在训练里已经炸过一次（job 13026569 整个作业报废）。
    #   现在两条后端都由 `normalize_sampling` 兜底，但默认值本身也要给对。
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--repetition-penalty", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    ap.add_argument("--attn", choices=["flash_attention_2", "sdpa", "eager"], default="flash_attention_2")
    ap.add_argument("--style", choices=["ours", "rich"], default=None,
                    help="explicit prompt style; deviation from profile marks output custom")
    ap.add_argument("--action-menu", choices=["auto", "on", "off"], default="auto",
                    help="auto follows EZ_SOLVER_ADMISSIBLE for ours; rich always shows its native numbered menu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--run-id", default=None,
                    help="shard-run identity written to the protocol manifest")
    args = ap.parse_args()
    defaults = ({"samples": 1, "max_steps": 50, "max_context_tokens": 40960,
                 "max_new_tokens": 32, "temperature": 0.0, "top_p": 0.95,
                 "top_k": -1, "repetition_penalty": 1.0, "style": "rich",
                 "stop_mode": "eos"}
                if args.protocol == "public-rich" else
                {"samples": 4, "max_steps": 30, "max_context_tokens": 40960,
                 "max_new_tokens": 96, "temperature": 1.0, "top_p": 0.95,
                 "top_k": -1, "repetition_penalty": 1.05,
                 "style": "ours", "stop_mode": "tool_call"})
    option_names = {"samples": "samples", "max_steps": "max_steps",
                    "max_context_tokens": "max_context_tokens",
                    "max_new_tokens": "max_new_tokens", "temperature": "temperature",
                    "top_p": "top_p", "top_k": "top_k",
                    "repetition_penalty": "repetition_penalty", "style": "style"}
    overrides = {}
    explicit_style = args.style
    for name, dest in option_names.items():
        if name == "style":
            continue
        supplied = getattr(args, dest)
        if supplied is None:
            setattr(args, dest, defaults[name])
        elif supplied != defaults[name]:
            overrides[name] = {"profile_default": defaults[name], "effective": supplied}
    env_profile_overrides = {
        "fewshot": any(os.environ.get(k, "0") == "1"
                        for k in ("EZ_SOLVER_FEWSHOT", "EZ_SOLVER_FEWSHOT_SCHEMA")),
        "admissible_menu": os.environ.get("EZ_SOLVER_ADMISSIBLE", "0") == "1",
        "parser_strict": os.environ.get("EZ_PARSER_STRICT", "0") == "1",
        "resolver_strict": os.environ.get("EZ_SOLVER_STRICT_RESOLVE", "0") == "1",
    }
    if args.protocol == "public-rich":
        args.style = explicit_style or "rich"
    elif explicit_style is not None:
        args.style = explicit_style
    else:
        args.style = solver_style()
    if args.style != defaults["style"]:
        overrides["style"] = {"profile_default": defaults["style"], "effective": args.style}
    if args.protocol == "paper-approx" and solver_style() != defaults["style"]:
        overrides["environment_style"] = solver_style()
    legacy_menu_value = os.environ.get("EZ_SOLVER_ADMISSIBLE", "0")
    if args.style == "rich" and args.action_menu == "off":
        ap.error("--action-menu off is incompatible with rich, which requires its native numbered menu")
    menu_requested = args.action_menu
    if args.style == "rich":
        menu_enabled = True
        menu_source = "rich_required"
    elif args.action_menu == "on":
        menu_enabled, menu_source = True, "cli"
    elif args.action_menu == "off":
        menu_enabled, menu_source = False, "cli"
    else:
        menu_enabled = legacy_menu_value == "1"
        menu_source = "legacy_environment" if menu_enabled else "legacy_environment_default_off"
    effective_flags = dict(env_profile_overrides)
    effective_flags["admissible_menu"] = menu_enabled
    if args.style == "rich":
        effective_flags["fewshot"] = False
        effective_flags["admissible_menu"] = True
        effective_flags["parser_strict"] = False
        effective_flags["resolver_strict"] = False
    if (effective_flags["parser_strict"] or effective_flags["resolver_strict"] or
            effective_flags["fewshot"]):
        overrides["environment_protocol_flags"] = env_profile_overrides
    if menu_enabled and args.style != "rich":
        overrides["action_menu"] = {"requested": menu_requested,
                                     "source": menu_source,
                                     "effective": True}
    defaults["stop_mode"] = "eos" if args.style == "rich" else "tool_call"
    effective_profile = args.protocol
    if args.protocol == "public-rich" and overrides:
        effective_profile = "custom"
    elif args.protocol == "paper-approx" and overrides:
        effective_profile = "custom"
    if not args.base:
        ap.error("--base 未给且 $ZTWANG_MODEL_PATH 为空")
    if not args.base_only and not args.ckpt:
        ap.error("--ckpt is required unless --base-only is set")
    if args.samples <= 0 or args.batch <= 0 or args.max_steps <= 0:
        ap.error("samples, batch, and max-steps must be positive")
    # 只拒绝**真正非法**的 top-k：-1 = 禁用（vLLM 口径），>=1 = 启用。
    # 0 按"禁用"处理（HF 的历史语义），不再像旧版那样把它当合法默认值、
    # 反而把 -1 当非法（那正好写反了）。
    if args.max_context_tokens <= args.max_new_tokens or args.top_k < -1 or args.temperature < 0:
        ap.error("max-context-tokens must exceed max-new-tokens, "
                 "top-k must be -1 (disable) or >= 1, and temperature must be >= 0")

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
        stop_mode=defaults["stop_mode"], max_batch_size=2)
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
                               max_episode_steps=args.max_steps,
                               with_expert=False, with_facts=False,
                               ignore_benchmark_done=False)
        try:
            samples.extend(_rollout_batch(env, policy, game_batch, records, rollout_cfg,
                                          args.max_steps, style=args.style,
                                          menu_enabled=menu_enabled))
        finally:
            env.close()
        if start == 0 or (start // args.batch + 1) % 25 == 0:
            print(f"[eval-ez] {min(start + len(rows), len(eval_rows))}/{len(eval_rows)} samples",
                  flush=True)

    expected_tasks = {task_instance_id(g): task_type_of(g) for g in games}
    if len(expected_tasks) != len(games):
        raise RuntimeError("selected valid_unseen games contain duplicate task ids")
    summary = aggregate_complete_evaluation(samples, expected_tasks, args.samples)
    decode = {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
              "repetition_penalty": args.repetition_penalty,
              "max_new_tokens": args.max_new_tokens, "max_steps": args.max_steps,
              "max_context_tokens": args.max_context_tokens, "stop_mode": defaults["stop_mode"]}
    if args.backend == "vllm":
        from ez_vllm_policy import normalize_sampling
        decode_effective = normalize_sampling(rollout_cfg)
    elif args.temperature == 0:
        decode_effective = {"do_sample": False, "temperature": None, "top_p": None,
                            "top_k": None, "repetition_penalty": args.repetition_penalty,
                            "stop_mode": defaults["stop_mode"]}
    else:
        decode_effective = {"do_sample": True, **decode}
    manifest = _manifest(args, policy, effective_profile=effective_profile,
                         style=args.style, decode=decode, overrides=overrides,
                         menu_enabled=menu_enabled, menu_requested=menu_requested,
                         menu_source=menu_source, legacy_menu_value=legacy_menu_value)
    report = {
        "schema_version": 2,
        "protocol": "alfworld_valid_unseen_official_benchmark_done",
        "protocol_manifest": manifest,
        "split": "eval_out_of_distribution",
        "select": args.select, "offset": args.offset,
        "expected_task_ids": list(expected_tasks),
        "summary": summary,
        "decode_effective": decode_effective,
        "elapsed_seconds": time.time() - started,
        "sample_rows": samples,
    }
    out = Path(args.out) if args.out else Path("runs/eval_ez") / (
        "base-only.json" if args.base_only else "solver-adapter.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval-ez] episode-weighted SR={summary['episode_weighted_sr']:.4f}; "
          f"per-sample whole-suite SR={summary['sample_suite_success_rates']} "
          f"population SD={summary['sample_suite_sr_population_std']:.4f}; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
