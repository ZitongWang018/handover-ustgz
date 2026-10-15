#!/usr/bin/env python3
"""Strict local-only Explorer-Zero evaluation v2 entrypoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from ez_eval_v2_contract import (SCHEMA, SAMPLES, canonical_hash, discover_suite,
                                 request_seed, sample_seed, sha256_file, strict_merge, validate_suite)


def inventory(root: Path) -> list[dict]:
    if not root.is_dir():
        raise ValueError(f"model/adapter path is not a local directory: {root}")
    return [{"path": p.relative_to(root).as_posix(), "size": p.stat().st_size,
             "sha256": sha256_file(p)} for p in sorted(root.rglob("*")) if p.is_file()]


def software_versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for name in ("alfworld", "textworld", "torch", "transformers", "peft", "vllm", "numpy"):
        try: out[name] = version(name)
        except PackageNotFoundError: out[name] = "not-installed"
    return out


def _set_strict_environment():
    # Old EZ flags are removed before importing prompt/parser modules.
    keys = ("EZ_SOLVER_STYLE", "EZ_SOLVER_FEWSHOT", "EZ_SOLVER_FEWSHOT_SCHEMA",
            "EZ_SOLVER_ADMISSIBLE", "EZ_SOLVER_STRICT_RESOLVE", "EZ_PARSER_STRICT",
            "EZ_SOLVER_BARE_FALLBACK", "EZ_ACTION_MENU", "EZ_MAX_STEPS", "EZ_MAX_NEW",
            "EZ_MAX_NEW_TOKENS", "EZ_MAX_CONTEXT_TOKENS", "EZ_TEMPERATURE", "EZ_TOP_P",
            "EZ_TOP_K", "EZ_REPETITION_PENALTY", "EZ_SAMPLES", "EZ_SEED", "EZ_PROTOCOL")
    for key in keys: os.environ.pop(key, None)
    os.environ["EZ_SOLVER_STYLE"] = "ours"
    os.environ["EZ_SOLVER_FEWSHOT"] = "0"
    os.environ["EZ_SOLVER_FEWSHOT_SCHEMA"] = "0"
    os.environ["EZ_SOLVER_ADMISSIBLE"] = "0"
    os.environ["EZ_SOLVER_STRICT_RESOLVE"] = "0"
    os.environ["EZ_PARSER_STRICT"] = "0"


def prepare(args) -> int:
    suite = discover_suite(args.config, args.split, shards=args.shards)
    validate_suite(suite, config_path=args.config)
    out = Path(args.output).expanduser()
    if out.exists(): raise FileExistsError(f"refusing to overwrite suite manifest: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(suite, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"prepared {suite['n_tasks']} {suite['split']} tasks in {len(suite['shards'])} non-empty shards; pool={suite['pool_sha256']}")
    return 0


def _checkpoint_info(args, base_digest):
    if args.base_only:
        return {"kind": "base-only", "base_inventory_sha256": base_digest}
    if not args.checkpoint:
        raise ValueError("adapter evaluation requires --checkpoint v3 training checkpoint")
    cp = Path(args.checkpoint).expanduser().resolve(strict=True)
    # Shared with the training branch: this validator proves schema v3, completion,
    # inventory integrity and exact base digest. A schema-v2/legacy adapter is rejected.
    from ez_train_v2_contract import validate_checkpoint_v3
    state = validate_checkpoint_v3(str(cp), expected_contract={
        "base_inventory_sha256": base_digest}, purpose="evaluation")
    leaf = Path(state["solver_path"]).resolve(strict=True)
    if leaf.name != "solver" and (leaf / "solver").is_dir():
        leaf = leaf / "solver"
    if not (leaf / "adapter_config.json").is_file():
        raise ValueError("validated v3 checkpoint does not expose a PEFT solver adapter leaf")
    return {"kind": "adapter", "checkpoint": str(cp), "checkpoint_manifest": state,
            "adapter_path": str(leaf), "adapter_inventory": inventory(leaf),
            "base_inventory_sha256": base_digest}


def _load_v2_policy(args, tokenizer_path, tok=None):
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True,
                                            local_files_only=True)
        if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    if args.backend == "vllm":
        from ez_vllm_policy import VLLMPolicy, build_vllm_engine
        eng = build_vllm_engine(args.base, gpu_memory_utilization=args.gpu_memory_utilization,
                                max_model_len=args.max_context_tokens,
                                max_lora_rank=32, max_num_seqs=args.batch, seed=args.seed)
        policy = VLLMPolicy(eng, tok)
        if not args.base_only: policy.set_lora("solver", args.adapter_path)
        return policy
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation=args.attn, local_files_only=True)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    if not args.base_only:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path, adapter_name="solver",
                                          is_trainable=False, local_files_only=True).eval()
        model.set_adapter("solver")
    from rollout import HFPolicy
    return HFPolicy(model, tok, enable_thinking=False)


def protocol_signature(suite, *, tokenizer_hash, chat_hash, backend, args):
    return {"pool_sha256": suite["pool_sha256"], "split": suite["split"],
        "samples_per_task": SAMPLES, "protocol": "paper-approx", "prompt_sha256": args.prompt_sha256,
        "tokenizer_sha256": tokenizer_hash, "scaffold_sha256": args.prompt_sha256,
        "chat_template_sha256": chat_hash, "parser": "ez_json_tool_call/v2",
        "menu": False, "stop": "tool_call", "decode": {"temperature": args.temperature,
        "top_p": args.top_p, "top_k": args.top_k, "repetition_penalty": 1.0},
        "max_context_tokens": args.max_context_tokens, "max_new_tokens": args.max_new_tokens,
        "max_steps": args.max_steps, "seed_policy": "sha256(run_seed,task_id,sample_index)->sample_seed; sha256(sample_seed,turn)->request_seed",
        "run_seed": args.seed, "backend": backend, "base_inventory_sha256": args.base_inventory_sha256,
        "attention_backend": "vllm" if backend == "vllm" else args.attn,
        "dtype": "bfloat16", "resolver": "no_hidden_admissible_recovery; malformed_projects_to_look",
        "fewshot": False, "parser_strict": False,
        "software": software_versions(), "runtime_source_sha256": runtime_sources()}


def runtime_sources():
    names = ("eval_ez.py", "ez_eval_v2.py", "ez_eval_v2_contract.py", "ez_prompts.py",
             "ez_proposer.py", "ez_eval_protocol.py", "prompts.py", "reward.py", "rollout.py",
             "ez_vllm_policy.py", "alfworld_env.py", "alfworld_compat.py", "ez_train_v2_contract.py")
    return {name: sha256_file(SRC / name) for name in names if (SRC / name).is_file()}


def prepare_run(args):
    if args.batch <= 0 or args.max_steps <= 0 or args.max_new_tokens <= 0 or args.max_context_tokens <= args.max_new_tokens:
        raise ValueError("batch/step/token limits must be positive and context must exceed max-new-tokens")
    if (not math.isfinite(args.temperature) or args.temperature < 0 or
            not math.isfinite(args.top_p) or not 0 < args.top_p <= 1 or
            not math.isfinite(args.gpu_memory_utilization) or not 0 < args.gpu_memory_utilization <= 1 or
            args.top_k < -1):
        raise ValueError("invalid sampling settings")
    if args.base_only == bool(args.checkpoint):
        raise ValueError("choose exactly one of --base-only or --checkpoint")
    suite_path = Path(args.suite_manifest).resolve(strict=True)
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    validate_suite(suite, config_path=args.config)
    fresh = discover_suite(args.config, suite["split"], shards=int(suite["requested_shards"]))
    if fresh["pool_sha256"] != suite["pool_sha256"] or fresh["tasks"] != suite["tasks"]:
        raise ValueError("actual installed ALFWorld pool differs from prepared suite")
    base = Path(args.base).expanduser().resolve(strict=True)
    from ez_train_v2_contract import base_file_inventory, base_inventory_sha256
    base_map = base_file_inventory(base)
    base_digest = base_inventory_sha256(base)
    arm_args = argparse.Namespace(base_only=args.base_only, checkpoint=args.checkpoint)
    arm = _checkpoint_info(arm_args, base_digest)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(base), trust_remote_code=True, local_files_only=True)
    chat = getattr(tok, "chat_template", None)
    chat_hash = hashlib.sha256(str(chat).encode()).hexdigest() if chat is not None else None
    token_files = {k: v for k, v in base_map.items() if k.startswith("tokenizer") or
                   k in ("vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json")}
    from ez_prompts import SOLVER_SYSTEM, SOLVER_USER_TEMPLATE
    prompt_hash = hashlib.sha256((SOLVER_SYSTEM + "\0" + SOLVER_USER_TEMPLATE).encode()).hexdigest()
    args.prompt_sha256 = prompt_hash
    args.base_inventory_sha256 = base_digest
    sig = protocol_signature(suite, tokenizer_hash=canonical_hash(token_files), chat_hash=chat_hash,
                             backend=args.backend, args=args)
    evidence = {"base_inventory": [{"path": k, **v} for k, v in sorted(base_map.items())],
                "software": software_versions(), "runtime_source_sha256": runtime_sources()}
    expected_shards = [{"rank": sh["rank"], "task_ids": sh["task_ids"],
                        "path": str((Path(args.results_dir).resolve() / f"rank{sh['rank']}.json"))}
                       for sh in suite["shards"]]
    contract = {"schema": SCHEMA, "run_id": args.run_id or uuid.uuid4().hex,
        "suite": suite, "suite_manifest_path": str(suite_path), "config_path": str(Path(args.config).resolve()),
        "base_path": str(base), "arm_identity": arm, "protocol_signature": sig,
        "expected_shards": expected_shards, **evidence,
        "runtime": {"backend": args.backend, "seed": args.seed, "batch": args.batch,
                    "max_steps": args.max_steps, "max_context_tokens": args.max_context_tokens,
                    "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
                    "top_p": args.top_p, "top_k": args.top_k, "gpu_memory_utilization": args.gpu_memory_utilization,
                    "attn": args.attn, "base_only": args.base_only}}
    contract["contract_sha256"] = canonical_hash(contract)
    out = Path(args.output)
    if out.exists(): raise FileExistsError(f"refusing to overwrite prepared run contract: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"prepared immutable run contract {contract['contract_sha256']} for {len(suite['shards'])} shards")
    return 0


def run_eval(args) -> int:
    _set_strict_environment()
    run_path = Path(args.run_contract).expanduser().resolve(strict=True)
    contract = json.loads(run_path.read_text(encoding="utf-8"))
    contract_hash = contract.pop("contract_sha256", None)
    if contract_hash != canonical_hash(contract): raise ValueError("prepared run contract hash is invalid")
    if contract.get("schema") != SCHEMA: raise ValueError("unsupported run contract schema")
    runtime = contract["runtime"]
    for key, value in runtime.items(): setattr(args, key, value)
    args.base = contract["base_path"]; args.run_id = contract["run_id"]
    args.checkpoint = contract["arm_identity"].get("checkpoint")
    args.base_only = contract["arm_identity"].get("kind") == "base-only"
    args.config = contract["config_path"]
    suite = contract["suite"]; validate_suite(suite, config_path=args.config)
    fresh = discover_suite(args.config, suite["split"], shards=int(suite["requested_shards"]))
    if fresh["pool_sha256"] != suite["pool_sha256"] or fresh["tasks"] != suite["tasks"]:
        raise ValueError("actual installed ALFWorld pool differs from prepared suite")
    shards = suite["shards"]
    if args.world_size != len(shards) or not 0 <= args.rank < len(shards):
        raise ValueError(f"rank/world_size must select one of {len(shards)} non-empty prepared shards")
    spec = shards[args.rank]
    if not spec["task_ids"]: raise ValueError("empty shard is forbidden")
    from ez_train_v2_contract import base_file_inventory, base_inventory_sha256
    base = Path(args.base).resolve(strict=True); args.base = str(base)
    base_map = base_file_inventory(base); base_digest = base_inventory_sha256(base)
    base_inv = [{"path": k, **v} for k, v in sorted(base_map.items())]
    if base_digest != contract["protocol_signature"]["base_inventory_sha256"] or base_inv != contract["base_inventory"]:
        raise ValueError("local base model bytes differ from prepared run contract")
    args.base_inventory_sha256 = base_digest
    arm = _checkpoint_info(args, base_digest)
    # Adapter leaf is validated by the shared v3 contract before model loading.
    if not args.base_only: args.adapter_path = arm["adapter_path"]
    if arm != contract["arm_identity"]:
        raise ValueError("checkpoint/adapter identity differs from prepared run contract")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True, local_files_only=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    chat = getattr(tok, "chat_template", None)
    chat_hash = hashlib.sha256(str(chat).encode()).hexdigest() if chat is not None else None
    token_files = {k: v for k, v in base_map.items() if k.startswith("tokenizer") or
                   k in ("vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json")}
    from ez_prompts import SOLVER_SYSTEM, SOLVER_USER_TEMPLATE
    args.prompt_sha256 = hashlib.sha256((SOLVER_SYSTEM + "\0" + SOLVER_USER_TEMPLATE).encode()).hexdigest()
    sig = protocol_signature(suite, tokenizer_hash=canonical_hash(token_files), chat_hash=chat_hash,
                             backend=args.backend, args=args)
    if sig != contract["protocol_signature"]:
        raise ValueError("runtime protocol differs from immutable prepared run contract")
    if software_versions() != contract["software"] or runtime_sources() != contract["runtime_source_sha256"]:
        raise ValueError("software or evaluation source differs from prepared run contract")
    expected_output = contract["expected_shards"][args.rank]["path"]
    if str(Path(args.output).resolve()) != expected_output:
        raise ValueError("rank output path differs from immutable run contract")
    policy = _load_v2_policy(args, args.base, tok=tok)
    run_id = args.run_id
    from alfworld_env import ALFWorldBatchEnv, load_alfworld_config
    from eval_ez import _rollout_batch
    from ez_eval_protocol import aggregate_complete_evaluation
    from rollout import RolloutConfig
    env_cfg = load_alfworld_config(args.config)
    by_id = {t["task_id"]: t for t in suite["tasks"]}
    tasks = [by_id[x] for x in spec["task_ids"]]
    rows = []
    cfg = RolloutConfig(group_size=1, max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
        top_k=args.top_k, repetition_penalty=1.0, max_context_tokens=args.max_context_tokens,
        stop_mode="tool_call", max_batch_size=args.batch)
    started = time.time()
    for si in range(SAMPLES):
        for start in range(0, len(tasks), args.batch):
            task_batch = tasks[start:start + args.batch]
            seeds = [sample_seed(args.seed, t["task_id"], si) for t in task_batch]
            recs = [{"task_id": t["task_id"], "task_type": t["task_type"],
                     "sample": si, "sample_seed": seed} for t, seed in zip(task_batch, seeds)]
            env = ALFWorldBatchEnv(env_cfg, split=suite["alfworld_split"], batch_size=len(task_batch),
                seed=seeds[0], max_episode_steps=args.max_steps, with_expert=False,
                with_facts=False, ignore_benchmark_done=False)
            try:
                batch = _rollout_batch(env, policy, [t["game_file"] for t in task_batch], recs, cfg,
                                       args.max_steps, style="ours", menu_enabled=False,
                                       capture_v2_trace=True)
            finally: env.close()
            by_tid = {r["task_id"]: r for r in batch}
            if set(by_tid) != {t["task_id"] for t in task_batch}:
                raise ValueError("environment reset did not preserve exact assigned task slots")
            for task, seed in zip(task_batch, seeds):
                row = by_tid[task["task_id"]]
                if row["game_file"] != task["game_file"] or row["sample_seed"] != seed:
                    raise ValueError("reset slot or sample seed does not match assigned task")
                out_row = {"task_id": task["task_id"], "task_type": task["task_type"],
                    "game_file": task["game_file"], "game_sha256": task["game_sha256"], "sample_index": si, "seed": seed,
                    "won": bool(row["won"]), "steps": row["steps"], "trace": row["trace"],
                    "initial_prompt_token_ids": row["initial_prompt_token_ids"],
                    "chat_template": row["chat_template"], "actions": row["actions"],
                    "kinds": row["kinds"], "truncated": any(t.get("truncated") for t in row["trace"])}
                out_row["record_sha256"] = canonical_hash(out_row)
                rows.append(out_row)
    expected = {t["task_id"]: t["task_type"] for t in tasks}
    check_rows = [{"task_id": r["task_id"], "task_type": r["task_type"],
                   "sample": r["sample_index"], "won": r["won"]} for r in rows]
    summary = aggregate_complete_evaluation(check_rows, expected, SAMPLES)
    result = {"schema": SCHEMA, "run_id": run_id, "rank": args.rank,
        "contract_sha256": contract_hash,
        "split": suite["split"], "pool_sha256": suite["pool_sha256"],
        "task_ids": list(spec["task_ids"]), "protocol_signature": sig,
        "arm_identity": arm, "base_inventory": base_inv, "software": software_versions(),
        "runtime_source_sha256": runtime_sources(),
        "summary": summary, "elapsed_seconds": time.time() - started, "sample_rows": rows}
    out = Path(args.output).expanduser()
    if out.exists(): raise FileExistsError(f"refusing to overwrite shard result: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote rank {args.rank}/{args.world_size} ({len(rows)} samples): {out}")
    return 0


def merge_cmd(args):
    expected = json.loads(Path(args.expected_manifest).read_text(encoding="utf-8"))
    specs = expected.get("expected_shards", [])
    results = [json.loads(Path(s["path"]).read_text(encoding="utf-8")) for s in specs]
    merged = strict_merge(expected, results)
    out = Path(args.output)
    if out.exists(): raise FileExistsError(f"refusing to overwrite merged result: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


def compare_cmd(args):
    from ez_eval_v2_contract import compare_pair
    a, b = [json.loads(Path(p).read_text(encoding="utf-8")) for p in (args.base_result, args.adapter_result)]
    result = compare_pair(a, b)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare-suite"); p.add_argument("--config", required=True); p.add_argument("--split", choices=sorted({"valid_seen", "valid_unseen"}), default="valid_unseen"); p.add_argument("--shards", type=int, default=8); p.add_argument("--output", required=True); p.set_defaults(fn=prepare)
    p = sub.add_parser("prepare-run"); p.add_argument("--suite-manifest",required=True); p.add_argument("--config",required=True); p.add_argument("--base",required=True); p.add_argument("--checkpoint"); p.add_argument("--base-only",action="store_true"); p.add_argument("--backend",choices=["hf","vllm"],required=True); p.add_argument("--seed",type=int,default=42); p.add_argument("--run-id"); p.add_argument("--results-dir",required=True); p.add_argument("--output",required=True); p.add_argument("--batch",type=int,default=16); p.add_argument("--max-steps",type=int,default=30); p.add_argument("--max-context-tokens",type=int,default=40960); p.add_argument("--max-new-tokens",type=int,default=1024); p.add_argument("--temperature",type=float,default=1.0); p.add_argument("--top-p",type=float,default=0.95); p.add_argument("--top-k",type=int,default=-1); p.add_argument("--gpu-memory-utilization",type=float,default=0.85); p.add_argument("--attn",choices=["flash_attention_2","sdpa","eager"],default="sdpa"); p.set_defaults(fn=prepare_run)
    p = sub.add_parser("run"); p.add_argument("--run-contract",required=True); p.add_argument("--rank",type=int,required=True); p.add_argument("--world-size",type=int,required=True); p.add_argument("--output",required=True); p.set_defaults(fn=run_eval)
    p = sub.add_parser("merge"); p.add_argument("--expected-manifest", required=True); p.add_argument("--output", required=True); p.set_defaults(fn=merge_cmd)
    p = sub.add_parser("compare-pair"); p.add_argument("--base-result", required=True); p.add_argument("--adapter-result", required=True); p.add_argument("--output", required=True); p.set_defaults(fn=compare_cmd)
    args = ap.parse_args(argv)
    try: return args.fn(args)
    except (ValueError, OSError, KeyError, ImportError) as e: ap.error(str(e))


if __name__ == "__main__": raise SystemExit(main())
