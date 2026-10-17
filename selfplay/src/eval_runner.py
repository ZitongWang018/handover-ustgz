#!/usr/bin/env python3
"""One matched full-pool ReAct evaluator for SkillRise baseline and EZ checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from evaluation import (SCHEMA, SEEDS, align_task_slots, canonical_hash, compare, discover_suite, merge,
                        request_seed, sha256_file, validate_suite)


def _load_yaml(path: str | Path) -> dict:
    import yaml
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("solver configuration must be a YAML mapping")
    return value


def _alfworld_config_path(config_path: Path, config: dict) -> Path:
    relative = config.get("env", {}).get("config_path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("config.env.config_path must name the one ALFWorld config")
    path = Path(relative).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve(strict=True)


def _solver_config(config: dict) -> dict:
    value = config.get("solver", {})
    if not isinstance(value, dict):
        raise ValueError("config.solver must be a mapping")
    return value


def _eval_config(config: dict) -> dict:
    value = config.get("eval", {})
    if not isinstance(value, dict):
        raise ValueError("config.eval must be a mapping")
    return value


def _versions() -> dict:
    result = {"python": sys.version.split()[0]}
    for name in ("alfworld", "textworld", "torch", "transformers", "peft", "vllm", "numpy"):
        try: result[name] = version(name)
        except PackageNotFoundError: result[name] = "not-installed"
    return result


def _source_identity() -> dict:
    names = ("eval_runner.py", "evaluation.py", "protocol.py", "alfworld_env.py",
             "alfworld_compat.py", "checkpoint.py", "rollout.py", "ez_vllm_policy.py")
    missing = [name for name in names if not (SRC / name).is_file()]
    if missing:
        raise ValueError("required evaluator source is missing: " + ", ".join(missing))
    return {name: sha256_file(SRC / name) for name in names}


def _base_identity(base: str) -> tuple[Path, dict, str]:
    from checkpoint import base_file_inventory
    path = Path(base).expanduser().resolve(strict=True)
    inventory = base_file_inventory(path)
    digest = canonical_hash(inventory)
    serial = [{"path": rel, **record} for rel, record in sorted(inventory.items())]
    return path, {"files": serial}, digest


def _arm(args, base_digest: str, solver_config: dict) -> dict:
    if bool(args.base_only) == bool(args.checkpoint):
        raise ValueError("choose exactly one of --base-only or --checkpoint")
    if args.base_only:
        return {"kind": "base-only", "base_inventory_sha256": base_digest}
    from checkpoint import validate_checkpoint_v3
    cp = Path(args.checkpoint).expanduser().resolve(strict=True)
    validated = validate_checkpoint_v3(str(cp), expected_contract={
        "base_inventory_sha256": base_digest}, purpose="evaluation")
    manifest = validated["manifest"]
    if manifest.get("resolved_config", {}).get("solver") != solver_config:
        raise ValueError("checkpoint solver configuration differs from the unified evaluation setting")
    source_files = manifest.get("source_files", {})
    protocol_record = source_files.get("selfplay/src/protocol.py")
    if not isinstance(protocol_record, dict) or protocol_record.get("sha256") != sha256_file(SRC / "protocol.py"):
        raise ValueError("checkpoint was trained with different SkillRise prompt/action protocol source")
    solver_path = Path(validated["solver_path"]).resolve(strict=True)
    if not (solver_path / "adapter_config.json").is_file():
        raise ValueError("validated checkpoint does not expose its solver adapter leaf")
    adapter_files = [{"path": p.relative_to(solver_path).as_posix(), "size": p.stat().st_size,
                      "sha256": sha256_file(p)} for p in sorted(solver_path.rglob("*")) if p.is_file()]
    return {"kind": "checkpoint", "path": str(cp),
        "manifest_sha256": canonical_hash(validated["manifest"]),
        "solver_path": str(solver_path), "solver_files": adapter_files,
        "base_inventory_sha256": base_digest}


def _config_settings(config: dict) -> dict:
    solver, ev = _solver_config(config), _eval_config(config)
    expected = {"style": "react", "max_turns": 30, "max_context_tokens": 32768,
        "keep_recent_turns": 15, "max_new_tokens": 1024, "temperature": 0.7,
        "top_p": 0.95, "top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0,
        "stop_mode": "action_tag", "menu_enabled": True}
    settings = {}
    for key in expected:
        if key not in solver:
            raise ValueError(f"config.solver is missing required field: {key}")
        settings[key] = solver[key]
    for key in ("seeds", "split", "batch_size", "shards", "backend", "gpu_memory_utilization"):
        if key not in ev:
            raise ValueError(f"config.eval is missing required field: {key}")
    settings["seeds"] = ev["seeds"]
    settings["split"] = ev["split"]
    settings["batch_size"] = ev["batch_size"]
    settings["shards"] = ev["shards"]
    settings["backend"] = ev["backend"]
    settings["gpu_memory_utilization"] = ev["gpu_memory_utilization"]
    seeds_valid = (isinstance(settings["seeds"], list) and
                   all(isinstance(seed, int) and not isinstance(seed, bool) for seed in settings["seeds"]) and
                   tuple(settings["seeds"]) == SEEDS)
    if (settings["style"] != "react" or not seeds_valid or
            settings["split"] != "valid_seen" or settings["max_turns"] != 30 or
            settings["max_context_tokens"] != 32768 or settings["keep_recent_turns"] != 15 or
            settings["max_new_tokens"] != 1024 or settings["stop_mode"] != "action_tag" or
            settings["menu_enabled"] is not True or settings["temperature"] != 0.7 or
            settings["top_p"] != 0.95 or settings["top_k"] != -1 or
            settings["min_p"] != 0.0 or settings["repetition_penalty"] != 1.0):
        raise ValueError("config must declare the approved ReAct / valid_seen / five-seed protocol")
    if settings["backend"] not in {"vllm", "hf"} or settings["batch_size"] < 1:
        raise ValueError("config has unsupported backend or batch size")
    if (isinstance(settings["shards"], bool) or not isinstance(settings["shards"], int) or
            not 1 <= settings["shards"] <= 8 or
            not isinstance(settings["gpu_memory_utilization"], (int, float)) or
            not 0 < settings["gpu_memory_utilization"] <= 1):
        raise ValueError("config has invalid evaluation shard or GPU memory settings")
    return settings


def prepare(args) -> int:
    config_path = Path(args.config).expanduser().resolve(strict=True)
    config = _load_yaml(config_path)
    settings = _config_settings(config)
    data_root = Path(args.alfworld_data).expanduser().resolve(strict=True)
    if not data_root.is_dir(): raise ValueError("--alfworld-data must point to an existing directory")
    os.environ["ALFWORLD_DATA"] = str(data_root)
    alfworld_config = _alfworld_config_path(config_path, config)
    suite = discover_suite(str(alfworld_config), settings["split"], settings["shards"])
    validate_suite(suite)
    if suite["config_path"] != str(alfworld_config):
        raise ValueError("ALFWorld configuration path differs from config.env.config_path")
    base, base_inventory, base_digest = _base_identity(args.base)
    arm = _arm(args, base_digest, _solver_config(config))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(base), trust_remote_code=True, local_files_only=True)
    token_files = {x["path"]: x for x in base_inventory["files"]
                   if x["path"].startswith("tokenizer") or x["path"] in
                   ("vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json")}
    prompt_source = (SRC / "protocol.py").read_bytes()
    signature = {"pool_sha256": suite["pool_sha256"], "split": suite["split"],
        "config_sha256": sha256_file(config_path), "config": config,
        "solver_settings": settings, "style": "react", "prompt_source_sha256": hashlib.sha256(prompt_source).hexdigest(),
        "tokenizer_files_sha256": canonical_hash(token_files),
        "tokenizer_class": type(tok).__qualname__, "raw_prompt_encoding": "tokenizer.encode(add_special_tokens=True)",
        "action_parser": "SkillRise play projection", "stop": "first decoded </action>, included",
        "seed_policy": "environment reset receives run seed; sha256(seed,task_id,turn) per generation request",
        "seeds": list(SEEDS), "max_turns": settings["max_turns"],
        "alfworld_data_root": str(data_root), "alfworld_config_path": str(alfworld_config),
        "max_context_tokens": settings["max_context_tokens"], "max_new_tokens": settings["max_new_tokens"],
        "temperature": settings["temperature"], "top_p": settings["top_p"],
        "top_k": settings["top_k"], "min_p": settings["min_p"],
        "repetition_penalty": settings["repetition_penalty"], "backend": settings["backend"],
        "dtype": "bfloat16", "base_inventory_sha256": base_digest,
        "batch_size": settings["batch_size"], "gpu_memory_utilization": settings["gpu_memory_utilization"],
        "software": _versions(), "runtime_source_sha256": _source_identity()}
    contract = {"schema": SCHEMA, "run_id": args.run_id or uuid.uuid4().hex,
        "suite": suite, "config_path": str(config_path),
        "alfworld_data_root": str(data_root), "alfworld_config_path": str(alfworld_config), "base_path": str(base),
        "base_inventory": base_inventory, "arm_identity": arm,
        "protocol_signature": signature,
        "expected_shards": [{"rank": sh["rank"], "task_ids": sh["task_ids"],
            "path": str((Path(args.results_dir).resolve() / f"rank{sh['rank']}.json"))}
            for sh in suite["shards"]]}
    contract["contract_sha256"] = canonical_hash(contract)
    out = Path(args.output)
    if out.exists(): raise FileExistsError(f"refusing to overwrite contract: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"prepared {len(suite['tasks'])} tasks x {len(SEEDS)} seeds; contract={contract['contract_sha256']}")
    return 0


def _load_policy(contract: dict, tokenizer):
    sig, arm = contract["protocol_signature"], contract["arm_identity"]
    if arm.get("kind") not in {"base-only", "checkpoint"}:
        raise ValueError(f"unsupported evaluation arm kind: {arm.get('kind')!r}")
    if sig["backend"] == "vllm":
        from ez_vllm_policy import VLLMPolicy, build_vllm_engine
        engine = build_vllm_engine(contract["base_path"], gpu_memory_utilization=sig["gpu_memory_utilization"],
            max_model_len=sig["max_context_tokens"], max_lora_rank=32,
            max_num_seqs=sig["batch_size"], seed=SEEDS[0],
            enable_lora=arm["kind"] == "checkpoint")
        policy = VLLMPolicy(engine, tokenizer)
        if arm["kind"] == "checkpoint": policy.set_lora("solver", arm["solver_path"])
        return policy
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(contract["base_path"], torch_dtype=torch.bfloat16,
        trust_remote_code=True, local_files_only=True).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    if arm["kind"] == "checkpoint":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, arm["solver_path"], is_trainable=False,
            local_files_only=True).eval()
    from rollout import HFPolicy
    return HFPolicy(model, tokenizer)


def run(args) -> int:
    from protocol import (action_tag_stop_len, build_skillrise_prompt, encode_raw_prompt,
                          project_skillrise_action)
    from alfworld_compat import find_compat_dir, install
    compat = find_compat_dir()
    if compat: install(verbose=False)
    run_path = Path(args.contract).resolve(strict=True)
    contract = json.loads(run_path.read_text(encoding="utf-8"))
    digest = contract.pop("contract_sha256", None)
    if digest != canonical_hash(contract) or contract.get("schema") != SCHEMA:
        raise ValueError("invalid immutable evaluation contract")
    contract["contract_sha256"] = digest
    data_root = Path(contract["alfworld_data_root"]).expanduser().resolve(strict=True)
    if not data_root.is_dir(): raise ValueError("prepared ALFWorld data root is missing or not a directory")
    os.environ["ALFWORLD_DATA"] = str(data_root)
    config = _load_yaml(contract["config_path"])
    if str(_alfworld_config_path(Path(contract["config_path"]), config)) != contract["alfworld_config_path"]:
        raise ValueError("ALFWorld config path differs from the prepared solver config")
    validate_suite(contract["suite"])
    rank = args.rank
    if not 0 <= rank < len(contract["suite"]["shards"]): raise ValueError("rank out of range")
    spec = contract["suite"]["shards"][rank]
    if Path(args.output).resolve() != Path(contract["expected_shards"][rank]["path"]).resolve():
        raise ValueError("output path differs from prepared contract")
    if _versions() != contract["protocol_signature"]["software"] or _source_identity() != contract["protocol_signature"]["runtime_source_sha256"]:
        raise ValueError("software or evaluator source differs from prepared contract")
    fresh = discover_suite(contract["alfworld_config_path"], contract["suite"]["split"],
                           len(contract["suite"]["shards"]))
    if fresh["pool_sha256"] != contract["suite"]["pool_sha256"] or fresh["tasks"] != contract["suite"]["tasks"]:
        raise ValueError("installed valid_seen task pool differs from the prepared contract")
    settings = _config_settings(config)
    if sha256_file(contract["config_path"]) != contract["protocol_signature"]["config_sha256"]:
        raise ValueError("solver config changed after prepare")
    base, base_inventory, base_digest = _base_identity(contract["base_path"])
    if base_digest != contract["protocol_signature"]["base_inventory_sha256"] or base_inventory != contract["base_inventory"]:
        raise ValueError("base model files changed after prepare")
    arm = _arm(argparse.Namespace(base_only=contract["arm_identity"]["kind"] == "base-only",
                                  checkpoint=contract["arm_identity"].get("path")), base_digest,
               _solver_config(config))
    if arm != contract["arm_identity"]: raise ValueError("checkpoint identity differs from prepared contract")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(base), trust_remote_code=True, local_files_only=True)
    task_by_id = {t["task_id"]: t for t in contract["suite"]["tasks"]}
    tasks = [task_by_id[tid] for tid in spec["task_ids"]]
    policy = _load_policy(contract, tok)
    from alfworld_env import ALFWorldBatchEnv, load_alfworld_config
    from rollout import RolloutConfig
    env_config = load_alfworld_config(contract["alfworld_config_path"])
    cfg = RolloutConfig(group_size=1, max_steps=settings["max_turns"],
        max_new_tokens=settings["max_new_tokens"], temperature=settings["temperature"],
        top_p=settings["top_p"], top_k=settings["top_k"], min_p=settings["min_p"],
        repetition_penalty=settings["repetition_penalty"], max_context_tokens=settings["max_context_tokens"],
        keep_recent_turns=settings["keep_recent_turns"], stop_mode="action_tag", max_batch_size=settings["batch_size"])
    rows, start_time = [], time.time()
    for seed in SEEDS:
        for start in range(0, len(tasks), settings["batch_size"]):
            task_batch = tasks[start:start + settings["batch_size"]]
            env = ALFWorldBatchEnv(env_config, split=contract["suite"]["alfworld_split"],
                batch_size=len(task_batch), max_episode_steps=settings["max_turns"],
                seed=seed, with_expert=False, with_facts=False, ignore_benchmark_done=False)
            try:
                _obs, views, admissible, used, _ = env.reset([t["game_file"] for t in task_batch], seed=seed)
                task_batch = align_task_slots(task_batch, used)
                init_obs = {t["task_id"]: (views[i].raw or "").strip() for i, t in enumerate(task_batch)}
                histories = {t["task_id"]: [] for t in task_batch}
                adm = {t["task_id"]: list(admissible[i]) for i, t in enumerate(task_batch)}
                active = {t["task_id"] for t in task_batch}
                won = {t["task_id"]: False for t in task_batch}
                traces = {t["task_id"]: [] for t in task_batch}
                for turn in range(settings["max_turns"]):
                    active_tasks = [t for t in task_batch if t["task_id"] in active]
                    if not active_tasks: break
                    prompts = [build_skillrise_prompt(init_obs[t["task_id"]], histories[t["task_id"]],
                        adm[t["task_id"]], turn,
                        history_length=settings["keep_recent_turns"]) for t in active_tasks]
                    prompt_ids = [encode_raw_prompt(tok, prompt) for prompt in prompts]
                    if any(len(ids) + settings["max_new_tokens"] > settings["max_context_tokens"] for ids in prompt_ids):
                        raise ValueError("raw SkillRise prompt plus generation budget exceeds 32768-token context")
                    seeds = [request_seed(seed, t["task_id"], turn) for t in active_tasks]
                    if settings["backend"] == "hf":
                        generated_ids = []
                        for ids, req_seed in zip(prompt_ids, seeds):
                            _, one_ids, _ = policy.generate_from_token_ids([ids], cfg, request_seeds=[req_seed])
                            generated_ids.append(one_ids[0])
                    else:
                        _, generated_ids, _ = policy.generate_from_token_ids(prompt_ids, cfg, request_seeds=seeds)
                    env_actions = ["look"] * len(task_batch)
                    slot_index = {t["task_id"]: i for i, t in enumerate(task_batch)}
                    finish_reasons = getattr(policy, "last_finish_reasons", [])
                    for j, (t, prompt, pids, raw_ids, req_seed) in enumerate(zip(active_tasks, prompts, prompt_ids, generated_ids, seeds)):
                        stop_len = action_tag_stop_len(tok, raw_ids)
                        output_ids = list(raw_ids[:stop_len])
                        raw = tok.decode(output_ids, skip_special_tokens=True)
                        action, tag_valid = project_skillrise_action(raw)
                        command = action if action else "look"
                        env_actions[slot_index[t["task_id"]]] = command
                        traces[t["task_id"]].append({"turn": turn, "raw_prompt": prompt,
                            "prompt_token_ids": pids,
                            "output_token_ids": output_ids, "raw_output": raw,
                            "action_tag_valid": bool(tag_valid), "parser_projection": action,
                            "executed_action": command, "request_seed": req_seed,
                            "admissible_actions": adm[t["task_id"]], "stop_len": stop_len,
                            "finish_reason": finish_reasons[j] if j < len(finish_reasons) else None,
                            "truncated": len(raw_ids) >= settings["max_new_tokens"]})
                    step = env.step(env_actions)
                    for i, t in enumerate(task_batch):
                        if t["task_id"] not in active: continue
                        tid = t["task_id"]
                        obs = str(step.obs[i])
                        traces[tid][-1].update({"observation": obs, "won": bool(step.won[i]),
                                               "done": bool(step.dones[i])})
                        histories[tid].append((traces[tid][-1]["executed_action"], obs))
                        adm[tid] = list(step.admissible[i])
                        won[tid] = bool(step.won[i])
                        if step.dones[i] or step.won[i]: active.remove(tid)
                for t in task_batch:
                    tid = t["task_id"]
                    row = {"task_id": tid, "task_type": t["task_type"],
                        "game_file": t["game_file"], "game_sha256": t["game_sha256"],
                        "seed": seed, "environment_reset_seed": seed,
                        "won": bool(won[tid]), "steps": len(traces[tid]), "trace": traces[tid]}
                    row["record_sha256"] = canonical_hash(row)
                    rows.append(row)
            finally:
                env.close()
    summary = __import__("evaluation").summarize(rows, tasks, require_six_types=False)
    result = {"schema": SCHEMA, "run_id": contract["run_id"], "contract_sha256": digest,
        "rank": rank, "task_ids": list(spec["task_ids"]),
        "protocol_signature": contract["protocol_signature"], "arm_identity": contract["arm_identity"],
        "base_inventory": contract["base_inventory"],
        "software": contract["protocol_signature"]["software"],
        "runtime_source_sha256": contract["protocol_signature"]["runtime_source_sha256"],
        "rows": rows, "summary": summary, "elapsed_seconds": time.time() - start_time}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists(): raise FileExistsError(f"refusing to overwrite rank result: {out}")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"rank {rank}: wrote {len(rows)} episodes to {out}")
    return 0


def merge_cmd(args) -> int:
    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    shards = [json.loads(Path(x["path"]).read_text(encoding="utf-8")) for x in contract["expected_shards"]]
    result = merge(contract, shards)
    out = Path(args.output)
    if out.exists(): raise FileExistsError(f"refusing to overwrite merged result: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"merged {result['summary']['n_episodes']} episodes; SR={result['summary']['episode_weighted_sr']:.4f}")
    return 0


def compare_cmd(args) -> int:
    a, b = [json.loads(Path(p).read_text(encoding="utf-8")) for p in (args.baseline, args.trained)]
    result = compare(a, b)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists(): raise FileExistsError(f"refusing to overwrite comparison: {out}")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"paired episode SR delta={result['paired_episode_sr_delta_mean']:+.4f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--config", default=str(SRC.parent / "config.yaml")); p.add_argument("--alfworld-data", required=True)
    p.add_argument("--base", required=True); p.add_argument("--checkpoint"); p.add_argument("--base-only", action="store_true")
    p.add_argument("--run-id"); p.add_argument("--results-dir", required=True); p.add_argument("--output", required=True); p.set_defaults(fn=prepare)
    p = sub.add_parser("run"); p.add_argument("--contract", required=True); p.add_argument("--rank", type=int, required=True); p.add_argument("--output", required=True); p.set_defaults(fn=run)
    p = sub.add_parser("merge"); p.add_argument("--contract", required=True); p.add_argument("--output", required=True); p.set_defaults(fn=merge_cmd)
    p = sub.add_parser("compare"); p.add_argument("--baseline", required=True); p.add_argument("--trained", required=True); p.add_argument("--output", required=True); p.set_defaults(fn=compare_cmd)
    args = ap.parse_args(argv)
    try: return args.fn(args)
    except (ValueError, OSError, KeyError, ImportError) as exc: ap.error(str(exc))


if __name__ == "__main__": raise SystemExit(main())
