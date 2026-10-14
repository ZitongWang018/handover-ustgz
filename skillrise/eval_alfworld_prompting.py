#!/usr/bin/env python
"""Eight-rank code-aligned SkillRise GRPO validation for ALFWorld."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

from eval_protocol import (
    ATTEMPTS, ENVIRONMENT_STEP_BUDGET, MAX_PROMPT_TOKENS, MAX_RESPONSE_TOKENS,
    MAX_TURNS_PER_ATTEMPT,
    PROFILE, SHARD_COUNT, SLOTS_PER_SHARD, SPLIT, ProtocolError, aggregate_rows,
    decode_vllm_token_ids, game_id, make_selection_manifest, sha256_bytes,
    sha256_json, shard_slots, tokenize_upstream_prompt,
    validate_selection_manifest, validate_shard_grid, verify_preflight,
    UPSTREAM_AGENT_SAMPLING, upstream_engine_kwargs,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SELFPLAY = ROOT / "selfplay"
UPSTREAM = HERE / "upstream"
UPSTREAM_CONFIG = UPSTREAM / "agent_system/environments/alfworld/configs/config_tw.yaml"
SOURCE_MANIFEST = UPSTREAM / "SOURCE_MANIFEST.json"
RUNTIME_COMPAT_INFO = {"mode": "not_initialized", "path": None}


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _runtime_imports():
    global RUNTIME_COMPAT_INFO
    sys.path.insert(0, str(SELFPLAY / "src"))
    compat = __import__("alfworld_compat", fromlist=["install", "find_compat_dir"])
    compat_dir = compat.find_compat_dir()
    require_compat = os.environ.get("SKR_REQUIRE_GLIBC_COMPAT", "0") == "1"
    if compat_dir:
        compat.install()
        RUNTIME_COMPAT_INFO = {"mode": "shim_installed", "path": compat_dir}
    elif require_compat:
        raise RuntimeError("SKR_REQUIRE_GLIBC_COMPAT=1 but no compatible libdownward shim was found")
    else:
        RUNTIME_COMPAT_INFO = {"mode": "native_fast_downward", "path": None}
    from alfworld_env import load_alfworld_config
    # Prefer the pinned ALFWorld environment source; dependencies remain in the
    # already configured Python environment.
    sys.path.insert(0, str(UPSTREAM / "agent_system/environments/alfworld"))
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
    return AlfredTWEnv, load_alfworld_config


def _upstream_imports():
    sys.path.insert(0, str(UPSTREAM))
    from agent_system.environments.alfworld.prompt import ALFWORLD_PLAY_PROMPT, get_alfworld_prompt
    from agent_system.environments.alfworld.memory import SimpleMemory
    from agent_system.environments.alfworld.projection import alfworld_projection
    return ALFWORLD_PLAY_PROMPT, get_alfworld_prompt, SimpleMemory, alfworld_projection


def _config_pool():
    AlfredTWEnv, load = _runtime_imports()
    data_root = os.environ.get("ALFWORLD_DATA", "")
    if not data_root or not Path(data_root).is_dir():
        raise ProtocolError("ALFWORLD_DATA must point to an existing local dataset directory")
    cfg = load(str(UPSTREAM_CONFIG))
    # The source config's TextWorld environment budget is 50. The rollout
    # collector applies its independent ten-turn cap per attempt.
    cfg["dagger"]["training"]["max_nb_steps_per_episode"] = ENVIRONMENT_STEP_BUDGET
    cfg["rl"]["training"]["max_nb_steps_per_episode"] = ENVIRONMENT_STEP_BUDGET
    base_env = AlfredTWEnv(cfg, train_eval=SPLIT)
    pool = list(base_env.game_files)
    if not pool:
        raise ProtocolError(f"No games found for {SPLIT}; check ALFWORLD_DATA")
    return cfg, pool, base_env


def _task_type(game_file: str) -> str:
    p = Path(game_file.replace("game.tw-pddl", "traj_data.json"))
    with p.open(encoding="utf-8") as f:
        value = json.load(f)["task_type"]
    if value not in ("pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep",
                     "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place"):
        raise ProtocolError(f"Unknown task type {value!r} in {p}")
    return value


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def prepare_selection(path: Path):
    if path.exists():
        raise FileExistsError(f"Selection manifest exists: {path}")
    _, pool, _ = _config_pool()
    root = os.environ.get("ALFWORLD_DATA", "")
    ids = [game_id(g, root or None) for g in pool]
    pool_hash = sha256_bytes(json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode())
    selection = make_selection_manifest(
        pool, _task_type, pool_order_sha256=pool_hash,
        data_config_sha256=_sha_file(UPSTREAM_CONFIG),
        source_manifest_sha256=_sha_file(SOURCE_MANIFEST),
    )
    selection["data_root"] = str(Path(root).resolve()) if root else None
    selection["pool_game_ids"] = ids
    validate_selection_manifest(selection)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, selection)
    return selection


def validate_selection_against_local(selection, pool):
    """Reject stale selection/source/data metadata before allocating a GPU model."""
    root = os.environ.get("ALFWORLD_DATA", "")
    ids = [game_id(g, root or None) for g in pool]
    current_pool_hash = sha256_bytes(json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode())
    expected = {
        "pool_order_sha256": current_pool_hash,
        "data_config_sha256": _sha_file(UPSTREAM_CONFIG),
        "source_manifest_sha256": _sha_file(SOURCE_MANIFEST),
    }
    mismatches = [key for key, value in expected.items() if selection.get(key) != value]
    if mismatches:
        raise ProtocolError("Selection manifest does not match local " + ", ".join(mismatches))
    return expected


def _prompt(get_prompt, template, initial_obs, memory, admissible, turn, attempt, style):
    curr, _ = memory.fetch(history_length=15)
    menu = ",\n ".join(f"'{a}'" for a in admissible if a != "help")
    result = get_prompt(
        phase="play", turn_idx=turn, traj_idx=attempt, init_observation=initial_obs,
        curr_traj=curr[0], past_traj={}, admissible_actions=menu,
        reflection={}, reflection_type="reflection_only",
    )
    if style == "react":
        return result
    marker = "- Your response should first by step-by-step reasoning about the current situation.\n- Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."
    if marker not in template:
        raise ProtocolError("Cannot safely derive custom zero-shot prompt from changed upstream template")
    return result.replace(marker, "- Your response should choose an admissible action for the current step and present it within <action> </action> tags.")


def _engine(base):
    import torch
    from vllm import LLM, SamplingParams
    torch.set_num_threads(1)
    model = LLM(model=base, **upstream_engine_kwargs())
    # Match generate_sequences_agent, not the unused vanilla generate_sequences path:
    # it sets no stop strings. Base config supplies detokenize=False and response length 1024.
    sampling = SamplingParams(**UPSTREAM_AGENT_SAMPLING)
    return model, sampling


def _restart_same_game(env):
    """Match AlfworldWorker.restart without reloading/recompiling the game."""
    tw = env
    for _ in range(4):
        if hasattr(tw, "batch_env") and hasattr(tw, "last_commands"):
            break
        tw = getattr(tw, "env", None)
        if tw is None:
            raise ProtocolError("Could not find TextWorld batch environment under Gym wrappers")
    else:
        raise ProtocolError("Could not find TextWorld batch environment under Gym wrappers")
    tw.last_commands = [None]
    tw.obs, infos = tw.batch_env.reset()
    return tw.obs, infos


def _generation_parity_check(llm, tokenizer, prompts, projection, admissible,
                             current_sampling, report_path):
    """Compare upstream constructor parameters with the active adapter params."""
    from vllm import SamplingParams

    if len(prompts) != 2 or len(admissible) != 2:
        raise ProtocolError("Generation parity check requires exactly two tokenized prompts")
    diagnostic_seed = 20261015
    reference_args = {**UPSTREAM_AGENT_SAMPLING, "logprobs": 0, "seed": diagnostic_seed}
    adapter_args = {
        key: getattr(current_sampling, key, value)
        for key, value in UPSTREAM_AGENT_SAMPLING.items()
    }
    adapter_args["logprobs"] = getattr(current_sampling, "logprobs", 0)
    adapter_args["seed"] = diagnostic_seed
    refs = llm.generate(prompts, SamplingParams(**reference_args), use_tqdm=False)
    adapted = llm.generate(prompts, SamplingParams(**adapter_args), use_tqdm=False)

    def results(outputs):
        decoded, ids, projected = [], [], []
        for output in outputs:
            text, token_ids = decode_vllm_token_ids(output.outputs[0], tokenizer)
            decoded.append(text)
            ids.append(token_ids)
            action, _ = projection([text], [admissible[len(decoded) - 1]], phase="play")
            projected.append(action[0])
        return decoded, ids, projected

    ref_text, ref_ids, ref_actions = results(refs)
    got_text, got_ids, got_actions = results(adapted)
    report = {
        "prompts_token_ids": [list(p["prompt_token_ids"]) for p in prompts],
        "reference_sampling_params": reference_args,
        "adapter_sampling_params": adapter_args,
        "reference_text": ref_text, "adapter_text": got_text,
        "reference_token_ids": ref_ids, "adapter_token_ids": got_ids,
        "reference_projection": ref_actions, "adapter_projection": got_actions,
        "token_ids_equal": ref_ids == got_ids,
        "decoded_text_equal": ref_text == got_text,
        "projection_equal": ref_actions == got_actions,
    }
    report["equivalent"] = all(report[k] for k in
                               ("token_ids_equal", "decoded_text_equal", "projection_equal"))
    _write_json(report_path, report)
    if not report["equivalent"]:
        raise ProtocolError(f"Generation parameter parity check failed; see {report_path}")
    return report


def run_rank(args, selection):
    if args.samples != 1:
        raise ProtocolError("Only samples=1 is supported; repeats require a separate suite-level protocol")
    validate_selection_manifest(selection)
    slots = shard_slots(selection, args.rank, args.world_size)
    if args.preflight:
        slots = slots[:2]

    base_path = Path(args.base).expanduser()
    if not base_path.is_dir() or not (base_path / "config.json").is_file():
        raise ProtocolError("--base must be an existing local model directory containing config.json; Hub download is disabled")

    _, pool, base_env = _config_pool()
    local_selection_checks = validate_selection_against_local(selection, pool)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=False, local_files_only=True)
    llm, sampling = _engine(args.base)
    template, get_prompt, SimpleMemory, projection = _upstream_imports()

    envs, rows = [], []
    generation_seconds = 0.0
    environment_seconds = 0.0
    parity_report = None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.out_dir / ("preflight.jsonl" if args.preflight else f"rank-{args.rank:02d}.jsonl")
    start_time = time.time()
    try:
        # This reproduces AlfworldWorker: init one TextWorld env per worker from
        # the complete ordered pool, then seed each worker independently.
        for slot in slots:
            env = base_env.init_env(batch_size=1)
            env.seed(int(slot["worker_seed"]))
            envs.append(env)
        initial_obs, adm = [], []
        workers = min(SLOTS_PER_SHARD, len(envs))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="alfworld-env") as executor:
            env_t0 = time.perf_counter()
            reset_results = list(executor.map(lambda env: env.reset(), envs))
            environment_seconds += time.perf_counter() - env_t0
        for (obs, infos), slot in zip(reset_results, slots):
            used = infos.get("extra.gamefile", [None])[0]
            if used != slot["game_file"]:
                raise ProtocolError(f"Slot {slot['slot']} reset selected {used!r}, expected {slot['game_file']!r}")
            initial_obs.append(obs[0])  # raw reset text, as passed to upstream get_alfworld_prompt
            adm.append(list(infos["admissible_commands"][0]))
        current_obs = list(initial_obs)

        memories = [[SimpleMemory() for _ in slots] for _ in range(ATTEMPTS)]
        for attempt_memories in memories:
            for memory in attempt_memories:
                memory.reset(1)
        cumulative_won = [False] * len(slots)
        attempts_by_slot = [[] for _ in slots]
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="alfworld-env") as executor, \
                trace_path.open("w", encoding="utf-8", newline="\n") as trace:
            for attempt in range(ATTEMPTS):
                if attempt:
                    # Upstream restart resets the same task; it does not advance the worker's pool.
                    env_t0 = time.perf_counter()
                    restart_results = list(executor.map(_restart_same_game, envs))
                    environment_seconds += time.perf_counter() - env_t0
                    for i, ((obs, infos), slot) in enumerate(zip(restart_results, slots)):
                        used = infos.get("extra.gamefile", [slot["game_file"]])[0]
                        if used != slot["game_file"]:
                            raise ProtocolError(f"Restart changed game for slot {slot['slot']}: {used!r}")
                        current_obs[i] = obs[0]
                        adm[i] = list(infos["admissible_commands"][0])
                recs = [{"attempt": attempt, "won": False,
                         "skipped_after_earlier_win": cumulative_won[i],
                         "steps": 0, "valid_action_tag_steps": 0, "trace": []}
                        for i in range(len(slots))]
                done = list(cumulative_won)
                for turn in range(MAX_TURNS_PER_ATTEMPT):
                    active = [i for i in range(len(slots)) if not done[i]]
                    if not active:
                        break
                    prompts, rendered_prompts, tokenized = [], [], []
                    for i in active:
                        prompt = _prompt(get_prompt, template, initial_obs[i], memories[attempt][i],
                                         adm[i], turn, attempt, args.style)
                        rendered, ids = tokenize_upstream_prompt(tok, prompt, MAX_PROMPT_TOKENS)
                        prompts.append(prompt)
                        rendered_prompts.append(rendered)
                        tokenized.append({"prompt_token_ids": ids})
                    gen_t0 = time.perf_counter()
                    if args.preflight and parity_report is None and len(tokenized) == 2:
                        parity_report = _generation_parity_check(
                            llm, tok, tokenized, projection, [adm[i] for i in active],
                            sampling, args.out_dir / "generation-parity.json",
                        )
                    outputs = llm.generate(tokenized, sampling, use_tqdm=False)
                    generation_seconds += time.perf_counter() - gen_t0
                    if len(outputs) != len(active):
                        raise ProtocolError("vLLM generation count differs from active prompt count")
                    prepared = []
                    for out, i in zip(outputs, active):
                        raw, generated_ids = decode_vllm_token_ids(out.outputs[0], tok)
                        admissible_before = list(adm[i])
                        projected, valid = projection([raw], [admissible_before], phase="play")
                        action = projected[0]  # preserve exact upstream fallback, including empty command
                        prepared.append((raw, generated_ids, admissible_before, action, bool(valid[0])))
                    env_t0 = time.perf_counter()
                    step_results = list(executor.map(
                        lambda pair: envs[pair[0]].step([pair[1]]),
                        [(i, prepared[j][3]) for j, i in enumerate(active)],
                    ))
                    environment_seconds += time.perf_counter() - env_t0
                    for j, (i, result) in enumerate(zip(active, step_results)):
                        raw, generated_ids, admissible_before, action, valid_tag = prepared[j]
                        observation_before = current_obs[i]
                        observation, reward, dones, infos = result
                        won = bool(infos["won"][0])
                        env_done = bool(dones[0])
                        adm[i] = list(infos["admissible_commands"][0])
                        memories[attempt][i].store({"text_obs": [observation[0]], "action": [action],
                                                    "reward": [reward[0]], "dones": [env_done], "won": [won]})
                        recs[i]["steps"] += 1
                        recs[i]["valid_action_tag_steps"] += int(valid_tag)
                        recs[i]["won"] = bool(recs[i]["won"] or won)
                        done[i] = env_done
                        current_obs[i] = observation[0]
                        event = {
                            "slot": slots[i]["slot"], "sample": 0, "attempt": attempt, "turn": turn,
                            "prompt": prompts[j], "rendered_prompt": rendered_prompts[j],
                            "prompt_token_count": len(tokenized[j]["prompt_token_ids"]),
                            "generated_token_ids": generated_ids, "raw_output": raw,
                            "projected_action": action, "valid_action_tag": valid_tag,
                            "admissible_actions": admissible_before,
                            "projected_is_admissible": action in admissible_before,
                            "executed_action": action,
                            "observation_before": observation_before,
                            "observation": observation[0],
                            "won": won, "done": env_done,
                        }
                        recs[i]["trace"].append(event)
                        trace.write(json.dumps(event, ensure_ascii=False) + "\n")
                        trace.flush()
                for i, rec in enumerate(recs):
                    cumulative_won[i] = bool(cumulative_won[i] or rec["won"])
                    rec["cumulative_won"] = cumulative_won[i]
                    attempts_by_slot[i].append(rec)
        for i, slot in enumerate(slots):
            rows.append({"slot": slot["slot"], "worker_seed": slot["worker_seed"],
                         "game_file": slot["game_file"], "game_id": slot["game_id"],
                         "task_type": slot["task_type"], "sample": 0,
                         "attempts": attempts_by_slot[i]})
    finally:
        for env in envs:
            try:
                env.close()
            except Exception:
                pass

    result = {
        "schema_version": 1, "profile": PROFILE, "style": args.style,
        "prompt_provenance": _prompt_provenance(args.style),
        "run_id": args.run_id,
        "rank": args.rank, "world_size": args.world_size, "samples": args.samples,
        "base_model": str(Path(args.base).resolve()),
        "model_identity": _safe_model_identity(args.base),
        "decode": {"temperature": 0.7, "top_p": 1.0, "top_k": -1,
                   "repetition_penalty": 1.0, "max_new_tokens": MAX_RESPONSE_TOKENS,
                   "min_tokens": 5, "stop": None, "ignore_eos": False, "detokenize": False,
                   "logprobs": 0, "engine_seed": 0, "prompt_tokens_max": MAX_PROMPT_TOKENS},
        "engine": upstream_engine_kwargs(),
        "selection_manifest_sha256": sha256_json(selection),
        "selection_local_checks": local_selection_checks,
        "selection_manifest": str(args.selection_manifest.resolve()),
        "rows": rows, "runtime_seconds": time.time() - start_time,
        "stage_seconds": {"generation": generation_seconds, "environment": environment_seconds},
        "generation_parity": parity_report,
        "software": _software_versions(),
        "tokenizer": {
            "name_or_path": getattr(tok, "name_or_path", None),
            "chat_template_sha256": sha256_bytes(str(getattr(tok, "chat_template", "")).encode("utf-8")),
            "enable_thinking": False,
            "eos_token_id": getattr(tok, "eos_token_id", None),
            "bos_token_id": getattr(tok, "bos_token_id", None),
            "pad_token_id": getattr(tok, "pad_token_id", None),
            "model_max_length": getattr(tok, "model_max_length", None),
        },
        "upstream_sources": {
            name: _sha_file(UPSTREAM / name) for name in (
                "examples/skillrise_alfworld/grpo_alfworld_qwen3_4b.sh",
                "verl/trainer/config/ppo_trainer.yaml",
                "agent_system/environments/alfworld/prompt.py",
                "agent_system/environments/alfworld/projection.py",
                "agent_system/environments/alfworld/memory.py",
                "agent_system/multi_turn_rollout/skillrise_rollout_loop.py",
                "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py",
                "agent_system/environments/alfworld/alfworld/agents/environment/alfred_tw_env.py",
            )
        },
        "task_type_map": list(__import__("eval_protocol").TASK_TYPES),
        "glibc_compat": RUNTIME_COMPAT_INFO,
        "vllm_attention_backend": os.environ.get("VLLM_ATTENTION_BACKEND", "default"),
        "diagnostics": {
            "valid_tag_steps": sum(a["valid_action_tag_steps"] for r in rows for a in r["attempts"]),
            "action_steps": sum(a["steps"] for r in rows for a in r["attempts"]),
            "note": "Malformed generation follows upstream projection; no look or other fallback is substituted.",
        },
    }
    outfile = args.out_dir / ("preflight.json" if args.preflight else f"rank-{args.rank:02d}.json")
    _write_json(outfile, result)
    print(json.dumps({k: result[k] for k in ("profile", "style", "rank", "diagnostics", "runtime_seconds")}, ensure_ascii=False), flush=True)
    return result


def _software_versions():
    versions = {}
    for package in ("vllm", "transformers", "torch", "alfworld", "textworld", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def _safe_model_identity(model_path):
    try:
        sys.path.insert(0, str(SELFPLAY / "src"))
        return __import__("ez_model_identity").inspect_model_identity(model_path)
    except Exception as exc:
        return {"status": "metadata_unavailable", "weights_verified": False,
                "requested_path": str(model_path),
                "reason": f"metadata inspection failed: {type(exc).__name__}: {exc}"}


def _prompt_provenance(style):
    if style == "react":
        return {"paper_exact": False, "kind": "upstream prompt",
                "source": "vendored SkillRise ALFWORLD_PLAY_PROMPT at the recorded source commit"}
    return {"paper_exact": False, "kind": "user-approved derived approximation",
            "source": "upstream play prompt with its two reasoning instructions removed"}


def merge_rank_outputs(out_dir: Path, selection_path: Path):
    with selection_path.open(encoding="utf-8") as f:
        selection = json.load(f)
    shards = []
    for rank in range(SHARD_COUNT):
        path = out_dir / f"rank-{rank:02d}.json"
        if not path.is_file():
            raise ProtocolError(f"Missing rank shard: {path}")
        with path.open(encoding="utf-8") as f:
            shards.append(json.load(f))
    rows = validate_shard_grid(shards, selection)
    merged = {
        "schema_version": 1, "profile": PROFILE, "source_commit": selection["source_commit"],
        "protocol_status": ("code-aligned upstream ReAct prompt; paper prompt/split/instance list undisclosed"
                            if shards[0]["style"] == "react" else
                            "user-approved derived prompt approximation; not an author-published baseline"),
        "split": selection["split"], "selection_manifest_sha256": sha256_json(selection),
        "run_id": shards[0]["run_id"], "style": shards[0]["style"],
        "prompt_provenance": shards[0]["prompt_provenance"],
        "base_model": shards[0]["base_model"], "model_identity": shards[0]["model_identity"],
        "decode": shards[0]["decode"], "engine": shards[0]["engine"],
        "tokenizer": shards[0]["tokenizer"], "software": shards[0]["software"],
        "upstream_sources": shards[0]["upstream_sources"],
        "vllm_attention_backend": shards[0]["vllm_attention_backend"],
        "glibc_compat": shards[0]["glibc_compat"],
        "n_slots": len(rows), "samples": 1, "metrics": aggregate_rows(rows), "rows": rows,
    }
    _write_json(out_dir / "merged.json", merged)
    return merged


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--style", choices=("react", "zero_shot"), default="react")
    p.add_argument("--base", default=os.environ.get("ZTWANG_MODEL_PATH", ""))
    p.add_argument("--selection-manifest", type=Path)
    p.add_argument("--prepare-selection", type=Path)
    p.add_argument("--out-dir", type=Path, default=HERE / "out")
    p.add_argument("--run-id", default=os.environ.get("SLURM_JOB_ID", "local"))
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=SHARD_COUNT)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--verify-preflight", type=Path,
                   help="Validate preflight trace completeness and print tag-rate diagnosis.")
    p.add_argument("--merge", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.environ["HF_HUB_OFFLINE"] = "1"
    if args.verify_preflight:
        with args.verify_preflight.open(encoding="utf-8") as f:
            report = verify_preflight(json.load(f))
        print(json.dumps(report, ensure_ascii=False))
        return 0
    if args.prepare_selection:
        selection = prepare_selection(args.prepare_selection)
        print(f"Wrote selection manifest with {len(selection['slots'])} slots")
        return 0
    if args.merge:
        if not args.selection_manifest:
            raise ProtocolError("--merge requires --selection-manifest")
        print(json.dumps(merge_rank_outputs(args.out_dir, args.selection_manifest)["metrics"], indent=2))
        return 0
    if not args.base:
        raise ProtocolError("--base or ZTWANG_MODEL_PATH is required")
    if not Path(args.base).expanduser().is_dir() or not (Path(args.base).expanduser() / "config.json").is_file():
        raise ProtocolError("--base must be an existing local model directory containing config.json")
    if args.world_size != SHARD_COUNT or not 0 <= args.rank < args.world_size:
        raise ProtocolError("This profile requires eight ranks (0..7)")
    if args.selection_manifest is None:
        raise ProtocolError("Prepare and pass --selection-manifest before GPU allocation")
    with args.selection_manifest.open(encoding="utf-8") as f:
        selection = json.load(f)
    run_rank(args, selection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
