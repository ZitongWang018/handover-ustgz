#!/usr/bin/env python
"""P1 验收门 #3（真实环境）：用**真实 ALFWorld 环境 + 真实 prompt** 跑两个相位的
内循环，HF 采样 vs vLLM 采样对比。

为什么必须用真实环境而不是造 prompt（教训，本脚本第一版踩过）：
    造出来的 prompt（重复 filler）是分布外的，4B base 会在 ~18 个 token 处直接收尾，
    于是两条路径都快得没意义，测出来的"加速比"全是假的。
    真实分布的观测 + 真实系统提示词才会走到 30 回合 × 上百 token 的真实路径。

测的东西：
    phase=proposer —— 调 `run_proposer_exploration`（**主瓶颈**，实测占单步 ~50%）
    phase=solver   —— 复刻 solve rate 的内循环（G_P 条轨迹 × 30 回合）
    两侧共用同一份 `build_*_messages` / `parse_tool_call` / env，
    因此耗时差 100% 来自采样实现。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from alfworld_compat import install as install_alfworld_compat  # noqa: E402

install_alfworld_compat()

from alfworld_env import ALFWorldBatchEnv, collect_game_files, load_alfworld_config  # noqa: E402
from ez_proposer import ProposerConfig, _slot_snapshot, parse_tool_call, run_proposer_exploration  # noqa: E402
from ez_prompts import append_tool_result, build_solver_messages  # noqa: E402
from rollout import HFPolicy, RolloutConfig  # noqa: E402

MAX_TURNS = 30


def make_env(alf_cfg, batch, max_steps):
    return ALFWorldBatchEnv(alf_cfg, split="train", batch_size=batch,
                            max_episode_steps=max_steps, with_facts=True,
                            ignore_benchmark_done=True)


def phase_proposer(policy, alf_cfg, games_n: int, g_total: int, log) -> dict:
    from ez_proposer import run_proposer_exploration as run_explore

    all_games = sorted(collect_game_files(alf_cfg, "train"))
    sel = all_games[:games_n]
    g_local = g_total // games_n
    batch = g_local * games_n
    env = make_env(alf_cfg, batch, MAX_TURNS)
    p_cfg = ProposerConfig(group_size=g_local, max_turns=MAX_TURNS,
                           max_new_tokens=160, generation_max_batch_size=0)
    log(f"[real] proposer: games={games_n} x G_local={g_local} -> slots={batch}")
    t0 = time.time()
    grouped = run_explore(env, policy, sel, p_cfg, log_fn=log)
    wall = time.time() - t0
    rolls = [r for grp in grouped for r in grp]
    turns = sum(getattr(r, "n_turns", 0) for r in rolls)
    return {"phase": "proposer", "wall_s": round(wall, 1), "slots": batch,
            "rollouts": len(rolls), "total_turns": turns,
            "accept": sum(1 for r in rolls if getattr(r, "accepted", False))}


def phase_solver(policy, alf_cfg, slots: int, log) -> dict:
    """复刻 solve rate 内循环：G_P 条轨迹同任务、逐回合 generate→step→回填。"""
    from ez_verify import ProposedTask

    all_games = sorted(collect_game_files(alf_cfg, "train"))
    game = all_games[0]
    env = make_env(alf_cfg, slots, MAX_TURNS)
    obs_list, views, _admissible, used, _e = env.reset([game] * slots)
    facts0 = list(getattr(env, "last_reset_facts", [None] * slots))
    s0 = _slot_snapshot(facts0[0], obs_list[0])
    task = ProposedTask(task_desc="cool a hot apple and put it in the fridge",
                        s0=s0, ref_final=s0, ref_final_delta=None,
                        subtasks=[], game_file=used[0])
    from ez_state import StateDelta
    task.ref_final_delta = StateDelta()

    cfg = RolloutConfig(group_size=1, max_steps=MAX_TURNS, max_new_tokens=96,
                        temperature=1.0, top_p=0.95, top_k=-1, min_p=0.0,
                        repetition_penalty=1.05, max_context_tokens=40960,
                        keep_recent_turns=12, stop_mode="tool_call", max_batch_size=0)
    msgs = [build_solver_messages((views[s].env_desc or views[s].raw), task.task_desc)
            for s in range(slots)]
    bufs = [policy.initial_prompt_ids(m, cfg) for m in msgs]
    adm = list(_admissible)
    log(f"[real] solver: slots={slots} game={os.path.basename(game)}")
    t0 = time.time()
    n_turns = 0
    active = [True] * slots
    for _step in range(MAX_TURNS):
        if not any(active):
            break
        idxs = [s for s in range(slots) if active[s]]
        texts, gen, _ = policy.generate_from_token_ids([bufs[s] for s in idxs], cfg)
        n_turns += 1
        acts = ["look"] * slots
        for k, s in enumerate(idxs):
            p = parse_tool_call(texts[k])
            acts[s] = p.action.strip() if p.kind == "action" else "look"
        bs = env.step(acts)
        for k, s in enumerate(idxs):
            bufs[s].extend(gen[k])
            tail = policy.assistant_end_ids()
            fb = append_tool_result([], acts[s], bs.obs[s])
            content = fb[0]["content"] if fb else "observation unavailable"
            bufs[s].extend(tail + policy.append_feedback_ids(content)
                           + policy.next_generation_prefix_ids(content))
            if bool(bs.dones[s]):
                active[s] = False
    wall = time.time() - t0
    return {"phase": "solver", "wall_s": round(wall, 1), "slots": slots,
            "rollouts": slots, "total_turns": n_turns * max(1, slots)}


def phase_envonly(alf_cfg, slots: int, log) -> dict:
    """只跑环境：16 个 slot × 30 回合 `env.step`，**不生成**。

    为什么要单独量这个（本轮最重要的一次对照）：
        生产里 8 个 rank 各自开 env_p(16) + env_s(16) + env_sr(8) ≈ 40 个 textworld
        子进程，全节点 = 320 个进程抢 64 核。所以"vLLM 把生成变快"之后，
        **剩下的时间几乎全是环境**。不量这个数就无法判断 8 卡生产能不能压进 10h。
        同一脚本在 sbatch 里单跑 1 份、再并行 8 份跑，就能直接读出超订倍数。
    """
    all_games = sorted(collect_game_files(alf_cfg, "train"))
    sel = all_games[:8]
    env = make_env(alf_cfg, slots, MAX_TURNS)
    log(f"[real] envonly: slots={slots} games={len(sel)}")
    t0 = time.time()
    t_reset = 0.0
    t_step = 0.0
    parse_s = 0.0
    turns = 0
    for _rep in range(3):                      # 3 局，降低单次噪声
        _r0 = time.time()
        env.reset([g for g in sel for _ in range(slots // len(sel))])
        t_reset += time.time() - _r0
        for _t in range(MAX_TURNS):
            _s0 = time.time()
            acts = ["look"] * slots
            env.step(acts)
            t_step += time.time() - _s0
            turns += slots
            parse_s += 0.0
    wall = time.time() - t0
    return {"phase": "envonly", "wall_s": round(wall, 2), "slots": slots,
            "total_turns": turns, "reset_s": round(t_reset, 2),
            "step_s": round(t_step, 2),
            "step_ms_per_slot": round(1000.0 * t_step / max(turns, 1), 3)}


def build_policy(mode, base, tok, log, gpu_mem, max_num_seqs):
    if mode == "hf":
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            base, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
            device_map="cuda:0", trust_remote_code=True)
        lc = LoraConfig(r=32, lora_alpha=32, lora_dropout=0.0, bias="none",
                        target_modules="all-linear", task_type="CAUSAL_LM")
        model = get_peft_model(model, lc, adapter_name="solver")
        model.eval()
        return HFPolicy(model, tok)
    if mode == "vllm":
        import gc

        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        from ez_vllm_policy import VLLMPolicy, build_vllm_engine, export_lora_adapter

        cpu = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16,
                                                   trust_remote_code=True)
        lc = LoraConfig(r=32, lora_alpha=32, lora_dropout=0.0, bias="none",
                        target_modules="all-linear", task_type="CAUSAL_LM")
        cpu = get_peft_model(cpu, lc, adapter_name="solver")
        adir = export_lora_adapter(cpu, "solver",
                                   os.path.abspath("vllm_lora_bench/solver"))
        del cpu
        gc.collect()
        eng = build_vllm_engine(base, gpu_memory_utilization=gpu_mem,
                                max_num_seqs=max_num_seqs)
        pol = VLLMPolicy(eng, tok, log_fn=log)
        pol.set_lora("solver", adir)
        return pol
    raise SystemExit(f"bad mode {mode}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hf", "vllm", "envonly"], required=True)
    ap.add_argument("--phase", default="proposer,solver")
    ap.add_argument("--base", default=os.environ.get("ZTWANG_MODEL_PATH"))
    ap.add_argument("--games", type=int, default=8, help="proposer 相位 game 数")
    ap.add_argument("--g-total", type=int, default=16)
    ap.add_argument("--sr-slots", type=int, default=8, help="solve rate 的 G_P")
    ap.add_argument("--gpu-mem", type=float, default=0.55)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()
    if not args.base:
        raise SystemExit("set ZTWANG_MODEL_PATH or --base")

    log = lambda m: print(m, flush=True)  # noqa: E731

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    proj = SRC.parent
    alf_cfg = load_alfworld_config(str(proj / "configs" / "base_config.yaml"))

    if args.mode == "envonly":
        out = [phase_envonly(alf_cfg, args.games * (args.g_total // args.games), log)]
        out[0]["turns_per_s"] = round(out[0]["total_turns"] / max(out[0]["wall_s"], 1e-9), 2)
        log("[real] " + json.dumps(out[0], ensure_ascii=False))
        if args.json_out:
            with open(args.json_out, "w") as fh:
                json.dump(out, fh, ensure_ascii=False, indent=2)
        return 0

    policy = build_policy(args.mode, args.base, tok, log, args.gpu_mem, args.max_num_seqs)

    out = []
    want = [p.strip() for p in args.phase.split(",") if p.strip()]
    if "proposer" in want:
        out.append({"mode": args.mode, **phase_proposer(policy, alf_cfg, args.games,
                                                        args.g_total, log)})
    if "solver" in want:
        out.append({"mode": args.mode, **phase_solver(policy, alf_cfg, args.sr_slots, log)})

    for r in out:
        r["turns_per_s"] = round(r["total_turns"] / max(r["wall_s"], 1e-9), 2)
        log("[real] " + json.dumps(r, ensure_ascii=False))
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
