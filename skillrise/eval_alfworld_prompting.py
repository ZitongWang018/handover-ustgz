#!/usr/bin/env python
"""SkillRise ALFWorld 提示词评测（zero-shot / ReAct），纯推理，不涉及任何训练。

为什么是"独立评测器"而不是直接跑上游仓库
---------------------------------------------------------------------------
上游 `Within-yao/SkillRise` 的评测是**嵌在 VERL 训练器的 validation 环节**里的
（`verl.trainer.main_ppo` + `trainer.val_before_train=True`），要拉 Ray/FSDP 一整栈。
但 ReAct 与 zero-shot 这两种设置**只走 prompting，不训练**，所以只需要它的
**环境语义**，不需要训练框架。本文件把那段语义**逐字搬过来**：
  * prompt 模板           ← `agent_system/environments/alfworld/prompt.py::ALFWORLD_PLAY_PROMPT`
                            + `get_alfworld_prompt` / `parse_current_trajectory` / `parse_reflection`
  * 历史拼装              ← `memory.py::SimpleMemory.fetch` 的
                            `Action {n}: {act}\nObservation {n}: {obs}`，最近 15 步
  * 动作解析              ← `projection.py::alfworld_projection(phase='play')` 逐字
  * admissible 的写法     ← `env_manager.py::build_text_obs`：`',\n '.join(f"'{s}'"...)`，剔除 'help'
推理换成我们自己的 vLLM（用户要求）；其余配置与上游脚本逐项对齐。

两种设置的唯一差别（见 ztwang/skillrise/DECISION_zero_shot_vs_react.md）：
  * `react`     ：仓库原样 prompt（"先 step-by-step reasoning，再在 <action> 里给动作"）
  * `zero_shot` ：同一 prompt **删掉那两条推理要求**，要求直接给 <action>
其余（数据 split、步数、解码、解析、admissible 写法）**完全相同**。

用法：
    python skillrise/eval_alfworld_prompting.py --style react     --split eval_in_distribution --n 8
    python skillrise/eval_alfworld_prompting.py --style zero_shot --split eval_in_distribution --n 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

# ── 上游 prompt 模板：逐字抄自 SkillRise/agent_system/environments/alfworld/prompt.py ──
ALFWORLD_PLAY_PROMPT = """
You are an expert agent operating in the ALFRED Embodied Environment.
{init_observation}{past_trajectories_reflections}{current_trajectory}

Your admissible actions of the current situation are: 
[{admissible_actions}]

Now it's your turn to take an action.

- Your response should first by step-by-step reasoning about the current situation.
- Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

#: zero-shot 变体：**只**删掉上面最后两条"推理要求"，其余逐字不动
ZERO_SHOT_TAIL = """
Now it's your turn to take an action.

- Your response should choose an admissible action for the current step and present it within <action> </action> tags.
"""

CURR_TRAJ_AT_TRAJ1 = '''
Below are the actions you took and the corresponding observations: 
{current_trajectory}'''


def build_play_prompt(style, init_observation, curr_traj, admissible, turn_idx):
    """对齐 get_alfworld_prompt(phase='play')，traj_idx 恒为 0（不做多轮尝试）。"""
    current_trajectory = "" if turn_idx == 0 else CURR_TRAJ_AT_TRAJ1.format(
        current_trajectory=curr_traj)
    adm = ",\n ".join(f"'{s}'" for s in admissible if s != "help")
    body = ALFWORLD_PLAY_PROMPT.format(
        init_observation=init_observation,
        past_trajectories_reflections="",
        current_trajectory=current_trajectory,
        admissible_actions=adm,
    )
    if style == "zero_shot":
        head = body[:body.index("\nNow it's your turn to take an action.")]
        body = head + ZERO_SHOT_TAIL
    return body.strip()


def format_history(records, history_length=15):
    """逐字对齐 memory.py::SimpleMemory.fetch 的拼装。"""
    lines = []
    recent = records[-history_length:]
    start = len(records) - len(recent)
    for j, (act, obs) in enumerate(recent):
        n = start + j + 1
        lines.append(f"Action {n}: {act}\nObservation {n}: {obs}")
    return "\n".join(lines)


def alfworld_projection_play(text):
    """逐字对齐 projection.py::alfworld_projection(phase='play')。

    返回 (action, valid)。**注意上游语义**：找不到 `<action>` 标签时，
    动作是"整段输出的**最后 30 个字符**"（会当非法动作发给环境），valid=0。
    """
    s = str(text).lower()
    start_tag, end_tag = "<action>", "</action>"
    si, ei = s.find(start_tag), s.find(end_tag)
    if si == -1 or ei == -1:
        return s[-30:], 0
    return s[si + len(start_tag):ei].strip().lower(), 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", choices=["react", "zero_shot"], required=True)
    ap.add_argument("--split", default="eval_in_distribution",
                    choices=["eval_in_distribution", "eval_out_of_distribution"])
    ap.add_argument("--base", default=os.environ.get("ZTWANG_MODEL_PATH", ""))
    ap.add_argument("--n", type=int, default=0, help="0 = 全量")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=30, help="上游 env.max_steps=30")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7, help="上游 val_kwargs.temperature=0.7")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    proj = Path(__file__).resolve().parent.parent          # <repo>/skillrise -> <repo>
    selfplay = proj / "selfplay"
    sys.path.insert(0, str(selfplay / "src"))
    sys.path.insert(0, str(selfplay / "third_party" / "py310deps"))
    from alfworld_env import (ALFWorldBatchEnv, collect_game_files,
                              load_alfworld_config)
    os.environ.setdefault("ALFWORLD_DATA", str(Path.home() / ".cache" / "alfworld"))

    cfg = load_alfworld_config(str(selfplay / "configs" / "base_config.yaml"))
    # ★ 不要 sorted()！上游 AlfredTWEnv 的 `env.game_files` **不是有序的**
    #   （实测 `gf == sorted(gf)` 为 False），且 eval 是按它自己的顺序取
    #   `val_batch_size`(=128) 个 slot。我原来加 sorted 就把顺序改掉了。
    games = list(collect_game_files(cfg, args.split))
    if args.n > 0:
        games = games[:args.n]
    print("[skillrise-eval] game_files 原序前 3: "
          + str([g.split("/")[-3][:32] for g in games[:3]]), flush=True)
    print(f"[skillrise-eval] style={args.style} split={args.split} "
          f"games={len(games)} samples={args.samples} max_steps={args.max_steps}", flush=True)

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    llm = LLM(model=args.base, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem,
              max_model_len=32768, enable_prefix_caching=True, seed=args.seed)
    sp = SamplingParams(n=1, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_new_tokens,
                        stop=["</action>"], include_stop_str_in_output=True)
    print(f"[skillrise-eval] 实际解码: {sp}", flush=True)

    rows = []
    for s_i in range(args.samples):
        # 每个 sample 内按 batch 跑；batch 内槽位与 used 对齐由 env 负责
        for start in range(0, len(games), args.batch):
            batch = games[start:start + args.batch]
            env = ALFWorldBatchEnv(cfg, split=args.split, batch_size=len(batch),
                                   max_episode_steps=args.max_steps,
                                   with_facts=False, ignore_benchmark_done=False)
            obs, views, admissible, used, _ = env.reset(batch)
            init_obs = [(views[i].raw or "").strip() for i in range(len(used))]
            hist = [[] for _ in used]
            adm = [list(admissible[i]) if admissible else [] for i in range(len(used))]
            done = [False] * len(used)
            won = [False] * len(used)
            nstep = [0] * len(used)
            nvalid = [0] * len(used)
            nact = [0] * len(used)
            for t in range(args.max_steps):
                idxs = [i for i in range(len(used)) if not done[i]]
                if not idxs:
                    break
                prompts = [build_play_prompt(args.style, init_obs[i],
                                             format_history(hist[i]), adm[i], t)
                           for i in idxs]
                outs = llm.generate(prompts, sp, use_tqdm=False)
                actions = ["look"] * len(used)
                for j, slot in enumerate(idxs):
                    raw = outs[j].outputs[0].text
                    acts, val = alfworld_projection_play(raw)
                    nvalid[slot] += val
                    nact[slot] += 1
                    actions[slot] = acts if acts else "look"
                bs = env.step(actions)
                for slot in idxs:
                    won[slot] = bool(bs.won[slot]) if bs.won else False
                    if bs.admissible:
                        adm[slot] = bs.admissible[slot]
                    hist[slot].append((actions[slot], bs.obs[slot]))
                    nstep[slot] += 1
                    if bs.dones[slot]:
                        done[slot] = True
            for i in range(len(used)):
                rows.append({"game": used[i], "split": args.split, "style": args.style,
                             "sample": s_i, "won": won[i], "steps": nstep[i],
                             "n_actions": nact[i], "n_valid_tag": nvalid[i],
                             "hit_cap": nstep[i] >= args.max_steps and not won[i]})
            env.close()
            print(f"[skillrise-eval] sample={s_i} batch@{start} done, "
                  f"累计 {len(rows)} episode", flush=True)

    n = len(rows)
    nwin = sum(1 for r in rows if r["won"])
    by = Counter()
    for r in rows:
        t = Path(str(r["game"]).replace("\\", "/")).parent.parent.name.split("-")[0]
        by[t] += 1 if r["won"] else 0
    sr = nwin / n if n else 0.0
    summary = {"style": args.style, "split": args.split, "n_episodes": n,
               "n_won": nwin, "sr": sr,
               "tag_valid_rate": (sum(r["n_valid_tag"] for r in rows)
                                  / max(1, sum(r["n_actions"] for r in rows))),
               "hit_cap_rate": sum(1 for r in rows if r["hit_cap"]) / max(1, n),
               "per_type_wins": dict(by),
               "decode": {"temperature": args.temperature, "top_p": args.top_p,
                          "max_new_tokens": args.max_new_tokens,
                          "max_steps": args.max_steps, "samples": args.samples},
               "base": args.base}
    print("\n=== SkillRise prompting eval ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    out = args.out or f"skillrise_eval_{args.style}_{args.split}.json"
    json.dump({"summary": summary, "rows": rows}, open(out, "w"),
              ensure_ascii=False, indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
