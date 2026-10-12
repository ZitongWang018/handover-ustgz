#!/usr/bin/env python
"""格式探针：回答「Qwen3-4B-Instruct-2507 到底吐了什么」。

背景
----
冒烟训练实测（2026-10-12）：
    reward_mean=1.6  group_std_mean=0.0  adv_abs_mean=0.0  loss=0.0
    eval/success_rate=0.0  eval/invalid_action_rate=1.0  eval/mean_steps=8.0
`invalid_action_rate=1.0` = 每一步动作都不是 admissible 里的合法命令。
奖励函数的问题已经修过（Bug#6 归一化），所以病根更靠前：**模型没吐出可解析的动作**。
而所有日志里 `grep -c "Action:"` 全是 0 —— 我们从来没真看过模型输出。

本脚本就把原始生成文本拿出来看。

要回答的 4 个是非题
------------------
1. 输出里有没有 `Action:` 这一行？后面原文是什么（逐字符 repr）？
2. 生成 token 数是否正好 = max_new_tokens（=被截断）？64 vs 256 差多少？
3. 是不是空输出 / 立刻 EOS？
4. greedy 和采样差多少？（greedy 也 100% 非法 -> 不是采样参数的问题）

用法
----
    cd ~/ztwang/selfplay
    python env_setup/probe_policy_fmt.py 2>&1 | tee env_setup/results/probe_policy_fmt.log
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ / "src"))

import torch  # noqa: E402

from alfworld_compat import install as install_alfworld_compat  # noqa: E402
from alfworld_env import ALFWorldBatchEnv, load_alfworld_config  # noqa: E402
from prompts import (  # noqa: E402
    build_system_prompt,
    format_observation,
    match_admissible,
    parse_action,
)
from reward import make_reward_cfg  # noqa: E402
from rollout import HFPolicy, RolloutConfig, run_rollouts  # noqa: E402
from train_grpo import build_model_and_tokenizer, load_cfg  # noqa: E402

MODEL_PATH = os.environ.get("ZTWANG_MODEL_PATH", str(Path.home() / "thinktool/models/qwen3_4b_inst_2507"))


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def show_generation(tag: str, text: str, n_tok: int, max_new: int, admissible) -> None:
    """统一打印一组生成结果 + 解析结果。"""
    parsed = parse_action(text)
    matched = match_admissible(parsed.action, admissible)
    print(f"\n--- [{tag}] ---")
    print(f"n_tok={n_tok}  max_new_tokens={max_new}  截断={'是' if n_tok >= max_new else '否'}")
    print(f"raw repr : {text!r}")
    print(f"has_marker={parsed.has_marker}  thought={parsed.thought!r}")
    print(f"action   : {parsed.action!r}")
    if matched:
        print("匹配 admissible : OK -> " + repr(matched))
    else:
        print("匹配 admissible : 未命中")
    if not matched:
        # 给出最接近的一条，方便看出差在哪（大小写 / 前后缀 / 句点）
        import difflib
        close = difflib.get_close_matches(parsed.action.lower(), [c.lower() for c in admissible], n=2, cutoff=0.4)
        print(f"最接近的合法命令: {close}")
        print(f"admissible 样例 : {admissible[:6]}")


def main() -> int:
    t0 = time.time()
    out: dict = {}

    install_alfworld_compat(verbose=True)

    cfg = load_cfg(str(PROJ / "configs/train_smoke.yaml"))
    cfg["model"]["path"] = MODEL_PATH          # yaml 里是字面量 $ZTWANG_MODEL_PATH
    print(f"[probe] model = {MODEL_PATH}")

    # ---------- 环境：拿一条真实任务 ----------
    alf_cfg = load_alfworld_config(str(PROJ / "configs/base_config.yaml"))
    env = ALFWorldBatchEnv(alf_cfg, split="train", batch_size=1, seed=42)
    game = env.game_files[0]
    obs, views, admissible, used, _expert = env.reset([game])

    hr("1) 环境首帧")
    print(f"game     : {game}")
    print(f"used[0]  : {used[0]}")
    print(f"task_desc: {views[0].task_desc!r}")
    print(f"env_desc : {views[0].env_desc[:200]!r}")
    print(f"admissible ({len(admissible[0])} 条): {admissible[0][:10]}")
    print(f"max_episode_steps = {env.max_episode_steps}")
    out["n_admissible"] = len(admissible[0])
    out["admissible"] = admissible[0]

    # ---------- prompt ----------
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {
            "role": "user",
            "content": format_observation(
                views[0].env_desc, admissible[0], first_turn=True, task_desc=views[0].task_desc
            ),
        },
    ]

    hr("2) 模型加载")
    model, tokenizer = build_model_and_tokenizer(cfg, 0, 1)
    policy = HFPolicy(model, tokenizer)
    device = next(model.parameters()).device
    print(f"[probe] device = {device}")

    prompt = policy.render(messages, add_generation_prompt=True)
    n_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    hr(f"3) 渲染后的完整 prompt（{n_prompt} token）")
    print(repr(prompt))
    out["n_prompt_tokens"] = n_prompt
    out["prompt"] = prompt

    # ---------- A. 单条生成，5 组采样 ----------
    hr("4) A 段：单条生成 × 5 组采样")
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    settings = [
        ("A1 greedy  max_new=256", dict(do_sample=False, max_new_tokens=256)),
        ("A2 T=1.0   max_new=256", dict(do_sample=True, temperature=1.0, top_p=0.95, max_new_tokens=256)),
        ("A3 T=1.0   max_new=64 ", dict(do_sample=True, temperature=1.0, top_p=0.95, max_new_tokens=64)),
        ("A4 T=1.0 + rep_penalty=1.05", dict(do_sample=True, temperature=1.0, top_p=0.95,
                                             repetition_penalty=1.05, max_new_tokens=256)),
        ("A5 T=1.0 + top_k=20", dict(do_sample=True, temperature=1.0, top_p=0.95, top_k=20, max_new_tokens=256)),
    ]
    out["generations"] = {}
    for tag, kw in settings:
        with torch.no_grad():
            o = model.generate(**enc, pad_token_id=tokenizer.pad_token_id,
                               eos_token_id=tokenizer.eos_token_id, **kw)
        ids = o[0, enc["input_ids"].shape[1]:].tolist()
        eos = tokenizer.eos_token_id
        while ids and eos is not None and ids[-1] == eos:
            ids.pop()
        text = tokenizer.decode(ids, skip_special_tokens=True)
        show_generation(tag, text, len(ids), kw["max_new_tokens"], admissible[0])
        out["generations"][tag] = {
            "text": text, "n_tok": len(ids), "max_new": kw["max_new_tokens"],
            "action": parse_action(text).action,
            "has_marker": parse_action(text).has_marker,
            "matched": match_admissible(parse_action(text).action, admissible[0]),
        }

    # ---------- B. 真链路：monkeypatch HFPolicy.generate 抓每一步 ----------
    hr("5) B 段：真链路 rollout（1 task × 2，max_steps=8）")
    rcfg = RolloutConfig(
        group_size=2, max_steps=8, temperature=1.0, top_p=0.95,
        repetition_penalty=1.05, max_new_tokens=64, max_context_tokens=8192,
    )
    dumps = []
    orig_generate = HFPolicy.generate

    def spy_generate(self, messages_list, c):
        texts, gen_ids, stops = orig_generate(self, messages_list, c)
        dumps.append({
            "n": len(texts),
            "texts": [t for t in texts],
            "n_tok": [len(g) for g in gen_ids],
            "stops": list(stops),
        })
        print(f"    [gen] n={len(texts)} n_tok={[len(g) for g in gen_ids]} stops={list(stops)}")
        for t in texts:
            print(f"      raw={t!r}")
        return texts, gen_ids, stops

    HFPolicy.generate = spy_generate
    try:
        # ⚠️ run_rollouts 里硬断言 `env.batch_size == n_tasks * group_size`。
        #    A 段用的单槽 env（batch_size=1）过不了这条，必须另开一个 2 槽的。
        env2 = ALFWorldBatchEnv(alf_cfg, split="train", batch_size=2, seed=42)
        tasks = [{"game_file": game, "task_id": "probe", "task_type": "probe"}]
        trajs = run_rollouts(env2, policy, tasks, rcfg, make_reward_cfg(cfg.get("reward")),
                             log_fn=lambda m: print("  " + m), tokenize=False)
    finally:
        HFPolicy.generate = orig_generate

    hr("6) B 段轨迹统计")
    for i, t in enumerate(trajs):
        s = t.stats
        print(f"traj{i}: n_steps={s.n_steps} n_valid={s.n_valid_steps} n_invalid={s.n_invalid} "
              f"n_fmt={s.n_format_violation} n_rep={s.n_repeat} won={s.won} trunc={s.truncated} "
              f"reward={t.reward:.4f}")
    print(f"dumps: {len(dumps)} 次 generate，共 {sum(d['n'] for d in dumps)} 条生成")
    out["trajs"] = [
        {"n_steps": t.stats.n_steps, "n_valid": t.stats.n_valid_steps,
         "n_invalid": t.stats.n_invalid, "n_fmt": t.stats.n_format_violation,
         "won": t.stats.won, "reward": t.reward} for t in trajs
    ]
    out["dumps"] = dumps

    # ---------- 结论 ----------
    hr("7) 结论")
    n_has_marker = sum(1 for v in out["generations"].values() if v["has_marker"])
    n_matched = sum(1 for v in out["generations"].values() if v["matched"])
    print(f"5 组采样里：出现 Action: 标记 {n_has_marker}/5，命中 admissible {n_matched}/5")
    tot_invalid = sum(t.stats.n_invalid for t in trajs)
    tot_valid = sum(t.stats.n_valid_steps for t in trajs)
    print(f"真链路 rollout：合法 {tot_valid} 步 / 非法 {tot_invalid} 步 -> "
          f"invalid_rate={tot_invalid / max(1, tot_invalid + tot_valid):.1%}")
    for tag, v in out["generations"].items():
        if v["n_tok"] >= v["max_new"]:
            print(f"  ⚠️ {tag} 被截断（n_tok={v['n_tok']} >= max_new={v['max_new']}）")
        if not v["text"].strip():
            print(f"  ⚠️ {tag} 输出为空（立刻 EOS）")
    out["summary"] = {
        "n_has_marker": n_has_marker, "n_matched": n_matched,
        "rollout_invalid_rate": tot_invalid / max(1, tot_invalid + tot_valid),
    }

    res_dir = HERE / "results"
    res_dir.mkdir(exist_ok=True)
    (res_dir / "probe_policy_fmt.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[probe] 全部耗时 {time.time() - t0:.1f}s -> {res_dir / 'probe_policy_fmt.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
