"""Explorer-Zero solver：在 proposer 生成的任务上 rollout，用状态快照算 Eq.7 奖励。

与主线 `rollout.run_rollouts` 的差异（刻意不合并，保持主线稳定）：
1. solver 用 **附录 G Fig.14 的 prompt**（`<tool_call>` 协议，与 proposer 格式一致），
   看到的任务描述是 **proposer 生成的 x**，不是环境自带的官方任务；
2. 沿途每步记录 **状态快照**，最终奖励走 `ez_verify.verify_solver_rollout`
   （Eq.5 终态匹配 + Eq.6 子任务命中 + Eq.7 组合），**不用 env 的 won/reward**；
3. 轨迹的 `Trajectory.stats.won` 用 final_match 填（评测口径一致），
   `traj.reward` 填 Eq.7 奖励，可直接喂主线 GRPO 的组归一化。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ez_proposer import _slot_snapshot, parse_tool_call, sanitize_explore_action
from ez_prompts import append_tool_result, build_solver_messages
from ez_prompts import admissible_suffix, recover_bare_command
from ez_state import StateSnapshot, assert_facts_mode
from ez_verify import ProposedTask, SolverRolloutResult, verify_solver_rollout
from rollout import (RolloutConfig, Trajectory, _align_slot_tasks)
from reward import TrajectoryStats


@dataclass
class EzSolverTrajectory:
    """一条 solver 轨迹 = 主线 Trajectory + EZ 验证结果。"""

    traj: Trajectory
    verify: SolverRolloutResult
    snapshots: List[StateSnapshot]


def _adm_for(bs, s):
    """`EZ_SOLVER_ADMISSIBLE=1` 时才把合法动作表交给 append_tool_result。

    `append_tool_result` 本身**早就支持** inbound 这张表（非空就拼进去），
    但调用点一直没传 —— 一个典型"能力悬空"。默认关，保证行为逐字节不变。
    """
    import os
    if os.environ.get("EZ_SOLVER_ADMISSIBLE", "0") != "1":
        return ()
    try:
        return bs.admissible[s]
    except Exception:  # noqa: BLE001
        return ()


def run_ez_solver_rollouts(env, policy, tasks: Sequence[ProposedTask], cfg: RolloutConfig,
                           tokenize: bool = True,
                           log_fn: Optional[Callable[[str], None]] = None) -> List[EzSolverTrajectory]:
    """tasks: G 个（同一 ProposedTask 重复 G 次 = 一个 GRPO 组；调用方负责拼组）。

    `env.batch_size` 必须等于 len(tasks)（每个 slot 一条独立 rollout）。
    """
    n = len(tasks)
    if env.batch_size != n:
        raise ValueError(f"env.batch_size={env.batch_size} != n_tasks={n}")
    # ★ 对齐陷阱：_align_slot_tasks 只能按 gamefile 区分任务。若同一房子里有两个
    #   **不同** 的 proposer 任务（同 game_file、不同 task_desc），槽位可能张冠李戴。
    #   这里强制"同 game_file ⇒ 同 task_desc"（重复副本可互换），违反就抛给调用方拆批。
    seen: Dict[str, str] = {}
    for t in tasks:
        key = t.game_file
        if key in seen and seen[key] != t.task_desc:
            raise ValueError(
                "同一 game_file 上有多个不同 task_desc，slot 对齐会混淆；请拆成两批调用")
        seen[key] = t.task_desc

    games = [t.game_file for t in tasks]
    obs_list, views, admissible, used, _ = env.reset(games)
    # ★ textworld 会打乱 slot→game 的对应，必须反查（与主线 run_rollouts 同理）
    slot_task = _align_slot_tasks(games, used, group_size=1)
    facts0 = list(getattr(env, "last_reset_facts", [None] * n))
    snap_hist: List[List[StateSnapshot]] = []
    for s in range(n):
        _s0 = _slot_snapshot(facts0[s], obs_list[s])
        assert_facts_mode(_s0, f"solver reset slot {s}")
        from collections import Counter
        expected = tasks[slot_task[s]].s0
        if _s0.mode != expected.mode or Counter(_s0.facts) != Counter(expected.facts):
            raise RuntimeError("solver reset state differs from ProposedTask.s0; refusing mismatched rollout")
        snap_hist.append([_s0])

    messages: List[List[Dict[str, str]]] = []
    token_buffers: List[List[int]] = []
    token_masks: List[List[int]] = []
    stats: List[TrajectoryStats] = []
    gen_ids_log: List[List[List[int]]] = [[] for _ in range(n)]
    for s in range(n):
        t = tasks[slot_task[s]]
        # ★ EZ：附录 G Fig.14 solver prompt；观测里塞 proposer 任务 x
        v = views[s]
        msg = build_solver_messages(v.env_desc or v.raw, t.task_desc)
        messages.append(msg)
        # Solve-rate estimation and training must sample from identical token-native
        # histories. `tokenize` controls only whether the final trajectory is retained.
        token_buffers.append(policy.initial_prompt_ids(msg, cfg))
        token_masks.append([0] * len(token_buffers[-1]))
        stats.append(TrajectoryStats(task_type="ez_generated",
                                     task_id=t.meta.get("uid", str(slot_task[s]))))

    def append_feedback(slot: int, content: str) -> None:
        tail = policy.assistant_end_ids()
        user = policy.append_feedback_ids(content)
        prefix = policy.next_generation_prefix_ids(content)
        token_buffers[slot].extend(tail + user + prefix)
        token_masks[slot].extend([0] * (len(tail) + len(user) + len(prefix)))

    active = [True] * n
    last_action = [""] * n
    max_steps = min(cfg.max_steps, env.max_episode_steps)
    t0 = time.time()

    for _step in range(max_steps):
        if not any(active):
            break
        idxs = [s for s in range(n) if active[s]]
        texts, gen_ids, _ = policy.generate_from_token_ids([token_buffers[s] for s in idxs], cfg)

        step_actions = ["look"] * n
        parsed_by_slot: Dict[int, Any] = {}
        for k, s in enumerate(idxs):
            parsed = parse_tool_call(texts[k])
            if parsed.kind == "action":
                # EZ's default policy is raw: preserve invalid commands as generated.
                executed, exact = parsed.action.strip(), parsed.action.strip().lower() in {
                    a.strip().lower() for a in admissible[s]}
            elif parsed.kind == "malformed":
                # ★ 先尝试"抢救自然命令"：ALFWorld 原生动作空间就是自然命令字符串
                #   （`"go to desk 1"`），模型很可能压根不吐 JSON。实测基座 malformed
                #   占比 53%，其中"非 JSON 开头"占 3%；这条只在能对上当前 admissible
                #   白名单时才生效，所以**不可能**执行非法动作。默认关（EZ_SOLVER_BARE_FALLBACK=1 打开）。
                _rec = recover_bare_command(texts[k], admissible[s])
                if _rec:
                    executed = _rec
                    exact = _rec.strip().lower() in {
                        a.strip().lower() for a in admissible[s]}
                else:
                    executed, exact = "look", False     # 格式违例：空转 + 计数
            else:
                executed, exact = "look", True      # sub_finish/finish 对 solver 无意义：空转
            step_actions[s] = executed
            parsed_by_slot[s] = (parsed, executed, exact, texts[k], gen_ids[k])
            # Every generated response is part of the policy trajectory, even if malformed
            # or a control action. Keep its real ids so subsequent feedback has context.
            messages[s].append({"role": "assistant", "content": texts[k].strip()})
            gen_ids_log[s].append(list(gen_ids[k]))
            token_buffers[s].extend(gen_ids[k])
            token_masks[s].extend([1] * len(gen_ids[k]))

        bs = env.step(step_actions)
        cur_facts = list(bs.facts) if bs.facts else [None] * n
        for s in idxs:
            _sn = _slot_snapshot(cur_facts[s], bs.obs[s])
            assert_facts_mode(_sn, f"solver step slot {s}")
            snap_hist[s].append(_sn)

        for s in idxs:
            parsed, executed, exact, text, gid = parsed_by_slot[s]
            st = stats[s]
            st.n_steps += 1
            if parsed.kind == "malformed" and executed == "look":
                st.n_format_violation += 1
                messages[s].append({"role": "user", "content":
                    "Your last response was not a valid tool call. Emit exactly one "
                    "<tool_call>{...}</tool_call> block."
                    + admissible_suffix(adm[s])})
                append_feedback(s, messages[s][-1]["content"])
                if bool(bs.dones[s]):
                    active[s] = False
                continue
            if parsed.kind in ("sub_finish", "finish"):
                # solver 协议里没有这两个声明：当作普通空转，只轻微计数
                st.n_invalid += 0
                messages[s].append({"role": "user", "content":
                    "Only environment actions are valid for you. Keep working on the task."
                    + admissible_suffix(adm[s])})
                append_feedback(s, messages[s][-1]["content"])
                if bool(bs.dones[s]):
                    active[s] = False
                continue
            if exact:
                st.n_valid_steps += 1
            else:
                st.n_invalid += 1
            if executed and executed.lower() == last_action[s].lower():
                st.n_repeat += 1
            last_action[s] = executed
            if bool(bs.dones[s]):
                active[s] = False
            else:
                append_tool_result(messages[s], executed, bs.obs[s],
                                   _adm_for(bs, s))
                append_feedback(s, messages[s][-1]["content"])
        for s in idxs:
            if active[s]:
                admissible[s] = bs.admissible[s]

    out: List[EzSolverTrajectory] = []
    for s in range(n):
        t = tasks[slot_task[s]]
        st = stats[s]
        if active[s]:
            st.truncated = True
        verify = verify_solver_rollout(t, snap_hist[s])
        st.won = verify.final_match
        if tokenize:
            ids, mask = token_buffers[s], token_masks[s]
        else:
            ids, mask = [], []
        traj = Trajectory(task_id=st.task_id, task_type=st.task_type, game_file=used[s],
                          messages=messages[s], input_ids=ids, loss_mask=mask, stats=st,
                          n_turns=len(gen_ids_log[s]), n_tokens=len(ids))
        traj.reward = verify.reward     # ★ Eq.7，不是 compute_reward
        out.append(EzSolverTrajectory(traj=traj, verify=verify, snapshots=snap_hist[s]))

    if log_fn:
        rw = [e.traj.reward for e in out]
        log_fn(f"[ez-solver] {n} rollouts | reward mean={np.mean(rw):.3f} | "
               f"final_match={sum(e.verify.final_match for e in out)}/{n} | "
               f"subtask_rate={np.mean([e.verify.subtask_rate for e in out]):.3f} | "
               f"{time.time()-t0:.1f}s")
    return out
