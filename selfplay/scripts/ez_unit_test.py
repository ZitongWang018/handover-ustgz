#!/usr/bin/env python
"""Explorer-Zero 离线单测：状态匹配 / 奖励 / 过滤 / 工具解析。

纯离线：不需要 GPU、不需要 ALFWorld 数据、不需要模型。
运行：python scripts/ez_unit_test.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ez_proposer import parse_tool_call                     # noqa: E402
from ez_state import (StateDelta, StateSnapshot, delta_covers,  # noqa: E402
                      match_final_state, match_reference, normalize_fact,
                      snapshot_from_obs_text, state_delta)
from ez_verify import (ProposedTask, SubtaskRef, action_type,  # noqa: E402
                       proposer_reward, redundant_pair, validity_filter,
                       verify_solver_rollout)
from rollout import _ToolCallStopCriteria, scan_bare_json_call  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [ok] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {extra}")


def snap(facts) -> StateSnapshot:
    return StateSnapshot(facts=frozenset(normalize_fact(f) for f in facts), mode="facts")


print("== 实例归一化 ==")
check("apple 1 -> apple", normalize_fact("in(apple 1, fridge 1)") == "in(apple, fridge)")
check("apple 1 与 apple 2 归一后相同",
      normalize_fact("on(apple 1, table 1)") == normalize_fact("on(apple 2, table 2)"))
check("多参数", normalize_fact("cooked(apple 1)") == "cooked(apple)")

print("== delta 与覆盖匹配 ==")
S0 = snap(["at(agent, kitchen)", "closed(fridge 1)", "in(apple 1, fridge 1)"])
S1 = snap(["at(agent, kitchen)", "open(fridge 1)", "in(apple 1, fridge 1)"])   # 开了冰箱
S2 = snap(["at(agent, kitchen)", "open(fridge 1)", "held(apple 1)"])          # 拿了苹果
S3 = snap(["at(agent, kitchen)", "open(fridge 1)", "held(apple 1)",
           "sliced(apple 1)"])                                                  # 多切了一刀

d_open = state_delta(S0, S1)
check("开冰箱 delta=removed closed", d_open.removed == frozenset({"closed(fridge)"}))
check("开冰箱 delta=added open", d_open.added == frozenset({"open(fridge)"}))
check("覆盖：S2 包含 open 变化", delta_covers(d_open, state_delta(S0, S2)))
check("不覆盖：S1 不含 take 变化",
      not delta_covers(state_delta(S0, S2), d_open))
check("容忍额外变化：S3 覆盖 S2 的 delta",
      delta_covers(state_delta(S0, S2), state_delta(S0, S3)))

print("== Eq.5/6/7 solver 奖励 ==")
# 终态 = 开着冰箱**且拿着苹果**（与 S1 仅开冰箱区分开，否则"部分完成"场景不成立）
ref_final = snap(["at(agent, kitchen)", "open(fridge 1)", "held(apple 1)"])
task = ProposedTask(
    task_desc="Put an apple in the fridge.",
    s0=S0, ref_final=ref_final,
    ref_final_delta=state_delta(S0, ref_final),
    subtasks=[SubtaskRef(description="Open the fridge.",
                         ref_snapshot=S1, ref_delta=state_delta(S0, S1))],
    proposer_actions=["open fridge 1", "take apple 1 from fridge 1"],
    proposer_action_types=["open", "take"],
)

# 全成功：终态命中 + 子任务命中 -> reward=1
full = verify_solver_rollout(task, [S0, S1, ref_final])
check("成功轨迹 reward=1", full.reward == 1.0 and full.final_match)
# 失败但有进展：开了冰箱没放进 -> reward=子任务率=1.0（唯一子任务完成）
partial = verify_solver_rollout(task, [S0, S1])
check("部分完成 reward=子任务率", (not partial.final_match) and partial.reward == 1.0
      and partial.subtask_hits == [True])
# 完全失败
none_ = verify_solver_rollout(task, [S0, S0])
check("完全失败 reward=0", none_.reward == 0.0 and not any(none_.subtask_hits))
# 无序清单：后完成子任务也算（历史状态命中）
reordered = verify_solver_rollout(task, [S0, ref_final, S1])
check("历史状态命中子任务", reordered.subtask_hits == [True])
# 实例替换：solver 用 apple 2（归一化后等价）
S1_b = snap(["at(agent, kitchen)", "open(fridge 1)", "in(apple 2, fridge 2)"])
S2_b = snap(["at(agent, kitchen)", "open(fridge 1)", "in(apple 2, fridge 2)"])
swap = verify_solver_rollout(task, [S0, S1_b, S2_b])
check("实例替换等价（apple 2 也算开冰箱）", swap.subtask_hits == [True])
# 无子任务任务：失败即 0
task_nosub = ProposedTask(task_desc="x", s0=S0, ref_final=ref_final,
                          ref_final_delta=state_delta(S0, ref_final))
check("无子任务失败=0", verify_solver_rollout(task_nosub, [S0, S1]).reward == 0.0)
check("无子任务成功=1", verify_solver_rollout(task_nosub, [S0, ref_final]).reward == 1.0)

print("== 冗余子任务判定（delta 同 + Jaccard>0.6）==")
d_a = state_delta(S0, S1)
check("完全相同描述+同delta -> 冗余",
      redundant_pair("Open the fridge.", d_a, "Open the fridge.", d_a))
check("不同delta 不冗余",
      not redundant_pair("Open the fridge.", d_a, "Open the fridge.", state_delta(S0, S2)))
check("delta 同但描述不同 -> 不冗余",
      not redundant_pair("Open the fridge.", d_a, "Unlock the cabinet.", d_a))

print("== 有效性过滤三规则 ==")
dup = ProposedTask(task_desc="Do two things.", s0=S0, ref_final=S2,
                   ref_final_delta=state_delta(S0, S2),
                   subtasks=[SubtaskRef("Open the fridge.", S1, d_a),
                             SubtaskRef("Open the fridge.", S1, d_a)],
                   proposer_actions=["open fridge 1"])
ok, reason = validity_filter(dup)
check("冗余子任务被拒", not ok and reason.startswith("redundant_subtask"), reason)
empty = ProposedTask(task_desc="", s0=S0, ref_final=S2,
                     ref_final_delta=state_delta(S0, S2))
ok, _ = validity_filter(empty)
check("空任务描述被拒", not ok)
# review P1-2：终态相对 s0 无变化（开局就 finish / 全程只 look）必须被拒，
# 否则 Eq.5 空 delta 匹配恒真 → solver 奖励恒 1、p̂ 恒 1，整条线退化。
nochange = ProposedTask(task_desc="Look around.", s0=S0, ref_final=S0,
                        ref_final_delta=state_delta(S0, S0),
                        proposer_actions=["look", "look"])
ok, reason = validity_filter(nochange)
check("空 delta 任务被拒", not ok and reason == "empty_final_delta", reason)
good = ProposedTask(task_desc="Open the fridge.", s0=S0, ref_final=S1,
                    ref_final_delta=state_delta(S0, S1),
                    proposer_actions=["open fridge 1"])
ok, reason = validity_filter(good)
check("正常任务通过", ok, reason)

print("== proposer 奖励 Eq.1-4 ==")
r, parts = proposer_reward(task, solve_rate=0.5)
check("p̂=0.5 难度=1", abs(parts["r_difficulty"] - 1.0) < 1e-9)
r2, parts2 = proposer_reward(task, solve_rate=0.0)
check("p̂=0 难度=0", abs(parts2["r_difficulty"] - 0.0) < 1e-9)
check("子任务项 = w_sub*K", abs(parts["r_subtask"] - 0.3 * 1) < 1e-9)
check("多样性项 = w_div*|动作类型|", abs(parts["r_diversity"] - 0.3 * 2) < 1e-9)
check("动作类型提取", action_type("open fridge 1") == "open")

print("== 工具调用解析 ==")
p = parse_tool_call('blah <tool_call>{"name":"alfworld_tool","arguments":{"action_type":"go_to","receptacle_name":"kitchen 1"}}</tool_call>')
check("go_to 解析", p.kind == "action" and p.action == "go to kitchen 1", p.raw)
p = parse_tool_call('<tool_call>{"name":"alfworld_tool","arguments":{"action_type":"heat","object_name":"apple 1","receptacle_name":"microwave 1"}}</tool_call>')
check("heat 解析", p.kind == "action" and p.action == "heat apple 1 with microwave 1", p.raw)
p = parse_tool_call('<tool_call>{"name":"alfworld_tool","arguments":{"action_type":"sub_finish","task_description":"Heat an apple."}}</tool_call>')
check("sub_finish 解析", p.kind == "sub_finish" and p.task_description == "Heat an apple.")
p = parse_tool_call('<tool_call>{"name":"alfworld_tool","arguments":{"action_type":"finish","task_description":"Heat an apple and put it on the countertop."}}</tool_call>')
check("finish 解析", p.kind == "finish" and "countertop" in p.task_description)
p = parse_tool_call('<tool_call>{"action_type":"look"}</tool_call>')
check("裸 JSON 兼容", p.kind == "action" and p.action == "look")
p = parse_tool_call("没有任何工具调用")
check("无工具调用 -> malformed", p.kind == "malformed")
p = parse_tool_call('<tool_call>{"name":"alfworld_tool","arguments":{"action_type":"fly"}}</tool_call>')
check("未知动作 -> malformed", p.kind == "malformed")
p = parse_tool_call('<tool_call>{"name":"alfworld_tool","arguments":{"action_type":"take","object_name":"apple 1"}}</tool_call>')
check("缺参数 -> malformed", p.kind == "malformed")

print("== 裸 JSON 工具调用（smoke 实测：Qwen3-4B-Instruct-2507 不吐 <tool_call> 标签）==")
# 现场真实输出的形态：先一行 reasoning，再裸 JSON，不带任何标签
_REAL = ('Go To: desk 1  \nReasoning: desk is a workspace.  \n\n'
         '{"name":"alfworld_tool","arguments":{"action_type":"go_to","receptacle_name":"desk 1"}}')
p = parse_tool_call(_REAL)
check("裸 JSON -> action", p.kind == "action" and p.action == "go to desk 1", p.raw)
check("裸 JSON -> end_idx 落在对象右括号后", 0 < p.end_idx <= len(_REAL.rstrip()), str(p.end_idx))

# 一轮连写多个动作（现场就是这个形态）：必须只取**第一个**，end_idx 指向第一个对象末尾
_MULTI = _REAL + ('\n\nOpen: drawer 1\n{"name":"alfworld_tool",'
                  '"arguments":{"action_type":"open","receptacle_name":"drawer 1"}}')
p = parse_tool_call(_MULTI)
check("连写多个动作 -> 只取第一个", p.kind == "action" and p.action == "go to desk 1", p.raw)
check("连写多个动作 -> end_idx 不含第二个", p.end_idx < len(_MULTI.rstrip()),
      f"end_idx={p.end_idx} len={len(_MULTI.rstrip())}")

# 字符串里带花括号：括号配平不能被字符串内的 { } 带偏
_NEST = '{"name":"alfworld_tool","arguments":{"action_type":"finish","task_description":"put an apple in {the} bowl"}}'
p = parse_tool_call(_NEST)
check("字符串内含花括号仍配平", p.kind == "finish" and "{the}" in p.task_description, p.raw)

# 工具调用前面有别的 JSON 对象 -> 必须跳过它继续找
p = parse_tool_call('prefix {"note":"hi"} then {"name":"alfworld_tool","arguments":{"action_type":"look"}}')
check("跳过不含 action_type 的 JSON", p.kind == "action" and p.action == "look", p.raw)

# 括号没配平（被 max_new_tokens 截断）-> malformed，不能瞎猜
p = parse_tool_call('{"name":"alfworld_tool","arguments":{"action_type":"go_to","receptacle')
check("截断的 JSON -> malformed", p.kind == "malformed")
check("纯文本 -> malformed", parse_tool_call("Go To: desk 1\nReasoning: blah").kind == "malformed")

print("== scan_bare_json_call 直接测 ==")
sp = scan_bare_json_call(_MULTI)
check("scan 返回 span", sp is not None and sp[1] <= len(_MULTI.rstrip()), str(sp))
check("scan 无对象 -> None", scan_bare_json_call("no json here") is None)
check("scan 忽略非工具 JSON", scan_bare_json_call('{"foo":1}') is None)
# 防回归：envelope 必须**整体**被识别，不能只吃掉内层 arguments 对象
_ENV = '{"name":"alfworld_tool","arguments":{"action_type":"go_to","receptacle_name":"desk 1"}}'
check("scan envelope 不吃内层", scan_bare_json_call(_ENV) == (0, len(_ENV)),
      str(scan_bare_json_call(_ENV)))
_LOOK = '{"name":"alfworld_tool","arguments":{"action_type":"look"}}'
check("scan 空 arguments 也可识别", scan_bare_json_call(_LOOK) == (0, len(_LOOK)),
      str(scan_bare_json_call(_LOOK)))

print("== tool_call 停止条件（EZ 专用）==")


class _FakeTok:
    """把 token id 直接当字符解码：id n -> 词表第 n 个字符。"""
    CHARS = ' abcdefghijklmnopqrstuvwxyz0123456789<>/{}_-":,'

    def __init__(self):
        self.chars = self.CHARS

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.chars[i] if 0 <= i < len(self.chars) else "?" for i in ids)


class _FakeTensor:
    """只实现 stopping criteria 用到的 shape / 二维切片。"""
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]) if rows else 0)

    def __getitem__(self, key):
        i, sl = key
        return self.rows[i][sl]


def _encode(tok: _FakeTok, text: str):
    return [tok.chars.index(c) for c in text]


tok = _FakeTok()
crit = _ToolCallStopCriteria(tok, [0], max_new_tokens=200)
partial = _encode(tok, "<tool_call>{a}")
done = _encode(tok, "<tool_call>{a}</tool_call>")
check("未闭合不停", crit(_FakeTensor([partial]), None) is False)
check("闭合后停", crit(_FakeTensor([done]), None) is True)
check("stop_len 记录", crit.stop_len(0, 999) == len(done))
crit2 = _ToolCallStopCriteria(tok, [0], max_new_tokens=200)
check("从没出现标签 -> 永不触发", crit2(_FakeTensor([partial]), None) is False
      and crit2.stop_len(0, 999) == 999)

# ★ 裸 JSON 停止（现场真实形态）：不吐标签，直接甩 JSON；停在第一个完整动作对象上
crit3 = _ToolCallStopCriteria(tok, [0], max_new_tokens=200)
_bare_open = _encode(tok, 'go to desk 1 {"name":"alfworld_tool","arguments":{"action_type":"go_to"')
_bare_done = _encode(
    tok, 'go to desk 1 {"name":"alfworld_tool","arguments":{"action_type":"go_to","receptacle_name":"desk 1"}}')
check("裸 JSON 未配平不停", crit3(_FakeTensor([_bare_open]), None) is False)
check("裸 JSON 配平后停", crit3(_FakeTensor([_bare_done]), None) is True)
check("裸 JSON stop_len 记录", crit3.stop_len(0, 999) == len(_bare_done))

# 停在**第一个**动作对象：后面还跟着第二个动作也必须立刻停。
# ⚠️ 必须**逐 token 增量喂入**——真实 generate 是每生成一个 token 判一次。
#    一次性喂整段文本的话 generated_len 恒等于全长，测不出"停在哪"。
crit4 = _ToolCallStopCriteria(tok, [0], max_new_tokens=200)
_FIRST = '{"name":"alfworld_tool","arguments":{"action_type":"look"}}'
_FULL = (_FIRST + ' then {"name":"alfworld_tool",'
                  '"arguments":{"action_type":"inventory"}}')
_first = _encode(tok, _FIRST)
_full = _encode(tok, _FULL)
_fired_at = None
for _n in range(1, len(_full) + 1):
    if crit4(_FakeTensor([_full[:_n]]), None):
        _fired_at = _n
        break
check("连写两个动作 -> 第一个配平即停", _fired_at == len(_first),
      f"fired_at={_fired_at} 期望={len(_first)}")
check("连写两个动作 -> 不跑到第二个动作", _fired_at is not None and _fired_at < len(_full),
      f"fired_at={_fired_at} full={len(_full)}")

# 非工具的 JSON 不该触发停止（避免把 reasoning 里的乱 JSON 当动作）
crit5 = _ToolCallStopCriteria(tok, [0], max_new_tokens=200)
check("非工具 JSON 不触发", crit5(_FakeTensor([_encode(tok, '{"foo":1}')]), None) is False)

print("== text fallback 降级模式 ==")
fa = snapshot_from_obs_text("You are in the kitchen.")
fb = snapshot_from_obs_text("You are in the kitchen.")
fc = snapshot_from_obs_text("You are in the bathroom.")
check("fallback 相同观测无变化", state_delta(fa, fb).is_empty())
check("fallback 不同观测有变化", not state_delta(fa, fc).is_empty())
check("fallback 匹配", match_reference(fa, fc, [fa, fc]))

print("== 多卡采样：跨 rank 不得重复（job 12989532 实测踩到的真 bug）==")
# ★ 背景：solver 相位旧写法是 `rng.sample(outer_pool, k)`，
#   而 rng 是 `random.Random(seed)` —— 各 rank 的 seed 完全相同
#   （`set_seed(seed, rank)` 里的 `seed + rank` 只是局部变量，改不到这里），
#   池子又经 all_gather 后完全一致 ⇒ 所有 rank 采到同一批任务。
#   实测 8 卡 k=1：rank1..rank7 全部采到同一个任务，全局只用到 2 个不同任务，
#   有效 batch 缩水到 25%。
#   修法两条：① rng 加 rank 偏移 ② 采样改成按 rank 分片（本例验证的是 ②）。
POOL = list(range(24))
for _world in (2, 4, 8):
    _k = max(1, 8 // _world)
    _picked = [tuple(POOL[(r + i * _world) % len(POOL)] for i in range(_k))
               for r in range(_world)]
    _all = [t for p in _picked for t in p]
    _uniq = len(set(_all))
    check(f"world={_world} 分片采样：{_uniq}/{len(_all)} 个不同任务（应==总数）",
          _uniq == len(_all), f"采到 {_picked}")
    check(f"world={_world} 各 rank 条数一致（红线 B）",
          len({len(p) for p in _picked}) == 1, f"{[len(p) for p in _picked]}")

# 对照：旧写法（rng 不带 rank 偏移）确实会采重 —— 把这个退化行为也钉住，
# 免得有人"优化"回 rng.sample 而没加 rank 偏移。
import random as _random                                     # noqa: E402

# ① 纯 rng（无人额外消耗）：8 个 rank 采到**完全相同**的一批
_old = []
for _r in range(8):
    _rng = _random.Random(42)
    _old.append(tuple(_rng.sample(POOL, k=1)))
check("旧写法(random.Random(42) 不分片)：8 个 rank 采到同一批（bug 复现）",
      len(set(_old)) == 1, f"{_old}")

# ② 真实情形：rank0 在 proposer 相位调过 rng.choice，所以只有它不同
_old2 = []
for _r in range(8):
    _rng = _random.Random(42)
    if _r == 0:
        for _ in range(8):                    # 复现行 558 的 `rng.choice(game_files)`
            _rng.choice(range(3553))
    _old2.append(tuple(_rng.sample(POOL, k=1)))
check("旧写法(rank0 消耗过 rng)：全局只有 2 个不同任务，有效 batch 25%",
      len(set(_old2)) == 2, f"{set(_old2)}")
check("旧写法退化了：rank1..7 全部撞车",
      len(set(_old2[1:])) == 1, f"{_old2[1:]}")

# ===========================================================================
# grpo_loss 的归约口径：token 级 vs 序列级（2026-10-12 交叉 review 抓到的真 bug）
#
# 为什么单独测：`grpo_loss` 在 train_grpo.py 里，而那个模块顶层要 import
# transformers / alfworld / rollout，离线环境根本 import 不进来。所以把归约
# 数学抽到 `src/grpo_norm.py`（只依赖 torch），这里测的就是**线上跑的那份代码**，
# 不是复刻实现。
#
# 判据不看 loss 的**值**（序列级下 Σadv=0 会让它恒为 0，看不出差别），
# 而是看**梯度** —— 梯度才是真正决定更新方向的东西。
# ===========================================================================
try:
    import torch  # noqa: E402
    from grpo_norm import grpo_reduce  # noqa: E402
    _HAS_TORCH = True
except ImportError as _e:                                    # pragma: no cover
    print(f"\n== grpo 归约口径 ==\n  ⚠️ 跳过（缺依赖：{_e}）")
    _HAS_TORCH = False

if _HAS_TORCH:
    print("\n== grpo 归约口径（token 级 vs 序列级）==")

    # G=2：一条短(20 token)答对，一条长(200 token)答错
    LEN = [20, 200]
    ADV = [1.0, -1.0]
    W = max(LEN)
    _mask = torch.zeros(2, W)
    for _i, _L in enumerate(LEN):
        _mask[_i, :_L] = 1.0

    def _grads(norm: str):
        """per_tok[i,t] = -adv_i * θ_i ⇒ ∂loss/∂θ_i 就是"序列 i 拿到的权重×(-adv_i)"。"""
        th = torch.zeros(2, requires_grad=True)
        adv = torch.tensor(ADV).unsqueeze(1)                 # (2,1)
        per_tok = -adv * th.unsqueeze(1)                     # (2,W)
        loss = grpo_reduce(per_tok, _mask, loss_norm=norm)
        loss.backward()
        return th.grad.clone(), float(loss.item())

    g_seq, v_seq = _grads("seq")
    g_tok, v_tok = _grads("token")

    # ① 序列级：两条序列等权，梯度幅度必须相等（与长度无关！）
    check("序列级：两条序列梯度幅度相等（等权，与长度无关）",
          abs(abs(g_seq[0]) - abs(g_seq[1])) < 1e-6, f"{g_seq.tolist()}")
    check("序列级：梯度幅度 == |adv|/G = 0.5",
          abs(abs(g_seq[0]) - 0.5) < 1e-6, f"{g_seq.tolist()}")

    # ② token 级：长度加权，200/20 = 10 倍
    _ratio = abs(g_tok[1]) / abs(g_tok[0])
    check(f"token 级：长度加权，长/短 = {_ratio:.1f}（应 == 200/20 = 10）",
          abs(_ratio - 10.0) < 1e-4, f"{g_tok.tolist()}")
    check("token 级确实放大了长序列（这就是 length bias）",
          abs(g_tok[1]) > abs(g_seq[1]) * 1.5, f"tok={g_tok.tolist()} seq={g_seq.tolist()}")

    # ③ 组内净优势：Σ adv = 0 ⇒ 序列级必须给 0 净信号，token 级不该给 0
    def _net(norm: str):
        s = torch.zeros(1, requires_grad=True)
        adv = torch.tensor(ADV).unsqueeze(1)
        per_tok = -adv * s                                   # 所有序列共用一个缩放
        loss = grpo_reduce(per_tok, _mask, loss_norm=norm)
        loss.backward()
        return float(s.grad[0].item())

    # ⚠️ 阈值用 1e-6 而不是 1e-9：float32 下 0.5-0.5 的累加误差就有 ~6e-8，
    #    拿 1e-9 卡只会在换机器上时随机红。
    check("序列级：组内奖励对称 ⇒ 净信号恒为 0（GRPO 应有性质）",
          abs(_net("seq")) < 1e-6, f"{_net('seq')}")
    check("token 级：同样的对称奖励却给出非零净信号（bug 复现）",
          abs(_net("token")) > 0.5, f"{_net('token')}")

    # ④ sanity：长度全相等时两种口径必须完全一致
    _meq = torch.ones(2, 50)
    _adv = torch.tensor(ADV).unsqueeze(1)
    _th1 = torch.zeros(2, requires_grad=True)
    _l1 = grpo_reduce(-_adv * _th1.unsqueeze(1), _meq, "seq")
    _th2 = torch.zeros(2, requires_grad=True)
    _l2 = grpo_reduce(-_adv * _th2.unsqueeze(1), _meq, "token")
    check("长度全相等时 seq == token（口径差异只来自长度）",
          abs(float(_l1.item()) - float(_l2.item())) < 1e-9,
          f"{_l1.item()} vs {_l2.item()}")

    # ⑤ mask 全 0 的行不能进平均（否则会稀释），也不能让 loss 断图
    _mz = torch.zeros(3, W)
    _mz[0, :10] = 1.0                                        # 只有第 0 行有效
    _th3 = torch.zeros(3, requires_grad=True)
    _l3 = grpo_reduce(-torch.tensor([1.0, -1.0, 1.0]).unsqueeze(1) * _th3.unsqueeze(1),
                      _mz, "seq")
    check("mask 全 0 的行有 grad_fn（不能断图，否则红线 B 死锁）",
          _l3.requires_grad and _l3.grad_fn is not None, str(_l3))
    _l3.backward()
    check("mask 全 0 的行拿不到梯度", float(_th3.grad[2].item()) == 0.0,
          f"{_th3.grad.tolist()}")
    check("有效行不受空行稀释（权重仍是 1/1 而非 1/3）",
          abs(abs(_th3.grad[0].item()) - 1.0) < 1e-6, f"{_th3.grad.tolist()}")

    # ⑥ 非法口径必须报错，不能悄悄走默认分支
    _bad = False
    try:
        grpo_reduce(torch.zeros(1, 4), torch.ones(1, 4), "oops")
    except ValueError:
        _bad = True
    check("非法 loss_norm 抛 ValueError", _bad)

print(f"\n{'全部通过 ✅' if FAIL == 0 else f'失败 {FAIL} 项 ❌'} ({PASS} passed)")
sys.exit(1 if FAIL else 0)
