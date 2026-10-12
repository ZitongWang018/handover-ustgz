#!/usr/bin/env python
"""F：独立证伪重写后的 `build_token_sequence`（双游标实现）。

核心思想：**另写一个从规格直接推出的参考实现** —— 不用「运行游标」，而是每段都
`render(messages[:i])` 现算基准长度，取 `render(messages[:i+1])[base:]` 作为增量。
这跟被验证实现的「rlen 游标推进」是完全不同的代码路径，能抓出游标漂移。

规格（描述性）：把整段对话用 chat 模板渲染成一条序列，其中每个 assistant 的**内容**
   不采用模板对文本的重新分词，而是替换为模型真实生成的 token id，并在其后接上
   「同一前缀 + 空 assistant 内容」渲染得到的模板尾。

覆盖 (a)(b)(c)(d)，见 team-lead 的验收清单。

运行（CPU 即可，只需 tokenizer）：
    python scripts/verify_tokens.py
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import traceback
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from rollout import HFPolicy, RolloutConfig, build_token_sequence  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(f"{name} :: {detail}")


# --------------------------------------------------------------------------- 参考实现


def reference_build(messages, gen_ids_per_assistant, policy):
    """从规格直接写的参考实现（无「运行游标」，每段现算 base）。"""
    ids: list[int] = []
    mask: list[int] = []
    ai = 0
    for i, msg in enumerate(messages):
        # base = 前 i 条消息的渲染（i==0 时为空序列；apply_chat_template 不接受空列表）
        base = [] if i == 0 else policy.tokenize_text(
            policy.render(messages[:i], add_generation_prompt=False))
        if msg["role"] == "assistant" and ai < len(gen_ids_per_assistant):
            gen = list(gen_ids_per_assistant[ai])
            ai += 1
            gp_full = policy.tokenize_text(policy.render(messages[:i], add_generation_prompt=True)) if i else []
            gpt = gp_full[len(base):]                       # 生成提示骨架（<|im_start|>assistant\n）
            empty = policy.tokenize_text(policy.render(
                list(messages[:i]) + [{"role": "assistant", "content": ""}],
                add_generation_prompt=False,
            ))
            tail = empty[len(gp_full):]                     # 模板尾（<|im_end|>\n）
            if gen:
                ids += gpt + gen + tail
                mask += [0] * len(gpt) + [1] * len(gen) + [1] * len(tail)
            else:
                # 规格 (d)：模型没生成任何 token → 整段 mask=0（不训练空输出）
                seq = gpt + tail
                ids += seq
                mask += [0] * len(seq)
        else:
            cur = policy.tokenize_text(policy.render(messages[:i + 1], add_generation_prompt=False))
            new = cur[len(base):]
            ids += new
            mask += [0] * len(new)
    return ids, mask


def spans_of(mask):
    """把 mask 里的连续 1 段切成 [(start, end)]。"""
    out = []
    s = None
    for i, m in enumerate(mask):
        if m and s is None:
            s = i
        elif not m and s is not None:
            out.append((s, i))
            s = None
    if s is not None:
        out.append((s, len(mask)))
    return out


# --------------------------------------------------------------------------- 工具


def make_policy(tok):
    p = HFPolicy.__new__(HFPolicy)
    p.tokenizer = tok
    p.enable_thinking = False
    return p


def tok_ids(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def find_gen_with_H(tok, want_sign, rng, trials=60000):
    """找一个 gen：H = len(tokenize(decode(gen))) - len(gen) 符号正确且非 0。"""
    nv = len(tok)
    for _ in range(trials):
        L = rng.randint(1, 4)
        gen = [rng.randrange(nv) for _ in range(L)]
        if any(tok.convert_ids_to_tokens(g).startswith("<") for g in gen):
            continue
        txt = tok.decode(gen, skip_special_tokens=True)
        if not txt.strip():
            continue
        h = len(tok_ids(tok, txt)) - L
        if (h > 0 and want_sign > 0) or (h < 0 and want_sign < 0):
            return gen, txt, h
    return None


def rand_text(rng, tok, max_new_tokens=12):
    """随机造一段「像人话」的文本。"""
    words = ["the", "a", "apple", "mug", "fridge", "shelf", "desk", "look", "go", "to",
             "in", "on", "take", "put", "hot", "cold", "clean", "lamp", "countertop", "tray"]
    n = rng.randint(3, max_new_tokens)
    return " ".join(rng.choice(words) for _ in range(n))


def make_conversation(rng, tok, n_turns, gen_for_turn):
    """gen_for_turn(rng, k) -> (content, gen)；content 与 gen 必须自洽（真实 rollout 里就是这样）。"""
    msgs = [{"role": "system", "content": "You are an expert agent. Output exactly two lines."}]
    gens = []
    for k in range(n_turns):
        msgs.append({"role": "user", "content": f"Observation: {rand_text(rng, tok, 20)}\n\nAdmissible Commands:\n- look\n- go to shelf 1"})
        content, gen = gen_for_turn(rng, k)
        msgs.append({"role": "assistant", "content": content})
        gens.append(list(gen))
    return msgs, gens


# --------------------------------------------------------------------------- (a)


def test_a_exactness(policy, tok):
    print("\n== (a) 精确还原性：mask span / 逐位 gen / mask=0 反查 ==")
    rng = random.Random(11)

    def gen_for_turn(r, k):
        text = f"Thought: step {k} done.\nAction: look"
        return text, tok_ids(tok, text)

    msgs, gens = make_conversation(rng, tok, 4, gen_for_turn)
    ids, mask, bounds = build_token_sequence(msgs, gens, policy)

    n_asst = sum(1 for m in msgs if m["role"] == "assistant")
    sp = spans_of(mask)
    check(f"mask=1 span 数 == assistant 数（{len(sp)} vs {n_asst}）", len(sp) == n_asst, f"spans={sp}")

    ok_gen = True
    ok_tail = True
    for k, (s, e) in enumerate(sp):
        gen = gens[k]
        if ids[s:s + len(gen)] != gen:
            ok_gen = False
        # span = gen + tail
        if e - s <= len(gen):
            ok_tail = False
    check("每个 span 的前 |gen| 个 token 与假 gen id 逐位一致", ok_gen and len(sp) == len(gens))
    check("每个 span 都包含 gen 之后的模板尾（span 长度 > |gen|）", ok_tail)

    # mask=1 decode
    m1 = [ids[i] for i, m in enumerate(mask) if m]
    m0 = [ids[i] for i, m in enumerate(mask) if not m]
    d1 = tok.decode(m1, skip_special_tokens=False)
    d0 = tok.decode(m0, skip_special_tokens=False)
    print("  --- decode(mask=1) ---")
    print("  " + repr(d1[:400]))
    print("  --- decode(mask=0) 前 500 字符 ---")
    print("  " + repr(d0[:500]))

    # 反向检查：mask=0 里不能出现任何 assistant 正文
    leaked = [m["content"][:24] for m in msgs if m["role"] == "assistant" and m["content"][:24] in d0]
    check("mask=0 里不出现任何 assistant 正文（无重复注入）", not leaked, f"泄漏={leaked}")

    # 正向：每个 assistant 正文应能在 mask=1 里找到（content 与 gen 自洽）
    missing = []
    for m in msgs:
        if m["role"] != "assistant":
            continue
        frag = m["content"][:24]
        if frag not in d1:
            missing.append(frag)
    check("mask=1 里能找到每段 assistant 正文", not missing, f"缺失={missing}")

    check("boundaries 条数 == assistant 数", len(bounds) == n_asst, f"{bounds}")

    # 与独立参考实现完全一致
    r_ids, r_mask = reference_build(msgs, gens, policy)
    check("(a) 输出与独立参考实现逐位一致", r_ids == ids and r_mask == mask,
          f"len impl={len(ids)} ref={len(r_ids)}; ids equal={r_ids == ids}")


# --------------------------------------------------------------------------- (a') 随机轰炸


def test_a_random(policy, tok):
    print("\n== (a') 随机 300 条对话：与独立参考实现逐位一致 ==")
    rng = random.Random(2026)
    nv = len(tok)
    bad = []
    total_tok = 0
    for t in range(300):
        n_turns = rng.randint(1, 6)

        def gen_for_turn(r, k):
            style = r.random()
            if style < 0.5:
                text = f"Thought: t{k}\nAction: {r.choice(['look', 'go to shelf 1'])}"
                g = tok_ids(tok, text)
            elif style < 0.8:
                g = [r.randrange(nv) for _ in range(r.randint(1, 4))]
                text = tok.decode(g, skip_special_tokens=True)
                if not text.strip():
                    text = f"REPLY-{k}"
                    g = tok_ids(tok, text)
            else:
                g = []
                text = ""
            return text, g

        msgs, gens = make_conversation(rng, tok, n_turns, gen_for_turn)
        try:
            ids, mask, _ = build_token_sequence(msgs, gens, policy)
        except Exception as exc:  # noqa: BLE001
            bad.append(f"case{t} 抛异常 {exc!r}")
            continue
        r_ids, r_mask = reference_build(msgs, gens, policy)
        total_tok += len(ids)
        if r_ids != ids or r_mask != mask:
            msg = f"case{t} 与参考不一致 len impl={len(ids)} ref={len(r_ids)} mask_eq={r_mask == mask}"
            if len(bad) == 0:      # 只 dump 第一个不符的细节
                d = next((j for j in range(min(len(mask), len(r_mask))) if mask[j] != r_mask[j]), None)
                print(f"    [dump] 第一个 mask 不一致位置={d}")
                print(f"    [dump] messages={[ (m['role'], m['content'][:40]) for m in msgs ]}")
                print(f"    [dump] gens lens={[len(g) for g in gens]}")
                print(f"    [dump] impl mask[:60]={mask[:60]}")
                print(f"    [dump] ref  mask[:60]={r_mask[:60]}")
                print(f"    [dump] ids[:60]={ids[:60]}")
                for j, (g, m) in enumerate(zip(gens, [x for x in msgs if x['role'] == 'assistant'])):
                    print(f"    [dump] gen[{j}] = {g}  decode={tok.decode(g, skip_special_tokens=True)[:40]!r}")
            bad.append(msg)
    check(f"300 条随机对话全部与参考实现逐位一致（累计 {total_tok} token）", not bad, f"{bad[:3]}")


# --------------------------------------------------------------------------- (b)


def test_b_cursor_decoupling(policy, tok):
    print("\n== (b) 游标解耦：H = len(tokenize(decode(gen))) - len(gen) != 0 ==")
    rng = random.Random(99)
    pos = find_gen_with_H(tok, +1, rng)
    neg = find_gen_with_H(tok, -1, rng)
    print(f"  H>0 样本: {pos}")
    print(f"  H<0 样本: {neg}")
    if not pos or not neg:
        check("找到 H>0 与 H<0 的 gen 样本", False, f"pos={pos is not None} neg={neg is not None}")
        return
    check("找到 H>0 与 H<0 的 gen 样本", True)

    for tag, (gen, txt, h) in (("H>0", pos), ("H<0", neg)):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Observation: room.\n\nAdmissible Commands:\n- look"},
            {"role": "assistant", "content": txt},
            {"role": "user", "content": "Observation: after action.\n\nAdmissible Commands:\n- look"},
            {"role": "assistant", "content": "Action: look"},
        ]
        gens = [list(gen), tok_ids(tok, "Action: look")]
        ids, mask, _ = build_token_sequence(msgs, gens, policy)
        sp = spans_of(mask)
        s0, e0 = sp[0]
        # 独立算 tail
        prefix_r = tok_ids(tok, policy.render(msgs[:2], add_generation_prompt=True))
        empty = tok_ids(tok, policy.render(msgs[:2] + [{"role": "assistant", "content": ""}], add_generation_prompt=False))
        tail = empty[len(prefix_r):]
        nxt = ids[s0 + len(gen): s0 + len(gen) + len(tail)]

        check(f"[{tag}] H={h} != 0", h != 0, f"H={h}")
        check(f"[{tag}] gen 之后紧跟的是**模板尾**", nxt == tail, f"got={nxt} want={tail}")
        residue = tok_ids(tok, tok.decode(gen, skip_special_tokens=True))[len(gen):]
        if residue:
            got_res = ids[s0 + len(gen): s0 + len(gen) + len(residue)]
            check(f"[{tag}] 后面**不是**'解码再编码'的残留 {residue}", got_res != residue or residue == tail,
                  "解码再编码残留混进来了")

        # 后续 user turn 的段长必须 == render 增量（没有"重复一整段"）
        # 用参考实现做全序列对拍
        r_ids, r_mask = reference_build(msgs, gens, policy)
        check(f"[{tag}] 全序列与独立参考实现逐位一致", r_ids == ids and r_mask == mask,
              f"len impl={len(ids)} ref={len(r_ids)}")

        # 显式量一下第二个 user turn 的增量
        u2_base = tok_ids(tok, policy.render(msgs[:3], add_generation_prompt=False))
        u2_cur = tok_ids(tok, policy.render(msgs[:4], add_generation_prompt=False))
        u2_delta = len(u2_cur) - len(u2_base)
        # 在 ids 里，第二个 user 段位于 [e0, e0+u2_delta)
        seg = ids[e0:e0 + u2_delta]
        check(f"[{tag}] 第二个 user turn 段长 == render 增量({u2_delta})，未重复整段",
              len(seg) == u2_delta and ids[e0:e0 + u2_delta] == u2_cur[len(u2_base):])


# --------------------------------------------------------------------------- (c)


def test_c_truncate_alignment(policy, tok):
    print("\n== (c) truncate_messages + gen 切片对齐 ==")
    rng = random.Random(7)
    cfg = RolloutConfig(max_context_tokens=250, keep_recent_turns=3)

    n_gen = 10
    msgs = [{"role": "system", "content": "You are an expert agent."}]
    gens = []
    for k in range(n_gen):
        msgs.append({"role": "user", "content": f"Observation: {rand_text(rng, tok, 60)}"})
        content = f"Thought: turn {k} reasoning about the room.\nAction: look {rand_text(rng, tok, 20)}"
        msgs.append({"role": "assistant", "content": content})
        gens.append(tok_ids(tok, content))

    full_r = tok_ids(tok, policy.render(msgs, add_generation_prompt=True))
    tr = policy.truncate_messages(list(msgs), cfg)
    n_asst_kept = sum(1 for m in tr if m["role"] == "assistant")
    gen_kept = gens[len(gens) - n_asst_kept:] if n_asst_kept else []

    print(f"  完整渲染 {len(full_r)} token（max_context_tokens={cfg.max_context_tokens}）")
    print(f"  裁剪后 {len(tr)} 条消息，其中 assistant {n_asst_kept} 条；gen_kept {len(gen_kept)} 条")
    check("完整渲染确实超过 max_context_tokens（触发了裁剪）", len(full_r) > cfg.max_context_tokens,
          f"{len(full_r)} vs {cfg.max_context_tokens}")
    check("裁剪后 assistant 条数 < 全部 assistant 数", n_asst_kept < n_gen, f"{n_asst_kept} vs {n_gen}")
    check("gen_kept 条数 == 裁剪后 assistant 条数", len(gen_kept) == n_asst_kept)

    # 配对方向：gen_kept[i] 必须对应 tr 里第 i 个 assistant（即**最后** n_asst_kept 次生成）
    tr_asst = [m["content"] for m in tr if m["role"] == "assistant"]
    pair_ok = True
    detail = []
    for i, (content, g) in enumerate(zip(tr_asst, gen_kept)):
        if tok_ids(tok, content) != list(g):
            pair_ok = False
            detail.append((content[:20], tok.decode(g)[:20]))
    check("gen_kept 与裁剪后 assistant **一一配对**（是最后 n 次生成，不是最早 n 次）", pair_ok, f"{detail[:3]}")

    # 显式确认切的是尾部
    tail_ids = [c.split("turn ")[1].split(" ")[0] for c in tr_asst]
    print(f"  裁剪后保留的 assistant 标记（turn 编号）：{tail_ids}")
    check("保留的是尾部（最后几条 turn）", tail_ids == [str(k) for k in range(n_gen - n_asst_kept, n_gen)],
          f"{tail_ids}")

    ids, mask, _ = build_token_sequence(tr, gen_kept, policy)
    gen_total = sum(len(g) for g in gen_kept)
    print(f"  训练序列 len(ids)={len(ids)}（gen 合计 {gen_total}）")
    check("len(ids) <= max_context_tokens*1.2 + gen 总量", len(ids) <= cfg.max_context_tokens * 1.2 + gen_total,
          f"len={len(ids)} 上限={cfg.max_context_tokens * 1.2 + gen_total:.0f}")
    check("训练序列远小于完整渲染（裁剪有效）", len(ids) < len(full_r), f"{len(ids)} vs {len(full_r)}")

    r_ids, r_mask = reference_build(tr, gen_kept, policy)
    check("(c) 输出与独立参考实现逐位一致", r_ids == ids and r_mask == mask)


def test_c2_short_but_long(policy, tok):
    """(c2) 边界：消息条数 <= 2*keep_recent_turns，但渲染超长 → 裁剪会不会**复制**前缀？

    `truncate_messages` 的返回是 `head[:-1] + [head[-1]] + tail`，其中
    `tail = messages[-max(2, 2*keep_recent_turns):]`。当 `len(messages) <= 2*keep`
    时，`tail == messages`，而 `head[-1]` 又是 `messages[1]`（首个 user），
    于是结果会变成 `[system, user0] + messages` —— **system / user0 各出现两次**。
    """
    print("\n== (c2) 裁剪边界：消息数少但渲染超长 ==")
    rng = random.Random(3)
    keep = 12                                    # 与 configs/train_qwen3_4b.yaml 一致
    cfg = RolloutConfig(max_context_tokens=300, keep_recent_turns=keep)
    n_turns = 8                                  # len(messages) = 17 <= 2*keep = 24
    msgs = [{"role": "system", "content": "You are an expert agent."}]
    for k in range(n_turns):
        msgs.append({"role": "user", "content": f"Observation: {rand_text(rng, tok, 200)}"})
        msgs.append({"role": "assistant", "content": f"Thought: turn {k}\nAction: look"})
    full_r = tok_ids(tok, policy.render(msgs, add_generation_prompt=True))
    tr = policy.truncate_messages(list(msgs), cfg)
    roles = [m["role"] for m in tr]
    n_sys = roles.count("system")
    dup_user0 = sum(1 for m in tr if m["role"] == "user" and m["content"] == msgs[1]["content"])
    print(f"  len(messages)={len(msgs)} (2*keep={2 * keep})；完整渲染 {len(full_r)} token > {cfg.max_context_tokens}")
    print(f"  裁剪后 {len(tr)} 条，roles 前 6 = {roles[:6]}，system 出现 {n_sys} 次，user0 出现 {dup_user0} 次")
    check("裁剪触发", len(full_r) > cfg.max_context_tokens)
    check("裁剪后不复制 system 消息", n_sys == 1, f"system 出现 {n_sys} 次")
    check("裁剪后不复制首个 user（任务描述）消息", dup_user0 == 1, f"user0 出现 {dup_user0} 次")
    check("裁剪后条数 <= 原条数", len(tr) <= len(msgs), f"{len(tr)} vs {len(msgs)}")


def test_c3_dup_impact(policy, tok):
    """(c3) 若 (c2) 复制的确发生，量化它对 build_token_sequence 的影响。"""
    print("\n== (c3) 复制前缀对训练序列的影响（若 (c2) 失败） ==")
    rng = random.Random(5)
    cfg = RolloutConfig(max_context_tokens=300, keep_recent_turns=12)
    n_turns = 8
    msgs = [{"role": "system", "content": "You are an expert agent."}]
    gens = []
    for k in range(n_turns):
        msgs.append({"role": "user", "content": f"Observation: {rand_text(rng, tok, 200)}"})
        content = f"Thought: turn {k}\nAction: look"
        msgs.append({"role": "assistant", "content": content})
        gens.append(tok_ids(tok, content))
    tr = policy.truncate_messages(list(msgs), cfg)
    n_asst_kept = sum(1 for m in tr if m["role"] == "assistant")
    gen_kept = gens[len(gens) - n_asst_kept:] if n_asst_kept else []
    ids, mask, _ = build_token_sequence(tr, gen_kept, policy)
    d0 = tok.decode([ids[i] for i, m in enumerate(mask) if not m], skip_special_tokens=False)
    task_line = msgs[1]["content"].split("Observation:")[0]
    n_task_in_prompt = d0.count("You are an expert agent.")
    print(f"  裁剪后 {len(tr)} 条消息；训练序列 len(ids)={len(ids)}")
    print(f"  system 提示在 mask=0 里出现 {n_task_in_prompt} 次（正常应为 1）")
    check("system 提示在训练序列的 mask=0 里只出现 1 次", n_task_in_prompt == 1, f"出现 {n_task_in_prompt} 次")





# --------------------------------------------------------------------------- (d)


class FakeView:
    def __init__(self, env_desc="You are in a room.", task_desc="put apple in fridge"):
        self.raw = env_desc
        self.env_desc = env_desc
        self.task_desc = task_desc


class FakeStep:
    def __init__(self, n):
        self.obs = ["You see nothing special."] * n
        self.rewards = [0.0] * n
        self.dones = [True] * n
        self.infos = {}
        self.game_files = []
        self.won = [False] * n
        self.admissible = [["look"]] * n
        self.expert_plan = [None] * n


class FakeEnv:
    def __init__(self, batch_size=1, max_episode_steps=4):
        self.batch_size = batch_size
        self.max_episode_steps = max_episode_steps
        self.game_files = ["/fake/task0/game.tw-pddl"]

    def reset(self, game_files=None, seed=None):
        gf = list(game_files or [])
        obs = ["welcome\n\nYou are in a room.\n\nYour task is to: put apple in fridge"] * len(gf)
        return obs, [FakeView() for _ in obs], [["look"]] * len(gf), gf, [None] * len(gf)

    def step(self, actions):
        return FakeStep(len(actions))

    def close(self):
        pass


class EmptyGenPolicy:
    """generate 永远返回空 gen id —— 用来打 run_rollouts 的 'mask 全 0' 守卫。"""

    def __init__(self, base):
        self.base = base

    def render(self, *a, **k):
        return self.base.render(*a, **k)

    def tokenize_text(self, *a, **k):
        return self.base.tokenize_text(*a, **k)

    def truncate_messages(self, *a, **k):
        return self.base.truncate_messages(*a, **k)

    def generate(self, messages_list, cfg=None):
        texts = ["Thought: x\nAction: look" for _ in messages_list]
        return texts, [[] for _ in messages_list], [0 for _ in messages_list]


def test_d_edges(policy, tok):
    print("\n== (d) 边界 ==")
    sysmsg = {"role": "system", "content": "sys"}
    usr = {"role": "user", "content": "Observation: room."}

    # d1: gen = []
    msgs = [sysmsg, usr,
            {"role": "assistant", "content": "REPLY-0"}, {"role": "user", "content": "Observation: next."},
            {"role": "assistant", "content": "REPLY-1"}]
    gens = [[], tok_ids(tok, "REPLY-1")]
    try:
        ids, mask, bounds = build_token_sequence(msgs, gens, policy)
        first_span = spans_of(mask)
        check("gen=[] 不崩，且该 assistant 段 mask 全 0",
              len(first_span) == 1 and bounds and bounds[0] == first_span[0][1], f"spans={first_span}")
        check("gen=[] 时 'REPLY-0' 出现在 mask=0 里（按模板渲染，不训练）",
              "REPLY-0" in tok.decode([ids[i] for i, m in enumerate(mask) if not m], skip_special_tokens=False))
    except Exception as exc:  # noqa: BLE001
        check("gen=[] 不崩", False, repr(exc))

    # d2: 连续两个 assistant
    msgs2 = [sysmsg, usr,
             {"role": "assistant", "content": "A1"}, {"role": "assistant", "content": "A2"},
             {"role": "user", "content": "Observation: u2"}]
    gens2 = [tok_ids(tok, "A1")]
    try:
        ids2, mask2, _ = build_token_sequence(msgs2, gens2, policy)
        sp2 = spans_of(mask2)
        d0 = tok.decode([ids2[i] for i, m in enumerate(mask2) if not m], skip_special_tokens=False)
        check("连续两个 assistant：不崩，只有第一个被 mask=1",
              len(sp2) == 1 and "A2" in d0, f"spans={sp2}")
        r_ids, r_mask = reference_build(msgs2, gens2, policy)
        check("连续两个 assistant：与参考实现一致", r_ids == ids2 and r_mask == mask2)
    except Exception as exc:  # noqa: BLE001
        check("连续两个 assistant：不崩", False, repr(exc))

    # d3: 只有 system+user，没有 assistant -> mask 全 0
    ids3, mask3, _ = build_token_sequence([sysmsg, usr], [], policy)
    check("只有 system+user：mask_sum == 0", sum(mask3) == 0, f"mask_sum={sum(mask3)}")

    # d4: messages 为空
    ids4, mask4, b4 = build_token_sequence([], [], policy)
    check("messages 为空：返回 ([],[],[])", ids4 == [] and mask4 == [] and b4 == [])

    # d5: run_rollouts 的 "mask 全 0" 守卫必须真的炸
    from reward import make_reward_cfg
    from rollout import run_rollouts
    env = FakeEnv(batch_size=1, max_episode_steps=4)
    pol = EmptyGenPolicy(policy)
    cfg = RolloutConfig(group_size=1, max_steps=4, max_context_tokens=4096)
    tasks = [{"game_file": "/fake/task0/game.tw-pddl", "task_id": "t0", "task_type": "tt"}]
    try:
        run_rollouts(env, pol, tasks, cfg, make_reward_cfg(None))
        check("run_rollouts 在 mask 全 0 时抛 RuntimeError", False, "没抛")
    except RuntimeError as exc:
        check(f"run_rollouts 在 mask 全 0 时抛 RuntimeError（{str(exc)[:50]}...）", True)
    except Exception as exc:  # noqa: BLE001
        check("run_rollouts 在 mask 全 0 时抛 RuntimeError", False, f"抛了别的 {exc!r}")


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/thinktool/models/qwen3_4b_inst_2507"))
    args = ap.parse_args()

    from transformers import AutoTokenizer

    model_path = os.path.expandvars(os.path.expanduser(args.model))
    print(f"[tok] {model_path}")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    policy = make_policy(tok)
    print(f"[tok] vocab={len(tok)} eos={tok.eos_token_id} im_end={tok.convert_tokens_to_ids('<|im_end|>')}")

    for fn in (test_a_exactness, test_a_random, test_b_cursor_decoupling,
               test_c_truncate_alignment, test_c2_short_but_long, test_c3_dup_impact,
               test_d_edges):
        try:
            fn(policy, tok)
        except Exception:  # noqa: BLE001
            FAILS.append(f"{fn.__name__} 崩溃")
            print(traceback.format_exc())

    print("\n" + ("F 全部通过 ✅" if not FAILS else f"F 失败 {len(FAILS)} 项 ❌"))
    for f in FAILS:
        print("  - " + f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
