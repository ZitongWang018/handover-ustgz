#!/usr/bin/env python
"""第三方代码审查：把 GRPO 关键代码段喂给 DeepSeek（复用 src/llm_client.py）。

只读审查，不改任何业务代码。
"""
from __future__ import annotations

import os
import sys
import time

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, os.path.abspath(SRC))

from llm_client import ChatClient  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

C = ChatClient()

# --------------------------------------------------------------------------- 代码段

ROLLOUT_CODE = r'''
# ---- src/rollout.py: HFPolicy.generate（GC toggle + 生成段起点）----
def generate(self, messages_list, cfg):
    torch = self.torch
    prompts = [self.render(self.truncate_messages(list(m), cfg), add_generation_prompt=True)
               for m in messages_list]
    # 生成必须用左 padding
    prev_side = self.tokenizer.padding_side
    self.tokenizer.padding_side = "left"
    try:
        enc = self.tokenizer(list(prompts), return_tensors="pt", padding=True,
                             add_special_tokens=False).to(self.device)
    finally:
        self.tokenizer.padding_side = prev_side
    # bug B 修法：生成段起点 = 含 padding 的初始总长度
    gen_start = int(enc["input_ids"].shape[1])
    criteria = _ActionStopCriteria(self.tokenizer, [gen_start]*enc["input_ids"].shape[0],
                                   cfg.max_new_tokens)
    gen_kwargs = dict(max_new_tokens=cfg.max_new_tokens, do_sample=cfg.temperature>0,
                      temperature=max(cfg.temperature,1e-5), top_p=cfg.top_p,
                      repetition_penalty=cfg.repetition_penalty,
                      pad_token_id=self.tokenizer.pad_token_id,
                      eos_token_id=self.tokenizer.eos_token_id,
                      stopping_criteria=[criteria])
    # bug A 修法：生成期间临时关掉 gradient_checkpointing
    _gc_on = bool(getattr(self.model, "is_gradient_checkpointing", False))
    if _gc_on:
        self.model.gradient_checkpointing_disable()
    try:
        with torch.no_grad():
            out = self.model.generate(**enc, use_cache=True, **gen_kwargs)
    finally:
        if _gc_on:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
    plens = [gen_start] * enc["input_ids"].shape[0]
    texts, gen_ids, stops = [], [], []
    for i in range(out.shape[0]):
        plen = int(plens[i])
        ids = out[i, plen:].tolist()
        stop = criteria.stop_len(i, len(ids))
        ids = ids[:stop]
        eos = self.tokenizer.eos_token_id
        while ids and eos is not None and ids[-1] == eos:
            ids.pop()
        gen_ids.append(ids); stops.append(len(ids))
        texts.append(self.tokenizer.decode(ids, skip_special_tokens=True))
    return texts, gen_ids, stops

# ---- src/rollout.py: build_token_sequence（双游标 ids vs rlen）----
def build_token_sequence(messages, gen_ids_per_assistant, policy):
    """返回 (input_ids, loss_mask, turn_boundaries)。
    assistant 段用真实生成 token id (mask=1)；其后的模板收尾 <|im_end|> 也 mask=1；
    其余（模板骨架 + 环境观测）用重新渲染的 token，mask=0。
    维护两个游标：ids=真实输出序列游标，rlen=纯模板渲染序列游标。
    """
    ids, mask, boundaries = [], [], []
    assistant_idx = 0
    rlen = 0
    prev_render = []
    def _render_ids(msgs, gen_prompt=False):
        return policy.tokenize_text(policy.render(msgs, add_generation_prompt=gen_prompt))
    for i, msg in enumerate(messages):
        if msg["role"] == "assistant" and assistant_idx < len(gen_ids_per_assistant):
            gen = list(gen_ids_per_assistant[assistant_idx]); assistant_idx += 1
            prefix_r = _render_ids(messages[:i], gen_prompt=True)
            if prev_render and prefix_r[: len(prev_render)] != prev_render:
                raise RuntimeError("chat 模板不是前缀稳定的")
            if len(prefix_r) < rlen:
                raise RuntimeError("渲染长度回退")
            gen_prompt_tokens = prefix_r[rlen:]       # <|im_start|>assistant\n 骨架, mask=0
            full_r = _render_ids(messages[:i+1], gen_prompt=False)
            if gen:
                empty_full = _render_ids(list(messages[:i]) +
                                         [{"role":"assistant","content":""}], gen_prompt=False)
                if empty_full[: len(prefix_r)] != prefix_r:
                    raise RuntimeError("空 assistant 渲染不以前缀开头")
                tail = empty_full[len(prefix_r):]    # 精确模板收尾 token
                ids.extend(gen_prompt_tokens); mask.extend([0]*len(gen_prompt_tokens))
                ids.extend(gen);            mask.extend([1]*len(gen))
                ids.extend(tail);           mask.extend([1]*len(tail))
                boundaries.append(len(ids))
            else:
                seq = gen_prompt_tokens + full_r[len(prefix_r):]
                ids.extend(seq); mask.extend([0]*len(seq))
            prev_render = full_r
            rlen = len(full_r)
        else:
            cur = _render_ids(messages[:i+1], gen_prompt=False)
            new = cur[rlen:]
            ids.extend(new); mask.extend([0]*len(new))
            prev_render = cur; rlen = len(cur)
    return ids, mask, boundaries

# ---- src/rollout.py: _align_slot_tasks（env slot 重排）----
def _align_slot_tasks(games, used, group_size):
    """textworld batch env 在 reset() 时 shuffle gamefiles，slot i 不是请求第 i 项。
    按「同一 gamefile 出现次数逐个消耗」对齐，返回 slot_task[s]。
    """
    n_slots = len(games)
    if [os.path.realpath(u) for u in used] == [os.path.realpath(x) for x in games]:
        return [i // group_size for i in range(n_slots)]
    avail = defaultdict(deque)
    for idx, gf in enumerate(games):
        avail[os.path.realpath(gf)].append(idx // group_size)
    slot_task = []
    for u in used:
        q = avail[os.path.realpath(u)]
        if not q:
            raise RuntimeError(f"env 装载的 gamefile 不在请求列表里：{u}")
        slot_task.append(int(q.popleft()))
    return slot_task
'''

TRAIN_CODE = r'''
# ---- src/train_grpo.py: per_token_logps（分块 log-softmax）----
_LOGSOFTMAX_CHUNK = 512
def per_token_logps(model, input_ids, attention_mask, chunk=_LOGSOFTMAX_CHUNK):
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits[:, :-1, :]          # bf16
    targets = input_ids[:, 1:]
    width = logits.size(1)
    if width <= chunk:
        return logits.float().log_softmax(dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    parts = []
    for s in range(0, width, chunk):
        lg = logits[:, s:s+chunk, :].float()
        tgt = targets[:, s:s+chunk]
        parts.append(lg.log_softmax(dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1))
    return torch.cat(parts, dim=1)

# ---- src/train_grpo.py: grpo_loss（无 KL，beta=0）----
def grpo_loss(model, batch, old_logps, clip_eps):
    new_logps = per_token_logps(model, batch.input_ids, batch.attention_mask)
    mask = batch.loss_mask[:, 1:]           # logp[:, i] 预测 input_ids[:, i+1]
    ratio = torch.exp(new_logps - old_logps)
    adv = batch.advantages.unsqueeze(1)     # 每条轨迹一个标量，广播到所有 token
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0-clip_eps, 1.0+clip_eps) * adv
    per_tok = -torch.min(surr1, surr2)
    denom = mask.sum().clamp(min=1.0)
    loss = (per_tok * mask).sum() / denom
    return loss, stats

# ---- src/train_grpo.py: build_model_and_tokenizer ----
def build_model_and_tokenizer(cfg, local_rank, world):
    if torch.cuda.is_available(): torch.cuda.set_device(local_rank)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=bfloat16, ...)
    model.config.use_cache = False
    if mcfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    if world > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)
    return model, tokenizer

# ---- src/train_grpo.py: main 里的 advantage + did_step + 保存 ----
with torch.no_grad():
    trajs = run_rollouts(env, policy, tasks, rollout_cfg, reward_cfg)
group_ids = [i // g for i in range(len(trajs))]
advs = group_advantages([t.reward for t in trajs], group_ids, use_std=True)
for t, a in zip(trajs, advs): t.advantage = a

order = list(range(len(trajs)))
n_micro = math.ceil(len(order) / micro)     # micro=1
adv_abs_mean = float(np.mean([abs(a) for a in advs]))
# 各 rank 是否更新必须一致，否则 DDP 挂死；用 MAX 归约统一决策
step_flag = torch.tensor([1.0 if adv_abs_mean > 1e-8 else 0.0], device=...)
if world > 1: dist.all_reduce(step_flag, op=dist.ReduceOp.MAX)
did_step = (args.dry_run is False) and step_flag.item() > 0.5
if did_step:
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        for k in range(n_micro):
            bt = make_batch([trajs[i] for i in order[k*micro:(k+1)*micro]], pad_id, device)
            old = per_token_logps(model, bt.input_ids, bt.attention_mask)
            batches.append((bt, old))
    total_active = sum(bt.n_active for bt, _ in batches) or 1
    for k, (bt, old) in enumerate(batches):
        loss, st = grpo_loss(model, bt, old, clip_eps)
        weight = bt.n_active / total_active
        (loss * weight).backward()
    torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
    optimizer.step()
if is_main(rank) and cfg["train"].get("save_every", 0) and ((it+1) % save_every == 0 ...):
    save_checkpoint(model, tokenizer, run_dir, f"iter{it+1}", cfg, {"iter": it+1})
'''

REWARD_CODE = r'''
# ---- src/reward.py: compute_reward ----
@dataclass
class RewardConfig:
    success: float = 1.0
    invalid_action_penalty: float = 0.10
    format_violation_penalty: float = 0.10
    repeat_penalty: float = 0.05
    valid_step_bonus: float = 0.01
    admissible_ratio_bonus: float = 0.0
    efficiency_bonus: float = 0.10
    step_penalty: float = 0.0
    use_sparse_only: bool = False
    clip_min: float = -1.0
    clip_max: float = 2.0

def compute_reward(stats, cfg, max_steps):
    """所有逐步骤累加的过程项都先除以 max_steps 归一化。"""
    c = cfg.resolved()
    denom = float(max_steps) if max_steps and max_steps > 0 else 1.0
    r = c.success * float(stats.won)
    r += c.invalid_action_penalty * (stats.n_invalid / denom)
    r += c.format_violation_penalty * (stats.n_format_violation / denom)
    r += c.repeat_penalty * (stats.n_repeat / denom)
    r += c.valid_step_bonus * (stats.n_valid_steps / denom)
    if c.admissible_ratio_bonus and stats.n_steps:
        r += c.admissible_ratio_bonus * (stats.n_valid_steps / stats.n_steps)
    if stats.won:
        r += c.efficiency_bonus * max(0.0, 1.0 - stats.n_steps / denom)
    r += c.step_penalty * (stats.n_steps / denom)
    r *= c.task_type_weights.get(stats.task_type, 1.0)
    return float(np.clip(r, c.clip_min, c.clip_max))

# ---- src/reward.py: group_advantages ----
def group_advantages(rewards, group_ids, eps=1e-4, use_std=True):
    """A_i = (R_i - mean_g) / (std_g + eps)；组内 std<eps 直接返回 0。"""
    rewards = np.asarray(rewards, dtype=np.float64); group_ids = np.asarray(group_ids)
    adv = np.zeros_like(rewards)
    for g in np.unique(group_ids):
        m = group_ids == g; grp = rewards[m]
        if grp.size <= 1: continue
        std = grp.std()
        if std < eps:
            adv[m] = 0.0; continue
        adv[m] = (grp - grp.mean()) / (std + eps) if use_std else (grp - grp.mean())
    return adv.tolist()

# 配置（configs/train_qwen3_4b.yaml）
# success: 1.0, invalid_action_penalty: 0.10, format_violation_penalty: 0.10,
# repeat_penalty: 0.05, valid_step_bonus: 0.01, efficiency_bonus: 0.10,
# clip_min: -1.0, clip_max: 2.0, max_episode_steps: 30, group_size: 4
'''

# --------------------------------------------------------------------------- 问题

Q1 = '''你是资深 RL/LLM 训练专家，请审查在 ALFWorld（文本家务任务，最长 30 步）上用 Qwen3-4B-Instruct-2507 做多轮 agent GRPO 训练的自研代码。ChatML 模板，policy 用 HF generate 采样，训练用手写 PPO-clip 目标，beta=0 无参考模型无 KL。

【代码段 1：rollout.py 关键函数】
'''

Q_ROLLOUT = '''请回答下面 3 个问题，每个问题给出明确结论 + 理由 + 若有问题给出改法（可给关键代码）：

问题 5：`build_token_sequence` 用双游标（`ids` 真实输出游标 vs `rlen` 纯模板渲染游标）拼训练序列。assistant 段用真实 gen_ids，模板收尾用「同一前缀 + 空 assistant 内容」渲染出的 `tail`。请仔细检查这套逻辑有没有 off-by-one 或索引错误？特别注意：`gen_prompt_tokens = prefix_r[rlen:]` 与 `tail = empty_full[len(prefix_r):]` 这两个切片的起点对不对；`rlen` 推进用的是 `full_r`（含真实 assistant 文本的完整模板渲染）的长度，而实际输出 ids 用的是 gen 的长度 —— 两者不一致时后续轮次的 `prefix_r[rlen:]` 和 `cur[rlen:]` 会不会错位？如果有 bug 请给出正确写法。

问题 6：生成段起点用了「含 padding 的初始总长度」`gen_start = enc["input_ids"].shape[1]`，而不是 `attention_mask.sum()`。代码注释说左 padding 时必须用前者。这个修法对吗？请分析：HF generate 在左 padding 下 `out[i]` 的布局是什么，`out[i, plen:]` 用哪个才正确？是否所有情况都成立（比如 batch 内长度相等、tokenizer 是否加了特殊 token）？有没有更稳妥的写法？

问题 7：训练时开 gradient_checkpointing、生成时临时 `gradient_checkpointing_disable()`、生成完再 enable，在 DDP（DistributedDataParallel，8 卡）下安全吗？会不会造成 rank 间不一致、hook 泄漏、或影响 DDP 的 reducer/梯度同步？请给出风险点和更安全的替代方案。'''

Q_TRAIN = '''请回答下面 3 个问题，每个问题给出明确结论 + 理由 + 若有问题给出改法（可给关键代码）：

问题 1：多轮 agent RL 里，loss mask 只覆盖 assistant 生成 token（含其后的模板收尾 `<|im_end|>`，mask=1），环境观测和模板骨架 mask=0。这样做对吗？特别地：模板收尾 token `<|im_end|>` 该不该 mask=1？mask=1 会不会让模型学到「过早结束」或在多轮场景下引入偏差？请给出建议。

问题 2：GRPO 的 advantage 是 trajectory 级标量（每条轨迹一个数），在 loss 里广播到该轨迹的每个 token 上，然后除以「全局 mask token 总数」归一化。这种"按 token 数归一化"（sum over all tokens / total_tokens，即 token-level mean）对吗？和 PPO/GRPO 论文里的 token 级归一化相比如何？不按 token 数归一化（比如按轨迹数平均）会有什么后果？当前这个 `denom = mask.sum()`（跨轨迹的全局 token 数）会不会让长轨迹 dominate、短轨迹梯度被稀释？请给出建议（含 Dr.GRPO 的 loss 归一化讨论）。

问题 4：当前实现 beta=0，完全没有 KL 惩罚、也没有参考模型。GRPO 在长时域多轮 agent 任务上会不会崩？请评估风险（reward hacking、熵坍塌、生成退化、长度爆炸等），并给出是否需要加 KL 的建议与代价（显存/算力）。'''

Q_REWARD = '''请回答下面 1 个问题（这是重点，请逐项仔细检查符号和量纲）：

问题 3：奖励函数把逐步骤累加的项都除以 `max_steps`（=30）归一化。请评估这个设计是否合理？会不会引入新问题？
特别请仔细核对**符号**：配置里 `invalid_action_penalty: 0.10`、`format_violation_penalty: 0.10`、`repeat_penalty: 0.05` 都是**正数**，而 `compute_reward` 里是 `r += c.invalid_action_penalty * (n_invalid/denom)` 这样的**加法**。函数文档却写着"全程非法动作 ≈ -0.20"。请判断：这是不是把惩罚项写成了奖励（符号错误）？正确写法应该是什么？这个 bug 在 GRPO 里会造成什么具体后果？
另外请评估：`max_steps` 归一化的量纲是否合理（过程项被压到 ±0.1 量级，而 success=1.0）？早期 invalid_rate 接近 1.0 时会发生什么？除以 `max_steps`（30）而不是实际 `n_steps` 会不会让短轨迹的过程项被低估？'''

SYSTEM = None


def ask(name: str, code: str, question: str, effort: str = "high", max_tokens: int = 32768) -> str:
    prompt = (
        "你是资深 RL/LLM 训练专家。下面是别人手写的 ALFWorld × Qwen3-4B 多轮 agent GRPO 训练代码"
        "（自研实现，不用 trl/peft）。\n\n【代码】\n" + code + "\n\n【问题】\n" + question
    )
    print(f"--- 调用 DeepSeek: {name} (effort={effort}) ...", flush=True)
    t0 = time.time()
    res = C.chat([{"role": "user", "content": prompt}], temperature=0.0,
                 max_tokens=max_tokens, effort=effort)
    print(f"    done {time.time()-t0:.1f}s  prompt_tok={res.prompt_tokens} "
          f"completion_tok={res.completion_tokens} len={len(res.text)}", flush=True)
    if not res.text:
        return f"<EMPTY CONTENT> reasoning_len={len(res.reasoning)}\n--- reasoning ---\n{res.reasoning}"
    return res.text


def main() -> int:
    jobs = [
        ("01_rollout", ROLLOUT_CODE, Q_ROLLOUT),
        ("02_train", TRAIN_CODE, Q_TRAIN),
        ("03_reward", REWARD_CODE, Q_REWARD),
    ]
    for name, code, q in jobs:
        try:
            text = ask(name, code, q)
        except Exception as e:  # noqa: BLE001
            text = f"<调用失败> {type(e).__name__}: {e}"
            print(text, flush=True)
        with open(os.path.join(OUT, f"deepseek_{name}.md"), "w", encoding="utf-8") as fh:
            fh.write(f"# DeepSeek 审查 - {name}\n\n{text}\n")
        print(f"saved -> {OUT}/deepseek_{name}.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
