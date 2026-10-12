# DeepSeek 审查 - 02_train

` (chat template initial assistant header) maybe it may be part of skeleton not generated; if it's generated, should be included too." Hmm.

Let's compose final answer.

Double-check "问题4" no reference model memory for 4B: Qwen3-4B bf16 weights ~8GB (4e9*2=8e9 bytes = 8GB). Actually 4B params bf16 ~8GB yes. Additional optimizer states if full fine-tune AdamW ~8GB*maybe 12=96GB? But they didn't ask. Could mention "显存成本：参考模型只加权重（~8GB/rank），不加 optimizer states，因为 frozen." This is helpful.
Compute: "Ref forward is another bf16 forward; if you cache per rollout, it's not every micro-batch? Actually need ref_logps for all generated sequences; once per rollout data, can cache. But if data is newly sampled each iteration, need compute once per new trajectory; then cached through microbatches across outer gradient accumulation. So overhead one forward per rollout token; compared to training forward and old_logps, maybe 1/3 extra." Good.

Need maybe mention "If adding reference model, ensure no DDP wrapper for ref; do not call `ref_model.train()`; keep `eval()`." Not necessary.

Let's now prepare final answer.## 问题 1：loss mask 只覆盖 assistant token，含 `<|im_end|>`，mask=1 对吗？

### 结论

**做法基本正确。** 模板骨架、环境观测、system/user prompt 等非模型生成 token 应 mask=0；assistant 从第一个生成 token 到 `<|im_end|>` 都应 mask=1。**`<|im_end|>` 应该 mask=1。**

### 理由

`<|im_end|>` 是模型生成的动作终止 token，表示“本轮 turn 结束，交回环境”。在多轮 agent 中，它本身就是一个重要决策：模型要学习何时停止当前 turn。如果把它 mask=0，则模型无法通过梯度学习“什么时候该结束 turn”，容易出现格式错误、停不下来或早停不受控。

它不是普通模板 token，而是策略生成 token 的一部分。

### “mask=1 会不会让模型学到过早结束？”

mask=1 本身不会导致模型系统性地早停。真正导致早停的是：

1. reward/advantage 设计不好；
2. trajectory 级 advantage 粗粒度广播；
3. 没有长度惩罚或早停惩罚；
4. 环境对空动作/无效动作惩罚不足。

如果早停能带来更高 reward，那模型当然会学早停；这跟 `<|im_end|>` 是否 mask=1 无关。反而如果 mask=0，模型会失去对终止 token 的控制能力。

### 建议

```text
- 所有 assistant 生成 token，包括 <|im_end|>，mask=1
- 环境观测 / 模板骨架 / padding / system / user prompt mask=0
- 如果担心早停，在 reward 中显式加：
  - 无效结束惩罚
  - 步数 penalty
  - 任务失败 penalty
- 监控平均 turn 数、平均轨迹长度、成功样本长度、<|im_end|> 概率
```

简单说：**mask=1 是对的；早停风险应通过 reward 和长度压控解决，而不是 mask 掉 `<|im_end|>`。**

---

## 问题 2：GRPO trajectory 级 advantage 广播到 token，并做全局 token-level mean 归一化，对吗？

### 结论

**当前实现不完全符合原始 GRPO 常见的 per-sequence 归一化。**  
当前 `denom = mask.sum()` 是 token-level mean，会引入长度 bias：**长轨迹 dominate 梯度，短轨迹梯度被稀释。**

### 理由

原始 GRPO / DeepSeekMath 的目标通常是：

\[
\mathcal{L}_i = -\frac{1}{|o_i|} \sum_t \min(r_t, \text{clip}(r_t)) A_i
\]

即先对每个轨迹按 token 数做一次 sequence-level mean，再对 batch/group 做平均。

而当前代码是：

\[
\mathcal{L} = -\frac{\sum_i \sum_t \text{per\_tok}_{i,t} \cdot \text{mask}_{i,t}}{\sum_i \sum_t \text{mask}_{i,t}}
\]

这是 token-level mean，等价于“每个 token 的 loss 权重相同”。

问题在于：advantage 是 trajectory 级标量，轨迹内每个 token 共享同一个 advantage。于是：

- 长轨迹有很多 token；
- 每个 token 都贡献一份梯度；
- 最终该轨迹的梯度贡献正比于长度。

例如：

- 轨迹 A：100 token，reward 好；
- 轨迹 B：10 token，reward 也好；

当前实现下，A 的梯度贡献大约是 B 的 10 倍。这个结果不一定是“最优”，尤其在 ALFWorld 这种稀疏 reward、长度差异较大的多轮 agent 任务中，会让长度成为隐式偏置。

### 按轨迹数平均会怎样？

如果改成：

```python
per_seq_loss = (per_tok * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
loss = per_seq_loss.mean()
```

则：

- 每条轨迹同等权重；
- 短轨迹不会被长轨迹稀释；
- 单个 token 的有效梯度量级约等于 `1 / len_i`；
- 通常更适合 trajectory-level advantage + sparse reward。

代价是：如果某条很短轨迹偶然获得高 reward，它内部少数 token 会获得较大梯度，可能造成一次比较尖锐的更新。这个可以通过 group advantage normalization、clip、grad clip 缓解。

### 当前 `denom=mask.sum()` 会不会让长轨迹 dominate？

**会。**  
如果外层 `bt.n_active` 指的是 token 数，那么整体实现就是标准的 token-level mean；如果 `n_active` 是轨迹数，那会有混合归一化，行为可能更奇怪，但核心风险仍然是：`per-batch loss` 已经被 token 数归一化，再用轨迹数加权，不是标准 GRPO sequence-level 目标。

### 建议改法

推荐改成 per-sequence mean：

```python
def grpo_loss(model, batch, old_logps, clip_eps):
    new_logps = per_token_logps(
        model,
        batch.input_ids,
        batch.attention_mask,
    )

    # logp[:, i] 预测 input_ids[:, i+1]
    mask = batch.loss_mask[:, 1:]

    ratio = torch.exp(new_logps - old_logps)
    adv = batch.advantages.unsqueeze(1)

    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    per_tok = -torch.min(surr1, surr2)

    # per-sequence normalization，和 GRPO 原始形式一致
    seq_len = mask.sum(dim=1).clamp(min=1.0)
    per_seq_loss = (per_tok * mask).sum(dim=1) / seq_len

    return per_seq_loss.mean(), stats
```

外层如果不是 token-level weighting，最好也改成按轨迹数平均：

```python
loss = per_seq_loss.mean()
(loss / n_micro).backward()
```

或者如果保留 micro-batch 权重，就用 `bt.n_traj / total_traj`，不要用 token 数权重。

### Dr.GRPO 相关讨论

Dr.GRPO 通常强调 GRPO 中的 length bias：当 advantage 是 trajectory 级标量，且 loss 采用 token-level mean 时，会自然放大长序列影响。修复思路一般就是：

- 使用 sequence-level normalization；
- 或在 reward 中显式加长度惩罚；
- 或在 advantage 标准化时对长度做去偏。

对于 ALFWorld 这种多轮 agent 任务，我更建议 **per-sequence mean + reward 长度惩罚**，而不是当前的全局 token-level mean。

---

## 问题 4：beta=0，没有 KL penalty / reference model，会不会崩？

### 结论

**风险很高，不建议长时域多轮 agent 任务里长期 beta=0。**  
GRPO 的 clip 只能限制每一轮更新相对旧策略的局部变化，不能阻止长期训练后 policy 严重偏移 base model。

### 具体风险

#### 1. Reward hacking

ALFWorld 如果 reward 设计不是纯稀疏任务成功，而有部分探索/进度 reward，模型可能学会：

- 反复 `look` / `examine`；
- 重复有效但无意义的动作；
- 利用环境解析漏洞制造高分轨迹。

没有 KL 时，policy 会无约束地放大这些行为。

#### 2. Entropy collapse / 生成退化

4B 小模型 + 离散动作空间 + 模板化文本，很容易在 GRPO 更新后变成：

- 动作分布极端；
- 只输出少数高频动作；
- 探索消失；
- 后续 rollout 多样性极低。

尤其在 ALFWorld 这种多轮任务里，熵坍缩后很难恢复。

#### 3. 语言能力退化

没有 KL，模型会逐渐偏离 base Qwen3 的语言分布。可能表现为：

- chat/function calling 格式错误；
- action text 无法被环境解析；
- 生成内容语法破碎。

虽然 parser reward 会惩罚无效动作，但 RL 通常会先经历一段格式退化区，可能直接导致训练崩掉。

#### 4. Length explosion

结合问题 2 的 token-level mean，长度膨胀风险被进一步放大。模型可能学到：

- 增加冗余 action；
- 延长 trajectory；
- 不必要地增加 turn 数。

如果 reward 没严格惩罚长度，no KL + token-level mean 会促进隐式长度偏差。

### 是否需要加 KL？

**建议加，尤其你是全参微调 Qwen3-4B，且没有 PEFT 约束。**

推荐形式：token-level KL to reference policy。

参考 KL 可以用 k3 estimator：

```python
# ref_logps 预先 no_grad 算好
kl = torch.exp(ref_logps - new_logps) - (ref_logps - new_logps) - 1.0
per_tok = -torch.min(surr1, surr2) + beta * kl
```

完整 loss 形如：

```python
def grpo_loss_with_kl(
    model,
    batch,
    old_logps,
    ref_logps,
    clip_eps,
    beta,
):
    new_logps = per_token_logps(
        model,
        batch.input_ids,
        batch.attention_mask,
    )

    mask = batch.loss_mask[:, 1:]

    ratio = torch.exp(new_logps - old_logps)
    adv = batch.advantages.unsqueeze(1)

    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    reward_term = -torch.min(surr1, surr2)

    # KL(current || ref) k3 estimator，逐 token
    kl = torch.exp(ref_logps - new_logps) - (ref_logps - new_logps) - 1.0

    per_tok = reward_term + beta * kl

    seq_len = mask.sum(dim=1).clamp(min=1.0)
    per_seq_loss = (per_tok * mask).sum(dim=1) / seq_len

    return per_seq_loss.mean(), {}
```

其中 `ref_logps` 可以这样预计算：

```python
with torch.no_grad():
    ref_logps = per_token_logps(
        ref_model,
        batch.input_ids,
        batch.attention_mask,
    )
```

### beta 建议

从低开始：

```text
beta = 0.01 ~ 0.05
```

具体看任务：

- ALFWorld 动作空间小，beta 可以稍低；
- 如果发现 entropy 快速下降、eval success 下降、KL 快速上升，提高 beta；
- 如果训练太慢，可以适当降低 beta。

### 如果不想加 reference model，至少加什么？

如果显存吃紧，至少要加以下之一：

1. **KL to old policy**  
   用 `old_logps` 做弱 KL：
   ```python
   kl_old = new_logps - old_logps
   per_tok += beta_old * kl_old
   ```
   这只限制单步更新，不限制长期漂移，但比完全没有好。

2. **Entropy regularization**
   ```python
   log_probs = logits.float().log_softmax(dim=-1)
   entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
   per_tok -= entropy_coef * entropy
   ```

3. **Reward 长度惩罚**
   ```text
   shaped_reward = success_reward - lambda * step_count
   ```

但这些不能完全替代 reference KL。

### 参考模型成本

以 Qwen3-4B 为例：

- 参考模型权重：约 8GB bf16/rank；
- 不参与 gradient，不需要 optimizer state；
- 计算上多一次 no-grad forward；
- 如果 rollout 数据不变，`ref_logps` 可以在每个 iter 只算一次并缓存，后续 micro-batch 复用。

总结建议：

```text
beta=0 不建议用于长时域多轮 agent RL。
推荐：
- 加 reference KL，beta=0.01~0.05
- loss 改为 per-sequence mean
- reward 加长度惩罚
- 监控 entropy、KL、平均长度、eval success
```

这样在 ALFWorld × Qwen3-4B 这种设置下会稳得多。
