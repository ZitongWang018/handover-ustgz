# DeepSeek 审查 - 01_rollout

## 问题 5

结论：**非空 `gen` 的主路径没有 off-by-one；两个切片起点正确。`rlen` 推进使用 `full_r` 长度不会导致后续轮次错位。但 `gen` 为空时的 `else` 分支有一个真实 bug。**

### 为什么主路径切片正确

设处理 assistant 消息前的 render 长度为 `R`，当前 assistant header 为 `H`，模板收尾（`<|im_end|>\n` 等）为 `T`，当前 assistant 真实生成 token 为 `G`，当前 assistant 渲染文本 token 为 `C`。

处理 assistant 时：

- `prefix_r = render(messages[:i], add_generation_prompt=True)`  
  结构为：`previous_render + H`，长度约 `R + len(H)`。
- 所以：
  ```python
  gen_prompt_tokens = prefix_r[rlen:]   # rlen == R
  ```
  切出的就是 `H`，正确。

- `empty_full = render(messages[:i] + [{"role":"assistant","content":""}], add_generation_prompt=False)`  
  结构为：`previous_render + H + T`。
- 所以：
  ```python
  tail = empty_full[len(prefix_r):]
  ```
  切出的就是 `T`，正确。

拼接后实际 token 序列是：

```text
previous_render + H + G + T
```

而 render 长度推进：

```python
rlen = len(full_r)   # full_r 结构为 previous_render + H + C + T
```

所以 `rlen` 是 **render 坐标系的长度**，不是实际 `ids` 的长度。当 `len(G) != len(C)` 时，`len(ids)` 和 `rlen` 确实会不一致。

但后续无论是：

```python
prefix_r[rlen:]
```

还是非 assistant 分支：

```python
cur[rlen:]
```

它们切的都是 **render 序列**，而不是实际 `ids`。`rlen` 始终表示“已处理消息的纯模板渲染长度”，所以后续从 render 中切下一段模板/观测时，起点仍然正确。实际 `ids` 长度不同只影响实际序列长度，不影响 render 切片位置。

因此主路径不会错位。`boundaries` 用 `len(ids)` 也是正确的。

### 真正的 bug：空 `gen` 分支

```python
else:
    seq = gen_prompt_tokens + full_r[len(prefix_r):]
    ids.extend(seq)
    mask.extend([0]*len(seq))
```

如果 `gen` 为空，但 `messages[i]["content"]` 非空，`full_r[len(prefix_r):]` 会包含 **真实 render 出的 assistant Content**，而模型实际没有生成这些 token。这会构造出一段模型并未生成的 assistant 内容。

空生成时，应和主路径一样使用 `empty_full` 的 tail，而不是 `full_r` 的 tail：

```python
seq = gen_prompt_tokens + tail
```

### 建议改进

可以显式维护 `render_len` 和 `actual_len`，并修掉空 `gen` 分支：

```python
def build_token_sequence(messages, gen_ids_per_assistant, policy):
    ids, mask, boundaries = [], [], []
    assistant_idx = 0
    render_len = 0
    actual_len = 0
    prev_render = []

    def _render_ids(msgs, gen_prompt=False):
        return policy.tokenize_text(policy.render(msgs, add_generation_prompt=gen_prompt))

    for i, msg in enumerate(messages):
        if msg["role"] == "assistant" and assistant_idx < len(gen_ids_per_assistant):
            gen = list(gen_ids_per_assistant[assistant_idx])
            assistant_idx += 1

            prefix_r = _render_ids(messages[:i], gen_prompt=True)
            if len(prefix_r) < render_len:
                raise RuntimeError("渲染长度回退")
            if prev_render and prefix_r[:len(prev_render)] != prev_render:
                raise RuntimeError("chat 模板不是前缀稳定的")

            header = prefix_r[render_len:]

            empty_r = _render_ids(
                messages[:i] + [{"role": "assistant", "content": ""}],
                gen_prompt=False,
            )
            if empty_r[:len(prefix_r)] != prefix_r:
                raise RuntimeError("空 assistant 渲染不以前缀开头")

            tail = empty_r[len(prefix_r):]
            full_r = _render_ids(messages[:i + 1], gen_prompt=False)

            if gen:
                ids.extend(header)
                mask.extend([0] * len(header))

                ids.extend(gen)
                mask.extend([1] * len(gen))

                ids.extend(tail)
                mask.extend([1] * len(tail))

                actual_len += len(header) + len(gen) + len(tail)
                boundaries.append(actual_len)
            else:
                # 修复：不要使用 full_r[len(prefix_r):]
                # 那会把未生成的 assistant Content 混进来
                seq = header + tail
                ids.extend(seq)
                mask.extend([0] * len(seq))
                actual_len += len(seq)

            prev_render = full_r
            render_len = len(full_r)

        else:
            cur = _render_ids(messages[:i + 1], gen_prompt=False)
            if len(cur) < render_len:
                raise RuntimeError("渲染长度回退")

            new = cur[render_len:]
            ids.extend(new)
            mask.extend([0] * len(new))
            actual_len += len(new)

            prev_render = cur
            render_len = len(cur)

    assert len(ids) == actual_len
    return ids, mask, boundaries
```

---

## 问题 6

结论：**这个修法是正确。对于左 padding 的 HF generate，应该用 `enc["input_ids"].shape[1]` 作为生成段起点，不能用 `attention_mask.sum()`。**

### 左 padding 下 `out[i]` 的布局

假设 batch 内有 3 条 prompt，长度分别为：

```text
L1 = 5
L2 = 7
L3 = 4
```

padding 后的 `input_ids` 长度为：

```text
L_pad = 7
```

左 padding 后每个样本为：

```text
sample1: [pad, pad, tok1, tok2, tok3, tok4, tok5]
sample2: [tok1, tok2, tok3, tok4, tok5, tok6, tok7]
sample3: [pad, pad, pad, tok1, tok2, tok3, tok4]
```

HF generate 会在整个序列末尾追加生成 token，因此 `out[i]` 布局为：

```text
out[1] = [pad, pad, prompt(5), gen...]
out[2] = [prompt(7), gen...]
out[3] = [pad, pad, pad, prompt(4), gen...]
```

生成 token 的起点是 padding 后的统一长度 `L_pad`，也就是：

```python
gen_start = enc["input_ids"].shape[1]
```

而 `attention_mask.sum()` 给出的是每条 prompt 的真实长度：

```text
[5, 7, 4]
```

如果用 `attention_mask.sum()` 去切：

```python
out[i, attention_mask.sum(i):]
```

对 sample1 会从 index 5 开始切，但 sample1 的 index 5 仍然是 prompt token，不是生成 token，所以会出错。

### 是否所有情况都成立？

- **batch 内长度相等**：此时没有 padding，`attention_mask.sum() == input_ids.shape[1]`，两个起点相同。
- **tokenizer 是否添加特殊 token**：这里 `add_special_tokens=False`，特殊 token 已经在 `render()` 中生成。即便将来改成 `add_special_tokens=True`，只要 padding side 仍是 `left`，padding 仍在最左侧，`input_ids.shape[1]` 依然包含“pad + 全部 prompt token”，生成起点仍在末尾，结论不变。
- **关键前提**：必须是左 padding。若是右 padding，`out[i]` 布局会不一样，这个写法不能直接使用。

### 更稳妥写法

可以写成：

```python
assert self.tokenizer.padding_side == "left"

enc = self.tokenizer(
    prompts,
    return_tensors="pt",
    padding=True,
    add_special_tokens=False,
).to(self.device)

prompt_lens = enc["attention_mask"].sum(dim=1)
gen_start = int(enc["input_ids"].shape[1])

...

for i in range(out.shape[0]):
    # 左 padding 下生成段起点统一是 padding 后的输入长度
    ids = out[i, gen_start:].tolist()
```

可以额外加一个 sanity check：

```python
pad_len = gen_start - prompt_lens[i].item()
assert enc["attention_mask"][i, :pad_len].sum() == 0
assert enc["attention_mask"][i, pad_len:].sum() == prompt_lens[i]
```

---

## 问题 7

结论：**在 DDP 8 卡下，如果所有 rank 都执行同一段生成逻辑，且 `finally` 中恢复 GC，当前做法基本安全，不会造成 DDP reducer hook 泄漏，也不会直接破坏梯度同步。但有几个工程风险，不推荐频繁 global toggle。**

### 为什么基本安全

1. **HF 的 gradient checkpointing enable/disable 并不会操作 DDP hook**  
   它通常只是设置模型内部的 `_use_gradient_checkpointing` / `_gradient_checkpointing_func`，不会注册或卸载 DDP 的 reducer autograd hook。DDP 的参数梯度 hook 是在构造 DDP 时安装的，和 GC 开关无关。

2. **生成期间在 `torch.no_grad()` 下进行**  
   没有 autograd graph，也没有 backward，因此 DDP reducer 不会参与，也不会有梯度 allreduce。生成期间 GC 是否开启不会影响 DDP 通信。

3. **rank 一致性由代码结构保证**  
   如果 8 个 rank 都运行同一个 `generate()`，都在进入 collective 前恢复 GC，则不会出现 rank 间 GC 状态不一致。

### 主要风险点

1. **`self.model` 可能是 DDP wrapper**  
   `self.model.gradient_checkpointing_disable()` 会通过 DDP 的 `__getattr__` 委托到内部 module。这个行为可用，但不够显式。更稳妥是直接操作底层 module：

   ```python
   base_model = self.model.module if hasattr(self.model, "module") else self.model
   ```

2. **`enable` 时硬编码了 `use_reentrant=False`**  
   如果原始 GC 配置不是这个，或者以后其他地方依赖原始 `_gradient_checkpointing_func`，会被覆盖。更安全是保存原始 GC 状态，甚至保存原 kwargs。

3. **生成时没调 `model.eval()`**  
   当前代码只在 `no_grad()` 下生成。如果模型仍处于 `training=True`，dropout 仍然会生效，会对 rollout 采样引入额外随机性。通常 rollout 应显式 `eval()`，之后恢复 `train()`。

4. **如果只有部分 rank 执行 rollout**  
   例如 eval 只在 rank 0 执行，其他 rank 不执行，那么 rank 0 短暂 disable/enable GC，其他 rank 不知道。只要 rank 0 在下一次 collective 前恢复，通常没问题；否则会导致状态不一致。DDP 下最好所有 rank 都执行相同 toggling，或只用 raw module 做生成。

### 更安全的替代方案

优先操作底层 raw module，并保存/恢复 `eval/train` 状态：

```python
def generate(self, messages_list, cfg):
    ...
    base_model = self.model.module if hasattr(self.model, "module") else self.model

    gc_on = getattr(base_model, "is_gradient_checkpointing", False)
    was_training = base_model.training

    if was_training:
        base_model.eval()

    if gc_on:
        base_model.gradient_checkpointing_disable()

    try:
        with torch.no_grad():
            out = base_model.generate(**enc, use_cache=True, **gen_kwargs)
    finally:
        if gc_on:
            base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if was_training:
            base_model.train()

    ...
```

如果你想完全避免生成期间开关 GC，可以牺牲速度，在 GC 开启时使用 `use_cache=False`：

```python
with torch.no_grad():
    out = base_model.generate(**enc, use_cache=False, **gen_kwargs)
```

这种写法不会修改 GC 状态，更“纯”，但长序列生成会慢很多。对于大规模 RL rollout，通常还是采用“生成时关 GC、生成后恢复”的做法，只要按上面的方式操作底层 module 并恢复状态即可。
