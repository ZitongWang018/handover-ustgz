# DeepSeek 审查 - 03_reward

结论：**确实是把“惩罚项”写成了“奖励项”，符号错误。** 并且 `/ max_steps` 的归一化不是“步内比例归一化”，而是“固定最大预算分数归一化”，在短轨迹上会稀释过程项影响。

---

## 1. 符号检查：三个 `penalty` 全部写反

当前代码：

```python
r += c.invalid_action_penalty * (stats.n_invalid / denom)
r += c.format_violation_penalty * (stats.n_format_violation / denom)
r += c.repeat_penalty * (stats.n_repeat / denom)
```

配置里：

```yaml
invalid_action_penalty: 0.10
format_violation_penalty: 0.10
repeat_penalty: 0.05
```

这些数是正数，但名字是 `penalty`。  
所以更 invalid、更多 format violation、更多 repeat，奖励反而**增加**。  
这与文档里“全程非法动作 ≈ -0.20”完全相反。

例如全程非法动作，如果 `n_invalid=30`、`denom=30`：

当前代码：

```python
r += 0.10 * (30/30)  # +0.10
```

如果同时 `n_format_violation=30`：

```python
r += 0.10 * (30/30)  # +0.10
```

如果还有 repeat：

```python
r += 0.05 * (30/30)  # +0.05
```

结果失败轨迹可能得到 +0.20 ~ +0.25 的正奖励，而不是负奖励。

---

### 正确写法

如果配置里保持这些系数为正的“惩罚幅度”，那么应该用减法：

```python
r = c.success * float(stats.won)

# 惩罚项：正系数，应该减
r -= c.invalid_action_penalty * (stats.n_invalid / denom)
r -= c.format_violation_penalty * (stats.n_format_violation / denom)
r -= c.repeat_penalty * (stats.n_repeat / denom)

# 奖励项：正系数，应该加
r += c.valid_step_bonus * (stats.n_valid_steps / denom)

if c.admissible_ratio_bonus and stats.n_steps:
    r += c.admissible_ratio_bonus * (stats.n_valid_steps / stats.n_steps)

if stats.won:
    r += c.efficiency_bonus * max(0.0, 1.0 - stats.n_steps / denom)

# step_penalty 如果非 0，也应该是惩罚，应该减
r -= c.step_penalty * (stats.n_steps / denom)

r *= c.task_type_weights.get(stats.task_type, 1.0)
return float(np.clip(r, c.clip_min, c.clip_max))
```

或者，如果你坚持用 `r += c.invalid_action_penalty * ...`，那配置里的 `invalid_action_penalty`、`format_violation_penalty`、`repeat_penalty` 必须改成负数，例如 `-0.10`、`-0.10`、`-0.05`。  
但通常工程上更推荐：**配置保持正数表示惩罚大小，代码里减法处理。**

---

## 2. 这个符号 bug 在 GRPO 里的具体后果

GRPO 的 advantage 是组内标准化：

```python
A_i = (R_i - mean_g) / (std_g + eps)
```

注意，这里能抵消掉的是**组内所有样本共有的常数偏移**，但不会抵消“与 invalid/format/repeat 计数成正比”的错误奖励。

现在 reward 变成：

```python
R ≈ success * 1.0
    + 0.10 * invalid_ratio_to_max
    + 0.10 * format_ratio_to_max
    + 0.05 * repeat_ratio_to_max
    + ...
```

这会导致：

### 2.1 失败样本中获得正向 advantage

早期 ALFWorld 成功率通常很低，很多 GRPO group 里可能一个成功样本都没有。  
这时 reward 主要由过程项决定。

如果某个轨迹 invalid 多、格式错误多、重复多，它反而会拿到更高 reward。  
在 `group_advantages` 后会成为正 advantage，策略会被强化成：

- 输出更多非法动作
- 输出更多格式违规
- 输出更多重复动作

也就是说，**模型会学到用这些错误行为刷 reward**，而不是完成任务。

### 2.2 成功样本会部分掩盖 bug

当组内出现成功样本时，`success=1.0` 会主导 reward，因为过程项最大也就 ±0.1 量级。  
所以有成功样本时，成功轨迹大概率仍然是最高 reward。

但在大量失败样本之间，错误排序仍然存在。  
更 invalid 的失败样本可能比更接近成功的失败样本获得更高 reward，这会给策略一个非常强的错误梯度。

### 2.3 与 `group_advantages` 的交互

`group_advantages` 会减均值，但不会消除这种错误，因为这不是一个组内公共常数偏移，而是和每个轨迹的 `n_invalid`、`n_format_violation`、`n_repeat` 相关。

所以结论是：  
**这个 bug 会系统性地把“非法动作”强化成正行为，尤其在早期成功样本稀疏时，可能直接导致策略崩溃或收敛到退化策略。**

---

## 3. `max_steps` 归一化是否合理？

### 3.1 数学上是什么

现在每个过程项是：

```python
coef * (count / max_steps)
```

这不是“实际轨迹内的比例”：

```python
count / n_steps
```

而是“相对最大步数预算的比例”。

所以它衡量的不是：

> “这个轨迹里非法动作占实际动作的比例是多少？”

而是：

> “这个轨迹里非法动作占最大允许步数的比例是多少？”

这两者语义不同。

---

### 3.2 过程项被压到 ±0.1 量级是否合理？

如果这是**轻量 shaping**，作为成功奖励的 tie breaker，那还算合理。  
因为：

```python
success = 1.0
invalid_penalty_max = -0.10
format_penalty_max = -0.10
repeat_penalty_max = -0.05
```

成功项仍然占据绝对主导，过程项不会压过成功项。  
这可以避免模型为了过程奖励而牺牲成功率，尤其是 dense reward 过强时容易出现的 reward hacking。

但代价是：

- 早期没有成功样本时，过程项是唯一学习信号。
- 这些过程项被压得很小，导致失败样本之间的 reward 差异很小。
- 例如 0 次 invalid 和 10 次 invalid 之间的差异只有：

```python
0.10 * (10 / 30) = 0.033
```

这个差异在原始 reward 空间很小。

不过注意，`group_advantages` 会除以组内 std，所以如果组内方差也很小，最终 advantage 可能仍会被放大。  
但原始 reward 的尺度仍然会影响和 `success=1.0` 的相对关系。

所以：

> 作为轻量 shaping：合理。  
> 作为强过程约束：太弱。

---

### 3.3 早期 `invalid_rate` 接近 1.0 时会发生什么？

如果 `invalid_rate = n_invalid / n_steps` 接近 1.0，但 `n_steps` 不一定等于 `max_steps`。

当前代码用的是：

```python
n_invalid / max_steps
```

假设 `n_steps=10`，这 10 步全是非法动作：

```python
invalid_rate = 10 / 10 = 1.0
```

但 reward 中非法动作项只有：

```python
0.10 * (10 / 30) = 0.033
```

所以从“实际非法率”角度看，它被严重低估了。

而在当前符号错误下，这些非法动作还会得到**正奖励**。  
所以早期 `invalid_rate` 接近 1.0 时，失败的轨迹可能得到 +0.033 ~ +0.10 甚至更高的正过程奖励。  
在无成功样本的 group 里，这些轨迹会变成正 advantage，进一步强化非法动作。

如果符号修复后，非法动作惩罚依然被 `max_steps` 稀释。  
早期 `invalid_rate≈1.0` 且轨迹较短时，惩罚会很弱，不能有效压制非法行为。

---

### 3.4 除以 `max_steps` 而不是 `n_steps` 是否低估短轨迹过程项？

**会低估，尤其是短轨迹。**

例如：

#### 例 1：1 步轨迹，1 次非法

当前：

```python
0.10 * (1 / 30) = 0.00333
```

如果用实际步数：

```python
0.10 * (1 / 1) = 0.10
```

两者相差 30 倍。

#### 例 2：3 步轨迹，3 次非法

当前：

```python
0.10 * (3 / 30) = 0.01
```

如果用实际步数：

```python
0.10 * (3 / 3) = 0.10
```

相差 10 倍。

所以如果目标是想惩罚“非法动作占比”，那 `/max_steps` 是错误分母。  
正确应该类似：

```python
denom = max(1, stats.n_steps)

r -= c.invalid_action_penalty * (stats.n_invalid / denom)
```

但这样也有代价：

- 非常短的轨迹，单次非法动作会产生很大惩罚。
- 例如 1 步轨迹，1 次非法就是 `-0.10`，和 30 步全非法一样大。
- 这更偏向“比例惩罚”，而不是“绝对数量惩罚”。

所以两种选择其实是不同语义：

| 分母 | 语义 | 特点 |
|---|---|---|
| `max_steps` | 相对最大预算的绝对数量 | 短轨迹过程项被稀释；成功项主导 |
| `n_steps` | 实际轨迹内比例 | 短轨迹过程项更明显；但短轨迹方差可能增大 |
| 不用归一化，直接 `sum_t r_t` | 每步 dense reward | 最直接，但长度会影响总奖励，容易主导 success |

---

## 4. 总体评价

这个奖励设计有四个核心问题：

1. **符号错误**：`invalid_action_penalty`、`format_violation_penalty`、`repeat_penalty` 被加而不是减。
2. **语义错位**：文档说的是“惩罚”，代码给的是“奖励”。
3. **归一化分母不是实际步数**：如果目标是衡量非法率，`/max_steps` 会严重低估短轨迹过程项。
4. **过程项与 success 的尺度差异过大**：作为轻量 shaping 可以接受，但早期成功稀疏时，过程项可能太弱；而当前符号错误又让这些弱信号方向相反，进一步加剧训练崩溃风险。

修复顺序应该是：

```text
先修符号 → 再决定过程项归一化语义 → 再调过程项系数尺度。
```

否则在符号错误的情况下讨论 `/max_steps` 是否合理，意义有限。
