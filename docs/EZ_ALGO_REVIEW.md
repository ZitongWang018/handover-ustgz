# Explorer-Zero 算法正确性核查报告

日期：2026-10-12
触发：用户要求「核查，算法真的没问题吗，包括 smoke 的」「用不同的模型 review 代码」

> ⚠️ 这份文档记的是**"算法语义对不对"**，不是"能不能跑通"。
> "跑通"是 smoke 的事（见 `EZ_DDP_PLAN.md` §8），"跑对"是这份文档的事。
> 一个能跑通但 loss 归约口径写错的训练，会安静地训出一个偏的模型，
> 而且**所有 smoke 判据都会全绿** —— 这正是必须单独查一遍的原因。

---

## 0. 结论速览

| # | 问题 | 谁提的 | 核实 | 处置 |
|---|---|---|---|---|
| P0-1 | `grpo_loss` 用 **token 级**平均，非论文的**序列级** | 模型2 独立发现 | ✅ 属实（我复核+实测） | ✅ 已修，默认 `seq` |
| P0-2 | solver `rng.sample` 各 rank 采到**同一批任务** | 模型1 + 模型2 都提 | ✅ 属实（我先前已定位） | ✅ 已修（按 rank 分片） |
| P0-3 | solver `k_local` 未跨 rank 同步 → 可致 NCCL 死锁 | 模型2 | ⚠️ 理论风险 | ✅ 已加固 `all_reduce MIN` |
| P1-1 | `np.std` 是 **ddof=0**，论文/TRL 用 **ddof=1** | 模型1 + 模型2 都提 | ✅ 属实 | ✅ 已修（默认 ddof=1） |
| P1-2 | `old_logps` 在 `train()` 下算，dropout 会让 ratio≠1 | 模型2 | ❌ **不成立** | 记档，不动 |
| P1-3 | `kl_approx` 公式不是 KL | 模型2 | ❌ **不成立**（是标准 k3） | 加注释防误"修" |
| P1-4 | `kl_approx`/`clip_frac` 恒 0（ratio≡1） | 模型1 | ✅ 属实但**不是 bug** | 记档，不动 |
| P1-5 | solver `all_roll` 未按组标准化 → 违反红线 A | 模型2 | ❌ **不成立** | 记档，不动 |
| P1-6 | `n_active` 加权造成双重归一化 | 模型2 | ❌ **不成立** | 记档，不动 |

**一句话**：两个模型独立 review，**交叉验证出同一个真 bug（P0-2）**，
模型2 额外挖出**一条谁都没注意到的真数学偏差（P0-1）**。
同时它自己也误报了 4 条 —— review 结论必须逐条回源码核实，不能直接采信。

---

## 1. 核查方法

1. **两个不同的模型独立 review**（不看彼此的输出）：
   - 模型1：`deepseek-chat`（非推理型，快）
   - 模型2：`deepseek-reasoner`（R1 推理型）
   - 给两者的 spec 一致：贴带行号的权威代码片段 + 明确要求"给文件:行号 + 推理过程"
2. **我自己逐条回源码核实**：每条结论都必须能在源码里定位，不接受"可能""需要确认"
3. **真 bug 先写最小复现脚本量化**，再改，再钉成回归用例

> ⚠️ 踩到的坑：`deepseek-reasoner` 会先把 token 预算烧在思维链
> （`reasoning_content`）上，`max_tokens` 给小时正文是**空串**。
> `hpc_tools/ds_call.py` 已加思维链回退 + `--max-tokens 20000`。
> 另一坑：DeepSeek 会**幻觉代码上下文**（它给的 selftest 函数签名跟真实文件
> 完全不符），所以 spec 必须贴带行号的原文，且改动只能靠"OLD 唯一匹配"落盘。

---

## 2. 🔴 P0-1：`grpo_loss` 的归约口径是 token 级，不是论文的序列级

**位置**：`src/train_grpo.py` `grpo_loss()`（现抽到 `src/grpo_norm.py`）

**现象**：

```python
denom = mask.sum().clamp(min=1.0)          # 整个 batch 的参与 token 总数
loss  = (per_tok * mask).sum() / denom     # ← 组里 G 条序列的 token 混在一起平均
```

**为什么是错的**：论文（DeepSeekMath 公式 4）是

```
J = (1/G) Σ_i [ (1/|o_i|) Σ_t ( ... ) ]
```

先**每条序列**内部按 token 平均，再**组间等权**平均。
而现在写的是 `(Σ_i Σ_t) / (Σ_i |o_i|)` —— 等于**按长度加权**。

`ratio ≡ 1` 时 `per_tok = -adv_i` 在**序列内是常数**，于是两种口径退化成：

```
token 级： loss = - Σ_i |o_i|·adv_i / Σ_i |o_i|     ← 长度加权
序列级：   loss = - (1/G) Σ_i adv_i                  ← 组内等权
```

GRPO 归一化后 `Σ_i adv_i = 0`，所以**序列级下"组内奖励对称"必然给出净信号 0**
（这是 GRPO *应有*的性质）。而 token 级下**长度本身变成了信号**。

**实测**（`scripts/ez_repro_lossnorm.py`）：

| 场景 | 序列级净信号 | token 级净信号 |
|---|---|---|
| G=2，长度 20 vs 200，一好一坏 | `0.000` | `-0.818`（长序列占 91% 权重） |
| G=8，长度 30..300，4 好 4 坏 | `0.000` | `-0.577` |
| G=8 随机 2000 组，长度 20~300 | 恒 `0.000` | \|均值\| `0.149`，最大 `0.614` |

最后一行最关键：**组内奖励完全对称时，序列级本该给 0，token 级却稳定给出 ±0.15 的
更新方向，方向纯由长度分布决定。** 这就是 Dr.GRPO 点名的 length bias ——
模型会去学"写长一点 / 写短一点"，而不是"答对一点"。

**修法**：`src/grpo_norm.py::grpo_reduce`

```python
n_tok   = mask.sum(dim=1)                  # (n,)
seq_sum = (per_tok * mask).sum(dim=1)
ok      = n_tok > 0                        # mask 全 0 的行不能进平均，否则稀释
loss    = (seq_sum[ok] / n_tok[ok].clamp(min=1.0)).mean()
```

配置开关 `train.loss_norm: seq | token`（默认 `seq`），选 `token` 时 rank0 打 WARN。

**回归用例**（`scripts/ez_unit_test.py`，测梯度而非 loss 值）：

| 用例 | 判据 |
|---|---|
| 序列级两条序列梯度幅度相等 | `|g0| == |g1| == 0.5`（与长度无关） |
| token 级长度加权 | `|g_long|/|g_short| == 200/20 == 10` |
| 组内奖励对称 ⇒ 序列级净信号为 0 | `< 1e-6` |
| 同上，token 级必须非 0 | `> 0.5` |
| 长度全相等时两口径一致 | `seq == token` |
| mask 全 0 的行拿不到梯度、不稀释有效行 | 权重仍是 `1/1` 而非 `1/3` |

> ⚠️ 为什么单独拆 `grpo_norm.py`：`grpo_loss` 在 `train_grpo.py` 里，而那个模块
> 顶层要 import `transformers` / `alfworld` / `rollout`，离线单测环境根本 import
> 不进来。以前只能复刻一份实现去测 —— **复刻的和跑的不是一个东西**，
> 改了跑的那份单测照样绿。抽出只依赖 torch 的纯函数后，单测测的就是线上跑的那份。

---

## 3. 🔴 P0-2：solver 的 `rng.sample` 各 rank 采到同一批任务

**位置**：`src/ez_train.py` `rng = random.Random(seed)`（行 530 附近）+ solver 采样

**现象**：`set_seed(seed, rank)` 里的 `seed + rank` 只改了**函数内的局部变量**，
改不到调用者作用域的 `seed`。所以 `rng = random.Random(seed)` 各 rank 状态**完全相同**，
而 `outer_pool` 经 all_gather 后也完全一致 ⇒ `rng.sample` 各 rank 采到同一批。

**实测**（`hpc_tools/ez_repro_rng.py`）：8 卡 k=1 时 rank1..7 全部采到同一个任务，
全局只用到 2 个不同任务，**有效 batch 缩水到 25%**，8 卡算力白扔 6/8。

**修法**（两条都做）：
1. `rng = random.Random(seed + rank)` —— 各 rank 随机流真正分开
2. 采样改成**按 rank 分片**，不再依赖 rng 状态同步：

```python
batch = [outer_pool[(rank + i * world) % len(outer_pool)] for i in range(k_local)]
```

**回归用例**：`world=2/4/8` 分片断言 8/8 个不同任务 + 各 rank 条数一致（红线 B），
外加两组对照用例把旧写法的退化行为钉死（防有人"优化"回 `rng.sample`）。

---

## 4. 🟡 P1-1：`group_advantages` 用 `np.std`（ddof=0），论文/TRL 用 ddof=1

**位置**：`src/reward.py` `group_advantages()`

`np.std` 默认是**总体标准差**（ddof=0），会**低估**组内离散度，从而**放大** advantage：

| G | ddof=0 相对 ddof=1 的放大 |
|---|---|
| 16 | `sqrt(16/15)` → **+3.3%** |
| 4 | `sqrt(4/3)` → **+15.5%** |
| 2 | `±1.000` vs `±0.707` → **+41%** |

TRL `GRPOTrainer` 用 `torch.std`（默认 `unbiased=True` 即 ddof=1），
DeepSeekMath 参考实现同样走无偏估计。**G 越小差得越离谱**，
而且放大后的幅度跟 `max_grad_norm` 的阈值也对不上。

**修法**：`group_advantages(..., ddof=1)`，配置 `train.adv_ddof: 1`（可回退 0）。

---

## 5. ❌ 被推翻的 review 结论（review 也要被 review）

这四条是模型2 提的，**回源码核实后不成立**。记在这里，免得下次有人（包括我）
又照着 review 去"修"出 bug。

### 5.1 `kl_approx` "不是 KL" —— **它就是标准 k3 估计**

代码：`((ratio - 1.0) - (new_logps - old_logps)).mean()`

令 `Δ = new - old`：

```
E_old[ exp(Δ) - 1 - Δ ]
  = Σ_old · (new/old)  -  1  -  E_old[Δ]
  = 1 - 1 - E_old[Δ]
  = -E_old[Δ]
  = KL(π_old ‖ π_new)          ✅
```

一阶项 `-Δ` 就是 k1 估计；k3 比 k1 多一个非负二阶项，**无偏且方差更低**。
"顺手改成 `(old - new).mean()`" 反而会退化成 k1。已在代码里加注释锁住。

### 5.2 `old_logps` 在 `train()` 下算 ⇒ dropout 让 ratio≠1 —— **本项目 dropout=0**

`LoraConfig(..., lora_dropout=0.0)`（`src/ez_train.py` 行 93），Qwen3 基座
attention_dropout 也是 0。所以 `train()`/`eval()` 前向结果一致，
`ratio` **严格等于 1**，`old_logps` 是安全的。

### 5.3 solver `all_roll` 未按组标准化 → 违反红线 A —— **已按 `task_id` 分组**

```python
for t in batch:
    grp = [r for r in all_roll if r.traj.task_id == t.meta["uid"]]
    advs = group_advantages([r.traj.reward for r in grp], [0] * len(grp), ...)
```

每个 task 独立成组、独立算 mean/std，红线 A（GRPO 组不可拆）成立。

### 5.4 `n_active` 加权造成双重归一化 —— **`n_active` 只进日志，不进 loss**

`make_batch` 返回的 `n_active` 只用于 `_diag` 诊断行（`adv_spread/n_active/...`）。
loss 缩放是 `loss / len(groups)`（proposer）和 `loss / len(batch)`（solver），
都是**组间等权**，没有按 token 数加权。

---

## 6. ⚪ 属实但不是 bug（记档，不改）

### 6.1 `ratio ≡ 1` ⇒ `clip_frac` 恒 0、`ratio_mean` 恒 1、clip 是死代码

单步 on-policy GRPO：采样完只更新一次，`old == 当前策略`，`ratio` 必然为 1。
**这是 GRPO 单步更新的正常性质，不是 bug。** 保留公式是为了将来做多 epoch
（µ 步小批量更新）时不用重写。`clip_eps=0.3` 目前不生效，属预期。

### 6.2 组内奖励全同分 ⇒ 梯度精确为 0，参数不动

`group_advantages` 里 `std < eps` 直接返回 0 ⇒ 优势全 0 ⇒ 梯度全 0。
**这是正确的**：组内无区分度就没有相对优势。这也是稀疏奖励下 `update_rel=0`
最常见的原因 —— 判"参数没动"要看**真实 backward grad_norm**，不能只看 `update_rel`
（详见 `EZ_DDP_PLAN.md` §0.5.2）。

---

## 7. 🟡 P0-3：solver `k_local` 加跨 rank 同步兜底

**位置**：`src/ez_train.py` solver 相位

原来只靠"池子已跨 rank 一致"这个**写在注释里的假设**。`k_local` 直接决定
backward 次数，次数不同 = NCCL 永久卡死（红线 B），而假设一旦破了，
要等 30 分钟 timeout 才炸，且日志里什么都看不出来。

已改为显式 `all_reduce(MIN)` 同步，并在不一致时打 WARN 指明"去查 all_gather"，
而不是让人去猜死锁原因。

---

## 8. 仍然欠着的复现偏差（未修，已知）

| 项 | 说明 |
|---|---|
| KL penalty β=0.001 | Table 3 有，主线 GRPO 内核无参考模型，暂未实现 |
| 多 epoch PPO 更新 | 现在每步只更新一次，`clip_eps` 不生效（见 §6.1） |
| `tasks_per_step=2` | 论文 batch/rollouts 编排未写死，取务实值（偏差 U5） |
| 评测组件 | valid_unseen ×4 待补 |
| 观测型子任务工件 | 待补 |

---

## 9. 回归基线（改完后）

| 测试 | 结果 |
|---|---|
| `scripts/ez_unit_test.py` | **81 passed** ✅（含新增 11 项归约口径 + 9 项多卡采样） |
| `scripts/unit_test.py` | 通过 ✅（含 ddof=1 新断言 + ddof=0 对照） |
| `scripts/ez_check_smoke_selftest.py` | **18/18 PASS** ✅ |
| 2 卡 debug smoke | job 12991093（本报告修订后重跑） |

> ⚠️ 改了 `loss_norm` 和 `adv_ddof` 会**改变 loss 值和梯度幅度**，
> 所以 smoke 必须重跑，不能拿旧基线判新代码。
