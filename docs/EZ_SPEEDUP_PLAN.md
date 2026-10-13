# Explorer-Zero 训练加速方案

> 目标：把 4 outer（200 步）从实测 ~41 小时压下来。
> 最后更新：2026-10-13

## 0. 现状实测（job 13003615，二期 8×A800）

| 指标 | 值 |
|---|---|
| 稳定单步（P1） | **~735s** |
| exploration（proposer 探索，2 条/rank） | ~130s |
| solve rate（[ez-solver] 8 rollouts） | ~200s |
| 真训练（backward+sync+optimizer） | **4.7s（0.6%）** |
| 其余（env reset / 通信 / persist 等） | ~400s |
| 峰值显存 | 10.3 GiB / 81.9 GiB（**13%**） |

**核心事实：99% 时间在 rollout（采样 + 等环境），真正的梯度计算只有 4.7s。**
而 rollout 里，GPU generate 只占一小半，大头是 ALFWorld 环境的开销
（PDDL 编译 + 每轮 env.step，全是 CPU 串行）。

## 1. 两个「加速 idea」的真相（先纠正认知）

### idea ②「环境异步」—— 其实已经做了

`ALFWorldBatchEnv` 内部就是 textworld 的 `AsyncBatchEnv`（`asynchronous=True`），
每个 slot 一个独立子进程，env.step 已经是并行的。

**真正缺的不是「异步」，是「并行度」**：每 rank 只有 1~2 条 rollout
（G_local=2），env 子进程并行度 = 2，GPU generate 的 batch = 2。batch 这么小，
GPU 和 env 都在「假忙」。

### idea ①「vLLM」—— 收益被高估了，且有硬伤

vLLM 只加速「generate」这一环，而 generate 只占 rollout 的一小半。
更致命的是：我们是**双 LoRA adapter（proposer/solver）+ 每个 step 都更新权重**，
vLLM 要支持「训练中动态热更新 LoRA + 双 adapter 切换」，这是 veRL/OpenRLHF
这类框架的**核心工程**，不是换一行 `model.generate` 能解决的。

## 2. 真正的瓶颈排序

1. **ALFWorld 环境硬开销（PDDL 编译 + env.step）** —— CPU 串行，GPU 帮不上，**是地板**。
2. **并行度太低**（每 rank 1~2 条 rollout）—— batch 太小，GPU/env 都空转。
3. **generate 慢**（HF 小 batch 生成）—— vLLM 能治，但收益受 #1 限制。

## 3. 方案（按性价比）

### 方案 A：并行化 exploration（收益最大，中风险）

**现状**（`ez_train.py:858`）：TPS 个 game 的 exploration 是 `for gf in games` **串行**。

```python
for gf in local_games:
    _one = run_proposer_exploration(env_p, policy, gf, p_cfg, log_fn=log)
```

**改法**：把 `run_proposer_exploration` 扩成批量版（一次 reset 多个 game，
每个 game 占 G_local 个 slot），配合 `tasks_per_step` 提到 8，让 8 个 game 的
exploration 一起跑。

- **收益**：exploration 阶段 8×（130s → ~20s），solve rate 阶段同理。
- **风险**：改的是 exploration 核心循环，slot 对齐 / fingerprint 分组 / 空组守卫
  都要重新验证。历史上这里踩过「textworld 打乱 slot」「空组 DDP 死锁」的坑。
- **工作量**：0.5~1 天 + 充分验证。

### 方案 B：vLLM 换 generate（收益中等，高风险，大工程）

分三步，任何一步都别跳过验证：

1. **只读推理**：先让 vLLM 单独 serve 一个冻结的 solver adapter，验证
   `stop_mode="tool_call"` 的停止条件、左 padding / gen_start 语义在 vLLM 里能对齐。
2. **LoRA 热同步**：每个训练 step 后把更新过的 LoRA 权重同步给 vLLM
   （vLLM 有 lora 热加载 API，但 4B + 双 adapter 每步 reload 的开销要实测）。
3. **双 adapter 切换**：proposer/solver 相位切换时在 vLLM 里切换 LoRA。

- **收益**：generate 快 5~20×，但**只作用于 rollout 里的 generate 部分**，
  折算到总时长约 1.5~2×（因为环境硬开销 #1 不变）。
- **风险**：最高。双 LoRA 动态同步是框架级工程，容易引入「采样分布和训练分布
  不一致」的静默 bug（on-policy RL 最怕这个）。
- **工作量**：2~4 周。

### 方案 C：已落地（低风险）

`generation_max_batch_size: 2 → 8`：solve rate 的 8 条 rollout 从「4 批 generate」
变「1 批」，省 ~75s/步（~10%）。

## 4. 现实天花板（必须认清）

即使 A + B 全做完，**每 outer 仍有不可压缩的环境硬开销**：

- ALFWorld 的 PDDL 编译 + 每轮 env.step 是 CPU 的 fast_downward 规划器，
  GPU 加速不了，只能靠多核并行摊薄。
- 二期单机 64 核，环境并行度上限约 16~32，这是地板。

**务实预期**：A 落地 → 4 outer 从 41h 降到 ~8~12h；A+B 全做 → ~4~6h。
论文那个「2h」是 H100 + 忽略环境开销的理想值，A800 + ALFWorld 现实到不了。

## 5. 分阶段推进建议

| 阶段 | 内容 | 预期 | 前置 |
|---|---|---|---|
| 0 | 方案 C（batch 8）| 已做，~10% | 无 |
| 1 | 方案 A（并行化 exploration）| 41h→~10h | 训练结束后 + 有空闲卡验证 |
| 2 | 方案 B 第 1 步（vLLM 只读推理）| 验证可行性 | 阶段 1 后 |
| 3 | 方案 B 第 2/3 步（LoRA 同步）| 再 ~2× | 阶段 2 通过 |
