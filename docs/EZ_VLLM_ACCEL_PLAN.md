# EZ 工程加速方案：vLLM 异步 rollout 后端

> 目标：在不牺牲论文口径（M_P=M_S=25, 4 outer, G=16, G_P=8）的前提下，把 4 outer 从 ~17h 压进 10h 以内。
> 手段：vLLM 连续批处理 + 轨迹级异步调度 + 环境工作池 + 前缀缓存。
> 最后更新：2026-10-13

## 0. 为什么必须做工程（配置已经到顶了）

忠实配置单步 ~530s，拆开：

| 阶段 | 耗时 | 说明 |
|---|---|---|
| exploration（2 slot × 30 turns） | ~135s | DDP 把 G=16 分到 8 卡 → 每卡只有 2 条，GPU 严重闲置 |
| solve rate（1 任务 × G_P=8） | ~200s | 8 条 rollout，串行 30 turns |
| update | ~5s | LoRA 很小 |
| 其他（env reset / PDDL 编译 / 通信） | ~190s | |

**地板 = 30 个回合 × 每回合 ~5s × 2 个相位 ≈ 300s/步**。200 步 ⇒ ~17h。
唯一能破的办法：**让"生成"和"环境 step"真正重叠**（现在是一条轨迹内严格串行），
并把多条轨迹的生成合并成连续批（现在每卡只有 1~2 条）。

## 1. 目标架构

```
        ┌──────────────┐
        │  请求队列     │  (task_uid, rollout_id, policy_version, token_ids, env_state)
        └──────┬───────┘
               │  首次分配选空闲 GPU；后续轮次固定留在原 GPU（保 prefix cache）
     ┌─────────┼─────────┬─── ... ──┐
   vLLM0     vLLM1     vLLM2      vLLM7      ← 8×单卡实例，TP=1，enable_prefix_caching
     └─────────┴─────────┴─── ... ──┘
               │  生成完动作
        ┌──────▼───────┐
        │  环境工作池    │  N 个 worker（N ≪ 轨迹数），并行 reset/step
        └──────┬───────┘
               │  观测回填 → 该轨迹立刻再入队下一轮
        ┌──────▼───────┐
        │  轨迹完成收集  │  按 task_uid 攒满 G_P 条 → p̂ → proposer 奖励
        └──────────────┘
```

关键点（对齐 verl Agent Loop）：
1. **轨迹级独立推进**，不再"整批等齐"。
2. **同一条轨迹的首轮选空闲实例，后续固定在原实例** —— 这样 prefix cache 才命中。
3. **只提交真实请求**，绝不为了凑 batch 生成 padding（当前 solve rate 有补齐浪费）。
4. **输入输出都用 token id**，严禁重新套 chat template（会改变训练输入）。

## 2. 分阶段实施（每阶段都能独立验证、独立回退）

### P0 — 环境与 PoC（0.5 天）
- 独立 venv 装 vLLM（**不污染训练用的 `thinktool`**）：`~/ztwang/vllm_venv`。
- PoC：加载 `qwen3_4b`，用 vLLM offline `LLM.generate` 跑一批 96-token 生成，
  与 HF `generate` 对比吞吐；确认 A800 上单卡能吃满。
- 产出：`scripts/ez_vllm_poc.py` + 速度/显存数字。

### P1 — solve rate 接 vLLM（收益最大、最好隔离）
**为什么先做这段**：solver 在整个 proposer 相位**冻结**，权重不用每步刷新 —— 最适合常驻引擎 + 前缀缓存。
- 新增 `src/ez_vllm_policy.py`：实现与 `HFPolicy` 同接口的 vLLM 版
  （`initial_prompt_ids` / `generate_from_token_ids` / `assistant_end_ids` / `append_feedback_ids` /
  `next_generation_prefix_ids`），**内部走 vLLM 的 token-in/token-out**；
  停止条件 `tool_call` 需要在 vLLM 侧用 `stop`/自定义 `StoppingCriteria` 复刻
  （对应 `rollout._ToolCallStopCriteria`）。
- `ez_train.py` 的 solve rate 段：把 `run_ez_solver_rollouts(env_sr, policy, ...)` 换成
  轨迹级异步循环 + vLLM 后端；env 用工作池并行。
- **验收**：同一份 solver 权重、固定 32 个任务、每任务 8 条采样 ——
  对比"256 条轨迹全部完成的时间"与 HF 版；并核对 p̂ 与 HF 版一致（同权重同种子）。
- 风险：停止条件复刻不一致 → 生成长度分布变化 → p̂ 偏。必须逐条对齐验证。

### P2 — exploration + solver rollout 接入
- exploration（proposer）每个 step 后权重刷新，需要 **策略版本边界**：
  完成当前版本的请求 → 暂停收请求 → 训练更新 → 刷 adapter + 版本号 → 清失效缓存 → 恢复采样。
- vLLM 的 LoRA：用 `LoRARequest` 挂 adapter；每步 `wake/sleep` 或增量更新 adapter 权重。
- **验收**：训练序列（`build_token_sequence`）与采样历史在 vLLM 下仍逐 token 一致。

### P3 — 环境工作池 + 并发调优
- 把 env 状态数与进程数解耦：有限 worker 管 env 实例，限制同时执行的 action。
- 先分别实测 reset / step / facts 提取耗时，再定 worker 数（推理实例和 tokenizer 也要 CPU）。
- 调参：`max_num_seqs` 16/32/64、`max_num_batched_tokens` 8192/16384、
  chunked prefill 开、`enable_prefix_caching` 开；同时看**完成速度 / 缓存命中 / 抢占次数 / 显存峰值**。

### P4 — 推理与训练共用八卡
- rollout 阶段八卡全给推理；update 阶段（实测仅 ~30s）切回训练。
- vLLM `sleep` 释放权重与 KV，`wake` 恢复；训练显存预算要同时覆盖训练模型 + 推理权重 + KV + 峰值。

## 3. 验收口径（统一一份 solver 权重 + 固定任务集）

- **256 条有效轨迹全部完成的时间**（主指标）
- 有效轨迹数 / GPU 小时
- 补齐后丢弃的生成量（目标 = 0）
- 各卡等待差异、缓存抢占次数、权重切换开销
- 正确性：任务对应、停止条件、采样参数、奖励与 HF 版一致

## 4. 风险与回退

| 风险 | 说明 | 回退 |
|---|---|---|
| 停止条件不一致 | vLLM 侧 tool_call 停止与 HF 版有偏差 | P1 先做逐条对齐测试，不一致就不换 |
| LoRA 权重同步出错 | 采样策略 ≠ 训练策略 → on-policy 破坏（静默） | 每次刷新后校验 adapter 版本号 + 抽样比对 logits |
| 显存冲突 | 训练 + 推理共存 | P4 用 sleep/wake；先测显存预算 |
| 工期 | 全套 = 2~4 周 | 每阶段独立可用；P1 单独就能把 solve rate 这段提速 |

## 5. 与现有训练的关系

- 忠实版（TPS=1, M_P=25）**同时在跑**（job 13007666，~17h），作为可用基线。
- 工程版开发期间**不动**忠实版代码路径：新增 `ez_vllm_policy.py` / 调度器，
  用开关（如 `ZTWANG_ROLLOUT_BACKEND=vllm`）切换，默认仍是 HF。
