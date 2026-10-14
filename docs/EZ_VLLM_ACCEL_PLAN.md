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

### P0 — 环境与 PoC（✅ 已完成 2026-10-13）

**环境**（独立 venv `~/ztwang/vllm_venv`，不污染训练用的 `thinktool`）：

| 包 | 固定版本 | 说明 |
|---|---|---|
| vllm | **0.8.5** | ⚠️ 必须 pin；pip 默认装的 0.31 是 **CUDA 13** 版，A800 驱动 570 跑不起来 |
| torch | 2.6.0+cu124 | 匹配驱动 570（cu13 要 ≥580） |
| transformers | **4.51.3** | ⚠️ 必须 pin；vllm 0.8.5 对 transformers 没上界 → pip 解析到 5.19 → tokenizer API 崩 |
| tokenizers | 0.21.4 | |
| xformers | 0.0.29.post2 | |

**PoC 结果**（`scripts/ez_vllm_poc.py`，job 13008318 @ A40 gpu3-9）：

- 模型加载 163.6s
- **32 seqs × 96 tokens 一次生成 = 1.95s，吞吐 1576 tok/s**（prefix caching 开）
- 对比 HF 版：32 slot 每回合 ~9.7s ⇒ vLLM 单次批量生成快 **5~10×**

**结论**：vLLM 0.8.5 在二期可用；token-in/token-out（`prompt_token_ids` 进、`outputs[0].token_ids` 出）
走的通，与 EZ 训练序列"assistant 段用真实生成 token id"的要求一致。

### P1 — solve rate 接 vLLM（✅ 已完成 2026-10-14）
**为什么先做这段**：solver 在整个 proposer 相位**冻结**，权重不用每步刷新 —— 最适合常驻引擎 + 前缀缓存。

- 新增 `src/ez_vllm_policy.py`：`VLLMPolicy` **继承** `HFPolicy`，分词面
  （`initial_prompt_ids` / `append_feedback_ids` / `assistant_end_ids` /
  `next_generation_prefix_ids` / `truncate_messages`）**一行都不重写** —— 两个后端给出的
  prompt token 序列逐字节相同。被换掉的只有 `_generate_batch` 那一层。
- **停止条件不进 vLLM，改成"生成满 + 事后精确截断"**：EZ 的判据是"括号配平的裸 JSON"
  （`scan_bare_json_call`），vLLM 的 `stop=` 表达不了。可以这么做的依据：自回归采样
  **逐行独立**，HF 的 per-row StoppingCriteria 只是提前冻结某行、不改它已吐出的 token。
  等价的单测：`scripts/ez_vllm_stop_equiv.py` → **309/309 PASS**（9 个历史坑例 + 300 随机序列）。
  代价：多生成"本会提前停掉"的 token（实测填充浪费 ≈1.4x），换 10x 吞吐，划算。
- LoRA 热同步：`export_lora_adapter`（**注意 PEFT 0.18 会多套一层 `<adapter>/`**，
  函数负责探测真实路径）+ `lora_workdir`（按 rank 分目录、只留最近 2 份、写前清空）。

### P2 — exploration + solver rollout 接入（✅ 已完成 2026-10-14）
- `ez_train.py` 新增开关 `ZTWANG_ROLLOUT_BACKEND=hf|vllm`（默认 hf）+ `sync_rollout_adapter()`，
  在 **4 处相位边界**刷 LoRA：proposer 步内（每步权重都变）/ solve rate 前（→solver）/
  恢复（→proposer）/ solver 步内（每步权重都变）；另有 2 处评测点（`mid_eval_solver` 的 `sync_fn`）。
  漏刷 = 采样策略 ≠ 训练策略 = **on-policy 静默破坏**，所以每次刷完都打日志。
- **发射方式必须换（关键坑）**：`torchrun --nproc_per_node=8` 让 8 个 rank 都看得到 8 张卡，
  而 vLLM 的 `LLM()` **没有 device 参数**，只看 `CUDA_VISIBLE_DEVICES` ⇒ 8 个引擎全挤 GPU0。
  "先 `torch.cuda.set_device(local)` 再改环境变量"**必崩**（torch 设备缓存已按 8 卡建好，
  `IndexError` @ `_dynamo/device_interface.py`）。最终方案：
  ```bash
  CUDA_VISIBLE_DEVICES=$i RANK=$i WORLD_SIZE=8 LOCAL_RANK=0 python src/ez_train.py ...
  ```
  **零 Python 代码改动**（`dist_info()` 本来就只读环境变量），已验收 8 进程并发起引擎无 OOM。
- 验收（`scripts/ez_vllm_real_bench.py`，真实 ALFWorld 环境）：

| 环节 | HF（旧） | vLLM（新） | 倍数 |
|---|---|---|---|
| proposer 探索 16 slot × 30 回合 | 787.8s | **92.2s** | 8.6x |
| solve rate 8 条轨迹 | ~250s | **14.6s** | 17x |
| 纯环境步进（对照，8 进程并发 vs 单进程） | 25.6s | 31.2s | 1.22x（几乎不争抢）|

  第三行说明：**环境不是瓶颈**，GPU 生成才是。

### P3 — 环境工作池 + 并发调优（结论：不需要）
实测环境步进在 8 进程并发下只慢 1.22x（11.5ms/slot-turn），且 env reset 只占 ~1s/步。
**P3 原计划的工作池/解耦没有必要**，跳过。

### P4 — 推理与训练共用八卡（结论：不需要 sleep/wake）
训练侧实测显存峰值 **10.5~11.9GB**（bf16 基座 8G + fp32 双 adapter + 梯度 + AdamW），
vLLM 取 `gpu_memory_utilization=0.40`（≈32GB）⇒ 合计 ~45GB / 80GB，**直接共存即可**，
不需要 vLLM 的 sleep/wake。

### 端到端实测（job 13026569，TPS=1 / 4 outer / M_P=M_S=25 / G=16 / G_P=8）
| 环节 | 实测 |
|---|---|
| proposer 探索（2 slot × 29 回合） | 34~43s |
| solve rate（1 任务 × G_P=8） | 21~25s |
| vLLM LoRA 热刷 | 0.3~0.5s/次（每步 2 次） |
| `all_gather_object` + 等最慢 rank | ~30s |
| backward | 1.7~4.3s |
| **P 步合计** | **77~112s**（对比旧 HF+TSP=8 的探索 787.8s + 250s×N） |

投影 **4 outer ≈ 4~5h**，达成「≤10h」目标。

> ⚠️ 一个容易搞错的前提：现在 `ez_qwen3_4b.yaml` 是 `tasks_per_step: 1`，且代码里
> **已经没有**"TPS 自动上调到 world"的逻辑。所以每步只有 **1 个全局 GRPO 组 = 16 条 rollout**，
> rank 分片后每 rank 2 条。旧作业日志里的「TPS=8」是**那份配置 + 那份代码**的产物，
> 意味着 8 倍数据量 —— 旧作业慢是两个因素叠加（HF 采样慢 + 8 倍数据）。

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
