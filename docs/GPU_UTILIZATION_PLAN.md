# EZ 训练 GPU 利用率优化方案

> 现状：`ez_train.py` 是**单进程 + HF generate + 逐步同步**的可跑通版本。
> 目标：在显存不爆的前提下把 GPU 利用率从"很低"拉到"接近打满"。
> 最后更新：2026-10-12

## 0. 先诊断：现在 GPU 在浪费什么

一次 proposer 探索 = 30 轮循环，每轮 `policy.generate(batch=16~32) → env.step()`：

| 浪费点 | 量级 | 原因（代码位置） |
|---|---|---|
| **① 每轮重算整段历史 prefill** | **最大头，约占总算力 60-80%** | `rollout.HFPolicy.generate` 无 `past_key_values`，第 t 轮把前 t 轮的对话全部重新 encode 一遍 → 总 prefill 是 O(T²)；30 轮探索 = 30 倍重复 |
| **② GPU 等 env.step** | 每步几十 ms ~ 数百 ms 空转 | textworld `AsyncBatchEnv` 子进程间同步；`ez_proposer.py` 里生成/环境严格串行 |
| **③ 不同步的 slot 被 idle 拖累** | 探索后期明显 | 有的 slot 早 finish、有的还在走，HF batch 生成必须整批等最慢的；没有 continuous batching |
| **④ fp32 基座常驻** | 显存 -50% 的空间被白占 | `ez_qwen3_4b.yaml: dtype float32`（继承主线的 bf16 舍入教训）——但 **EZ 用 LoRA，那条教训不适用** |
| **⑤ 单进程** | 7/8 张卡空闲 | `ez_train.py` 明确 `world==1`，8 卡 DDP 是 TODO |
| **⑥ env 装载（PDDL 编译）反复发生** | 每步几次 reset | game 池没预热，reset 是主要 CPU 开销 |

## 1. 改造清单（按"收益/风险"排序）

### P0-A. 基座换 bf16（EZ 专用，**立即做，零风险**）
- **为什么安全**：主线的 fp32 铁律来自"AdamW 直接更新 bf16 基座权重被舍入抹掉"。EZ 里**基座冻结**，更新的只有 LoRA adapter（peft 默认 fp32 创建），adapter 更新量 ~1e-4 而 fp32 ulp ≈ 1e-7 → 永远不会被抹掉。
- **收益**：4B 基座 16G → 8G；省下的显存直接换成更大的 rollout batch / 更长上下文。
- **改法**：`configs/ez_qwen3_4b.yaml` 的 `model.dtype: bfloat16`；`build_dual_lora` 里显式 `LoraConfig(...)` 后用 `.to(torch.float32)` 保险（peft 默认已是 fp32）。
- ⚠️ 必须同时保留 `_autocast_bf16(cache_enabled=False)`（防权重缓存 +8G）。

### P0-B. KV cache 增量解码（**收益最大，需改 rollout 循环**）
- **做法**：给每个 slot 维护独立的 `past_key_values`；每轮只 encode **新增的观测 tokens**（几十个），`generate` 时传入 cache 继续。
  - transformers 4.57 的 `generate(..., past_key_values=cache)` 支持传 DynamicCache；
  - 每轮结束后把新生成的 token 追加进 cache，再 encode 新观测段。
- **显存账**（Qwen3-4B：36 层 / 8 KV heads / head_dim 128 / bf16）：
  每 token KV ≈ 2×36×8×128×2B ≈ **144 KB/token**
  - 8k 上下文 × 1 条 = 1.2 GB → **16 条 = 19 GB**，A800/H100 80G 完全放得下；
  - 若开到 32 条 = 38 GB，仍可，但要与训练显存分时复用。
- **收益**：生成算力砍 50-70%（30 轮探索从 30×prefill 变成 1×prefill + 29×decode）。
- **风险**：cache 与 `truncate_messages`（截历史）冲突——截断时必须**重建** cache。建议：EZ 里不用截断，直接按 max_context 控制轮数。

### P0-C. 替换 rollout 引擎为 vLLM/SGLang（**天花板最高，但有安装门**）
- **为什么值得**：continuous batching（闲置 slot 自动让位）、paged attention（显存利用率 ~90% vs HF 的 60%）、prefix caching（system prompt 共享）、**dynamic LoRA 热加载**（正好匹配 EZ 的双 adapter：切换 adapter 就是一次 API 调用，不用改权重）。
- **架构**：训练进程（HF + peft）与推理进程（vLLM）分离；每 K 步把 adapter 权重推给 vLLM（`load_lora`），rollout 全部走 vLLM 的 OpenAI 兼容接口。
- **收益**：吞吐 3-10×（经验值），配合 ① 后总提升 5-15×。
- ⚠️ **硬约束**：**三期完全无外网**（pypi/github 不可达），vLLM 及其 CUDA 依赖装不上；二期有网可装但 vLLM 0.8.5 与 transformers 4.57 不匹配（已记录）。落地路径：
  1. 先在二期/本地把 wheel 全量下载（`pip download vllm==<ver> -d wheels/`），核对 CUDA 版本；
  2. `tools/sync_ztwang.py` 推三期（wheel 目录几十 GB，SFTP 慢，一次性付出）；
  3. 或用 Apptainer/SIF 镜像（二期 `slimerl/slime:v0.3.0` 已在计划里）。
  → **结论：①/② 先做（纯代码），vLLM 作为第二阶段独立攻坚。**

### P1-D. 训练/rollout 流水线重叠
- 现在：proposer 探索完 → 估 p̂ → 训练，三段严格串行，GPU 在 env 阶段空转。
- 改法：异步队列。solver rollout 估 p̂（`env_sr`）与 proposer 的 GRPO 更新可并行——**估 p̂ 用旧的 solver 权重本来就是论文语义**（冻结 solver），所以可以放到后台 worker。
- 实现：`threading`/`multiprocessing` + 双缓冲，或直接上 Ray。收益：消除 ② 的空转，~15-30%。

### P1-E. 8 卡 DDP 分片（把 ⑤ 补上）
- EZ 结构天然适合分片：每个 rank 分到 `G/world` 条探索轨迹 + `G/world` 条 solver rollout，梯度 all-reduce。
- LoRA 只有 adapter 参与 all-reduce → **通信量极小**，DDP 足够，不需要 FSDP/ZeRO。
- 关键点（从主线踩过的坑抄）：
  - `model.to(device)` 必须在 DDP/分片之前（主线 job 12986001 的教训）；
  - 每个 micro-batch 用 `no_sync()`，只在最后一个同步；
  - `init_process_group(..., timeout=30min)`（主线已修）；
  - **任务池必须跨 rank 分片或去重**，别 8 张卡练同一批任务（EZ 特有：proposer 的 uid 生成要带 rank）。
- 期望：8 卡 ≈ 单卡时间 ÷ 6~7（环境侧 CPU 开销不完全可分摊）。

### P1-F. env 侧优化
- **game 池预热**：`ALFWorldBatchEnv` 建一次、所有相位复用（已是），但 `reset` 装载 PDDL 是大头 → 每个 game 用完不卸载（保持常驻 slot 绑定同一批 game），减少 reset 次数。
- **双缓冲**：生成进行时后台 reset 下一批（与 D 合并做）。
- `AsyncBatchEnv` slot 数：CPU 侧 96 核，可支撑比 GPU batch 更大的 env pool（1 个 slot 约 1 核），**batch 16 → 32 的瓶颈在显存不在 CPU**。

### P2-G. 其他
- `torch.compile(model)` 包 LoRA 后的 forward：训练步提速 ~10-20%（首次编译 2-5 分钟；注意 GC + compile 的兼容性，需 smoke）。
- micro-batch 从 1 提到 2-4（P0-A 省出的显存正好用在这）；`per_token_logps` 内部已按 chunk 算 logits，主要收益在通信/调度摊薄。
- `num_workers`/`OMP_NUM_THREADS`：8 卡时设 6-8，别让 env 子进程和训练抢核。
- proposer `max_new_tokens=160` 偏大且停止条件永远不触发（`Action:` 判据对 `<tool_call>` 无效）→ 加 `<tool_call>...</tool_call>` 级别的停止条件，**每轮省下半数生成 token**（review P2-14，改动小收益直接）。

## 2. 显存预算表（H100/A800 80G）

| 配置 | 基座 | 训练侧峰值 | rollout KV（16×8k） | 合计 | 可行性 |
|---|---|---|---|---|---|
| 现状 fp32 | 16 G | grad+Adam(fp32 adapter) ~2 G + logits chunk ~10-20 G | 19 G | **~50-60 G** | 能跑但紧 |
| **P0-A bf16** | 8 G | ~2 G + 10-20 G | 19 G | **~40-50 G** | ✅ 舒适 |
| + P0-B 增 cache 到 32 slot | 8 G | 同 | 38 G | ~60-70 G | ⚠️ 需分时复用 |
| + 训练/rollout 同卡共存 | 8 G | 同 | 视 batch | 需显存池化 | 推荐分离卡 |

## 3. 落地顺序（建议）

1. **P0-A（bf16）+ P2-G 停止条件** —— 半天，纯配置/小改，立即 -50% 显存 + 省一半生成 token。
2. **P0-B（KV cache 增量解码）** —— 1-2 天，`rollout.HFPolicy.generate` 的重构版（新增 `generate_cached`，不改主线）。
3. **P1-E（8 卡 DDP）** —— 1-2 天，README 里已有主线的全部踩坑笔记可抄。
4. **P1-D/F（重叠流水线 + env 双缓冲）** —— 2-3 天。
5. **P0-C（vLLM）** —— 独立攻坚：先解决离线 wheel/镜像，再改推理接口（`HFPolicy` 已有接口抽象，换实现即可）。

预期效果：1+2+3 做完，8 卡跑完论文 4 outer loops（200 步）从"单卡 25-50 小时"降到 **3-6 小时**量级；再加 4/5 可以压到 **1-2 小时**。
