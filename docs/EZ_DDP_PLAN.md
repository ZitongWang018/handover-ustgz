# Explorer-Zero 8 卡 DDP 方案

> 目标：把 `src/ez_train.py` 从"单进程 scaffold"改成能在 8×A800 / 8×H100 上跑，
> 且**算法语义与论文一致**（这是前提，速度是次要的）。
>
> 结论先说：EZ 的 DDP **不能照抄**主线 `train_grpo.py`。主线是"一条轨迹一个样本、
> 样本之间互相独立"，所以随便切；EZ 的 GRPO 组是**同一个 game 的 G 条 rollout**，
> 组是算法的最小不可分单元——切错地方不报错，但算出来的 advantage 是错的。

---

## 0.5 落地状态（2026-10-12 更新）

| 阶段 | 状态 | 证据 |
|---|---|---|
| 方案文档（本文件） | ✅ | — |
| P0 代码落地 | ✅ | `ez_train.py` / `train_grpo.py` / `ez_smoke_8gpu.sbatch` / `ez_check_smoke_ok.py` 关4 |
| 单卡冒烟硬通过 | ✅ | job `12989496`：`结论：**通过** ✅ 关0/1/2b/3 全过`，`update_rel=5.00e-01，252/504 张量真的动了` |
| P0 后单卡回归（确认没改坏 world==1） | ✅ | job `12989528`（`scripts/ez_smoke_1gpu.sbatch`）|
| 8 卡冒烟 | ⏳ 排队 | job `12989532` @ `i64m1tga800u8ka`（8 卡作业被策略限定只能投 800u8ka / a40u8ka）|

### 0.5.1 路上清掉的一个"假故障"（值得记住）

单卡冒烟曾长期打出这个指纹：

```
[outer 0][P0] 诊断 ... | 真实 backward grad_norm=6.1474e+00 (504 个张量有 grad)
[outer 0 P0] update_rel=0.000e+00（必须为 >>1e-10，否则梯度是死的）
```

"梯度明明非零，参数却一步不动"——看着像 `optimizer.step()` 被精度抹掉。**不是。**

真因在 `update_probe` 选张量的方式：旧实现取"**第一个** `numel>=4096` 的 LoRA 张量"，
按 `named_parameters()` 顺序恰好命中 `layers.0.self_attn.q_proj.lora_A.<adapter>`。
而 peft 把 **`lora_B` 初始化成全 0**：

```
y = W x + B(A x) · s      ⇒   ∂L/∂A = Bᵀ(∂L/∂y) xᵀ ≡ 0        （B == 0）
```

所以**该张量在第 0 步的梯度是精确的 0**（不是"很小"，是 0）。AdamW 里 `exp_avg=0`
⇒ Δp 精确为 0 ⇒ 打印 `0.000e+00`。梯度链一直是通的，是 probe 挑了个天生零梯度的张量。

修法（`ez_train.py::update_probe`）：对**所有带 grad 的参数**逐个测位移、报 `N/M 张量真的动了`，
并用有界 metric `rel = ‖Δp‖/(‖p‖+‖Δp‖)`（zero-init 张量给 1.0，不给 ∞）。
另外诊断行把 `lora_A` / `lora_B` 的 grad_norm **分开打**，这件事以后永远看得见：

```
[outer 0 P0] update_rel=5.000e-01（=max ‖Δp‖/(‖p‖+‖Δp‖)；252/504 张量真的动了；max‖Δp‖=5.571e-02）
[outer 0][P0] 诊断 ... | 真实 backward grad_norm=4.3530e+00 (504 个张量有 grad；A=0.0000e+00 B=4.3530e+00)  ← A=0 是正常的（B 初始全 0）
```

`252/504` 恰好一半 = 所有 `lora_B` 动了、所有 `lora_A` 没动，完全对上。

**教训**：GRPO 里 "update_rel == 0" 有三种成因（梯度链断 / 组内奖励同分 / probe 挑错张量），
`ez_check_smoke_ok.py` 现在用三条独立判据把它们分开：关2 对**组奖励 spread** 判、
关2b 对**张量计数**判、关0 用不依赖奖励的梯度链自检。

---

## 0. 现状：现在用 torchrun 8 进程跑，第 3 行就会炸

| 位置 | 现状 | 后果 |
|---|---|---|
| `ez_train.py:168` | 只调 `dist_info()`，**从未 `init_process_group`** | `DDP(...)` 处的 `build_model_and_tokenizer` 直接抛 "Default process group has not been initialized" |
| `ez_train.py:60` | `log0(0, msg)` 把 rank 硬编码成 0 | world>1 时**8 个 rank 全都会打印**，且全部写同一个 `train_log.jsonl` → 日志交错成垃圾 |
| `ez_train.py:171` | `run_dir` 用 `time.strftime('%m%d-%H%M')` | 各 rank 独立取时间，跨分钟会算出**不同的目录名** → 各写各的 ckpt |
| `ez_train.py:169` | `set_seed(seed, rank)` 之后又用 `random.Random(seed)` | rank 间 seed 不同但 `rng` 又被 `seed` 重置 → 8 个 rank 抽到**同一批 game**，8× 重复劳动 |
| `ez_train.py:379` | `ez_save` 无 rank 判断 | 8 个 rank 同时往 `ckpt.tmp` 写 → 互相踩 |

所以第一步不是"优化 DDP"，是**把单进程 scaffold 变成合法的多进程程序**。这属于 P0，
跟算法无关，做完就能验证"8 卡能不能起来"。

---

## 1. 三条语义红线（踩了就静默错，不报错）

### 红线 A：GRPO 组不可拆

`group_advantages([...])` 对组内奖励做 **mean/std 归一化**。如果把一个 G=16 的组拆到
4 个 rank 上（每 rank 4 条），每个 rank 只看到 4 个奖励 → 算出的 mean/std 与真实组不同，
而且是**各 rank 各自一套**。`grpo_loss` 用这些 advantage 加权 logp，DDP 再把 4 个 rank 的
梯度平均 → 得到的既不是"16 条一组的 GRPO"，也不是任何有意义的量。

**规则：分片必须在"组"这一层之上（按 game 切），或者在切了组之后把奖励 all_gather 回来重算。**

### 红线 B：每个 rank 的 backward 次数必须完全相同

DDP 每个 `backward()` 触发一次 all-reduce。rank 之间 backward 次数不同 → 先到的 rank
在某个 all-reduce 上等，后到的 rank 永远发不出那次通信 → **死锁**（或者 30min 后
NCCL watchdog abort）。

EZ 里有两个天然的"次数不定"来源：
1. `accepted` 的条数各 rank 不同（提案是否被接受取决于生成本身）。但注意：
   **p_hat 那段 `run_ez_solver_rollouts(tokenize=False)` 不 backward**，所以它安全。
2. `len(groups) == TPS`（每个 game 一组）。如果按 game 分片，各 rank 的组数 = TPS/world，
   **TPS 不能被 world 整除时就有 rank 多一组** → 死锁。

**规则：`TPS % world == 0` 必须硬断言；`did_step` 这类"这一步要不要更新"的决定要用
`dist.all_reduce(op=MAX)` 统一（主线 `train_grpo.py:615` 已经踩过这个坑，直接沿用）。**

### 红线 C：proposer 与 solver 是两个策略，DDP 只认一套参数

`build_dual_lora` 用 peft 挂了 `proposer` / `solver` 两个 adapter，靠 `set_adapter()`
切换 `requires_grad` 来隔离。**但 DDP 是在 wrap 的那一刻，按"哪些参数 requires_grad=True"
注册 all-reduce hook 的。**

如果按现在的顺序（`get_peft_model(adapter_name="proposer")` 之后立刻 DDP 包起来），
DDP 只认 proposer 的参数；等切到 solver 相位时，solver adapter 的梯度**永远不会被
同步** → 8 个 rank 的 solver 各自漂移，评测时只有一个 rank 是对的。

**规则：wrap DDP 之前让两个 adapter 都 `requires_grad=True`，并设
`find_unused_parameters=True`。** 相位切换时未参与计算的那个 adapter 自然"unused"，
DDP 按 `find_unused_parameters` 语义跳过，而不是报错。

> ⚠️ 不要用 `find_unused_parameters=False` + 动态 toggling —— 那是死路：
> DDP 的 hook 集在 wrap 时就固定了，之后新变 trainable 的参数不会被同步。

---

## 2. 推荐方案：按 game 分片（Phase-1，语义零改动）

```
                 全局 TPS 个 game（rank0 采样后 broadcast，保证不重不漏）
                 ├── rank0: game[0::8]     → 每个 game 跑完整 G=16 条 rollout
                 ├── rank1: game[1::8]     → 组完整，group_advantages 语义不变
                 ...
                 └── rank7: game[7::8]
   每个 rank 独立完成：探索 → 提案 → p_hat 估计 → 自己的 GRPO loss
   然后 DDP all-reduce 把 8 份梯度平均 → 等价于"在 TPS 个 game 上做一次 GRPO 更新"
```

**为什么这样最省事**：`run_proposer_exploration` / `run_ez_solver_rollouts` /
`proposer_reward` / `group_advantages` **一行都不用改**，只改外层调度。

### 2.1 需要改的具体点

| # | 位置 | 改法 |
|---|---|---|
| 1 | `ez_train.py:168` | `rank, world, local = setup_dist()`（从 `train_grpo` import，别重写） |
| 2 | `ez_train.py:60` | `log0(rank, msg)`，并把 `_LOG_FH` 改成 `train_log.rank{rank}.jsonl`；rank0 的那份仍叫 `train_log.jsonl` 供现有判读脚本读 |
| 3 | `ez_train.py:171` | rank0 算 `run_dir` 名 → `dist.broadcast_object_list` 广播 |
| 4 | `ez_train.py:277` | rank0 用全局 rng 抽 `TPS` 个 game → 广播列表 → 各 rank 取 `games[rank::world]`；断言 `TPS % world == 0` |
| 5 | `ez_train.py:328` | `(loss / len(groups)).backward()` 里的 `len(groups)` 是**本 rank 的组数**（=TPS/world）。DDP 会对梯度取平均 → 全局等价于 `1/TPS`，正确。**不要**再手动乘 world |
| 6 | `ez_train.py:379` | `ez_save` 只在 `is_main(rank)` 时执行（peft 保存的是 adapter，很小，不值得每 rank 存一份） |
| 7 | `task_pool` / `persist_pool` | 各 rank 只有自己那份 → `persist_pool` 改为 rank0 gather 后写；否则 `task_pool.jsonl` 只有 1/8 |
| 8 | `env_p/env_s/env_sr` | 各 rank 建自己的（本来就是这样），但要给 `torch.manual_seed` + 每 rank 不同的 env 实例 |
| 9 | `build_dual_lora` 里 | 见红线 C：wrap 前让两个 adapter 都 trainable；`DDP(..., find_unused_parameters=True)` |

### 2.2 关键代码骨架

```python
# --- 1) 分片采样：全局 TPS 固定，各 rank 拿不重不漏的一份 ---
games: List[str] = []
if is_main(rank):
    games = [rng.choice(game_files) for _ in range(TPS)]
obj = [games]
dist.broadcast_object_list(obj, src=0)
games = obj[0]
if TPS % world:
    raise SystemExit(f"tasks_per_step({TPS}) 必须能被 world_size({world}) 整除，否则各 rank "
                     f"backward 次数不同 -> DDP 死锁")
local_games = games[rank::world]

# --- 2) 相位切换 + 更新决策统一 ---
did_step_flag = torch.tensor([1.0 if any_have_signal else 0.0], device=device)
if world > 1:
    dist.all_reduce(did_step_flag, op=dist.ReduceOp.MAX)   # 红线 B
```

```python
# --- 3) build_dual_lora 尾部：让 DDP 认全两个 adapter ---
# 两个 adapter 都必须 requires_grad=True 才能被 DDP 注册 hook；
# 相位隔离改由"未参与计算 -> unused -> 跳过"来实现。
for n, p in model.named_parameters():
    if "lora_" in n:
        p.requires_grad_(True)
...
model = DDP(model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True)     # 红线 C：必须是 True
```

---

## 3. 显存与通信账（LoRA 让 DDP 在这里非常划算）

| 项 | 单卡数值 | 8 卡 |
|---|---|---|
| 基座权重（Qwen3-4B bf16） | ~8.0 GB | 每卡一份 8 GB（DDP 不做参数切分） |
| LoRA adapter fp32（r=32，all-linear） | ~0.2 GB × 2 | 每卡 ~0.4 GB |
| AdamW 状态（只覆盖 adapter，fp32×2） | ~0.8 GB | 每卡 ~0.8 GB |
| 激活 + KV cache（G=16，8k 上下文） | 19–35 GB | 同左 |
| **DDP all-reduce 通信量 / 步** | — | **~0.4 GB（=2×adapter 参数量）** |

要点：
1. **通信量极小**（LoRA 只同步 adapter，不是 8GB 全参）→ 8 卡节点内 NVLink/PCIe 完全吃得下，
   **不需要 ZeRO / FSDP**。上 ZeRO 反而会把简单问题复杂化。
2. 因此"8 卡 = 8× 吞吐"这个假设**在生成阶段成立**，因为生成是逐 rank 独立的；
   all-reduce 只在 backward 发生，占比很小。
3. 显存瓶颈仍是**激活 + KV cache**（见 `GPU_UTILIZATION_PLAN.md`），跟卡数无关。
   8 卡不会让单卡放得下更长的上下文。

---

## 4. 环境进程数：8 卡真正的隐性杀手

每个 rank 要建 **3 个 ALFWorld batch env**：

```
每 rank env slot 数 = G (env_p) + G (env_s) + G_P (env_sr) = 16 + 16 + 8 = 40
8 卡总计          = 8 × 40 = 320 个并发 env slot
```

textworld 是**进程内**的纯 Python 环境（不是子进程），但 320 个 slot 同时 step 会：
- 吃 CPU：节点只有 96 核（三期）/ 64 核（二期）→ 严重超订，env.step 变慢，GPU 空转
- 吃内存：每个 slot 持有 game 状态 + 观测历史，几十 MB 级 → 十几 GB

**处理办法（按优先级）**：
1. `--cpus-per-task` 给足，并把 `OMP_NUM_THREADS` 压到 1–2（否则 320 个 slot × 6 线程 = 爆炸）
2. **env_sr 按需懒建**：p_hat 估计只在"有提案被接受"时才用，可以先只建 `env_p`/`env_s`，
   首次需要 G_P 估计时再建 `env_sr`（省 1/5 的 slot）
3. 若仍不够：走 §6 的方案 B，把 slot 数从 `G` 降到 `G/world`

---

## 5. 有效 batch 怎么随卡数变（决定超参要不要改）

现在 `TPS=2`、`G=16` → 每步全局 2 个 game × 16 条 = 32 条 rollout。

按 §2 的分片，**全局 TPS 不变**，所以：
- world=1：rank0 跑 2 个 game
- world=8：每 rank 跑 `2/8` —— **不整除，直接断言失败**

两个选项，**必须显式选一个并写进 config**：

| 选项 | 做法 | 效果 |
|---|---|---|
| **A. 固定全局 batch**（保守，推荐起步） | `TPS = world × 1`（每 rank 1 个 game）→ world=8 时 TPS=8 | 全局 batch 从 32 涨到 128 条；与论文 "batch 16" 的偏差需要重新核对（论文没写死编排，见 U5） |
| **B. 固定每 rank batch** | `TPS` 视为**每 rank** 的值，全局 = TPS×world | 全局 batch 随卡数线性涨，lr 需要按 √world 或 world 缩放 |

**建议**：先用 A 把 8 卡跑通（`TPS=8`），跑通后再做 A/B 对比。**不要**在没跑通的时候
同时改 batch 语义 —— 那样出了问题分不清是 DDP bug 还是超参问题。

---

## 6. 方案 B（rollout 级分片 + 奖励 all_gather）——什么时候才值得做

如果 §4 的 env slot 问题压不住，或者想把生成吞吐再翻一倍，就需要把 `G` 也切开：

```
每个 game 的 G 条 rollout 切给 world 个 rank（每 rank G/world 条）
  ↓
各 rank 只算自己那几条的 reward（标量）
  ↓
dist.all_gather 把整个组的 reward 收齐（1 float × G，代价可忽略）
  ↓
每个 rank 用同一套全局 mean/std 给自己的 rollout 算 advantage   ← 红线 A 被满足
  ↓
local_loss = (world / G) × Σ_{i∈本rank} a_i · ℓ_i              ← 关键缩放
  ↓
DDP 平均 8 份 → (1/G) Σ_all a_i ℓ_i ，与单卡完全等价
```

**为什么 `world/G` 这个缩放是必须的**：DDP 的语义是"把各 rank 的梯度取平均"，即
`(1/world)·Σ_r ∇L_r`。而我们想要的是 `(1/G)·Σ_all`。因为 `(1/world)·Σ_r (world/G) Σ_{i∈S_r} = (1/world)(world/G) Σ_all = (1/G) Σ_all` ✓。
漏掉这个 `world` 因子 → 梯度小 world 倍 → lr 等效缩小 8 倍 → 看起来"能跑但学不动"。

**代价**：proposer 的"提案"是绑在具体某条 rollout 上的（谁调了 `finish` 谁提案），
分片后提案只存在于某一个 rank → 需要把 `ProposedTask` 也 all_gather 一遍，还要处理
"同一个 game 多个 rank 都提案"的去重。**复杂度明显上升，不建议作为第一步。**

---

## 7. 分阶段落地

### P0 — 让 8 卡能起来（不改算法语义，半天）— ✅ 已落地
- [x] `setup_dist()` 接入；`log0(rank, ...)`；per-rank 日志文件
      （`train_log.rank{r}.jsonl`；rank0 额外维护 `train_log.jsonl`，判读脚本不用改）
- [x] `run_dir` 广播；`ez_save` 仅 rank0；`persist_pool` 仅 rank0（池子靠 all_gather 保证一致）
- [x] 两 adapter 都 trainable 后再 DDP，`find_unused_parameters=True`
      （`build_model_and_tokenizer(..., wrap_ddp=False)` → 挂双 LoRA → 全开 requires_grad → DDP）
- [x] `TPS % world == 0` 兜底：不整除时自动上调到 world 的倍数 + 大声 WARNING（红线 B）
- [x] 分片采样（rank0 抽 + broadcast + `games[rank::world]`）；`uid` 带 rank 防跨 rank 撞号
- [x] 死锁的显式守卫：空 GRPO 组 → 全局 MIN all_reduce 后**全员同时退出**；
      solver 空池 → 全局一致的跳过决定；空组 backward → 零 loss 占位（绝不 `continue`）
- [ ] **验收**：8 卡跑完 1 个 outer，8 个 rank 的 `update_rel` 都不为 0
      → job `12989532` 排队中，判据见 §8 关4（已实现）

#### P0 里两个容易写错、但**必须**写对的地方（本次踩点）

1. **`policy` 必须持有 DDP 包装后的对象**。`HFPolicy` 内部直接 `model(...)`；
   如果传 `raw` 或 `raw.module`，就绕过了 DDP 的 forward hook，梯度**永远不会 all-reduce**
   —— 8 个 rank 各自静默跑自己的，速度不涨、指标还对不上，且不报任何错。
   旧代码那句 `model.module if hasattr(model, "module") else model` 正是这个雷。

2. **DDP 必须在双 adapter 都可训练之后才 wrap**（红线 C）。DDP 在构造那一刻按
   "哪些参数 requires_grad" 固定 bucket 成员；先 wrap 再切 `requires_grad`，
   非活跃 adapter 的参数**从未进过 bucket 列表**，切到它的相位时梯度不会被同步。

### P1 — 效率（1 天）
- [ ] `env_sr` 懒建（省 1/5 slot）；`OMP_NUM_THREADS=1`（8 卡脚本里已设）
- [ ] `no_sync()` 包掉除最后一个 micro-batch 之外的所有 backward（主线同款）
- [ ] 每 rank 的 env 数写进启动自检日志（防止 320 slot 静默超订）

### P2 — 方案 B（仅当 P1 不够）
- [ ] rollout 级分片 + reward all_gather + `world/G` 缩放
- [ ] `ProposedTask` all_gather + 去重

### 不做（明确排除）
- **ZeRO / FSDP**：adapter 只有 0.4 GB，通信省不出东西，只会引入切分 bug
- **Tensor Parallel**：4B 模型单卡放得下，TP 是负收益
- **多机**：单节点 8 卡已够，跨节点 NCCL 只会让"各 rank 生成耗时不均"放大成死锁

---

## 8. 8 卡冒烟验收标准（沿用现成脚本）

```bash
# P0 完成后，把冒烟脚本换成 8 进程版本跑（见 P0 验收）
EZ_SMOKE_G=2 EZ_SMOKE_G_P=2 EZ_SMOKE_TPS=8 EZ_SMOKE_TURNS=16 \
  torchrun --nproc_per_node=8 src/ez_train.py --config configs/ez_qwen3_4b.yaml --dry-run
```

判据（`scripts/ez_check_smoke_ok.py` 已实现三关，8 卡额外加两条）：
1. 三关全过（facts / `update_rel` / 奖励区分度）——单卡判据不变
2. **`update_rel` 的数值在 8 个 rank 之间同量级**（差 100× 以上说明某个 rank 没参与梯度同步）
3. **每个 rank 的 `train_log.rank*.jsonl` 都存在且非空**（缺文件 = 那个 rank 提早挂了）
4. **rank0 的 `len(groups)` == TPS/world**（对不上说明分片写错了）
5. **`dist` 初始化日志里 world_size == 8**（防止 torchrun 参数没生效，退化成 8 个独立单卡进程）

> 第 5 条最容易出事：torchrun 没生效时**不会报错**，只是 8 个进程各自在单卡上跑，
> 速度没变快、日志却多了 8 倍。启动自检里必须显式打印 world_size。

---

## 9. 时间与收益预期

| 场景 | 单卡 | 8 卡 P0 | 8 卡 P1 |
|---|---|---|---|
| ALFWorld 4 outer loops | ~15 GPU-node-hours（≈15h 单卡） | ≈2h（理论 8×）| ≈1.5h（含 env 重叠收益） |
| 排队（二期 `i64m1tga800u8ka`） | — | ~26h | ~26h |

**排队时间远大于计算时间**——所以 P0 必须一次做对，不要拿 8 卡作业当调试工具。
单卡 `debug` 分区（A40 30min，几分钟起算）才是调试位，8 卡只跑已经验证过的代码。
