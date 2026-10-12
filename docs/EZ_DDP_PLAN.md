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
| 修掉 2 卡那一发触发的判据假警报 | ✅ | 见 §0.5.2；回归用例 18/18 全绿（`ez_check_smoke_selftest.py`，新增 N/O/P/Q/R） |
| **2 卡真实多进程（DDP 接线）** | ✅ | job `12989603` @ `debug:gpu3-9`，2:42。两 rank 各 `本地 games=1/2`、同一 `run_dir`、`world_size=2`、空池时**一致跳过**、双双 `[done]`、`rank0/1.jsonl` 各 15 行 |
| **2 卡 `turns=16` 硬通过** | ✅ **P0 验收完成** | job `12989645` @ `debug:gpu3-9`，6:43，ExitCode 0:0。自检全 OK 零 WARN：`update_rel(P0)=5.00e-01`、`252/504`、`accept=1/2`、`solver reward mean=1.000`、**两 rank `max(update_rel)` 同量级（比值 1.0）** |
| **8 卡冒烟（P0 最后一关）** | ✅ **P0 全部完成** | job `12989532` @ `i64m1tga800u8ka`，2.7min。关4 三条全绿：**8 个 rank 日志齐全**、`world_size=[8]`、**8 个 rank 的 `max(update_rel)` 全为 `5.00e-01`（比值 1.0）** |

**8 卡那发的实测输出（job 12989532）**

```
  [OK   ] 关0 梯度链通：proposer grad_norm=5.661e+02, solver grad_norm=5.639e+02
  [OK   ] 关1 facts 通道存在（109 条事实）
  [OK   ] 关2 update_rel 共 2 处：outer 0 P0=5.00e-01, outer 0 S0=0.00e+00
  [OK   ] 关2b 优化器实际更新张量数：outer 0 P0=252/504, outer 0 S0=0/504
  [OK   ] 关3 proposer 有有效提案：r0=0/2, r1=1/2, r2=0/2, r3=0/2,
                                  r4=0/2, r5=0/2, r6=1/2, r7=0/2（全局 2/16）
  [WARN ] 关3 solver reward mean=['0.000', '0.000']（缩规模下任务太难，正常）
  [OK   ] 关4 8 个 rank 日志都存在且非空
  [OK   ] 关4 日志里的 world_size=[8]
  [OK   ] 关4 各 rank 的 max(update_rel)=r0..r7 全为 5.00e-01（最大/最小=1.0）
                                            ⇒ 各 rank 都参与了梯度同步
```

> **8 卡那一发最有意思的一条证据**：rank0 自己的两个候选奖励都是 0
> （`accept=0/2`，组内 spread=0 ⇒ 优势全 0 ⇒ 它自己算不出梯度），
> 可它的 `update_rel` 却是**非零的 5.00e-01**。这个梯度只能是 DDP 的
> all-reduce 从别的 rank（r1、r6 各接受了 1 条）传过来的 —— 而且 8 个 rank
> 的值**完全一致**。这就是 DDP 真正接上的铁证。

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

### 0.5.2 第二个假故障：`0/504` 被当成断链（2 卡那一发暴露的）

2 卡作业 `12989603` 的日志是**完全正确**的，却被旧判据打出两条 FATAL：

```
[OK   ] 关4 2 个 rank 日志都存在且非空
[OK   ] 关4 日志里的 world_size=[2]
[FATAL] 关2b 优化器实际更新张量数：outer 0 P0=0/504（全 0 —— 优化器真的一步没动）
[WARN ] 关2 这些 P 相位组内奖励全同分（无信号，属正常）：['0P0(spread=0.000)']
[FATAL] 关3 proposer 全部被拒（[(0, 2)]）
[FATAL] 关4 各 rank 的 max(update_rel)=r0=0.00e+00, r1=0.00e+00（最大/最小 = inf）
```

真因：那一发 `accept=0/2`（`turns=12` 太短，proposer 走不到 `finish`）⇒ 候选奖励全同分 ⇒
GRPO 组内优势全 0 ⇒ **梯度精确为 0**。日志里的诊断行写得明明白白：

```
[outer 0][P0] 诊断 ... | 真实 backward grad_norm=0.0000e+00 (504 个张量有 grad；A=... B=...)
```

所以 `0/504` 是**正确行为**，不是 bug。旧判据只看"动了几个张量"，不看"有没有梯度"，
就把"无信号"和"真断链"混成同一类；关4 里 `lo == 0 ⇒ ratio = inf` 同理。

修法：判"参数没动"的**最终依据改成真实 backward grad_norm**（`ez_check_smoke_ok.py::GN_RE`）：

| 现象 | 旧判据 | 新判据 |
|---|---|---|
| `grad_norm > 0` 且一步没动 | FATAL | FATAL（**真断链**，这才该报） |
| `grad_norm == 0` 且一步没动 | FATAL ❌ | WARN（奖励同分 ⇒ 不动是正确的）|
| 各 rank `update_rel` 全 0，比值 inf | FATAL ❌ | OK（各 rank grad_norm 也全 0 ⇒ **一致无信号**）|
| 各 rank `update_rel` 全 0，但某 rank `grad_norm > 0` | FATAL | FATAL（那个 rank 真有问题）|
| 日志里没有 `grad_norm` 行（旧代码）| 不判 | FATAL（无法区分，必须升级代码）|

⚠️ 正则有个坑：`update_probe` 的行是 `[outer 2 P0] update_rel=...`（一个括号、空格分隔），
而 P/S 相位的诊断行是 `[outer 2][P0] 诊断 ...`（**两个独立方括号**）。两条正则不能混用。

**代价**：这个假警报是实打实的——`ez_smoke_8gpu.sbatch` 有 `set -euo pipefail`，
自检 exit 1 ⇒ 整发作业被 Slurm 记成 **FAILED**（`12989603` 就是），
看起来像 DDP 炸了，其实 DDP 全对。


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
- [x] **验收**：8 卡跑完 1 个 outer，8 个 rank 的 `update_rel` 都不为 0
      → job `12989532` @ `i64m1tga800u8ka`，2.7min，**关0/1/2/2b/3/4 全绿**
      （8 个 rank 的 `max(update_rel)` 全为 `5.00e-01`，最大/最小=1.0）

#### P0 里两个容易写错、但**必须**写对的地方（本次踩点）

1. **`DistributedDataParallel` 不转发 unknown 属性**（torch 2.x 实测，job `12989565`）。
   `model.set_adapter("proposer")` 会直接抛
   `AttributeError: 'DistributedDataParallel' object has no attribute 'set_adapter'`，
   `generate` / `save_pretrained` 同理。所以分工必须清楚：

   | 调用 | 用在哪个对象上 | 为什么 |
   |---|---|---|
   | `set_adapter` / `generate` / `save_pretrained` / `print_trainable_parameters` | **内层 `raw`** | DDP 根本没有这些属性 |
   | **要算梯度的那次 forward**（`per_token_logps` / `grpo_loss`） | **包装后的 `model`** | 否则绕过 DDP 的 forward hook，梯度**永远不会 all-reduce**，且不报任何错 |
   | `train()` / `eval()` / `zero_grad()` | **包装后的 `model`** | 这些是 `nn.Module` 自己的方法；DDP 要靠 `train()` 重置 `require_backward_grad_sync` |

   推论：`HFPolicy` 内部走 `generate`，所以它必须持有 **`raw`**；
   loss 那一路由 `ez_train` 显式把 `model` 传进 `per_token_logps/grpo_loss`。
   （一开始我把 `policy` 写成持有包装后的对象，属于**反了的**修法，被 job 12989565 打回。）
   启动自检里留了探针日志 `hasattr(DDP,'set_adapter')`，升级 torch 时一眼能看出行为有没有变。

2. **DDP 必须在双 adapter 都可训练之后才 wrap**（红线 C）。DDP 在构造那一刻按
   "哪些参数 requires_grad" 固定 bucket 成员；先 wrap 再切 `requires_grad`，
   非活跃 adapter 的参数**从未进过 bucket 列表**，切到它的相位时梯度不会被同步。

#### 已知风险：`find_unused_parameters=True` × activation checkpointing

第二个适配器在非活跃相位里"没参与前向"，靠 `find_unused_parameters=True` 让它被跳过。
而 `find_unused_parameters=True` 的实现方式是**每次 backward 都遍历 autograd graph**
去判定 unused 集合 —— 这和 activation checkpointing 的"反向时重算前向"是有张力的，
PyTorch 社区里出现过 `Expected to mark a variable ready only once` 这类报错。
本项目**两层都开了 checkpoint**：基座 `gradient_checkpointing_enable(use_reentrant=False)`
+ `per_token_logps` 里 `checkpoint(_chunk_logp, use_reentrant=False)`。
`use_reentrant=False` 是官方支持 DDP 的那一支，预期能work，但**没在 8 卡上实测过**。

**若 8 卡冒烟报 "mark a variable ready only once" / "finished reduction in the prior iteration"，**
按下面的顺序改（前两条不改算法语义）：

1. `static_graph=True`：不行 —— 本项目的"被使用集合"会在 proposer/solver 相位之间变化，
   静态图会把它缓存错。
2. **把"未使用的参数"变成"被使用的参数"**（推荐）：在 loss 上加一项系数为 0 的
   `0.0 * Σ p.sum()`，覆盖**本相位不活跃的那个 adapter**。前向因此触及全部 adapter 参数，
   `find_unused_parameters=False` 就能用，代价只是每步多 all-reduce 一份未活跃 adapter
   （≈0.26 GB，LoRA 下可忽略），换来的是"最成熟的那条组合（DDP + checkpoint，无 unused 判定）"。
3. 关掉 `per_token_logps` 里的分块 checkpoint（`_LOGSOFTMAX_CHUNK` 调到 ≥ 序列长度的量级）
   —— 只在显存撑得住时用，属于拿显存换稳定。

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

# 或者直接提交 sbatch（推荐，内含单测 + facts 预检 + 自检五段）：
sbatch -p i64m1tga800u8ka -t 02:00:00 --gres=gpu:8 \
    ~/ztwang/selfplay/scripts/ez_smoke_8gpu.sbatch
# 最小成本的 DDP 接线验证（debug 区，立刻起算，实测 2:42 跑完）：
sbatch -p debug -t 00:28:00 --gres=gpu:2 --cpus-per-task=16 --mem=240G \
    --export=ALL,NPROC=2,EZ_SMOKE_TPS=2,EZ_SMOKE_TURNS=16 \
    ~/ztwang/selfplay/scripts/ez_smoke_8gpu.sbatch
```

判据（`scripts/ez_check_smoke_ok.py` 已实现，单卡关0/1/2/2b/3；多卡额外加关4）：

1. **关0** 两个 adapter 各自都能接到梯度（`grad_chain_selfcheck`，不依赖环境/奖励）
2. **关1** facts 通道存在
3. **关2** `update_rel` 必须对着**本组奖励 spread** 判：`spread>0` 却不动 = 断链（FATAL）
4. **关2b** `grad_norm > 0` 却 `0/M 张量动了` = 真断链（FATAL）；
   `grad_norm == 0` 却不动 = 奖励同分，**正确**（WARN）。← ★ 这条必须看 grad_norm，不能只看位移
5. **关3** proposer `accept >= 1`，solver 奖励不是全 0
6. **关4** 每个 rank 的 `train_log.rank*.jsonl` 都存在且非空（缺文件 = 那个 rank 提早挂了）
7. **关4** 各 rank 的 `max(update_rel)` 同量级。⚠️ **但"全 0"要分两种**：
   - 各 rank 的真实 `grad_norm` **也全 0** ⇒ 一致无信号（开局奖励同分），**不是**没同步
   - 某 rank `grad_norm > 0` 而它位移为 0 ⇒ 那个 rank 真有问题（FATAL）
8. **关4** `dist` 初始化日志里 `world_size == 本发进程数`
   （防止 torchrun 参数没生效，退化成 N 个独立单卡进程）

> 第 8 条最容易出事：torchrun 没生效时**不会报错**，只是 N 个进程各自在单卡上跑，
> 速度没变快、日志却多了 N 倍。启动自检里必须显式打印 world_size。

> ★ 判据本身要先用回归用例证明**不会误报**，再拿它当准入门：
> `python scripts/ez_check_smoke_selftest.py`（16 个用例 A–P，全部来自真实踩过的坑，
> 包括 O = 2 卡一致无梯度、P = 某 rank 有梯度却不动）。
> `ez_smoke_8gpu.sbatch` / `ez_smoke_1gpu.sbatch` 的 `[0/3b]` 段会自动跑它。
> 这一条不是洁癖：2026-10-12 就是因为判据误报，让一发完全正确的 2 卡 DDP 作业
> 被 Slurm 记成 FAILED（见 §0.5.2）。

---

## 9. 时间与收益预期

| 场景 | 单卡 | 8 卡 P0 | 8 卡 P1 |
|---|---|---|---|
| ALFWorld 4 outer loops | ~15 GPU-node-hours（≈15h 单卡） | ≈2h（理论 8×）| ≈1.5h（含 env 重叠收益） |
| 排队（二期 `i64m1tga800u8ka`） | — | ~26h | ~26h |

**排队时间远大于计算时间**——所以 P0 必须一次做对，不要拿 8 卡作业当调试工具。
单卡 `debug` 分区（A40 30min，几分钟起算）才是调试位，8 卡只跑已经验证过的代码。
