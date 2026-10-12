# selfplay：ALFWorld × Qwen3-4B-Instruct 的 GRPO 训练（设计与协议）

> 代码在 `~/ztwang/selfplay/`。本文只讲**为什么这么做**、**训练/评测协议**、**已知局限与下一步**。
> 操作层面的东西（怎么跑、目录、坑）看 `selfplay/README.md`。
> 最后更新：2026-10-12

---

## 1. 定位：这是 SPADE 迁移路线里的「B」

`docs/SPADE_ALFWORLD.md` 的结论是：SPADE（arXiv:2608.19197）里没有 ALFWorld，而且
"让设计器现场写可执行环境"这个核心机制**装不进** ALFWorld（ALFWorld 的世界只能由 PDDL/TextWorld 造）。
所以路线分三步：

| 路线 | 内容 | 状态 |
|---|---|---|
| **A** | 用官方 4B checkpoint 零样本评 ALFWorld，看能力能不能迁移 | 待做（先要能跑 ALFWorld） |
| **B** | **在 ALFWorld 上做 GRPO，拿到固定环境基线** | ✅ 本仓库 |
| **C** | 改造设计器 prompt / 校验器，让它生成"ALFWorld 风格"的文本世界；真 ALFWorld 当留出集 | 后续 |

本仓库是 **B**：把 ALFWorld 当固定环境，用 GRPO 训 Qwen3-4B，产出：
1. 一个训得动的多轮智能体 RL 流水线（可直接复用到路线 C）；
2. ALFWorld 上的 GRPO 基线数字（路线 A/C 的对照组）。

---

## 2. 环境层：为什么是「进程内批量环境」而不是 AgentGym 的 HTTP 服务

机器上现成的 `agentenv-alfworld` 是 AgentGym 那套：一个 FastAPI server 持有一堆环境，客户端走 HTTP。
它很好用，但**用来做 RL rollout 有两个硬伤**：

1. **世界编译是主要开销**。ALFWorld 的 `.tw-pddl` 每次 `reset()` 都要跑一遍 PDDL 编译 + 规划，
   单局几秒量级。HTTP server 是**一个进程**，只能串行编译；而 `textworld.gym` 的 `AsyncBatchEnv`
   会为每个 slot 起独立子进程，`batch_size=N` 就 N 路并行编译。
2. **分组采样不好控制**。GRPO 要求"同一任务的多条轨迹"归为一组。批量 env 的
   `TextworldBatchGymEnv.gamefiles` + `seed()` 可以由我们完全控制每一批玩哪些局，
   再用 `infos["extra.gamefile"]` 反查每 slot 实际拿到的局，分组干净。

代价是我们得自己写适配层（`src/alfworld_env.py`，约 150 行）——比让两个 Python 版本跨进程通信便宜。

---

## 3. 算法：自研 GRPO

### 3.1 为什么不用 TRL 的 `GRPOTrainer`

读了 `thinktool` 环境里 TRL 0.26.2 的 `grpo_trainer.py`：
`rollout_func` 的调用点**只在 `self.use_vllm` 为真时**存在（`vllm_mode` 为 `server` 或 `colocate` 两个分支）。
而本机 vLLM 是 0.8.5.post1，TRL 0.26 需要 ≥0.10.2，加上 transformers 4.57.3 与 vLLM 0.8.5 本身不兼容。
绕开 vLLM 就只能放弃 `rollout_func`，而普通路径又无法做"多轮 + 只对模型词元算 loss"。

结论：**自己写**。多轮智能体 RL 的 loss mask 本来就该自己掌控，代码量反而更小。

### 3.2 目标函数

对每个任务采 `G` 条轨迹，奖励 `R_i`，组内标准化得 advantage：

```
A_i = (R_i − mean_g) / (std_g + eps)          # 组内无区分度（std<eps）时该组 A_i ≡ 0
```

单次迭代（`num_iterations=1`，不带参考模型 `β=0`）：

```
ratio_t  = exp( logπ_θ(a_t|s_t) − logπ_old(a_t|s_t) )
L        = − (1/Σm_t) Σ_t  m_t · min( ratio_t·A , clip(ratio_t, 1±ε)·A )
```

* `m_t` = loss mask，**只在模型自己生成的词元上为 1**；
  chat 模板骨架、环境观测、`<|im_end|>` 之外的内容全部 0。
* `π_old` 在更新前用当前策略一次性算出（`num_iterations=1` 下与 rollout 时一致）。
* 8 卡 DDP 数据并行：每个 rank 跑自己的任务与轨迹，梯度全归约。

### 3.3 轨迹 → 词元（`src/rollout.py::build_token_sequence`）

多轮 agent 的序列有两个来源：

* **assistant 段 = 真实生成的 token id**（不是把生成文本重新 tokenize——那会和实际采样有偏差）；
* **非 assistant 段 = 重新渲染 chat 模板得到的 token**（mask=0），靠"渲染前 i+1 条消息"的
  增量长度来切片，前提是模板前缀稳定（Qwen 的 ChatML 满足，代码里有断言）。

---

## 3.4 轨迹与任务的对应关系（slot 打乱问题）

这是本项目**唯一一个曾经静默出错**的地方，值得单独记一笔。

`textworld` 的批量环境在 `reset()` 时会打乱请求的 gamefiles
（`textworld/gym/envs/textworld_batch.py:96-118`：`rng.shuffle(gamefiles)` 后按顺序 `load`），
因此**批量槽位 `slot i` 通常不是请求列表里的第 `i` 个 task**。

后果分两层：

* **正确性层**：`group_ids = [i // group_size ...]` 这类写法会把不同 task 混进同一组，
  GRPO 的组内归一化 advantage 就变成"跨任务比较"，**训练不会报错，只会学歪**。
* **归因层**：eval 输出里 `task_id` 与真实轨迹错配（曾出现 `pick_and_place_simple-Pencil-None-Shelf-308`
  名下挂着 egg/garbagecan 的动作序列）。

约定（已写进 docstring，并有两道断言守着）：

| 接口 | 契约 |
|---|---|
| `_align_slot_tasks(games, used, group_size)` | 用 `defaultdict(deque)` 按出现次数逐个消耗，**支持重复 gamefile 与 `group_size > 1`** |
| `run_rollouts` / `run_expert` | 返回顺序 **task-major / 组连续**：第 `i` 个 task 的 `g` 条轨迹占据 `[i*g, (i+1)*g)` |
| `Trajectory.game_file` | 存**实际装载**的 gamefile（`used[s]`），不是请求值 |
| `eval_alfworld.py` | 逐条校验 `realpath(traj.game_file) == realpath(task["game_file"])` |
| `train_grpo.py` | 校验每个 group 内 `game_file` 唯一一致，违反即 `RuntimeError` |

压测命令：`python src/smoke_test.py --stage rollout --batch 8 --group-size 4`。

---

## 4. 奖励：稀疏是主要矛盾

ALFWorld 的官方奖励 = 整局是否完成（0/1），单局最多 50 步。
把纯 0/1 直接丢进 GRPO，早期几乎必然是"一组全 0 → advantage 全 0 → 零梯度"，
`train_grpo.py` 里对这种情况直接跳过更新（不硬造噪声梯度）。

默认配置（`configs/train_qwen3_4b.yaml` 的 `reward`）在 `success` 之外挂了几个**很轻**的过程项：

| 项 | 默认 | 作用 |
|---|---|---|
| `success` | `1.0` | 主奖励 |
| `invalid_action_penalty` | `0.10` | 输出不在 `admissible_commands` 里 |
| `format_violation_penalty` | `0.10` | 没按 `Thought:/Action:` 格式输出 |
| `repeat_penalty` | `0.05` | 连续重复同一动作（死循环） |
| `valid_step_bonus` | `0.01` | 每步合法，提供微弱方差 |
| `efficiency_bonus` | `0.10` | 成功且步数少 |

`use_sparse_only: true` 会把以上全部清零，**只剩 0/1** —— 需要"论文式最素设定"时切这个开关。
过程项的选择本身就是可做的消融（见 §6）。

---

## 5. 评测协议

| 项 | 设定 |
|---|---|
| 训练集 | `json_2.1.1/train`（ALFWorld 官方 train split） |
| 分布内评测 | `valid_seen`（`eval_in_distribution`） |
| **分布外评测（主指标）** | **`valid_unseen`**（`eval_out_of_distribution`，论文口径） |
| 单局最大步数 | 训练 30；**最终报数用 50** |
| 动作空间 | 始终给 `admissible_commands` |
| 解码 | 评测 `temperature=0`（贪心），训练 `1.0` |

`src/eval_alfworld.py` 支持四种被测主体，用于把数字钉在可比的位置：

| `--agent` | 说明 |
|---|---|
| `echo` | 永远 `look`：验证链路 + 成功率地板 |
| `random` | 从 admissible 均匀随机：随机基线 |
| `scripted` | 确定性启发式优先级：非学习的"中等水平"参照 |
| `deepseek` | **DeepSeek-V4-Pro**（`effort=max`）直接玩：强模型参照（需外网，只能二期跑） |
| `hf` | 本地 Qwen3-4B（base 或我们训出的 `runs/.../ckpt`） |

---

## 6. 已知局限 / 待做

1. **吞吐**：HF `generate` 逐轮生成（没有 vLLM），50 步 × 8 轨迹的迭代会明显慢于 vLLM 版本。
   下一步可以接 vLLM 0.10+ 的 colocate 模式，但要先把环境升到匹配的 transformers。
2. **上下文增长**：轨迹最长可到 6–8k token，超过 `max_context_tokens` 会丢中间轮次（保留 system+任务+最近若干轮），
   这会改变策略所见。值得做一次"可训/可截断"的对照。
3. **奖励塑形的消融**：`use_sparse_only` vs 默认过程项，是目前最该跑的对照实验。
4. **hint 信号**：SPADE 的训练信号核心是 hint-based regret；ALFWorld 的 PDDL 能推出专家计划，
   可以做成"下一步正确动作"的 privileged hint。这是路线 C 的接口。
5. **`invalid_action_mode=raw`** 会把模型输出的非法动作**原样**丢给 TextWorld（诚实、无 off-policy 偏移），
   代价是浪费步数；`best_match` 会把动作模糊对齐到合法命令（推进更顺但有 off-policy 偏移）。
   两种都实现了，默认 `raw`。
6. **三期落地**：三期无外网，`thinktool` 环境不在三期。需要走
   "二期 `pip download` 出 wheel → 搬到三期 → `--target` 离线装"（`env_setup/setup_deps.sh` 已支持 `OFFLINE=1`）。
