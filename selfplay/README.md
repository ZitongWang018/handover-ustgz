# selfplay — ALFWorld × Qwen3-4B-Instruct 的 GRPO 智能体训练

把 **Qwen3-4B-Instruct-2507** 放在 **ALFWorld** 文本具身环境里做多轮 ReAct + GRPO 训练，
并给出可复现的评测（留出集 `valid_unseen`）与外部基线（DeepSeek-V4-Pro）。

背景与本项目的定位见 `../docs/SPADE_ALFWORLD.md`：
SPADE（arXiv:2608.19197）本身**不含** ALFWorld，这是把"自适应环境下的自博弈"迁移到
具身文本智能体的第一步（路线 B：固定环境基线）。

---

## 1. 快速开始

```bash
# 0) 一次性打通环境（造 GLIBC 兼容层 + 装 textworld/alfworld 依赖）
bash env_setup/setup_deps.sh

# 1) 分层冒烟测试（强烈建议逐层来）
python src/smoke_test.py --stage env      --batch 2     # 环境能 reset/step 吗
python src/smoke_test.py --stage rollout  --batch 2     # 随机策略能走完一局吗
python src/smoke_test.py --stage tokens   --batch 2     # chat 模板/token mask 拼得对吗
python src/smoke_test.py --stage all      --batch 2     # 4B 模型 rollout + 前向/反向

# 2) 训练（8 卡）
sbatch scripts/train_a800x8.sbatch                      # 二期 A800 × 8
sbatch scripts/train_h100x8.sbatch                      # 三期 H100 × 8（需先离线搬依赖）

# 3) 评测
AGENT=echo     sbatch --export=ALL,AGENT=echo     --job-name=sf-eval scripts/eval.sbatch
AGENT=random   sbatch --export=ALL,AGENT=random   --job-name=sf-eval scripts/eval.sbatch
AGENT=deepseek sbatch --export=ALL,AGENT=deepseek,N=20 --job-name=sf-eval-ds scripts/eval.sbatch
AGENT=hf       sbatch --export=ALL,AGENT=hf,MODEL=.../runs/qwen3-4b-alfworld-grpo/ckpt scripts/eval.sbatch
```

---

## 2. 目录

```
selfplay/
├── env_setup/
│   ├── patch_libdownward.py     # 把 TextWorld 规划器的 GLIBC 需求从 2.32 降到 2.17（+符号垫片）
│   ├── setup_deps.sh            # 一键打通运行环境（只写 ztwang 内）
│   └── ENV_BRINGUP.md           # 实测过的依赖清单与踩坑记录
├── src/
│   ├── alfworld_compat.py       # 兼容层：import textworld 之前必须 install()
│   ├── alfworld_env.py          # ALFWorld 批量环境适配（自研，不依赖 AgentGym）
│   ├── prompts.py               # ReAct system prompt / 观测渲染 / 动作解析
│   ├── reward.py                # 奖励函数 + group-normalized advantage
│   ├── rollout.py               # ReAct rollout + 轨迹→词元(mask) 拼接
│   ├── train_grpo.py            # 自研 GRPO 训练入口（torchrun）
│   ├── agents.py                # 评测用策略：HTTP API / 随机 / echo / 脚本化
│   ├── llm_client.py            # OpenAI 兼容客户端（默认 DeepSeek-V4-Pro）
│   ├── eval_alfworld.py         # 评测脚本
│   └── smoke_test.py            # 分层冒烟测试
├── configs/
│   ├── base_config.yaml         # ALFWorld 环境配置（自持副本）
│   ├── train_qwen3_4b.yaml      # 训练超参
│   └── secrets.env              # API key（已 gitignore）
├── scripts/                     # sbatch / 环境脚本
└── third_party/                 # 兼容层产物 + --target 装的依赖 + wheel 缓存（gitignore）
```

---

## 3. 关键设计决策

| 事项 | 决策 | 原因 |
|---|---|---|
| 环境访问方式 | **进程内**（`textworld.gym` 的 `AsyncBatchEnv`），不走 HTTP 服务 | 世界编译（PDDL 规划）是主要开销，批量 env 会在 N 个子进程里并行编译；HTTP 单进程会串行 |
| 训练框架 | **自研 GRPO**，不用 TRL 的 `GRPOTrainer` | TRL 0.26 的 `rollout_func` 只在 `use_vllm=True` 时生效，而本机 vLLM 0.8.5 与 transformers 4.57 不匹配（TRL 要求 ≥0.10.2）；且多轮轨迹的 loss mask 需要细粒度控制 |
| 生成后端 | HF `generate`（DDP 每 rank 各自生成） | 免掉 vLLM 版本地狱；8 卡各跑自己的任务，天然数据并行 |
| 动作空间 | 始终把 `admissible_commands` 给模型 | ALFWorld 稀疏奖励 + 长时域，不收窄动作空间基本训不动 |
| loss 的词元掩码 | assistant 段用**真实生成的 token id**，观测/模板段 mask=0 | 多轮智能体 RL 的标准做法：只对模型自己产出的词元算 loss |
| 奖励 | 默认 `success + 轻量过程项`，可一键切"纯 0/1" | 纯二值奖励在 GRPO 里早期极易"组内全 0 → advantage 全 0 → 零梯度"；过程项只用来打破僵局，`use_sparse_only: true` 可切回论文式最素设定 |
| 断点 | 每次保存**先写临时目录再原子替换**，只留最新一个 | 用户铁律二 |

---

## 4. 已知坑（都已在代码里处理）

1. **`libdownward.so` 需要 glibc 2.32**：二期 Ubuntu 20.04（glibc 2.31）、三期 Rocky 8（glibc 2.28）都跑不了。
   → `env_setup/patch_libdownward.py` + `src/alfworld_compat.py`。
   注意三个反直觉点：只改版本字符串不改 `vna_hash` 没用（glibc 会报 `version 'GLIBC_2.17' not found`）；
   `LD_PRELOAD` 垫片也没用（版本需求记在 `libc.so.6` 头上，不在调用方）；
   而 libdownward 用到 glibc 2.32 才有的 `__libc_single_threaded`，必须用 `ctypes.CDLL(..., RTLD_GLOBAL)`
   预加载垫片，dlopen 时才能从全局域解析到。
2. **生成必须左 padding**，训练前向必须右 padding。
3. **DDP 各 rank 的 backward 次数必须一致**：轨迹条数固定（`n_tasks_per_iter × group_size`），
   "这一步要不要更新"用 `all_reduce(MAX)` 统一决策，否则会挂死。
4. **chat 模板必须前缀稳定**：`build_token_sequence` 靠"渲染前 i+1 条消息"的增量 token 拼序列，
   已加断言；Qwen 的 ChatML 满足。
5. **三期无外网** → 依赖必须先在二期 `pip download` 成 wheel 再搬过去（`third_party/wheels/`）。
6. **⚠️ textworld 批量环境会打乱 slot 顺序**（最容易踩、后果最严重的一个）：
   `textworld/gym/envs/textworld_batch.py:96-118` 在 `reset()` 里对 gamefiles 做 `rng.shuffle`
   再按顺序 `load`，所以 **slot i ≠ 请求列表第 i 个 task**。
   - 如果 `run_rollouts` 按 slot 顺序返回，`train_grpo.py` 的 `group_ids = [i // g ...]`
     会把**不同 task 混进同一个 GRPO 组** → 组内归一化 advantage 算错，**不报错地毁掉训练**。
   - eval 的 `zip(trajs, chunk)` 也会逐条归因错误（曾出现 `...Pencil-None-Shelf-308` 名下挂着 egg/garbagecan 的轨迹）。
   - **约定**：`run_rollouts` / `run_expert` 返回顺序**保证 task-major / 组连续**
     （第 i 个 task 的 g 条轨迹占据 `[i*g, (i+1)*g)`），并且 `Trajectory.game_file` 是**实际装载**的那个。
     对齐由 `_align_slot_tasks`（按出现次数逐个消耗，支持 `group_size > 1`）完成，
     `eval_alfworld.py` 与 `train_grpo.py` 各有一道断言守着，破坏契约会立刻抛错而不是静默出错。
   - 复现/压测：`python src/smoke_test.py --stage rollout --batch 8 --group-size 4`。

---

## 5. 资源与提交

| | 二期 A800 | 三期 H100 |
|---|---|---|
| 分区 | `i64m1tga800u8ka`（8 卡仅此） | `acd_u` |
| account | `root` | `d_tonggehuang_team` |
| 单机 | 64c / 1TB | 96c / 2TB |
| 外网 | ✅ | ❌ |

统一规范见 `../docs/CLUSTER_GUIDE.md`（第 0 节两条铁律优先）。
