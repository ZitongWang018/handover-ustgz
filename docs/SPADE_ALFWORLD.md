# SPADE × ALFWorld × Qwen3-4B：可行性与实施路线

> 目标论文：**SPADE: Self-Play in Adaptive Synthetic Executable Environments**
> arXiv:2608.19197（v3，2026-08-31）· 代码 <https://github.com/spade-rl/spade> · 项目页 <https://spade-rl.github.io>
> 本文只回答一个问题：**想让 Qwen3-4B 在 ALFWorld 上跑这个工作，现实吗？怎么做？**
> 最后更新：2026-10-11

---

## 1. 结论先行

**你的判断是对的：ALFWorld 不在 SPADE 里，而且不是"漏做了"，是结构上放不进去。**

三条硬证据：

1. 论文摘要与正文只声明**两个 setting**：`cognitive games` 与 `multi-turn tool use`。
2. 评测集是 8 个留出的 math / science / code / reasoning benchmark + BFCL-v4 multi-turn + ACEBench-Agent，**全部是静态题集或 API 调用，没有交互式具身环境**。
3. 仓库全量文件树里，`alfworld` / `textworld` / `webshop` / `jericho` / `sciworld` 命中数为 **0**。它的 `games` 指的是 RLVE 那套数学博弈（`coin_square_game`、`delta_nim_game`、`euclid_game`、`multiple_flipping_game` 等 11 个），**不是文字冒险游戏**。

所以"在 ALFWorld 上复现 SPADE"本质上是**把方法迁移到一个新 setting**，属于增量贡献，不是复现。这既是风险（要自己承担说服力），也是机会（论文没占这个坑）。

---

## 2. SPADE 到底在做什么

单模型扮演两个角色，共享同一套参数，交替训练：

| 角色 | 干什么 | 用什么信号训练 |
|---|---|---|
| **Environment Designer** | 写**完整可执行的 Python 环境代码**，带 Gym 风格 `reset()` / `step()`，同时给一个 privileged hint | hint-based regret：agent 有 hint 的得分 − 没 hint 的得分 |
| **Reasoning Agent** | 在这些生成出来的环境里做题 | 任务本身的 return |

关键设计：
- 环境是**生成出来的代码**，不是既有 benchmark。设计器被 regret 信号推着去生成"agent 刚好不会做、但做得出来"的任务 → 自适应课程。
- 设计器要**接地**：从预训练语料里采样文档当上下文（`CORPUS_FILE`），并且维护**累积的环境记忆**（`env_memory.py`）。论文说这两条是成功的关键组件。
- 训练算法：**GRPO**，一次 400 rollouts。

**核心代码结构**（`main` 分支）：

```
spade/core/
  orchestrator.py            SPADE 主循环
  fixed_env_orchestrator.py  ← 固定环境基线（ALFWorld 要从这里接）
  game_generator.py          ← 设计器产出环境
  hint_generator.py          ← privileged hint
  difficulty_controller.py   ← 自适应难度
  env_memory.py              ← 环境记忆
  env_validator.py           ← 生成代码的结构/运行时校验
  corpus.py                  ← 语料接地
  envs/
    env_adapter.py           ← ★ 统一接口，新增环境从这里派生
    rlve_adapter.py          ← 现成范式：接 RLVE 环境
    gem_adapter.py           ← 现成范式：接 GEM
    synthetic_game_adapter.py
    synthetic_game_env.py
    tool_use_base_env.py
train_spade_slime.py         ← Slime/SGLang/Megatron 后端
train_spade_tinker.py        ← Tinker 后端
slurm/train_spade_slurm_apptainer.sbatch   ← ★ 官方 Slurm 启动器
```

---

## 3. 为什么 ALFWorld 装不进"设计器"框架

这是整件事最关键的一点，决定了方案怎么设计：

| | SPADE 的环境 | ALFWorld |
|---|---|---|
| 本体 | **一段可执行的 Python 代码**（设计器现场写出来的） | **一个固定的模拟器**（TextWorld 引擎 + PDDL） |
| 谁能造 | 模型自己造 | 只有 PDDL/TextWorld 能造，模型造不了 |
| 接口 | Gym `reset()` / `step()`，自己定 | 已有，`init_env(batch_size=N)`，`step` 收 admissible commands |
| 观测 | 设计器自己设计 | 固定文本 + `admissible_commands` |
| 奖励 | 设计器自己写 verifier | 固定：任务完成 0/1 |
| 时域 | 设计器控制 | 长（ALFWorld 单局可达 ~50 步），**稀疏奖励** |

**结论：ALFWorld 只能当"固定环境"或"留出评测集"，不能当设计器的产出目标。**

设计器能产出的，最多是**"像 ALFWorld 那样的文字世界"**——即让设计器生成新的文本环境代码（房间/容器/物品/导航/操作），保持 ALFWorld 的观测格式，把真 ALFWorld 当留出测试。这条才是真正的方法迁移。

---

## 4. 四条可行路线

| 路线 | 做什么 | 改动量 | 产出价值 | 建议 |
|---|---|---|---|---|
| **A. 直接评测** | 下载官方 **4B games / tool-use checkpoint**，零样本评 ALFWorld | 几乎为零（只搭 ALFWorld 评测） | 回答"SPADE 训练出的能力能不能迁移到具身文本智能体" | ★★★ **先做这个**，1 周内出信号 |
| **B. 固定环境基线** | 写 `alfworld_adapter.py`，用 `fixed_env_orchestrator` 在 ALFWorld 上跑 GRPO | 中（1 个 adapter + 数据管线） | 拿到"ALFWorld 上的 GRPO 基线"，是 A 的对照组，也是后续 C 的必备件 | ★★★ 必做，是 C 的地基 |
| **C. SPADE for text-worlds** | 改造设计器 prompt + `env_validator`，让它生成 **ALFWorld 风格**的文本环境；agent 学，真 ALFWorld 当留出集 | 大（prompt + 校验规则 + hint 设计 + reward） | **真正的论文级增量**：adaptive curriculum 在稀疏奖励具身任务上是否有效 | ★★ 主力工作 |
| **D. 用 TextWorld 程序化生成** | 不用设计器，改用 TextWorld 的生成 API 造 ALFWorld 式任务 | 中 | 只是"更多固定环境"，丢掉 SPADE 的核心思想 | ✗ 不推荐单独做，可当 C 的对照 |

**推荐组合：A → B → C。** A 花一周验证方向，B 建地基并拿到对照组，C 才是要写论文的部分。

---

## 5. 分阶段计划

### 阶段 0：环境搭建（三期 H100 × 8）

SPADE 官方 Slurm 启动器 `slurm/train_spade_slurm_apptainer.sbatch` 的要求：

```bash
#SBATCH --gpus=8 --cpus-per-task=64 --mem=1900G --nodes=1 --time=03:00:00
```

- **容器**：`apptainer build slime.sif docker://slimerl/slime:v0.3.0`（内含 Slime + SGLang + Megatron-LM）
- **必须初始化 submodule**：`git submodule update --init --recursive`（4B 配方会 source `slime/scripts/models/qwen3-4B-Instruct-2507.sh`）
- **Ray 的 `TMPDIR` 必须在节点本地盘**，不能放 NFS/GPFS，否则 dashboard 起不来（报 `Module MetricsHead failed to start`）。三期 `/tmp` 是 15T 本地盘 ✓
- 三期 **无外网**，容器必须在**二期**构建好（或从别处拉）再传过去

### 阶段 1：跑通论文 Games 设定 @ 4B

```bash
bash cmd/games/train_spade_4b.sh          # SPADE
bash cmd/games/train_fixed_rlve_4b.sh     # 固定环境基线（对照）
```

先复现原设定，确认 pipeline 通了、指标对得上，再动 ALFWorld。**不要跳过这步**——SPADE 栈很重，先排除环境问题。

### 阶段 2：接 ALFWorld（路线 B）

ALFWorld 纯文本版**不需要 GPU、不需要 OpenGL**，只用 CPU：

```bash
pip install alfworld[full]      # python 3.9+
alfworld-download               # PDDL + game files + 检测器 → ~/.cache/alfworld/
```

然后写 `spade/core/envs/alfworld_adapter.py`，照 `rlve_adapter.py` 的范式实现：

- `reset()` → 返回文本观测 + `admissible_commands`
- `step(action)` → 文本观测 + reward（0/1）
- **用 `admissible_commands` 约束动作空间**——这是让稀疏奖励 RL 收敛的关键，直接用会好训很多
- hint 定义：ALFWorld 的 PDDL 能推出专家计划，**hint = 下一步正确动作（或整条计划）**，正好接上 `hint_generator.py`

任务类型共 6 类（pick_and_place / look_at_obj_in_light / pick_heat_then_place / pick_cool_then_place / pick_clean_then_place / pick_two_obj_and_place），可以按类型分 curriculum。

### 阶段 3：SPADE 化（路线 C）

改造点：
1. **设计器 prompt**：从"写数学博弈"改成"写 ALFWorld 风格的文本世界"，观测格式对齐真 ALFWorld
2. **`env_validator.py`**：加文本世界的结构校验（必须是可导航的房间图、物品可寻、任务可解）
3. **hint 语义**：生成的文本环境里，hint 必须是"可执行的正确动作序列"
4. **评测**：真 ALFWorld 当**留出集**，只评测不训练 → 这是论文最看重的泛化证据

---

## 6. 三期落地：要从二期传什么

三期无外网，所有东西都得在二期准备好再 rsync 过去（实测 **120–190 MB/s**，见 `CLUSTER_GUIDE.md`）。

| 需要的东西 | 来源 | 体积 |
|---|---|---|
| Qwen3-4B-Instruct-2507 HF 权重 | **二期已有** `~/thinktool/models/qwen3_4b_inst_2507` | 7.6G |
| `Qwen3-4B-Instruct-2507_torch_dist` | 在二期用 slime 转换器从上面转出来 | ~15G |
| `slime.sif` 容器 | 二期 `apptainer build` 或 `docker pull` 后转 | 10–20G |
| SPADE 仓库 + slime submodule | 二期 `git clone --recurse-submodules` | ~1G |
| ALFWorld 数据 | 二期 `alfworld-download` | 3–4G |
| SPADE 官方 4B checkpoint | 二期从 hf-mirror 下（`HF_ENDPOINT=https://hf-mirror.com`） | ~8G |
| 接地语料 `CORPUS_FILE` | **论文没发布**，得自己造 JSONL | 自定 |

转换 Megatron 格式（在二期容器里做）：

```bash
cd slime && source ../cmd/models/qwen3-8B.sh   # 4B 用 slime/scripts/models/qwen3-4B-Instruct-2507.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint ${MODEL_ROOT}/Qwen3-4B-Instruct-2507 \
    --save ${MODEL_ROOT}/Qwen3-4B-Instruct-2507_torch_dist
```

转移手法用 `tools/xfer.sh`（见下节）。

---

## 7. 风险与坑（提前知道，别踩）

1. **论文的语料没开源。** `CORPUS_FILE` 必须自己造，而论文明确说"接地语料"是关键组件之一。**你没法严格复现论文数字**，只能复现方法。
2. **ALFWorld 是稀疏奖励 + 长时域**，SPADE 原设定的环境是短时域、密集反馈的。GRPO + 二值奖励方差极大，很可能训不动。→ 先上 `admissible_commands` 约束 + 按任务类型分课程，必要时加过程奖励。
3. **hint 的定义是成败关键。** 论文整个训练信号都建立在 hint-based regret 上。ALFWorld 的 hint 怎么定义（下一步动作？完整计划？部分计划？）直接决定成败，值得单独做消融。
4. **AIME / BFCL 评测数据都不随仓库发布**，得自己准备。
5. **Ray `TMPDIR` 不能放共享盘**，否则 Ray dashboard 起不来。
6. **登录节点不能跑这个。** 必须 8 卡整机，按 `slurm/train_spade_slurm_apptainer.sbatch` 的规格提交。
7. **三期 `~/output` 已堆到 406G**，训练前先按"只留一个断点"的规则清理，`OUTPUT_DIR` 指到固定目录并加 `--save_total_limit 1`。

---

## 8. 下一步（如果要动手）

1. 在**二期**把 SPADE 仓库 clone 下来（含 submodule），跑 `cmd/games/train_spade_4b.sh` 的 `--help` / dry-run，确认栈完整
2. 在**二期**拉 `slimerl/slime:v0.3.0` 建 `.sif`，同时下 ALFWorld 数据和 4B checkpoint
3. 用 `tools/xfer.sh` 把上面全部推到**三期**
4. 三期上先复现 4B games 数字（阶段 1）
5. 写 `alfworld_adapter.py`，跑通路线 B
6. 路线 C 的设计器 prompt 改造

---

## 参考

- 论文 <https://arxiv.org/abs/2608.19197> ｜ 代码 <https://github.com/spade-rl/spade> ｜ 项目页 <https://spade-rl.github.io>
- 模型与数据 <https://huggingface.co/spade-rl>
- ALFWorld <https://github.com/alfworld/alfworld>（ICLR 2021, arXiv:2010.03768）
- 依赖栈：Slime <https://github.com/THUDM/slime> + SGLang + Megatron-LM
