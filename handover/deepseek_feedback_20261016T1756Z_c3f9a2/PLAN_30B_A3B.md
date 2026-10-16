# 复现计划：Qwen3-30B-A3B 版（供 GPT 生成代码用）

- **交接会话 ID**：`c3f9a2`
- **UTC 时间**：2026-10-16T17:56Z（本地 2026-10-17 01:56 CST）
- **用途**：给维护方（GPT）生成「评测代码 + 训练代码」的**输入规格**。
  代码必须**沿用仓库现有 ReAct 框架的写法**（`selfplay/` 下 `evaluate.py` / `train.py` /
  `run_eval.sh` / `run_train.sh` 同一套风格），**只是换模型**。
- **本文件只陈述事实与要求，不含我自创的实现代码。**

---

## 0. 现状（已确认的事实）

1. GitHub `origin/main` 最新 = **`d615200`**（`fix(ez): isolate rollout engines and
   preserve four evaluated snapshots`，2026-10-16 07:08 +0800）。
   该提交已做：**rollout 引擎隔离**（修此前 vLLM EngineCore 启动失败）、
   **四个 outer 快照保留与逐个评测**（`selfplay/evaluate_run.py` 487 行 +
   `selfplay/scripts/run_experiment.sh` 117 行 + `selfplay/src/training_runtime.py` 249 行）。
2. 服务器（二期 HKUST-GZ，用户 `xyu608`）上已跑通的流程：
   - **基线评测**（4B）：`python selfplay/evaluate.py prepare --base-only ...` +
     `bash selfplay/scripts/run_eval.sh <contract>` → 完成
     （`valid_seen` 全量池 140 任务 × 5 种子 = 700 episode）。
   - **训练**（4B）：`bash selfplay/scripts/run_train.sh --baseline-contract <合同> ...`
     → 曾因 vLLM EngineCore 启动失败，后由 `d615200` 修复引擎隔离。
3. 仓库当前 `README.md` 明文：**换模型/协议后必须重新准备并运行基线**，
   旧合同不能沿用。⇒ **换 30B-A3B 后，基线合同与基线结果必须重做。**
4. 评测/训练两臂必须使用**同一份 `selfplay/config.yaml`** 固定协议；
   训练入口会把 proposer/引擎上下文升到 40960（solver 保持 32768），
   该 resolved config **不能**用于准备评测合同。

## 1. 模型信息

### 1.1 目标模型

| 项 | 值 | 状态 |
|---|---|---|
| 模型名 | **Qwen3-30B-A3B-Instruct-2507**（用户指定，替换 4B） | — |
| 集群路径（上次 2026-10-16 01:45 实测 `~/thinktool/models/` 里有这些条目） | `30b-a3b-inst-2507`、`qwen3-30b-a3b-inst-2507`（**两者都可能存在，须核实哪个是真实目录、哪个是别名/链接**） | **待核验** |
| 完整路径候选 | `/hpc2hdd/home/xyu608/thinktool/models/qwen3-30b-a3b-inst-2507` | **待核验** |
| 文件清单（safetensors 分片数、config.json、tokenizer 文件、总大小） | 未取得（集群当时网络不可达） | **待核验** |

### 1.2 模型特性（Qwen3-30B-A3B-Instruct-2507 公开规格，供代码适配参考）

- **MoE**：总参数 30B，**每 token 激活 3B**；48 层，128 路由专家 + 共享专家，
  每 token 激活 top-8 专家。
- 隐藏维 2048；注意力 32 Q heads / 32 KV heads，head_dim 128。
- 词表 151,936（Qwen2.5 系列 tokenizer）。
- 原生上下文 256K（YaRN）。**支持 thinking 模式**（`enable_thinking` 开关）。
- **bf16 权重约 61GB**。
- 推理速度接近 3~8B 稠密模型（激活参数量小），但**显存占用按 30B 全量算**。

### 1.3 换模型带来的具体风险点（代码生成时必须处理）

| # | 风险 | 说明 |
|---|---|---|
| R1 | **VRAM** | bf16 61GB。二期是 **A800 80G**。评测 `gpu_memory_utilization: 0.85` 只剩 ~19G 给 KV/激活，**可能 OOM**；训练时 HF 模型 + vLLM 引擎**同卡**（`0.35`），61GB×2 侧 + 引擎，**非常紧张**。需实测并给出安全值或改造方案。 |
| R2 | **vLLM 0.8.5.post1 对 Qwen3-MoE 的支持** | 需要验证 `Qwen3MoeForCausalLM` 能否在 v0.8.5 下正常加载 + `enable_lora`（**LoRA on MoE**）是否可用。 |
| R3 | **LoRA target 爆炸** | 现配置 `lora_targets: all-linear`。MoE 有 128 路由专家 × 多模块，`all-linear` 会挂到极多模块上，参数/显存/速度都可能失控。需给出 30B-A3B 下合理的 target（或沿用 all-linear 并实测）。 |
| R4 | **thinking 模式** | instruct-2507 有 `enable_thinking`。本框架用**纯文本 ReAct 提示 + `<action>` 标签**，thinking 若默认开启会污染输出格式。需在 tokenizer/chat 配置层明确**关闭**或确认默认关闭。 |
| R5 | **40960 上下文 + MoE 的 KV cache** | 训练 vLLM 引擎 `max_model_len: 40960`，30B-A3B 的 KV cache 按 32 KV heads × 48 层计算，需实测是否放得下。 |
| R6 | **tokenizer 一致性** | 合同会哈希 tokenizer 文件；30B-A3B 的 tokenizer 与 4B 不同，属**正常更换**，但必须整组一致（base inventory 会校验）。 |

## 2. 路径与目录结构（全程只在用户自己的 `ztwang` 下）

**唯一授权根目录**：`/hpc2hdd/home/xyu608/ztwang`（用户 `xyu608`，本期确认路径；
历史路径 `/home/ztwang` 无效）。**禁止**写 `~` 下其它位置、`/tmp`、任何他人目录。

```
/hpc2hdd/home/xyu608/ztwang/
├── selfplay/                        # 代码（沿用现有框架，见 §3/§4 的改动点）
│   ├── evaluate.py
│   ├── evaluate_run.py              # d615200 新增：训练目录四快照评测
│   ├── train.py
│   ├── config.yaml                  # 共享协议配置（换模型不改协议字段）
│   ├── scripts/
│   │   ├── run_eval.sh
│   │   ├── run_train.sh
│   │   └── run_experiment.sh        # d615200 新增：训练+四快照评测组合流程
│   ├── src/                         # protocol/ez_solver/ez_proposer/training_* ...
│   └── runs/                        # 训练 run 目录（每 run 一个子目录）
├── results/                         # 评测结果（base=基线；每个 run 的评测放 run 目录内）
│   └── base/                        # 30B-A3B 基线合同 + 合并结果（**需重做**）
├── logs/                            # sbatch 日志
└── handover/                        # 交接材料（本文件所在）
```

**固定数据/模型路径**（只读引用，禁止写入/修改）：
- ALFWorld 数据根：`/hpc2hdd/home/xyu608/.cache/alfworld`
- 模型目录：`/hpc2hdd/home/xyu608/thinktool/models/`（只读；30B-A3B 具体子目录**待核验**）

## 3. 第一部分：评测代码（沿用 ReAct 框架）

### 3.1 要求

沿用 `evaluate.py prepare` + `scripts/run_eval.sh` 的现有协议，**只做两件事**：
1. `--base` 指向 30B-A3B 路径。
2. 处理 §1.3 的 R1/R2/R4/R6（显存安全值、vLLM MoE 支持验证、thinking 关闭、tokenizer 整组一致）。

**协议本身不变**（这些是 `config.yaml`/合同固定的口径，不许改）：
- split `valid_seen`、种子 `[0,1,2,3,4]`、`shards: 8`、`batch_size: 32`
- solver：ReAct 风格纯文本提示（先 reasoning 再 `<action>...</action>`）、
  `add_special_tokens=True`、最近 15 组历史、合法动作菜单（剔除 `help`）、
  `max_turns 30`、`temperature 0.7`、`top_p 0.95`、`max_new_tokens 1024`、
  `max_context_tokens 32768`、`stop_mode action_tag`、`menu_enabled true`

### 3.2 运行方式（与 4B 完全同构）

```bash
cd /hpc2hdd/home/xyu608/ztwang
export SKR_PYTHON=/hpc2hdd/home/xyu608/miniconda3/envs/thinktool/bin/python
export PYTHONPATH=$PWD/selfplay/src:$PWD/selfplay/third_party/py310deps
export ALFWORLD_DATA=/hpc2hdd/home/xyu608/.cache/alfworld
export TMPDIR=/hpc2hdd/home/xyu608/ztwang/.tmp
export PIP_CACHE_DIR=/hpc2hdd/home/xyu608/ztwang/.pipcache

python selfplay/evaluate.py prepare \
  --config selfplay/config.yaml \
  --alfworld-data /hpc2hdd/home/xyu608/.cache/alfworld \
  --base /hpc2hdd/home/xyu608/thinktool/models/<30B-A3B-实际目录> \
  --base-only \
  --results-dir results/base30b \
  --output results/base30b/contract.json

sbatch --account=root --partition=i64m1tga800u8ka --nodes=1 --ntasks-per-node=1 \
  --gres=gpu:8 --cpus-per-task=64 --mem=400G --time=0-06:00:00 \
  --job-name=ez-base30b \
  --output=/hpc2hdd/home/xyu608/ztwang/logs/ez-base30b-%j.out \
  --error=/hpc2hdd/home/xyu608/ztwang/logs/ez-base30b-%j.err \
  --wrap="cd /hpc2hdd/home/xyu608/ztwang && export SKR_PYTHON=... TMPDIR=... \
          PYTHONPATH=... && bash selfplay/scripts/run_eval.sh results/base30b/contract.json"
```

> 注：结果目录用**新名字**（`results/base30b`），**绝不覆盖** 4B 的 `results/base/`。

## 4. 第二部分：训练代码（沿用 ReAct 框架）

### 4.1 要求

沿用 `train.py` + `scripts/run_train.sh`，训练超参**保持论文 Table 3 口径**（不许改）：
- 4 outer loops；每轮 proposer 25 步 + solver 25 步
- GRPO `G=16`、`tasks_per_step=1`、proposer 的 solve-rate 采样 `G_P=8`
- LoRA `r=32, α=32`；AdamW `lr=1e-4`；`clip_eps=0.3`；`KL β=0.001`
- `w_sub=0.3, w_div=0.3`；proposer 上下文 40960，solver 32768
- 数据来自 ALFWorld `train` split

**换 30B-A3B 后必须额外处理**：§1.3 的 R1~R6。
其中 **R3（LoRA target）** 需要 GPT 给出明确决定：沿用 `all-linear` 还是给
30B-A3B 定制 target 列表；**R1（VRAM）** 需要给出 A800-80G 下训练可行的
`gpu_memory_utilization` 实测安全值。

### 4.2 运行方式（与 4B 完全同构；必须绑 30B 基线合同）

```bash
bash selfplay/scripts/run_train.sh \
  --base /hpc2hdd/home/xyu608/thinktool/models/<30B-A3B-实际目录> \
  --alfworld-data /hpc2hdd/home/xyu608/.cache/alfworld \
  --baseline-contract results/base30b/contract.json \
  --backend vllm \
  --run-id explorer-zero-30b-1017-XXXX
```

提交方式同前：`sbatch --wrap`（8 卡、`i64m1tga800u8ka`、日志落在
`/hpc2hdd/home/xyu608/ztwang/logs/` 内、时间上限需 GPT 按 30B 的实测步速给建议，
4B 参考值：12h 可能不够）。

### 4.3 四快照评测（GPT 已在 d615200 实现，**不需要新写**，只需适配 30B）

```bash
bash selfplay/scripts/run_experiment.sh \
  --run-id explorer-zero-30b-1017-XXXX \
  --run-dir /hpc2hdd/home/xyu608/ztwang/selfplay/runs/explorer-zero-30b-1017-XXXX \
  --base /hpc2hdd/home/xyu608/thinktool/models/<30B-A3B> \
  --alfworld-data /hpc2hdd/home/xyu608/.cache/alfworld \
  --baseline-contract results/base30b/contract.json \
  --baseline-results results/base30b/contract.merged.json \
  --backend vllm
```

评测结果放在 run 目录内独立子目录；四个 outer 快照串行评测；每个 outer 完成后
与 30B 基线 `compare`。

## 5. 依赖与环境（已就位，勿重建）

- Python：`/hpc2hdd/home/xyu608/miniconda3/envs/thinktool/bin/python`（3.10.19）
- 关键依赖（基线评测已实测通过）：`alfworld 0.3.3`、`textworld 1.6.2`、
  `torch 2.6.0`、`transformers 4.57.3`、`peft 0.18.0`、`vllm 0.8.5.post1`、`numpy 1.26.4`
- 离线 wheel 仓库：`selfplay/third_party/wheels/`（94 个 wheel）
- planner shim：`selfplay/third_party/glibc232/`（`libdownward.so` +
  `libglibc_single_threaded_shim.so`，已恢复并验证可用）
- **若 30B-A3B 需要额外依赖**（如 flash-attn 特定版本、moe 相关）：**离线安装受限**，
  新 wheel 需提前放 `selfplay/third_party/wheels/`，且**安装必须用
  `--no-index --find-links=... --target=py310deps`**，缓存/临时目录压进 `ztwang` 内
  （`TMPDIR`/`PIP_CACHE_DIR` 指向 `ztwang/.tmp`、`ztwang/.pipcache`）。
- **不要**把 `skillrise/` 或其它目录塞进 `PYTHONPATH`（历史教训：会遮蔽 `alfworld`）。

## 6. 给 GPT 的任务清单（改哪些 / 不改哪些）

**必须交付**：
1. 30B-A3B 的**基线评测**（重做合同 + 5 种子结果）——代码沿用现有入口。
2. 30B-A3B 的**训练** + **四个 outer 快照逐个评测** + 与基线 `compare`。
3. 对 §1.3 R1~R6 的**逐项处理决定**（VRAM 安全值、vLLM MoE 验证结论、
   LoRA target 决策、thinking 关闭方式、40960 KV 实测、tokenizer 整组一致）。
4. 一份**实测显存/步速报告**（首步耗时 → 总时长估算 → sbatch `--time` 建议）。

**禁止**（不改协议口径）：split/种子/步数/温度/top_p/max_new/menu/stop_mode/
训练超参/奖励口径 一律沿用现有值；换模型只允许动「模型路径」与「资源类参数
（显存利用率、LoRA target、批量、上下文上限的**资源适配**，且需在报告中说明理由）」。

**边界**：所有读写只在 `/hpc2hdd/home/xyu608/ztwang` 内；模型目录与数据目录只读；
`results/base/`（4B 基线）不覆盖、不删除；新建目录用新名字。

## 7. 待核验清单（服务器网络当前不可达，以下未取得）

1. 30B-A3B 的实际目录名与文件清单（`qwen3-30b-a3b-inst-2507` 与 `30b-a3b-inst-2507`
   哪个是实体/别名；safetensors 分片、总大小、config.json 内 `model_type`）。
2. 二期 A800-80G 节点当前可用性与排队情况。
3. 4B 基线结果（`results/base/contract.merged.json`）是否仍在集群上（供对照参考，
   **30B 基线必须独立重做**，不能沿用）。
4. `d615200` 的新代码是否已同步到集群（本地已 fetch；集群同步需网络恢复后按
   AGENTS.md §3 的 fast-forward 规则操作）。
