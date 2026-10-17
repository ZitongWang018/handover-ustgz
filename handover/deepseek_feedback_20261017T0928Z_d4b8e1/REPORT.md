# 交接：30B-A3B 基线评测失败（vLLM 不支持 MoE+LoRA）＋ 结果与论文对照

- **会话 ID**：`d4b8e1`｜**UTC**：2026-10-17T09:28Z（本地 2026-10-17 17:28 CST）
- **结论先行**：**30B-A3B 基线评测无法在现有 vLLM 0.8.5 上运行**，8 个 rank 全部同一错误。
  4B 基线已完成，其数字**只能与 SkillRise 论文对照（且不是复现）**，
  **不能与 Explorer-Zero 论文的 17.14 比较**。
- **是否改实现**：**没有**。本轮只提交任务、读日志、重算统计。

## 1. 失败事实（原始证据）

| 项 | 值 |
|---|---|
| 作业 | `13133474` `ez-base30b` |
| sacct | `FAILED 00:05:40 ExitCode 1:0 NodeList gpu1-42` |
| 命令 | `bash selfplay/scripts/run_eval.sh results/base30b/contract.json`（`--gres=gpu:8`） |
| 日志 | `~/ztwang/logs/ez-base30b-13133474.out` / `.err` |

**`.out` 中的原文（行号）**
```
logs/ez-base30b-13133474.out:143  ERROR 10-17 02:54:07 [core.py:396] ValueError: Qwen3MoeForCausalLM does not support LoRA yet.
（同一条错误在 :166 :189 :212 :235 … 共出现 8 次）
```
**`.err` 中**
```
RuntimeError: Engine core initialization failed. See root cause above.
one or more evaluator ranks failed; merge was skipped
```

**⇒ 8/8 rank 均失败于引擎初始化阶段**，评测从未开始。

## 2. 根因（两条，缺一不可）

1. **vLLM 0.8.5.post1 的 `Qwen3MoeForCausalLM` 不支持 LoRA**
   （抛错位置 `vllm/v1/worker/lora_model_runner.py`）。
2. **`enable_lora=True` 在评测路径里是硬编码**：
   `selfplay/src/ez_vllm_policy.py:100` `enable_lora=True,`
   —— 没有 CLI、没有 config 开关。因此**连 `--base-only`（不需要 adapter 的基线臂）
   也会被强制走 LoRA 加载路径并失败**。

这是 `PLAN_30B_A3B.md` §1.3 中 **R2** 的实测确认。

## 3. 两条可行修复路径（**均需维护方决定 / 或用户明确授权**）

| 方案 | 改动 | 优点 | 缺点 |
|---|---|---|---|
| **A（推荐）** | 让 `enable_lora` 依「是否有 adapter」条件化：base-only → `False`；训练 arm → `True` | 30B 能用 **vLLM** 跑；**与 4B 同后端、同协议签名，可直接比** | 需改代码（约 1 处条件判断）；且 `enable_lora` 属于协议签名相关字段，需确认不加进 `protocol_signature` |
| **B** | 把 `selfplay/config.yaml` 的 `eval.backend` 由 `vllm` 改为 **`hf`** | **代码已支持**（`eval_runner.py:137` 允许 `{"vllm","hf"}`；`:218` 走 `HFPolicy`，不需要 LoRA）；零代码改动 | ① 合同里的 `backend` 会进 `protocol_signature`（`eval_runner.py:179`）⇒ **签名改变，不能与 4B 的 vllm 结果直接比**；② HF 逐 token 生成，8 卡 700 episode 会**明显更慢** |
| C | 升级 vLLM 到支持 MoE+LoRA 的版本 | 根治 | **集群无外网**，需离线 wheel，短期不可行 |

**注**：A 与 B 都会影响协议身份，**都必须为该臂单独准备合同并单独评测基线**，
不能复用已有的 `results/base30b/contract.json`。

## 4. 已完成的 4B 基线结果（可直接引用）

来源：`~/ztwang/results/base/contract.merged.json`（`episode_weighted_sr = 0.32`），
本报告用 `rows` 重算逐类型：

| 类型 | n | won | SR |
|---|---:|---:|---:|
| pick_and_place_simple | 175 | 133 | **76.0%** |
| look_at_obj_in_light | 65 | 26 | **40.0%** |
| pick_heat_then_place_in_recep | 80 | 18 | **22.5%** |
| pick_clean_then_place_in_recep | 135 | 27 | **20.0%** |
| pick_cool_then_place_in_recep | 125 | 15 | **12.0%** |
| pick_two_obj_and_place | 120 | 5 | **4.2%** |
| **6 类 macro** | 700 | | **29.1%** |

episode 加权 SR = **32.0%**；逐 seed `0.393/0.343/0.293/0.307/0.264` ⇒ **0.320 ± 0.050**。

## 5. 与论文的对照结论（**关键，请勿误引**）

### 5.1 只能"对照"、不能称"复现"：SkillRise（arXiv 2607.26784）Table 1 · ALFWorld · Qwen3-4B · ReAct

| 类型 | SkillRise ReAct | 本文 4B 基线 | 差 |
|---|---:|---:|---:|
| Pick | 45.5 | 76.0 | **+30.5** |
| Look | 30.3 | 40.0 | +9.7 |
| Clean | 21.4 | 20.0 | −1.4 |
| Heat | 12.5 | 22.5 | +10.0 |
| Cool | 18.1 | 12.0 | −6.1 |
| Pick2 | 23.5 | 4.2 | **−19.3** |
| **Avg（6 类 macro）** | **25.0** | **29.1** | **+4.1** |

**不可声称复现**，理由（均为可核对的事实）：

- SkillRise 原文未披露其 ReAct 的 prompt 全文、是否给"当前可用动作"菜单、回合上限、
  生成上限、temperature/top_p 取值。
- 本方协议（`selfplay/config.yaml` 固定）：**每步注入合法动作菜单（剔除 `help`）**、
  `max_turns 30`、`max_new_tokens 1024`、`temperature 0.7`、`top_p 0.95`、
  `valid_seen` 全量池 140 任务 × 5 种子。
- 逐类型差异量级（Pick +30.5 / Pick2 −19.3）**远大于**协议未披露项可能解释的范围，
  更可能反映**动作菜单注入**与**任务实例不同**的影响。
  ⇒ 只能报"在共享 ReAct 设定下的自测基线"，**不能写成 SkillRise 复现值**。

### 5.2 **不能**比较：Explorer-Zero Table 1（ALFWorld）

| | EZ 论文 | 本文 |
|---|---|---|
| 基线值 | **17.14 ± 5.7** | 29.1 macro / 32.0 episode |
| split | **`valid_unseen`** | **`valid_seen`** |
| 动作格式 | **`<tool_call>` JSON**（Appendix G） | **`<action>` 标签** |
| 每任务采样 | 4 samples（Table 1 表注） | 5 seeds |
| 结构 | 论文另报 "Ours 85.07 ± 1.6" | 未训练基座，尚无训练臂 |

⇒ **split、动作协议、采样口径三重不同，任何直接相减都无意义。**
  （仓库 `README.md` 亦已明确："不声称复现 EZ 原协议的绝对分数"。）

### 5.3 完全没做

SkillRise 的 **Tau-bench（Retail/Airline）** 与 **WebShop** 两列未评测（本项目只做 ALFWorld）。

## 6. 我做了什么 / 没做什么

**做了**（均为用户明确批准的执行动作）：
1. 提交 30B 基线评测（`sbatch --wrap`，未新建文件）
2. 读日志、`sacct`、从 `rows` 重算逐类型统计
3. 写本交接文件

**没做**：
- 未修改任何源码/配置（`enable_lora`、`eval.backend` 均未动）
- 未重试该失败作业（§4）
- 未触碰他人目录；`results/base/`（4B）只读、未覆盖
- 未推 main；本反馈在 `deepseek-feedback/d4b8e1`

## 7. 需要维护方决定

1. **选 A 还是 B**（见 §3），以及是否需要为 30B 单独定义协议身份的处理方式。
2. 若选 A：`enable_lora` 是否应计入 `protocol_signature`？
   （不计入则 30B 与 4B 可直接比；计入则两臂不可比。）
3. 30B-A3B 的 `gpu_memory_utilization` 需给出建议值：**评测 0.85 × 80G = 68G，
   而权重约 57~61G**，KV/激活余量很小——即使解决 LoRA 问题，**仍可能 OOM**（R1）。
4. `results/base30b/contract.json`（sha `7f177287…`）是否作废并重建。
