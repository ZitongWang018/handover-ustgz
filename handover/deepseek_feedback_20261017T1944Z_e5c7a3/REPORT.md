# 交接：30B-A3B baseline 实测无效（指示模型＋裸文本）＋ 训练代码 review

- **会话 ID**：`e5c7a3`｜**UTC**：2026-10-17T19:44Z（本地 2026-10-18 03:44 CST）
- **基线提交**：`9c3ea6b`（本报告基于该 commit；集群工作树已同步到该版本）
- **用户明确决定**：**使用 `Qwen3-30B-A3B-Instruct-2507`（指令版），不用 base 版。**
- **本轮是否改实现**：**没有**。只提交评测、读日志、重算统计、跑测试、做数值等价验证。

---

## 1. 30B-A3B baseline 已跑通，但**结果无效**

### 1.1 运行事实

| 项 | 值 |
|---|---|
| 作业 | `13159927` `ez-base30b-v2` |
| sacct | **`COMPLETED 00:22:43 ExitCode 0:0`**（节点 gpu1-23） |
| 合同 | `results/base30b-v2/contract.json`，`contract=8e58ce30…`，140 tasks × 5 seeds |
| 结果 | `results/base30b-v2/contract.merged.json`（395,883,989 B） |

```
merged 700 episodes; SR=0.0843
```

| 指标 | 30B-A3B | 对照：4B（`qwen3_4b`） |
|---|---:|---:|
| episode 加权 SR | **8.43%** | 32.00% |
| 6 类 macro | **8.3%** | 29.1% |
| 逐 seed episode SR | 0.0643 / 0.0571 / 0.0857 / 0.1143 / 0.1000 ⇒ **0.084 ± 0.023** | 0.320 ± 0.050 |

逐类型（30B）：Look 20.0%｜Pick 17.7%｜Pick2 4.2%｜Clean 3.7%｜Cool 3.2%｜**Heat 1.2%**

**修复生效**：`a074f5c` 的 `enable_lora = arm["kind"] == "checkpoint"` 已上集群
（`eval_runner.py:209`），base-only 不再走 LoRA 路径，EngineCore 正常启动、无 OOM。

### 1.2 ★ 为什么这个数字无效（trace 实证）

从 `contract.merged.json` 的 `rows[*].trace[*].raw_output` 取实际输出（700 行全部含 trace，
共 20,046 步）：

```
step0（won=False）: " For example, if you want to go to the sinkbasin 1, you should write:
                      <action>go to sinkbasin 1</action>\n\n"
step0（won=True） : " For example, your response should be in the format:/nStep 1: ... \n
                      Step 2: ...\n...\nFinal decision: <action> go to coffeetable 1 </action>\n\n"
```

**模型在逐字复读提示词里的示例/格式说明，而不是回答问题。**

配套统计：

| 指标 | 值 | 含义 |
|---|---:|---|
| `action_tag_valid` 率 | **0.994** | 标签格式齐备 ⇒ **格式对、语义全错** |
| `truncated` 率 | 0.004 | 不是长度截断问题 |
| 总步数 | 20,046 | 30 步封顶正常走完 |

### 1.3 根因

本协议（`selfplay/src/protocol.py`，与 4B 共用）向模型喂**原始纯文本**、
`add_special_tokens=True`、**不套 chat template**——这是 **base 模型**的用法。

- **4B**：`qwen3_4b` 是 **base** ⇒ 天然按 few-shot 风格续写并产出动作 ⇒ 32.0% ✓
- **30B**：`qwen3-30b-a3b-inst-2507` 是 **instruct** ⇒ 未套 `<|im_start|>…<|im_start|>assistant`
  时，模型把整段提示当成"待续写文本"⇒ 复读示例 ⇒ 8.43% ✗

**⇒ 8.43% 不反映 30B-A3B 的能力，是「指示模型 × 裸文本提示」的协议错配。**

### 1.4 用户已定：**继续用 instruct**

⇒ 必须为 instruct 模型引入 chat template。这属于**协议变更**，影响面：

1. `protocol.py` 的提示构造（`add_special_tokens` 与模板包装）
2. `protocol_signature` 里的 `prompt_source_sha256` / `tokenizer_files_sha256` 会变
3. **所有既有基线合同与结果全部失效**（4B 的 `results/base/` 含 700 episode 也要重做）
4. 需要同时明确 **thinking 模式开关**（`enable_thinking`）：
   若保留 thinking，`<action>` 会出现在 `<｜end▁of▁thinking｜>` 之后，投影逻辑必须能定位；
   若关闭，需确认模板/参数的正确关闭方式。**本报告不替维护方选择，只列出影响。**

**替代路径（供参考，本报告不主张）**：换用 `Qwen3-30B-A3B-Base`。
集群当前**没有**该权重（`~/thinktool/models/` 只有 `30b-a3b-inst-2507` 与
`qwen3-30b-a3b-inst-2507`，两者均为 instruct），且集群无外网，需离线获取。

---

## 2. 训练代码 review（`9c3ea6b`）

### 2.1 做了什么（按提交自身描述与代码核对）

新增 `--memory-profile qwen3_moe_cpu --backend hf`：
8 个数据并行 rank，每卡一份**冻结 BF16 基座 + FP32 LoRA**；**采样与反向共用同一份 HF 模型**
（不启第二份 vLLM 基座，也不依赖 vLLM 0.8.5 的 MoE LoRA）；AdamW 动量与计数常驻 CPU，
按单参数临时更新回传；反向/梯度自检用 `torch.autograd.graph.save_on_cpu`。

新增文件：`cpu_adamw.py`(167) / `training_memory.py`(195) /
`tests/test_30b_training.py`(114) / `tests/test_cpu_adamw.py`(103)。

### 2.2 客观验证（我在集群上跑的，命令见 `commands.txt`）

| 检查 | 结果 |
|---|---|
| 全量测试 `pytest selfplay/tests/ -q` | **101 passed, 1 skipped** ✓ |
| **CPU AdamW 与 `torch.optim.AdamW` 数值等价** | 20 步后最大逐元素差 **0.000e+00（bit-exact）** ✓ |
| 优化器 state 是否真在 CPU | ✓ `state_dict()` 内所有张量 `device=cpu` |
| profile 前置校验 | 严格：要求 `backend=hf` / `model_type=qwen3_moe` / `dtype=bfloat16` / `lora=True` / `lora_targets=all-linear` / `gradient_checkpointing=True` / `world_size=8`，不满足直接抛错 ✓ |
| **超参是否被擅改** | **没有**。`training_memory.py` 只改 4 项：`train.generation_max_batch_size` 2→1（唯一超参变化，已披露）、`runtime.memory_profile`、`runtime.optimizer_state_device=cpu`、`runtime.activation_offload=True`，并返回 backend=`hf`。**G=16 / GP=8 / TPS=1 / 4 loops×(25+25) / lr=1e-4 / clip=0.3 / KL=0.001 / LoRA r32·α32 / w_sub·w_div=0.3 全部未动** ✓ |

**结论：实现本身可信**（数学等价、超参未动、前置校验严格、测试覆盖到 CPU AdamW 与 30B profile）。

### 2.3 三个仍然断链的地方（都不是这份代码的 bug，是链路问题）

**① 评测口径断链（维护方自己在 REPORT 里也写了）**

> "不能将 HF checkpoint 和原 vLLM baseline 强行按同签名比较。"

30B 训练走 **HF**，而 30B baseline 是用 **vLLM** 跑的 `results/base30b-v2/`。
⇒ 训练完成后**无法与现有 baseline 配对比较**。需要**另做一套 HF 口径的 base-only baseline**
（同一 HF solver 设置），再与新 checkpoint 在同一 HF 设置下配对。

**② `run_experiment.sh` 不能用于该 profile**

维护方明确：**只用 `run_train.sh`**；`evaluate_run.py` 的 vLLM 四快照路径
**不适用于该 profile**（vLLM 0.8.5 仍不能加载 MoE LoRA checkpoint）。

**③ 速度与资源完全未实测**

CPU offload + HF MoE 逐专家 + `save_on_cpu` + 每步 H2D 参数拷回
⇒ 维护方原话："不得据 CPU 检查承诺 vLLM 速度或三天内完成"。
另外 4 份快照各含完整优化器状态、all-linear adapter 与 CPU 动量都很大，
需按 `runtime/memory_rank*.json` 实测后预留磁盘与整机 RAM。

---

## 3. 我做了什么 / 没做什么

**做了**（均为用户批准的只读或执行动作）：
1. 同步 `9c3ea6b` 到集群工作树（`git archive` + tar，未做 git 级 reset/clean）
2. 提交 30B baseline 评测并等待完成；读 `sacct`/日志；从 `rows` 重算逐类型与逐 seed
3. 从 `rows[*].trace[*].raw_output` 提取实际模型输出做诊断
4. 在集群跑 `pytest selfplay/tests/ -q`（101 passed）
5. **CPU AdamW 数值等价测试**（对比 `torch.optim.AdamW`，20 步，0.000e+00）
6. 核查 `training_memory.py` 的 profile 变更字段与前置信

**没做**：
- 未修改任何源码、配置、提示模板、解析器、启动脚本
- 未重试失败运行；未启训练
- 未删除/移动任何既有文件；`results/base/`（4B）只读
- 未碰他人目录；未推 main（本反馈在 `deepseek-feedback/e5c7a3`）

## 4. 需要维护方决定

1. **instruct 的 chat template 与 thinking 开关**：模板形态、是否 `enable_thinking=False`、
   `<action>` 投影如何与 thinking 共存。这会改变 `protocol_signature`，
   **4B 与 30B 的所有基线必须在新协议下重做**。
2. **评测后端统一**：为了让训练前后可配对，建议 30B 的训练与评测统一到 **HF**；
   是否同时把 4B 也迁到 HF（否则 4B 的 vLLM 结果与新协议不可比）。
3. **30B 训练是否会撞显存/内存**：`gpu_memory_utilization` 之外的 HF 侧占用、
   4 份快照与 CPU 动量的磁盘峰值，需实测（本报告未测）。
4. `results/base30b-v2/`（vLLM 跑的无效 baseline）与 `results/base30b/`（更早的失败尝试）
   如何处置——**我不删除，等指示**。
