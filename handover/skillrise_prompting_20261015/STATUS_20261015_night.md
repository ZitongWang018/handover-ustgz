# 交接：本轮状态说明 + 两个入口的区分（2026-10-15 深夜）

本文件是**状态交接**，不是结果报告。写它的直接原因：上一轮把**两条完全不同的评测链路
混在一起讲**，造成理解混乱。这里把"哪个入口是哪件事"钉死。

---

## 0. 最重要的一张表：两个入口，两件事，**不要混**

| | **SkillRise prompting 评测** | **Explorer-Zero（EZ）复现** |
|---|---|---|
| 目标论文 | 《SKILLRISE: Agentic RL for Cross-Task Skill Evolution》arXiv **2607.26784** | Explorer-Zero（EZ） |
| 目标 | ALFWorld 上 **ReAct** 与 **zero-shot** 两个 prompting 基线的成功率 | **基座**与**训练后 solver** 的结果 |
| split | `eval_in_distribution`（**valid_seen**） | **`valid_unseen`** |
| 入口 | `skillrise/eval_quick_both_1gpu.sbatch` | `selfplay/scripts/run_ez_eval_v2_8gpu.sh` |
| 评测器 | `skillrise/eval_quick.py`（自研） | `selfplay/src/ez_eval_v2.py`（GPT 版 v2） |
| 报的指标 | 6 类 macro（对应 SkillRise Table 1 的 Avg.） | 每任务 4 sample 的 SR 均值±标准差（对应 EZ Table 1） |
| 目标值 | ReAct **25.0** / Zero-shot **4.4** | 基座 **17.14 ± 5.7** / Ours 85.07 ± 1.6 |
| 模型（本次用） | `~/thinktool/models/qwen3_4b_inst_2507` | **未定**（GPT 标记为 `weights_unverified`） |

**两个入口的产出不可互相引用。** SkillRise 的 macro 数字**不能**当作 EZ 的基座结果，
反之亦然。GPT 在 `handover/explorer_zero_repro_plan_20261015/README.md` 里明确写过这一点。

## 1. 本轮新产出的数据（SkillRise 链路）

### 1.1 已完成：Zero-shot × 128 槽 × 2 种子

文件：`raw_2seed/zero_shot_128_2seed.json`（256 条 episode 明细）

| seed | macro |
|---|---|
| 0 | 16.4 |
| 1 | 21.3 |
| **mean ± std (n=2)** | **18.9 ± 3.5** |

对照上一轮 5 种子：**20.3 ± 3.9**（14.1 / 20.1 / 23.2 / 23.9 / 20.0）⇒ 在噪声范围内一致。

### 1.2 仍在运行（未完成）

作业 **13093588**（react）与 **13093589**（zero_shot），`EZ_SEEDS=0,1`，
每作业顺序跑「128 槽 × 2 种子 → 全 140 × 2 种子」。
截至本文件生成：zero_shot 已完成 128 段、进入 140 段；react 仍在 128 段第 2 个种子。
**本目录不含这两格的结果。**

## 2. ★ 本轮最重要的发现：**同一个 seed，两次跑结果不同**

| seed | 上一轮（5 个种子在同一进程内顺序跑） | 本轮（2 个种子在同一进程内顺序跑） |
|---|---|---|
| 0 | 14.1 | **16.4** |
| 1 | 20.1 | **21.3** |

同一份代码、同一份 128 槽清单、同一个模型、同样显式传了 `SamplingParams(seed=...)`，
**值仍然不同**。

**推断的原因（未做对照实验，属推断）**：vLLM 引擎的 RNG 状态在多个 seed 之间是
**连续推进**的。因此"5 种子序列里的 seed=0"与"2 种子序列里的 seed=0"
所对应的底层随机状态不同，采样结果随之分叉。

**由此得到的可操作纪律**：
1. `seed` 只保证**同一次运行内可重放**，**不保证跨运行、跨进程、跨卡数复现**。
2. **不同种子数量的运行结果不可混在一起比较。**
3. 报数字必须用 `mean ± std`，且必须写明**种子列表**与**同批跑了几个种子**。
4. n=2 的 std 不可信；正式数字至少 5 个种子。

这条与 GPT 在审计中的提醒一致（"seed 本身不证明底层 vLLM 每请求 RNG 行为"）。

## 3. 本轮修复的一个提交错误（记录以备复现）

提交 2 种子运行时，第一次用了：

```bash
sbatch --export=ALL,EZ_STYLE=$S,EZ_SEEDS=0,1 ...     # ← 错
```

**逗号被 Slurm 当作 `--export` 的字段分隔符**，`0,1` 被截成 `EZ_SEEDS=0`，
作业日志显示 `种子列表: [0]` —— 只跑了一个种子。

正确写法：

```bash
export EZ_SEEDS=0,1
sbatch --export=ALL ...                              # 走 shell 环境继承
```

**结论：`--export` 的值里不能出现逗号。**

## 4. 文件清单

| 路径 | 内容 |
|---|---|
| `raw_2seed/zero_shot_128_2seed.json` | Zero-shot × 128 槽 × 2 种子的 256 条 episode 明细（每条含 `seed`） |
| `raw_2seed/summary_2seed.txt` | 上述文件的汇总输出原文 |
| `raw/`（上一轮） | 5 种子版本的 640+640 条明细、`selection_128.json`、`summary_128.txt` |
| `ez_training_evidence/` | Explorer-Zero 训练 run 的原始日志（另一件事，见上一轮 README §9） |
| `README.md`（上一轮） | SkillRise 链路的完整协议、模型对照、128 槽本质、问题清单 |

## 5. 尚未完成 / 待确认

1. **react × 128 与两格的 140 段**仍在跑（见 §1.2）。
2. **EZ 链路一次都没跑过**：GPT 已交付 v2 训练与 v2 评测入口 + 37 个新测试（本地全绿：
   `pytest selfplay/tests/` = 115 passed / 1 skipped），但**尚未用真实模型在 GPU 上执行过**。
   执行前需先决定 `--base` 用哪个权重——GPT 明确拒绝替你决定，因为无法证明权重身份。
3. **`skillrise/EVAL_PROTOCOL.md` 已被 GPT 改写**：从"定稿协议"降级为
   "历史 quick diagnostic 记录"，并注明修正了此前对种子、Avg、128 槽、模型身份的过度推断。
   原 89 行版本见 commit `72f07f9`。
4. **集群与 GitHub 的代码不同步**：集群 `~/ztwang` 的 git HEAD 停在 `0758f1f`
   （另一条线，且有未暂存改动），`origin/main` 为 `1ffd794`。
   `hpc_tools/sync_ztwang.py` **会静默漏传新增文件**（本轮已实际发生，导致 10 个测试假失败）。
   **交付前必须逐文件核对，不要只信同步工具。**
