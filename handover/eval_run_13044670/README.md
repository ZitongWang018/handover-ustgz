# 交接：ALFWorld valid_unseen 基座评测 run 13044670

本目录用于**把一次完整评测运行的原始产物交给另一个模型独立审核**。
本文件只陈述可核验的事实，**不包含结论、判断或改进建议**。所有数字均抄自同目录的
`FACTS_manifest_summary.json` 与 `FACTS_sample_rows.tsv`，可直接比对。

---

## 1. 这次运行是什么

| 项 | 值 |
|---|---|
| 目的 | 在 ALFWorld `valid_unseen`（官方 `eval_out_of_distribution`）上评测 **基座模型**（不挂 LoRA adapter），作为复现工作的 baseline control |
| Slurm job id | `13044670` |
| Slurm 作业名 | `ez-evalfull` |
| 节点 | `gpu1-1`（8× A800 80GB，分片并行） |
| 提交时刻 | 2026-10-14 21:22 左右（CST） |
| 结果文件生成 | 2026-10-14 21:40 左右（CST） |
| 提交命令 | `sbatch --export=ALL,ZTWANG_MODEL_PATH=$HOME/thinktool/models/qwen3_4b,EZ_PROTOCOL=paper-approx,EZ_BASE_ONLY=1,EZ_BACKEND=vllm,EZ_SAMPLES=4 scripts/ez_eval_full_8gpu.sbatch` |
| 入口脚本 | `ez_eval_full_8gpu.sbatch`（本目录内附一份副本） |
| 评测主程序 | `selfplay/src/eval_ez.py` |
| 协议记账模块 | `selfplay/src/ez_eval_protocol.py`（本目录内附一份副本） |
| 代码版本 | git commit `643e2ae`（本次运行**未对该 commit 做任何修改**） |
| 输出目录（集群） | `~/ztwang/selfplay/eval_out/full_paper-approx-base-only-job13044670/` |

## 2. 协议 manifest（逐项原值）

来自 `FACTS_manifest_summary.json` 的 `protocol_manifest`：

```
protocol_requested                     paper-approx
protocol_effective                     paper-approx
faithfulness                           "paper approximation; Figure 14 says Example Omitted and the paper's sampling details are incomplete"
prompt_style                           ours
prompt_sha256                          0d3f4d9b1ba4165d6e0ad7e442813f7a4c98fe43c3d2f2fc66dce0379d129951
tokenizer_chat_template_sha256         a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8
enable_thinking                        False
tokenizer                              /hpc2hdd/home/xyu608/thinktool/models/qwen3_4b
base_model                             /hpc2hdd/home/xyu608/thinktool/models/qwen3_4b
checkpoint                             None
adapter                                None
base_only                              True
backend                                vllm
parser.type                            ez_json_tool_call
parser.strict                          False
parser.strict_is_diagnostic_only       False
resolver.type                          public_whitespace_normalization_plus_unique_resolver
resolver.strict                        False
resolver.menu_enabled                  True
resolver.menu_scope                    "initial prompt and execution feedback"
resolver.natural_command_recovery      True
fewshot.enabled                        False
fewshot.requested_flags.EZ_SOLVER_FEWSHOT        0
fewshot.requested_flags.EZ_SOLVER_FEWSHOT_SCHEMA 0
fewshot.schema_variant                 False
decode.temperature                     1.0
decode.top_p                           0.95
decode.top_k                           -1
decode.repetition_penalty              1.05
decode.max_new_tokens                  96
decode.max_steps                       30
decode.max_context_tokens              40960
```

`decode_effective`（后端**实际收到**的采样参数，归一化之后）：

```json
{"temperature": 1.0, "top_p": 0.95, "top_k": -1, "min_p": 0.0, "repetition_penalty": 1.05}
```

模型目录 `/hpc2hdd/home/xyu608/thinktool/models/qwen3_4b` 的 `README.md` 中含
`base_model: Qwen/Qwen3-4B-Base` 字样；`config.json` 的 `model_type` 为 `qwen3`，
`architectures` 为 `["Qwen3ForCausalLM"]`。**本目录未包含该模型卡的完整副本。**

## 3. 原始结果

### 3.1 总体（`FACTS_manifest_summary.json` → `summary`）

```
episode_weighted_sr                 0.05970149253731343
task_type_macro_sr                  0.0559715821812596
episode_bernoulli_std               0.23693295322966484
sample_suite_success_rates          [0.05223880597014925, 0.05223880597014925,
                                     0.07462686567164178, 0.05970149253731343]
sample_suite_sr_population_std      0.00913988709993723          (ddof=0)
sample_suite_sr_sample_std          0.010553832555023094         (ddof=1)
n_tasks                             134
n_episodes                          536
samples_per_task                    4
n_task_types                        6
```

`sample_suite_success_rates` 是 4 个「每 sample 的整集 SR」：第 i 个数 =
134 个任务各自第 i 次采样的成功率。

### 3.2 逐任务类型（`summary.task_type`）

| task_type | task_macro_sr | episode_weighted_sr | n_tasks | n_episodes |
|---|---|---|---|---|
| look_at_obj_in_light | 0.000000 | 0.000000 | 18 | 72 |
| pick_and_place_simple | 0.208333 | 0.208333 | 24 | 96 |
| pick_clean_then_place_in_recep | 0.032258 | 0.032258 | 31 | 124 |
| pick_cool_then_place_in_recep | 0.095238 | 0.095238 | 21 | 84 |
| pick_heat_then_place_in_recep | 0.000000 | 0.000000 | 23 | 92 |
| pick_two_obj_and_place | 0.000000 | 0.000000 | 17 | 68 |

### 3.3 分片

8 个分片，声明见 `FACTS_manifest_summary.json` → `shards`。
前 7 片各 `task_count=17 / episode_count=68`，第 8 片 `task_count=15 / episode_count=60`。

### 3.4 作业退出状态

`logs/job.out` 全文（共 421 字节）：

```
[eval-shard] rank=0 exit=0
[eval-shard] rank=1 exit=0
[eval-shard] rank=2 exit=0
[eval-shard] rank=3 exit=0
[eval-shard] rank=4 exit=0
[eval-shard] rank=5 exit=0
[eval-shard] rank=6 exit=0
[eval-shard] rank=7 exit=0
merged tasks=134 episodes=536 episode_weighted_SR=0.0597; wrote eval_out/full_paper-approx-base-only-job13044670/merged.json
[eval-full] result=eval_out/full_paper-approx-base-only-job13044670/merged.json
```

`logs/job.err` 为 **0 字节**。

## 4. 文件清单

| 文件 | 大小 | 内容 |
|---|---|---|
| `FACTS_manifest_summary.json` | 5.8 KB | 协议 manifest + decode_effective + 全部聚合统计（含逐类型、逐任务 sample mean）+ 分片声明。**这是拿去审核的主文件** |
| `FACTS_sample_rows.tsv` | 68 KB | 536 行 episode 明细，表头 `task_id / task_type / sample / won / steps / max_steps_reached / n_actions` |
| `FACTS_per_task.tsv` | 15 KB | 134 行，表头 `task_id / task_type / sample_mean / n_samples` |
| `run13044670_all.tar.gz` | 312 KB | 完整原始产物：`merged.json`（6.4 MB，单文件含 536 行的完整 trace）、8 个 `shard_*.json`、8 个 `shard_*.log`、`expected_shards.json`、job `.out`/`.err` |
| `logs/job.out` / `logs/job.err` | 421 B / 0 B | Slurm 作业级 stdout / stderr |
| `logs/shard_00.log` … `logs/shard_07.log` | 各 ~20 KB | 8 个分片各自的完整 stdout |
| `ez_eval_full_8gpu.sbatch` | 4.8 KB | 入口脚本副本（含环境变量开关清单） |
| `ez_eval_protocol.py` | 5.1 KB | 协议记账模块副本（`align_records_to_slots` / `aggregate_complete_evaluation`） |

**注意**：`merged.json` 里 `sample_rows` 每行含该 episode 的**完整动作序列**；
`FACTS_sample_rows.tsv` 只保留了长度（`n_actions`），原始动作序列需要解 tar 包。

## 5. 本目录**不**包含什么

- 模型权重、tokenizer 文件（只给了路径与 sha256）
- 训练侧产物（`ez_runs/` 下的 checkpoint、`train_log.rank0.jsonl`）
- 其它 run 的产物（早期 `ms30/ms50/ms100/rich` 等一批对照）
- 论文 PDF 或论文正文的截图/摘录
- 任何环境（`alfworld` / `textworld`）的版本号——**本 run 的 manifest 未记录软件版本**

## 6. 外部对照值（出处与核实状态）

| 值 | 来源 | 核实状态 |
|---|---|---|
| valid-unseen baseline `17.14 ± 5.7` | Explorer-Zero 稿件 Table 1 | **未由本次交接方核对原文 PDF**；该数字转录自项目内部文档 `docs/EXPLORER_ZERO_REPRO_PLAN.md`。稿件据内部记录为匿名 ICLR 2027 under-review |
| valid-unseen Ours `85.07 ± 1.6` | 同上 | 同上 |
| strict prompt weighted SR `3.7%` / macro `3.5%`、rich prompt weighted `10.4%` / macro `12%` | 第三方公开仓库 `Masoudjafaripour/VLM-VLA-Embodied-Agents` 的 `baseline/ALFWorld/alfworld_eval/RESULTS.md` | 由调研方读取该仓库得出；**该仓库使用与本次不同的 scaffold（rich rules + 编号菜单 + 50 步 + greedy + max_new_tokens=32 + 每任务 1 采样），与本次 run 的协议不同** |

## 7. 供审核者判断的待核实问题（**问题，不是结论**）

1. `paper-approx` 这个近似协议与论文真实评测设置之间，哪些项是确定的、哪些项是本次自行补全的？
2. `decode.max_steps = 30` 的依据是论文 Table 3 的训练回合数；评测口径是否也应是 30？
3. resolver 的 `menu_enabled=True`（初始 prompt 与每步反馈中都注入 admissible actions 菜单）
   在论文的 Figure 14 骨架里**没有被记录**，它对结果的影响方向与幅度是未知的。
4. `backend = vllm` 与 HF 原生解码之间是否存在可测量的差异？
5. 逐类型结果中 `pick_and_place_simple` 的 `episode_weighted_sr = 0.208333`
   与另外三类 `0.000000` 并存，这个分布形态是否与"未微调基座模型"的预期一致？
6. `episode_bernoulli_std = 0.2369` 与论文所报 `± 5.7`（单位与定义未核实）之间可否比较？

## 8. 时间线上的诚实记录

- 本次运行的代码 = commit `643e2ae`，**运行期间未改动**。
- 运行**之后**，交接方在本地修复了 `selfplay/tests/test_ez_eval_audit.py` 末尾一个
  字面 `\n` 导致的 `SyntaxError`（该文件在 `643e2ae` 中即已损坏，438 行测试无法被收集；
  已用 GitHub API 拉取原文比对确认是仓库内既有问题，非本地传输损坏）。
  修复后 `pytest selfplay/tests/` 为 `64 passed, 1 skipped`。
  **该修复不涉及本次评测所执行的任何代码路径。**
- 更早的同日运行（`ms30 / ms50 / ms100 / t0.4 / t0.6 / rich` 等一批对照）使用不同的、
  且当时已知有缺陷的解析与协议实现，其产物不在本目录内，**不建议与本目录数字并列**。
