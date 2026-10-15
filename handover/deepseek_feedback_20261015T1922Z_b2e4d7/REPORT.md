# DeepSeek 执行代理反馈：训练作业 vLLM EngineCore 启动失败

- **会话 ID**：`b2e4d7`
- **UTC 时间**：2026-10-15T19:22Z（本地 2026-10-16 03:22 CST）
- **状态**：**训练失败，已停止，未重试、未改配置。** 基线评测已完成（见 §5）。
- **是否改过实现**：**改过 1 处**，经用户明确授权，且**与本次失败无关**（见 §4）。

## 1. 本次授权（原文）

- "同步到集群，并且开始跑训练，记得中间存 4 个 ckpt，每个 ckpt 都要走一次评测"
- "你来帮我改吧，只需要修改点名称就行，如果怕撞名字可以用时间当后缀"（授权改实现，范围=断点命名）
- "写交接包，并且 kill 掉"

## 2. 失败事实（原始证据）

**作业**：`13098262` `ez-train`
**sacct**：`FAILED 00:22:44 ExitCode 1:0`；`rank 5 local_rank 5 exitcode 1`

**`.out` 关键行**
```
ERROR 10-16 03:17:58 [core.py:396] EngineCore failed to start.
```

**`.err` 关键行**
```
Exception raised from throwTimeoutError at /pytorch/torch/csrc/distributed/c10d/socket.cpp:1025
torch.distributed.DistNetworkError: The client socket has timed out after 600000ms
    while trying to connect to (10.1...)
```
同一错误在多处出现（不同进程地址），说明**多个 rank 同时**在 rendezvous 上超时。

**时间线**
| 时刻 | 事件 |
|---|---|
| 02:55 | 作业提交并启动 |
| 02:58:39 | 8 个 vLLM 引擎开始 `Initializing a V1 LLM engine` |
| 03:17:58 | `EngineCore failed to start.`（约 20 分钟后） |
| 03:18 | 作业以 exit=1 结束，总时长 22:44 |

**日志路径**（授权目录内）
- `~/ztwang/logs/ez-train-13098262.out`
- `~/ztwang/logs/ez-train-13098262.err`

## 3. 已通过的检查（说明失败点不在训练逻辑）

`selfcheck` 全部通过：
```
[selfcheck] base dtype=torch.bfloat16 | lora参数样例=[...q_proj.lora_A.proposer.weight] | gc=True
[selfcheck] facts 通道 OK（152 条事实）
[selfcheck] env slots: env_p=2 env_s=2 env_sr=8(lazy) → 本 rank 当前 4，懒建后 12；8 rank 全局 96
[selfcheck] 梯度链 proposer grad_norm=1.6301e+02 (504 个张量有 grad) | solver grad_norm=1.6354e+02 (504 个张量有 grad)
```
⇒ 模型、LoRA、事实通道、梯度链均正常；**失败发生在 vLLM 引擎启动阶段，训练循环从未开始**。

## 4. 我改的那 1 处（授权范围内，且与失败无关）

**文件**：`selfplay/src/ez_train.py`（`ez_save_v3` 内，`os.replace(tmp_dir, ckpt_dir)` 成功之后）
**改动**：**纯新增 10 行，0 删除**

```python
archive_dir = run_dir / f"ckpt.{tag}"
if archive_dir.exists():
    shutil.rmtree(archive_dir)
shutil.copytree(ckpt_dir, archive_dir, copy_function=os.link)
```

**目的**（用户要求"中间存 4 个 ckpt，每个都评测"）：原代码每次保存都写同一个
`run_dir/ckpt`，后写覆盖前写，只留最后一个。上面的追加把每次封存结果用
`os.link`（硬链接，几乎不占额外空间）另存为 `ckpt.{tag}`。

**为什么不影响本次失败**
- 该代码**只在 `ez_save_v3` 被调用时执行**；`ez_save` 的调用点位于
  `ez_train.py:1269`（proposer phase 每 5 步或 phase 结束）与 `:1460`（solver 同理）。
- 本次运行**连第 1 步都没有完成**，**任何保存点都没有到达**。
- 因此这段代码**从未被执行**。

**为可复核起见**：改动可用 `git diff` 与本地 `deepseek-feedback/b2e4d7` 分支比对；
`py_compile`（本地）与 `ast.parse`（集群端）均通过。

## 5. 同时完成的另一件事：基线评测（成功）

**作业** `13097438` `ez-base-eval`：**COMPLETED 00:38:03 ExitCode 0**

**结果**（未训练基座 `qwen3_4b`，SkillRise ReAct 协议，`valid_seen` 全量池，
140 任务 × 5 种子 = 700 episode）：

| 类型 | n | won | SR |
|---|---:|---:|---:|
| pick_and_place_simple | 175 | 133 | 76.0% |
| look_at_obj_in_light | 65 | 26 | 40.0% |
| pick_heat_then_place_in_recep | 80 | 18 | 22.5% |
| pick_clean_then_place_in_recep | 135 | 27 | 20.0% |
| pick_cool_then_place_in_recep | 125 | 15 | 12.0% |
| pick_two_obj_and_place | 120 | 5 | 4.2% |
| **6 类 macro** | 700 | | **29.1%** |

`episode_weighted_sr = 0.320`；逐 seed `0.393/0.343/0.293/0.307/0.264` ⇒ **0.320 ± 0.050**；
逐 seed macro `0.381/0.312/0.272/0.251/0.241` ⇒ **0.291 ± 0.056**。

**产出**：`~/ztwang/results/base/{contract.json, contract.merged.json, rank0..7.json}`

**口径提醒（重要）**：此结果**不可**直接与 Explorer-Zero 论文 Table 1 的
`17.14 ± 5.7` 比较——论文是 `valid_unseen` + `<tool_call>` JSON，本次是
`valid_seen` + SkillRise `<action>`。两者协议不同。

## 6. 我的判断（推测，非结论）

`selfcheck` 打印 `8 rank 全局 96` 个 env slot。训练为单节点 8 rank，每 rank 创建多个
vLLM 引擎，**推测**单节点上的引擎进程数/端口/显存压力导致 EngineCore 之间的
分布式连接无法在 600 s 内建立。

**我没有证据确认**此推测，**故不作为结论**。相关可调项（仅列出，未改动）：
`config.yaml` 的 `runtime.vllm.gpu_memory_utilization: 0.35`、
`runtime.world_size: 8`、`runtime.vllm.max_num_seqs: 64`、
`runtime.vllm.max_num_batched_tokens: 8192`，以及 `ez_train.py` 里 env slot 的懒建策略。

## 7. 我没有做的事（§4 要求）

- **没有重试**该训练（§4：不要反复重试）
- **没有修改任何配置**（§4/§2：改配置需明确授权）
- **没有修改训练/奖励/rollout 逻辑**（唯一改动见 §4，范围=断点命名，且未被执行）
- 没有推 main；本反馈走 `deepseek-feedback/b2e4d7`

## 8. 需要维护方决定

1. EngineCore 启动超时的根因：是引擎数量/显存/端口，还是代码里的启动顺序问题？
2. 若需调整任何配置（引擎数、`gpu_memory_utilization`、world_size 等），
   请明确文件与目标值——**我按 §2 不会自行决定**。
3. 基线评测的 29.1% macro / 32.0% episode 是否作为训练对照的正式基线？
