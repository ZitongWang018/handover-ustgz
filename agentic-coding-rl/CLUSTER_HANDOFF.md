# 交给集群执行 AI：八卡 A 系列 GPU 的 GRPO / GRPO＋DIS / Score Centering 对照

## 1. 本次任务与当前状态

目标是在**同一计算节点分配的 8 张 NVIDIA A 系列 GPU**上，先验证三条 SWE 训练链路，再运行 GRPO、GRPO＋DIS、Score Centering 三臂对照。三者共用模型、任务、长度、采样与评测预算；各方法的 loss/normalization 差异按下文记录。当前用户尚未确认具体 GPU 型号/显存，以及计算节点的容器支持，必须先检查，不能把“八卡 A”自行认定为 A800 80GB。

仓库：<https://github.com/ZitongWang018/handover-ustgz>，子目录 `agentic-coding-rl/`。最低功能版本：`6e5bef1e26c49075e4b7afe56211460f272ec3e9`（已含 GRPO＋DIS）；使用更晚提交时确认该提交仍是其祖先，记录实际 HEAD。

已完成的是实现准备与 CPU 验证：GRPO＋DIS/启动器的 31 项测试、100 组随机梯度对照通过；没有实际 GPU 训练或 SWE 分数。此计数不代表已经运行 Score Centering 的全部上游数值/分布式测试。本次执行要补上目标机器的运行证据。详细实现说明：[GRPO_DIS.md](GRPO_DIS.md)。

本计划做的是**本仓库统一协议下的基线复现**。当前 `grpo` 是记录 rollout 概率作分母的 bypass GRPO；不是 SAO 原文保留 latest old policy 的标准 GRPO。`grpo_dis` 与其共用 GRPO advantage，替换为 DIS 目标。第三臂 `score_centering` 使用固定 Miles 内的现成实现，当前默认是 **Score Centering＋MIS**，不是关闭 importance weighting 的纯 centering 消融。不得将这些结果标为 SAO 或 Score Centering 原论文绝对分数复现。

## 2. 执行授权与工作边界

先读仓库根 [AGENTS.md](../AGENTS.md)，再读本文件。由用户明确要求按本文执行时，任务范围为：在用户指定的真实 `ztwang` 根目录内准备本项目运行配置、必要依赖/数据/模型转换产物、调度作业脚本和运行记录；申请并使用指定的八卡资源；执行下面三臂的 smoke 和已授权的正式训练，以及第 9.1 节中同一 allocation 内的 Score Centering 验证；保存证据并反馈。

- 用户必须提供或确认：真实 `ztwang` 根目录、本仓库位置、本地模型位置、队列/GRES/账号、资源与时长预算。历史 `/home/ztwang` 不构成确认。
- 源码、tracked 配置、算法、奖励、测试和训练协议保持只读。仅生成本次独立的 `*.local.json`、运行目录和作业脚本。路径、当前节点地址、实际 GPU UUID、已核准容器引用等站点字段可以按事实填写；研究超参数沿用第 3 节。
- 用户给定的模型若不是默认模型，不得只换 checkpoint 路径后运行。先反馈模型类型、并行、parser/TITO 与资源配置的适配需求。
- 不安装系统服务，不提权，不清理其他任务，不修改全局环境，不批量 kill/prune。正常结束只关闭本作业自己启动的服务；没有针对其他运行的取消授权。
- 同步仓库仅在确认干净、无覆盖风险后 fast-forward。遇到错误保留现场，不现场改实现“跑通”。按根 AGENTS 的交接要求向 `deepseek-feedback/<唯一ID>` 分支追加反馈，不直接推 main。
- 只收到文档链接而没有执行指令时，不将文档本身当作提交作业或安装依赖的授权。

## 3. 仓库现有默认值：三臂共用的资源与预算

权威来源：[configs/local.example.json](configs/local.example.json)、[launcher.py](launcher.py)。不要根据下面的说明重新手写一套不一致的默认配置；应复制当前模板，填写站点字段。

| 项目 | 默认值 / 含义 |
|---|---|
| 模型 | Qwen3-4B-Instruct-2507；本地 HF checkpoint |
| 训练 / 评测数据 | SWE-Gym Lite / SWE-bench Verified 全量；`eval_limit=0` 表示不截取 |
| Agent | Harbor 的 mini-swe-agent |
| 调度模式 | CLI 默认 `async`，同一节点 4 卡 actor＋4 卡 rollout；没有 critic |
| 并行 | actor TP=2、PP=1、CP=1、EP=1；每个 rollout engine 1 卡 |
| 每个任务采样数 G | 8 |
| 每批采样任务数 | 2，即每批 16 条进入训练的轨迹 |
| 全局 batch | 16；正常无跳过时每批一个 actor optimizer update |
| 训练批数 | `num_rollouts=100`，正常对应约 100 次 actor update |
| 单臂名义训练量 | 100×16=1600 条进入更新的轨迹；不含额外生成、丢弃、重采样和评测 |
| actor 学习率 | `1e-6`，constant schedule |
| 优化器 | Adam，beta1=0.9、beta2=0.98，weight decay=0.1 |
| 训练采样 | temperature=1、top_p=1、top_k=-1 |
| 轨迹长度预算 | `max_seq_len=16384`，包括 prompt / action / observation |
| 单次生成上限 | `max_response_len=2048`；不是 2048 轮，也不是整条轨迹总长 |
| GRPO clipping | epsilon low=0.2、high=0.28；对应普通 PPO-style clipping |
| GRPO＋DIS 边界 | ratio 严格处于 `(0.2, 4.0)` 才保留 PG 项；不使用上一行 clipping |
| Score Centering | K=128；IS 模式为 MIS，边界 `[0.5, 5.0]`；不叠加通用 TIS/MIS hook |
| 正则化 | loss-level reference KL 系数 0.001；reward-level KL=0；entropy=0 |
| 保存频率 | 每 10 批；`--smoke` 改为每批保存 |
| 评测频率 | 加 `--eval` 才启用；训练前＋每 20 批，单题单 rollout |
| 评测采样 | temperature=0、top_p=1、top_k=-1 |
| 随机种子 | 训练及 rollout seed=42；不是多训练种子实验 |
| 异步限制 | 最大 weight staleness=4；finished buffer 容量系数=2 |
| Harbor 并发 / timeout | 并发 16；server agent timeout=3600 秒，client timeout=7200 秒 |
| 训练容器 | 模板为 `radixark/miles:latest`；运行前记录并固定获准的实际镜像身份 |

`critic_lr`、`critic_warmup_rollouts`、`gae_lambda` 等字段虽然在模板里，但**不作用于本次三臂**。不要因为看见这些字段就声称已经运行 critic 或 SAO。

Score Centering 另有方法定义上的差异：保留组内减均值、关闭组内标准差归一化，使用 per-token loss；GRPO / GRPO＋DIS 保留现有的组内标准差和默认轨迹均值归约。其 PG 是独立的 score-centering loss，不用 PPO clip。它还自动启用 FP32 LM head 并存储生成时的 top-K 概率。这是完整 recipe 间的对照，不是只开关 centering 一项的消融；不要隐去这些差异。

当前没有显式设置成 300 次 agent interaction 的上限。时间上限、单次生成上限、累计上下文上限分别是不同限制。不要向用户报告“默认训练 100 步”时把 optimizer update、rollout batch、agent turn 混为一谈。

## 4. 与 SAO 原文相同和不同的地方

下表只比较论文中明确给出的项目，不推测未公开设置。[SAO §3.1、§4.1–4.2](https://arxiv.org/html/2607.07508v1)

| 项目 | 本仓库当前默认 | SAO 原文 coding 设置 |
|---|---|---|
| 模型 | 4B Instruct | Qwen3-30B-A3B-Thinking-2507 |
| 每批轨迹 | 16 | 128；GRPO 为 16 prompts×8 |
| Scaffold / 长度 | mini-swe-agent / 16K | OpenHands / 128K，最多 300 轮 |
| actor LR / GRPO 组大小 | `1e-6` / 8 | 相同 |
| DIS ratio 区间 | `(0.2, 4.0)` | 对应其 coding 的 epsilon low=0.8、high=3.0 |
| 评测 temperature | 0 | 1 |

论文 SAO 本身为 G=1 并使用 critic；本次两条 GRPO 基线以及 Score Centering 都保持 G=8，不应为了“贴近 SAO”把它们的 G 改成 1。

结论：**有默认步数和超参数，但只是适合先验证的小规模 pilot；整体没有与 SAO 原文对齐。** 100 批也不是从论文获得的统一停止步数。论文的 coding 训练任务清单、完整运行依赖等信息不足以据此确认与本仓库一致。

此次不要擅自切 30B、128K、OpenHands，也不要只把数字改大后叫“原文复现”。先在同一 4B 协议下验证三种基线。未来 SAO 的比较应共用这一协议，或者在正式更换协议后重跑全部主对照。

## 5. 提交前必须取得的集群事实

把下列结果写入本次运行记录；不需要读取其他用户目录或凭据。

1. **调度器**：是否 Slurm；允许的 partition、account/QOS、八卡 A 系列 GRES 名称、节点时间/CPU/RAM 上限。本文的 Slurm 示例不适用于未知调度器。
2. **实际硬件**：分配到的 GPU 型号、每卡显存、8 张 GPU 的身份、驱动版本；记录拓扑。只在获分配节点上做 GPU 检查，不在登录节点训练。
3. **容器**：能否使用管理员允许的 Docker＋NVIDIA GPU runtime＋Docker Compose；容器中能否看到且只看到本次分配的 8 张 GPU。Docker daemon 和 sandbox 的使用必须符合站点规定。
4. **路径/网络/磁盘**：模型、数据、repo、cache、run 都在真实根目录内；本地模型匹配 4B Instruct-2507；HF/镜像是否可下载；Docker 数据盘及项目目录容量足够；task 容器能访问当前计算节点的 session/router 地址。

**容器判断是硬前提。**当前 `in_container.sh` 调用 Docker，Harbor 使用 Docker Compose 创建 SWE sandbox。只有 Apptainer/Singularity、没有 Docker 权限时，不能直接执行这套入口。反馈“需要容器/环境适配”，不要自行安装 Docker、改成 sudo、把 Docker 命令机械替换成 Apptainer 或私改源码。

不能只凭“A100 支持”推断当前镜像可运行。镜像可能带较新 CUDA/依赖，必须在实际 A 系列节点做容器 GPU probe。CPU 测试使用的 torch 2.6.0+cpu 不是训练依赖，**不得安装进 GPU 训练环境**。

## 6. 准备模型、数据和固定版本

命令均在经用户确认的项目路径执行；环境与下载准备遵循用户允许的范围。已有准备产物先核对，不覆盖。

```bash
cd /USER_CONFIRMED_ZTWANG_ROOT/handover-ustgz/agentic-coding-rl
git remote -v
git status --short
git rev-parse HEAD
# 确认干净且用户允许同步后，才进行 fast-forward 更新。

cp -n configs/local.example.json configs/cluster.local.json
# 填写 workspace_root、paths、实际分配的 GPU、已核准镜像、当前节点可达地址。
# 原有配置已存在时先检查，不能用模板覆盖它。

bash scripts/setup_harbor.sh configs/cluster.local.json
.deps/harbor/.venv/bin/python run.py prepare-data --config configs/cluster.local.json --kind train
.deps/harbor/.venv/bin/python run.py prepare-data --config configs/cluster.local.json --kind eval
```

上游代码由 `UPSTREAM.lock.json` 固定。Harbor 安装在项目下独立 `.venv`。bootstrap/依赖与数据准备可能需要网络；不能自动下载模型。`prepare-data` 不带 limit 才是默认全量，失败保留生成目录并交接，不能静默删掉失败任务。

模型转换占用 GPU，在获分配资源内执行：

```bash
export SWE_RL_CONFIG="$PWD/configs/cluster.local.json"
bash scripts/in_container.sh convert
```

如果已有匹配的 Megatron checkpoint，跳过转换，记录其身份。转换目标已存在时入口会拒绝覆盖；不能为绕过此检查删除已有目录。

`runtime.session_external_host` 必须是**当前计算节点、task 容器能访问到的地址**，不是登录节点地址。动态分配的节点可能每次不同，应在 allocation 内完成站点配置并验证。`node_ip=127.0.0.1` 是本节点 Ray 配置，不能据此把 task 访问的 external host 也填成 loopback。

## 7. 八卡作业的组织与提交

建议每次申请单节点 8 卡，先在一个 allocation 中顺序运行三个 smoke；通过后，再在足够时长的 allocation 中顺序运行三个正式臂。不要同时让三臂各自占用同一组 8 卡，也不要把 4 张 actor 卡误当成总申请卡数。

当前仓库提供的是单机训练入口，**没有已核准的站点专用 sbatch 文件**。执行 AI 可以在本次独立 run 目录中生成作业脚本，按下面的结构填写站点字段；这不授权改训练实现。

Slurm 提交形状（占位值必须先核实）：

```bash
sbatch \
  --partition="CONFIRMED_A_SERIES_PARTITION" \
  --gres="CONFIRMED_EIGHT_GPU_GRES" \
  --nodes=1 --ntasks=1 \
  --cpus-per-task="CONFIRMED_CPU_BUDGET" \
  --mem="CONFIRMED_RAM_BUDGET" \
  --time="CONFIRMED_WALLTIME" \
  --output="EXISTING_UNIQUE_RUN_DIR/slurm-%j.out" \
  "APPROVED_JOB_SCRIPT"
```

站点要求时再添加实际 account/QOS。不要在 `#SBATCH` 注释中使用希望 shell 展开的 `$VARIABLE`。submission 输出目录必须事先创建，作业脚本、配置路径均需解析并留在根目录内。

**GPU 映射尤其需要核对**：Slurm 的 `CUDA_VISIBLE_DEVICES` 可能是重映射编号，而 `docker --gpus device=...` 使用宿主机 GPU 身份。确认 `SLURM_JOB_GPUS` / `SLURM_STEP_GPUS` / UUID 与 allocation 一致；优先使用本次获分配 GPU 的 UUID 填 `runtime.gpus`。不能直接照抄模板 `0,1,2,3,4,5,6,7` 后碰到其他人的卡。`runtime.num_gpus=8` 和 `async_actor_gpus=4` 保持默认。

作业脚本应依次完成：

1. 记录 job ID、节点、真实根目录、HEAD、配置哈希、实际 GPU/驱动、资源和镜像身份。
2. 在当前 allocation 内完成并核对当前节点的 local config，设置绝对路径 `SWE_RL_CONFIG`。
3. 启动本作业自己的 Harbor 服务并保存 stdout/stderr，等 `/health` 成功；失败不继续训练。
4. 运行 `doctor --gpu`；正式臂还需 `--eval` 使其检查评测数据。
5. 顺序运行第 8 节命令；任何一个命令失败就停止后续臂，保留日志。
6. 正常结束或收到作业信号时，只关闭自己创建的 Harbor 服务。`run.py serve` 会再启动服务子进程，不能只杀 wrapper 父 PID；用本作业独立进程组管理并核对归属，不能按 `python`/`ray`/`sglang` 名称全局杀进程。

不要在登录节点提前启动 Harbor，再让未知计算节点用 `127.0.0.1` 连接它。不要把服务与训练分到不同节点却继续使用单节点网络配置。

## 8. allocation 内的准确运行命令

先启动服务（在作业中作为本作业自己的受控后台服务；日志路径由独立 launch 目录指定）：

```bash
.deps/harbor/.venv/bin/python run.py serve --config "$SWE_RL_CONFIG"
```

服务健康后，运行检查与三臂 smoke。Score Centering 的原生数值/协议验证要求见第 9.1 节。若要自动化，应将上面的服务放入受控后台，而不是让前台命令阻塞后面的训练：

```bash
bash scripts/in_container.sh doctor --gpu

bash scripts/in_container.sh train \
  --algo grpo --mode async --run-id g-smoke-UNIQUE --smoke
bash scripts/run_grpo_dis.sh \
  --mode async --run-id d-smoke-UNIQUE --smoke
bash scripts/in_container.sh train \
  --algo score_centering --mode async --run-id sc-smoke-UNIQUE --smoke
```

`--smoke` 保留正常模型、batch、长度、G，只将每臂各缩成 2 批并每批保存；不加 `--eval`，避免 smoke 先跑完整 Verified。不能自行把 G 改成 1 来“省卡”，那会改变 GRPO。

三臂 smoke 均通过第 9 节的检查后，使用新 run ID 执行已授权的 100 批正式对照：

```bash
bash scripts/in_container.sh doctor --gpu --eval

bash scripts/in_container.sh train \
  --algo grpo --mode async --run-id g-s42-UNIQUE --eval
bash scripts/run_grpo_dis.sh \
  --mode async --run-id d-s42-UNIQUE --eval
bash scripts/in_container.sh train \
  --algo score_centering --mode async --run-id sc-s42-UNIQUE --eval
```

`UNIQUE` 必须换成本次短 ID。路径过长会超过 Ray UNIX socket 限制，`cache_root` 保持短路径，run ID 也保持简短。三个正式臂使用同一配置文件、checkpoint、数据和镜像；每一臂结束释放 GPU 后才启动下一臂。

**评测预算不能忽略**：默认 100 批、每 20 批全量评测，加训练前评测，常规约 6 轮、3000 条 Verified 评测轨迹/臂；三臂合计约 9000 条，实际以日志触发次数为准。这不属于各臂的 1600 条名义训练轨迹（三臂合计 4800 条）。两步 smoke 不能直接推算包含全量评测的作业时长。Score Centering 还有候选概率传输、存储和 FP32 head 的额外开销，不能假定其单步耗时与 GRPO 相同。

若用户希望只做首尾评测，可明确批准三臂共同把 `eval_interval` 改为 100；当前默认未改。其他缩小 batch、截取数据、改变长度、降低 G、跳过评测等也都需要明确说明，不能作为执行 AI 的默认排错措施。

当前没有自动续训入口。超时/失败后不能直接重用旧 run ID，不得把换个 ID 从头训练称为 resume。保留 checkpoint、记录实际完成批数，交接后续恢复需求。

## 9. 什么算跑通，什么算复现证据

smoke 至少应同时看到：

- 三臂均有真实 SWE agent 多轮轨迹；task ID、工具交互、grader 输出与 reward 对得上，而不是只启动了 Ray。抽查代表轨迹并保留各臂的全部实际日志。
- 正常进入 actor backward/optimizer；loss、梯度有限，发生了预期训练更新；有 checkpoint 和训练 dump。
- GRPO group 保持 8；检查混合成功/失败组是否存在。大量全零奖励不是崩溃，但不能据此说算法有效。
- DIS 日志含 kept/low-mask/high-mask 指标且值合理；没有 NaN、概率行错位或双重 TIS 修正。指标为 Miles 聚合口径，不把单 microbatch 的中间和误当最终全局比例。
- Score Centering 通过第 9.1 节检查，实际候选概率字段与 action/observation 行对齐；不能只因 loss 有值就认定 centering 已正确生效。
- `manifest.json` 记录正确算法、配置、数据哈希、代码/插件哈希、镜像身份；GRPO＋DIS 有独立 algorithm details。

本次三臂都保留同样 KL 系数 0.001、entropy=0。DIS mask 只作用于 PG 项；区间外 token 仍可能有 KL 正则梯度，这是当前明确定义的行为，不应误判为 DIS 未屏蔽。

正式结果至少报告：各自完成的批数和训练轨迹/token、有效/丢弃比例、policy lag、训练/评测用时、Verified 解决数及分母、初始模型结果、最终 checkpoint 与评测绑定关系。区分模型失败与 Docker/agent/grader 基础设施失败；不能把基础设施故障静默计为正常未解决，发现这种情况先保留证据反馈。

当前只跑 seed=42，能形成一组配对 pilot；不能据此宣称多种子稳定提升。100 批后不显著也不能直接否定长程方法，需要先检查模型能力、有效奖励和长程/滞后是否实际发生。

### 9.1 Score Centering 的额外验证

依据固定 Miles 的 [原生说明](https://github.com/radixark/miles/blob/ee3f3d4650bc7d901c736f81297b3ec524159fb0/examples/infra_features/score_centering/README.md)，该方法需要**实际生成时**的候选概率，不是事后用新权重重算。顺序为数值/数据管线检查、稳定端点协议检查、真实 SWE smoke，再做正式评测。

**数值和管线测试**：在获准的训练镜像/匹配环境中、固定 `.deps/miles` checkout 下运行。下面 `python` 指该训练环境的解释器，不是宿主机 Harbor venv；缺少 pytest 或上游依赖时先记录并反馈，不擅自改变训练环境的依赖版本。

```bash
cd "$PROJECT/.deps/miles"
python -m pytest \
  tests/fast/backends/training_utils/test_score_centering.py \
  tests/fast/backends/training_utils/test_score_centering_pipeline.py
```

记录实际 collected/passed/failed/skipped 数量和完整输出。文件存在、未执行、全 skip 都不算通过。检查包含独立梯度 oracle、constant-reward score correction 和多轮数据传递；top-K 尾部近似本身不保证任意分布下精确抵消，不能把近似误差简单判为实现错误。

**真实服务协议**：在本次 allocation 内准备与训练相同依赖和概率设置、权重保持不变的 SGLang 端点，串行运行探针。不能在三臂训练占满 GPU 时额外启动抢占显存的模型，也不能边更新权重边做温度一致性比较。端点需要同时支持 `/generate` 和 `/v1/chat/completions`，不是 Harbor 的 21100 端口或带 session 前缀的 URL；实际地址由执行 AI 从自己启动的服务日志确认。

```bash
# 在获准的匹配训练环境、.deps/miles 目录中执行。
export MILES_LIVE_SCORE_CENTERING_ENDPOINT="http://CONFIRMED_ENGINE_HOST:CONFIRMED_PORT"
export MILES_LIVE_SCORE_CENTERING_MODEL="/CONFIRMED_ZTWANG_ROOT/models/Qwen3-4B-Instruct-2507"
export MILES_LIVE_SCORE_CENTERING_SERVED_MODEL="CONFIRMED_SERVED_MODEL_NAME"
export MILES_LIVE_SCORE_CENTERING_ARTIFACT_DIR="/CONFIRMED_ZTWANG_ROOT/UNIQUE_RUN_DIR/sc-protocol"
python -m pytest --confcutdir=tests/manual tests/manual/test_score_centering_live.py
```

服务端应保留 `SGLANG_RETURN_ORIGINAL_LOGPROB=0`，使用匹配的 SGLang/router 协议并确认 FP32 LM head 生效。探针会生成少量短响应，检查 native/OpenAI 概率元数据及不同温度；必须保存响应材料。若失败，反馈具体依赖版本/响应差异，不能改成关闭 centering 或不采集候选概率来“通过”。

**SWE smoke 的数据与指标**：

- 确认生成请求实际采集 K=128，而不仅是启动命令带了参数；`rollout_topk_token_ids`、`rollout_topk_log_probs` 和 sampled-token `rollout_log_probs` 存在并正确对齐。
- observation/padding 的无效候选占位不进入训练校正；重试、多轮合并和截断后的行数与 response length 一致；概率来自实际生成分布。
- 日志出现 `sc_correction`、`sc_train_head_mass`、`sc_rollout_head_mass`、`sc_tail_ratio`、`sc_importance_weight` 及 train/rollout mismatch 指标，值有限，结合头部概率质量解释尾部近似。不是要求每个指标必须非零或必须随训练单调变化。
- `score_centering_is=mis`，MIS 边界保持 `[0.5, 5.0]`；不叠加 `--use-tis` 或 DIS；保留组内减均值、关闭标准差归一化，并按原生 per-token 口径聚合。
- K=128 的 ID/logprob 数组约增加 1 KiB/response position（未计传输副本），FP32 head 也有额外开销。OOM 时保存证据，不私自降低 K、batch 或长度后混入原三臂对照。

将结果分开报告：“上游数值检查”“live 概率协议”“真实 SWE smoke”“100 批训练/Verified 评测”。任何一层未执行或失败，都不能被后一层的 reward/loss 日志替代。

## 10. 最终向用户交付什么

给出简洁状态表和可核验路径，不只发“任务已提交”：

| 项目 | 必需内容 |
|---|---|
| 源码与设置 | HEAD、上游 SHA、实际 local config 的安全快照/哈希、模型身份、数据/task manifest |
| 集群运行 | job ID、节点、GPU 型号/显存/UUID、开始/结束时间、状态、实际资源、日志路径 |
| 三臂证据 | smoke/正式各自 run ID；实际批数；manifest、train.log、trace、checkpoint 路径 |
| Score Centering | 原生测试数量/结果、live 探针响应、候选概率与多轮对齐检查、SC 指标、额外资源成本 |
| 评测 | 初始/各检查点解决数与总数；完整 Verified 或子集；基础设施错误；评测配置 |
| 未完成项 | 等待、失败、超时、还在运行的作业及其证据；不要补造字段 |
| GitHub 交接 | 按根 AGENTS 新增独立反馈包和反馈分支，附 commit 链接；不上传模型和大轨迹 |

建议报告正文开头直接回答：“三臂 smoke 是否通过；Score Centering 数值/协议是否验证；正式各完成多少批；Verified 是否测完；哪里失败；下一步需要维护方处理什么。”未发生的训练/评测明确写“未运行”。
