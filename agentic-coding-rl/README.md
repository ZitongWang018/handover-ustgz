# SWE 长程 RL 算法实验底座

用同一个 **Miles + SGLang + Megatron + Harbor** 管线做真实仓库修复任务。支持 GRPO、GRPO＋DIS、GSPO、TIS/MIS、Score Centering、单 rollout PPO；后续在 PPO 分支接 SAO。新目录独立于本仓库的 `selfplay/`、`spade/`，不调用它们的代码。

**交付状态：准备代码和启动配置，不是 SAO 论文结果的复现。** 当前入口不实现 SAO；单 rollout PPO 不等于 SAO。没有在本地 Windows 上运行 CUDA、Docker sandbox 或真实 SWE 训练。CPU 检查结果见 [VALIDATION.md](VALIDATION.md)。

交给另一位 AI 去集群申请八卡 A 系列并运行 GRPO / GRPO＋DIS / Score Centering，请优先阅读 [集群执行交接文档](CLUSTER_HANDOFF.md)，其中列出了默认步数/超参数、SAO 差异、站点待确认项、三臂提交顺序，以及 Score Centering 的数值/真实概率协议验证和验收证据。

## 一次准备，之后只改配置和启动参数

服务器前提：Linux、已分配的 NVIDIA GPU、Docker GPU runtime、Docker Compose、Git、Python 3.12、uv。Harbor 在宿主机独立虚拟环境运行，训练在 Miles 官方容器中运行。无需把 Docker socket 挂进训练容器。模型权重由你提供，不自动下载。

服务器执行代理先读仓库根 [AGENTS.md](../AGENTS.md)。下面的命令是供你审核后在已授权的服务器执行，**不构成让代理自动安装依赖或启动训练的授权**。代码准备已经获得本次用户授权；实际服务器 `ztwang` 根目录、GPU 分配和 Docker 权限仍由操作者确认。

```bash
# 在你确认的 ztwang 根目录内；不要用历史路径代替确认。
git clone https://github.com/ZitongWang018/handover-ustgz.git
cd handover-ustgz/agentic-coding-rl
cp configs/local.example.json configs/local.json
```

编辑 `configs/local.json`，主要填这些：

| 字段 | 填什么 |
|---|---|
| `workspace_root` | 本次确认的、真实存在的 `ztwang` 绝对路径 |
| `paths.hf_checkpoint` | HF checkpoint 目录，保留 tokenizer、chat template、config 与权重 |
| `paths.torch_dist_checkpoint` | 转换后 Megatron checkpoint 的新目录；已有同版本的可直接填 |
| `runtime.session_external_host` | **task Docker 容器可以访问到的训练宿主机 IP**，通常不能填容器内的 `127.0.0.1` |
| `runtime.gpus` / `num_gpus` | 获分配的 GPU ID 和数量，例 `0,1,2,3,4,5,6,7` / `8` |
| `runtime.async_actor_gpus` | 异步模式的 actor 卡数，默认 4；其余 4 张用于 rollout；critic 与 actor 共卡 |
| `training.*` | 学习率、组大小、每次更新轨迹数、长度、步数、clip、KL、seed 等 |

`paths.*` 可填绝对路径或相对于 `workspace_root` 的路径，必须留在该目录内。默认假设 checkout 叫 `handover-ustgz`，改了目录名也要修改数据/输出路径。缓存默认位于根目录下 `.cache/swe-rl`，保持短路径，避免 Ray UNIX socket 路径超长。

默认模型为 **Qwen3-4B-Instruct-2507**；默认上下文 16K、单次生成 2K，只是小规模起步设置。它不是 SAO 原文的 30B/128K/OpenHands 协议，也未在 8 卡上实测吞吐或显存。不同型号/显存的 8 卡不能直接等价。模型变体必须与 `model.megatron_type`、TITO/parser 一起对齐，`doctor` 会检查主要架构及 RoPE；任意 checkpoint 不能只换路径。

### 1. 固定上游代码和 Harbor 环境

```bash
bash scripts/run.sh bootstrap
bash scripts/setup_harbor.sh configs/local.json
```

两个 Git commit 固定在 [UPSTREAM.lock.json](UPSTREAM.lock.json)。`.deps/` 本地实际 clone，体积和 Git 历史不重复推入本仓库；服务器执行 bootstrap 会重建相同版本。不需要初始化无关项目的 submodule。

`setup_harbor.sh` 使用 Harbor 自带 `uv.lock` 和独立 `.venv`，不安装到全局 Python。训练依赖使用官方 Miles 镜像；`runtime.image` 暂为 `radixark/miles:latest`，**镜像尚未锁定到远端 digest，也未验证其与固定代码的运行兼容性**。首次成功 smoke 后应把它换成 `radixark/miles@sha256:...` 并固定后续所有对比；每次训练会记录实际容器 image ID。不要在容器里 `git pull`。

### 2. 一键准备 SWE 任务和 JSONL

```bash
# 宿主机执行，使用刚创建的 Harbor 环境。
.deps/harbor/.venv/bin/python run.py prepare-data --kind train
.deps/harbor/.venv/bin/python run.py prepare-data --kind eval
```

默认训练集为 SWE-Gym Lite，评测为 SWE-bench Verified；转换调用固定版本的 Harbor adapter，下载数据并生成任务定义。首次 trial 仍需要下载/构建各仓库 Docker 镜像，注意磁盘和网络。训练不会使用 Verified test 作训练集。

想先检查少量任务可加 `--limit 8`。这是小样本诊断，不能报告为完整 Verified 成绩。数据和 task 文件有 SHA256 manifest；adapter 自身使用数据集默认 revision，因此这是内容留档，不是预先固定的 HF revision。准备失败保留现场，不覆盖数据；若需要改变子集、agent 或上下文，请使用新的 `paths.train_jsonl` / `eval_jsonl` / `tasks_dir`。

Gold patch 只在 Harbor verifier/oracle 的任务文件中，不放入 Miles 的 `prompt` 和 rollout metadata。任务定义保留 `solution/` 供 oracle 使用；标准 Harbor 非 oracle agent 不将其作为输入。

Harbor adapter 会固定任务内容，但其 sandbox 中安装的 mini-swe-agent/litellm 版本当前仍由上游默认解析。正式对比前还应固定实际 agent 安装版本和镜像 digest；目前不声称所有第三方运行依赖均已冻结。

### 3. 转换 checkpoint（已有可跳过）

```bash
# 宿主机包装器自动进入官方训练容器；只转换，不训练。
bash scripts/in_container.sh convert
```

转换写入 `paths.torch_dist_checkpoint`；目标已存在就停止，不覆盖。训练仍然需要 HF 目录，不能删。

### 4. 启动 Harbor 服务（终端 A）

```bash
.deps/harbor/.venv/bin/python run.py serve
```

默认服务在宿主机 `21100` 端口；需要 task 容器访问训练宿主机 `23000..23007` session 端口及 `23100` router 端口。异步默认 16 条并发轨迹，CPU/内存/磁盘也要能承载 16 个 sandbox，可同时下调 `total_trajectories_per_update`、`group_size` 和 `max_concurrent`。

Harbor 上游会检查操作者自己的 Docker Hub 登录状态，首次使用可能需要你自行 `docker login`。本脚本不收集或提交凭据。服务输出在独立 `runs/harbor-*` 目录；自动全局 Docker prune 已通过 `DOCKER_PRUNE_INTERVAL_SEC=inf` 禁用，已有容器不会被批量清理。遗留容器/镜像只由操作者按明确清单处理。

### 5. 检查并做两步 GPU smoke（终端 B）

```bash
bash scripts/in_container.sh doctor --gpu
bash scripts/in_container.sh train --algo grpo --mode async --run-id grpo-smoke-s42 --smoke
```

`--smoke` 保留配置的任务长度和 batch，只把训练缩为两次更新；PPO 额外保留 critic warmup。它验证链路，不证明收敛。检查 `train.log`、`traces/`、checkpoint 和 reward 是否有效，确认是实际 SWE 多轮交互。不要把日志中不断重采样失败任务的情况当成正常训练。

无需 GPU 的命令预览：

```bash
python3 run.py train --algo grpo --mode async --run-id preview --dry-run
```

### 6. 正式训练和对比

```bash
bash scripts/in_container.sh train --algo grpo --mode async --run-id grpo-s42 --eval
bash scripts/in_container.sh train --algo grpo_dis --mode async --run-id grpo-dis-s42 --eval
bash scripts/in_container.sh train --algo gspo --mode async --run-id gspo-s42 --eval
bash scripts/in_container.sh train --algo grpo_tis --mode async --run-id grpo-tis-s42 --eval
bash scripts/in_container.sh train --algo grpo_mis --mode async --run-id grpo-mis-s42 --eval
bash scripts/in_container.sh train --algo score_centering --mode async --run-id sc-s42 --eval
bash scripts/in_container.sh train --algo ppo_single --mode async --run-id ppo-single-s42 --eval
```

**以上是顺序选择运行的命令，不是在同一组 GPU 上同时启动。** 每条结束后容器退出；下一条使用新的独立容器和 run ID。同步模式改为 `--mode sync`。`--eval` 运行训练前基线及定期评测；不传则只训练。默认评测是在 Harbor/mini-swe-agent 协议下，不能声称复现原文 OpenHands 的绝对分数。

`SWE_RL_CONFIG=/absolute/path/to/ztwang/.../my.local.json bash scripts/in_container.sh ...` 可以切换配置。直接 Python 启动时用 `--config`。不要在容器 wrapper 里再传第二个 `--config`。

模型、训练超参和路径改配置即可；算法、同步/异步、run ID、是否 smoke、是否评测由 CLI 控制。配置中的轨迹预算相同：默认 `2 prompts × 8 = 16`，PPO 自动变为 `16 prompts × 1 = 16`；每批对应一个 actor optimizer step。PPO warmup 是额外 critic-only rollout，比较总成本时必须计入。

## 结果与算法语义

每次运行生成 `runs/<run-id>/manifest.json`、`train.log`、`traces/`、`checkpoints/`，PPO 还有 `checkpoints_critic/`。manifest 记录上游 SHA、配置、实际命令、源文件哈希、数据哈希和实际容器 ID。重复 run ID 会停止；当前入口**不提供自动续训**，避免不完整恢复 critic/rollout/RNG 后误称严格续训。保留失败日志和 checkpoint，由后续专门的恢复功能处理。

| `--algo` | 实际算法定义 |
|---|---|
| `grpo` | 上游 GRPO advantage，ratio 分母用真实 rollout logprob（bypass 版本），不是 SAO DIS |
| `grpo_dis` | 同一 GRPO advantage＋独立 DIS loss；直接 learner/rollout ratio，严格区间外 PG 归零，默认 `(0.2, 4.0)` |
| `gspo` | 上游 GSPO 序列比率；clip 单独用 `gspo_clip_*` |
| `grpo_tis` | 上游三策略 decoupled PPO/GRPO ＋ token TIS；重算 proximal logprob，与 bypass 基线有明确区别 |
| `grpo_mis` | 同上，importance weights 超出配置区间时 mask；不是 SAO DIS |
| `score_centering` | 上游 `2609.20807` 的 top-k 实现，默认 K=128、MIS；去组内标准差，保留减均值 |
| `ppo_single` | G=1、critic、跳过 observation 的固定 λ GAE、PPO loss；不是 SAO |

统一训练采样默认 temperature=1、top_p=1、top_k=-1。Score Centering 额外存储候选概率，需匹配固定 Miles 的 SGLang/router 协议；CPU 测试无法证明容器中的实际响应满足协议，首次 smoke 应检查候选字段。论文实现的数值测试不是其 SWE 学习效果的复现。

**GRPO＋DIS 直接入口：**在完成上述数据、checkpoint 和 Harbor 服务准备后：

```bash
# 先做两步 smoke；正式运行用新 run ID，并改 --smoke 为 --eval。
bash scripts/run_grpo_dis.sh --run-id grpo-dis-smoke --smoke
bash scripts/run_grpo_dis.sh --run-id grpo-dis-s42 --eval
```

`--mode sync` 可切换同步模式，默认仍为异步。旧 `local.json` 不必重建；可添加 `training.dis_ratio_low` / `dis_ratio_high` 覆盖默认值。它们与普通 PPO 的 `clip_low` / `clip_high` 独立，后者不会用于 DIS loss。算法、梯度、KL/entropy 的边界及完整检查说明见 [GRPO_DIS.md](GRPO_DIS.md)。

比较固定 trajectory/token 预算与固定墙钟时间两条曲线，同时记录解决率、生成 token 数、工具轮数、policy lag、ratio/mask 比例及 critic EV。只在相同模型、数据、agent、上下文、采样和评测协议下比较；不同算法的 clip、normalization、额外 critic/候选概率成本要写明。训练 reward 上升不等于 Verified 提升。

## 后续实现 SAO 从哪里改

见 [ALGORITHM_MAP.md](ALGORITHM_MAP.md)。不需要更换 SWE 环境或训练框架。当前 DIS 使用 Miles 的 custom-loss 插件，代码位于本目录 `algorithms/`，不改上游 tracked 文件；manifest 包含插件源文件哈希。如果未来需要修改上游本身，必须明确更新版本/补丁和来源检查，不能静默绕过记录。

主要来源：[Miles Harbor recipe](https://github.com/radixark/miles/tree/ee3f3d4650bc7d901c736f81297b3ec524159fb0/examples/swe-agent-harbor-docker)、[PPO](https://github.com/radixark/miles/tree/ee3f3d4650bc7d901c736f81297b3ec524159fb0/examples/ppo)、[Score Centering](https://github.com/radixark/miles/tree/ee3f3d4650bc7d901c736f81297b3ec524159fb0/examples/infra_features/score_centering)、[SAO 原文](https://arxiv.org/abs/2607.07508)。
