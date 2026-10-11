# HKUST-GZ HPC 集群使用规范与规则

> 本文档是 `xyu608` 账号在 HKUST-GZ（香港科技大学·广州）HPC 平台上的**唯一权威操作规范**。
> 覆盖 **二期**（A800/A40，登录节点 `hpc2login`）与 **三期 ACD**（H100，登录节点 `hpc3login`）。
> 后续任何协作者（包括 AI 助手）在提交作业、读写文件、训练模型前，**必须先读完本文档并遵守其中的硬性规则**。
> 最后更新：2026-10-11

---

## ⛔ 第 0 条：两条不可违反的铁律

**这两条优先于本文档其它所有内容。任何情况下都不得违反。**

### 铁律一：只能在 `ztwang/` 里创建和编辑文件

- **所有新建、修改、运行的文件，一律放在 `~/ztwang/` 内**（二期 `/hpc2hdd/home/xyu608/ztwang`，三期 `/data/user/xyu608/ztwang`）。
- **外部资源只读**：`~/thinktool/models/`、他人目录下的模型、预训练 checkpoint 等，**可以读取使用，绝不写入、绝不修改、绝不移动**。
- 实验输出（checkpoint、日志、结果）也写在 `ztwang/` 下，或按第 2 节规则写到高速盘。

```bash
# ✅ 正确
python ~/ztwang/selfplay/src/train.py --model ~/thinktool/models/qwen3_4b_inst_2507
sbatch ~/ztwang/tools/sbatch/gpu_h100x8.sbatch

# ❌ 错误：在工作区外创建/修改文件
cd ~/thinktool && vim train.py           # 改外部仓库
mkdir ~/new_experiment                   # 在家目录根下新建
```

原因：家目录根下已有 10+ 个历史项目，外部仓库可能被他人或上游更新覆盖，且无法追踪归属。`ztwang/` 是唯一纳入版本管理的目录。

### 铁律二：训练必须存断点，且**只保留一个**

- 任何超过 1 小时的训练任务，`sbatch` 脚本里**必须**配置周期性 checkpoint。
- **同时只保留最新一个断点**——新的写好后删掉旧的。不允许堆积 `ckpt-1000`、`ckpt-2000`、`ckpt-3000`... 占满存储。
- 实现方式（任选其一）：

```bash
# 方式 A：训练框架自带（推荐），以 HuggingFace Trainer 为例
--save_steps 500 --save_total_limit 1 --save_only_model

# 方式 B：脚本里显式「存新的、删旧的」
CKPT_DIR=~/ztwang/selfplay/runs/$SLURM_JOB_ID/ckpt
mkdir -p "$CKPT_DIR"
# ... 训练中周期性保存到 "$CKPT_DIR/current" ...
# 覆盖式保存：每次先删旧的再写新的
rm -rf "$CKPT_DIR/prev" && mv "$CKPT_DIR/current" "$CKPT_DIR/prev" \
  && mkdir "$CKPT_DIR/current"
```

- **为什么**：三期存储 17P、二期家目录 HDD 已用 96%，一个 8B 模型的优化器状态就有几十 GB。多个断点叠加快，是账号被限流的头号原因。
- 断点写在**高速盘**（二期 `~/jhspoolers`，三期 `/data/user/xyu608/ztwang/runs`），不要写家目录 HDD。

---

## 0. 给 AI 协作者的第一段话

如果你是一个 AI 助手，被要求在这台集群上干活，请先确认四件事：

1. **先读第 0 条铁律**：只在 `ztwang/` 里写文件；训练必须存断点且只留一个。
2. **你连的是哪个节点**：登录节点（`mgmt-*` / `ACD-Login-*`）只用于编辑、提交、看日志，**绝不允许跑训练或大规模计算**。
3. **你要写的路径在哪**：代码放 `~/ztwang/`；二期大文件放 `~/jhspoolers/`（SSD），三期放 `~/ztwang/runs/`。
4. **你申请了哪个分区**：
   - 二期：8 卡整机**只能**提交到 `i64m1tga800u8ka` / `i64m1tga40u8ka`，否则被拒。
   - 三期：`acd_u`（免费共享）/ `acd_ue`（独占）/ `emergency_acd`（紧急），单节点固定 8 张 H100，只能整机申请。

违反以上任一条会导致作业被拒、账号被限流、或拖垮共享存储——这是共享集群，不是私人机器。

---

## 0.1 两个集群怎么选

| | **二期**（hpc2login） | **三期 ACD**（hpc3login） |
|---|---|---|
| GPU | NVIDIA **A800** 80GB / A40 48GB | NVIDIA **H100 80GB HBM3** |
| 登录节点 | `hpc2login.hpc.hkust-gz.edu.cn` | `hpc3login.hpc.hkust-gz.edu.cn` |
| 用户组 | `tonggehuang_team` | `formal` |
| 家目录 | `/hpc2hdd/home/xyu608` | `/data/user/xyu608` |
| **外网** | ✅ 可访问（git/curl 正常） | ❌ **完全隔离**，只能走校内镜像仓 |
| 一天前的 8 卡等待 | ~26 小时 | ~15 小时 |
| 现状 | 1600+ 作业，长期拥堵 | 相对宽松，但卡也基本占满 |
| **适合** | 需要联网拉代码/依赖的实验 | 已有环境、纯训练跑量 |

> 详细对比见 `docs/CLUSTER_SNAPSHOT.md`，三期专项规则见 `docs/HPC3_GUIDE.md`。

---


## 1. 账号与登录

| 项目 | 值 |
|---|---|
| 用户名 | `xyu608` |
| 邮箱前缀 / SSO | `xyu608` |
| 用户组 | 二期 `tonggehuang_team`；三期 `formal` |
| UID / GID | 204227 / 201375 |
| 二期登录节点 | `hpc2login.hpc.hkust-gz.edu.cn`（登录后主机名 `mgmt-*`） |
| 三期登录节点 | `hpc3login.hpc.hkust-gz.edu.cn`（登录后主机名 `ACD-Login-*`） |
| 一期登录节点 | `hpc1login.hpc.hkust-gz.edu.cn` |
| 二期 Web 门户 | <https://hpc2login.hpc.hkust-gz.edu.cn/> |
| 三期 Web 门户 | <https://hpc3login.hpc.hkust-gz.edu.cn/hpc-user/#/login> |

本机（Windows）已配置免密登录，直接在终端执行：

```bash
ssh hpc2        # 二期（日常开发、需要联网时用）
ssh hpc3        # 三期 ACD（H100，纯训练用）
ssh hpc1        # 一期
```

- SSH 别名定义在 `C:\Users\dasou\.ssh\config`，同一把私钥 `C:\Users\dasou\.ssh\hpc2gz_ed25519` 同时授权于二期与三期
- 二期登录 shell 下 `sbatch` 在 PATH 中（`/opt/slurm/bin`）；**非交互式执行需显式加 PATH**：
  ```bash
  export PATH=/opt/slurm/bin:$PATH      # 仅二期需要
  ```
- 三期 `sbatch` / `squeue` 在 `/bin`，登录 shell 与非交互 shell 都直接可用，**无需设 PATH**
- 三期 Slurm 版本 23.11.11，集群名 `chess`；二期 Slurm 23.02.3

> 注意：**两个集群都曾出现 `~/.ssh` 权限异常**（目录是 `drw-------`，缺 `x` 位导致属主都无法进入，免密登录失效）。
> 若免密登录失效，先修复：`chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys`

---


## 2. 存储布局（必须遵守的写入规则）

### 2.1 二期（HDD 家目录，配额紧张）

| 路径 | 类型 | 容量/状态 | 用途 |
|---|---|---|---|
| `/hpc2hdd/home/xyu608` | HDD 家目录 | 3.9P，**已用 96%，余 173T** | 代码、小文件、配置。**不要放大数据** |
| `/hpc2ssd/JH_DATA/spooler/xyu608` | SSD | 309T，已用 93% | **软链为 `~/jhspoolers`**，放数据集/checkpoint/临时大文件 |
| `/hpc2hdd/JH_DATA/jhai_data/xyu608` | HDD 平台盘 | — | 软链为 `~/jhaidata` |
| `/hpc2hdd/JH_DATA/share/xyu608` | HDD 共享盘 | — | 软链为 `~/.jhshares` |
| `/hpc2hdd/JH_DATA/filedownload/xyu608` | HDD | — | 软链为 `~/.filedownload` |
| `/tmp` | 本地盘 | 节点本地 | 每节点独立，作业结束即丢，适合解压中间产物 |

### 2.2 三期 ACD（分布式存储，比二期宽松得多）

| 路径 | 类型 | 容量/状态 | 用途 |
|---|---|---|---|
| `/data/user/xyu608` | 家目录 | 挂在 `/data`（19P，**已用 38%，余 12P**） | 代码、模型、输出；比二期宽裕，但仍**不要囤多个断点** |
| `/share` | 共享目录 | — | 平台软件、cuda 工具链、数据集 |
| `/tmp` | 本地盘 | 节点本地 | 作业结束即丢 |

三期家目录**没有** `jhspoolers` 那套软链，训练产物直接写 `~/ztwang/runs/<jobid>/`。

### 硬性写入规则

1. **代码与产出统一在 `~/ztwang/`**（第 0 条铁律）。
2. **断点只留一个**（第 0 条铁律二）：一个 8B 模型的优化器状态可达几十 GB，堆积断点是账号被限流的头号原因。
3. 二期家目录 HDD 已用 96%，大文件（数据集、模型权重、tar 包）一律写 `~/jhspoolers/`（SSD）。定期检查体积：
   ```bash
   du -sh ~/* 2>/dev/null | sort -h        # ⚠️ 二期家目录 300G+，会跑几分钟
   ```
4. 二期家目录里存在几个 **8GB+ 的匿名残留文件**，属于历史下载残留，属于清理候选（清理前需人工确认）：
   - `~/05815e010bd027d7be39c8e65c81c56c`（8.4G）
   - `~/1203d0fddb2c60c613e83cab431f1d76`（8.4G）
   - `~/Slime_running/1203d0fddb2c60c613e83cab431f1d76`（8.3G，疑似重复）
   - `~/docker/thinktool_v0_9_1.tar.gz`（11.9G）
   - `~/hechenyu/longlive2_docker.tar.gz`（10.8G）
   - `~/clip/MiniClip.zip`（1.2G）
5. **`~/.recycle_bin` 不自动释放空间**，删除大文件后需手动清空。

---


## 3. 目录约定

```
~/ztwang/                     <- 所有代码的唯一根目录（本仓库）
├── docs/                     <- 规范、速查文档
├── tools/                    <- 脚本工具（远程控制、巡检、sbatch 模板）
│   └── sbatch/               <- 作业模板，改两行即可提交
└── selfplay/                 <- selfplay 项目工作区
```

其它已有项目目录（历史成果，**不要随意改动或重命名**）：

| 目录 | 内容 | git 仓库 |
|---|---|---|
| `~/thinktool` | 257G，最大。`rllm` / `AFlow` / `models`（本地模型） | `rllm`, `AFlow` 等 |
| `~/hechenyu` | LongLive 视频生成 | `LongLive` |
| `~/MAS` | ViF 多智能体 | `ViF`, `ViF_ours` |
| `~/Slime_running` | slime RL 训练框架 + Megatron-LM | `slime`, `Megatron-LM` |
| `~/Theory` | IA / K-model 理论研究 | `MultiEMO` 等 |
| `~/clip` | MiniClip / llm2clip | `MiniClip` |
| `~/theory_final` | ReVision | `ReVision` |
| `~/huangzhe` | DLLM-Searcher | `DLLM-Searcher` |

三期家目录（`/data/user/xyu608`）的已有内容 —— **同样只读，不要改动**：

| 目录 | 体积 | 内容 |
|---|---|---|
| `~/output` | **406G** | 3 组 `checkpoints-sft-bunny-*` 断点（⚠️ 断点堆积的反面教材） |
| `~/model` | 61G | 模型 |
| `~/models` | 624M | 模型目录 |
| `~/revision_sft` | 5.3G | ReVision SFT |
| `~/res` | 1.1G | 权重分片 |
| `~/ReVision` | 78M | ReVision 项目（非 git 仓库） |
| `~/checkpoints-pretrain-bunny-*` | 44M | 预训练断点 |
| `~/aistudio` `~/pkgs` | — | 平台/依赖 |


---

## 4. Slurm 作业提交规范

### 4.1 集群与账号基本信息

| 项目 | 值 |
|---|---|
| Slurm 版本 | 23.02.3 |
| 命令路径 | `/opt/slurm/bin` |
| Slurm account | `root` |
| 默认分区 | `a128m512u` |
| 默认 QOS | 无（走 fairshare） |

### 4.2 分区规格表

| 分区 | 节点数 | 单节点配置 | 时限 | 用途 |
|---|---|---|---|---|
| `i64m1tga800u` | 11 | 64 核 / 1TB / **A800×8** | 7 天 | 日常 GPU 训练（**单卡~多卡，非整机**） |
| `i64m1tga800ue` | 11 | 同上 | 7 天 | 同上（紧急队列副本） |
| `long_gpu` | 11 | 同上 | 14 天 | GPU 长任务 |
| `i64m1tga800u8ka` | 3 | 同上 | 7 天 | ⚠️ **8 卡整机专用队列** |
| `i64m1tga40u` | 3 | 64 核 / 1TB / **A40×8** | 7 天 | A40 日常 |
| `i64m1tga40u8ka` | 1 | 同上 | 7 天 | ⚠️ **8 卡整机专用队列** |
| `emergency_gpu` | 10 | A800×8 | 14 天 | 紧急 GPU |
| `i64m512u` | 107 | 64 核 / 512GB | 7 天 | CPU 大部队 |
| `a128m512u` | 16 | 128 核 / 512GB | 7 天 | CPU 高核数 |
| `i96m3tu` | 4 | 96 核 / **3TB** | 7 天 | 大内存 |
| `long_cpu` | 106 | 64 核 | 14 天 | CPU 长任务 |
| `debug` | 6 | 混合 | **30 分钟** | 调试、自检 |

### 4.3 三期 ACD 分区规格表

三期是**独立集群**（ClusterName `chess`），登录节点 `hpc3login`，与二期不共享文件系统、不共享队列。

| 分区 | 节点数 | 单节点配置 | 时限 | 优先级 | 用途 |
|---|---|---|---|---|---|
| `acd_u` ⭐默认 | 40 | 96 核 / **2TB** / **H100×8** | **7 天** | 100 | 免费共享队列，日常用这个 |
| `acd_ue` | 40 | 同上 | 7 天 | 200 | 独占队列（价格中档） |
| `emergency_acd` | 40 | 同上 | 7 天 | 300 | 紧急队列（价格高档） |
| `debug` | 1（`ACD1-63`） | 同上 | **30 分钟** | 1 | 调试、自检 |

关键参数：

| 项目 | 值 |
|---|---|
| Slurm 版本 | 23.11.11，命令在 `/bin`（无需设 PATH） |
| Slurm account | `d_tonggehuang_team`（显示为 `d_tonggeh+`） |
| 默认 QOS | `normal` |
| 单用户配额 | **CPU 128 核 / GPU 16 张 / 单作业 ≤7 天**（官方口径） |
| QOS 硬限 | `MaxTRESPerUser cpu=192, gres/gpu=16`，`MaxJobsPerUser=20` |
| 可提交用户组 | `formal`, `admin` |
| 节点规格 | 2×48C Xeon（96 核）/ 2TB RAM / 8×H100 80GB HBM3 |
| 集群总算力 | 320 GPU（`acd_u` 内 40 节点）/ 3840 CPU / 78TB 内存 |
| 网络 | 400Gb/s RoCE v2，节点内 GPU 互联 800GB/s |

⚠️ **三期与二期的三个关键差异**：

1. **三期完全无法访问外网**（`github.com`、`pypi.org` 均不可达）。代码/依赖必须在二期或本机准备好后传进去，或使用校内镜像仓。**因此三期上的 `~/ztwang` 不能用 `git pull`，只能从本机 SFTP 推送。**
2. **`PrivateData=jobs`**：`squeue` 看不到别人的作业（只会显示 0），**判断拥堵情况要看节点状态而不是队列长度**：
   ```bash
   sinfo -h -N -o "%T" | sort | uniq -c     # 节点状态分布
   sbatch --test-only ... --wrap true       # 直接问起算时间（最准）
   ```
3. **单节点固定 8 卡，只能整机申请**——`--gres=gpu:8` 是常态，不存在"只申请 1 张"的常见用法（技术上可以，但碎片化排队更长）。

### 4.4 ⚠️ 硬性规则（违反会被 Slurm 直接拒绝）

1. **二期 8 卡（整机）作业只能提交到 `i64m1tga800u8ka`（A800）或 `i64m1tga40u8ka`（A40）。**
   提交到 `i64m1tga800u` 等普通分区会报错：
   ```
   ERROR: 8-GPU jobs are ONLY allowed on partitions 'i64m1tga800u8ka' and 'i64m1tga40u8ka'.
   ```
   **三期没有这个限制**，`acd_u` 直接申请 `--gres=gpu:8` 即可。
2. **节点是独占申请的**：二期一台 64 核 / 1TB / 8 卡；三期一台 96 核 / 2TB / 8 卡。
   `--mem` 不要超过节点物理内存（二期 A800 节点 1TB、CPU 节点 512GB；三期 2TB）。
3. **登录节点不跑计算**。所有训练/推理/大规模数据处理必须走 `sbatch` 或 `srun`。
4. **`--time` 必须写**，且不要动辄申请 7 天；短任务写短时限有利于 backfill 插队，能显著缩短等待。
5. **不要用 `--exclusive`**（节点本身已独占）。
6. **训练脚本必须带 checkpoint 且只保留一个**（见第 0 条铁律二）。
7. 日志统一输出到 `~/ztwang/logs/` 或项目自己的 `logs/`，文件名带 `%x-%j` 便于排查。

### 4.5 提交三件套

```bash
# 二期需要，三期不需要（三期 slurm 在 /bin）
export PATH=/opt/slurm/bin:$PATH

sbatch my_job.sbatch           # 提交
squeue -u xyu608               # 看自己的队列
scancel <JOBID>                # 取消
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode   # 历史作业
```

### 4.6 提交前必做的自检

```bash
bash ~/ztwang/tools/check_cluster.sh     # 一键看负载、配额、队列、预计起算时间
```

或手动预测起算时间（**不会真的提交作业**）：

```bash
sbatch --test-only --partition=i64m1tga800u8ka --gres=gpu:a800:8 \
       --ntasks=1 --cpus-per-task=32 --mem=200G --time=1-00:00:00 --wrap "true"
```

---

## 5. 环境与依赖

### 5.0 三期 ACD 环境（与二期完全独立）

| 项目 | 值 |
|---|---|
| conda | `/share/anaconda3`（登录即激活，`CONDA_EXE=/share/anaconda3/bin/conda`） |
| 已有环境 | `ReVision` `revision` `pytorch` `pytorch251` `pytorch271` `pytorch2.14.0` `vllm` `vllm017` `flash_attn3` `jax_gpu` `video_dit` `open3d` |
| module | `module avail` 可用；`/data/modulefiles/opensource`、`/data/modulefiles/soft`（`apptainer` `singularity` `ffmpeg` `julia` `uv` `open3d` 等） |
| CUDA | `/share/cuda132`、`/share/cuda133`、`/share/bak_cuda128` |
| 数据集 | `/share/dataset` |
| **外网** | ❌ **完全隔离**，`pip` / `conda` / `git` 直连均不可用 |

三期家目录已有内容（**只读使用，不要修改**）：`ReVision/`、`models/`、`model/`、`output/`、`res/`、`revision_sft/`、`pkgs/`、`checkpoints-pretrain-bunny-*/`、`aistudio/`

⚠️ 在三期装依赖：因无外网，需先在二期或本机 `pip download`，再 SFTP 传进去离线安装。

### 5.1 二期 conda

`~/miniconda3`（conda 26.1.1），已有环境：

| 环境 | 用途 |
|---|---|
| `ReVision` | ReVision 项目 |
| `thinktool` | thinktool / rllm |
| `aflow` | AFlow |
| `longlive` / `longlive2` | LongLive 视频生成 |
| `SyBC` | — |
| `agentenv-alfworld` | Agent 环境 |
| `agentenv-sciworld` | Agent 环境 |
| `agentenv-searchqa` | Agent 环境 |
| `agentenv-webshop` | Agent 环境 |

作业脚本里激活方式：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate <env>
```

### 5.2 module（部分常用）

`cuda/11.0 ~ 13.0`、`compilers/gcc-11.1.0 ~ 13.1.0`、`anaconda3`、`apptainer-1.4.5`、
`singularity-ce-4.1.3`、`mpi/openmpi-4.1.5`、`hdf5`、`ffmpeg`、`git-lfs`、`uv`、`julia/1.10.9`

```bash
module avail          # 查看全部
module load cuda/12.8
```

---

## 6. 模型资产

### 6.1 本地已有模型 `~/thinktool/models/`

| 目录 | 实际模型 | 体积 | 说明 |
|---|---|---|---|
| `qwen3_4b_inst_2507` | **Qwen3-4B-Instruct-2507** | 7.6G | ✅ 指令版，3 分片，权重完整，context 262144 |
| `qwen3_4b` | Qwen3-4B（基座） | 7.6G | context 40960 |
| `qwen3_8b` | **Qwen3-8B** | 16G | ✅ 5 分片，权重完整，hidden 4096 |
| `qwen3-14b` | Qwen3-14B | 28G | 完整 |
| `qwen3-30b-a3b-inst-2507` | Qwen3-30B-A3B-Instruct-2507 | 57G | MoE |
| `30b-a3b-inst-2507` | Qwen3-30B-A3B-Instruct-2507 | 56G | MoE（重复副本） |

> **关于 "Qwen3-7B"：Qwen3 系列没有 7B 规格。**
> Qwen3 的规格是 0.6B / 1.7B / 4B / 8B / 14B / 32B（稠密）+ 30B-A3B / 235B-A22B（MoE）。
> 如果需要 7B 量级，对应的是 **Qwen3-8B**（本地已有）；
> 如果确实要 7B，那是上一代 **Qwen2.5-7B-Instruct**，可读路径：
> - `/hpc2hdd/JH_DATA/jhai_data/mpeng053/pretrain_model/Qwen2.5-7B-Instruct`
> - `/hpc2hdd/JH_DATA/spooler/syang724/models/Qwen_Qwen2.5-7B-Instruct`

### 6.2 HF 缓存 `~/.cache/huggingface/hub/`

`Qwen3-4B`、`Qwen3-4B-Instruct-2507`、`Qwen3-8B`、`Qwen3-14B`、`Qwen3-30B-A3B-Instruct-2507`

### 6.3 使用他人目录下的模型

只读使用没问题（`mpeng053`、`leeyuxiaoli`、`syang724` 等目录可读），但：
**不要写入他人目录，也不要依赖其长期存在。**需要长期使用请复制到自己目录或 SSD。

---

## 7. AI 协作约定

后续可能由 GPT 等外部模型协助开发，协作时必须遵守：

0. **第 0 条两条铁律优先级最高**：
   - 只能在 `ztwang/` 里新建/修改文件，外部 ckpt、模型、他人目录**一律只读**；
   - 任何训练都必须写 checkpoint，且**同时只保留一个**。
1. **改动前先读本文档和 `docs/ENVIRONMENT.md`**，不要把集群当成普通 Linux 机器。
2. **不要在登录节点执行训练/推理**，只提交作业。
3. **不要批量删除或移动文件**。删除家目录下任何 >1GB 的文件前必须经用户确认。
4. **不要写 `~/.ssh`、`~/.bashrc`、`~/miniconda3` 等环境目录**，除非用户明确要求。
5. **不要把训练产物写到 `ztwang/` 之外**；二期大文件用 `~/jhspoolers`（SSD），三期写 `~/ztwang/runs/<jobid>/`。
6. **提交作业前用 `--test-only` 预测起算时间**，避免盲目排队。
7. **所有产出（脚本、配置）写进 `~/ztwang/`**，并同步到 GitHub 私有仓库。
8. **`.gitignore` 必须生效**：模型权重、数据集、checkpoint、`*.tar.gz`、日志不得入库。
9. **换行符必须是 LF**。在 Windows 上编辑 `.sh` / `.sbatch` / `.py` 后，务必确认没有变成 CRLF，
   否则集群上会报 `bad interpreter: /bin/bash^M`。仓库已配置 `.gitattributes` 强制 `eol=lf`，
   但**直接往集群传文件时（不走 git）不会自动转换**——用 `dos2unix` 或 `sed -i 's/\r$//'` 处理。
10. **不要用 `du -sh ~/*` 之类全盘扫描命令**，二期家目录 300G+ 在 HDD 上会跑几分钟并可能拖慢共享存储。
    `tools/check_cluster.sh` 默认已跳过体积统计，需要时加 `--du` 参数。
11. **三期无外网**：不要在三期执行 `git clone/pull`、`pip install`、`conda install`，会挂住。
    同步代码用 `python hpc_tools/sync_ztwang.py --host hpc3`。


---

## 8. 常见故障速查

| 现象 | 原因 | 处理 |
|---|---|---|
| `8-GPU jobs are ONLY allowed on ...` | 8 卡作业提交到普通分区 | 改用 `i64m1tga800u8ka` / `i64m1tga40u8ka` |
| `sbatch: command not found` | 非交互 shell PATH 缺失 | `export PATH=/opt/slurm/bin:$PATH` |
| 免密登录失效 | `~/.ssh` 权限被改 | `chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys` |
| 作业长期 `Priority` 挂起 | 队列拥堵 / 时限申请过长 | 缩短 `--time`，或换 `a128m512u`、`debug` |
| `Nodes required ... DOWN, DRAINED` | 目标节点维护 | 换分区或去掉 `-w` 指定节点 |
| 家目录写入报 quota | HDD 96% 满 | 清理大文件，改写到 `~/jhspoolers` |
| 作业被 OOM Kill | `--mem` 不足 | 提高 `--mem`（上限 1TB / 512GB） |

---

## 9. 快速参考卡片

```bash
# ---- 连接 ----
ssh hpc2                       # 二期（A800/A40，可上外网）
ssh hpc3                       # 三期（H100，无外网）

# ---- Slurm 环境 ----
export PATH=/opt/slurm/bin:$PATH     # 仅二期需要；三期在 /bin

# ---- 看负载 ----
bash ~/ztwang/tools/check_cluster.sh        # 一键巡检（自动识别集群）
sinfo -s                                    # 分区汇总
squeue -u xyu608                            # 我的作业
squeue -h -o "%T" | sort | uniq -c          # 二期：全集群作业状态
sinfo -h -N -o "%T" | sort | uniq -c        # 三期：只能看节点状态

# ---- 提交 ----
sbatch ~/ztwang/tools/sbatch/gpu_h100x8.sbatch    # 三期 H100 × 8
sbatch ~/ztwang/tools/sbatch/gpu_a800x8.sbatch    # 二期 A800 × 8

# ---- 预测起算时间（不提交）----
sbatch --test-only -p acd_u --gres=gpu:8 --time=1:00:00 --wrap true
sbatch --test-only -p i64m1tga800u8ka --gres=gpu:a800:8 --time=1:00:00 --wrap true

# ---- 从本机同步代码到集群 ----
python hpc_tools/sync_ztwang.py                 # -> 二期
python hpc_tools/sync_ztwang.py --host hpc3     # -> 三期

# ---- 看日志 ----
tail -f ~/ztwang/logs/*.out

# ---- 磁盘 ----
df -h
du -sh ~/* | sort -h        # ⚠️ 二期 HDD 上很慢
```
