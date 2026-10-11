# HKUST-GZ HPC 集群使用规范与规则

> 本文档是 `xyu608` 账号在 HKUST-GZ（香港科技大学·广州）HPC 二期平台上的**唯一权威操作规范**。
> 后续任何协作者（包括 AI 助手）在提交作业、读写文件、训练模型前，**必须先读完本文档并遵守其中的硬性规则**。
> 最后更新：2026-10-11

---

## 0. 给 AI 协作者的第一段话

如果你是一个 AI 助手，被要求在这台集群上干活，请先确认三件事：

1. **你连的是哪个节点**：登录节点 `mgmt-*` 只用于编辑、提交、看日志，**绝不允许在登录节点跑训练或大规模计算**。
2. **你要写的路径在哪**：代码放 `~/ztwang/`，大文件放 `~/jhspoolers/`（SSD），**不要往家目录 HDD 里塞 checkpoint 或数据集**。
3. **你申请了哪个分区**：8 卡整机作业**只能**提交到 `i64m1tga800u8ka` / `i64m1tga40u8ka`，提交到普通 GPU 分区会被 Slurm 直接拒绝。

违反以上任一条会导致作业被拒、账号被限流、或拖垮共享存储——这是共享集群，不是私人机器。

---

## 1. 账号与登录

| 项目 | 值 |
|---|---|
| 用户名 | `xyu608` |
| 邮箱前缀 / SSO | `xyu608` |
| 用户组 | `tonggehuang_team` |
| UID / GID | 204227 / 201375 |
| 二期登录节点 | `hpc2login.hpc.hkust-gz.edu.cn` |
| 一期登录节点 | `hpc1login.hpc.hkust-gz.edu.cn` |
| Web 门户 | <https://hpc2login.hpc.hkust-gz.edu.cn/> |

本机（Windows）已配置免密登录，直接在终端执行：

```bash
ssh hpc2        # 二期（日常用这个）
ssh hpc1        # 一期
```

- SSH 别名定义在 `C:\Users\dasou\.ssh\config`，私钥 `C:\Users\dasou\.ssh\hpc2gz_ed25519`
- 登录 shell 为 `/bin/bash`，`sbatch` / `squeue` 等命令在登录 shell 下自动可用（`/opt/slurm/bin`）
- **非交互式执行**（脚本、AI 自动化）时 PATH 不含 Slurm，必须显式加：

```bash
export PATH=/opt/slurm/bin:$PATH
```

> 注意：集群曾出现 `~/.ssh` 权限异常（目录缺 `x` 位导致属主都无法进入）。
> 若免密登录突然失效，先检查并修复：`chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys`

---

## 2. 存储布局（必须遵守的写入规则）

| 路径 | 类型 | 容量/状态 | 用途 |
|---|---|---|---|
| `/hpc2hdd/home/xyu608` | HDD 家目录 | 3.9P，**已用 96%，余 173T** | 代码、小文件、配置。**不要放大数据** |
| `/hpc2ssd/JH_DATA/spooler/xyu608` | SSD | 309T，已用 93% | **软链为 `~/jhspoolers`**，放数据集/checkpoint/临时大文件 |
| `/hpc2hdd/JH_DATA/jhai_data/xyu608` | HDD 平台盘 | — | 软链为 `~/jhaidata` |
| `/hpc2hdd/JH_DATA/share/xyu608` | HDD 共享盘 | — | 软链为 `~/.jhshares` |
| `/hpc2hdd/JH_DATA/filedownload/xyu608` | HDD | — | 软链为 `~/.filedownload` |
| `/tmp` | 本地盘 | 节点本地 | 每节点独立，作业结束即丢，适合解压中间产物 |

### 硬性写入规则

1. **大文件（数据集、checkpoint、模型权重、tar 包）一律写 `~/jhspoolers/`（SSD）**，不要写家目录。
2. HDD 整体已用 96%，家目录配额紧张。定期检查体积：
   ```bash
   du -sh ~/* 2>/dev/null | sort -h
   ```
3. 家目录里目前存在几个 **8GB+ 的匿名残留文件**，属于历史下载残留，属于清理候选（清理前需人工确认）：
   - `~/05815e010bd027d7be39c8e65c81c56c`（8.4G）
   - `~/1203d0fddb2c60c613e83cab431f1d76`（8.4G）
   - `~/Slime_running/1203d0fddb2c60c613e83cab431f1d76`（8.3G，疑似重复）
   - `~/docker/thinktool_v0_9_1.tar.gz`（11.9G）
   - `~/hechenyu/longlive2_docker.tar.gz`（10.8G）
   - `~/clip/MiniClip.zip`（1.2G）
4. **`~/.recycle_bin` 不自动释放空间**，删除大文件后需手动清空。

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

### 4.3 ⚠️ 硬性规则（违反会被 Slurm 直接拒绝）

1. **8 卡（整机）作业只能提交到 `i64m1tga800u8ka`（A800）或 `i64m1tga40u8ka`（A40）。**
   提交到 `i64m1tga800u` 等普通分区会报错：
   ```
   ERROR: 8-GPU jobs are ONLY allowed on partitions 'i64m1tga800u8ka' and 'i64m1tga40u8ka'.
   ```
2. **节点是独占申请的**：一台机器 64 核 / 1TB 内存 / 8 卡。`--mem` 不要超过 1TB（A800 节点）或 512GB（CPU 节点）。
3. **登录节点不跑计算**。所有训练/推理/大规模数据处理必须走 `sbatch` 或 `srun`。
4. **`--time` 必须写**，且不要动辄申请 7 天；短任务写短时限有利于 backfill 插队，能显著缩短等待。
5. **不要用 `--exclusive`**（节点本身已独占）。
6. 日志统一输出到 `~/ztwang/logs/` 或项目自己的 `logs/`，文件名带 `%x-%j` 便于排查。

### 4.4 提交三件套

```bash
export PATH=/opt/slurm/bin:$PATH

sbatch my_job.sbatch           # 提交
squeue -u xyu608               # 看自己的队列
scancel <JOBID>                # 取消
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode   # 历史作业
```

### 4.5 提交前必做的自检

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

### 5.1 conda

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

1. **改动前先读本文档和 `docs/ENVIRONMENT.md`**，不要把集群当成普通 Linux 机器。
2. **不要在登录节点执行训练/推理**，只提交作业。
3. **不要批量删除或移动文件**。删除家目录下任何 >1GB 的文件前必须经用户确认。
4. **不要写 `~/.ssh`、`~/.bashrc`、`~/miniconda3` 等环境目录**，除非用户明确要求。
5. **不要往家目录 HDD 写大文件**，用 `~/jhspoolers`。
6. **提交作业前用 `--test-only` 预测起算时间**，避免盲目排队。
7. **所有产出（脚本、配置、日志路径）写进 `~/ztwang/`**，并同步到 GitHub 私有仓库。
8. **`.gitignore` 必须生效**：模型权重、数据集、checkpoint、`*.tar.gz`、日志不得入库。
9. **换行符必须是 LF**。在 Windows 上编辑 `.sh` / `.sbatch` / `.py` 后，务必确认没有变成 CRLF，
   否则集群上会报 `bad interpreter: /bin/bash^M`。仓库已配置 `.gitattributes` 强制 `eol=lf`，
   但**直接往集群传文件时（不走 git）不会自动转换**——用 `dos2unix` 或 `sed -i 's/\r$//'` 处理。
10. **不要用 `du -sh ~/*` 之类全盘扫描命令**，家目录 300G+ 在 HDD 上会跑几分钟并可能拖慢共享存储。
    `tools/check_cluster.sh` 默认已跳过体积统计，需要时加 `--du` 参数。

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
ssh hpc2

# ---- Slurm 环境 ----
export PATH=/opt/slurm/bin:$PATH

# ---- 看负载 ----
sinfo -s                                   # 分区汇总
sinfo -o "%20P %6D %14F %22G"              # 分区+GPU
squeue -u xyu608                           # 我的作业
squeue -h -o "%T" | sort | uniq -c         # 全集群作业状态

# ---- 提交 ----
sbatch ~/ztwang/tools/sbatch/gpu_a800x8.sbatch

# ---- 预测起算时间（不提交）----
sbatch --test-only -p i64m1tga800u8ka --gres=gpu:a800:8 --time=1:00:00 --wrap true

# ---- 看日志 ----
tail -f ~/ztwang/logs/*.out

# ---- 磁盘 ----
du -sh ~/* | sort -h
```
