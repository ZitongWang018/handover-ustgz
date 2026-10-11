# HPC 三期（ACD）使用指南

> 三期是**独立集群**，与二期不共享文件系统、不共享队列、不共享登录节点。
> 官方文档：<https://docs.hpc.hkust-gz.edu.cn/docs/hpc3/>
> 最后更新：2026-10-11

---

## 1. 一句话总览

**三期 = 40 台 × 8 张 H100 80GB HBM3，单机 96 核 / 2TB 内存，7 天时限，但完全不能上外网。**

适合"环境已备好、纯训练跑量"；不适合"需要 git clone / pip install 的开发阶段"。

---

## 2. 与二期的完整对比

### 2.1 硬件与卡数

| 项目 | 二期 | 三期 ACD |
|---|---|---|
| GPU 型号 | NVIDIA **A800** 80GB SXM4 / A40 48GB | NVIDIA **H100 80GB HBM3** |
| 单机 GPU 数 | 8 | 8 |
| **Slurm 可见 GPU 总数** | **232**（29 节点 × 8） | **328**（41 节点 × 8） |
| 其中已分配（实测） | 195 | 299 |
| 其中空闲（实测） | 37 | 29 |
| 单机 CPU | 64 核（2×32 Xeon 8358P） | **96 核**（2×48 Xeon） |
| 单机内存 | 1TB（gpu1）/ 2TB（gpu2） | **2TB** |
| 机间网络 | 200Gb/s InfiniBand | **400Gb/s RoCE v2** |
| 机内 GPU 互联 | NVLink | **800GB/s 双向** |
| 存储 | 309TB SSD + 3.9PB HDD | **17PB** 分布式 |
| 峰值算力 | 185.5 Pflops@FP16 | **1078 Pflops@FP16** |

> 说明：二期硬件清单上是 60 台 A800 + 15 台 A40 节点（共 600 卡），但绝大部分被 `*_rent`（包机）分区独占，**普通用户队列里实际只能看到 232 卡**。三期硬件清单是 68 节点，Slurm 中 `acd_u` 有 40 节点，其余为包机/预留。

### 2.2 作业时限

| 队列 | 二期时限 | 队列 | 三期时限 |
|---|---|---|---|
| `i64m1tga800u` / `i64m1tga800ue` | 7 天 | `acd_u`（默认） | **7 天** |
| `i64m1tga800u8ka`（8卡专用） | 7 天 | `acd_ue` | 7 天 |
| `i64m1tga40u` / `8ka` | 7 天 | `emergency_acd` | 7 天 |
| `long_gpu` / `emergency_gpu` | **14 天** | — | 无 14 天队列 |
| `long_cpu` / `emergency_cpu` | 14 天 | — | — |
| `debug` | 30 分钟 | `debug` | 30 分钟 |

**结论：二期有 14 天长任务队列，三期最长只有 7 天。**
三期需要更长时长时，只能去科大 GO 提交 IT 工单申请延长，**且只允许延长一次、不得超过 7 天**。

### 2.3 8 卡作业排队情况（2026-10-11 21:46 实测）

用 `sbatch --test-only ... --wrap true` 问出来的预计起算时间（不会真的提交）：

| 集群 | 申请 | 分区 | 预计起算 | **距今** |
|---|---|---|---|---|
| 二期 | A800 × 8 | `i64m1tga800u8ka` | 10-13 00:14 | **~26 小时** |
| 二期 | A800 × 1 | `i64m1tga800u` | 10-16 13:43 | ~4.7 天 |
| **三期** | **H100 × 8** | **`acd_u`** | **10-12 12:58** | **~15 小时** |
| 三期 | H100 × 1 | `acd_u` | 立即 | **0** |

**读数要点：**

1. **三期的 8 卡比二期快约 10~11 小时**，因为三期机队更整齐（40 台同规格），且没有二期那种"被 14 天大作业长期占住"的问题。
2. **单卡反而更慢**——三期 29 张空闲卡全散落在各节点（每节点 1~3 张），凑不出第 8 张；二期同样如此（37 张空闲卡凑不出整机）。
   → **想跑就有：要么马上用单卡，要么等整机。**
3. 二期 8 卡队列只有 3 台机器（`gpu1-1/2/3`），**队伍短但机器少**；三期 40 台同构机，**吞吐大得多**。

### 2.4 状态与"看不见的作业"

| 项目 | 二期 | 三期 |
|---|---|---|
| Slurm 版本 | 23.02.3 | 23.11.11 |
| 集群名 | （默认） | `chess` |
| `PrivateData` | 未隐藏 | **`jobs`** ← 看不到别人的作业 |
| 当前作业数 | 运行 700 / 排队 918 | **无法用 squeue 统计** |
| 判断拥堵的方法 | `squeue -h -o "%T" \| sort \| uniq -c` | `sinfo -h -N -o "%T" \| sort \| uniq -c`（看节点状态） |

> ⚠️ 三期执行 `squeue` 会显示 0 个作业，**这不代表集群空闲**。必须看节点状态或直接 `--test-only`。
> 本次实测三期状态：9 个节点 allocated + 112 个 mixed（40 节点 × 3 个分区 + 1 debug），**没有完全空闲的节点**。

---

## 3. 登录与环境

| 项目 | 值 |
|---|---|
| 登录节点 | `hpc3login.hpc.hkust-gz.edu.cn`（登录后主机名 `ACD-Login-1`） |
| 本机别名 | `ssh hpc3` |
| Slurm 命令 | 在 `/bin`，**无需** `export PATH` |
| Slurm account | `d_tonggehuang_team` |
| 默认 QOS | `normal`（`MaxTRESPerUser cpu=192, gres/gpu=16`，`MaxJobsPerUser=20`） |
| 用户组 | `formal` |
| 家目录 | `/data/user/xyu608`（`/data` 共 19P，已用 38%） |
| conda | `/share/anaconda3`（登录即激活） |
| 已有 conda 环境 | `ReVision` `revision` `pytorch` `pytorch251` `pytorch271` `pytorch2.14.0` `vllm` `vllm017` `flash_attn3` `jax_gpu` `video_dit` `open3d` |
| 数据集 | `/share/dataset` |
| CUDA 工具链 | `/share/cuda132` `/share/cuda133` `/share/bak_cuda128` |
| module | `/data/modulefiles/opensource`、`/data/modulefiles/soft` |

### 3.1 ⚠️ 三期没有外网

实测结果：

```bash
curl https://github.com   → 连不上（7s 超时）
curl https://pypi.org     → 连不上
```

因此：

- ❌ `git clone` / `git pull` / `pip install` / `conda install` 直连全部失败
- ✅ 代码用 **本机 → SFTP 推送**：`python hpc_tools/sync_ztwang.py --host hpc3`
- ✅ 依赖先在二期或本机 `pip download`，再传进去离线 `pip install --no-index --find-links=./pkgs`
- ✅ 校内软件镜像仓：见 <https://docs.hpc.hkust-gz.edu.cn/docs/hpc3/on-campus-repository>
- 已有环境迁移参考：<https://docs.hpc.hkust-gz.edu.cn/docs/hpc3/hpc2-migrate>

**所以三期上的 `~/ztwang` 不配 git remote，它只是二期的下游镜像。**

### 3.2 首次登录踩到的坑

`~/.ssh` 初始权限为 `drw-------`（缺 `x` 位），导致连属主都进不去，密钥登录失效。
修复：`chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys`（二期是同一个毛病）。

---

## 4. 三期家目录现状（只读使用，不要改动）

| 目录 | 体积 | 内容 |
|---|---|---|
| `~/output` | **406G** | 3 组 SFT/预训练断点（`checkpoints-sft-bunny-*`） |
| `~/model` | 61G | 模型 |
| `~/models` | 624M | 模型目录 |
| `~/revision_sft` | 5.3G | ReVision SFT 数据/结果 |
| `~/res` | 1.1G | 导出权重分片 |
| `~/ReVision` | 78M | ReVision 项目（**非 git 仓库**，只有一堆 `log_*.out/err`） |
| `~/checkpoints-pretrain-bunny-*` | 44M | 预训练断点 |
| `~/aistudio` `~/pkgs` | — | 平台/依赖 |

> **`~/output` 406G 是"多个断点堆积"的典型案例**——这正是第 0 条铁律二（断点只留一个）要解决的问题。
> 这些历史目录**不要删**（可能还有用），但新实验一律写 `~/ztwang/runs/<jobid>/`，且只留一个断点。

---

## 5. 提交作业

```bash
ssh hpc3
# 不需要 export PATH

# 先看忙不忙（注意：不能靠 squeue）
sinfo -h -N -o "%T" | sort | uniq -c

# 预测起算时间（不真提交）
sbatch --test-only -p acd_u --gres=gpu:8 --mem=200G --time=1-00:00:00 --wrap true

# 提交
sbatch ~/ztwang/tools/sbatch/gpu_h100x8.sbatch
squeue -u xyu608
scancel <JOBID>
```

### 分区选择建议

| 场景 | 用哪个 |
|---|---|
| 日常训练（免费） | `acd_u` ⭐ |
| 需要独占资源 | `acd_ue`（价格中档） |
| 赶 deadline | `emergency_acd`（价格高档） |
| 调环境、验证脚本 | `debug`（**30 分钟**，1 台机 `ACD1-63`） |

### 配额提醒

单用户 **GPU 上限 16 张**（= 2 台整机）、CPU 128 核、`MaxJobsPerUser=20`。
所以**同时最多跑 2 个 8 卡作业**，第 3 个会被 QoS 拒绝。

---

## 6. 三期特有故障速查

| 现象 | 原因 | 处理 |
|---|---|---|
| `squeue` 显示 0 个作业 | `PrivateData=jobs` 隐藏了他人作业 | 属正常，看 `sinfo` 节点状态 |
| `git pull` / `pip install` 卡住 | 三期无外网 | 从本机 SFTP 推送；依赖离线安装 |
| `sbatch: error: Batch job submission failed: Invalid qos specification` | 超过 16 卡配额或 20 作业上限 | 减少并发，等作业结束 |
| 8 卡作业一直 `Priority` | 凑不出整机（空闲卡碎片化） | 缩短 `--time`；或先用单卡 |
| `debug` 分区报时限错误 | `debug` 上限 30 分钟 | `--time=00:29:00` |
| 免密登录失效 | `~/.ssh` 权限被改 | `chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys` |

---

## 7. 快速参考卡片（三期）

```bash
# ---- 连接（本机）----
ssh hpc3

# ---- 集群概览 ----
sinfo -s                                          # 分区汇总
sinfo -o "%20P %6D %8c %10m %12G %12l"            # 分区+CPU+内存+GPU+时限
sinfo -h -N -o "%T" | sort | uniq -c              # 节点状态（替代 squeue）

# ---- 排队预测（不提交）----
sbatch --test-only -p acd_u --gres=gpu:8 --mem=200G --time=1:00:00 --wrap true

# ---- 提交 ----
sbatch ~/ztwang/tools/sbatch/gpu_h100x8.sbatch

# ---- 从本机同步代码到三期 ----
python hpc_tools/sync_ztwang.py --host hpc3

# ---- 看 GPU 型号（在一个作业里）----
nvidia-smi --query-gpu=name,memory.total --format=csv
```
