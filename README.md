# ztwang — HPC 工作区

HKUST-GZ HPC **二期 + 三期** 的个人工作区。代码、文档、作业模板统一放在这里。

> 远程仓库：**https://github.com/ZitongWang018/handover-ustgz**（Private）
> 集群副本：二期 `~/ztwang/`（有 git remote）、三期 `~/ztwang/`（无外网，靠 SFTP 同步）
> 协作流程见 `docs/GITHUB.md`

## ⛔ 两条铁律（先看这个）

1. **只能在 `ztwang/` 里创建和编辑文件**。外部 ckpt / 模型 / 他人目录一律**只读**。
2. **训练必须存断点，且只保留一个**。不允许堆积 `ckpt-1000`、`ckpt-2000`…

详见 `docs/CLUSTER_GUIDE.md` 第 0 节。

## 结构

```
ztwang/
├── docs/                    集群规范与速查
│   ├── CLUSTER_GUIDE.md     ★ 使用规范与规则（主文档，必读）
│   ├── HPC3_GUIDE.md        ★ 三期 ACD (H100) 专项指南
│   ├── ENVIRONMENT.md       环境事实清单（路径/命令/模型，机器可读）
│   ├── CLUSTER_SNAPSHOT.md  两集群状态快照与对比
│   └── GITHUB.md            GitHub 协作与提交流程
├── tools/                   工具
│   ├── hpc.py               远程控制脚本（从本机跑远端命令）
│   ├── check_cluster.sh     一键巡检（自动识别二期/三期）
│   ├── sync_ztwang.py       本机 → 集群 目录镜像（--host hpc2|hpc3）
│   ├── push_github.py       建库/推送（token 走环境变量）
│   ├── gh_cluster_setup.py  二期接 remote + 同步
│   └── sbatch/              作业模板
│       ├── gpu_h100x8.sbatch   H100 × 8 整机（三期 acd_u）
│       ├── gpu_a800x8.sbatch   A800 × 8 整机（二期 8ka 队列）
│       ├── gpu_a800.sbatch     A800 单卡/多卡
│       ├── cpu_job.sbatch      CPU 多核
│       └── test_debug.sbatch   30 秒自检
├── selfplay/                selfplay 项目工作区
│   ├── src/                 源码
│   ├── configs/             实验配置
│   └── scripts/             启动脚本
└── logs/                    运行日志（不入库）
```

## 30 秒上手

```bash
ssh hpc2        # 二期（A800/A40，可上外网）
ssh hpc3        # 三期（H100，无外网）

export PATH=/opt/slurm/bin:$PATH    # 仅二期需要

bash ~/ztwang/tools/check_cluster.sh              # 看集群忙不忙
sbatch --test-only -p acd_u --gres=gpu:8 \
       --time=1:00:00 --wrap true                 # 预测三期 8 卡起算时间
sbatch ~/ztwang/tools/sbatch/gpu_h100x8.sbatch    # 提交
squeue -u xyu608                                  # 看队列
```

## 怎么选集群

| | 二期（hpc2） | 三期 ACD（hpc3） |
|---|---|---|
| GPU | A800 80GB / A40 | **H100 80GB** |
| 可见卡数 | 232 | **328** |
| 外网 | ✅ | ❌ **完全隔离** |
| 8 卡等待 | ~26 小时 | **~15 小时** |
| 最长时限 | 14 天 | 7 天 |

- **需要 `git clone` / `pip install`** → 二期
- **环境已备好、纯训练跑量** → 三期（H100 更快，排队更短）

## 三条最重要的事

1. **只在 `~/ztwang/` 里写文件**，外部 ckpt 只读。
2. **训练必须存断点且只留一个**（三期 `~/output` 已堆到 406G 是反面教材）。
3. **8 卡作业**：二期只能提交到 `i64m1tga800u8ka` / `i64m1tga40u8ka`；三期用 `acd_u`。

详见 `docs/CLUSTER_GUIDE.md`。
