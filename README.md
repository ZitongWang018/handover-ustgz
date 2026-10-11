# ztwang — HPC 工作区

HKUST-GZ HPC 二期的个人工作区。代码、文档、作业模板统一放在这里。

> 远程仓库：**https://github.com/ZitongWang018/handover-ustgz**（Private）
> 集群副本：`~/ztwang/`，已接好 remote。协作流程见 `docs/GITHUB.md`。

## 结构

```
ztwang/
├── docs/                   集群规范与速查
│   ├── CLUSTER_GUIDE.md    ★ 集群使用规范与规则（主文档，必读）
│   ├── ENVIRONMENT.md      环境事实清单（路径/命令/模型，机器可读）
│   ├── CLUSTER_SNAPSHOT.md 集群状态快照
│   └── GITHUB.md           GitHub 协作与提交流程
├── tools/                  工具
│   ├── hpc.py              远程控制脚本（从本机跑远端命令）
│   ├── check_cluster.sh    一键巡检负载/配额/队列
│   ├── sync_ztwang.py      本机 → 集群 目录镜像
│   ├── push_github.py      建库/推送（token 走环境变量）
│   ├── gh_cluster_setup.py 集群侧接 remote + 同步
│   └── sbatch/             作业模板
│       ├── gpu_a800x8.sbatch   A800 × 8 整机（8ka 专用队列）
│       ├── gpu_a800.sbatch     A800 单卡/多卡
│       ├── cpu_job.sbatch      CPU 多核
│       └── test_debug.sbatch   30 秒自检
├── selfplay/               selfplay 项目工作区
│   ├── src/                源码
│   ├── configs/            实验配置
│   └── scripts/            启动脚本
└── logs/                   运行日志（不入库）
```

## 30 秒上手

```bash
ssh hpc2
export PATH=/opt/slurm/bin:$PATH

bash ~/ztwang/tools/check_cluster.sh              # 先看集群忙不忙
sbatch --test-only -p i64m1tga800u8ka \
       --gres=gpu:a800:8 --time=1:00:00 --wrap true   # 预测起算时间
sbatch ~/ztwang/tools/sbatch/gpu_a800x8.sbatch    # 提交
squeue -u xyu608                                  # 看队列
```

## 三条最重要的事

1. **8 卡作业只能提交到 `i64m1tga800u8ka` / `i64m1tga40u8ka`**，其它分区会被拒。
2. **登录节点不跑计算**，一律 `sbatch`。
3. **大文件写 `~/jhspoolers`（SSD）**，家目录 HDD 已用 96%。

详见 `docs/CLUSTER_GUIDE.md`。
