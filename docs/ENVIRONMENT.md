# 环境事实清单

> 机器可读的集群事实。供 AI 助手在无交互探查时直接引用。
> 采集时间：2026-10-11

## 身份

```yaml
user: xyu608
uid: 204227
gid: 201375
group: tonggehuang_team
groups: [tonggehuang_team, HKUST-GZ, domain_users, Redbird_MPhil_Programs, College_of_Future_Tech]
login_shell: /bin/bash
home: /hpc2hdd/home/xyu608
```

## 主机

```yaml
cluster: HKUST-GZ HPC 二期 (hpc12)
login_nodes: [hpc2login.hpc.hkust-gz.edu.cn]   # 实际登录到 mgmt-6
login_nodes_phase1: [hpc1login.hpc.hkust-gz.edu.cn]
portal: https://hpc2login.hpc.hkust-gz.edu.cn/
ssh_server: OpenSSH_8.2p1 Ubuntu-4ubuntu0.8
auth_methods: [publickey, password]
vpn: 校内网络直连可用（出现 10.120.x.x / 10.121.x.x 内网 IP）
```

## 路径

```yaml
home_hdd: /hpc2hdd/home/xyu608          # 3.9P, 96% used, 余 173T
ssd: /hpc2ssd                            # 309T, 93% used
ssd_user: /hpc2ssd/JH_DATA/spooler/xyu608        # 软链 ~/jhspoolers
jhai_data: /hpc2hdd/JH_DATA/jhai_data/xyu608     # 软链 ~/jhaidata
share: /hpc2hdd/JH_DATA/share/xyu608             # 软链 ~/.jhshares
filedownload: /hpc2hdd/JH_DATA/filedownload/xyu608  # 软链 ~/.filedownload
workdir: /hpc2hdd/home/xyu608/ztwang
```

## Slurm

```yaml
version: 23.02.3
bindir: /opt/slurm/bin            # 非交互 shell 需手动加入 PATH
account: root
default_partition: a128m512u
default_qos: null
```

### 分区

| 分区 | 节点 | 单节点 | GRES | 时限 |
|---|---|---|---|---|
| i64m512u / i64m512ue | 107 | 64c/512G | — | 7d |
| i64m512r / i64m512re | 13 | 64c/512G | — | 7d |
| a128m512u / a128m512ue | 16 | 128c/512G | — | 7d |
| long_cpu | 106 | 64c | — | 14d |
| i96m3tu / i96m3tue | 4 | 96c/3T | — | 7d |
| emergency_cpu | 107 | 64c | — | 14d |
| i64m1tga800u / i64m1tga800ue | 11 | 64c/1T | gpu:a800:8 | 7d |
| long_gpu | 11 | 64c/1T | gpu:a800:8 | 14d |
| i64m1tga800u8ka | 3 | 64c/1T | gpu:a800:8 | 7d |
| i64m1tga40u / i64m1tga40ue | 3 | 64c/1T | gpu:a40:8 | 7d |
| i64m1tga40u8ka | 1 | 64c/1T | gpu:a40:8 | 7d |
| emergency_gpu | 10 | 64c/1T | gpu:a800:8 | 14d |
| emergency_gpua40 | 3 | 64c/1T | gpu:a40:8 | 7d |
| debug | 6 | 混合 | 部分 | **30m** |
| czhangcn_rent / slurm01_qos / jiahengwei_rent / jingtang_rent / linshiju_rent / pengshan_rent | 1~4 | 专用 | a800 | infinite |

### 约束（Slurm 服务端强制）

```
8-GPU jobs are ONLY allowed on partitions 'i64m1tga800u8ka' and 'i64m1tga40u8ka'
```

## conda

```yaml
root: /hpc2hdd/home/xyu608/miniconda3
version: 26.1.1
activate: source ~/miniconda3/etc/profile.d/conda.sh
envs: [ReVision, SyBC, aflow, agentenv-alfworld, agentenv-sciworld,
       agentenv-searchqa, agentenv-webshop, longlive, longlive2, thinktool]
```

## 本地模型 `~/thinktool/models/`

| 目录 | 模型 | 体积 | 分片 | 备注 |
|---|---|---|---|---|
| qwen3_4b_inst_2507 | Qwen3-4B-Instruct-2507 | 7.6G | 3 | 36 层, hidden 2560, ctx 262144 |
| qwen3_4b | Qwen3-4B (base) | 7.6G | — | ctx 40960 |
| qwen3_8b | Qwen3-8B | 16G | 5 | 36 层, hidden 4096, ctx 40960 |
| qwen3-14b | Qwen3-14B | 28G | — | |
| qwen3-30b-a3b-inst-2507 | Qwen3-30B-A3B-Instruct-2507 | 57G | — | MoE |
| 30b-a3b-inst-2507 | Qwen3-30B-A3B-Instruct-2507 | 56G | — | 重复副本 |

### 可用但不在本账号目录的模型（只读）

```
/hpc2hdd/JH_DATA/jhai_data/mpeng053/pretrain_model/    # Qwen2.5-7B-Instruct, Qwen2-7B-Instruct, ...
/hpc2hdd/JH_DATA/spooler/syang724/models/              # Qwen_Qwen2.5-7B-Instruct, ...
/hpc2hdd/JH_DATA/jhai_data/leeyuxiaoli/Models/         # Qwen3.5-4B, Qwen3.5-9B, ...
```

### 重要结论

**Qwen3 系列不存在 7B 规格。** 官方规格为 0.6B / 1.7B / 4B / 8B / 14B / 32B（稠密）
与 30B-A3B / 235B-A22B（MoE）。7B 量级请用 **Qwen3-8B**（本地已有）；
若必须 7B，则对应上一代 **Qwen2.5-7B-Instruct**。

## HF 缓存 `~/.cache/huggingface/hub/`

```
models--Qwen--Qwen3-4B
models--Qwen--Qwen3-4B-Instruct-2507
models--Qwen--Qwen3-8B
models--Qwen--Qwen3-14B
models--Qwen--Qwen3-30B-A3B-Instruct-2507
```

## 家目录项目

| 目录 | 体积 | 仓库 |
|---|---|---|
| thinktool | 257G | rllm, AFlow, learn-claude-code, worktree-workflow |
| hechenyu | 52G | LongLive |
| MAS | 15G | ViF, ViF_ours |
| docker | 12G | 镜像 tar 包（无仓库） |
| Slime_running | 8.6G | slime, Megatron-LM |
| Theory | — | IA/MultiEMO, K-model |
| clip | — | MiniClip |
| theory_final | 8.0M | ReVision |
| huangzhe | 131M | DLLM-Searcher |
| haiji | 1.0K | 无 |
