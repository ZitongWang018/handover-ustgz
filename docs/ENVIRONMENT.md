# 环境事实清单

> 机器可读的集群事实。供 AI 助手在无交互探查时直接引用。
> 覆盖 **二期** 与 **三期 ACD** 两个集群（两者完全独立）。
> 采集时间：2026-10-11

## 身份

```yaml
user: xyu608
uid: 204227
gid: 201375
group: tonggehuang_team          # 二期
group_phase3: formal             # 三期
groups: [tonggehuang_team, HKUST-GZ, domain_users, Redbird_MPhil_Programs, College_of_Future_Tech]
login_shell: /bin/bash
home: /hpc2hdd/home/xyu608        # 二期；三期是 /data/user/xyu608
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
internet: 可访问（github.com/pypi 正常）
```

### 三期主机

```yaml
cluster: HKUST-GZ HPC 三期 ACD
slurm_cluster_name: chess
login_node: hpc3login.hpc.hkust-gz.edu.cn      # IP 10.120.81.30，实际登录到 ACD-Login-1
portal: https://hpc3login.hpc.hkust-gz.edu.cn/hpc-user/#/login
ssh_server: OpenSSH_10.5
auth_methods: [publickey, password]
internet: ❌ 完全隔离（github.com / pypi.org 均连不上）
docs: https://docs.hpc.hkust-gz.edu.cn/docs/hpc3/
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

---

# 三期 ACD 事实清单

## 路径

```yaml
home: /data/user/xyu608                  # 挂在 /data（19P, 38% used, 余 12P）
share: /share                           # 平台软件/cuda/dataset
share_anaconda: /share/anaconda3
share_cuda: [/share/cuda132, /share/cuda133, /share/bak_cuda128]
share_dataset: /share/dataset
workdir: /data/user/xyu608/ztwang
tmp: /tmp                                # 节点本地，作业结束即丢
```

> 三期**没有** `~/jhspoolers` 等二期软链。训练产物写 `~/ztwang/runs/<jobid>/`。

## Slurm

```yaml
version: 23.11.11
cluster_name: chess
bindir: /bin                             # 无需设 PATH（与二期不同）
account: d_tonggehuang_team              # 显示为 d_tonggeh+
default_partition: acd_u
default_qos: normal
private_data: jobs                       # ⚠️ squeue 看不到别人的作业
select_type: select/cons_tres
gres_types: [gpu]
controllers: [ACD-Manage-3, ACD-Manage-2]
```

### 分区

| 分区 | 节点 | 单节点 | GRES | 时限 | 优先级 |
|---|---|---|---|---|---|
| `acd_u` ⭐默认 | 40 | 96c / 2T | gpu:8 | 7d | 100 |
| `acd_ue` | 40 | 96c / 2T | gpu:8 | 7d | 200 |
| `emergency_acd` | 40 | 96c / 2T | gpu:8 | 7d | 300 |
| `debug` | 1 (`ACD1-63`) | 96c / 2T | gpu:8 | **30m** | 1 |

节点：`ACD1-{1-4,6-9,11-26,28,30-33,36,50-52,54-56,59,61,64,67}`（acd_* 三分区共用同一批 40 台）

### 配额与限制

```yaml
allow_groups: [formal, admin]
qos_normal:
  MaxTRESPerUser: cpu=192, gres/gpu=16
  MaxJobsPerUser: 20
user_quota_doc:  CPU 128 核 / GPU 16 张 / 单作业 ≤7 天
max_time: 7-00:00:00                     # 可申请延长一次，不得超 7 天
```

## 硬件

```yaml
gpu_model: NVIDIA H100 80GB HBM3         # nvidia-smi 实测
gpu_total_visible: 328                   # 41 节点 × 8
gpu_allocated: 299                       # 2026-10-11 实测
gpu_free: 29                             # 分散在各节点，凑不出整机
cpus_per_node: 96                        # 2 × 48
mem_per_node: 2000000M                   # ≈2TB
interconnect: 400Gb/s RoCE v2
intra_node_gpu: 800GB/s 双向
storage: 17PB 分布式
nodes_visible: 41                        # acd_u 40 + debug 1
fp16_peak: 1078 Pflops                   # 官方口径
```

## conda

```yaml
root: /share/anaconda3                   # 登录即激活
envs: [ReVision, revision, pytorch, pytorch251, pytorch271, pytorch2.14.0,
       vllm, vllm017, flash_attn3, jax_gpu, video_dit, open3d]
```

## 家目录现状（只读，勿改）

| 目录 | 体积 | 内容 |
|---|---|---|
| `output` | **406G** | 3 组 `checkpoints-sft-bunny-*` 断点 |
| `model` | 61G | 模型 |
| `models` | 624M | 模型目录 |
| `revision_sft` | 5.3G | ReVision SFT |
| `res` | 1.1G | 权重分片 |
| `ReVision` | 78M | 项目（**非 git 仓库**） |
| `checkpoints-pretrain-bunny-2M` | 44M | 预训练断点 |
| `aistudio` `pkgs` `output` | — | 平台/依赖 |

## 三期特有约束

1. **无外网** → 代码靠 `python hpc_tools/sync_ztwang.py --host hpc3` 推送；依赖需离线安装。
2. **`PrivateData=jobs`** → 判断拥堵必须看 `sinfo` 节点状态或 `--test-only`，不能看 `squeue`。
3. **单机固定 8 卡** → 常规用法是整机 `--gres=gpu:8`。
4. **单用户 GPU 上限 16 张** → 同时最多 2 个 8 卡作业。
5. **最长 7 天**（二期有 14 天队列，三期没有）。

---

# 系统级约束（2026-10-12 实测）

## glibc / 操作系统

| | 二期 | 三期 |
|---|---|---|
| 登录节点 | Ubuntu 20.04, **glibc 2.31** (`mgmt-3`) | Rocky Linux 8.10, **glibc 2.28** (`ACD-Login-1`) |
| 计算节点 | Ubuntu 20.04, **glibc 2.31** (`gpu3-9` via `srun -p debug`) | Rocky 8.6, **glibc 2.28** (`ACD1-63`) |

> **登录节点和计算节点是同一个镜像**，不要指望"计算节点版本更新"来绕开依赖问题。

### 后果：TextWorld 的规划器跑不起来（已解决）

`fast_downward_textworld==20.6.4` 里的 `libdownward.so` 在 glibc≥2.32 上编译，
需要 `GLIBC_2.32`（唯一用到的新符号是 `__libc_single_threaded`）与 `GLIBCXX_3.4.29`。
而 ALFWorld 的 `.tw-pddl` 游戏走 `textworld.envs.pddl.Env`，其 `__init__` 第一句就是
`fast_downward.load_lib()` → 直接 `OSError: version 'GLIBC_2.32' not found`。

**解决方案**（只写自己目录，不动外部环境）：`ztwang/selfplay/env_setup/patch_libdownward.py`
+ `ztwang/selfplay/src/alfworld_compat.py`。关键点：

1. 把 `.gnu.version_r` 里对 `libc.so.6` 的 `GLIBC_2.32` 需求改成 `GLIBC_2.17`，
   **必须同时改 `vna_hash`**（只改字符串会报 `version 'GLIBC_2.17' not found`）；
2. `__libc_single_threaded` 用一个一行 C 垫片提供，在 `import fast_downward` 之前
   用 `ctypes.CDLL(..., RTLD_GLOBAL)` 预加载 —— **`LD_PRELOAD` 救不了**，
   因为版本需求记在 `libc.so.6` 头上而不是调用方；
3. 运行时把 `fast_downward.interface.DOWNWARD_LIB_PATH` 指向 patched 副本
   （`load_lib()` 每次调用都读这个模块全局变量）。

## 可用的训练环境（二期）

| 环境 | Python | 关键包 |
|---|---|---|
| `thinktool` | 3.10.19 | torch 2.6.0+cu124, transformers 4.57.3, **trl 0.26.2**, vllm 0.8.5.post1, peft 0.18.0, accelerate 1.12.0, deepspeed 0.18.3, ray 2.53.0, sglang 0.4.6, flash-attn 2.7.4, numpy 1.26.4 |
| `agentenv-alfworld` | 3.9.25 | torch 2.8.0, textworld 1.6.2, alfworld 0.3.3, fast_downward_textworld 20.6.4, gym 0.26.2, numpy 2.0.2 |

⚠️ **`trl 0.26.2` 的 `rollout_func` 只在 `use_vllm=True` 时生效**，而 vLLM 0.8.5 < TRL 要求的 0.10.2，
所以 selfplay 用的是自研 GRPO（见 `docs/SELFPLAY.md`）。

`~/.cache/alfworld` = 3.5G（`json_2.1.1/{train,valid_seen,valid_train,valid_unseen}` + `logic/` + `detectors/`）。

## PyPI wheel 可用性（二期有外网，已实测）

```
textworld-1.6.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl   ✅
alfworld-0.3.3-py3-none-any.whl                                              ✅
fast_downward_textworld-20.6.4                                                ✅
gym==0.26.2                                                                  ⚠️ 无匹配 wheel，需退到 sdist / 复制
```


