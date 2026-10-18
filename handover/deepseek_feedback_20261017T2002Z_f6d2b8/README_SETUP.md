# 环境速查：SSH / 集群登录 / 项目结构 / 链接

> 本文件**不含**训练与评测的协议配置（那部分见仓库 `selfplay/config.yaml` 与 `README.md`）。
> 也**不含任何私钥、密码、token**。

---

## 1. SSH 配置

配置文件位置：`C:/Users/dasou/.ssh/config`

```ssh-config
Host hpc2
    StrictHostKeyChecking accept-new
    HostName hpc2login.hpc.hkust-gz.edu.cn
    User xyu608
    IdentityFile C:/Users/dasou/.ssh/hpc2gz_ed25519
    IdentitiesOnly yes

Host hpc3
    StrictHostKeyChecking accept-new
    HostName hpc3login.hpc.hkust-gz.edu.cn
    User xyu608
    IdentityFile C:/Users/dasou/.ssh/hpc2gz_ed25519
    IdentitiesOnly yes
```

- **认证方式：公钥认证，无密码**。两个集群共用同一把密钥。
- 私钥路径：`C:/Users/dasou/.ssh/hpc2gz_ed25519`
  公钥路径：`C:/Users/dasou/.ssh/hpc2gz_ed25519.pub`
- ⚠️ 私钥文件当前权限为 `-rw-r--r--`。建议收紧为仅本人可读
  （Windows 下用 `icacls` 限制到当前用户），避免被同机其他账号读取。

## 2. 登录方式

```bash
ssh hpc2      # 二期（A800/A40，有外网）
ssh hpc3      # 三期 ACD（H100，完全无外网）
```

### 两个集群的关键差异

| 项 | 二期 `hpc2` | 三期 `hpc3` |
|---|---|---|
| 账号 / 组 | `xyu608` / `tonggehuang_team` | `xyu608` / `formal` |
| 家目录 | `/hpc2hdd/home/xyu608` | `/data/user/xyu608` |
| GPU | A800 80G、A40 48G | H100 80G |
| 外网 | ✅ 有 | ❌ **完全没有**（github/pypi 均不可达） |
| Slurm account | `root` | `d_tonggehuang_team` |
| Slurm 命令位置 | `/opt/slurm/bin`（**需 `export PATH`**） | `/bin`（无需设 PATH） |
| 最长时限 | 14 天（`long_*`/`emergency_*`） | 7 天 |
| 实测瓶颈 | 8 卡整节点常常要排队 | 无 8 卡限制 |

**三期无外网 ⇒ 代码不能用 `git pull`，只能从二期或本地同步过去。**

## 3. 项目结构

仓库根目录（也是唯一允许的工作边界）：
`/hpc2hdd/home/xyu608/ztwang`

```
ztwang/
├── AGENTS.md                         代理操作硬规则（必读，见 §4）
├── README.md                         项目说明：协议、训练/评测入口、口径边界
├── selfplay/                         ← 主体代码
│   ├── train.py                      训练入口（CLI）
│   ├── evaluate.py                   评测入口（prepare / run / merge / compare）
│   ├── evaluate_run.py               已训练 run 的四快照评测
│   ├── config.yaml                   共享协议配置（提示/解码/训练超参）
│   ├── alfworld.yaml                 ALFWorld 环境配置
│   ├── scripts/
│   │   ├── run_train.sh              训练启动（8 rank torchrun）
│   │   ├── run_eval.sh               评测启动（按合同分片）
│   │   └── run_experiment.sh         训练 + 四快照评测组合流程
│   ├── src/                          实现
│   │   ├── protocol.py               共享提示、chat 模板、tokenization、动作投影
│   │   ├── ez_solver.py              solver rollout
│   │   ├── ez_proposer.py            proposer：可验证任务生成
│   │   ├── ez_train.py               交替自博弈训练主循环
│   │   ├── ez_vllm_policy.py         vLLM 推理封装
│   │   ├── rollout.py                HF 推理路径
│   │   ├── eval_runner.py            评测执行 / 合同校验
│   │   ├── evaluation.py             评测集发现与合并
│   │   ├── checkpoint.py             断点封存与校验
│   │   ├── training_contract.py      训练前基线合同校验
│   │   ├── training_runtime.py       训练运行时（引擎/进程编排）
│   │   ├── training_memory.py        显存/内存 profile（含 30B CPU-offload）
│   │   ├── training_snapshots.py     outer 快照
│   │   ├── cpu_adamw.py              CPU 常驻动量的 AdamW
│   │   ├── grpo.py / grpo_norm.py    GRPO 损失与归一化
│   │   ├── reward.py                 奖励
│   │   ├── alfworld_env.py / alfworld_compat.py   环境与 planner shim
│   │   └── ...
│   ├── tests/                        回归测试（协议/合同/训练/评测）
│   ├── third_party/                  离线依赖（不进 git）
│   │   ├── py310deps/                78 个依赖（从 wheels 装）
│   │   ├── glibc232/                 ALFWorld planner shim（libdownward.so 等）
│   │   └── wheels/                   94 个离线 wheel
│   └── runs/                         训练 run（每 run 一个子目录，不进 git）
└── handover/                         各方交接材料（只追加，不覆盖）
```

### 运行时产出位置（都不进 git）

```
results/<name>/         评测合同 + 每 rank 明细 + 合并结果
selfplay/runs/<run-id>/ 训练断点、快照、训练日志
logs/                   sbatch 的 stdout/stderr
.tmp/  .pipcache/       临时文件与 pip 缓存（**必须放这里，不要写 /tmp 或 ~**）
```

### 关键外部路径（只读）

| 用途 | 路径 |
|---|---|
| 模型目录 | `/hpc2hdd/home/xyu608/thinktool/models/` |
| ALFWorld 数据 | `/hpc2hdd/home/xyu608/.cache/alfworld` |
| Python 解释器 | `/hpc2hdd/home/xyu608/miniconda3/envs/thinktool/bin/python` |

## 4. 链接

| 内容 | 链接 |
|---|---|
| **GitHub 仓库** | https://github.com/ZitongWang018/handover-ustgz |
| **AGENTS.md**（代理硬规则，必读） | https://github.com/ZitongWang018/handover-ustgz/blob/main/AGENTS.md |
| README（协议与入口） | https://github.com/ZitongWang018/handover-ustgz/blob/main/README.md |
| 配置 | https://github.com/ZitongWang018/handover-ustgz/blob/main/selfplay/config.yaml |

**Fork/分支约定**：远端固定为 `ZitongWang018/handover-ustgz`；
执行代理的反馈走 `deepseek-feedback/<唯一会话ID>` 分支，**不直接推 main**。

## 5. 与项目无关但同机的其它 SSH 入口（勿混用）

`~/.ssh/config` 里还有 `connect.westc.seetacloud.com`、`seetacloud-codex`、
`hpc`（`jms.sjtu-lumia.cn`）、`a800`（`103.79.25.68`）等条目，**都不属于本项目**，
不要在这些机器上操作本项目的数据。
