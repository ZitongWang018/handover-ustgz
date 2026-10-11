# GitHub 协作规范

> 远程仓库：**https://github.com/ZitongWang018/handover-ustgz**（Private）
> 本文件面向在本仓库协作的人与 AI 助手，约定代码如何流转。

## 1. 三个位置，一个真相源

| 位置 | 角色 | 说明 |
|---|---|---|
| **GitHub `handover-ustgz`** | **唯一真相源** | 所有最终代码/文档以此为准 |
| 集群 `~/ztwang/` | 运行环境 | 实际跑实验的地方；`git` 已初始化并接好 remote |
| 本机 `WorkBuddy/.../ztwang/` | 中转工作区 | 无 GPU 时写代码/文档的副本 |

**规则：任何改动先提交到 GitHub，再向另外两处同步。** 不要反过来在集群上手改后不回传。

## 2. 常规流程

### 2.1 在 GitHub 上改（推荐，AI 协作主通道）

```bash
# 本机或集群
cd ~/ztwang
git pull                      # 拉最新
# ... 改代码 ...
git add -A
git commit -m "feat: 说明改了什么"
git push
```

### 2.2 集群侧认证（重要）

集群上的 `origin` 是 HTTPS 地址，`git pull/push` 需要 Personal Access Token。
token **没有**常驻在集群上（集群家目录属组为 `tonggehuang_team`，明文存密钥有外泄风险）。

需要拉取/推送时，向助手说一句「同步集群仓库」即可，由助手用一次性 token 完成。
若你希望自己动手，临时导出再删除：

```bash
# 只在当前 shell 生效，退出即失效
export GIT_ASKPASS=~/my_askpass.sh   # 内容见下，或直接改用交互式输入
git pull
unset GIT_ASKPASS
```

> ⚠️ 不要把 token 写进 `~/.git-credentials` 或任何会入库的文件。

### 2.3 从本机推到集群

本机 `hpc_tools/sync_ztwang.py` 会把本地 `ztwang/` 目录整体镜像到集群 `~/ztwang/`：

```bash
python hpc_tools/sync_ztwang.py
```

## 3. 目录约定

```
ztwang/
├── docs/            规范与事实文档（本目录）
├── tools/           工具脚本（hpc.py 远程控制、check_cluster.sh 巡检、同步脚本）
├── tools/sbatch/    Slurm 作业模板
├── selfplay/        当前主线项目
│   ├── src/         源码
│   ├── configs/     实验配置
│   └── scripts/     启动脚本
└── logs/            运行日志（.gitignore 已排除）
```

## 4. 什么绝对不能入库

`.gitignore` 已覆盖以下内容，提交前不要用 `-f` 强推：

- 模型权重：`*.safetensors` `*.bin` `*.pt` `*.pth` `*.ckpt`
- 数据集与缓存：`data/` `datasets/` `*.jsonl` `*.parquet` `hf_cache/`
- 镜像与压缩包：`*.tar` `*.tar.gz` `*.zip` `*.safetensors.index.json`
- 日志与输出：`logs/` `outputs/` `wandb/` `*.log`
- 凭据：`*.token` `.env` `*.pem` `id_rsa*`

**判据：只要单个文件 > 10 MB，或可以用脚本重新生成，就不要提交。**
GitHub 单仓库建议 < 1 GB，硬上限 5 GB；超了无法回滚，只能重建仓库。

## 5. 提交信息规范

```
<类型>: <一句话说明>

类型：feat / fix / docs / refactor / exp / chore
```

- `exp:` 用于实验性改动（改超参、换数据），便于日后区分"代码变更"与"实验变更"
- 一个提交只做一件事，不要把格式化与逻辑改动混在一起

## 6. 给 AI 协作助手的约定

1. **动手前先读** `docs/CLUSTER_GUIDE.md`，里面列了集群硬约束
2. 新增脚本必须提交到仓库，不要只留在集群上
3. 所有 shell 脚本以 **LF** 换行（`.gitattributes` 已强制 `eol=lf`）
4. 不要在仓库里放绝对路径以外的环境假设；集群路径见 `docs/ENVIRONMENT.md`
5. 生成大文件（模型、数据集）只写下载脚本，不提交产物
