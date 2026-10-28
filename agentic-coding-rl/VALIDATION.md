# 准备检查记录

检查环境：本地 Windows，Python 3.12，Git for Windows / Git Bash。未连接训练服务器，未分配 GPU。

已执行：

- 实际 clone Miles、Harbor，核对 `UPSTREAM.lock.json` 中两个 commit；tracked 上游文件没有修改。
- CPU 单元检查：不同算法和同步/异步下的轨迹预算、GPU 分配、单 rollout 使用 critic、TIS/MIS 无重复比率修正、Score Centering 配置、训练/评测 ID 隔离、gold patch 不进入 rollout metadata、路径越界拒绝、不同模型 RoPE 参数和原生算法字段。
- 生成六种算法 × 两种模式的训练命令，包含实际读取固定 Miles 的模型架构定义；不是自行重写模型参数。
- 包装脚本逐个通过 Bash 语法检查，Python 文件通过编译检查，JSON 文件通过解析。
- `bootstrap` 在已经 clone 的目录上重新运行，验证幂等性与来源检查。

可重跑的 CPU 命令：

```bash
python3 run.py bootstrap
python3 -m unittest discover -s tests -v
python3 -m py_compile launcher.py run.py tests/test_launcher.py
for f in scripts/*.sh; do bash -n "$f"; done
```

尚未验证：

- 目标 GPU 的显存、性能和正确训练更新。
- 容器拉取、固定 Miles 与该容器 SGLang/Megatron/router 的实际兼容性。
- Harbor 环境依赖安装、Docker 仓库镜像构建、实际 SWE agent 交互与 grader。
- Score Centering 的实际 generation-time top-k 元数据协议。
- 真实训练结果、SWE-bench Verified 分数、SAO 算法或论文结论。

入口可供目标机器首跑；通过两步 GPU smoke、检查轨迹和 checkpoint 后，才能把其状态提升为“训练链路已验证”。学习效果和论文对齐需另外的预算、seed 和评测结果。
