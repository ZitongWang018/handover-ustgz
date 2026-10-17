# 30B Explorer-Zero 训练交接（2026-10-18）

本轮仅完成代码与本地 CPU 验证，没有访问服务器、安装依赖、启动训练/评测或切换权重。用户报告 30B baseline 已可运行；GitHub 尚未提供该轮完整轨迹供本轮独立复核。本报告不声称 30B 八卡已实测成功、无 OOM 或达到论文效果。

## 训练实现

新增显式 `--memory-profile qwen3_moe_cpu --backend hf`，继续使用八个数据并行 rank，每卡一份冻结 BF16 基座和独立 proposer/solver FP32 LoRA。采样与反向共用同一份 HF 模型，不启动第二份 vLLM 基座，也不依赖当前 vLLM 0.8.5.post1 的 MoE LoRA 能力。不是量化、FSDP 或张量并行。

- AdamW 的两组动量与计数常驻 CPU，按单个参数临时更新并回传 GPU；无持久 CPU 参数副本。恢复时不把全部状态转到 GPU，也不克隆全部 CPU 动量。
- 正式两个相位的反向和梯度自检都使用 autograd `save_on_cpu`；保留非重入梯度检查点。模型通过固定单卡 device map 流式加载，避免同时在八个进程主存中物化完整基座。
- 位移诊断只在 CPU 抽查最多三个张量、各 4096 个元素，不再为诊断复制整组 GPU adapter；该值明确不是全局位移最大值。
- 仅将训练生成的微批上限从 2 改为 1；全局 G=16、GP=8、TPS=1、四轮各 25+25 步、上下文、提示、奖励、LoRA all-linear/r32/alpha32 和学习率等不变。所有资源覆盖写入训练 manifest 与哈希。
- 每个 rank 的独立 `runtime/memory_rank*.json` 统计实际参数及优化器状态字节。它不预测激活、KV、通信和碎片峰值，不能用来证明整机内存够用。
- 沿用完整封存、每 rank RNG、任务池、优化器、训练游标、四个 outer 不可变快照及续训验证。旧 4B 默认加载和优化器路径保留。

## 运行方式（供获得授权后的执行者使用）

先读仓库 `AGENTS.md`，核验真实授权目录、模型/数据/缓存路径、可用资源和本轮操作授权。以下都是待替换的路径，不授权启动服务器作业。prepare baseline 和训练必须使用同一解释器、原始配置及环境。

```bash
export SKR_PYTHON=/path/to/approved/python
bash /path/to/repo/selfplay/scripts/run_train.sh \
  --base /path/to/existing/Qwen3-30B-A3B-Instruct-2507 \
  --alfworld-data /path/to/existing/alfworld/data \
  --baseline-contract /path/to/completed/base30b/contract.json \
  --memory-profile qwen3_moe_cpu --backend hf \
  --run-id UNIQUE-30b-run
```

同样命令加 `--smoke` 只在已授权时做八卡短流程检查；它仍加载完整 30B，不能据此承诺显存足够，且其 checkpoint 不可正式续训或评测。正式续训需相同参数与 `--run-id`，再加 `--resume /path/to/run/ckpt`，不能隐式恢复或改 profile。

all-linear MoE 的 adapter 和 CPU 动量都很大；四份快照还各包含完整优化器状态。执行前按实际报告检查整机 RAM 与可用磁盘，预留保存/恢复、激活卸载及临时 checkpoint 的峰值空间。CPU 卸载有传输开销，HF MoE 逐专家计算可能较慢；不得据 CPU 检查承诺 vLLM 速度或三天内完成。

## baseline 与训练后评测边界

本轮未修改 `config.yaml`、原始评测器、提示/动作协议或其哈希清单，因此 a074f5c 下匹配的 baseline 合同仍按原校验规则使用。HF 是训练采样后端，不表示把既有 vLLM baseline 改成 HF。

**只用 `run_train.sh`，不要用 `run_experiment.sh` 自动评测该 profile。** 当前 vLLM 0.8.5.post1 仍不能加载 MoE LoRA checkpoint。已有 `evaluate_run.py` 的原始 vLLM 四快照评测路径尚不适用于该训练 profile；不关闭 LoRA冒充训练臂，不放宽合同检查。

后续如果仍用现有依赖，可在独立的 HF 评测配置/结果目录，用原 `evaluate.py` 分别准备和运行 HF base-only 与 HF checkpoint 合同，再在同一 HF 设置下配对比较；原 solver 设置必须保持一致。不能将 HF checkpoint 和原 vLLM baseline 强行按同签名比较。该后续评测没有在本轮执行。

## 协作与审查

使用用户指定的 deepseek-worker skill，API model `deepseek-flash`。worker 只生成只读 scratch 提案，GPT 检查状态、文件边界、原始 SHA256 后选择性应用并测试；worker 没有提交、推送或修改 live 项目的权限。

成功提案：`ede48fe8-c192-4f9b-ba60-a758f0aa1fd6`（内存模块）、`3a3fa93a-09c0-408d-9cd4-411e351a68e6`（CPU AdamW）、`0d6112f3-c145-4ea9-9ab4-2278ba83c702`（入口）、`6d0a6779-ad80-4df2-9b2f-36cf1b732881`（运行时）。部分只读 shell 检查被 worker 沙箱拒绝，不视为通过。两个大提案因输出超限失败，保留失败材料且未应用其部分内容，随后拆小任务。

GPT 另外修正了草案的偏差：避免恢复时复制全部 CPU 动量；对齐 AdamW 标量 bias correction；纠正部分补丁的缩进与 hunk 格式；保留默认 4B loader/诊断行为；修正 CPU 更新临时空间估算。补充了 PyTorch AdamW 对照、真实 tiny Qwen3-MoE 双 adapter 更新、正式封存恢复下一步一致、profile/baseline 合同及固定设备加载测试。

完整 CPU 回归：`100 passed, 2 skipped, 2 warnings`（25.89 秒）。跳过的是需要真实 CUDA 的状态驻留检查，以及 Windows 无创建目录符号链接权限的检查；对应 CPU 状态和模拟 alias 检查已通过。两个 warning 来自现有 tiny PEFT 保存测试的 base config 提示。39 个 Python 文件通过 AST 检查，训练源码清单完整；CLI 参数已核对。真实 8×A800 的显存、主存、吞吐、轨迹和非有限 loss 检查仍待获授权后的目标机验收。
