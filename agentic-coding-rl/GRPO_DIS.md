# GRPO＋DIS：入口与算法核对

已提供独立算法实现和启动入口。它保留当前 GRPO 的任务、group advantage、采样、异步调度和评测配置，只替换策略梯度目标；不训练 critic，不等于完整 SAO。代码通过 Miles custom-loss 接口加载，固定上游没有改动。

## 启动

按 [README](README.md) 准备配置、数据、checkpoint 并启动 Harbor 后：

```bash
bash scripts/run_grpo_dis.sh --run-id grpo-dis-smoke --smoke
bash scripts/run_grpo_dis.sh --run-id grpo-dis-s42 --eval
```

同一个通用入口也可以直接调用：

```bash
bash scripts/in_container.sh train --algo grpo_dis --mode async --run-id grpo-dis-s42 --eval
```

上面两条正式训练命令等价，选择其一；每次新 run 使用新 ID。同步模式传 `--mode sync`。预览命令时用 `python3 run.py train --algo grpo_dis --run-id preview --dry-run`，不需要 GPU，不会创建 run 目录。GPU 包装器使用 `SWE_RL_CONFIG` 切换配置。

旧配置自动获得默认边界；新配置示例含：

```json
"dis_ratio_low": 0.2,
"dis_ratio_high": 4.0
```

这是 `training` 下的字段。默认由 SAO coding 设置 `epsilon_low=0.8, epsilon_high=3.0` 对应得到；以直接 ratio 边界配置，避免与 PPO epsilon 混用。[SAO §3.1、§4.1](https://arxiv.org/html/2607.07508v1)

## 实际目标

令 `p` 为当前 learner 对实际动作 token 的概率，`q` 为生成时记录的实际采样概率：

```text
r_t = exp(log p_t - log q_t)
m_t = 1{dis_ratio_low < r_t < dis_ratio_high}
L_PG = -reduce_original(m_t * r_t * stop_gradient(A_t))
```

`A_t` 沿用 Miles 的 GRPO 组内归一化 advantage；`q` 也 stop-gradient。该 surrogate 的梯度是 `-m*r*A*grad(log p)`，等价于将 IS 权重视为固定权重的 score-function 梯度。不能把 `r*A*log p` 的每一项都保留梯度，否则会多出一项。

- 严格区间，端点本身屏蔽，与原文 Eq. (3) 一致；判定与 advantage 正负无关。
- 不使用 `batch.log_probs` 的重算 old policy，不做 PPO `min/clip`，不再叠加 TIS/MIS。
- 在 log 空间判断区间，只对保留 token 求指数，避免 `exp(10000)*0` 的 NaN backward。
- observation/padding 不参与策略梯度，也不消费 action-token 分母。它们的非有限占位值在计算前归零。
- action token 的非有限 logprob/advantage 会报错，不静默当作可接受数据。
- DIS 拒绝的 token 只把分子贡献置零，**保留原始 action-token／sequence 分母**，不按剩余 token 重新放大。
- 默认与已有 GRPO 相同，先算每条轨迹的 action-token 均值，再由 Miles 按 batch/DP/microbatch 缩放。没有改变 group size、batch、训练温度或评测温度。

**KL／entropy 与 PG mask 分开。**为保持原有 baseline 的正则化语义，reference-KL 和 entropy 仍作用于所有原始 action token。因此区间外 token 的 PG 梯度为零，但非零 KL/entropy 系数仍可产生正则化梯度。若需要纯 DIS 目标，把这两个系数同时设为零，并对照组使用相同设置。原文没有完整交代这些额外正则化的实现细节，这里明确记录我们的选择。

当前入口只接受 `top_p=1, top_k=-1`，避免拿未重归一化的 full-vocabulary 概率冒充过滤后采样概率。兼容使用原生 response-aligned logprob、TP/CP mask 和 reducer；不直接在本地重写 Megatron softmax kernel。

## 运行记录与指标

`manifest.json` 增加 DIS 版本、边界、严格端点和归一化定义，同时记录 `algorithms/*.py` 的 SHA256。旧的 run 不会自动覆盖或续训。

主要训练指标：`pg_loss`、`dis_kept_fraction`、`dis_low_mask_fraction`、`dis_high_mask_fraction`、`dis_ratio_weight`、`dis_log_ratio`，以及原有 `kl_loss`／`entropy_loss`。指标遵循 Miles 的聚合约定：默认是轨迹均值再做 batch 均值；不是直接把一个 microbatch 的标量当全局结果。

## 已做和未做的验证

CPU 测试环境单独安装 CPU PyTorch，**不要把测试 requirements 装进 GPU 训练镜像**：

```bash
python3 -m venv .venv-tests
.venv-tests/bin/python -m pip install -r requirements-test.txt
.venv-tests/bin/python -m unittest discover -s tests -v
```

测试包含正负 advantage、边界、极端 ratio、masked NaN、混合精度、有限差分 gradcheck、真实 softmax 的梯度/optimizer-step oracle、默认和定制边界、实际 CLI dry-run、子进程 hook 导入、原生参数 schema/function registry、原始归一化分母和 microbatch scaling。

`tests/native_cpu.py` 直接执行固定 Miles 的纯 reducer／布局／dispatcher 函数体，移除 GPU 导入和编译装饰器；CP=2 用两个逻辑 rank 检查切片和归约，**不是 NCCL 或真实分布式测试**。adapter 的 CPU logits extractor 也是测试夹具，未冒充实际 TP/CUDA kernel。

仍需目标 GPU 两步 smoke：真实 SGLang 行为 logprob、Megatron backward、GPU TP/CP 集成、Docker SWE verifier 和 checkpoint。没有真实 SWE 学习曲线或 Verified 分数，不声称已复现论文 27.0%。完整 SAO 仍待实现。
