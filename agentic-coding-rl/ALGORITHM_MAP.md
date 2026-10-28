# 从现有基线到 SAO

这是后续开发定位图，不是已实现 SAO 的声明。所有路径相对于 `.deps/miles/`。

| 部分 | 复用／修改点 | 需要对齐的语义 |
|---|---|---|
| SWE agent、沙箱、reward | `examples/swe-agent-harbor-docker/`，Harbor 独立服务 | 保持 agent/任务/测试不变，terminal reward 来自 verifier |
| 流式异步采样 | `train_async.py`、`miles/rollout/fully_async_rollout.py` | G=1 后完成一条即可进入组完成队列；仍然按指定全局 batch 更新，不是逐条 optimizer step |
| 真实行为概率 | `miles/rollout/agentic/`、session、`miles/ray/rollout/train_data_conversion.py` | 保存每个 action token 实际生成时的 logprob，多版本轨迹不能用最新权重重算替代 |
| DIS loss | **已在本目录 `algorithms/dis_core.py`、`algorithms/grpo_dis.py` 实现**，通过 `custom_loss_function_path` 接入 | GRPO＋DIS 入口 `--algo grpo_dis`；未来 SAO 可复用 tensor core，当前 hook 只接 GRPO |
| GAE | `miles/backends/training_utils/loss/hub/advantages.py`、`math_utils.py` | 已跳过 observation；增加论文的长度自适应 λ，并分别核对 actor advantage 与 critic target |
| critic | `miles/backends/megatron_utils/actor.py`、`train_async.py` | 独立模型已有；冻结 attention、critic:actor 更新频率、更新后 value 的时序仍需实现和验证 |
| 指标 | loss metrics、rollout version、`dump-details` | lag/ratio、有效 action token、双侧 mask、EV、训练和 rollout 时间 |

推荐消融顺序：异步 GRPO → GRPO+DIS → 单 rollout PPO → PPO+DIS → 加入论文的 GAE/critic 设计。每步报告固定生成预算和固定时间，不把异步化收益与 advantage 改进混在一起。

第三方 [miles-values](https://github.com/yyht/miles-values/tree/main-0618) 的 adaptive GAE、critic epochs 可作为实现参考，但它不是当前主干，不能整段无审查复制：路径、并行切分、value target 和数据时序均需对齐。没有把它列为已验证的 SWE 复现。

SAO 原文是 Qwen3-30B-A3B、128K、OpenHands 的 SWE 设置；当前默认是 4B、16K、mini-swe-agent。前者用于论文对齐，后者用于预算较小的算法对比，两者的绝对分数不可直接比较。
