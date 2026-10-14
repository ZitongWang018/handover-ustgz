# EZ 评测协议与解释边界

本文区分论文报告、公开的 rich-agent 对照和本仓库评测实现。命令生成与评测完成不代表论文复现已经成立；所有结果必须连同 JSON 中的 protocol_manifest、完整任务行和逐步 trace 一起解释。

## 两种明确的评测协议

paper-approx 是默认评测入口的近似协议：JSON tool-call、30 步、temperature 1、每任务 4 次采样、无 demo（除非显式环境开关改变 prompt）。论文 Table 1 报告 valid-unseen 上 baseline 17.14±5.7、Ours 85.07±1.6，并使用每个任务 4 个采样。论文 Figure 14 展示 JSON 格式和空的 thinking 前缀，但标注了 “Example Omitted”；遗漏的示例具体内容未知。论文没有给出足以确定 sampling 参数的完整信息。因此这里的默认无 demo 不能称为精确复现，也不能把 one-shot 等同于 n0；paper-approx 会把这一边界写进 manifest。

public-rich 是另一套公开 scaffold 的 zero-shot 对照：rich rules、编号 admissible actions、最近 5 轮历史、失败动作最近 10 条、最多 50 步、greedy、每步最多生成 32 tokens、不加 few-shot。它按每任务一次采样运行。公开结果中 strict prompt 报告 weighted SR 3.7%、macro SR 3.5%；rich prompt 报告 weighted SR 10.4%、macro SR 12%。这些数字来自其公开设置，不能与本仓库结果直接比较，也不表示复现相同环境就会得到相同分数。当前 TextWorld/ALFWorld 安装版本的命令方言可能与公开仓库版本不同；trace 会保留实际生成文本、菜单解析方式、执行命令和环境观测。

custom 用于显式改动协议后的结果。改动样本数、步数、解码参数、prompt style 或实际生效的环境开关后，输出会标为 custom 并记录 requested 与 effective decode、环境变量、prompt/tokenizer hash、模型/adapter、seed、后端、软件版本和上下文上限。rich 菜单解析不受 EZ JSON strict 开关影响；这些 rich 路径忽略的开关仍会在环境快照中列出。

samples=4 的统计是 4 份完整任务集的 SR。报告同时提供 episode-weighted SR、6 类 task-macro SR、每个 sample 的 whole-suite SR 及 ddof=0/ddof=1 标准差、每任务 sample 均值和总体标准差。episode_bernoulli_std 是单次 Bernoulli 结果的离散度，不是跨训练 seed 的误差，也不是论文 ± 的替代品。avg@4 不是 pass@4；不会把“4 次中至少成功一次”统计成论文 SR。

全量入口只接受 134 个 valid-unseen 任务，按声明的 shard 文件集合合并，要求每个 task/sample 恰好出现一次且所有 shard 的 manifest、实际解码参数、seed、上下文和运行 ID 一致。单独 quick/mid eval 是训练期子集指标，通常只有单样本；不能称为论文最终指标。

## Matched base 与 adapter 命令

同一 base、seed、split、prompt、采样和解码参数下，先运行 base control，再运行 solver adapter。完整任务集的 paper-approx 示例：

```bash
python src/eval_ez.py --protocol paper-approx --base "$ZTWANG_MODEL_PATH" --base-only --backend vllm --n 0 --samples 4 --select stratify --seed 42 --out eval_out/base-paper-approx.json
python src/eval_ez.py --protocol paper-approx --base "$ZTWANG_MODEL_PATH" --ckpt "$EZ_CKPT" --backend vllm --n 0 --samples 4 --select stratify --seed 42 --out eval_out/solver-paper-approx.json
```

公开 rich control 使用每任务一个 greedy sample；在相同参数下分别跑 base 与 adapter：

```bash
python src/eval_ez.py --protocol public-rich --base "$ZTWANG_MODEL_PATH" --base-only --backend vllm --n 0 --samples 1 --select stratify --seed 42 --out eval_out/base-public-rich.json
python src/eval_ez.py --protocol public-rich --base "$ZTWANG_MODEL_PATH" --ckpt "$EZ_CKPT" --backend vllm --n 0 --samples 1 --select stratify --seed 42 --out eval_out/solver-public-rich.json
```

全量 8 卡入口 scripts/ez_eval_full_8gpu.sbatch 使用唯一 run-id 和显式 shard manifest；paper approximation 默认 4 samples，public rich 默认 1 sample。可用 EZ_MAX_STEPS、EZ_MAX_NEW_TOKENS、EZ_MAX_CONTEXT_TOKENS、EZ_TEMPERATURE、EZ_TOP_P、EZ_TOP_K、EZ_REPETITION_PENALTY 显式覆盖，任何偏离 profile 默认值的运行都会标成 custom。完整入口固定 8 shards；EZ_SHARDS 若设置，只接受 8。

## 模型与 oracle 解释

公开论文/代码称模型为 Qwen3-4B。该名称不能推断它是 Qwen3-4B-Base；Qwen3-4B 模型卡将训练阶段列为 Pretraining & Post-training，而 Base 模型卡单独列为 Pretraining。[Qwen3-4B model card](https://huggingface.co/Qwen/Qwen3-4B) [Qwen3-4B-Base model card](https://huggingface.co/Qwen/Qwen3-4B-Base) 当前代码保留用户显式传入的 base 路径，不会擅自把它替换成 Base 权重。所有报告应该读取 manifest 中的实际模型路径。

ez_eval_oracle_check.py 直接把 ALFWorld expert 的原始命令送进环境，绕过模型生成和 JSON parser。它检查环境动作/判胜链路，不能单独归因模型 parser 或证明模型是唯一问题。专家也可能超过步数预算、计划为空或遇到环境执行错误。脚本按任务输出 win、budget truncation、空 expert-plan 步数和环境状态；这些诊断不形成“模型成功率上限”，也不因 expert 未全胜就断言框架错误。失败结果需结合原始命令、观测、版本和 trace 分析。

## 可追溯来源

- Explorer-Zero 为匿名 ICLR 2027 under-review 稿件。本地抽取的稿件版本中，Table 1 在第 7 页给出每任务四样本结果；Table 3 在第 15 页列出训练超参；Figure 14 在第 24 页给出 solver prompt，并将示例标为 “Example Omitted”。这些是当前审计拿到的稿件版本页码，不能推断被省略的例子内容。
- Rich prompt/结果核对基于公开实现的 [RESULTS.md](https://github.com/Masoudjafaripour/VLM-VLA-Embodied-Agents/blob/main/baseline/ALFWorld/alfworld_eval/RESULTS.md)、[llm_agent.py](https://github.com/Masoudjafaripour/VLM-VLA-Embodied-Agents/blob/main/baseline/ALFWorld/alfworld_eval/llm_agent.py)、[prompts.py](https://github.com/Masoudjafaripour/VLM-VLA-Embodied-Agents/blob/main/baseline/ALFWorld/alfworld_eval/prompts.py) 和 [README.md](https://github.com/Masoudjafaripour/VLM-VLA-Embodied-Agents/blob/main/baseline/ALFWorld/alfworld_eval/README.md)。其 strict 与 rich weighted/macro SR 见 RESULTS.md；这些公开结果不表示与本文本环境版本匹配后会得到相同分数。
- ReAct 的公开 ALFWorld notebook 可用于对照交互 scaffold：[ReAct alfworld.ipynb](https://github.com/ysymyth/ReAct/blob/master/alfworld.ipynb)；论文页：[ReAct](https://arxiv.org/abs/2210.03629)。
