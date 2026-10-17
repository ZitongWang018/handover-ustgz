# 在 SkillRise ReAct 设定下训练 Explorer-Zero

服务器执行代理（包括 DeepSeek）必须先读 [AGENTS.md](AGENTS.md)：仅在已确认的 `ztwang` 目录内执行明确批准的任务，默认不得修改实现；问题通过独立 GitHub 反馈分支完整交接。

本仓库在 ALFWorld 上实现 Explorer-Zero 的交替 proposer/solver 自博弈，并把 solver 训练与评测统一到 SkillRise 风格 ReAct 协议。基座模型冻结，训练 proposer 与 solver LoRA；最终比较未训练基座和训练后的 solver adapter。两臂使用同一 `valid_seen` 全量池、提示/动作投影、解码设置和五个评估种子。

这是论文方法在共享 ReAct 设定下的实验，不是 Explorer-Zero 论文原协议的逐项复现，也不声称复现其绝对分数。论文 Table 3 的 ALFWorld 设置包括 4 个 outer loops、每轮 proposer/solver 各 25 步、proposer 与 solver 最多 30 turns、G=16、solver solve-rate rollout 数 GP=8、LoRA r32/α32、AdamW、lr=1e-4、clip=0.3、KL=0.001、subtask/diversity 权重各 0.3。共享实现沿用这些算法和训练超参数，并按每步一个任务、每步全局 16 条 solver rollout 组织 GRPO；训练阶段每个 proposer proposal 的 GP=8 次 solve-rate 采样只用于 proposer 奖励，不进入 solver 梯度。论文 Table 3 报告 8×H100，当前目标训练服务器为 8×A800。

当前目标为 Qwen3-30B-A3B-Instruct-2507，使用 tokenizer 自带 chat template 和 HF 配对评测。论文上下文为 40,960 tokens；本设置 solver 固定 32,768，训练 proposer 为 40,960。solver 仍使用最近 15 组历史、可用动作菜单和 `<action>...</action>` 投影，与论文 Appendix G 的 `<think>`/`<tool_call>` JSON 不同。Proposer 保留 `sub_finish` 快照和 `finish` 终止任务约束；本地模板不声称逐字复用论文。

生成上限 1024、ReAct 温度 0.7 与 top_p 0.95 属于共享评测工程设置，不是 Table 3 的论文超参数。

## 固定共享 solver 协议

- 评测用完整 `valid_seen` 池、种子 0–4，每局最多 30 步；训练数据来自 ALFWorld `train` split。
- 每轮重建同一 ReAct 提示正文，作为单个 user message 调用 `apply_chat_template(tokenize=True, add_generation_prompt=True)`。模板只应用一次，不再二次添加特殊 token；Instruct-2507 仅支持 non-thinking。
- 提示包含当前合法动作（排除 `help`），最多保留最近 15 组已执行动作和观测。
- 解码为 temperature 0.7、top_p 0.95、max_new_tokens 1024；生成在 `</action>` 停止并保留闭合标签。
- 每一步 solver 训练从冻结 proposer 的 rank 0 生成新 proposal 并广播，再为该任务采样全局 G=16 条新 solver rollout；最多尝试 8 次取得有效 proposal，仍失败则停止训练，不回退到历史 task pool。历史 pool 用于审计和续训状态，不作为新 solver 更新的替代样本。
- 保存每轮实际提示与生成 token，按完整轨迹 response token 数加权。

提示、tokenizer、任务池或解码协议变化会改变评测合同；必须为新协议重新准备并运行基线，旧 quick 结果不能直接沿用。

当前唯一推荐设置为 chat/HF，评测 batch=1、shards=8；训练也使用 HF。旧 raw/vLLM 结果保留为历史实验，不删除、不覆盖，不能用于新设置的配对比较。代码保留显式 raw 编码供回归检查，不会自动根据模型猜测输入格式。

## 准备基线并训练

训练需要本地模型目录、ALFWorld 数据目录和 8 张 GPU；不会下载模型或数据。若已经有匹配的 base-only 基线合同，直接传入现有合同；不要覆盖它。仅在尚无合同的情况下，才用原始 `selfplay/config.yaml` 准备合同。训练只读取并校验合同，不会启动基线 prepare 或 evaluation；是否已有完整基线评测结果应单独记录。训练与评测配置必须继续使用同一原始 `selfplay/config.yaml`；训练生成的 resolved config 含 40,960-token proposer/runtime 设置，只用于记录训练，不能拿来准备评测合同。

```bash
export SKR_PYTHON=/path/to/approved/python
"$SKR_PYTHON" selfplay/evaluate.py prepare \
  --config selfplay/config.yaml \
  --alfworld-data /path/to/alfworld/data \
  --base /path/to/local/model \
  --base-only \
  --results-dir results/base30b-chat-hf-UNIQUE \
  --output results/base30b-chat-hf-UNIQUE/contract.json
bash selfplay/scripts/run_eval.sh results/base30b-chat-hf-UNIQUE/contract.json
```

准备合同会固定当前数据根路径、ALFWorld YAML、实际发现的完整 `valid_seen` 池、五种子、评测源码/软件版本、配置和基座/tokenizer文件身份。合同校验拒绝 checksum 或嵌套身份不一致、错误 split、不同模型/solver/source/config/data，或当前发现的任务池与合同不同。合同必须来自 `--base-only` 基线臂。

baseline 与 checkpoint 评测都使用同一配置的 HF 后端；checkpoint 明确加载 solver adapter，不会退回 base-only。合同固定输入格式、有效 chat template SHA256、tokenizer 和源码身份；缺失模板或模板漂移直接拒绝。HF 不使用 vLLM 的显存比例参数作为上限。最新运行方式与验证边界见 [聊天协议与配对评测交接](handover/gpt_feedback_20261018_chat_hf/REPORT.md)。

```bash
bash selfplay/scripts/run_train.sh \
  --base /path/to/local/model \
  --alfworld-data /path/to/alfworld/data \
  --baseline-contract results/base30b-chat-hf-UNIQUE/contract.json \
  --backend hf --memory-profile qwen3_moe_cpu \
  --run-id experiment-001
```

`--baseline-contract` 是必需参数。训练 manifest 和 checkpoint 会嵌入经过校验的完整合同、合同 SHA 和共同身份字段。训练验证使用传入的 base inventory，不会为合同校验重新哈希模型文件或加载权重。

续训必须提供同一 run 的封存 checkpoint、基座、数据和基线合同：

```bash
bash selfplay/scripts/run_train.sh \
  --base /path/to/local/model \
  --alfworld-data /path/to/alfworld/data \
  --baseline-contract results/base30b-chat-hf-UNIQUE/contract.json \
  --backend hf --memory-profile qwen3_moe_cpu \
  --run-id experiment-001 \
  --resume /path/to/repo/selfplay/runs/explorer-zero-experiment-001/ckpt
```

Checkpoint 封存双 adapter、AdamW 状态、任务池、训练游标和各 rank RNG；续训必须沿用同一 run、合同、profile 和 world size。微型恢复测试验证下一步参数一致，不证明真实环境轨迹逐位复现；旧 raw 协议 checkpoint 不能在新 chat 配置下直接续训。

训练入口 `run_train.sh` 只负责训练，不会自动评测。要在所有训练 rank 退出、GPU 释放后串行评测四个 outer 快照，使用组合工作流脚本；它先校验 run 路径和已有 baseline 合同/结果，再训练，训练正常退出后才启动评测：

```bash
bash selfplay/scripts/run_experiment.sh \
  --run-id experiment-001 \
  --run-dir /path/to/repo/selfplay/runs/explorer-zero-experiment-001 \
  --base /path/to/local/model \
  --alfworld-data /path/to/alfworld/data \
  --baseline-contract results/base30b-chat-hf-UNIQUE/contract.json \
  --baseline-results results/base30b-chat-hf-UNIQUE/contract.merged.json \
  --backend hf --memory-profile qwen3_moe_cpu
```

评测阶段逐个调用原有 `evaluate.py prepare`、`run_eval.sh` 和 `evaluate.py compare`，每个 outer 完成后才进入下一个。已有 baseline 不会重跑或写入。单独评测已完成的训练目录可运行 `python selfplay/evaluate_run.py --run-dir <run-dir> --baseline-results <base-merged.json> [--results-dir <run-dir>/evaluation]`；评测结果必须放在 run 目录内独立子目录。若目标目录已有完整且验证通过的四轮结果，会校验后跳过；若结果不完整，脚本保留现场并要求指定新的 `--results-dir`，不会删除或覆盖已有文件。

每个 outer 的合同、合并结果、对照结果和命令日志保存在其子目录；顶层 `summary.json`/`summary.csv` 汇总四轮与新 baseline 的对照。快照索引、封存、当前模板和合同身份在 GPU 评测前校验。两臂必须使用相同 `selfplay/config.yaml`、基座、数据、软件和评测源码；当前协议改变后，必须生成并运行新的 baseline，旧合同不能复用。

## 结果状态与代码入口

30B 训练使用同一 HF 基座负责采样与更新，AdamW 状态及保存的反向激活卸载到 CPU，训练生成微批为 1；保留 all-linear 双 LoRA。`run_experiment.sh` 现可串联该 profile 与四快照 HF 评测。旧交接包描述当时版本，当前设置以本 README 和最新交接为准。

本轮只完成本地代码和 CPU 验证，未启动服务器作业。旧反馈的 raw/vLLM 成功率不能代表新 chat/HF 的效果；新分数、八卡峰值内存与速度均待实测。CPU 卸载与 HF MoE 可能较慢，不承诺 vLLM 速度或固定完成时限。

```text
selfplay/train.py                 训练入口
selfplay/evaluate.py              固定协议评测与比较入口
selfplay/config.yaml              共享评测/训练基础配置
selfplay/src/training_contract.py 训练前基线合同校验
selfplay/src/protocol.py          共享提示、tokenization 与动作投影
selfplay/src/ez_solver.py         solver rollout
selfplay/src/ez_proposer.py       可验证任务生成
selfplay/tests/                   协议与合同回归测试
```
