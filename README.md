# 在 SkillRise ReAct 设定下检验 Explorer-Zero

服务器执行代理（包括 DeepSeek）必须先读 [AGENTS.md](AGENTS.md)：仅在已确认的 `ztwang` 目录内执行明确批准的任务，默认不得修改实现；问题通过独立 GitHub 反馈分支完整交接。

本仓库复现 EZ 的交替自博弈训练，其中 solver 训练与评估统一采用 SkillRise 风格 ReAct 协议。训练交替更新 proposer 和 solver adapter，基座模型保持冻结；评估只加载训练得到的 solver adapter。基线是未训练的基座模型。比较时两臂使用同一 valid_seen 全量任务池、提示、动作投影、解码参数和五个评估种子。

这里采用 SkillRise 最近一次 quick prompting 实测的 ReAct 设置，不声称复现 SkillRise 全部系统，也不声称达到 Explorer-Zero 原论文的绝对分数。任何提示、tokenizer、数据池或解码协议变化，都需要重新评估基线；旧 quick 数值不能直接沿用。

## 共享 solver 设置

- 评估使用完整 valid_seen 任务池和种子 0、1、2、3、4；每局最多 30 步。
- 每轮用原始纯文本提示重新构造输入，不套 chat template；tokenizer 按 `add_special_tokens=True` 编码。
- 提示展示当前可用动作，排除 `help`；历史只保留最近 15 组已执行动作和环境观测。
- 模型输出使用 `<action>...</action>` 投影，生成在 `</action>` 处停止并保留闭合标签。
- 训练和评估统一使用 temperature 0.7、top_p 0.95、max_new_tokens 1024、上下文上限 32768。
- solver 训练在 ALFWorld train split 和 proposer 生成的任务上使用同一提示、动作与 tokenization 规则。每轮实际提示和生成 token 分开保存，按整条轨迹的 response token 数加权。

训练配置采用 4 个 outer loops，每轮 proposer 和 solver 各 25 步；GRPO group size 16、每步 1 个任务、solve-rate 采样 8 次。LoRA rank 32，学习率 1e-4，clip epsilon 0.3，KL 系数 0.001，子任务和多样性权重各 0.3。上下文上限 32768，低于 EZ 原设置的 40960，因此这是明确的协议差异。

Proposer 内部使用 JSON、`finish` 和 `sub_finish` 生成可验证任务；这些控制标记不属于 solver 的评估动作协议。

## 运行

训练需要本地模型目录、ALFWorld 数据目录和 8 张 GPU。配置默认值见 `selfplay/config.yaml`；训练脚本不下载模型或数据。

```bash
bash selfplay/scripts/run_train.sh \
  --base /path/to/local/model \
  --alfworld-data /path/to/alfworld/data \
  --backend vllm \
  --run-id experiment-001
```

继续训练时提供已封存 checkpoint：

```bash
bash selfplay/scripts/run_train.sh \
  --base /path/to/local/model \
  --alfworld-data /path/to/alfworld/data \
  --backend vllm \
  --run-id experiment-001 \
  --resume /path/to/repo/selfplay/runs/explorer-zero-experiment-001/ckpt
```

`--resume` 指向该 run 的封存 `ckpt` 目录；`--run-id` 必须与原训练 run 相同。

先为未训练基座准备固定评估合同，再运行完整评估：

```bash
python selfplay/evaluate.py prepare \
  --config selfplay/config.yaml \
  --alfworld-data /path/to/alfworld/data \
  --base /path/to/local/model \
  --base-only \
  --results-dir results/base \
  --output results/base/contract.json
bash selfplay/scripts/run_eval.sh results/base/contract.json
```

`--alfworld-data` 是 ALFWorld 数据根目录，必须在 `prepare` 时显式提供；合同会固定该路径，评估 rank 启动时按合同恢复。ALFWorld YAML 路径从配置中的 `env.config_path` 解析。

对训练 checkpoint 使用相同配置准备另一份合同，将 `--base-only` 换成 `--checkpoint /path/to/checkpoint`，并使用单独的 `--results-dir` 和 `--output`。随后比较两份合并结果：

```bash
python selfplay/evaluate.py compare \
  --baseline results/base/contract.merged.json \
  --trained results/trained/contract.merged.json \
  --output results/comparison.json
```

评估合同固定任务池、配置、基座文件、协议源码和软件版本。checkpoint 必须由当前协议训练；协议源码变化后需重新训练或重新评估，不能把旧基线当作匹配对照。

## 训练与评估记录

训练 checkpoint 保存封存配置、proposer task pool、优化器状态、随机数状态、双 adapter 和续训游标。学习率保持常量，没有 scheduler 状态。vLLM 的重启不承诺逐位复现。

本仓库当前没有 GPU 训练或完整五种子评估结果，因此不报告训练提升、绝对分数或吞吐结论。代码路径、设置和命令提供了可复核实验协议；结果需在目标环境实际运行后再填写。

## 目录

```text
selfplay/train.py                 训练入口
selfplay/evaluate.py              固定协议的评估与比较入口
selfplay/config.yaml              训练和评估配置
selfplay/src/protocol.py          共享提示、tokenization 与动作投影
selfplay/src/ez_solver.py         生成任务上的 solver rollout
selfplay/src/ez_proposer.py       可验证任务生成
selfplay/tests/                   训练、协议和评估测试
selfplay/vendor/skillrise/        SkillRise 许可与来源说明
```
