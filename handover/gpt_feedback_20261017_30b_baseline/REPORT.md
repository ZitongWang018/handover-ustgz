# 30B base-only baseline 评测交接（2026-10-17）

## 本次范围

本轮范围仅为 30B base-only baseline 评测，不启动训练。评测 factory 对 base-only 使用 `enable_lora=False`；checkpoint 评测才启用 LoRA（`enable_lora=True`）。8 卡仍按一张卡一个 rank 分片执行。

沿用当前固定评测协议：完整 `valid_seen` 池、5 个 seed（0–4）、最多 30 turns、32,768-token context、temperature 0.7、top_p 0.95、max_new_tokens 1024、batch size 32、GPU memory utilization 0.85。不得降低上下文或采样参数，不切换 HF backend，也不替换模型权重。源码哈希已变化，之前失败的 30B baseline 合同必须原样保留；新评测使用唯一的新目录 `results/base30b-loraoff-<唯一ID>`，不可覆盖旧合同或结果。

评测脚本使用同一个 `SKR_PYTHON`：prepare 与 `run_eval.sh` 都从该解释器运行。以下命令中的模型、数据、仓库和解释器路径需由获批执行者替换；将 `UNIQUE_ID` 替换为本次唯一标识，确保输出目录此前不存在：

```bash
export SKR_PYTHON=/path/to/approved/python
BASELINE_DIR=/path/to/repo/results/base30b-loraoff-UNIQUE_ID

"$SKR_PYTHON" /path/to/repo/selfplay/evaluate.py prepare \
  --config /path/to/repo/selfplay/config.yaml \
  --alfworld-data /path/to/alfworld/data \
  --base /path/to/local/30b/model \
  --base-only \
  --results-dir "$BASELINE_DIR" \
  --output "$BASELINE_DIR/contract.json"
bash /path/to/repo/selfplay/scripts/run_eval.sh "$BASELINE_DIR/contract.json"
```

`prepare` 没有 `--backend` 参数；backend 从现有配置的 `eval.backend` 读取，本命令沿用 `selfplay/config.yaml` 中的 `vllm` 设置。

运行前执行代理必须重新阅读仓库 `AGENTS.md`，核验本次真实授权根目录与全部输入/输出路径；不得把以往 `thinktool` 路径或命令视为本轮服务器授权。这个任务不调用 `run_experiment.sh`、`run_train.sh` 或训练入口。服务器执行只能按明确批准范围进行；AGENTS 约束下的问题应保存证据并交接，不在服务器自行改代码。

## 事实、限制与更正

- 30B base-only 评测尚未做 GPU 实测；不能保证 OOM 已解决，也不能报告评测已成功。
- vLLM 0.8.5.post1 对该模型的 MoE LoRA 训练仍不支持。本任务不训练；不以切换 backend 或换权重绕过。
- 旧交接中该模型的 KV heads 应为 4，而非 32。Instruct 2507 仅用于 non-thinking 设定。
- 两模型单卡训练不属于本轮范围。此前 4B 结果是历史结果，不能当作新源码哈希下的配对 baseline。
- 附带语法整理：前次 `d615200` 测试后，`selfplay/src/training_runtime.py` 与 `selfplay/tests/test_training_runtime.py` 文件末尾各残留一行字面 `\n`，导致 SyntaxError；已仅删除这两行，训练逻辑和参数未改。这不是 30B 训练适配。此前 `d615200` 对应最终工作树未在该整理后验证；本轮对修正后的最终源码已完成完整重测。
- 本轮 CPU 验证：`pytest selfplay/tests -q --tb=short -rs` **81 passed, 1 skipped, 1 warning（39.85s）**；34 个 Python 文件 AST 检查通过，`evaluate.py prepare --help` 参数核验通过。唯一 skip 是 Windows 真实目录 symlink 测试因 `WinError 1314` 权限不足；模拟 alias 的防护覆盖通过。warning 是既有 tiny PEFT 测试找不到 base config 的提示。未进行 GPU 或 ALFWorld 实测。
