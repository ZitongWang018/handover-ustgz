# Explorer-Zero runtime 与评测交接（2026-10-16）

## 本轮实现边界

本轮为训练与评测增加了运行时隔离和交接路径：训练侧在创建单卡 vLLM EngineCore 时清理 torchrun、Elastic 与 vLLM data-parallel 启动元数据，保留设备掩码等普通运行环境。这个改动针对训练进程中的 runtime 隔离；原先 EngineCore timeout 的根因尚未证实，不能据此宣称该问题已被解释或在目标服务器上解决。

每个完整 outer 结束时发布一个不可变、带完整性封印的 solver 快照。评测入口要求四个 outer 快照按序齐全，并在启动评测前核对训练合同、索引、快照游标、seal 和 inventory。评测在训练所有 rank 退出后单独启动，逐个快照运行原有 prepare、评测、compare 流程；不会与 live training 共用 GPU。它校验嵌入训练 manifest 的完整 baseline 合同及现存 baseline merged 结果，包括合同哈希、协议签名、base-only 身份、配置与评测源码/软件身份、base/tokenizer inventory、环境配置和完整 valid_seen 任务池，以及完整任务×种子轨迹和重算 summary。baseline 不重跑、不写入。

本轮没有访问服务器，没有启动或操作训练/评测作业，没有改变模型权重，也没有安装或下载依赖。目标服务器上的 runtime 行为与 GPU 资源释放尚未验收。

## 运行入口

README 中的组合命令如下。请把路径换成目标机上已经存在的模型、ALFWorld 数据、匹配 baseline 合同/结果和本次 run 目录：

```bash
bash selfplay/scripts/run_experiment.sh \
  --run-id experiment-001 \
  --run-dir /path/to/repo/selfplay/runs/explorer-zero-experiment-001 \
  --base /path/to/local/model \
  --alfworld-data /path/to/alfworld/data \
  --baseline-contract results/base/contract.json \
  --baseline-results results/base/contract.merged.json \
  --backend vllm
```

`run_train.sh` 只训练，不会自动评测。`evaluate_run.py` 只评测已完成训练目录，不会启动训练。只有 `run_experiment.sh` 会在预检 baseline 与 run 路径后先调用训练；训练正常退出并释放各 rank 后，再串行评测四个快照。单独调用 `evaluate_run.py` 时，结果写入 run 目录下的 evaluation 子目录；已有完整结果只校验后跳过，部分结果保留并要求使用新的结果目录。

## 尚待确认

- DeepSeek 报告中的 baseline 32% 尚无 raw episode/shard 文件供独立复核；当前只能将它记为外部报告值，不能作为本地已验证结果。
- 需要在目标 8 卡环境完成真实验收，确认训练结束后所有 rank 退出、GPU 可供评测使用，以及四个快照顺序评测的完整产物。
- 记录目标机实际 vLLM 版本和依赖版本，并与绑定的 baseline/runtime 合同核对。
- 完整 `selfplay/tests` 独立执行：**76 passed, 1 skipped, 1 warning（25.91s）**。唯一 skip 是 Windows 目录 symlink 测试因 `WinError 1314` 缺少创建 symlink 的权限；模拟 alias 的防护测试通过。warning 是微型 PEFT 测试找不到 base config 的既有提示。三个入口的 `--help` 均通过，Git Bash 对 `run_experiment.sh`、`run_train.sh`、`run_eval.sh` 的语法检查均通过。未进行 GPU 或 ALFWorld 实测。

## 服务器执行约束

任何服务器操作都必须遵守仓库 `AGENTS.md` 的入口与授权规则，只执行明确批准的任务。不得根据本交接文档自行改服务器代码、改模型权重、启动额外作业或推断已有报告已通过验证。
