# Explorer-Zero 新训练入口 v2

本包记录从头训练的可复核入口。用户已选择不抢救旧 checkpoint；本入口不迁移旧 schema、不尝试从旧目录续训。它复用经过审查的 GRPO、状态验证、全局组分片和双 LoRA 同步原语；旧实现文件保留作历史追溯，新 run 只使用独立的 schema-v3 入口。

## 固定协议边界

- 配置：[`selfplay/configs/ez_repro_v2.yaml`](../../selfplay/configs/ez_repro_v2.yaml)。新入口要求本地已有 base 模型目录、ALFWorld 数据目录和显式 backend；禁止 Hub 回退、下载、安装或环境变量静默改写协议。ambient solver/parser/eval 开关与配置冲突时拒绝启动。
- 正式默认：8 rank、全局 `G=16`、`TPS=1`、`GP=8`，`N=4`、每 outer 的 P/S 各 25 步；LoRA `r=32, alpha=32`、学习率 `1e-4`、clip `0.3`、KL `0.001`、`w_sub=w_div=0.3`。训练 context `40960` 对齐 Table 3；`max_new_tokens=1024` 与论文未披露的 decode 值是本地显式选择，不标成作者设置。
- 训练池固定为 train split。schema-v3 manifest 锁定模型目录逐文件内容、tokenizer/config/generation metadata、ALFWorld 配置与逻辑文件、pool 顺序和任务文件/可用 trajectory、源码字节、软件版本及实际 backend 参数。HF cache 中指向本机 regular file 的模型文件 symlink 按目标内容计入；checkpoint 内不允许 symlink。
- 不运行 valid_unseen 中途评测，不据最终测试挑 checkpoint。`--smoke` 是独立的一步 P+一步 S 检查，保持 `G=16/TPS=1`；smoke checkpoint 可校验完整性，但明确禁止 resume 和 evaluation。
- vLLM 生成 RNG 不提供 bitwise 恢复保证。checkpoint 会保存 rank-local Python/NumPy/Torch RNG、optimizer、task pool、两个 adapter 和下一个训练位置；manifest 明确记为 `stateful_resume_non_bitwise`。

## 命令

从仓库根目录启动本地 8 卡 smoke（只验证实际计算链路，不代表 GPU 性能或论文结果）：

```bash
SKR_PYTHON=/path/to/active-env/bin/python \
SKR_PROJECT_ROOT="$PWD" \
bash selfplay/scripts/run_ez_train_v2_8gpu.sh \
  --config selfplay/configs/ez_repro_v2.yaml \
  --base /local/models/qwen3_4b \
  --alfworld-data /local/data/alfworld \
  --backend vllm --run-id smoke-20261015 --smoke
```

Smoke 结束后，正式 fresh run 必须使用新的唯一 run id，且不带 `--smoke`：

```bash
SKR_PYTHON=/path/to/active-env/bin/python \
SKR_PROJECT_ROOT="$PWD" \
bash selfplay/scripts/run_ez_train_v2_8gpu.sh \
  --config selfplay/configs/ez_repro_v2.yaml \
  --base /local/models/qwen3_4b \
  --alfworld-data /local/data/alfworld \
  --backend vllm --run-id fresh-20261015-a
```

正式恢复需指向同一 run 的 `ckpt`，保留相同配置、base、数据、源码和软件身份；v3 validator 会拒绝任何身份或 world/config 不匹配：

```bash
SKR_PYTHON=/path/to/active-env/bin/python \
SKR_PROJECT_ROOT="$PWD" \
bash selfplay/scripts/run_ez_train_v2_8gpu.sh \
  --config selfplay/configs/ez_repro_v2.yaml \
  --base /local/models/qwen3_4b \
  --alfworld-data /local/data/alfworld \
  --backend vllm --run-id fresh-20261015-a \
  --resume /path/to/runs/explorer-zero-v2-fresh-20261015-a/ckpt
```

脚本使用当前已分配的 8 个 `CUDA_VISIBLE_DEVICES`，每个 rank 映射到一张可见卡，并用同一个 `SKR_PYTHON` 启动 torch distributed。脚本不会提交调度器作业，也不会自动创建环境。当前没有在 GPU 上运行或验证；目标机器的 GPU smoke、实际 adapter 更新和吞吐仍待执行。

## Checkpoint API

`selfplay/src/ez_train_v2_contract.py` 提供纯标准库函数：

```python
validate_checkpoint_v3(path, expected_contract=None, purpose="resume")
validate_checkpoint_v3(path, expected_contract={"base_inventory_sha256": digest}, purpose="evaluation")
```

函数显式区分 resume 与 evaluation，返回 `state`、`manifest`、`inventory` 和 `solver_path`。Evaluation 只验证完整 marker、源码/训练合同及 solver adapter，不反序列化 optimizer；旧 schema、未封存目录、smoke checkpoint、内容不匹配和互相冲突的 `next_position`/游标别名都会拒绝。

## 配对评测入口

评测工具由独立的 v2 eval 分支维护；下面只列出可复核流程，不表示已经运行或取得分数。先在 CPU 端锁定当前实际发现的 task pool 与 8 个 shard：

```bash
python selfplay/src/ez_eval_v2.py prepare-suite \
  --config selfplay/configs/base_config.yaml --split valid_unseen \
  --shards 8 --output OUT/suite.json
```

推荐用对照计划生成器一次准备配对的 base 与 solver arm。它只写计划和 `run.sh`，不会自动提交或执行；先审阅文件，再手动执行：

```bash
python selfplay/scripts/ez_eval_v2_controls.py \
  --suite-manifest OUT/suite.json \
  --config selfplay/configs/base_config.yaml \
  --base /local/models/qwen3_4b \
  --checkpoint /local/runs/explorer-zero-v2-RUN/ckpt \
  --out-dir OUT/paired --backend vllm

# 人工审核 OUT/paired/plan.json 与 OUT/paired/run.sh 后，才手动执行：
bash OUT/paired/run.sh
```

然后分别执行配对的 base 与 adapter arm。`valid_unseen` 每任务 4 个独立 sample；主报告为完整 suite 的 sample-level SR 均值及明确的 suite 间标准差，不是 Pass@4，也不把四次结果 OR 在一起。Eval 的 30-turn 上限和 40960 context 借用训练预算，`max_new_tokens=1024` 是本地选择；这些仍是 paper-approx，不是论文已完整披露的最终 evaluation 参数。

配对计划运行并严格合并后，输出为 `OUT/paired/base-merged.json`、`OUT/paired/adapter-merged.json` 和 `OUT/paired/paired-difference.json`。单臂调试时，launcher 会输出本次唯一 run 目录和实际 merged 文件路径；不要从过期目录手工拼接 `merged.json`。

```bash
bash selfplay/scripts/run_ez_eval_v2_8gpu.sh \
  --base /local/models/qwen3_4b --config selfplay/configs/base_config.yaml \
  --out-root OUT/base --backend vllm --base-only

bash selfplay/scripts/run_ez_eval_v2_8gpu.sh \
  --base /local/models/qwen3_4b --config selfplay/configs/base_config.yaml \
  --out-root OUT/solver --backend vllm \
  --checkpoint /local/runs/explorer-zero-v2-RUN/ckpt
```

`selfplay/scripts/ez_eval_v2_controls.py` 只生成待审查的 plan 和 shell 文件，不自动执行命令。最终 unseen 应在 dev protocol 锁定后运行一次；valid_unseen 已被此前开发诊断使用，不能再称为从未查看过的 test set。

## CPU 验收与限制

本次本地完整 CPU 回归为 **114 passed、2 skipped、9 warnings**。两个 skip 分别是未设置 `EZ_QWEN3_TOKENIZER` 的真实 tokenizer 探针，以及 Windows 当前账户无法创建 symlink 的 fixture；warning 是 CPU 环境禁用 CUDA autocast 和 tiny PEFT fixture 的缺省 base config 提示。训练专属回归位于 `selfplay/tests/test_ez_train_v2.py`。其中 Tiny Llama + PEFT 双 adapter 检查 optimizer 非空后的 schema-v3 保存、adapter 重载、active adapter 保持、optimizer/RNG 恢复、下一步权重精确一致和 checkpoint 完整性；另有配置构造、PG/KL 目标分解、环境身份、两 rank Gloo preflight 及恢复游标检查。没有 Qwen3 实际 tokenizer 或正式模型验证。

这些检查不证明 GPU OOM、安全余量、速度、训练收敛或 Explorer-Zero 分数复现。没有 GPU 实测结果，也没有执行本文件中的任何训练命令。旧 checkpoint 按本轮决定不抢救；新鲜训练和评测必须从 schema-v3 开始。
