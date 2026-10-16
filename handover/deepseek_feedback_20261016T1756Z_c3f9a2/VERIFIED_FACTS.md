# 计划更新：模型与集群核查结果（2026-10-17 02:15）

> 这是对 `PLAN_30B_A3B.md` §1/§7「待核验」项的**补充事实**（追加，不覆盖原文件）。

## 1. 模型已核实

| 项 | 值 |
|---|---|
| **选定路径** | `/hpc2hdd/home/xyu608/thinktool/models/qwen3-30b-a3b-inst-2507`（**完整，57G，26 个文件**） |
| `config.json` 的 `model_type` | **`qwen3_moe`** ✓ |
| 关键结构 | hidden 2048 / 48 层 / **128 专家** / moe_intermediate 768 / max_position **262144**（256K） |
| tokenizer | tokenizer.json + tokenizer_config.json + vocab.json + merges.txt 齐全 ✓ |
| 分片 | `model-00001..16-of-00016.safetensors`（16 片） |

**注意**：同名短目录 `/hpc2hdd/home/xyu608/thinktool/models/30b-a3b-inst-2507`
**缺 `model-00016-of-00016.safetensors`**（只有 25 个文件），**不要用**。
两目录都是真实目录（非符号链接），共占约 113G。

## 2. 已完成的落实动作

1. `d615200` 代码已同步到集群工作树（`evaluate_run.py` / `run_experiment.sh` /
   `training_runtime.py` 均到位）。集群 git HEAD 未动（仍 `0758f1f`，未做 git 级更新）。
2. **30B 基线合同已生成并验证**：`~/ztwang/results/base30b/contract.json`（191KB）
   - `contract_sha256 = 7f1772872415a0a933b1baf9fe06c63961b779b8895d18dcfa49d81f7a84a7db`
   - `arm_identity.base_inventory_sha256 = b586ba22fd73030115db11a5f21bee54344607e4e7abdd3897f845c4fb48425b`
   - `expected_shards = 8`；任务池在 `suite.tasks`；JSON 完整可解析
3. **30B 基线评测已提交**：job **13133474**（`ez-base30b`，8 卡 `i64m1tga800u8ka`，
   6h 上限；提交时分区 1 mix + 2 alloc，**排队等整节点**）。
   日志：`~/ztwang/logs/ez-base30b-13133474.{out,err}`

## 3. 依旧待办

- 评测的 `gpu_memory_utilization: 0.85`（61G 权重 + KV）**未实测验证**，
  若 OOM 属预期内风险（计划 R1），届时把日志交回。
- 训练仍等 GPT 处理 R1~R6 后开始（不先于评测跑）。
