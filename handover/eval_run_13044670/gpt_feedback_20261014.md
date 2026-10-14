# 13044670 运行后的审查与修复反馈

本文是对 2026-10-14 评测运行 `13044670` 的后续审查记录，不是该次运行的原始产物，也不回写或替代原目录中的 README、JSON、TSV、日志或 tar。原始运行使用代码提交 `643e2ae`；后续修复提交为 `7deef0acc9ebff574032eab1908211b801b4dd53`。SOL 制定逻辑边界并 review，LUNA 按边界实现。本次补充用于区分“运行时事实”和“运行后修复”，不把新协议记账追溯套到旧结果上。

## 最重要的审计发现

运行 `643e2ae` 的 ours prompt 实际没有给模型展示 admissible action 菜单：初始 solver prompt 未传入动作表，后续反馈菜单也处于关闭状态。该次 manifest 却固定记录 `resolver.menu_enabled=True`。因此，原 manifest 对菜单状态的记录不准确；该运行应按实际无菜单输入理解。原始文件保留这一事实，不修改其 manifest，也不把旧结果重标为新协议。

## 版本边界与修复内容

版本链如下：

- `643e2ae` 是本次运行实际使用的代码版本，已经包含前一轮修复，例如 placement 动作解析、解析器变量错误、真实环境反馈、样本网格对应，以及严格分片合并。
- `6904dfc` 只修复测试文件中的字面 `\n` 问题，没有改变评测路径。
- `7deef0acc9ebff574032eab1908211b801b4dd53` 是本次运行后新增的菜单状态记账与对照计划修复，不属于 `13044670` 的运行代码。

后续代码改动包括：

1. 菜单开关提供 `auto/on/off`。`auto` 沿用 `EZ_SOLVER_ADMISSIBLE=1` 的旧环境变量语义，显式 CLI 选项优先于环境变量。ours 的同一个 resolved 状态传给初始 prompt 和每轮执行反馈，避免 manifest、初始输入和后续输入各说各话；rich profile 始终使用其编号菜单，显式关闭会被拒绝。
2. Manifest 记录请求值、实际生效值、解析来源、旧环境变量值与菜单范围；prompt hash 纳入实际 prompt style、菜单状态和对应模板。ours trace 增加初始任务/观测、每轮 admissible actions 及当轮是否实际展示菜单等信息，供逐步核对。
3. full 与 checkpoint 评测 sbatch 脚本转发菜单选项，并保留已有参数路径。新增离线模型身份摘要工具，只读取受限的本地元数据；默认不读取权重张量内容，状态明确为 `weights_unverified`。
4. 新增 CPU-only 对照计划生成器，支持 base-only 与可选 solver adapter 配对。A/B 仅改变菜单开关，B/C 仅改变步数预算，D 是 public-rich 多因素 scaffold 对照。计划生成器只写 `plan.json` 和供人工检查/执行的 `run.sh`，不会自行执行命令或提交作业。

| arm | 菜单 | 步数 | 每任务样本数 | 协议 |
|---|---:|---:|---:|---|
| A `paper_no_menu` | 关闭 | 30 | 4 | `paper-approx` |
| B `json_menu` | 开启 | 30 | 4 | `custom` |
| C `json_menu_50` | 开启 | 50 | 4 | `custom` |
| D `public_rich` | 开启（编号动作菜单） | 50 | 1 | `public-rich` |

有关实现位置和完整协议说明见 [评测入口](../../selfplay/src/eval_ez.py)、[prompt 与菜单处理](../../selfplay/src/ez_prompts.py)、[模型身份工具](../../selfplay/src/ez_model_identity.py)、[对照计划生成器](../../selfplay/scripts/ez_eval_controls.py)、[full 评测脚本](../../selfplay/scripts/ez_eval_full_8gpu.sbatch)、[checkpoint 评测脚本](../../selfplay/scripts/ez_eval_ckpt.sbatch)、[评测审计说明](../../docs/EZ_EVAL_AUDIT.md) 和 [13044670 专项审计](../../docs/EZ_EVAL_RUN_13044670_AUDIT.md)。

## 模型身份与历史数字的边界

官方 [Qwen3-4B 模型 README](https://huggingface.co/Qwen/Qwen3-4B/raw/main/README.md) 的 `base_model` 字段列出 `Qwen/Qwen3-4B-Base`，这是模型来源/祖先声明。Qwen3-4B 本身也可以包含该字段，所以它不能证明本地目录中的当前权重就是 Qwen3-4B-Base。此次没有对集群模型目录做身份核验，没有更换权重，也没有启动或修改集群作业；本地离线身份工具同样不会把 README 声明当成权重证明。

原始运行报告的 episode-weighted SR 为 **5.97%**，它只描述 `643e2ae` 下这次实际运行。由于当时 manifest 错记菜单开启，不能把后来修正的 manifest 或新菜单语义套用到该数值上，也不能据此宣称修复带来了分数变化。原运行 README 还写“manifest 未记录软件版本”；检查原 JSON 可见其中实际包含软件版本信息。该 README 的这句话有误，但本反馈不改原始快照。

## 验证范围

修复版本通过了 79 项完整 CPU 回归，以及暂存内容 AST 一致性和两个 shell 脚本的 `bash -n` 检查。这些结果验证代码路径与脚本语法，不代表 GPU/集群重评，也不提供性能或成功率提升证据。本次未运行模型、未读取模型权重、未启动或取消任何集群作业。

## 从 `selfplay/` 目录复查

下列命令只展示如何调用工具；模型及数据路径需由使用者按本地环境设置。基座示例显式关闭菜单，以免受环境变量影响；如需沿用旧环境变量行为可用 `--action-menu auto`，也可显式指定 `on` 或 `off`。

```bash
# 基座 full-eval 示例；--n 0 表示使用完整 valid_unseen split。
python src/eval_ez.py --protocol paper-approx \
  --base "$ZTWANG_MODEL_PATH" --base-only --backend vllm --n 0 \
  --samples 4 --seed 42 --action-menu off \
  --out eval_out/review-paper-approx-base.json

# 只生成对照计划和人工可检查的 run.sh；不会运行评测或提交作业。
python scripts/ez_eval_controls.py --base "$ZTWANG_MODEL_PATH" \
  --alfworld-data "$ALFWORLD_DATA" --out-dir eval_plans/controls-s42 \
  --seed 42 --backend vllm

# 可选：生成包含 base/solver 配对 arms 的计划。
python scripts/ez_eval_controls.py --base "$ZTWANG_MODEL_PATH" \
  --ckpt "$EZ_CKPT" --alfworld-data "$ALFWORLD_DATA" \
  --out-dir eval_plans/paired-controls-s42 --seed 42 --backend vllm

# 只读本地有限模型元数据；默认不扫描权重内容。
python scripts/ez_model_identity.py "$ZTWANG_MODEL_PATH"
```

计划输出目录须为新路径。生成的 `run.sh` 是供人工审查的文本，不会自动执行；请勿把计划生成等同于完成评测。
