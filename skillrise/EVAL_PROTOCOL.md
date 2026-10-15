# SkillRise ALFWorld zero-shot / ReAct：历史快速评测记录

> **状态：历史 custom quick diagnostic，非论文复现定稿，也非当前推荐入口。** 本文保留 2026-10-15 已归档测量值和当时入口参数，修正此前对种子、Avg、128 槽和模型身份作出的过度推断。可复核统计见[Explorer-Zero 复现审计计划](../handover/explorer_zero_repro_plan_20261015/README.md)及[`audit_current_evidence.json`](../handover/explorer_zero_repro_plan_20261015/audit_current_evidence.json)。

## 1. 这批结果实际覆盖什么

归档的两个 JSON 各含 5 个 seed × 128 条 episode，共 640 条；selection 文件为 `eval_in_distribution`（valid-seen）128 槽。逐 seed 对比 game-file 多重集后，两种 style 都是完整网格。128 槽来自 SkillRise 上游验证 worker 的 seeded shuffled-cycle 选择；该清单有 91 个不同游戏、31 个重复游戏，最大重复 3 次。它不是 128 个不同任务，也不能据此推断 Explorer-Zero 论文使用同一清单或把 128 定义为其验证批次。

归档行记录的历史参数如下，保留用于解释既有结果：

| 项 | 归档值 | 边界 |
|---|---|---|
| split / 槽 | `eval_in_distribution` / 固定 128-slot selection | SkillRise valid-seen quick diagnostic；不是 EZ 的 valid-unseen 全量评测 |
| seed | 0、1、2、3、4 | raw rows 可核对完整；seed 本身不证明底层 vLLM 每请求 RNG 行为 |
| step cap | 30 | 这是该 quick run 参数，不是从 EZ 论文正文恢复出的完整推理实现 |
| temperature / top_p | 0.7 / 0.95 | 归档参数；论文没有披露可唯一确定这些值的完整采样协议 |
| max new tokens | 1024 | 归档参数；不是 EZ 论文配置。旧 EZ `max_new_tokens=96` 另列为待诊断高风险补全值 |
| engine / parser | 自研 vLLM quick evaluator；上游投影解析器 | 逐回合 prompt/token IDs/执行动作未随这些 raw 行保存，不能证明实际完整轨迹与作者代码相同 |
| 模型记录 | raw summary 标注 `qwen3_4b_inst_2507` | 这是本次 SkillRise quick run 的记录；不据此替换 EZ 指定权重，也不证明其与论文 `Qwen3-4B` 身份相同 |

当时记录的入口仍可在原始交接 README 中追溯：

```bash
sbatch --export=ALL,EZ_STYLE=react skillrise/eval_quick_both_1gpu.sbatch
sbatch --export=ALL,EZ_STYLE=zero_shot skillrise/eval_quick_both_1gpu.sbatch
```

这两条是历史入口参数，不表示该入口现在就是论文精确实现或 Explorer-Zero 正式评测命令。

## 2. 可由 raw JSON 重算的结果

`selfplay/scripts/ez_repro_audit.py` 只用 Python 标准库读取 handover JSON/JSONL，核对行数、seed、游戏多重集、won 类型与 summary，计算以下指标；它不载入模型/环境，也不运行命令。

| Style | 640 条 episode 加权 SR | 5 个 seed 的 episode 加权 SR mean ± sample SD (ddof=1) | 5 个 seed 的六类 macro mean ± sample SD (ddof=1) | wins |
|---|---:|---:|---:|---:|
| ReAct | 30.46875% (195/640) | 30.46875 ± 2.7621 pp | 30.7570 ± 2.5676 pp | 195 |
| zero-shot | 19.53125% (125/640) | 19.53125 ± 3.5801 pp | 20.2696 ± 3.8791 pp | 125 |

`mean ± SD` 在旧 `summary_128.txt` 中指六类 task-macro：ReAct 30.8 ± 2.6，zero-shot 20.3 ± 3.9（n=5，按审计得到的 sample SD/ddof=1）。SkillRise 论文中的 “Avg” 是否等于六类 macro，现有证据不能确认；此前把二者认定为同一口径的说法已撤回。episode-weighted 与六类 macro 是不同口径，不能混叫 Avg。每个 seed 中六类都存在；审计工具对缺类 fixture 会把六类 macro 记为 null，而不会对剩余类别重新平均。

两个 style 各有 155 个 `(seed, game)` 重复组；其中 ReAct 26 组、zero-shot 19 组出现组内胜负不一致。这个计数只是样本依赖诊断，不能归因于 seed 或重复本身的因果作用。

早期无显式 seed 的 ReAct quick run 曾记录 36.6 和 37.0，后续五 seed 的 macro 记录为 30.8 ± 2.6。两批观测的差异本身不能证明“未传 seed 等价于升温”、存在系统性偏高、或由单一因素造成；没有 matched repeated experiment 可以支持这种结论。历史记录数字保留，不据其差值外推原因。

## 3. 协议差异与复现边界

- Explorer-Zero 论文 Table 1 的 ALFWorld 主结果在 `valid_unseen`，报告四个 sample-level evaluation 的均值/标准差；这批 SkillRise quick raw 是 `valid_seen` 上 128 槽、每槽 5 个 seed 的评测，两者不是同一个实验。
- 这里的 ReAct 式上游 prompt 与本项目自拟 zero-shot prompt，其模板/解析/动作菜单不能单凭本次 raw 证明已与上游逐字一致；作者没有定义本项目的自拟 zero-shot 变体。zero-shot 结果只能按该归档设置称为 custom quick diagnostic。
- selection 中的重复是可核实的；“这是论文为 prompting 选的 128 个 held-out tasks”或“128 必然是验证 batch”没有证据支持。
- 论文 Qwen3-4B 与 `Qwen3-4B-Base` 是不同命名。文件夹名、README 的 `base_model` 祖先字段均不能证明本地权重就是 Base。当前 EZ 计划保留用户已指定的本地 `qwen3_4b` 路径，先核验权重元数据/文件清单；不因 SkillRise 指令版 quick 结果自动换权重。
- raw episode 没有 slot id、prompt/template hash、tokenizer/checkpoint hash、逐轮输出/观测/执行动作轨迹。它能证明统计网格和存储行相符，不能证明 trajectory、parser、模型权重或复现实现完全正确。

完整论文对齐边界、当前 training evidence 分类和后续验收顺序以[新复现计划](../handover/explorer_zero_repro_plan_20261015/README.md)为准。不要用本 quick diagnostic 的分数选择 Explorer-Zero 基座、推断训练质量或声称 EZ 论文复现成功。
