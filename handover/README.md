# handover — 交接材料索引

本目录存放**交给外部模型 / 外部协作者独立审核**的运行产物快照。
原始运行快照记录事实且不追改；`gpt_feedback_*.md` 独立保存后续审查和修复说明。

| 目录 | 内容 | 审核对象 |
|---|---|---|
| `eval_run_13044670/`（含[后续反馈](eval_run_13044670/gpt_feedback_20261014.md)） | ALFWorld `valid_unseen` 基座评测（paper-approx 协议，134 任务 × 4 采样 = 536 episode）。含协议 manifest、536 行 episode 明细、逐任务/逐类型聚合、8 分片日志、完整 `merged.json`（tar 包内）；反馈文档独立记录后续审计与修复 | 评测协议一致性与数字合理性；后续修复反馈 |
| `skillrise_prompting_20261015/` | SkillRise ALFWorld prompting quick-run 的原始 JSON、selection、summary 与训练日志快照；历史事实保留 | 输入完整性、统计口径与证据边界；不作为 Explorer-Zero 结果 |
| `explorer_zero_repro_plan_20261015/` | 本地只读审计 JSON、论文/上游协议矩阵与分阶段复现计划；不启动作业、不执行训练 | Explorer-Zero 复现边界、当前证据与后续验收计划 |
| `explorer_zero_train_v2_20261015/`（含[实现反馈](explorer_zero_train_v2_20261015/gpt_feedback_v2.md)） | 从头训练的 schema-v3 入口、严格 checkpoint 合同、配对评测流程、便携 8 卡启动方式及 CPU 验收边界；不迁移旧 checkpoint、不运行 GPU 作业 | 新鲜训练合同、恢复资格与尚待目标 GPU 验证的事项 |

每个子目录的 `README.md` 里都有：
1. 运行事实（job id、节点、命令、代码 commit）
2. 协议 manifest 的逐项原值
3. 原始结果（不做二次加工）
4. 文件清单与"本目录不含什么"
5. 外部对照值 + 其**核实状态**
6. 待核实问题清单（问题，不是结论）
7. 时间线上的诚实记录（含本次运行前后发生过的改动）
| `skillrise_prompting_20261015/STATUS_20261015_night.md` | **状态交接**（非结果报告）：把「SkillRise prompting 评测」与「Explorer-Zero 复现」**两条链路的入口/口径/指标钉死**（上一轮把它们混讲导致混乱）；记录 **同 seed 两次跑结果不同**的实测证据与由此得到的纪律；记录 `sbatch --export` 里逗号截断的提交错误；列出未完成项（140 段在跑、EZ 链路未跑过、协议文档被改写、集群与 GitHub 代码不同步） | **两个入口是否被正确区分**、同 seed 不复现的推断是否成立 |
