# Explorer-Zero 复现计划与现有证据审计

**状态：计划和本地 CPU/静态审计，不是 runtime 修复、训练成功结论或 GPU 复现结果。** 本轮不改训练、奖励、rollout 执行路径，不访问服务器、不启动/取消作业、不换模型。旧 handover/raw 与 summary 保持原样；本目录只提供新的可复核审计和计划。

## 1. 目标与科学边界

目标是按 Explorer-Zero 论文复现 ALFWorld 的 zero-shot 基线和训练后的 solver 结果。论文中的主结果口径是 ALFWorld `valid_unseen`，Qwen3-4B，Figure 14 所示 JSON `<tool_call>` solver 格式，Table 1 的 `n=0` baseline（无 human demonstrations），以及每个 task prompt 的四个 sample-level evaluation。论文没有完整披露最终 eval 的回合上限；本计划暂沿用 Table 3 训练 solver 的 30-turn 上限作为 **paper-approx 选择**，后续需核验，不能写成论文已明确的最终评测设置。要报告的是四份完整任务集的成功率均值与离散度；**不是**“四次中至少成功一次”的 Pass@4，也不是六类 macro 替代的论文主指标。最终使用 benchmark 官方 `won` 判据。

有效的全量 task 集必须从实际目标环境中发现、固定、哈希并验证完整 grid。预期 `valid_unseen` 有 134 个任务只是待核验预期：运行前要记录真实文件清单与 hash，结果须完整覆盖每个 task × sample。若实际数量不同，先查原因，不把 134 写成硬编码事实。

论文附录 G 的 Figure 14（PDF 第 24 页）确实展示空 thinking tags 与 JSON `<tool_call>` 结构，但图注为 “Details omitted for space”，user 部分带 `# Example Omitted`。遗漏示例的具体内容和 `n=0` 是否隐含任何未展示 prompt 细节均未知。因此可实现 `paper-approx`，不能声称 paper-exact；不能擅自补一条 one-shot example 并说与 n=0 等价。temperature、top-p、采样 RNG、具体 stop/decode 细节等未由论文给到足以唯一确定运行行为的规范。

论文写 `Qwen3-4B`。仓库中用户已有的 `qwen3_4b` 是本轮待核验的本地目标路径；它的目录别名、README `base_model` 祖先字段或某次分数都不能证明权重身份。`Qwen3-4B-Base` 是另一个模型名，不能把二者混为一谈。本计划不替换路径、不下载权重；先离线核对配置、tokenizer、权重文件清单与必要哈希，再记录 `weights_unverified` 或更高证据等级。

SkillRise 的 128-slot valid-seen ReAct/自拟 zero-shot 评测不是 EZ 的基座评测，也不是训练结果。它与 Explorer-Zero 的模型身份、split、prompt、菜单和指标不同，不能拿其 `30.8` macro 或 `2507` 指令版观测给 EZ 选型或归因。

## 2. 论文明确写出的设置

来源是本地核验的论文抽取文本（原 PDF SHA256：`05e75e1cea1bca2c408764297595099964d97da08ee49924f9b38004a221fad1`）。页码按论文 PDF 页码，不按抽取文本行号。完整的分类见同目录 [`protocol_matrix.json`](protocol_matrix.json)。

| 位置 | 可直接支持的事实 | 不应越界推断 |
|---|---|---|
| §4.1，第 6 页 | 通常报告 ALFWorld unseen validation 与 benchmark success；四个 sampled evaluations per task；EZ 自生成训练任务不来自人工编写数据 | 论文没有公开本地运行的确切 pool 文件清单/哈希、split 中每个文件 ID 或 sampler RNG |
| Table 1，第 7 页 | ALFWorld `Qwen3-4B` baseline n=0：17.14 ± 5.7；Ours：85.07 ± 1.6；表注称每个 task prompt 四个 sample 的均值/标准差 | ± 的 ddof/具体聚合实现未充分说明；不能把表格数值反推成唯一 decoding 参数 |
| Table 2，第 7 页 | valid-seen ablation：one-shot proposer 59.29 ± 9.4；interactive proposer 65.71 ± 2.3；`+ subtask` 84.29 ± 1.5 | 这是 proposer/solver 训练消融，不是 Table 1 zero-shot/ReAct 对照 |
| §3.1–3.2，第 4–6 页 | 每 proposer group 共用同一初始状态；任务需经规则 valid filter 才进入 solver 训练池；solver 的 final state 用 Eq.5 比较，subgoal 用 Eq.6 检查 visited states；无子任务时 subgoal reward 为 0；Eq.7 成功终态 reward=1，否则 reward 为完成子目标比例 | 论文没给 ALFWorld 完整可执行 verifier 规格；同类实体的 type/index 忽略规则不能替代实体关系绑定检查 |
| Eq.1–4，第 5 页 | valid proposer reward 为 `R_p = R_difficulty + w_sub K + w_div |A|`，各项非负；invalid proposal reward 为 0 | invalid proposal 的 0 reward 仍属于 proposer 的 GRPO 组；不是先把 invalid 从 proposer 组 advantage 中屏蔽掉 |
| Appendix D / Table 3，第 15 页 | 4 outer loops；每 loop proposer/solver 25/25 steps；proposer exploration budget 30 turns；max solver turns 30；context 40,960；GRPO G=16；每 proposed task 求 solve-rate 的 G_P=8；rollouts/step=16；batch=16；LoRA r=32, α=32；AdamW lr=1e-4；clip 0.3；KL β=0.001；w_sub=0.3；w_div=0.3；8×H100 | Table 3 的 30-turn 是训练上限，不足以证明最终 eval 也使用 30 turns。本计划沿用它是待核验的 paper-approx 选择。w_sub 是 **proposer** 子任务奖励权重；solver Eq.7 的子任务比例不再乘 0.3。temperature、response 上限、batch 的精确张量/累积语义、KL 实现细节等仍需区分论文明确值与实现选择 |
| Appendix G / Figure 14，第 24 页 | ALFWorld solver 的 tool-call JSON scaffold 和 `# Example Omitted` 可见 | 因示例被省略，不得宣称恢复了作者完整 prompt；本方案 prompt 属 paper-approx |

§4.1 的 mean/std 跨四个 sample。计划统一显式记录两种 suite-level SD（ddof=0、ddof=1），并另报每次完整 suite SR，避免把一个未披露的 ddof 假装成事实。论文原表的 ± 仍按原样引用，不重算或改写。

## 3. 现有输入审计结果

运行只读工具：

```powershell
python selfplay/scripts/ez_repro_audit.py --repo-root .
python selfplay/scripts/ez_repro_audit.py --repo-root . --output handover/explorer_zero_repro_plan_20261015/audit_current_evidence.json
```

脚本只读三份 SkillRise raw JSON 与四份训练 evidence JSONL，使用 Python 标准库，不 import torch、transformers、ALFWorld/TextWorld，不调用 shell/Git/网络，不启动任何任务。它核对每个 style 的 seeds 0–4、每 seed 128 rows、game-file 多重集与 selection 一致、won 类型为 boolean、task type 与 game path/selection 一致、summary 能由 rows 重算。审计报告写相对逻辑文件名和 SHA256，不把机器绝对路径带入输出；`--output` 使用原子替换且拒绝写入/覆盖输入目录。

2026-10-15 归档内容：

| Style | episodes | wins | episode-weighted SR | seed-weighted mean ± sample SD (ddof=1) | seed 6-type macro mean ± sample SD (ddof=1) | 同 seed/game 混合胜负重复组 |
|---|---:|---:|---:|---:|---:|---:|
| ReAct | 640 | 195 | 30.46875% | 30.46875 ± 2.7621 pp | 30.7570 ± 2.5676 pp | 26 / 155 |
| zero-shot | 640 | 125 | 19.53125% | 19.53125 ± 3.5801 pp | 20.2696 ± 3.8791 pp | 19 / 155 |

两种 style 共享同一份 selection，其多重集有 91 个 unique game、31 个重复 game、最大 multiplicity 3；每个 style 每个 seed 都逐多重集吻合。旧 `summary_128.txt` 的 ReAct 30.8 ± 2.6 与 zero-shot 20.3 ± 3.9 是六类 macro 的 n=5 汇总。混合结果组仅为诊断，不能据此说 seed 造成胜负变化。

归档了四个 `ez_training_evidence/*.jsonl`：三份有 `[dry-run]` 配置行并记有单 outer `[done]`；第四份 1013-0320 记录了 TPS 自动上调警告和三条 reward mean=0 的 solver 行，但这份快照没有 `[done]` marker。审计只按行号分类这些证据，不推断哪份就是用户所说的完整正式 run，也不由 `[done]` 推断多个 outer 完成。

raw 评测不含 row slot id、实际完整 prompt/token IDs、逐回合 observation/executed action/verifier trace、prompt hash、tokenizer hash、base/adapter checkpoint hash、评测源码 commit 与 raw 文件到 selection manifest 的绑定哈希。故本审计能确认统计输入完整性，**不能证明评测 trajectory、解析、模型身份或执行行为正确**。先前未 seed 36.6/37.0 与 seeded 30.8±2.6 的差值也不能证明“未 seed 等价升温”或系统性偏高。

### 训练 checkpoint 候选的历史线索（未访问远程）

本地 Git 历史中，`c3a83e026cfea8f41505fc4ff47dbd249ed9ae46` 的提交说明记载 job `13026569` 在 outer 0 的 50 个 P/S steps 后，mid-eval 的 `top_k=0` 触发 vLLM 错误，并把当时 1.4 小时称作“报废”。但该提交父版本 `c3a83e0^` 的 `ez_train.py` 已先调用 `ez_save(...)`，然后才进入 outer-end mid-eval；当前代码也仍先保存再评测。因此 **eval 崩溃本身不能证明训练 checkpoint 全部丢失**。历史 launcher 默认输出根目录为 `$HOME/jhspoolers/ez_runs`；`ckpt-outer0-S24` 是优先核验的候选 checkpoint 名。

后续操作者可在已授权本地/目标目录中查找类似 `$HOME/jhspoolers/ez_runs/ez-qwen3-4b-alfworld-*/ckpt/train_state.json` 或 `.../checkpoints/ckpt-outer0-S24/train_state.json` 的状态文件；这只是路径线索，run-dir 与 job id 的映射、文件是否仍存在、是否包含完整 optimizer/per-rank RNG/task pool 均**未核实**。本轮没有访问服务器。检查后只分类为 `evaluatable_only`（能可靠载入 adapter 与基座）、`continuable`（完整可验证 resume state 且配置匹配）或 `not_verified`；不得用 loss 曲线或 eval 崩溃直接判断续训资格。

## 4. 分阶段计划与验收门槛

### A. 冻结审计事实（当前阶段）

- 用本脚本在 CPU 上验证 raw 行/selection 多重集、summary、缺类与恶意 fixture；留存输入 SHA、source commit、审计脚本 SHA。
- 冻结原始交接文件，不覆盖旧 JSON、summary、日志或模型目录。
- 整理 checkpoint 候选清单与文件 hash；记录 base path 的 model config/tokenizer/权重名、大小、可得 hash，不下载、不换权重。
- 当前实现只产静态计划/只读 audit。没有 GPU runtime 或训练通过结果。

### B. 目标环境和可执行性核验

- 在目标服务器上先确认现有依赖、TextWorld/ALFWorld 版本、实际可用 GPU 与磁盘；不自动安装包，不运行 `alfworld-download`。
- 读取当前安装环境真实 `valid_unseen` task pool，记录排序语义、文件列表/hash、数量、六类任务覆盖与官方 `won` 终止行为；**此时不因预期 134 而跳过发现/核验**。
- 对本地既有 `qwen3_4b` 做只读身份审计。确认 base-only 与目标 solver adapter 是同一 base lineage；别名不能作为证据，不能用 SkillRise `qwen3_4b_inst_2507` 替换。
- 用少量不进入最终 unseen 报告的固定 dev episodes 做环境 oracle、prompt/parser 行为核查、base/adapter HF 与 vLLM greedy token-level parity。Parity 至少比生成 token IDs、decode、projection/executed command；小样本只证明兼容，不证明随机采样统计等价。
- 重点检查 `max_new_tokens=96`：它是当前旧 EZ eval/default config 候选补全值，论文未披露，可能截断 JSON/tool-call。先在固定 dev 统计 finish reason、长度、闭合 tool-tag 比率、有效命令和环境进度，再冻结足够上限。base/adapter 共用同值，不用 `valid_unseen` 分数选上限；不能把 SkillRise 的 1024 直接说成 EZ 论文配置。

### C. 固定 matched dev 对照

- 固定任务 list/hash、模型路径与权重 hash、seed、backend、context、stop/decode、官方 won 标准和结果 schema。
- 对同一 base 比较 `base-only` 与恢复出来的 solver adapter；增加 no-menu/menu 单变量臂用于诊断格式依赖。prompt / admissible exposure 属协议变量，结果不能合并成一个未经区分的分数。
- `valid_unseen` 已经被历史 mid-eval 使用过（包括子集评估记录），未来把它冻结不会让它变成从未接触的 test。最终可按论文指定 validation split 报告，但要披露 prior exposure，避免反复用 unseen 分数调 prompt、截断长度或权重。

### D. 训练信号和状态 verifier 验收

- 先检查候选 checkpoint 完整度、base 匹配、两个 adapter、optimizer、Python/NumPy/Torch 与 rank-local RNG、task pool、outer/phase/next step、world size、源码/config/env identity。完整性不足则只做可评估 adapter 分类；不盲目 resume。
- paper G=16 是**一个全局任务组**。8 卡应将同一组分为每 rank 2 条 rollout；不能把全局 `TPS=1` 自动增为 8 个组。`G_P=8` 是每个 proposed task 的 solver solve-rate 估计 rollouts。保持训练语义、seed、任务组与 rollout 数一致，不能把增加数据量称为加速。
- proposer 的 invalid proposal reward=0，应留在 proposer GRPO reward group 中，使格式/无效提案面对正确的零奖励；validity filter 只决定任务能否进入 solver pool。Equation 1 的 valid reward 为 `R_p = R_difficulty + w_sub K + w_div |A|`，其中各项非负；零 reward invalid 不应得到错误的正 advantage。不要把其它项目的“归一化前删 invalid”规则搬来。保留所有 valid proposals/solver rollouts 的负 advantage。
- solver 按 Eq.7：最终状态成功 reward=1；未成功按完成子目标比例计分；K=0 时 subgoal reward=0。`w_sub=0.3` 仅在 proposer Eq.3 子任务数奖励中，不再乘到 solver 的子任务比例上。
- 验收 state verifier：原始初始状态与 reset 初始 facts 一致；对象 type 可忽略实例序号，但多关系必须保持同实体绑定；final 全匹配/部分匹配/无子任务 0 分支；visited-state 子目标匹配；无 finish/非法 proposal 拒绝进入 solver；named-object/task relevance/filter 要有反例。实现中的 heuristic 要标 heuristic，不能宣称作者精确规则。
- 训练 smoke 只证明结构不崩。学习信号验收需看组内实际 rewards/advantages、valid negative advantage 保留、finite policy-gradient 与参数 delta、adapter 同步；奖组同分时 reward-normalized PG advantage 为零。但 KL β=.001 若 policy 已偏离 reference 仍可能单独提供梯度，因此不能把“同分必然总梯度和总更新为零”泛化；检查并分别记录 PG/KL 信号。只看 loss 或 `loss != 0` 不足以证明有效更新。

### E. 受控续训或新训

- 若 checkpoint 经 D 验收为 `continuable`，只能从保存的 `next_outer/phase/step` 按签名一致继续；不得重复已完成的 outer 更新。若没有可信可续训状态，则明确开始新四 outer run，不能把它说成修复/续接原 run。
- 论文 Table 3 对 ALFWorld 是 4×(25 P + 25 S)=200 update opportunities；TP/TS=30，G=16，G_P=8，rollouts/step=16，context=40960。Table 3 另报每 proposer step ≈500 s、每 solver step ≈200 s；按 200 步简单相乘约 70,000 s（19.4 node-hours），与 Appendix D 的“roughly 15 GPU-node hours”只是近似估算口径，不能当保证。目标为 8×A800，与论文 8×H100 不同，不声称速度/显存等价。
- 运行中按 outer 固定 dev 验收，留存每个 checkpoint 的可重放 manifest、adapter 参数 hash、optimizer/RNG/progress。只按预注册 dev 规则挑选 checkpoint，不根据最终 unseen 结果多轮调参。
- 速度只测：相同 G/TPS/G_P、相同 token/turn budget 下有效轨迹/秒、GPU-hours、generation/environment/backward/sync/update 时间、CPU worker 峰值与 adapter 刷新开销。不能用增加 TPS/数据量换来的吞吐冒充等预算提速。

### F. 最终锁定评测和报告

- 完成后锁定 checkpoint hash 和协议 manifest。base-only 和 solver 使用相同 model base、task list/hash、prompt scaffold、context、decode、stop、seed/backend 与 benchmark won；adapter 差异是唯一 treatment。
- 在发现并哈希的完整 `valid_unseen` task list 上各做四份完整 sample grid，每份 task 恰一次；报告四份 suite SR、均值、ddof=0/ddof=1 SD、逐任务/类型明细、原始 trace/hash、实际 task 数、software/backend/model identity、prior exposure。不得将 `any-of-four` 记为 Pass@1，也不以 6-type macro 取代 episode/task suite metric。
- 不因最终验证集分数迭代 prompt、菜单、模型或输出上限；若需要调整，回 dev 后重新注册协议并标为下一轮实验。

## 5. 当前已知限制/未完成项

- `audit_current_evidence.json` 是输入审计产物，不是 checkpoint 审计、模型 identity 证明或轨迹 verifier。
- 本轮没有扫描目标服务器 checkpoint，没有验证 job 13026569 的 run-dir/文件现存性；`ckpt-outer0-S24` 只是值得检查的历史候选。
- 本轮没有启动训练/评测、运行环境 oracle、实际 HF/vLLM parity、GPU/NCCL 多卡或正式 valid-unseen grid。
- 训练源有保存顺序候选证据，但本轮没有作 resume；环境变量、实际有效参数和 per-rank state 必须在候选 checkpoint 本体核实。
- 完整作者 zero-shot prompt、具体 sampling、评测 pool manifests、部分 verifier edge cases 仍未知。符合计划也只能称“按公开协议尽可能对齐”，不能声称还原所有隐藏细节。

### 下一轮有界代码改动（本轮未实施）

这些是通过当前审计后可单独派发的 runtime 工作包；每项先复核现代码与 checkpoint schema，再按下列边界实现。它们不授权改奖励定义、采样算法或训练目标。

| 工作包 | 文件边界 | 验收用例 | 明确不做 |
|---|---|---|---|
| Resume 协议签名 | `selfplay/src/ez_train.py` 的 `_resume_signature`、`ez_save` 与恢复入口；将 resolved training manifest 中的 style/menu/parser/fewshot/backend 和相关源码 hash 纳入签名 | 改任一语义字段时拒绝 resume；相同完整签名可连续保存/恢复；缺字段或旧 schema 只允许在显式兼容范围内加载用于 eval，不绕过签名启动续训 | vLLM sampling RNG 尚未保存，不能宣称 bitwise resume；不自动把旧 checkpoint 当可续训状态 |
| 训练指标语义 | `selfplay/src/ez_train.py` 记录处与其日志测试 | 保留 raw sequence-sum `train/loss`；另报明确的 mean objective、PG/KL 分量、全局 reward std、非零 advantage 组比例、partial reward、截断率和 active adapter delta；用手造 batch 验证每个分母/聚合口径 | 不用改日志名掩盖旧值；不调整优势、奖励、loss 或优化器算法 |
| Eval pool 与配对绑定 | `selfplay/src/eval_ez.py`、`selfplay/scripts/ez_eval_controls.py`、`selfplay/src/ez_eval_merge.py` 与对应 manifest/tests | 发现实际全量 pool 并保存排序/文件 hash；base 与 adapter 共用 task/sample/seed/trace manifest；输入缺片、额外行、task/hash 不同或 parser/prompt/stop/resolver 未配对时拒绝合并 | 不把 134 硬编码为事实；prompt/stop/resolver 的变化必须在配对 arm 同步，不从 unseen 分数反复挑协议 |
| 训练启动与 vLLM 刷新 | `selfplay/scripts` 的 launcher、`selfplay/src/ez_vllm_policy.py` 和入口测试 | 显式记录 backend 与可见 GPU 到 rank 映射；确认 LoRA 更新后推理端加载的是当前 adapter 版本；HF 与 vLLM 入口分别做 smoke/parity 验收 | 当前标准 A800 launcher 默认 HF，另有 vLLM 专用入口；不能把两条入口混称为同一个后端。保留已审过的 global G=16 语义，不重写训练算法、不自动切 backend |

## 6. 本轮验收

```powershell
python -m pytest selfplay/tests/test_ez_repro_audit.py -q
python selfplay/scripts/ez_repro_audit.py --repo-root . --output handover/explorer_zero_repro_plan_20261015/audit_current_evidence.json
```

CPU fixture 覆盖真实 handover 数字、缺/重复 seed、游戏 multiset 错位、`won` 非 bool、缺 task type 的 macro-null、selection task-type 冲突、原子输出与输入保护、审计工具 stdlib-only/不执行命令。GPU 与续训验收仍待目标环境执行。
