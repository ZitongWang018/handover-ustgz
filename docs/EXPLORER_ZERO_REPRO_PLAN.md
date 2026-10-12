# Explorer-Zero 复现评估与计划

> 论文：*Explorer-Zero: Improving Self-Play for Long-Horizon Tasks with Decomposable Verifiable Rewards*（ICLR 2027 在审，用户提供 PDF，29 页含附录）
> 本文回答三个问题：能不能复现？哪里不确定？怎么排计划？
> 最后更新：2026-10-12

## 0. 结论先行

**可以复现。** 三个理由：

1. **成本极低**：论文附录 D 写明一个 ALFWorld run ≈ **15 GPU-node-hours（8×H100）**，即一张 8 卡 H100 节点跑 ~2 小时。我们的三期配额（16×H100）随便跑得动，二期 8×A800 也行（15 node-hours ≈ 2 小时 8 卡）。
2. **基础设施大头已有**：我们有 `alfworld_env.py`（ALFWorld batch env）、`rollout.py`（rollout 管线）、`train_grpo.py`（GRPO 训练循环，刚修完 9 个实测 bug）、`reward.py`（组内归一化）。缺的只是 **proposer 角色 + 状态快照/比对 + 交替训练循环** 三块。
3. **论文披露度高**：超参全在 Table 3，prompt 骨架在附录 G（Fig 13/14 给了 ALFWorld proposer/solver 的完整骨架），验证规则、过滤规则、训练算法（Algorithm 1）都有。复现的主要障碍不是"藏着掖着"，而是**少数关键工程细节一笔带过**（见第 2 节不确定项）。

**预期**：主结果（ALFWorld 85.07±1.6）复现到 ±5 分以内算成功；精确对齐不现实，因为 prompt 有省略、状态匹配规则只有一句话。

## 1. 论文给出的关键规格（复现的"确定部分"）

### 训练配置（Table 3，ALFWorld 列）
| 项 | 值 |
|---|---|
| 基座 | **Qwen3-4B（base，非 instruct）** —— 我们二期 `~/thinktool/models/` 已有 |
| 外循环 N | 4 轮（每轮先训 proposer 再训 solver） |
| proposer/solver 步数 | 25 / 25 per loop（AppWorld 是 15/50） |
| proposer 探索预算 | 30 turns |
| solver 最大回合 | 30 |
| context length | 40960 |
| GRPO group size G | 16 |
| 每 proposed task 的 solver rollout（估 solve rate） | G_P = 8 |
| 训练步 batch | 16 |
| 微调 | **LoRA rank 32, α=32** |
| 优化器 / lr | AdamW / 1e-4 |
| clip ε1=ε2 | 0.3 |
| KL penalty β | 0.001（对参考策略） |
| subtask 奖励权重 w_sub | 0.3 |
| diversity 奖励权重 w_div | 0.3 |
| 硬件 | 8×H100；proposer step ≈500s，solver step ≈200s |

### 方法核心（确定部分）
- **proposer** 从 s₀ 探索，用三个控制工具：`declare`（实际用 `finish` 提交最终任务）、`sub_finish`（声明子任务并快照状态）、`finish`（声明最终任务并快照终态）。快照 = 相对 s₀ 的**状态 delta**。
- **solver 奖励**（Eq.7）：成功 = 1；失败 = 子任务完成率（Eq.6，无序清单，命中过的历史状态也算）。
- **验证规则**（§4.1）：ALFWorld/ScienceWorld **比对 object type、忽略实例编号**（apple 1 和 apple 2 等价）；AppWorld 比对任务提到的、参考轨迹中变化过的应用的状态变化。
- **proposer 奖励**（Eq.1）：R_difficulty = 1−|2p̂−1|（solve rate 0.5 最优）+ w_sub·K（非冗余子任务数，Jaccard>0.6 且 delta 相同视为冗余）+ w_div·|动作类型集合|。
- **有效性过滤**（3 条规则）：有 finish 任务描述；无冗余子任务；任务描述里提到的每个物体/应用都出现在 proposer 轨迹中。
- **评测**：ALFWorld unseen valid split，每个任务采样评估 4 次报均值±std。
- **solver 只见** task_description + 观测，**不见** proposer 轨迹和验证工件。

## 2. 不确定项清单（按风险排序）

| # | 不确定项 | 风险 | 影响 | 对策 |
|---|---|---|---|---|
| U1 | **ALFWorld 状态匹配的具体序列化**：论文只有一句 "compare object types while ignoring instance indices"。用 textworld 的哪些 facts？位置/温度/开合/清洁度哪些算"task-relevant changes"？solver 多做了无关变化还算匹配吗（子集匹配还是精确匹配）？ | **高** | 直接决定奖励函数，错一点整条曲线就偏 | 自己定规格：dump 全部 facts → 按 proposer delta 中出现的谓词子集匹配；做单元测试 + 人工抽查 50 条 |
| U2 | **prompt 省略部分**：附录 G 明说 "Details omitted for space"，`<think></think>` 留空的怪癖、观测格式、错误处理都没给全 | 中 | proposer 出题质量漂移 | 按骨架补全，对齐生成任务的接受率（论文 Appendix E.3 有 filter acceptance 趋势可对） |
| U3 | **proposer/solver 是否可见 admissible_commands**：prompt 里只列了动作空间，没说每步给不给合法动作表 | 中 | 影响 action 合法率和我们 raw/best_match 模式的取舍 | 两个变体各跑一次 smoke，看哪个接受率高 |
| U4 | **rollout 采样温度未在 Table 3**：GRPO 默认 1.0 但没写明 | 低 | 指标小幅漂移 | 用 T=1.0/top_p=0.95，与我们现有配置一致 |
| U5 | **每步任务/组的确切编排**：batch 16、G 16、G_P 8 之间怎么组织（几个 s₀、每组几条轨迹）只能从文字推断 | 低 | 效率差异，不影响正确性 | 按最自然的读法实现：每步 16 个任务、每任务 G=16 条 solver rollout |
| U6 | **训练框架未披露**：LoRA+KL 的具体实现（TRL？verl？自研？） | 低 | 工程选择自由 | 自研（在 train_grpo.py 上加 LoRA 和 KL），或直接用 peft |
| U7 | **Qwen3-4B base 的 prompt 适配**：base 模型遵循工具调用格式能力差，论文用空 think 标签 + 严格格式，冷启动解析失败率未知 | 中 | proposer 阶段早期可能大量 parse fail | 先用 few-shot 或先小步 SFT 打底（这本身可作为一个 ablation） |
| U8 | **评测 split 细节**："unseen validation split" = valid_unseen 的哪些 task type、多少条 | 低 | 数字不可比 | 用全量 valid_unseen（论文口径对应 eval_out_of_distribution） |

## 3. 复现计划

### Phase 0 — 设计冻结（半天，纯本地）
- [ ] 定稿状态匹配规格（U1）：state dump 格式、delta 提取、"task-relevant" 谓词筛选规则、匹配算法；写单测（用 3 个手工场景：加热匹配、实例替换匹配、多余动作容忍）。
- [ ] 按附录 G 骨架写 proposer/solver prompt 全稿（补全省略细节处做显式决策并记录）。
- [ ] 决定：LoRA r32（贴近论文）vs 全参（贴我们现有 fp32 管线）；决定 KL β=0.001 是否保留。
- [ ] 产出：`docs/EZ_STATE_MATCH.md`（规格）+ 两份 prompt 文件。

**验收**：单测全绿；规格文档过一遍人工 review。

### Phase 1 — 环境快照机制（1-2 天）
- [ ] `alfworld_env.py` 加 `dump_state()`（textworld facts 序列化）+ `state_delta(s0, s1)`。
- [ ] 新文件 `ez_verify.py`：参考状态管理、`s ≃ s_ref` 匹配器、Eq.5/6/7 奖励实现、有效性三规则过滤。
- [ ] 接进 `unit_test.py`。
- **验收**：对 20 个手工构造的 (s_ref, s_solver) 对，匹配结果与人判一致。

### Phase 2 — proposer 冒烟（2-3 天）
- [ ] `ez_proposer.py`：探索 loop + `sub_finish`/`finish` 工具解析 + 快照落盘（jsonl）。
- [ ] proposer prompt 接入；parse 失败率统计。
- [ ] 用**冻结的 base Qwen3-4B**（不训练）跑 50 条探索轨迹，人工抽查生成任务质量 + filter 接受率。
- **验收**：接受率 ≥60%（论文没给绝对值，E.3 有趋势可对）；抽查任务物理可达、描述清晰。

### Phase 3 — solver 冒烟（2-3 天）
- [ ] 生成任务池（Phase 2 产物）→ solver GRPO 小规模训练（单卡 A800/H100，3-5 steps × G=16）。
- [ ] 验证 Eq.7 奖励分布：失败轨迹的子任务完成率是否有区分度（对照 Prop 1：collapse 率应显著低于二值奖励）。
- **验收**：collapse 率从 ~100%（二值）降到 <50%；`update_rel > 1e-10`（我们的准入门）。

### Phase 4 — 完整交替训练（主实验，8 卡）
- [ ] 交替循环：4 outer × (25 proposer + 25 solver steps)，LoRA r32，实现 Algorithm 1（含 G_P=8 的 solve-rate 估计和 proposer 奖励）。
- [ ] 断点只留最新一个（铁律二）；日志含 collapse 率、filter 接受率、solve-rate 分布。
- [ ] 提交：三期 `acd_u` 8×H100（预计排队 ~15h，跑 ~2-4h）；二期 8×A800 备选。
- **验收**：跑完 4 outer loops，无 NaN/挂死。

### Phase 5 — 评测对齐（1 天）
- [ ] valid_unseen 全量评测，每任务 4 次采样，报 mean±std。
- **成功线**：≥ 80（论文 85.07±1.6 的下沿留容差）；**及格线**：显著 > Self-Challenging 复现值。
- [ ] 跑 Table 2 关键消融：one-shot proposer vs interactive、±subtask。

### 成本估算
| 阶段 | 资源 |
|---|---|
| 0-2 | 本地 CPU |
| 3 | 单卡几小时（debug 分区/单卡 H100 立即可用） |
| 4 | 8 卡 × ~4h（含余量）≈ 32 GPU-node-hours，含排队 1 天内完成 |
| 5 | 单卡评测几小时 |

## 4. 与我们主线（SPADE 路线 B GRPO 基线）的关系

- **不冲突，是互补的第二论文线**：主线用人工任务训 GRPO（对应论文 RL full=95.71 那一行），本计划复现的是自造任务线（85.07 那一行）。两条线共享 `alfworld_env.py` / `rollout.py` / GRPO 内核。
- **代码复用映射**：`train_grpo.py` 的组归一化、clip、DDP 骨架直接可用；新增 ez_verify / ez_proposer / 交替循环三个模块。
- **风险提示**：我们的 8 卡 job 12986056 未确诊，Phase 4 之前必须先解决 8 卡稳定性，否则复现也会踩同一个坑。
