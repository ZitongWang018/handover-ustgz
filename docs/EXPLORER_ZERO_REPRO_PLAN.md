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

### Phase 0 — 设计冻结（半天，纯本地）✅ 代码已落地
- [x] 定稿状态匹配规格（U1）：`src/ez_state.py`（归一化 / delta / delta_covers / match_reference）+ 单测。
- [x] 按附录 G 骨架写 proposer/solver prompt 全稿：`src/ez_prompts.py`。
- [x] 决定：**LoRA r32 α32**（贴论文，且 4B 全参 fp32 显存放不下）；KL β=0.001 **暂缓**（我们的 GRPO 内核是 β=0 无参考模型版，LoRA 低 lr 下影响小，列为 TODO）。
- 产出：`src/ez_state.py` / `src/ez_verify.py` / `src/ez_prompts.py`（规格直接落在代码 + 42 项单测里）。

### Phase 1 — 环境快照机制（1-2 天）✅ 代码已落地
- [x] `alfworld_env.py` 加 `with_facts` + `last_reset_facts` + `BatchStep.facts`（textworld `facts=True` 通道）。
- [x] `ez_state.py`（快照/delta/匹配）+ `ez_verify.py`（Eq.5/6/7 奖励 + 有效性三规则 + Eq.1-4 proposer 奖励）。
- [x] 接进 `scripts/ez_unit_test.py`（42 项）。
- **验收**：离线全绿 ✅；集群侧真 facts 待 `scripts/ez_check_env_facts.py` 确认。

### Phase 2 — proposer 冒烟（2-3 天）✅ 代码已落地
- [x] `ez_proposer.py`：探索 loop + `sub_finish`/`finish` 工具解析 + 快照落盘（`task_pool.jsonl`）。
- [x] proposer prompt 接入；`tool_call` 停止条件（`_ToolCallStopCriteria`，按 `</tool_call>` 停）。
- [ ] **待集群**：用冻结 base Qwen3-4B 跑 50 条探索轨迹，人工抽查任务质量 + filter 接受率 ≥60%。

### Phase 3 — solver 冒烟（2-3 天）
- [ ] 生成任务池（Phase 2 产物）→ solver GRPO 小规模训练（单卡 A800/H100，3-5 steps × G=16）。
- [ ] 验证 Eq.7 奖励分布：失败轨迹的子任务完成率是否有区分度（对照 Prop 1：collapse 率应显著低于二值奖励）。
- **验收**：collapse 率从 ~100%（二值）降到 <50%；`update_rel > 1e-10`（我们的准入门）。

### Phase 3.5 — EZ 端到端冒烟（**代码已就绪，等开服务器**）

代码状态（2026-10-12）：7 个源文件 + 配置 + 2 个脚本已写完，三 agent 交叉 review 抓出的
2 个 P0（`old logps` 未脱图 / G_P 撞 env batch）+ 6 个 P1 已全部修复，42 项离线单测全绿。
**下面这条命令就是 Phase 2+3 的合并验收，一条命令跑完。**

```bash
# 二期 · A40 48G · debug 分区（30min 上限，几分钟内起算，先跑这个）
sbatch -p debug -t 00:30:00 --gres=gpu:1 ~/ztwang/selfplay/scripts/ez_smoke_1gpu.sbatch

# 二期 · A800 80G · 单卡（想量真实显存峰值）
sbatch -p emergency_gpu -t 02:00:00 --gres=gpu:1 ~/ztwang/selfplay/scripts/ez_smoke_1gpu.sbatch

# 三期 · H100（改 account/partition，见脚本头部【三期变体】注释）
sbatch -p acd_u -A d_tonggehuang_team --qos=normal -t 01:00:00 --gres=gpu:1 .../ez_smoke_1gpu.sbatch
```

脚本内部按顺序跑四段：离线单测 → `ez_check_env_facts.py`（**facts 通道，EZ 的命根子**）
→ `ez_train.py --dry-run` → `ez_check_smoke_ok.py`（产物自检）。

冒烟规模会自动缩小（`G=2 G_P=2 TPS=1 turns=3`，因为论文默认 G=16×30turns 在 A40 的
30min 里跑不完）；想加大用 `--export=ALL,EZ_SMOKE_G=4,EZ_SMOKE_TURNS=5`。

**三关判定（`ez_check_smoke_ok.py` 的判据，别把这三种情况混为一谈）**

| 现象 | 判定 | 该怎么办 |
|---|---|---|
| 日志无 "facts 通道 OK" | **FATAL** | textworld 版本不够，改 `request_infos`；否则奖励是废的 |
| `update_rel = 0` **且** 奖励有区分度 | **FATAL** | 梯度链断了（no_grad / 断图 / lr=0），必修 |
| `update_rel = 0` **且** 奖励全同分 | **WARN** | 不是 bug：GRPO 组内优势全 0 时梯度天然为 0。缩规模下常见，加大 turns 再看 |
| proposer `accept=0/N` | **FATAL** | 提案器一条有效任务都没产出，查 prompt/解析/有效性过滤 |
| 三关全过 | **PASS** | 允许开 8 卡正式作业 |

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
