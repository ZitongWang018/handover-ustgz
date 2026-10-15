> **历史计划提示（2026-10-15）**：本文保留 2026-10-12 的原始评估，不代表当前审计结论；其中“可以复现”、运行成本和基础设施就绪等断言未经本轮验证。旧 smoke 通过不证明训练信号有效或论文结果已复现。当前复现边界、证据审计和分阶段验收见[新计划](../handover/explorer_zero_repro_plan_20261015/README.md)。

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

### Phase 3.5 — EZ 端到端冒烟 ✅ **单卡硬通过**（2026-10-12 job 12989496）

```
===== EZ 冒烟自检 =====
  [OK   ] 关0 梯度链通：proposer grad_norm=5.666e+02, solver grad_norm=5.644e+02
  [OK   ] 关1 facts 通道存在（109 条事实）
  [OK   ] 关2 update_rel 共 2 处：outer 0 P0=5.00e-01, outer 0 S0=0.00e+00
  [OK   ] 关2b 优化器实际更新张量数：outer 0 P0=252/504, outer 0 S0=0/504
  [OK   ] 关3 proposer 有有效提案：[(2, 4)]
  [WARN ] 关3 solver reward mean=['0.000', ...]（全 0：solver 一次都没完成任务）
结论：**通过** ✅ 关0/1/2b/3 全过：1 处 update_rel>0，单步最多更新 252 个张量
```

**2 卡真实多进程（DDP 接线验证）✅ 接线全对**（2026-10-12 job 12989603 @ `debug:gpu3-9`，2:42）

两 rank 各自 `本地 games=1/2`、同一 `run_dir`、`world_size=2`、DDP 包装成功、
探针 `hasattr(DDP,'set_adapter')=False`、池子为空时**两 rank 一致跳过 solver 相位**、
双双 `[done]` 干净退出、`train_log.rank0/1.jsonl` 各 15 行 —— DDP 该验的全验了。

但那一发 `accept=0/2`（`turns=12` 太短）⇒ 候选全同分 ⇒ 优势全 0 ⇒
`真实 backward grad_norm=0.0000e+00` ⇒ `0/504 张量动了` 是**正确**的。
旧判据只看位移、不看梯度，误报出两条 FATAL，把这一发判成 FAILED（见下）。

判据已经从"三关"扩到"关0/1/2/2b/3 + 多卡关4"，`ez_check_smoke_ok.py` 里逐条实现：

| 关 | 判据 | 落地 |
|---|---|---|
| 关0 | 两个 adapter 各自都能接到梯度（`grad_chain_selfcheck`，**不依赖环境/奖励**） | 启动自检，任一 adapter 为 0 直接 `SystemExit` |
| 关1 | facts 通道存在 | 启动预检 `assert_facts_mode` |
| 关2 | `update_rel` 对**本组奖励 spread** 判（专治"梯度断" vs "奖励同分"） | job 12989143 的教训 |
| 关2b | 位移计数 + **真实 backward grad_norm**（`grad_norm>0` 却 0 位移才是断链） | job 12989450 / 12989603 的教训 |
| 关3 | proposer `accept>=1`、solver 奖励不是全 0 | — |
| 关4 | 多卡：rank 日志齐 / `world_size==N` / 各 rank `update_rel` 同量级（**全 0 且 grad_norm 全 0 不算错**） | 见 `EZ_DDP_PLAN.md` §8 |

**`update_rel=0.000e+00` 的假警报（2026-10-12 job 12989450 → 12989496 定位）**

日志曾长期同时打印「`真实 backward grad_norm=6.1474e+00 (504 个张量有 grad)`」和
「`update_rel=0.000e+00`」——看着像优化器更新被精度抹掉。**不是。**
`update_probe` 取"第一个 `numel>=4096` 的 LoRA 张量"，按 `named_parameters()` 顺序恰好
命中 `layers.0.self_attn.q_proj.lora_A`；peft 把 **`lora_B` 初始化成全 0**，
于是 `∂L/∂A = Bᵀ(∂L/∂y)xᵀ ≡ 0` —— 该张量第 0 步梯度**精确为 0**，AdamW 的 `exp_avg=0`
⇒ Δp 精确为 0。梯度链一直是通的，是 probe 挑了个天生零梯度的张量。

修完后同一步的日志：

```
[outer 0 P0] update_rel=5.000e-01（=max ‖Δp‖/(‖p‖+‖Δp‖)；252/504 张量真的动了；max‖Δp‖=5.571e-02）
[outer 0][P0] 诊断 ... | 真实 backward grad_norm=4.3530e+00 (504 个张量有 grad；A=0.0000e+00 B=4.3530e+00)
[outer 0][P0] accept=2/4 | rew=[0.0, 0.0, 1.5, 1.8] | rew_spread=1.800
```

`252/504` 恰好一半 = 所有 `lora_B` 动了、所有 `lora_A` 没动，完全对上；
`A=0.0000e+00 B=4.3530e+00` 是同一件事的第二个证据。
**教训：GRPO 里 `update_rel==0` 有三种成因（梯度链断 / 组内奖励同分 / probe 挑错张量），
必须用三条独立判据分开，不能用一条猜。**

代码状态（2026-10-12）：7 个源文件 + 配置 + 5 个脚本，三 agent 交叉 review 抓出的
2 个 P0（`old logps` 未脱图 / G_P 撞 env batch）+ 6 个 P1 已全部修复。
61 项离线单测 + 16 项判读回归（`scripts/ez_check_smoke_selftest.py` 用例 A–P）全绿。

```bash
# 二期 · A40 48G · debug 分区（30min 上限，几分钟内起算，先跑这个）
sbatch -p debug -t 00:28:00 --gres=gpu:1 ~/ztwang/selfplay/scripts/ez_smoke_1gpu.sbatch

# 二期 · A800 80G · 单卡（想量真实显存峰值）
sbatch -p emergency_gpu -t 02:00:00 --gres=gpu:1 ~/ztwang/selfplay/scripts/ez_smoke_1gpu.sbatch

# 8 卡 DDP 验收（必须先过单卡；8 卡作业被策略限定只能投这两个分区）
sbatch -p i64m1tga800u8ka -t 01:00:00 --gres=gpu:8 ~/ztwang/selfplay/scripts/ez_smoke_8gpu.sbatch

# 三期 · H100（改 account/partition，见脚本头部【三期变体】注释）
sbatch -p acd_u -A d_tonggehuang_team --qos=normal -t 01:00:00 --gres=gpu:1 .../ez_smoke_1gpu.sbatch
```

脚本内部按顺序跑五段：离线单测 → **判读逻辑回归**（`ez_check_smoke_selftest.py`，纯 CPU 秒级）
→ `ez_check_env_facts.py`（**facts 通道，EZ 的命根子**）→ `ez_train.py --dry-run`
→ `ez_check_smoke_ok.py`（产物自检）。

冒烟规模会自动缩小（`G=2 G_P=2 TPS=1 turns=12`，因为论文默认 G=16×30turns 在 A40 的
30min 里跑不完）；想加大用 `--export=ALL,EZ_SMOKE_G=4,EZ_SMOKE_TURNS=16`。
**turns 别压到 3**：prompt 要求探索 10-20 个动作后才 `finish`，轮数不够时 proposer
必然 `accept=0` → 任务池空 → solver 相位整段被跳过，"冒烟看着跑完但其实一条真链路都没走"
（2026-10-12 job 12989069 实测踩到）。turn=16 时 accept 能到 2/4。

**已修的现场问题（2026-10-12）**

| 现象 | 根因 | 修法 |
|---|---|---|
| `fmt_bad=6/6`（解析 100% 失败） | Qwen3-4B-Instruct-2507 **不吐 `<tool_call>` 标签**，直接甩裸 JSON | `parse_tool_call` 增加裸 JSON 回退（括号配平扫描） |
| 一轮生成 8 个动作、顶到 max_new_tokens | `</tool_call>` 停止判据对这种输出永不触发 | `_ToolCallStopCriteria` 改为"闭合标签 **或** 第一个完整动作对象"即停 |
| 停止在**内层** `arguments` 的 `}` 上（截断工具调用） | `{"action_type":"look"}` 自己也是配平对象 | 扫描时跳过紧跟 `"arguments":` 之后的对象 |
| 跑满 turns 仍 `accept=0` | prompt 要求探索 10-20 动作才 finish，模型走到上限也不收尾 | prompt 加"一次一个动作"+few-shot；临近上限注入 `FINISH_NUDGE`（`NUDGE_WHEN_LEFT=3`） |
| `update_rel=0.000e+00` 但 `grad_norm` 非零 | probe 挑中了 `lora_A`（B 初始全 0 ⇒ 该张量梯度恒 0） | probe 测**所有**带 grad 的张量，报 `N/M` 计数；A/B 分开打 |

诊断脚本 `src/ez_debug_proposer.py` 可复现上述全部证据（打印 prompt / 原始输出 /
GC 在 peft 包装下是否真的关掉——**GC 兜底实测正常，不是本次的原因**）。

**判读表（`ez_check_smoke_ok.py` 的判据，别把这几种情况混为一谈）**

| 现象 | 判定 | 该怎么办 |
|---|---|---|
| 日志无 "facts 通道 OK" | **FATAL** | textworld 版本不够，改 `request_infos`；否则奖励是废的 |
| `update_rel = 0` **且** 奖励有区分度 | **FATAL** | 梯度链断了（no_grad / 断图 / lr=0），必修 |
| `update_rel = 0` **且** 奖励全同分 | **WARN** | 不是 bug：GRPO 组内优势全 0 时梯度天然为 0。缩规模下常见，加大 turns 再看 |
| `0/M 张量真的动了` 但 `grad_norm` 非零 | **FATAL** | 真断链（probe 已排除挑错张量） |
| `0/M 张量真的动了` 且 `grad_norm` **也为 0** | **WARN** | 不是 bug：奖励全同分 ⇒ 优势全 0 ⇒ 精确零梯度，不动是对的 |
| `N/M 张量真的动了` 且 N≈M/2 | **正常** | LoRA 第 0 步：B 初始全 0 ⇒ 所有 A 无梯度，只有 B 动 |
| 日志里根本没有 `真实 backward grad_norm` 行 | **WARN/FATAL** | 跑的是旧代码，无法区分"断链"与"无信号"，先升级 |
| proposer `accept=0/N` | **FATAL** | 提案器一条有效任务都没产出。`fmt_bad>0` 是解析/格式问题；`fmt_bad=0` 通常是 **turns 不够**（prompt 要求探索 10-20 动作才 finish，用 `EZ_SMOKE_TURNS=16`）|
| 多卡：某 rank 的 `train_log.rank*.jsonl` 缺失/为空 | **FATAL** | 那个 rank 提早挂了，看它的 `.err` |
| 多卡：各 rank `update_rel` 差 ≥100× **且该 rank grad_norm 非零** | **FATAL** | 该 rank 没参与梯度同步（policy 可能拿了 `raw.module`） |
| 多卡：各 rank `update_rel` 全 0 **且 grad_norm 也全 0** | **OK** | 一致无信号（开局奖励同分），不是没同步 |
| 多卡：`world_size=1` | **WARN** | torchrun 没生效，N 个进程各跑各的（速度不涨、日志多 N 倍） |
| 全部通过 | **PASS** | 允许开 8 卡正式作业 |

⚠️ 判"参数没动"的最终依据是**真实 backward grad_norm**，不是 `update_rel`。
`update_rel` 只能告诉你"动没动"，无法区分"有梯度却不动（错）"和"没梯度所以不动（对）"。
`ez_check_smoke_selftest.py` 里有 16 个回归用例把这条钉住（含 O/P 两个 2 卡场景）。

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
