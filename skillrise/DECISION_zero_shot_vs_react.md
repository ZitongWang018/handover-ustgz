# SkillRise ALFWorld 复现：zero-shot / ReAct 映射决定与依据

> 本文记录**一个必须由人拍板的选择**及其依据。用户已明确授权："你按仓库语义自行判断"。
> 目的是：如果之后发现映射错了，能立刻定位到是**哪一条判断**错了、并单点改回。

## 1. 事实：仓库里到底有什么

上游仓库 `Within-yao/SkillRise`（本地克隆 `_work/SkillRise`，仅关注 ALFWorld）：

- **没有**任何字面叫 `zero-shot` / `zero_shot` / `react` 的脚本、开关或配置项
  （全仓 grep：`grep -rniE "zero[_ -]?shot|react" --include=... .` → 除 `verl/` 外零命中）。
- 提供的两个可运行入口：
  | | `examples/skillrise_alfworld/grpo_alfworld_qwen3_4b.sh` | `examples/skillrise_alfworld/skillrise_alfworld_qwen3_4b.sh` |
  |---|---|---|
  | `env.env_name` | `alfworld/AlfredTWEnv` | `skillrise_alfworld/AlfredTWEnv` |
  | `env.num_attempts` | `1` | `K=3` |
  | `do_reflection` | `+env.do_reflection=False` | 多轮反射 |
  | `algorithm.adv_estimator` | `grpo` | `skillrise` |
  | `+env.meta_mode` | 无 | `skillrise` |
- `agent_system/environments/alfworld/prompt.py` 里有三个模板：
  - `ALFWORLD_PLAY_PROMPT` —— **明确要求"先 step-by-step reasoning，再在 `<action></action>` 里给出一个合法动作"**
  - `ALFWORLD_REFLECT_PROMPT` —— 反思模板（Reflexion 语义）
  - 若干 trajectory/reflection 拼接模板
- `env_manager.py` 的反射开关：`do_reflection`、`reflection_type ∈ {history_and_reflection, reflection_only, history_only}`（默认 `reflection_only`）。

## 2. 决定

**采用"prompting 框架"解释**，即三者里只有 prompt 不同，其余配置完全一致：

| 设置 | prompt | 其余配置 |
|---|---|---|
| **zero-shot** | 去掉 `ALFWORLD_PLAY_PROMPT` 里"先分步推理"那一条，要求**直接**给出 `<action>` | `alfworld/AlfredTWEnv`、`num_attempts=1`、`do_reflection=False`、解码/步数/数据 split 与 ReAct 完全相同 |
| **ReAct** | **仓库原样** `ALFWORLD_PLAY_PROMPT`（先推理、再 `<action>`） | 同上，逐项相同 |

两者都用 `num_attempts=1` + `do_reflection=False`：**反思（reflection）是 Reflexion，属于第三种设置，本次不碰**，
否则 zero-shot 与 ReAct 之间会混入第二个变量。

## 3. 依据（为什么不是另一种解释）

1. README 自述 "outperforming **prompting-based methods** and RL baselines" ⇒ 论文里的对照包含**prompting 类基线**，
   而 ALFWorld 文献里这一类的两个经典代表正是 **zero-shot prompting** 与 **ReAct prompting**。
2. 仓库 prompt 里那句"先推理、再 `<action>`"**本身就是 ReAct 的定义性特征**，
   而它出现在 `ALFWorld_PLAY_PROMPT` 这个**默认**模板里 ⇒ ReAct 就是"仓库原样"。
3. 另一种候选解释（zero-shot = `grpo` 那套 `num_attempts=1`；react = 反射多轮）
   存在两处不成立：① 那两者的差异是 **RL 算法与 attempts 数**，不是"框架"；
   ② 反射那套对应的是 **Reflexion**，把它叫 react 与文献惯例冲突。
4. 用户要求"直接基于仓库提供的代码照抄复现"+"完整对齐其所有配置"⇒ 本方案**不改任何非 prompt 配置**，
   只把 prompt 的推理那一条做成开关，改动面最小。

## 4. 残留不确定性（明确标注，不当成事实）

- **U1（高）**：论文里 zero-shot / ReAct 基线的**确切 prompt 文本与解码参数**未在仓库中给出；
  本方案的 zero-shot prompt 是**由 ReAct prompt 最小删改而来**，属**推断**，非论文原文。
  若拿到论文正文，应逐字核对。
- **U2（中）**：这两个基线在论文里**报的是哪个 split**（`eval_in_distribution`=valid_seen 还是 `valid_unseen`）。
  仓库脚本默认 `env.alfworld.eval_dataset=eval_in_distribution`；两处都会跑，分别报。
- **U3（中）**：论文报的数字**每任务采样几次**、是 greedy 还是采样（仓库脚本给的是
  `val_kwargs.temperature=0.7, do_sample=True`）。会按仓库原样 + 记录实际生效参数。
- **U4（低）**：是否对 zero-shot 也应给 admissible actions（本方案给，因为两者只差推理一条）。

## 5. 单点可改回

上述判断只影响**一个开关**（prompt 里是否保留"先分步推理"那一句）。
如果映射被推翻，改这一个开关即可重跑，**不需要重做环境、数据、加速或评测链路**。

---

## 6. 追加决定：模型用 `qwen3_4b_inst_2507`（指令版），不是 base

上游 `env.sh` 里 `SKILLRISE_MODEL_PATH` 的默认值只写 `/path/to/Qwen3-4B`，
**并没有说是 base 还是 instruct**。不猜，直接做**同题对照**（同一 prompt / split /
步数 / 解码，只换模型权重）：

| 模型 | 轨迹数 | 胜 | SR | **`tag_valid_rate`**（真的产出 `<action>` 的比例） | `hit_cap` |
|---|---|---|---|---|---|
| `thinktool/models/qwen3_4b`（base） | 4 | 0 | 0.0000 | **0.292** | 1.00 |
| `thinktool/models/qwen3_4b_inst_2507` | 8 | 1 | **0.1250** | **0.643** | 0.88 |

（job 13046956 与 13047182，均为 `react` / `eval_in_distribution` / `max_steps=30` /
`temperature=0.7` / `top_p=0.95` / `max_new_tokens=512`）

**依据**：base 版有 **71%** 的回合走上游 `alfworld_projection` 的
"取整段输出末 30 字符"兜底分支（= 无效动作），指令版把这一比例降到 **36%**，
并且**真的能赢下任务**。

**采用**：`SKILLRISE_MODEL_PATH=$HOME/thinktool/models/qwen3_4b_inst_2507`

**残留不确定性 U5（中）**：论文实际用的 Qwen3-4B 变体未在仓库中写明；
本决定基于"能产出有效轨迹"这一**可观测判据**，而非论文原文。
若拿到论文/作者配置应核对。改回只需换 `ZTWANG_MODEL_PATH`，**不影响其它任何环节**。

**U6（中，未处理）**：`tag_valid_rate` 仅 64%，即仍有 36% 的回合被上游语义判为非法动作
（动作仍是"末 30 字符"，照样发给环境）。**未做任何"改进"**——因为任何偏离上游的
解析放宽都会破坏对齐。若论文数字明显更高，这一项应优先复核。
