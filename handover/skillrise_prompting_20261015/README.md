# 交接：SkillRise zero-shot / ReAct 在 ALFWorld 上的评测（2026-10-15）

本目录用于把这次工作的**原始产物**交给另一个模型独立审核。
**只陈述可核验的事实，不含结论性建议。** 所有数字均可回溯到 `raw/` 里的原始 JSON。

---

## 1. 任务与目标

在 ALFWorld 上复现 GitHub 仓库 `Within-yao/SkillRise` 的两个 **prompting 基线**：
**zero-shot** 与 **ReAct**。用户明确：这两者在论文里**只有评测，不涉及训练**。

论文：**《SKILLRISE: Agentic Reinforcement Learning for Cross-Task Skill Evolution》**
（Zhejiang University / NUS / SJTU / Meituan），arXiv **2607.26784**。

目标值（论文 Table 1，Qwen3-4B，ALFWorld，Pass@1 %，**6 类任务 macro 平均**）：

| Method | Pick | Look | Clean | Heat | Cool | Pick2 | **Avg.** |
|---|---|---|---|---|---|---|---|
| Zero-shot | 1.5 | 6.1 | 3.6 | 4.2 | 2.8 | 8.6 | **4.4** |
| ReAct | 45.5 | 30.3 | 21.4 | 12.5 | 18.1 | 23.5 | **25.0** |

论文原文另有一句：*"We evaluate on **128 held-out task instances** from each environment."*
（ALFWorld / WebShop / ScienceWorld 三个环境**都**写 128。）

## 2. 本次结果

### 2.1 已完成：**128 槽** × 5 个固定种子（`raw/react_128_seeds01234.json`、`raw/zero_shot_128_seeds01234.json`）

| 设置 | macro (mean ± std, n=5 seeds) | 逐 seed |
|---|---|---|
| **ReAct** | **30.8 ± 2.6** | 26.5 / 30.4 / 32.0 / 33.1 / 31.7 |
| **Zero-shot** | **20.3 ± 3.9** | 14.1 / 20.1 / 23.2 / 23.9 / 20.0 |

episode 加权 SR（640 episode）：ReAct 30.5%（195/640）、Zero-shot 19.5%（125/640）。
论文对照：ReAct **25.0**、Zero-shot **4.4**。
⇒ ReAct 高 5.8（≈2.2σ）；Zero-shot 高 15.9（≈4σ）。**两者都高于论文。**

### 2.2 **未完成**：全 140 × 5 种子

提交于 2026-10-15 18:26，截至本文件生成时**仍在运行**（作业 13086144 / 13086145 / 13084003）。
完成后会产出 `quick_{react,zero_shot}_140_seeds01234_*.json`。
**本目录不含这四格数据。**

## 3. 运行事实

| 项 | 值 |
|---|---|
| 集群 | HKUST-GZ 二期（`hpc2login...`），账号 `xyu608` |
| 作业 | **13086144**（react，1 卡，gpu1-11）/ **13086145**（zero_shot，1 卡，gpu1-35）；8 卡备份版 **13084003/13084004** |
| 入口 | `skillrise/eval_quick_both_1gpu.sbatch`（= 本协议正式入口） |
| 评测器 | `skillrise/eval_quick.py` |
| 汇总 | `skillrise/summarize_quick.py` |
| 代码 commit | `2ad3fe8`（本目录建立时的 main 顶端） |
| 每作业结构 | 引擎初始化 ≈3 min → 128 槽 × 5 种子 → 全 140 × 5 种子 |
| 实测速率 | ≈25 min / 种子（1 卡共享节点，GPU 被他人作业抢占） |

## 4. 评测协议（`skillrise/EVAL_PROTOCOL.md` 已定稿）

| 项 | 值 | 依据 |
|---|---|---|
| 模型 | `~/thinktool/models/qwen3_4b_inst_2507` | 见 §5 的模型对照 |
| split | `eval_in_distribution`（valid_seen） | 上游 `grpo_alfworld_*.sh` 默认 |
| max_steps | 30 | 上游 `env.max_steps` |
| temperature / top_p | 0.7 / 0.95 | 上游 `actor_rollout_ref.rollout.val_kwargs` |
| max_new_tokens | 1024 | 上游 `data.max_response_length` |
| 采样种子 | **0,1,2,3,4（必须显式传）** | 见 §6 |
| 后端 | vLLM 0.8.5.post1（自研推理循环） | 用户指定 |
| prompt | 上游 `prompt.py::ALFWORLD_PLAY_PROMPT` 逐字 | — |
| 动作解析 | 上游 `projection.py::alfworld_projection(phase='play')` 逐字 | 无 `<action>` 标签时取整段输出**末 30 字符** |
| 历史格式 | 上游 `memory.py`：`Action {n}: {act}\nObservation {n}: {obs}`，最近 15 步 | — |
| 合法动作表 | 每步都注入（剔除 `'help'`），写法 `',\n '.join(f"'{s}'")` | 上游 `build_text_obs` |

两个设置的**唯一差别**：`zero_shot` 删去 `ALFWORLD_PLAY_PROMPT` 中"先 step-by-step
reasoning"那两条，其余逐字相同。该映射为用户批准的**自拟近似**
（上游清单原文：`not an author-published baseline`）。

## 5. 模型对照（决定用指令版，非 base）

上游 `env.sh` 只写 `SKILLRISE_MODEL_PATH="/path/to/Qwen3-4B"`，**未说明 base 还是 instruct**。
同题对照（同 prompt / split / 步数 / 解码，只换权重）：

| 模型 | 轨迹数 | 胜 | SR | `tag_valid_rate` | hit_cap |
|---|---|---|---|---|---|
| `thinktool/models/qwen3_4b`（base） | 4 | 0 | 0.0000 | 0.292 | 1.00 |
| `thinktool/models/qwen3_4b_inst_2507` | 8 | 1 | 0.1250 | 0.643 | 0.88 |

`tag_valid_rate` = 真正产出 `<action>…</action>` 的回合占比；其余回合走"末 30 字符"兜底分支。

## 6. 关键缺陷与修复：`SamplingParams` 未传 `seed`

**修复前**：`SamplingParams(n=1, temperature=0.7, top_p=0.95, max_tokens=1024, stop=[...])`
—— **没有 `seed` 字段**。vLLM 会给每个请求随机分配种子。

**观测到的后果**（同一配置、同一清单、同参数，仅重跑）：

| 128 槽 ReAct（未传 seed） | macro |
|---|---|
| 第 1 次（02:15） | 36.6 |
| 第 2 次（15:10） | 37.0 |

**修复后**（显式传 5 个种子）：**30.8 ± 2.6，范围 26.5 ~ 33.1**。
⇒ 前述 36.6 / 37.0 **双双落在修复后的分布范围之外**。
同一现象在逐类型上更剧烈（同配置）：ReAct 的 Look 42.9 → 64.3、Pick2 19.0 → 4.8。

**修复方式**：`SamplingParams(..., seed=seed)`；`eval_quick.py` 新增 `--seed-list`，
每个种子独立跑一整轮，rows 内记录 `"seed"`；`summarize_quick.py` 输出逐 seed macro 与 mean ± std。

## 7. 评测集：128 槽 vs 全 140（**此前未记录**）

`raw/selection_128.json` 的 `selection_semantics` 字段原文：

> `first game from each worker-seeded TextWorld shuffled cycle over the complete unsorted ALFWorld pool; duplicates retained`

实测该文件：

| 项 | 值 |
|---|---|
| `slot_count` | 128 |
| `pool_count` | 140 |
| **128 槽中去重后的不同游戏数** | **91** |
| **被重复的游戏数 / 最大重复次数** | **31 / 3** |

另：`AlfredTWEnv(cfg,'eval_in_distribution').game_files` **不是有序的**
（实测 `gf == sorted(gf)` 为 `False`），而上游的槽位分配依赖该顺序。

⇒ 128 槽**不是**"140 里挑选 128 个不同的"，而是 128 次**有放回**抽样，且顺序依赖文件系统返回次序。
全 140（`--n 0`，不排序）无此自由度。

## 8. 文件清单

| 文件 | 内容 |
|---|---|
| `raw/react_128_seeds01234.json` | ReAct × 128 槽 × 5 种子的**全部 640 条** episode 明细（含每条的 `seed` / `won` / `steps` / `n_actions` / `n_valid_tag` / `hit_cap`） |
| `raw/zero_shot_128_seeds01234.json` | 同上，zero-shot |
| `raw/selection_128.json` | 上游口味的 128 槽固定清单（GPT 版 `--prepare-selection` 生成） |
| `raw/summary_128.txt` | 上述两文件的汇总输出原文（逐 seed macro + mean±std + 逐类型表） |
| `ez_training_evidence/*.jsonl` | **另一件事**：Explorer-Zero 训练 run 的 `train_log.rank0.jsonl` 原文（见 §9） |

## 9. 附：Explorer-Zero（EZ）训练复查的原始日志

用户另问"EZ 训练有没有问题"。删除了 12 个有问题的 EZ run 之后，**删前把日志留档在本目录**
（`ez_training_evidence/`）。日志中的原文行：

```
# ez-qwen3-4b-alfworld-1013-0320
[ez-solver] 8 rollouts | reward mean=0.000 | final_match=0/8 | subtask_rate=0.000
[proposer] 3/16 accepted | mean_turns=29.5 | fmt_bad=281
grad_norm=1.70e+02 (504 个张量有 grad)

# ez-qwen3-4b-alfworld-1013-0224
update_rel=0.000e+00（0/504 张量真的动了；max‖Δp‖=0.000e+00）
真实 backward grad_norm=0.0000e+00   ← 日志自带解释："奖励全同分 ⇒ 优势全 0 ⇒ 梯度精确为 0"
[done] 1 outer loops in 4.7 min
```

**未核实项**：这批 run 的 loss / reward 曲线存在 `swanlog/run-*.swanlab`（SwanLab 私有格式），
**本次未解析**。哪些 run 属于"用户认为已完成的那次完整训练"亦未确认。

## 10. 本目录**不含**什么

- 全 140 × 5 种子的结果（**仍在运行**）
- EZ 训练的 loss/reward 曲线（`swanlog/*.swanlab` 未解析）
- 模型权重、tokenizer
- 上游仓库完整代码（仅 `skillrise/upstream/` 快照，在仓库另一处，不在本目录）
- 论文 PDF

## 11. 供审核者判断的问题（**问题，不是结论**）

1. 论文的 **zero-shot prompt** 具体是什么？正文只列 "Zero-shot, ReAct, Reflexion" 三个名字，
   未给模板。本次的 zero-shot 是**从 ReAct prompt 最小删改而来**。
2. 论文的 "128 held-out task instances" **来自哪个池**、**如何选取**？论文未定义；
   上游实现依赖文件系统返回顺序，实测含 31 个重复。
3. 论文的 prompting 基线**是否经过 RL 训练的验证批次机制**？（本次的 128 槽取自该机制，
   但 prompting 基线通常应在固定评测集上跑。）
4. 论文**未写** temperature / top_p / 采样次数。本次沿用仓库脚本的 0.7 / 0.95。
5. 本次两个设置都给合法动作表（继承上游 `build_text_obs`），**论文是否也如此**未知。
   这可能整体抬高成功率。
6. `eval_in_distribution`（valid_seen）**是否正确**？论文只说 "held-out"，未指明 split。
7. 逐类型差异在 n=14~28 下不可靠（同配置不同种子的逐类波动可达 ±20 分），
   **论文的逐类型数字是否也受同等噪声影响**？
