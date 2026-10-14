# SkillRise zero-shot / ReAct 复现：全链路重查（第 1 轮）

> 触发原因：用户指出 **ReAct 总平均虽对上了（25.5 vs 25.0），但逐类型几乎全不一样**
> （Pick +26、Clean −14、Heat +19、Cool −14、Pick2 −15）⇒ 总平均"对上"很可能是巧合。
> 本文只记录**查到的差异事实**与**待修项**，不写结论性判断。

## A. 已确认的硬差异（会直接改变逐类型分布）

### A1. 评测实例数不对：论文是 **128**，我跑了 **140**
- 论文正文（ALFWorld 段）："We evaluate on **128 held-out task instances** from each environment."
- 仓库：`examples/skillrise_alfworld/grpo_alfworld_qwen3_4b.sh` 里 `val_data_size=128`；
  `skillrise_alfworld/env_manager.py` 的注释明确写 `the 128 distinct eval tasks, each once`；
  eval 走 `config.data.val_batch_size`（=128）个 slot，`val_task_mode` 默认 `'repeat'`。
- 我实际跑的：`collect_game_files(cfg, 'eval_in_distribution')` 的**全量 140** 个。
⇒ **两边不是同一个集合**，逐类型不可比。

### A2. `max_new_tokens` 不对：仓库是 **1024**，我用了 **512**
- 仓库脚本：`data.max_response_length=1024`。
- 我：`--max-new-tokens 512`。
⇒ 更容易被截断 ⇒ 缺 `</action>` ⇒ 走上游 `alfworld_projection` 的
"取整段输出末 30 字符"兜底分支 ⇒ **非法动作率虚高**。我实测 `tag_valid_rate` 仅 ~0.64，
这条是首要嫌疑。

### A3. `prepare.py` 是 VERL 模板残留，与 ALFWorld 无关
`examples/data_preprocess/prepare.py` 实际加载的是 `hiyouga/geometry3k`（几何数据集），
生成 `~/data/verl-agent/text/{train,test}.parquet`。
**ALFWorld 的任务实例来自环境侧**（`config_tw.yaml` 的
`dataset.eval_id_data_path='$ALFWORLD_DATA/json_2.1.1/valid_seen'`），
**不是**从 parquet 读的。我一开始差点把 parquet 当成评测集来源。

## B. 待查明（还没定论）

### B1. 「128 个」具体是哪 128 个
上游 `alfred_tw_env.py` L203-209：
- train：`game_files[:num_train_games]`
- eval ：`game_files[:num_eval_games]`
`config_tw.yaml` 里 `num_eval_games: -1` ⇒ 取该 split 的**全部**文件；
那 128 是**上层 env_manager 按 `val_batch_size` 取的批次**。
**尚未确认**：是"文件列表前 128"还是"按 seed 洗牌后的 128"，以及它内部
`self.game_files` 的排序依据。**这决定我们能否逐实例对齐。**

### B2. 为什么我们的 `collect_game_files('eval_in_distribution')` 给 140，而 `valid_seen` 有 187 个文件
我们的 selfplay 侧有 `task_types` / solvable 之类的过滤。**未查明过滤条件**，
但这意味着**我们的 140 与上游的 128 之间可能还差一层筛选**。

### B3. 论文的 **zero-shot** 到底怎么定义
论文只写："We compare against three prompting baselines: **Zero-shot**, **ReAct**
(Yao et al., 2022b), and **Reflexion** (Shinn et al., 2023)."
**正文与已抽取文本中未找到 zero-shot 的 prompt 定义**。
我的变体（删掉"先分步推理"两条、保留合法动作表与 `<action>` 格式脚手架）实测 **19.3**，
论文 **4.4**，高 4.4 倍 ⇒ 我的定义与论文不同，**但论文写了什么还不知道**。

### B4. 采样参数
- 仓库：`actor_rollout_ref.rollout.val_kwargs.temperature=0.7`、`do_sample=True`（我照抄了）。
- 论文正文：**未找到**温度 / top_p / 采样次数的说明（`temperature` 全篇仅 1 处命中，
  在 `16591`，属另一段上下文）。

## C. 下一步（按优先级）

1. **修 A2**（`max_new_tokens` → 1024），重跑，看 `tag_valid_rate` 是否显著上升。
2. **查 B1/B2**：把"上游那 128 个实例"精确复现出来（含排序/种子），否则逐类型永远对不上。
3. **查 B3**：继续在论文（含附录）里找 zero-shot / ReAct 的 prompt 模板；
   若确实未披露，**如实记录为"论文未披露"**，并把已试变体与各自数字列成表备查。
4. 三个都落定后再重跑，并把结果与 Table 1 逐类型并排。

## D. 本轮产出（可复现）

- `skillrise/eval_alfworld_prompting.py` —— 独立评测器（逐字搬运上游 prompt/解析/历史格式）
- `skillrise/skillrise_eval.sbatch` —— 8 卡分区入口（评测只用 1 卡）
- `skillrise/summarize.py` —— 换算成论文口径（6 类 macro，对应 Table 1 的 Avg. 列）
- 已跑：`out/react_eval_in_distribution_n0_*.json`、`out/zero_shot_eval_in_distribution_n0_*.json`
  （各 140 episode；**因 A1/A2 未修，此两格数据不得对外使用**）
