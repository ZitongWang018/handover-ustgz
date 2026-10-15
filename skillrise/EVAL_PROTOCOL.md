# SkillRise zero-shot / ReAct 评测协议（定稿）

> 状态：**已定稿**（2026-10-15）。这是本项目对 ALFWorld 上 zero-shot / ReAct 两个
> prompting 基线的**正式评测口径**。任何对外引用的数字都必须来自本协议。

## 1. 入口

```bash
# 推荐（单卡，秒起；不要用 8 卡分区，见 §5）
sbatch --export=ALL,EZ_STYLE=react       skillrise/eval_quick_both_1gpu.sbatch
sbatch --export=ALL,EZ_STYLE=zero_shot   skillrise/eval_quick_both_1gpu.sbatch
```

一个作业内**顺序**跑两段：① 128 槽 → ② 全 140，**每段各跑 5 个种子**。

| 文件 | 作用 |
|---|---|
| `eval_quick.py` | 评测器（逐字搬运上游 prompt / 解析 / 历史格式；推理用自研 vLLM） |
| `eval_quick_both_1gpu.sbatch` | **正式入口**，单卡 |
| `summarize_quick.py` | 汇总：逐 seed 的 6 类 macro + **mean ± std**；支持旧版无 seed 的输出 |

## 2. 固定配置（不得随意改）

| 项 | 值 | 依据 |
|---|---|---|
| 模型 | `~/thinktool/models/qwen3_4b_inst_2507` | 同题对照：`tag_valid` 0.29→0.64 且有真实胜场 |
| split | `eval_in_distribution`（valid_seen） | 上游脚本默认 |
| 评测集 | ① 128 槽（`out/selection_128.json`）② 全 140 | 见 §4 |
| 种子 | **0,1,2,3,4**（必须显式传） | 见 §3 |
| `max_steps` | 30 | 上游 `env.max_steps` |
| `temperature` / `top_p` | 0.7 / 0.95 | 上游 `val_kwargs` |
| `max_new_tokens` | 1024 | 上游 `data.max_response_length` |
| 后端 | vLLM 0.8.5（自研推理循环） | 用户指定 |
| 解析 | `projection.py::alfworld_projection(phase='play')` 逐字 | 无 `<action>` 标签时取整段末 30 字符 |
| 合法动作表 | **每步都给**（剔除 'help'） | 上游 `build_text_obs` |

## 3. ⚠️ 最重要的一条：必须显式传采样种子

```python
SamplingParams(n=1, temperature=0.7, top_p=0.95, max_tokens=1024,
               seed=seed,            # ← 不传这一项，结果不可复现
               stop=["</action>"], include_stop_str_in_output=True)
```

**不传 `seed` 的后果（实测）**：vLLM 会给**每个请求**随机分配种子，
等价于抬高有效温度 ⇒ **同一配置跑两遍结果不同，且系统性偏高**。

证据：未传 seed 时 ReAct 跑出 **36.6 / 37.0**；显式传 5 个种子后为
**30.8 ± 2.6（范围 26.5~33.1）** —— 前两个值**双双落在范围之外**。

## 4. 评测集：128 槽 vs 全 140

- **全 140**：`AlfredTWEnv(cfg,'eval_in_distribution').game_files` 的全量，**不排序**。
  无抽样自由度、可被第三方复现 ⇒ **报数字用这个**。
- **128 槽**：上游 `data.val_batch_size=128` 的取值方式，
  `selection_semantics` 原文为 *"first game from each worker-seeded TextWorld shuffled cycle
  over the complete unsorted ALFWorld pool; **duplicates retained**"*。
  实测：**128 个槽里只有 91 个不同游戏，31 个被重复（最多 3 次）**。
  论文对 ALFWorld/WebShop/ScienceWorld **都**写 128 ⇒ 它是**框架验证批大小**，不是评测集设计。
  ⇒ 用于**与论文口径对照**，不作为主报数字。

## 5. 报告规则

1. **报 `mean ± std`（n=5 seeds），不报单次值。**
2. **逐类型差异在 n=14~28 下不可靠**：同配置不同种子的逐类波动可达 **±20 分**。
   逐类型表**只能看形状**，不能拿单个类的差异下结论。
3. **zero-shot 是"自拟近似"**，上游清单里标注为
   `not an author-published baseline`。**无论数字多接近，都不能说成"复现了论文的 zero-shot"。**
4. 报告时须同时给出：模型路径、split、评测集、种子列表、解码参数。

## 6. 集群注意（踩过的坑）

- **评测一律用单卡 + `i64m1tga800u`**：8 卡作业**只能**提 `i64m1tga800u8ka` /
  `i64m1tga40u8ka`，而它们常被整节点占满，**塞不进 `mix` 节点的空隙**（实测白等 76 分钟）；
  单卡能直接塞进 `mix` 空隙，**秒起**。
- **每 GPU 最多 8 核 CPU**（填 16 会被 `sbatch` 直接拒）。
- 依赖装配：`PYTHONPATH=selfplay/src:py310deps`，且
  **`py310deps/numpy` 已改名为 `numpy.disabled_20261015`**（不改会被源码目录遮蔽，抛
  `ImportError: should not try to import numpy from its source directory`）。
  不要往 PYTHONPATH 里塞 `skillrise/upstream/`，否则它会提供另一份 `alfworld`，
  导致 tatsu 解析 `alfred.twl2` 时状态崩坏。

## 7. 已知未解决 / 需进一步确认

- 论文的 zero-shot prompt 定义（正文只列名字，未给模板）。
- "128 个"具体是哪 128 个（论文未定义；上游选择依赖文件系统返回顺序）。
- 论文未写温度 / top_p / 采样次数。
- 上游版启动器（`eval_alfworld_prompting.py` + `run_eval_8gpu.sh`）在本集群
  **从未成功跑通**（tatsu 错误），当前结果全部来自本协议的自研入口。
