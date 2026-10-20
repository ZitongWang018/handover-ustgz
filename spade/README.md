# Qwen3-30B-A3B 的 SPADE tau2 评测

当前只做 tau2-bench 文本评测，不启动训练、games、BFCL、ACEBench 或其他实验，也不导入原仓库的 `selfplay/` 实现。被测模型为本地 **Qwen3-30B-A3B-Instruct-2507**；用户模拟器和自然语言断言裁判都使用 **DeepSeek 4.1 Flash**，由 `DEEPSEEK_API_KEY` 提供运行时凭据。

## 固定来源与评测口径

- `upstream/`：SPADE 官方仓库，固定 `ebd40ec872fc5630cac299cb5e38e7c89743bef5`。沿用其 `spade/core/eval/tau2_tasks.py`、`tau2_evaluator.py` 及官方 tau2 参数。
- `tau2-bench/`：Sierra 官方仓库，固定 `a2c024725189473d2d7cea3a5cfdbcc67478e41f`（2026-08-18）。这是按 SPADE 论文公开时间选择的版本，**不是已核实的作者历史环境版本**；SPADE 公开依赖没有固定 tau2 commit。避免自动跟随当前 tau2 2.0 的新增协议和默认裁判。
- 正式任务：retail/base 114、telecom/base 114、airline/base 50；每任务 4 次试验，共 1,112 条轨迹。
- 官方参数：最多 200 步、并发 48；agent temperature=0、top_p=0.8、top_k=20、max_tokens=16384；user temperature=0、max_tokens=2048。试验种子使用固定 tau2 实现的默认值 300。
- 主要分数为 SPADE 的严格分母 Pass@1：成功次数除以全部预定试验数，不是四次取最好。基础设施错误、缺失/重复试验、空数据均不能当作有效完成。
- 显式设置原生 `run_tasks(max_retries=0)`，关闭其默认 3 次基础设施重采样；这是一项已记录的执行差异。API 调用也不自动重试，失败保留现场并退出。

按用户要求更换了 simulator 和 NL judge，因此这是 **DeepSeek 用户/裁判设置下的对照**，不能直接宣称复现论文 GPT 用户/裁判的绝对分数。DeepSeek 使用 `deepseek-flash` API 别名；截至 2026-10-20，官方更新说明其对应 V4.1-Flash-1017。运行时记录服务返回的模型标识；未调用 API 时不能声称已核验具体服务权重。DeepSeek 显式关闭 thinking，以配合原设置的温度和用户输出预算。

## 准备已有环境

以下是之后在用户授权的服务器目录内运行的命令，当前未执行任何服务器作业。执行代理先读仓库根目录 `AGENTS.md`，确认实际授权的 `ztwang` 根目录；所有模型、代码、数据、缓存和输出都必须在其中。

```bash
git submodule update --init spade/upstream spade/tau2-bench
```

tau2 评测不需要初始化 Slime 或 Tinker 子模块，不需要 Ray、Megatron、ALFWorld、grounding corpus，也不需要下载 SPADE 的其他基准。两个官方仓库的源码和数据均通过 gitlink 固定，并保留各自许可证。

评测客户端使用用户已准备的独立 **Python 3.12** 环境，按固定 tau2 仓库的 `pyproject.toml` 安装核心依赖。SPADE 此路径按源码导入，只需其 tau2 模块依赖，不需要安装整个 `[eval]` extra（该 extra 含不相关的 GEM 等依赖）。本封装不自动安装、升级或修改环境。GPU 服务可以使用另一套已有 SGLang 环境；客户端通过 OpenAI-compatible HTTP 访问它。

## 八卡模型服务

默认使用官方 SGLang 服务方式，TP=8、DP=1、静态显存比例 0.7、Qwen tool-call parser。先预览；只有添加 `--execute` 才会启动前台服务。输出目录必须全新，模型和上游源码保持只读。

```bash
GPU_PYTHON=/path/inside/ztwang/gpu-env/bin/python
"$GPU_PYTHON" spade/run_tau2.py serve \
  --root /home/ztwang \
  --model /home/ztwang/models/Qwen3-30B-A3B-Instruct-2507 \
  --output /home/ztwang/runs/tau2-serve-UNIQUE
# 用户授权资源且预览通过后，同一命令加 --execute。
```

服务名默认 `qwen3-30b-a3b-instruct-2507`，地址默认 `http://127.0.0.1:30000`。服务进程不接收 DeepSeek 凭据，不会调用 `pkill`、停止全局 Ray 或安装依赖。若服务退出或启动失败，保留日志，不处理他人进程。

## 冒烟与完整评测

API key 不写入 `.env`、JSON/YAML、命令行或 Git；通过交互输入设置环境变量：

```bash
read -r -s -p 'DeepSeek API key: ' DEEPSEEK_API_KEY; printf '\n'
export DEEPSEEK_API_KEY
EVAL_PYTHON=/path/inside/ztwang/eval-env/bin/python

"$EVAL_PYTHON" spade/run_tau2.py evaluate \
  --root /home/ztwang \
  --model /home/ztwang/models/Qwen3-30B-A3B-Instruct-2507 \
  --endpoint http://127.0.0.1:30000 \
  --output /home/ztwang/runs/tau2-smoke-UNIQUE \
  --smoke
# 默认只预览；用户授权后加 --execute，三个领域各 1 个任务、1 次试验。

"$EVAL_PYTHON" spade/run_tau2.py evaluate \
  --root /home/ztwang \
  --model /home/ztwang/models/Qwen3-30B-A3B-Instruct-2507 \
  --endpoint http://127.0.0.1:30000 \
  --output /home/ztwang/runs/tau2-full-UNIQUE
# 用户确认冒烟轨迹和服务身份正常后，加 --execute 运行完整 1,112 条轨迹。
```

冒烟结果单独标记，不能进入完整评测表。已存在目录、缺少密钥、来源不匹配、任务不完整或发生基础设施错误时应停止并交接；不覆盖历史结果、不自动换模型或裁判、不用重采样改善失败分数。原始轨迹、参数身份和错误保留在本次独立输出目录。

通过现有 endpoint 评测时，`/v1/models` 只能核验服务名，无法独立证明远端加载的权重与本地文件相同。需要结合 GPU 服务启动日志与模型文件清单核对；不能仅凭模型别名断言身份完全一致。

## 本地验证

```bash
python -m pytest spade/tests -q
```

本地测试不访问 API、不需要 GPU、不产生真实成功率。上游源码不修改；自定义部分只负责路径/模型身份检查、DeepSeek 路由、独立输出和完整性审查。当前代码和验证状态见 `handover/gpt_feedback_20261020_tau2/`。
