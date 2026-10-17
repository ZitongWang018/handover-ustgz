# 30B Instruct 聊天输入与 HF 配对评测修复（2026-10-18）

基于 `9c3ea6b`，审查输入为 `deepseek-feedback/e5c7a3`。本轮没有访问服务器、下载权重、安装依赖、启动或取消作业。旧反馈的 8.43% 是旧 raw/vLLM 设置下的报告值，不删除、不改写；本轮未独立重算该完整轨迹，也未证明 chat template 能带来多少提升。

## 当前唯一推荐设置

Qwen3-30B-A3B-Instruct-2507；solver `prompt_format: chat_template`；baseline 与 checkpoint 评测均为 HF、batch=1、shards=8。完整 valid_seen、种子 0–4、30 turns、32768 solver context、最近 15 组历史、动作菜单、`</action>` 停止、temperature=0.7/top_p=0.95/max_new_tokens=1024 不变。生成基准与训练后比较均使用同一配置；HF 的 batch=1 是显式资源适配，不减少任务或种子。

每个 solver 决策把原 ReAct 正文作为单个 user message，用 tokenizer 自带模板执行 `apply_chat_template(tokenize=True, add_generation_prompt=True)`。直接使用其 token IDs，不二次 tokenize，不再添加特殊 token。不为仅支持 non-thinking 的 Instruct-2507 引入其他模型的 thinking 分隔符。proposer 原本已使用 chat template，此轮保留其方式。

训练采样和评测调用同一个 `encode_solver_prompt`。合同固定 prompt_format、编码方式和实际有效 chat template SHA256；模型/tokenizer 文件身份、源码、软件、完整任务网格等检查继续保留。训练在加载大模型前核对 tokenizer 模板；评测运行与四快照预检再次核对。缺失模板、格式不符、模板漂移、旧合同或不匹配的比较均拒绝，绝不静默回退 raw/base-only。

HF checkpoint 评测明确加载封存的 solver adapter。HF 也改成直接加载到当前可见卡，避免八进程在 CPU 同时物化完整 30B。组合脚本透传 `--memory-profile`，训练前拒绝旧 raw/vLLM 基线；所有训练 rank 退出后，沿用既有四个不可变 outer 快照逐个 prepare/run/compare，并输出 summary.json/summary.csv。没有放松签名要求或删除旧产物。

轨迹新增 `rendered_prompt`、`prompt_format`、`action_is_admissible`。`raw_prompt` 仍是 ReAct 正文；`prompt_token_ids` 是实际模型输入。标签格式正确不等于动作在菜单内，更不等于任务成功。

## 执行者后续操作

先读并遵守 `AGENTS.md`；核验授权真实目录、解释器、模型/数据、缓存/临时目录、可用 GPU/RAM/磁盘和本轮运行授权。仅在干净工作树上安全更新，禁止 `/tmp`、覆盖式 tar 同步、清理历史文件或自行修改实现。本文不是服务器运行授权。

旧 raw 合同和 checkpoint 不能直接在新 chat 配置下续训。为新的 chat/HF 设置准备新的 baseline 和全新训练 run，不覆盖旧目录。以下路径与 UNIQUE 标识需要替换，prepare 和运行必须用同一解释器：

```bash
export SKR_PYTHON=/path/to/approved/python
"$SKR_PYTHON" selfplay/evaluate.py prepare \
  --config selfplay/config.yaml \
  --alfworld-data /path/to/existing/alfworld/data \
  --base /path/to/existing/Qwen3-30B-A3B-Instruct-2507 \
  --base-only \
  --results-dir results/base30b-chat-hf-UNIQUE \
  --output results/base30b-chat-hf-UNIQUE/contract.json
bash selfplay/scripts/run_eval.sh results/base30b-chat-hf-UNIQUE/contract.json
```

先检查真实轨迹中的角色边界、有效动作及环境反馈。不要仅凭 ExitCode 0 或标签格式率认定成功，更不要把 chat 的预期收益当成实测。需要 raw/chat 因果对照时，用独立配置/目录固定相同任务、种子、HF 后端和其他设置；不得混用旧 vLLM 分数代替对照。

基线完成并检查后，组合流程可用：

```bash
bash selfplay/scripts/run_experiment.sh \
  --run-id UNIQUE-30b-chat \
  --run-dir /path/to/repo/selfplay/runs/explorer-zero-UNIQUE-30b-chat \
  --base /path/to/existing/Qwen3-30B-A3B-Instruct-2507 \
  --alfworld-data /path/to/existing/alfworld/data \
  --baseline-contract results/base30b-chat-hf-UNIQUE/contract.json \
  --baseline-results results/base30b-chat-hf-UNIQUE/contract.merged.json \
  --backend hf --memory-profile qwen3_moe_cpu
```

只训练用 `run_train.sh` 的相同训练参数；只评测已完成 run 用 `evaluate_run.py --run-dir ... --baseline-results ...`。续训保持所有参数、run-id、合同和 profile，再提供 `--resume .../ckpt`；失败产物保留，不隐式覆盖或恢复。训练仍是单 HF 基座、CPU AdamW 状态与激活卸载，数学、LoRA 目标、奖励、G/GP/TPS 和更新次数未改。HF 显存不受 vLLM 的 utilization 配置限制。

## 协作、审查和验证

使用 deepseek-worker skill，API model `deepseek-flash`。三个只读草案任务分别为 `b87739c8-5051-4699-99c0-ffbf0b94ba2e`、`55de4c57-ea59-4231-ba92-e68bd8519e44`、`11d09e32-b0cc-40cc-ab10-1ff487ef5df3`，GPT 检查完成状态、精确文件范围与原始 SHA256 后选择性应用。worker 未改 live 项目或推送。一个草案中的 shell 检查被沙箱拒绝，未视为测试成功。

GPT 修正了草案的参数名不匹配、合同字段读取层级、不存在的 CLI 参数提示、补丁 hunk 格式等问题，并增加输入 ID 类型检查、实际模板漂移检查、HF 流式加载、动作合法性诊断和配对模板身份检查。追加只读复核 `d63f1cf4-135e-47e4-9bf5-4573ca15d099` 已完成：据其意见进一步验证 checkpoint 训练时绑定的实际模板/tokenizer 类与编码源码，以及 transformers/tokenizers 版本；将已有 tokenizer 类检查前移到训练模型加载前；评测类检查同步补齐。其关于 rank 的担忧在原 launcher 已逐卡屏蔽的情况下不是现有故障，但现已显式拒绝 HF 多卡可见的误启动。复核只覆盖其快照中的文件，不替代 GPT 对后续修正的审查与实测。

最终完整 CPU 回归为 **112 passed、2 skipped、2 warnings（28.51 秒）**。覆盖真实本地 tokenizer 的恰好一次聊天编码、HFPolicy 输入 ID 一致、prepare/run/merge 的 HF/chat 路径、真实小型 Qwen3-MoE solver adapter 加载与 logits 一致、合同拒绝、正式封存的四快照编排，以及真实 Bash 脚本在 CPU 替身下的参数转发/错误基线拒绝/训练失败后不评测。两个 skip 分别需要真实 CUDA 和 Windows 符号链接权限；warning 为既有 tiny PEFT 保存提示。41 个 Python 文件通过 AST，训练源码清单完整，三个启动脚本语法与 CLI 参数检查通过。本地版本记录在 artifacts.json，与交接报告中的集群版本不同，不能据本地 CPU 测试宣称目标依赖/GPU 已验收。

第一轮回归曾发现两个旧 raw 夹具尚未更新，以及一次 OneDrive 临时文件原子替换 PermissionError。前者按新合同补齐并验证；后者在授权的本地非同步临时目录完整重测通过，没有削弱原子封存或增加静默重试。

新协议的真实 30B 基线、八卡训练峰值/速度、完整四快照结果和训练收益仍未取得。本轮不能保证无 OOM、达到论文分数或在固定时间内完成。
