# 30B 推理加速交接

本次授权：与 DeepSeek worker 协作完成推理加速代码、由 GPT review 并同步 GitHub。不访问服务器、不安装依赖、不启动或取消集群作业。

## 实现范围

- 目标是 baseline 和训练完成后的 solver 评测；30B 训练 rollout 继续使用 HF CPU-offload profile。本次不声称解决训练过程中全部生成开销。
- 原始底座和封存 checkpoint 保持只读；只把 solver adapter 合并到新的 BF16 完整模型。不能把 proposer 一起合并，不能用合并模型续训原 run。
- 导出保留底座、checkpoint、solver 文件清单和校验和；缺失完成 manifest、来源不符或文件损坏时拒绝启动 GPU 评测。
- 对比合并前后全部参数/buffer 名称和形状，并直接检查导出 safetensors 的名称、形状、BF16 浮点精度与分片索引；支持共享权重别名。合并期间再次检查底座、solver 和 checkpoint 小型封存文件。导出结果仍标为 checkpoint 臂，不会伪装成 base-only。
- baseline 与训练后模型共同使用 chat/vLLM、完整 valid_seen、种子 0–4、原有提示/动作菜单/解析器、30 步及相同解码参数。
- 默认八个单卡评测分片，每卡最多四个环境进入批量生成；单条轨迹仍按环境返回逐步执行。这不是跨环境异步补位调度，也不保证任何固定加速倍数。
- 评测结束保留各阶段耗时与吞吐统计，供实际服务器判断 GPU 生成与 CPU 环境哪个占主导。

## 复现边界

旧 raw/vLLM、旧 chat/HF 合同和分数都保留，但不能直接配对本次结果。当前源码、配置、后端与软件版本必须重新生成匹配 baseline 合同。

BF16 合并与不同推理内核可能导致舍入差异，特别是 MoE routing；同一 seed 不保证 HF 与 vLLM 逐 token 一致。服务器应先核对少量固定提示的 token、logits、动作投影和轨迹，再完整评测；不能把 CPU 测试当作真实 30B GPU 等价性证据。

每个 outer 的完整权重导出约 60 GB，四个导出需另留约 240 GB 磁盘以及足够 CPU 内存。导出和验证需要读取完整模型，计时必须包括这部分；失败现场保留，不自动清理已有文件。

## 协作与验证

使用 `deepseek-worker` 技能的隔离草稿工作流，配置模型标识 `deepseek-flash`。Worker 的 shell 被其只读策略拒绝；其输出只作为未验证补丁，由 GPT 检查来源哈希、文件边界并在主工作区实际运行测试。未放宽 worker 沙箱。

默认 worker 状态目录的原子 rename 遇到 EXDEV，改用该技能公开的 `DEEPSEEK_WORKER_STATE` 参数把任务状态保存在本地 D 盘；未修改技能实现、全局认证或任何服务器环境。

修改前测试：112 passed，2 skipped（CUDA residency、Windows 符号链接权限），2 个既有 PEFT warning。

最终完整回归：**145 passed、2 skipped、2 warnings，54.29 秒**。另通过三个启动脚本语法检查、两个命令入口 help、所有 Python 文件语法检查和 git diff 检查。测试包含真实小型 Qwen3-MoE 的不同 proposer/solver 更新、导出后 logits 与 solver 的 BF16 容差对比，以及四轮自动评测、chat/vLLM 批量种子与损坏拒绝路径。两个跳过项目与修改前相同。

本地测试环境：torch 2.7.1+cpu、transformers 4.57.6、PEFT 0.17.1、safetensors 0.8.0、tokenizers 0.22.2。未安装 vLLM/GPU 进行真实推理测试；引擎接口、分批与种子测试使用 stub，不能代替服务器 vLLM 0.8.5.post1 和真实 30B 的验证。

复审处理：

- 两次模型导出 worker 因输出长度限制失败，未应用其不完整草稿；以成功的 CLI/测试草稿为基础，由 GPT 拆分导出模块并修正安全与参数完整性检查。
- worker 补丁收集器遇到 Markdown 嵌套代码块时漏收后半段；GPT 对照完整原始响应及源文件哈希恢复完整补丁，不依赖工具的 completed 标记判断实现正确性。
- 修复 solver 子目录与 checkpoint 根目录混淆、配置读取失败静默回退、重复大文件校验、HF finish reason 错位和缺少合并来源仍复用旧结果的问题。
- 不采用“缓存验证后永不再检查”的建议；保留每次导出前后读取源权重，避免长流程期间使用变化的模型。分片可缺少某些任务类型，但全量池必须包含六类，这是既有指标要求，不放宽。
- 自动评测的 CPU 合并 stdout/stderr、用时和成功/失败记录写在各 outer 目录；失败现场保留。若完成标记存在但后续文件/环境校验失败，结果仍不可复用，不通过删除文件伪装成功。

修改后的检查命令与文件清单见同包 commands.txt、artifacts.json。未访问服务器、下载模型、切换权重、改训练 loss/reward/超参数、启动或取消任何集群作业。尚无服务器 GPU 实测、实际加速倍数或新的成功率。历史交接和原有未跟踪测试目录保持原样，不纳入本次提交。
