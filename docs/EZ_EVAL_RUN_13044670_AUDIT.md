# EZ 13044670 评测交接补充审计

本文是既有 13044670 交接记录的补充，不改写原记录，也不重述未经本次代码核对的运行结果。本文记录新增的评测协议审计、可追溯证据和后续重跑要求。

13044670 的 ours 实际没有向初始提示传入 admissible actions，后续 feedback 也没有开启 admissible 环境开关，因此模型实际看到的菜单是关闭的；旧 manifest 固定写 `menu_enabled=true` 与实际 prompt 不一致。本补充明确更正这一协议记录，不改原 handover 或历史结果文件。本轮修复把 action menu 改为显式 `--action-menu auto|on|off`。ours 的 `auto` 保持兼容 `EZ_SOLVER_ADMISSIBLE=1`，默认关闭；显式 CLI 值优先于环境变量。rich scaffold 必须显示它自己的逐步编号动作表，故显式 `off` 会失败。菜单开启会改变模型看到的提示并标为 custom；菜单关闭时，格式恢复器仍可能基于环境 admissible list 安全恢复自然命令。ours trace/manifest 现在区分“模型可见菜单”和“恢复器可用菜单”。

评测 JSON 的 `protocol_manifest.software` 记录当次可取得的 ALFWorld、TextWorld、PyTorch、Transformers、PEFT 与 vLLM 包版本。早期 README/交接文字没有列出软件版本，不等同于当次 JSON 没有版本记录；不要为此改写历史快照。若旧报告 JSON 缺失，应将版本标为未知，而不是从当前环境补填。

本地路径核验现在可记录 README 标题、`base_model` 祖先来源、license 链接、config 字段、cache snapshot commit hint 和权重 shard 名称/大小。轻量清单状态固定为 `weights_unverified`；这些声明不是权重内容的证明。官方 Qwen3-4B README 把 Qwen3-4B-Base 列为祖先来源，同时以 Qwen3-4B 为模型标题；当前 model card 的 config 提示 eos 151645/max position 40960，而 Base config 提示 eos 151643/max position 32768。它们是待与集群实际目录核对的区分线索，不证明 13044670 使用了哪组权重。集群目录当前未作在线读取，本补充不宣称已完成身份核验。

后续同条件对照由 `scripts/ez_eval_controls.py` 生成手动计划：`paper_no_menu` 与 `json_menu` 只差菜单，`json_menu` 与 `json_menu_50` 只差步数预算，`public_rich` 是独立的多因素 scaffold control。计划默认覆盖 134 个任务、8 shards；带 `--ckpt` 时每个 arm 同时写 base 与 solver adapter 命令。生成器只写 JSON 和人工可执行 shell 文件，不运行命令、不调用 sbatch/SSH，也不加载模型。每次运行必须使用新的输出目录、共同 seed/backend/data/base，并核对每份结果的 manifest 与实际 prompt trace。

参考：

- [Qwen3-4B README metadata](https://huggingface.co/Qwen/Qwen3-4B/raw/main/README.md)
- [Qwen3-4B config](https://huggingface.co/Qwen/Qwen3-4B/raw/main/config.json)
- [Qwen3-4B-Base config](https://huggingface.co/Qwen/Qwen3-4B-Base/raw/main/config.json)
- 当前评测入口、菜单协议与计划生成方式见 [EZ_EVAL_AUDIT.md](EZ_EVAL_AUDIT.md)。
