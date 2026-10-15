# 新鲜训练与配对评测实现反馈

本文件记录针对 Explorer-Zero 新训练与评测路径的代码交付边界。旧 checkpoint 按本轮决定不抢救、不迁移；新的训练从 schema-v3 fresh run 开始。训练与评测分别有独立入口和合同，不把旧训练的隐式环境变量、旧 checkpoint schema 或中途 valid_unseen 评测带入新 run。

训练侧新增 `ez_train_v2.py`、schema-v3 纯标准库合同、显式配置和便携八卡启动脚本。Table 3 的训练 context 40960 保持对齐；max-new-tokens 1024 是本地选择。合同覆盖完整 base 文件内容、tokenizer/config、实际 train pool 与可用 trajectory、ALFWorld domain/grammar/config、源码和软件版本、后端及 generation 配置。Checkpoint 对两个 adapter、optimizer、task pool、所有 rank RNG 和训练进度做 inventory/hash，再封存并原子提升；resume 拒绝旧 schema、smoke checkpoint、身份漂移及互相冲突的训练位置字段。HF cache 的本地模型文件 symlink 按目标内容散列；checkpoint 自身仍须自包含。

训练报告在 rank 0 始终写 `metrics.jsonl`，保存 objective mean、原始 sequence objective sum、PG/KL 分量、奖励组统计、梯度、active adapter 更新、partial/final/truncation 和生成统计。SwanLab 只有 config 明确启用并选择 local mode 才初始化。vLLM engine seed 使用 base seed 加 global rank，并将 `max_model_len`/`max_lora_rank` 从解析合同实际传到引擎。vLLM stateful resume 不承诺 bitwise token continuation。

评测侧提供 CPU suite discovery、base/solver protocol contract、8-shard 执行、严格 merge 和 paired difference 工具；每个任务的 4 个 sample 保持独立，主报告是 sample-level suite SR 均值，不是 Pass@4。对照计划生成脚本只写 plan/shell 文件，不自动执行。评测命令和训练 smoke/fresh/resume 命令见本目录 [README](README.md)。

本轮仅进行了 CPU/静态验证，没有启动 GPU、训练或评测作业，也没有改模型权重。完整本地 CPU suite 为 **114 passed、2 skipped、9 warnings**；skip 是未设置 `EZ_QWEN3_TOKENIZER` 的真实 tokenizer 探针和 Windows symlink 权限限制，warning 是 CPU 禁用 CUDA autocast与 tiny PEFT fixture 的缺省 base config 提示。CPU 用例验证 Tiny PEFT 双 adapter 的保存/重载、optimizer/RNG 恢复、next-position 防篡改、训练 decode 参数传递、vLLM rank seed、PG/KL 目标分解、单卡 rank 映射、两 rank Gloo fresh/resume preflight 和 metrics 本地落盘。没有 Qwen3 实际 tokenizer 或正式模型验证；这些结果不证明 GPU 显存、性能、收敛或论文分数复现。
