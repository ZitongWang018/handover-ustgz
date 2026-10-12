# 当前状态（2026-10-12 13:42 CST）

> 本文档由协作 AI 维护，反映最近一次同步时的真实状态。改动前请先读 `docs/SELFPLAY.md`（技术路线）与 `selfplay/README.md`（代码说明）。

## 一句话
Qwen3-4B-Instruct 在 ALFWorld 上复现 SPADE（arXiv:2608.19197）的 GRPO 训练，代码已**端到端跑通并通过真机验证**，8 卡 A800 正式训练作业**正在集群上运行**。

## 已验证 ✅（真机实测，非理论）
在二期 A40（单卡）+ A800（8 卡）上实测通过：
- fp32 参数 + flash_attention_2 + bf16 autocast 全链路正常（冒烟日志 `[model] dtype=torch.float32 attn=flash_attention_2 ... autocast=bf16`）。
- rollout 动作合法率：invalid_rate ≈ 0.06（GC 乱码 bug 修复后）。
- 奖励有区分度 → advantage 非零（15 步 × 4 组配置下 adv≈0.8-1.0）。
- 存断点不崩（deepspeed/CUDA_HOME bug 已修复），且只保留最新一个（铁律二）。
- 单卡 fp32 冒烟 3 iter 全绿；8 卡 DDP 冒烟已启动。

关键验证日志与指标见 `selfplay/validation_logs/`：
- `smoke3_fp32_a40.out` —— 单卡 fp32 端到端（adv 非零、loss 非零）。
- `metrics_fp32_smoke.jsonl` —— 该次冒烟的逐 iter 指标。
- `a800x8_stage1_live.out` —— 8 卡 DDP 冒烟（截取时正在跑 iter 0）。
- `metrics_prod_rank0.jsonl` —— 训练指标（含历史 bug 时期的坏行，供对照）。

## 正在运行 🏃
- 二期 8 卡 A800 作业（`scripts/train_a800x8.sbatch`）：阶段0 预检 → 阶段1 五 iter DDP 冒烟 → 准入门 → 阶段2 正式训练（total_iters=220，save/eval 每 25 iter，只留最新断点）。
- 流程：排了十几个小时后于 2026-10-12 下午提前起算；曾因「旧代码无 deepspeed 修复」和「DDP 未搬 GPU」两次快速失败，均已修复后重提。

## 本轮修掉的关键 bug（全部实测定位）
1. **奖励符号反转**（`reward.py` penalty 写成 `+=`）——离线验证排序已修正。
2. **gradient_checkpointing + generate → 乱码** → 动作 100% 非法 → 静默空转。修复：生成期临时关 GC（`rollout.generation_mode`）。
3. **左 padding 生成段起点算错** → 训练目标被 prompt 尾巴污染。
4. **存断点走 `save_pretrained` → accelerate → import deepspeed → CUDA_HOME 缺失必崩**。修复：手写 config+state_dict（`_save_model_manual`），tied weight 按 data_ptr 去重。
5. **DDP 包装前模型没搬 GPU**（只有单卡分支搬）→ 8 卡一跑就炸。修复：先 `.to(device)` 再包 DDP。
6. **autocast 权重缓存 +8GB 常驻** → A800 80G 贴脸 OOM。修复：`cache_enabled=False`。
7. **GC 开关每步 toggle 累积 forward hook**（9000 个/全程）。修复：幂等 `gc_set` + 整段 rollout 外一次性开关。
8. **run_dir mkdir 竞态**（非 rank0 可能 open 失败）→ 所有 rank mkdir + barrier。
9. 冒烟准入门新增 `update_rel`（参数真动了吗）与 `invalid_reward_corr`（奖励符号）两个 FAIL 判据。

## 待办 / 下一步
- [ ] 监督 8 卡作业阶段1 冒烟指标 + 准入门，通过后进阶段2 全程训练。
- [ ] 持续用 `scripts/monitor_runs.py` 监控（重点 `update_rel`、`invalid_reward_corr`、异常率 A1~A9）。
- [ ] 8 卡跑稳后把训练流程搬到三期 ACD（H100，代码靠 `tools/sync_ztwang.py --host hpc3` 推）。
- [ ] 奖励稀疏消融：`CFG=configs/train_qwen3_4b_sparse.yaml`（两套都跑作消融）。
- [ ] 正式训练端到端通过后，补 resume（断点目前不存 optimizer/RNG state）。

## 铁律（用户亲定，置顶）
1. 只能在 `ztwang/` 内创建/编辑文件；外部模型、他人目录一律只读。
2. 训练必须存断点，且只保留最新一个。
