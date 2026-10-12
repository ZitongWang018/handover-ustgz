#!/bin/bash
# ============================================================================
# Explorer-Zero **8 卡 DDP** 冒烟 —— P0 验收（EZ_DDP_PLAN.md §7/§8）
#
# 单卡冒烟（ez_smoke_1gpu.sbatch）过的是"链路通不通"；这一发过的是
# **"DDP 接线对不对"**，两者查的问题几乎不重叠：
#   · 单卡永远查不出"policy 拿了 raw.module 绕过 DDP forward hook"
#   · 单卡永远查不出"某个 rank 少做一次 backward"（会静默卡到 30min timeout）
#   · 单卡永远查不出"两个 adapter 没同时 trainable 就 wrap"（梯度静默不同步）
#
# ---------------------------------------------------------------------------
# 用法（分区/账号在命令行给，脚本里不写死）：
#
#   # ① 二期 · debug 分区恰好有一台自带 8 张 A40 的机器（gpu3-9），30min 上限。
#   #    够跑完这一发（实测单卡 dry-run 5.7min + 环境/模型加载约 4min），
#   #    而且**立刻起算**，是验 DDP 接线最快的路子。
#   sbatch -p debug -t 00:28:00 --gres=gpu:8 \
#       ~/ztwang/selfplay/scripts/ez_smoke_8gpu.sbatch
#
#   # ①b 二期 · debug 分区 **2 卡**变体（验 DDP 接线的最小成本；实测 2:42 跑完）
#   #     ⚠️ debug 分区每 GPU 最多 8 核 ⇒ 2 卡最多 16 核、内存也要一起降
#   #     ⚠️ turns 必须 ≥16：prompt 要求"探索 10-20 个动作后才 finish"，
#   #        turns=12 时 proposer 永远走不到 finish ⇒ accept=0 ⇒ 关3 FATAL
#   #        （2026-10-12 job 12989603 实测：2 卡 DDP 全部正确，唯一失败项就是它）
#   sbatch -p debug -t 00:28:00 --gres=gpu:2 \
#       --cpus-per-task=16 --mem=240G \
#       --export=ALL,NPROC=2,EZ_SMOKE_TPS=2,EZ_SMOKE_TURNS=16 \
#       ~/ztwang/selfplay/scripts/ez_smoke_8gpu.sbatch
#
#   # ② 二期 · 正式 8 卡分区（排队久，但时限长；A800 版换 800u8ka）
#   sbatch -p i64m1tga40u8ka -t 02:00:00 --gres=gpu:8 \
#       ~/ztwang/selfplay/scripts/ez_smoke_8gpu.sbatch
#
#   # ③ 三期 · H100（account/qos 与二期不同，且 #SBATCH 里的 output 路径是
#   #    二期家目录，必须用 -o/-e 覆盖成三期的）
#   sbatch -p acd_u -A d_tonggehuang_team --qos=normal -t 02:00:00 --gres=gpu:8 \
#       -o /data/user/xyu608/ztwang/logs/%x-%j.out \
#       -e /data/user/xyu608/ztwang/logs/%x-%j.err \
#       ~/ztwang/selfplay/scripts/ez_smoke_8gpu.sbatch
#   （三期还需先备好 data/alfworld 与 data/models/qwen3_4b_inst_2507）
#
# ⚠️ 8 卡排队时间远大于计算时间（二期 ~26h）。所以这一发**必须一次做对**：
#    提交前先本地确认 `python scripts/ez_check_smoke_selftest.py` 全绿。
#
# ---------------------------------------------------------------------------
# 通过标准（在单卡之上多一关；判据细节见 ez_check_smoke_ok.py）：
#   [关0] 两个 adapter 各自都能接到梯度（grad_chain_selfcheck）
#   [关1] facts 通道存在
#   [关2/2b] update_rel > 1e-10 且「N/M 张量真的动了」的 N > 0
#   [关3] proposer accept>=1，solver 奖励不是全 0
#   [关4] 每个 rank 的 train_log.rank*.jsonl 都存在且非空
#   [关4] 各 rank 的 max(update_rel) 同量级、日志里的 world_size == 本发进程数
#   ★ 判"参数没动"的最终依据是**真实 backward grad_norm**（GN_RE），不是 update_rel：
#        grad_norm=0（奖励全同分 ⇒ 优势全 0）  -> 不动是正确的，不算失败
#        grad_norm>0 却一步没动                -> 真断链，FATAL
#        各 rank update_rel 全 0 但 grad_norm 也全 0 -> 一致无信号，不是没同步
#     （2026-10-12 job 12989603：旧判据在这两点上误报出两条 FATAL，把一发
#      完全正确的 2 卡 DDP 判成 FAILED —— 现已修，见 ez_check_smoke_selftest.py N/O/P）
#
# 规模：EZ_SMOKE_TPS=8 让每 rank 恰好分到 1 个 game（TPS/world=1）。
#   ⚠️ TPS 必须是 world 的倍数，否则各 rank 的 backward 次数不同 -> NCCL 死锁。
#      ez_train.py 里会自动上调到 world 的倍数并打 WARNING，但**别依赖那个兜底**。
# ============================================================================
#SBATCH --job-name=ez-smoke8
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# 64c/1T 是**二期单机的上限**（三期是 96c/2T，想用满请在命令行覆盖：
#   --cpus-per-task=96 --mem=0）。8 卡时每进程抢满 64 线程会拖慢 NCCL 同步点，
#   所以脚本里把 OMP_NUM_THREADS 压到 1 —— 真正吃 CPU 的是 ALFWorld 子进程。
#SBATCH --cpus-per-task=64
# ⚠️ 别写 1T：二期单机实际内存是 1024000 MB（≈1000 GiB），写 1T 会被拒
#    "Memory specification can not be satisfied"（实测 2026-10-12）。
#SBATCH --mem=900G
#SBATCH --output=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.out
#SBATCH --error=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.err

set -euo pipefail

# ---------- 集群自适应（二期 /hpc2hdd/home、三期 /data/user）----------
if [[ -d /data/user/xyu608 ]]; then
  CLUSTER="hpc3"; PROJ=/data/user/xyu608/ztwang/selfplay
  ALFWORLD_DATA="${ALFWORLD_DATA:-$PROJ/data/alfworld}"
  MODEL_PATH="${ZTWANG_MODEL_PATH:-$PROJ/data/models/qwen3_4b_inst_2507}"
  CONDA_ENV="${ZTWANG_ENV:-vllm017}"; DEPS="third_party/py311deps"
  source /share/anaconda3/etc/profile.d/conda.sh
else
  CLUSTER="hpc2"; PROJ=$HOME/ztwang/selfplay
  ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
  MODEL_PATH="${ZTWANG_MODEL_PATH:-$HOME/thinktool/models/qwen3_4b_inst_2507}"
  CONDA_ENV="${ZTWANG_ENV:-thinktool}"; DEPS="third_party/py310deps"
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
cd "$PROJ"
conda activate "$CONDA_ENV"

export PYTHONPATH="$PROJ/src:$PROJ/$DEPS:${PYTHONPATH:-}"
export ZTWANG_MODEL_PATH="$MODEL_PATH"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# ★ 8 卡时 OMP_NUM_THREADS 必须压到 1：64 核 / 8 进程，每进程抢 64 个线程会
#   把 CPU 打满，同步点（NCCL）被拖慢，8 卡能跑出比单卡还慢的成绩（P1 项）。
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# NCCL 的"各 rank 耗时不均"容忍度：ALFWorld 单步推理可能几分钟，
# torch 侧 setup_dist 已把 timeout 放宽到 30min，这里只是把 NCCL 的 watchdog 也放宽。
export NCCL_DEBUG=WARN
# ⚠️ `NCCL_ASYNC_ERROR_HANDLING` 在 NCCL ≥2.13（torch 2.x 捆绑的版本）里**已弃用**，
#    要写 `TORCH_NCCL_ASYNC_ERROR_HANDLING` 才生效；老名字只会被静默忽略。
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1   # 兼容老版本 NCCL，留着无害

CFG=${CFG:-configs/ez_qwen3_4b.yaml}
NPROC=${NPROC:-8}
export EZ_SMOKE_G=${EZ_SMOKE_G:-2}
export EZ_SMOKE_G_P=${EZ_SMOKE_G_P:-2}
export EZ_SMOKE_TURNS=${EZ_SMOKE_TURNS:-12}
export EZ_SMOKE_TPS=${EZ_SMOKE_TPS:-$NPROC}

echo "=== job=$SLURM_JOB_ID cluster=$CLUSTER node=$(hostname) date=$(date) ==="
echo "PROJ=$PROJ CFG=$CFG conda=$CONDA_ENV NPROC=$NPROC"
echo "model=$ZTWANG_MODEL_PATH"
echo "alfworld_data=$ALFWORLD_DATA"
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total --format=csv
python -V
df -h "$HOME" | tail -1

# ---------- [0/3] 离线单测（纯 CPU）----------
echo "--- [0/3a] EZ 离线单测 ---"
python scripts/ez_unit_test.py
echo "--- [0/3b] 冒烟判读逻辑自测（不需要 GPU）---"
python scripts/ez_check_smoke_selftest.py

# ---------- [1/3] facts 通道预检（只要 rank0 跑一次，纯 CPU 侧）----------
echo "--- [1/3] facts 通道预检 ---"
if ! python scripts/ez_check_env_facts.py; then
  echo "!!! textworld 的 infos 里没有 facts —— EZ 状态匹配会退化成观测哈希，奖励是废的 !!!"
  exit 1
fi

# ---------- [2/3] 8 进程 dry-run ----------
echo "--- [2/3] EZ dry-run 8 卡（G=$EZ_SMOKE_G G_P=$EZ_SMOKE_G_P TPS=$EZ_SMOKE_TPS turns=$EZ_SMOKE_TURNS）---"
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
torchrun --nproc_per_node="$NPROC" --nnodes=1 \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    src/ez_train.py --config "$CFG" --dry-run
echo "--- [2/3] dry-run 完成，各卡显存峰值 ---"
nvidia-smi --query-gpu=index,memory.peak --format=csv 2>/dev/null || true

# ---------- [3/3] 产物自检（含 8 卡专有关4）----------
echo "--- [3/3] 产物自检 ---"
D=$(ls -dt runs/*/ | head -1)
echo "run_dir=$D"
ls -la "$D" || true
for f in "$D"train_log.rank*.jsonl; do
  [ -e "$f" ] || continue
  echo "  $(basename "$f"): $(wc -l < "$f") 行"
done
if ! python scripts/ez_check_smoke_ok.py --config "$CFG" --run-dir "$D"; then
  echo "!!! 8 卡冒烟自检未通过 —— 别开正式作业 !!!"
  exit 1
fi

echo "=== done: $(date) → 8 卡 P0 验收通过 ==="
