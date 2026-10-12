#!/bin/bash
# ============================================================================
# Explorer-Zero 单卡冒烟 —— 在烧 8 卡 × 十几小时之前，先用一张"立刻拿得到"的卡
# 把整条 EZ 链路跑通。EZ 比主线多两个死亡点（facts 通道 / 双 adapter 梯度），
# 这两处炸了在 8 卡作业里也要等排完队才看得到。
#
# ---------------------------------------------------------------------------
# 用法（分区/时限在命令行给，脚本里不写死）：
#   # 二期 · A40 48G · debug 分区，常年有空卡，几分钟内起算（推荐先跑这个）
#   sbatch -p debug -t 00:30:00 --gres=gpu:1 \
#       ~/ztwang/selfplay/scripts/ez_smoke_1gpu.sbatch
#
#   # 二期 · A800 80G · 单卡，想顺便量真实显存峰值时用
#   sbatch -p emergency_gpu -t 02:00:00 --gres=gpu:1 \
#       ~/ztwang/selfplay/scripts/ez_smoke_1gpu.sbatch
#
#   # 三期 · H100（注意 account/qos 与二期不同，见下方【三期变体】）
#   sbatch -p acd_u -A d_tonggehuang_team --qos=normal -t 01:00:00 --gres=gpu:1 \
#       ~/ztwang/selfplay/scripts/ez_smoke_1gpu.sbatch
#
# 【三期变体】三期要改这两行 #SBATCH（本脚本内置的 account 是二期的 root）：
#     #SBATCH --account=d_tonggehuang_team
#     #SBATCH --partition=acd_u
#   并用 ZTWANG_ENV=vllm017 覆盖 conda 环境（三期无 thinktool 环境）。
#
# ---------------------------------------------------------------------------
# 冒烟通过标准（四关 + 单卡跳过关4，全过才允许开 8 卡正式作业）：
#   [关0] 梯度链        两个 adapter 各自都能接到梯度（启动自检，不依赖环境/奖励）
#   [关1] facts 通道    脚本第一段打印 "[ok] reset 后 facts 有内容（N 条）"，N>0
#   [关2] update_rel    dry-run 结尾 P0/S0 都打印 update_rel 且 **> 1e-10**
#                       ⚠️ 判据必须对着**本组奖励的 spread** 看：
#                          rel=0 且 spread>0 -> 梯度链断（FATAL）
#                          rel=0 且 spread=0 -> GRPO 组内优势全 0，正常（WARN）
#   [关2b] 张量计数     "N/M 张量真的动了" 的 N>0。比 rel 更直接：
#                       LoRA 第 0 步 B 初始全 0 ⇒ 所有 lora_A 梯度恒 0，
#                       N≈M/2 是**正常**的；N=0 才是真断链。
#   [关3] 奖励非退化    proposer 行 accept>=1，solver 行 reward 不是全 0
#   任一条不过 → 直接 exit 1，别往下开 8 卡。
#
# 规模说明：--dry-run 会**自动缩规模**（G=2 G_P=2 TPS=1 turns=12），因为论文
#   默认的 G=16 × 30 turns 在 A40 的 30 min 上限里跑不完。想加大：
#   sbatch --export=ALL,EZ_SMOKE_G=4,EZ_SMOKE_TURNS=16 ...
#   ⚠️ turns 别压得太低：prompt 要求探索 10-20 个动作后才 finish，给 3 轮的话
#      proposer 永远走不到 finish -> accept=0 -> 任务池空 -> solver 相位整段跳过，
#      冒烟会"看着跑完但其实一条真链路都没走"（实测 job 12989069 就是这么假通过的）。
# ============================================================================
#SBATCH --job-name=ez-smoke
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.out
#SBATCH --error=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.err

set -euo pipefail

# ---------- 集群自适应（二期 /hpc2hdd/home、三期 /data/user）----------
if [[ -d /data/user/xyu608 ]]; then
  CLUSTER="hpc3"
  PROJ=/data/user/xyu608/ztwang/selfplay
  ALFWORLD_DATA="${ALFWORLD_DATA:-$PROJ/data/alfworld}"
  MODEL_PATH="${ZTWANG_MODEL_PATH:-$PROJ/data/models/qwen3_4b_inst_2507}"
  CONDA_ENV="${ZTWANG_ENV:-vllm017}"
  DEPS="third_party/py311deps"
  source /share/anaconda3/etc/profile.d/conda.sh
else
  CLUSTER="hpc2"
  PROJ=$HOME/ztwang/selfplay
  ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
  MODEL_PATH="${ZTWANG_MODEL_PATH:-$HOME/thinktool/models/qwen3_4b_inst_2507}"
  CONDA_ENV="${ZTWANG_ENV:-thinktool}"
  DEPS="third_party/py310deps"
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
cd "$PROJ"
conda activate "$CONDA_ENV"

export PYTHONPATH="$PROJ/src:$PROJ/$DEPS:${PYTHONPATH:-}"
export ZTWANG_MODEL_PATH="$MODEL_PATH"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[[ "$CLUSTER" == "hpc2" ]] && export ALFWORLD_GLIBC_COMPAT_DIR="$PROJ/third_party/glibc232"
[[ -n "${CONDA_PREFIX:-}" ]] && export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

CFG=${CFG:-configs/ez_qwen3_4b.yaml}

echo "=== job=$SLURM_JOB_ID cluster=$CLUSTER node=$(hostname) date=$(date) ==="
echo "PROJ=$PROJ CFG=$CFG conda=$CONDA_ENV"
echo "model=$ZTWANG_MODEL_PATH"
echo "alfworld_data=$ALFWORLD_DATA"
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total --format=csv
python -V
df -h "$HOME" | tail -1

# ---------- [0/3] 离线单测（纯 CPU，秒级；先证明核心逻辑没被改坏）----------
echo "--- [0/3] EZ 离线单测 ---"
python scripts/ez_unit_test.py

# ---------- [1/3] facts 通道预检（EZ 的命根子，没有就直接停）----------
echo "--- [1/3] facts 通道预检 ---"
if ! python scripts/ez_check_env_facts.py; then
  echo "!!! textworld 的 infos 里没有 facts —— EZ 状态匹配会退化成观测哈希，奖励是废的 !!!"
  echo "!!! 先解决 textworld>=1.5 或 alfworld_env 的 request_infos，再来冒烟          !!!"
  exit 1
fi

# ---------- [2/3] dry-run：1 step proposer + 1 step solver ----------
echo "--- [2/3] EZ dry-run（自动缩规模 G=${EZ_SMOKE_G:-2} G_P=${EZ_SMOKE_G_P:-2} turns=${EZ_SMOKE_TURNS:-12}）---"
export MASTER_ADDR=$(hostname)
export NCCL_DEBUG=WARN
torchrun --nproc_per_node=1 --nnodes=1 \
    --master_addr="$MASTER_ADDR" --master_port=29531 \
    src/ez_train.py --config "$CFG" --dry-run

echo "--- [2/3] dry-run 通过，显存峰值见下 ---"
nvidia-smi --query-gpu=memory.peak --format=csv 2>/dev/null || true

# ---------- [3/3] 产物自检：参数真的动了？奖励不是退化 0？ ----------
echo "--- [3/3] 产物自检 ---"
if ! python scripts/ez_check_smoke_ok.py --config "$CFG"; then
  echo "!!! 冒烟自检未通过：update_rel≈0 或奖励全 0，别开 8 卡 !!!"
  exit 1
fi

echo "=== done: $(date)  → 单卡冒烟四关全过，可以开 8 卡（先跑 ez_smoke_8gpu.sbatch）==="
