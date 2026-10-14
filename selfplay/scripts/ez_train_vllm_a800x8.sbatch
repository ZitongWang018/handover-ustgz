#!/bin/bash
# EZ 工程加速版：vLLM 采样后端 + 8 卡 A800 正式训练。
#
# 与 ez_train_a800x8.sbatch 的差别**只有一行**：ZTWANG_ROLLOUT_BACKEND=vllm。
# 算法（M_P/M_S/G/G_P/loss/奖励/组归一化）一字未改。
#
# 用法：
#   sbatch --export=ALL,ZTWANG_MODEL_PATH=... scripts/ez_train_vllm_a800x8.sbatch
# 冒烟（1+1 步、G=16、TPS=8 自动、12 turns）：
#   sbatch --export=ALL,ZTWANG_MODEL_PATH=...,EZ_EXTRA=--dry-run scripts/ez_train_vllm_a800x8.sbatch
#SBATCH --job-name=ez-vllm-train
#SBATCH --account=root
#SBATCH --partition=i64m1tga800u8ka
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --time=7-00:00:00
#SBATCH --output=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.out
#SBATCH --error=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.err

set -euo pipefail
: "${ZTWANG_MODEL_PATH:?Set ZTWANG_MODEL_PATH explicitly to the intended Qwen3-4B base checkpoint}"
PROJ="$HOME/ztwang/selfplay"
cd "$PROJ"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${ZTWANG_ENV:-thinktool}"

export PYTHONPATH="$PROJ/src:$PROJ/third_party/py310deps:${PYTHONPATH:-}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
export EZ_RUN_OUTPUT_DIR="${EZ_RUN_OUTPUT_DIR:-$HOME/jhspoolers/ez_runs}"
export EZ_SWANLAB_MODE="${EZ_SWANLAB_MODE:-auto}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN

# ★★ 加速开关：采样走 vLLM（连续批处理 + 前缀缓存 + LoRA 热刷）
export ZTWANG_ROLLOUT_BACKEND=vllm
export ZTWANG_VLLM_GMEM="${ZTWANG_VLLM_GMEM:-0.40}"      # 训练模型同卡共存，留 60% 给它
export ZTWANG_VLLM_MAX_SEQS="${ZTWANG_VLLM_MAX_SEQS:-64}"
export ZTWANG_VLLM_BATCHED="${ZTWANG_VLLM_BATCHED:-16384}"

mkdir -p "$EZ_RUN_OUTPUT_DIR" "$HOME/ztwang/logs"
echo "job=${SLURM_JOB_ID:-local} host=$(hostname) base=$ZTWANG_MODEL_PATH"
echo "backend=$ZTWANG_ROLLOUT_BACKEND gmem=$ZTWANG_VLLM_GMEM extra=${EZ_EXTRA:-}"
nvidia-smi -L

if [ "${EZ_SKIP_UNITTEST:-0}" != "1" ]; then
  python scripts/ez_unit_test.py
  python scripts/ez_check_smoke_selftest.py
  python scripts/ez_check_env_facts.py
fi

export MASTER_ADDR="$(hostname)"
export MASTER_PORT="$((29500 + ${SLURM_JOB_ID:-12345} % 1000))"

# ★★ 发射方式必须换：**不能用 torchrun**。
#    原因：`torchrun --nproc_per_node=8` 让 8 个 rank 都看得到全部 8 张卡，而
#    **vLLM 的 `LLM()` 没有 device 参数**，它按 `CUDA_VISIBLE_DEVICES` 自己挑卡 ——
#    8 个 rank 的引擎会全部落到物理卡 0（显存 8 倍超订 → OOM）。
#
#    试过且**否决**的做法：先 `torch.cuda.set_device(local)`，再改
#    `os.environ["CUDA_VISIBLE_DEVICES"]`。实测必崩（job 13009409）：
#        IndexError: list index out of range
#        torch/_dynamo/device_interface.py: caching_worker_device_properties["cuda"][device]
#    torch 的设备缓存已按 8 张卡建好，环境变量一改就对不上号。
#
#    改用"启动时就只暴露一张卡"：训练侧与 vLLM 子进程看到的都只有一张卡（= 物理卡 i），
#    两边一致，且**零 Python 代码改动**（`dist_info()` 本来就只读 RANK/WORLD_SIZE/LOCAL_RANK）。
#    已验收：scripts/ez_vllm_device_probe.sbatch → P2-GATE-DEVICE PASS。
NPROC="${ZTWANG_NPROC:-8}"
# ★ 续跑：作业中途挂掉（或被抢占）时，用 EZ_RESUME=<run_dir>/ckpt 直接续，
#   会恢复 adapter 权重 + optimizer + RNG + task_pool + outer/phase/step。
EXTRA_ARGS="${EZ_EXTRA:-}"
if [ -n "${EZ_RESUME:-}" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --resume $EZ_RESUME"
  echo "[launch] resume from $EZ_RESUME"
fi
PIDS=()
for i in $(seq 0 $((NPROC - 1))); do
  CUDA_VISIBLE_DEVICES=$i RANK=$i WORLD_SIZE=$NPROC LOCAL_RANK=0 \
    python src/ez_train.py --config configs/ez_qwen3_4b.yaml $EXTRA_ARGS \
    > "$HOME/ztwang/logs/ez-vllm-rank${i}-${SLURM_JOB_ID:-local}.log" 2>&1 &
  PIDS+=($!)
done
# 先用 `tail -f` 把 8 份 rank 日志汇到 slurm out，再等全部结束。
for f in "$HOME/ztwang/logs"/ez-vllm-rank*-"${SLURM_JOB_ID:-local}".log; do
  tail -n +1 -F "$f" 2>/dev/null | sed "s|^|[$(basename "$f" .log)] |" &
done
RC=0
for p in "${PIDS[@]}"; do
  wait "$p" || RC=1
done
echo "all ranks exited, rc=$RC"
exit $RC
