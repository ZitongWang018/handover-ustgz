#!/bin/bash
# 独立评测：加载任意 EZ 断点，在官方 valid_unseen 上算 SR（不改权重、不训练）。
#
# 用途：
#   1) 补训练期间漏掉的评测点（如 outer0 崩在评测那一步）
#   2) 随时对任意 checkpoints/ckpt-* 或 ckpt/ 做独立评测
#
# 用法：
#   sbatch --export=ALL,ZTWANG_MODEL_PATH=...,EZ_CKPT=<ckpt 目录> scripts/ez_eval_ckpt.sbatch
#   可选：EZ_N=16 EZ_SAMPLES=2 EZ_SELECT=head EZ_BACKENDS="vllm hf"
#
# ★ 默认用 --select head：与训练中 mid_eval_solver 取的是**同一批** 16 个任务
#   （ez_train 是 eval_game_files[:_n]，不 shuffle），所以补出来的点能和后面的点连成曲线。
#SBATCH --job-name=ez-evalckpt
#SBATCH --account=root
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=0-02:00:00
#SBATCH --output=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.out
#SBATCH --error=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.err

set -uo pipefail
PROJ="$HOME/ztwang/selfplay"
cd "$PROJ"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thinktool

export PYTHONPATH="$PROJ/src:$PROJ/third_party/py310deps:${PYTHONPATH:-}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
export ZTWANG_MODEL_PATH="${ZTWANG_MODEL_PATH:-$HOME/thinktool/models/qwen3_4b}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
mkdir -p logs eval_out

: "${EZ_CKPT:?Set EZ_CKPT to a checkpoint dir containing solver/}"
# 允许直接给 run 目录，自动补 /ckpt
if [ ! -d "$EZ_CKPT/solver" ] && [ -d "$EZ_CKPT/ckpt/solver" ]; then
  EZ_CKPT="$EZ_CKPT/ckpt"
fi
[ -d "$EZ_CKPT/solver" ] || { echo "FATAL: $EZ_CKPT 下没有 solver/"; exit 2; }

N="${EZ_N:-16}"
SAMPLES="${EZ_SAMPLES:-2}"
SELECT="${EZ_SELECT:-head}"
BATCH="${EZ_BATCH:-$N}"
BACKENDS="${EZ_BACKENDS:-vllm hf}"
TAG="$(basename "$(dirname "$EZ_CKPT")")-$(basename "$EZ_CKPT")"
export TAG
echo "job=${SLURM_JOB_ID} ckpt=$EZ_CKPT n=$N samples=$SAMPLES select=$SELECT batch=$BATCH backends=[$BACKENDS]"
nvidia-smi -L

GPU=0
for B in $BACKENDS; do
  OUT="eval_out/${TAG}-${B}-n${N}s${SAMPLES}.json"
  echo "########## backend=$B -> $OUT (gpu$GPU) ##########"
  CUDA_VISIBLE_DEVICES=$GPU python src/eval_ez.py \
      --backend "$B" \
      --base "$ZTWANG_MODEL_PATH" \
      --ckpt "$EZ_CKPT" \
      --n "$N" --samples "$SAMPLES" --select "$SELECT" --batch "$BATCH" \
      --seed 42 --out "$OUT" \
      > "logs/eval_${TAG}_${B}.log" 2>&1
  echo "[$B] exit=$?"
  GPU=$((GPU + 1))
done

echo "=================== SUMMARY ==================="
python3 - <<'EOF'
import json, glob, os
for f in sorted(glob.glob("eval_out/" + os.environ.get("TAG", "*") + "*.json")):
    d = json.load(open(f))
    print(f"{os.path.basename(f)}")
    print(f"   SR={d['sample_level_success_mean']:.4f} ± {d['sample_level_success_std']:.4f}"
          f"  n_tasks={d['n_tasks']} samples={d['sample_count']} select={d.get('select')}"
          f"  backend={d.get('backend')}")
    if d.get("decode_effective"):
        print(f"   实际采样参数={d['decode_effective']}")
EOF
echo "=== 逐文件结果 ==="
for f in eval_out/*.json; do
  python3 -c "
import json,sys
d=json.load(open('$f'))
print('$f', 'SR=%.4f'%d['sample_level_success_mean'], 'n=%d'%d['n_tasks'],
      'won=%d/%d'%(round(d['sample_level_success_mean']*d['sample_count']), d['sample_count']))
" 2>/dev/null
done
