#!/bin/bash
# 快速通道（单卡版）：只评测，1 张卡够用；避开 8 卡分区的长队。
# 快速通道：用**已三次成功跑通**的那套依赖装配 + 自研 harness（skillrise/eval_quick.py），
# 配合上游版的固定 128 槽清单，先把两个评测跑出结果。
# 不修改上游版（eval_alfworld_prompting.py / run_eval_8gpu.sh / *_8gpu.sbatch）任何一行。
#
#   sbatch --export=ALL,EZ_STYLE=react,EZ_SELECTION_MANIFEST=$PWD/skillrise/out/selection_128.json \
#          skillrise/eval_quick.sbatch
#SBATCH --job-name=skr-quick1g
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
PROJ="$HOME/ztwang"; cd "$PROJ"
PY="$HOME/miniconda3/envs/thinktool/bin/python"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
export ZTWANG_MODEL_PATH="${ZTWANG_MODEL_PATH:-$HOME/thinktool/models/qwen3_4b_inst_2507}"
# ★ 三次成功跑通过的那套装配：py310deps 直接优先（其中 numpy 已改名禁用）
export PYTHONPATH="$PROJ/selfplay/src:$PROJ/selfplay/third_party/py310deps"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
mkdir -p logs skillrise/out
STYLE="${EZ_STYLE:-react}"
SEL="${EZ_SELECTION_MANIFEST:?set EZ_SELECTION_MANIFEST}"
OUT="skillrise/out/quick_${STYLE}_$(date +%m%d_%H%M%S).json"
echo "job=$SLURM_JOB_ID style=$STYLE py=$PY model=$ZTWANG_MODEL_PATH sel=$SEL"
"$PY" -c "import numpy,textworld;print('numpy',numpy.__file__);print('tw',textworld.__file__)"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "$PY" skillrise/eval_quick.py --style "$STYLE" \
  --selection-manifest "$SEL" --base "$ZTWANG_MODEL_PATH" \
  --samples "${EZ_SAMPLES:-1}" --batch "${EZ_BATCH:-32}" \
  --max-steps "${EZ_MAX_STEPS:-30}" --max-new-tokens "${EZ_MAX_NEW:-1024}" \
  --temperature "${EZ_TEMPERATURE:-0.7}" --top-p "${EZ_TOP_P:-0.95}" --out "$OUT"
echo "[quick] exit=$? out=$OUT"
