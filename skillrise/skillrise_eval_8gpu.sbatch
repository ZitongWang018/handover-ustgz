#!/bin/bash
# Eight-GPU Slurm template for the current HKUST cluster. Override partition,
# gres, CPUs and memory via sbatch CLI on other Slurm installations.
#SBATCH --job-name=skr-eval
#SBATCH --account=root
#SBATCH --partition=i64m1tga800u8ka
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=0-06:00:00
#SBATCH --output=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.out
#SBATCH --error=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.err

set -euo pipefail
PROJ="${SKR_PROJECT_ROOT:-$HOME/ztwang}"
export SKR_PROJECT_ROOT="$PROJ"
# ★ 必须指定 python 解释器：PATH 里的 `python` 是 miniconda base 的 3.13，
#   而依赖（textworld/alfworld/py310deps）都是 3.10 的，用错解释器会抛
#   "should not try to import numpy from its source directory"（job 13049955/13049956 就是这么挂的）。
export SKR_CONDA_BASE="${SKR_CONDA_BASE:-$HOME/miniconda3}"
export SKR_CONDA_ENV="${SKR_CONDA_ENV:-thinktool}"
export SKR_PYTHON="${SKR_PYTHON:-$SKR_CONDA_BASE/envs/$SKR_CONDA_ENV/bin/python}"
export ZTWANG_MODEL_PATH="${ZTWANG_MODEL_PATH:-$HOME/thinktool/models/qwen3_4b_inst_2507}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
echo "[skr-eval-8gpu] python=$SKR_PYTHON model=$ZTWANG_MODEL_PATH data=$ALFWORLD_DATA"
exec bash "$PROJ/skillrise/run_eval_8gpu.sh"
