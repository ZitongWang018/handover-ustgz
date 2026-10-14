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
exec bash "$PROJ/skillrise/run_eval_8gpu.sh"
