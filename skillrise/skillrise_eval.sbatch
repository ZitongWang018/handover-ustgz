#!/bin/bash
# Deprecated compatibility notice. Use skillrise_eval_8gpu.sbatch with a fixed
# selection manifest; the former single-process CLI no longer matches the
# current 128-slot, eight-rank evaluation protocol.
#SBATCH --job-name=skr-eval-deprecated
#SBATCH --account=root
#SBATCH --partition=i64m1tga800u8ka
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=0-00:05:00
#SBATCH --output=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.out
#SBATCH --error=/hpc2hdd/home/xyu608/ztwang/logs/%x-%j.err

echo "This entry point is deprecated; use skillrise/skillrise_eval_8gpu.sbatch." >&2
exit 2
