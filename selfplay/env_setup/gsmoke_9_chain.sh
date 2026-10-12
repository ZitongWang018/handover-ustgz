#!/bin/bash
#SBATCH --job-name=gsmoke
#SBATCH --partition=i64m1tga800u
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/hpc2hdd/home/xyu608/ztwang/selfplay/logs/gsmoke_chain_%j.out
#SBATCH --error=/hpc2hdd/home/xyu608/ztwang/selfplay/logs/gsmoke_chain_%j.err
#
# GPU 全链路接力：一次拿到 GPU 就把所有 GPU 步骤跑完（分区超订，避免反复排队）。
#   gsmoke_2_all      -> stage all(fa2->sdpa) + 计时/显存探针
#   gsmoke_3_train    -> torchrun 4 iter + 断点"只留一个"验证
#   gsmoke_3b_dryrun  -> torchrun --dry-run --max-iters 1
#   gsmoke_3c_stepcount -> 进程内数 optimizer.step()，证明 dry-run 不更新参数
#   gsmoke_4_throughput -> n_tasks=2/group=4/steps=30/new=96 跑 2 iter 测吞吐与显存
# 单步失败不阻断后续（不加 set -e）。
set -uo pipefail
PROJ=$HOME/ztwang/selfplay
LOGD=$PROJ/logs; mkdir -p "$LOGD"
RUN=$PROJ/runs/qwen3-4b-alfworld-grpo
echo "########## GPU CHAIN START $(date) node=$(hostname) job=${SLURM_JOB_ID:-?} ##########"
for step in gsmoke_2_all gsmoke_3_train gsmoke_3b_dryrun gsmoke_3c_stepcount gsmoke_4_throughput; do
  echo; echo "########## >>> $step  $(date) ##########"
  t0=$(date +%s)
  bash "$PROJ/env_setup/$step.sh" 2>&1 | tee "$LOGD/$step.log"
  rc=${PIPESTATUS[0]}
  echo "########## <<< $step rc=$rc  ($(($(date +%s)-t0))s) ##########"
done
echo; echo "########## runs/ 目录最终状态 ##########"
ls -la "$RUN" 2>&1
echo "########## GPU CHAIN DONE $(date) ##########"
