#!/bin/bash
# P0：vLLM PoC —— 验证 vLLM 能在 A800/A40 上加载 qwen3_4b 并跑 token-in/token-out 生成。
#SBATCH --job-name=vllm-poc
#SBATCH --account=root
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=/hpc2hdd/home/xyu608/ztwang/logs/vllm-poc-%j.out
#SBATCH --error=/hpc2hdd/home/xyu608/ztwang/logs/vllm-poc-%j.err

set -x
echo "=== job=$SLURM_JOB_ID node=$(hostname) ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
~/ztwang/vllm_venv/bin/python -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__,'cuda',torch.version.cuda)"

~/ztwang/vllm_venv/bin/python ~/ztwang/selfplay/scripts/ez_vllm_poc.py \
    --model ~/thinktool/models/qwen3_4b \
    --n 32 --max-new-tokens 96 \
    --dump ~/ztwang/logs/vllm_poc_result.json
echo "=== POC_DONE rc=$? ==="
