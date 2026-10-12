#!/bin/bash
# 第二轮探针：验证 DeepSeek V4 Pro 基线能否在 ALFWorld 上真正打通「生成 -> 解析 -> 执行 -> 判分」闭环。
# 目的：确认 harness 不是只会输出 0% 的空壳（scripted/echo 本来就弱，不能用来判 harness 对不对）。
#
# 用法：
#   srun -p i64m512u --account=root -c 8 --mem=32G -t 00:40:00 bash ~/ztwang/selfplay/env_setup/probe_deepseek.sh
set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thinktool

PROJ=$HOME/ztwang/selfplay
cd "$PROJ"

export PYTHONPATH="$PROJ/src:$PROJ/third_party/py310deps"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
export ALFWORLD_GLIBC_COMPAT_DIR="$PROJ/third_party/glibc232"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

N=${N:-6}
MAX_STEPS=${MAX_STEPS:-25}
CONC=${CONC:-5}

echo "=== node=$(hostname) date=$(date) N=$N MAX_STEPS=$MAX_STEPS CONC=$CONC ==="

python src/eval_alfworld.py \
    --agent deepseek \
    --api-model deepseek-v4-pro \
    --effort max \
    --api-max-tokens 2048 \
    --concurrency "$CONC" \
    --split eval_out_of_distribution \
    --n "$N" --batch "$N" --max-steps "$MAX_STEPS" --tag probe2

echo "=== 逐条动作序列（人工核对解析是否正确）==="
python - <<'PY'
import json, glob, os
fs = sorted(glob.glob(os.path.expanduser("~/ztwang/selfplay/runs/eval/*probe2.json")))
print("files:", fs)
if fs:
    d = json.load(open(fs[-1]))
    print("summary:", json.dumps(d["summary"], ensure_ascii=False))
    for t in d["results"]:
        print(f"\n### {t['task_id']} [{t['task_type']}] won={t['won']} steps={t['steps']} "
              f"invalid={t['invalid']} fmt_bad={t['format_bad']} repeats={t['repeats']}")
        for i, a in enumerate(t["trajectory_actions"]):
            print(f"   {i+1:2d}. {a}")
        print("   last_obs:", t["last_observation"].replace("\n", " ")[:200])
PY

echo "=== PROBE2 DONE: $(date) ==="
