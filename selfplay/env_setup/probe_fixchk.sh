#!/bin/bash
# 修 slot/task 置换 bug 后的端到端验证：
#   1) expert  OOD n=8   —— 对齐守卫下必须跑通
#   2) random  OOD n=8   —— 对齐守卫下必须跑通
#   3) deepseek OOD n=6  —— 逐条核对 task_id 与动作语义一致（复现老 bug 已消失）
# 用法： srun ... bash ~/ztwang/selfplay/env_setup/probe_fixchk.sh
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

echo "=== node=$(hostname) date=$(date) ==="

echo "--- [1/3] expert OOD n=8 ---"
python src/eval_alfworld.py --agent expert --split eval_out_of_distribution \
    --n 8 --batch 8 --max-steps 50 --tag fixchk

echo "--- [2/3] random OOD n=8 ---"
python src/eval_alfworld.py --agent random --split eval_out_of_distribution \
    --n 8 --batch 8 --max-steps 20 --tag fixchk

echo "--- [3/3] deepseek OOD n=6 ---"
python src/eval_alfworld.py --agent deepseek \
    --api-model deepseek-v4-pro --effort max \
    --api-max-tokens 2048 --concurrency 5 \
    --split eval_out_of_distribution --n 6 --batch 6 --max-steps 25 --tag fixchk

echo "=== DeepSeek task_id <-> 动作序列 一致性核对 ==="
python - <<'PY'
import json, os

f = os.path.expanduser("~/ztwang/selfplay/runs/eval/eval_out_of_distribution_deepseek_fixchk.json")
d = json.load(open(f))
print("file:", f)
print("summary:", json.dumps(d["summary"], ensure_ascii=False))
for t in d["results"]:
    seg = t["task_id"].split("/")[0]
    toks = seg.split("-")
    ents = [x for x in toks[1:] if x and x != "None" and not x.isdigit()]
    acts = t["trajectory_actions"]
    text = " | ".join(acts)
    nospace = text.replace(" ", "")
    hits = [e for e in ents if e.lower() in text.lower() or e.lower() in nospace.lower()]
    print(f"\n### {t['task_id']}  [{t['task_type']}] won={t['won']} steps={t['steps']}")
    print(f"    entities={ents}  hit={hits}  -> {'OK' if len(hits) == len(ents) else 'CHECK'}")
    for i, a in enumerate(acts):
        print(f"    {i+1:2d}. {a}")
PY

echo "=== FIXCHK DONE: $(date) ==="
