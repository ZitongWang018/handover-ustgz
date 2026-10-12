#!/usr/bin/env bash
# C 的补充：用 verify_rollout_order.py 覆盖三种 group_size 组合（真跑 ALFWorld 环境）。
# 因为 smoke_test.py --stage rollout 目前会在 build_token_sequence 处崩（RandomPolicy 没有
# tokenize_text），这里用同一套断言思路（tokenize=False）独立补上 g=4 / g=1 / g=3 的证据。

set -u

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thinktool

PROJ=$HOME/ztwang/selfplay
cd "$PROJ"

export PYTHONPATH="$PROJ/src:$PROJ/third_party/py310deps"
export ALFWORLD_DATA="$HOME/.cache/alfworld"
export ALFWORLD_GLIBC_COMPAT_DIR="$PROJ/third_party/glibc232"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

FAILS=0
for combo in "8 4" "8 1" "6 3"; do
  set -- $combo
  echo ""
  echo "################################################################"
  echo "### verify_rollout_order batch=$1 group_size=$2"
  echo "################################################################"
  python scripts/verify_rollout_order.py --batch "$1" --group-size "$2" --max-steps 6
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "!!! [batch=$1 g=$2] 退出码 $rc"
    FAILS=$((FAILS + 1))
  fi
done

echo ""
if [ "$FAILS" -eq 0 ]; then
  echo "### verify_order_sweep.sh: 全部通过 ✅"
else
  echo "### verify_order_sweep.sh: $FAILS 个组合失败 ❌"
fi
exit "$FAILS"
