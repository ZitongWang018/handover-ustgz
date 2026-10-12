#!/usr/bin/env bash
# C：slot/task 对齐修复的集群端到端验证（在 CPU 计算节点上跑，不需要 GPU）。
#
# 依次执行：
#   1) smoke_test.py --stage rollout --batch 8 --group-size 4 --max-steps 6
#   2) smoke_test.py --stage rollout --batch 8 --group-size 1 --max-steps 6
#   3) smoke_test.py --stage rollout --batch 6 --group-size 3 --max-steps 6
#   4) scripts/unit_test.py
#   5) scripts/verify_align_property.py   （A：属性测试）
#   6) scripts/verify_rollout_order.py    （C：独立复算真实环境返回顺序）
#   7) scripts/verify_guards.py           （B：守卫在错的时候真的会炸）
#
# 用法（登录节点上执行，用 srun 派到计算节点）：
#   bash /hpc2hdd/home/xyu608/ztwang/selfplay/env_setup/verify_align.sh
# 或
#   srun -p i64m512u --account=root -c 8 --mem=32G -t 00:40:00 \
#        bash /hpc2hdd/home/xyu608/ztwang/selfplay/env_setup/verify_align.sh

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
run() {
  local name="$1"; shift
  echo ""
  echo "################################################################"
  echo "### $name"
  echo "################################################################"
  "$@"
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "!!! [$name] 退出码 $rc"
    FAILS=$((FAILS + 1))
  fi
  return 0
}

run "smoke rollout batch=8 g=4"    python src/smoke_test.py --stage rollout --batch 8 --group-size 4 --max-steps 6
run "smoke rollout batch=8 g=1"    python src/smoke_test.py --stage rollout --batch 8 --group-size 1 --max-steps 6
run "smoke rollout batch=6 g=3"    python src/smoke_test.py --stage rollout --batch 6 --group-size 3 --max-steps 6
run "unit_test"                    python scripts/unit_test.py
run "A 属性测试"                    python scripts/verify_align_property.py
run "C1 顺序独立复算 g=4"           python scripts/verify_rollout_order.py --batch 8 --group-size 4 --max-steps 6
run "C2 顺序独立复算 g=1"           python scripts/verify_rollout_order.py --batch 8 --group-size 1 --max-steps 6
run "C3 顺序独立复算 g=3"           python scripts/verify_rollout_order.py --batch 6 --group-size 3 --max-steps 6
run "B 守卫自证"                    python scripts/verify_guards.py
run "F token 序列验证"              python scripts/verify_tokens.py

echo ""
echo "################################################################"
if [ "$FAILS" -eq 0 ]; then
  echo "### verify_align.sh: 全部通过 ✅"
else
  echo "### verify_align.sh: $FAILS 个步骤失败 ❌"
fi
echo "################################################################"
exit "$FAILS"
