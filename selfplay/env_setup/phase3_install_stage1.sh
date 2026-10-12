#!/bin/bash
# phase3_install_stage1.sh -- 三期(ACD) 第一步安装：把 ALFWorld 依赖离线装进 ~/ztwang 下。
#
# 在【三期登录节点】执行。硬边界：
#   * 只写 /data/user/xyu608/ztwang/** 与 /data/user/xyu608/.cache/alfworld/**
#   * 绝不 `pip install`（无 --target）进 /share/anaconda3/envs/**（只读、别人在用）
#   * 只用 `pip install --target <ztwang 下的目录>`，运行期靠 PYTHONPATH 注入
#
# 用法：  bash ~/ztwang/selfplay/env_setup/phase3_install_stage1.sh /share/anaconda3/envs/vllm017/bin/python
#   参数 = 目标 conda env 的 python（决定 wheel 集：3.11->wheels, 3.12->wheels312）
set -u

PYEXE="${1:-/share/anaconda3/envs/vllm017/bin/python}"
ROOT=/data/user/xyu608/ztwang/selfplay
B=$ROOT/third_party

[ -x "$PYEXE" ] || { echo "no python: $PYEXE"; exit 1; }
PYVER=$("$PYEXE" -c 'import sys;print("%d.%d"%sys.version_info[:2])')

case "$PYVER" in
  3.11) W=$B/wheels;    T=$B/py311deps ;;
  3.12) W=$B/wheels312; T=$B/py312deps ;;
  *) echo "unsupported python $PYVER (需 3.11 或 3.12)"; exit 1 ;;
esac

echo "=== phase3 Stage-1 install ==="
echo "python : $PYVER  ($PYEXE)"
echo "wheels : $W"
echo "target : $T"
echo "=============================="

[ -d "$W" ] || { echo "缺少 wheel 目录 $W —— 先跑 phase3_push_stage1.sh"; exit 2; }
mkdir -p "$T"

echo "--- pip install --target (离线, --no-deps) ---"
"$PYEXE" -m pip install --no-cache-dir --no-index --find-links "$W" --no-deps \
  -r "$B/requirements_env_common.txt" --target "$T"
RC=$?
echo "pip rc=$RC"
[ $RC -eq 0 ] || { echo "pip 安装失败"; exit $RC; }

echo
echo "--- 纯 import 级自检 (只在登录节点, 不构造环境/不 step/不 srun) ---"
export SELFPLAY_REPO="$ROOT"
export SELFPLAY_DEPS="$T"
export PYTHONPATH="$ROOT/src:$T${PYTHONPATH:+:$PYTHONPATH}"
export ALFWORLD_DATA=/data/user/xyu608/.cache/alfworld
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PYEXE")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

RES=$ROOT/env_setup/results
mkdir -p "$RES"
"$PYEXE" "$ROOT/env_setup/phase3_verify_imports.py" | tee "$RES/phase3_verify_imports.log"
RC=${PIPESTATUS[0]}
echo "verify rc=$RC"

echo
if [ $RC -eq 0 ]; then
  echo "### STAGE-1 INSTALL OK ###"
else
  echo "### STAGE-1 INSTALL FAILED (rc=$RC) —— 见上方 [FAIL] 行 ###"
fi
exit $RC
