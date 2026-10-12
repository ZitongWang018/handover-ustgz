#!/bin/bash
# ============================================================================
# selfplay 运行环境一键打通（二期/三期通用）
# ----------------------------------------------------------------------------
# 做三件事，**全部只写 ~/ztwang 内**，不动任何外部 conda 环境：
#   1) 造 GLIBC 兼容层：patched libdownward.so + __libc_single_threaded 垫片
#   2) 用 `pip install --target` 把 textworld / alfworld / gym 装进自己的目录
#   3) 把全部 wheel 缓存到 third_party/wheels/（三期无外网，靠这批 wheel 离线安装）
#
# 用法：
#   bash env_setup/setup_deps.sh                 # 二期，目标环境 thinktool(py3.10)
#   TARGET_ENV=vllm017 DEPS_TAG=py311deps bash env_setup/setup_deps.sh   # 三期
#
# 变量：
#   SOURCE_ENV  提供原始 libdownward.so 的只读环境（默认 agentenv-alfworld）
#   TARGET_ENV  训练/评测用的环境（默认 thinktool）
#   DEPS_TAG    --target 目录后缀（默认按 TARGET_ENV 的 python 版本推）
#   OFFLINE=1   只装 wheel 缓存里的包（三期用）
# ============================================================================
set -euo pipefail

PROJ=${PROJ:-$HOME/ztwang/selfplay}
SOURCE_ENV=${SOURCE_ENV:-agentenv-alfworld}
TARGET_ENV=${TARGET_ENV:-thinktool}
OFFLINE=${OFFLINE:-0}

CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
[[ -f "$CONDA_SH" ]] || CONDA_SH=/share/anaconda3/etc/profile.d/conda.sh

echo "=== selfplay 环境打通 ==="
echo "PROJ=$PROJ  SOURCE_ENV=$SOURCE_ENV  TARGET_ENV=$TARGET_ENV  OFFLINE=$OFFLINE"

# ---------------------------------------------------------------- 0) 兼容层来源
LIBDW=""
for cand in \
    "$HOME/miniconda3/envs/$SOURCE_ENV/lib/python3.9/site-packages/fast_downward/libdownward.so" \
    "/share/anaconda3/envs/$SOURCE_ENV/lib/python3.9/site-packages/fast_downward/libdownward.so"; do
  [[ -f "$cand" ]] && LIBDW="$cand" && break
done
if [[ -z "$LIBDW" ]]; then
  echo "[!] 找不到原始 libdownward.so（SOURCE_ENV=$SOURCE_ENV）。"
  echo "    先在一个 py3.9 环境里装 textworld：pip install textworld==1.6.2 fast_downward_textworld==20.6.4"
  exit 1
fi
echo "原始库: $LIBDW"

# ---------------------------------------------------------------- 1) 造兼容层
COMPAT_DIR="$PROJ/third_party/glibc232"
mkdir -p "$COMPAT_DIR"
python3 "$PROJ/env_setup/patch_libdownward.py" --src "$LIBDW" --out-dir "$COMPAT_DIR"

# ---------------------------------------------------------------- 2) 目标环境 + 依赖
source "$CONDA_SH"
conda activate "$TARGET_ENV"
PYVER=$(python -c 'import sys;print(f"py{sys.version_info.major}{sys.version_info.minor}")')
DEPS_TAG=${DEPS_TAG:-${PYVER}deps}
D="$PROJ/third_party/$DEPS_TAG"
WHEELS="$PROJ/third_party/wheels"
mkdir -p "$D" "$WHEELS"
echo "目标解释器: $(python -V)  ->  $D"

PKGS=(textworld==1.6.2 alfworld==0.3.3 fast_downward_textworld==20.6.4 gym==0.26.2)

if [[ "$OFFLINE" == "1" ]]; then
  PIP_ARGS=(--no-index --find-links "$WHEELS")
  echo "[offline] 只用 $WHEELS 里的 wheel"
else
  PIP_ARGS=()
  # 顺便把 wheel 缓存下来给三期用
  pip download --dest "$WHEELS" --no-deps "${PKGS[@]}" || true
fi

pip install "${PIP_ARGS[@]}" --target "$D" --no-deps --upgrade "${PKGS[@]}" || {
  echo "[warn] 一次性安装失败，逐个来（有些包没有 wheel，要退到 sdist 或复制）"
  for p in "${PKGS[@]}"; do
    pip install "${PIP_ARGS[@]}" --target "$D" --no-deps --upgrade "$p" || echo "  [skip] $p"
  done
}

# gym 0.26.2 在 PyPI 上经常没有匹配的 wheel，退路是直接复制（纯 Python 包）
if ! ls "$D"/gym >/dev/null 2>&1; then
  SRC_GYM=""
  for cand in "$HOME/miniconda3/envs/$SOURCE_ENV/lib/python3.9/site-packages/gym" \
              "/share/anaconda3/envs/$SOURCE_ENV/lib/python3.9/site-packages/gym"; do
    [[ -d "$cand" ]] && SRC_GYM="$cand" && break
  done
  if [[ -n "$SRC_GYM" ]]; then
    echo "[fallback] 从只读环境复制 gym: $SRC_GYM"
    cp -r "$SRC_GYM" "$D/"
  else
    echo "[!] gym 装不上也复制不到，请手工处理"
  fi
fi

# ---------------------------------------------------------------- 3) 自检
export PYTHONPATH="$PROJ/src:$D:${PYTHONPATH:-}"
export ALFWORLD_GLIBC_COMPAT_DIR="$COMPAT_DIR"
echo "=== 自检 ==="
python - <<'PY'
import sys
sys.path.insert(0, __import__("os").path.join(__import__("os").path.expanduser("~"), "ztwang/selfplay/src"))
from alfworld_compat import install
install()
import fast_downward
lib = fast_downward.load_lib(); fast_downward.close_lib(lib)
print("  libdownward load: OK")
import numpy, textworld
print("  numpy:", numpy.__version__, "| textworld:", textworld.__version__)
import alfworld
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
print("  alfworld import: OK")
PY

echo
echo "=== 完成 ==="
echo "在 sbatch / 交互里这样用："
echo "  export PYTHONPATH=\"$PROJ/src:$D:\$PYTHONPATH\""
echo "  export ALFWORLD_DATA=\${ALFWORLD_DATA:-\$HOME/.cache/alfworld}"
echo "  export ALFWORLD_GLIBC_COMPAT_DIR=\"$COMPAT_DIR\""
