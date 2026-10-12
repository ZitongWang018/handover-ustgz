#!/bin/bash
# Step7: 补齐 gym/pandas/cv2 等「宿主环境原本自带、因此没进 --target 清单」的包，
#        然后在独立 py3.11 里做一次真正的「离线安装 + import」验证。
set -u
B=$HOME/ztwang/selfplay/third_party
W=$B/wheels
P311=$B/py311
PY=$HOME/miniconda3/envs/thinktool/bin/python

echo "########## 1) 补下载宿主自带但依赖链需要的 wheel ##########"
EXTRA="cloudpickle==3.1.2 opencv-python-headless==4.12.0.88 pandas==2.3.3 werkzeug==3.1.4 python-dateutil==2.9.0.post0 pytz==2025.3 six==1.17.0 tzdata==2025.3"
echo "$EXTRA" | tr ' ' '\n' > "$B/requirements_extra_host.txt"

for pair in "310 cp310" "311 cp311"; do
  set -- $pair; PYV=$1; ABI=$2
  for spec in $EXTRA; do
    "$PY" -m pip download -d "$W" --only-binary=:all: --no-deps \
        --python-version "$PYV" --implementation cp --abi "$ABI" "$spec" \
        >/dev/null 2>>"$B/wheels_missing/dl_extra_${PYV}.stderr" || echo "  miss: cp$PYV $spec"
  done
done
echo "wheels now: $(ls "$W"/*.whl | wc -l)"

echo
echo "########## 2) jericho wheel 里到底有没有 .so ##########"
unzip -l "$W"/jericho-*.whl 2>/dev/null | grep -E '\.so|\.pyd' | head || echo "  (无原生扩展 => 真纯 python)"
unzip -l "$W"/fast_downward_textworld-*.whl 2>/dev/null | grep -E 'libdownward|\.so' | head

echo
echo "########## 3) 在独立 py3.11 里做离线安装 + import 验证 ##########"
if [ ! -x "$P311/bin/python3" ]; then echo "!!! no py311"; exit 1; fi
# 用一个干净的 venv，模拟三期「全新环境 + 只有 wheels 目录」
rm -rf "$B/_testvenv311"
"$P311/bin/python3" -m venv "$B/_testvenv311"
TV=$B/_testvenv311
"$TV/bin/python" -V

echo "--- 离线安装（--no-index --find-links）---"
"$TV/bin/python" -m pip -q install --no-cache-dir --no-index --find-links "$W" \
    textworld alfworld ai2thor gym gym-notices jericho fast_downward_textworld \
    cloudpickle opencv-python-headless pandas werkzeug numpy==1.26.4
echo "install rc=$?"

echo "--- import 验证（不含 torch 分支）---"
LD_LIBRARY_PATH=$HOME/miniconda3/envs/thinktool/lib "$TV/bin/python" - <<'EOF'
import os, sys
B = os.path.expanduser("~/ztwang/selfplay/third_party")
sys.path.insert(0, os.path.join(B, "..", "src"))
os.environ["ALFWORLD_GLIBC_COMPAT_DIR"] = os.path.join(B, "glibc232")
from alfworld_compat import install
install(verbose=False)

import numpy, gym, jericho, ai2thor, cv2, pandas
print("numpy", numpy.__version__, "| gym", gym.__version__, "| jericho", jericho.__version__)
print("cv2", cv2.__version__, "| pandas", pandas.__version__)
import textworld
print("textworld", textworld.__version__)
import textworld.gym
print("textworld.gym OK")
import fast_downward
lib = fast_downward.load_lib(); fast_downward.close_lib(lib)
print("fast_downward.load_lib() -> OK")
EOF
echo "import rc=$?"

echo
echo "########## 4) 重新生成 manifest ##########"
bash "$HOME/ztwang/selfplay/env_setup/_install_step4_sdists.sh" 2>&1 | tail -2
echo "########## DONE step7 ##########"
