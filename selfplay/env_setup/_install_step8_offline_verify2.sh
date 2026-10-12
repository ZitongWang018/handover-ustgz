#!/bin/bash
# Step8: 修掉 step7 的两个问题（缺 opencv-python / pytz），重做一次干净 venv 的离线验证。
set -u
B=$HOME/ztwang/selfplay/third_party
W=$B/wheels
P311=$B/py311
PY=$HOME/miniconda3/envs/thinktool/bin/python

echo "########## 1) 补 opencv-python + pytz ##########"
for pair in "310 cp310" "311 cp311"; do
  set -- $pair; PYV=$1; ABI=$2
  for spec in "opencv-python==4.12.0.88" "pytz"; do
    "$PY" -m pip download -d "$W" --only-binary=:all: --no-deps \
        --python-version "$PYV" --implementation cp --abi "$ABI" "$spec" \
        >/dev/null 2>>"$B/wheels_missing/dl_extra2_${PYV}.stderr" || echo "  miss: cp$PYV $spec"
  done
done
ls "$W" | grep -Ei 'opencv|pytz'
echo "wheels now: $(ls "$W"/*.whl | wc -l)"

echo
echo "########## 2) 干净 venv 离线安装 ##########"
rm -rf "$B/_testvenv311" "$B/_pipcache311"
"$P311/bin/python3" -m venv "$B/_testvenv311"
TV=$B/_testvenv311
"$TV/bin/python" -m pip -q install --no-cache-dir --no-index --find-links "$W" \
    textworld ai2thor gym gym-notices jericho fast_downward_textworld \
    cloudpickle opencv-python opencv-python-headless pandas werkzeug numpy==1.26.4 pytz
echo "rc(core)=$?"
# alfworld 的依赖里有 torch / torchvision，必须由宿主环境提供（CUDA 版本要对齐），故 --no-deps
"$TV/bin/python" -m pip -q install --no-cache-dir --no-index --find-links "$W" --no-deps alfworld
echo "rc(alfworld --no-deps)=$?"

echo
echo "########## 3) import 验证（torch-free 子集）##########"
LD_LIBRARY_PATH=$HOME/miniconda3/envs/thinktool/lib "$TV/bin/python" - <<'EOF'
import os, sys
B = os.path.expanduser("~/ztwang/selfplay/third_party")
sys.path.insert(0, os.path.join(B, "..", "src"))
os.environ["ALFWORLD_GLIBC_COMPAT_DIR"] = os.path.join(B, "glibc232")
from alfworld_compat import install
install(verbose=True)

import numpy, gym, jericho, ai2thor, cv2, pandas, alfworld
print("numpy  ", numpy.__version__)
print("gym    ", gym.__version__)
print("jericho", jericho.__version__)
print("cv2    ", cv2.__version__)
print("pandas ", pandas.__version__)
print("alfworld", os.path.dirname(alfworld.__file__))
import textworld
print("textworld", textworld.__version__)
import textworld.gym
print("textworld.gym OK")
import fast_downward
lib = fast_downward.load_lib(); fast_downward.close_lib(lib)
print("fast_downward.load_lib() -> OK")
EOF
echo "rc(import)=$?"

echo
echo "########## 4) 若上面 libstdc++ 报错，试 conda 的 libstdc++ ##########"
LD_LIBRARY_PATH=$HOME/miniconda3/envs/thinktool/lib "$TV/bin/python" -c \
  "import ctypes.util,ctypes; print('libstdc++ resolved ->', ctypes.util.find_library('stdc++'))" 2>&1 | tail -2
strings /lib/x86_64-linux-gnu/libstdc++.so.6 2>/dev/null | grep -c 'GLIBCXX_3.4.29' || true
strings $HOME/miniconda3/envs/thinktool/lib/libstdc++.so.6 2>/dev/null | grep -c 'GLIBCXX_3.4.29' || true

echo
echo "########## 5) manifest ##########"
bash "$HOME/ztwang/selfplay/env_setup/_install_step4_sdists.sh" 2>&1 | tail -2
echo "########## DONE step8 ##########"
