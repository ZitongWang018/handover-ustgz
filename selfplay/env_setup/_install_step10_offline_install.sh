#!/bin/bash
# Step10: 全量 --no-deps 离线安装（离线脚本的真实用法），然后 import 验证。
set -u
B=$HOME/ztwang/selfplay/third_party
W=$B/wheels
P311=$B/py311
TV=$B/_testvenv311

cp -f /tmp/alfworld_py310deps_requirements.txt "$B/requirements_py310deps.txt" 2>/dev/null || true

cat > "$B/requirements_offline_extra.txt" <<'EOF'
# 宿主环境(thinktool)自带、因此没被 pip --target 收进 py310deps 清单，但依赖链需要
cloudpickle==3.1.2
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.3.3
werkzeug==3.1.4
python-dateutil==2.9.0.post0
pytz==2026.5
six==1.17.0
tzdata==2025.3
flask==3.1.3
blinker==1.9.0
itsdangerous==2.2.0
EOF

ALL=$(cat "$B/requirements_py310deps.txt" "$B/requirements_offline_extra.txt" | grep -v '^#' | grep -v '^$' | tr '\n' ' ')
echo "########## specs: $(echo $ALL | wc -w) ##########"

echo "########## 重建 venv 并全量 --no-deps 离线安装 ##########"
rm -rf "$TV"
"$P311/bin/python3" -m venv "$TV"
"$TV/bin/python" -m pip install --no-cache-dir --no-index --find-links "$W" --no-deps -q $ALL
echo "rc=$?"
"$TV/bin/python" -m pip list 2>/dev/null | wc -l

echo
echo "########## import 验证（torch-free 子集 + conda libstdc++）##########"
LD_LIBRARY_PATH=$HOME/miniconda3/envs/thinktool/lib "$TV/bin/python" - <<'EOF'
import os, sys
B = os.path.expanduser("~/ztwang/selfplay/third_party")
sys.path.insert(0, os.path.join(B, "..", "src"))
os.environ["ALFWORLD_GLIBC_COMPAT_DIR"] = os.path.join(B, "glibc232")
from alfworld_compat import install
install(verbose=True)

import numpy, gym, jericho, ai2thor, cv2, pandas, alfworld, flask, textworld
print("numpy    ", numpy.__version__)
print("gym      ", gym.__version__)
print("jericho  ", jericho.__version__)
print("cv2      ", cv2.__version__)
print("pandas   ", pandas.__version__)
print("flask    ", flask.__version__)
print("alfworld ", os.path.dirname(alfworld.__file__))
print("textworld", textworld.__version__)
import textworld.gym
print("textworld.gym OK")
import fast_downward
lib = fast_downward.load_lib(); fast_downward.close_lib(lib)
print("fast_downward.load_lib() -> OK  (py3.11 + 全离线 wheel)")
print("ALL GOOD")
EOF
echo "rc(import)=$?"

echo
echo "########## manifest ##########"
bash "$HOME/ztwang/selfplay/env_setup/_install_step4_sdists.sh" 2>&1 | tail -2
echo "########## DONE step10 ##########"
