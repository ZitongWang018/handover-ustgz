#!/bin/bash
# Step9: 补 flask（ai2thor 依赖）并重做离线验证；这次带自动补包循环。
set -u
B=$HOME/ztwang/selfplay/third_party
W=$B/wheels
P311=$B/py311
PY=$HOME/miniconda3/envs/thinktool/bin/python
TV=$B/_testvenv311

dl() {  # dl <pkg-spec>  -> 下 cp310+cp311 wheel 到 $W
  local spec="$1"
  for pair in "310 cp310" "311 cp311"; do
    set -- $pair
    "$PY" -m pip download -d "$W" --only-binary=:all: --python-version "$1" \
        --implementation cp --abi "$2" "$spec" \
        >/dev/null 2>>"$B/wheels_missing/dl_${1}.stderr" || echo "  miss cp$1: $spec"
  done
}

echo "########## 1) flask（带依赖解析）##########"
dl flask
ls "$W" | grep -iE 'flask|itsdangerous|blinker|jinja|markupsafe|click|werkzeug'
echo "wheels now: $(ls "$W"/*.whl | wc -l)"

echo
echo "########## 2) 干净 venv 离线安装（自动补包，最多 8 轮）##########"
rm -rf "$TV"
"$P311/bin/python3" -m venv "$TV"
CORE="textworld ai2thor gym gym-notices jericho fast_downward_textworld cloudpickle \
      opencv-python opencv-python-headless pandas werkzeug numpy==1.26.4 pytz flask"
for round in 1 2 3 4 5 6 7 8; do
  out=$("$TV/bin/python" -m pip install --no-cache-dir --no-index --find-links "$W" $CORE 2>&1)
  if echo "$out" | grep -q 'Successfully installed\|already satisfied'; then
    if ! echo "$out" | grep -q 'No matching distribution'; then echo "round $round: OK"; break; fi
  fi
  miss=$(echo "$out" | grep -oE "Could not find a version that satisfies the requirement [^ ]+" | awk '{print $NF}' | sort -u | tr '\n' ' ')
  echo "round $round missing: ${miss:-<none>}"
  [ -z "$miss" ] && { echo "$out" | tail -15; break; }
  for m in $miss; do
    m=${m%%;*}
    dl "$m"
  done
done

echo
echo "########## 3) alfworld --no-deps ##########"
"$TV/bin/python" -m pip -q install --no-cache-dir --no-index --find-links "$W" --no-deps alfworld
echo "rc=$?"

echo
echo "########## 4) import 验证（torch-free 子集，带 conda libstdc++）##########"
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
print("fast_downward.load_lib() -> OK  (py3.11, offline-installed)")
EOF
echo "rc(import)=$?"

echo
echo "########## 5) manifest ##########"
bash "$HOME/ztwang/selfplay/env_setup/_install_step4_sdists.sh" 2>&1 | tail -2
echo "########## DONE step9 ##########"
