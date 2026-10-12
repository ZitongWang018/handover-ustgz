#!/bin/bash
# Step6: 校验 wheel 集合 —— ① 每个 cp310 二进制包都有 cp311 对应 ② 在独立 py3.11 里真装真 import
set -u
B=$HOME/ztwang/selfplay/third_party
W=$B/wheels
P311=$B/py311

echo "########## 1) wheel tag 审计 ##########"
cd "$W"
mkdir -p "$B/_tmp"
PKGS=$B/_tmp/_pkgs.txt
ls *.whl | sed 's/\.whl$//' | awk -F'-' '{print $1}' | sort -u > "$PKGS"
while read -r p; do
  has310=$(ls "${p}"-*cp310*.whl 2>/dev/null | head -1)
  has311=$(ls "${p}"-*cp311*.whl 2>/dev/null | head -1)
  any=$(ls "${p}"-*none-any.whl 2>/dev/null | head -1)
  # 归一化包名做匹配（wheel 里 _ 与 - 混用）
  if [ -z "$has310" ] && [ -z "$has311" ]; then
    p2=$(echo "$p" | tr '_' '-')
    has310=$(ls "${p2}"-*cp310*.whl 2>/dev/null | head -1)
    has311=$(ls "${p2}"-*cp311*.whl 2>/dev/null | head -1)
    any=$(ls "${p2}"-*none-any.whl 2>/dev/null | head -1)
  fi
  if [ -n "$any" ]; then
    printf '  %-28s pure-python (%s)\n' "$p" "$(basename "$any")"
  elif [ -n "$has310" ] && [ -n "$has311" ]; then
    printf '  %-28s cp310+cp311 OK\n' "$p"
  elif [ -n "$has310" ] && [ -z "$has311" ]; then
    printf '  %-28s !!! 只有 cp310，缺 cp311 -> %s\n' "$p" "$(basename "$has310")"
  elif [ -z "$has310" ] && [ -n "$has311" ]; then
    printf '  %-28s !!! 只有 cp311，缺 cp310\n' "$p"
  else
    printf '  %-28s ??? 未归类\n' "$p"
  fi
done < "$PKGS"

echo
echo "总 wheel 数: $(ls "$W"/*.whl | wc -l)"

echo
echo "########## 2) py3.11 真装真 import ##########"
if [ -x "$P311/bin/python3" ]; then
  "$P311/bin/python3" -V
  "$P311/bin/python3" -m pip install -q --no-index --find-links "$W" gym jericho numpy
  "$P311/bin/python3" - <<'EOF'
import gym, jericho, numpy, os, sys
print("python", sys.version.split()[0])
print("gym", gym.__version__, "| jericho", jericho.__version__, "| numpy", numpy.__version__)
print("jericho file:", jericho.__file__)
EOF
  echo "--- fast_downward (py3.11) 能否 load_lib ---"
  "$P311/bin/python3" -m pip install -q --no-index --find-links "$W" fast_downward_textworld
  "$P311/bin/python3" - <<'EOF'
import os, sys, ctypes
B = os.path.expanduser("~/ztwang/selfplay/third_party")
sys.path.insert(0, os.path.join(B, "..", "src"))
os.environ["ALFWORLD_GLIBC_COMPAT_DIR"] = os.path.join(B, "glibc232")
from alfworld_compat import install
install(verbose=False)
import fast_downward
lib = fast_downward.load_lib()
fast_downward.close_lib(lib)
print("fast_downward.load_lib() on py3.11 -> OK")
EOF
else
  echo "!!! py311 不存在: $P311"
fi
echo "########## DONE step6 ##########"
