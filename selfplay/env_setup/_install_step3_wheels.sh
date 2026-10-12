#!/bin/bash
# Step3: 把 py310deps 的完整依赖闭包下载成 wheel（cp310 + cp311 两套），供三期离线安装。
# 只写 ~/ztwang/selfplay/third_party/wheels/ 和 wheels_sdist/。
set -x
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thinktool
PY=$HOME/miniconda3/envs/thinktool/bin/python
T=$HOME/ztwang/selfplay/third_party/py310deps
W=$HOME/ztwang/selfplay/third_party/wheels
WS=$HOME/ztwang/selfplay/third_party/wheels_sdist
mkdir -p "$W" "$WS"

REQ=/tmp/alfworld_py310deps_requirements.txt
: > "$REQ"
for d in "$T"/*.dist-info; do
  [ -d "$d" ] || continue
  n=$(grep -m1 '^Name: ' "$d/METADATA" | cut -d' ' -f2)
  v=$(grep -m1 '^Version: ' "$d/METADATA" | cut -d' ' -f2)
  [ -n "$n" ] && [ -n "$v" ] && echo "$n==$v" >> "$REQ"
done
# dist-info 里没有、但 pip --target 单独落下的东西（例如 gym 是从别人环境 copy 的）补上
grep -q '^gym==' "$REQ" || echo 'gym==0.26.2' >> "$REQ"
grep -q '^gym-notices==' "$REQ" || echo 'gym-notices==0.1.0' >> "$REQ"

sort -u -o "$REQ" "$REQ"
echo "########## requirements ($(wc -l < $REQ)) ##########"
cat "$REQ"

echo "########## download cp310 wheels ##########"
"$PY" -m pip download -d "$W" --only-binary=:all: \
  --python-version 310 --implementation cp --abi cp310 -r "$REQ" || echo "!!! cp310 download had errors"

echo "########## download cp311 wheels (三期用) ##########"
"$PY" -m pip download -d "$W" --only-binary=:all: \
  --python-version 311 --implementation cp --abi cp311 -r "$REQ" || echo "!!! cp311 download had errors"

echo "########## download sdists (兜底) ##########"
"$PY" -m pip download -d "$WS" --no-binary=:all: --no-deps -r "$REQ" || echo "!!! sdist download had errors"

echo "########## wheels/ ##########"
ls -la "$W" | head -100
echo "########## wheels_sdist/ ##########"
ls -la "$WS" | head -100
echo "########## DONE step3 ##########"
