#!/bin/bash
# Step3b: 逐包下载 wheel（cp310 + cp311），单包失败不影响其它。
# 只写 ~/ztwang/selfplay/third_party/{wheels,wheels_missing}。
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thinktool
PY=$HOME/miniconda3/envs/thinktool/bin/python
BASE=$HOME/ztwang/selfplay/third_party
W=$BASE/wheels
MISS=$BASE/wheels_missing
REQ=${REQ:-/tmp/alfworld_py310deps_requirements.txt}
mkdir -p "$W" "$MISS"

echo "### requirements: $REQ"
cat "$REQ"

for pair in "310 cp310" "311 cp311"; do
  set -- $pair
  PYV=$1
  ABI=$2
  FAIL=$MISS/failed_${PYV}.txt
  : > "$FAIL"
  echo "########## cp$PYV ##########"
  n_ok=0; n_fail=0
  while read -r spec; do
    [ -z "$spec" ] && continue
    if "$PY" -m pip download -d "$W" --only-binary=:all: --no-deps \
         --python-version "$PYV" --implementation cp --abi "$ABI" "$spec" \
         >/dev/null 2>>"$MISS/dl_${PYV}.stderr"; then
      n_ok=$((n_ok+1))
    else
      n_fail=$((n_fail+1))
      echo "$spec" | tee -a "$FAIL"
    fi
  done < "$REQ"
  echo "cp$PYV: ok=$n_ok fail=$n_fail  (failures -> $FAIL)"
done

echo "########## result ##########"
ls "$W" | wc -l
echo "--- wheels/ ---"
ls -la "$W"
echo "--- wheels_missing/ ---"
for f in "$MISS"/failed_*.txt; do echo "[$f]"; cat "$f"; done
echo "########## DONE step3b ##########"
