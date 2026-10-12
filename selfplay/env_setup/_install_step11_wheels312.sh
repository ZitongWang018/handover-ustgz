#!/bin/bash
# Step11: 为三期的 py3.12 分支预备 wheel（若最终选 vllm/pytorch271）。
# 只写二期 ~/ztwang/selfplay/third_party/。不碰三期。
#
# 依据：主 agent 的 selfplay/src/*.py 只 import
#   torch / transformers / numpy / yaml / textworld / alfworld / fast_downward
#   （GRPO 是手写的，不依赖 trl/peft/accelerate/deepspeed）
# => 三期需要的第三方闭包 = 环境侧闭包，**不需要**训练侧包。
set -u
B=$HOME/ztwang/selfplay/third_party
PY=$HOME/miniconda3/envs/thinktool/bin/python
W312=$B/wheels312
MISS=$B/wheels_missing
mkdir -p "$W312" "$MISS"

# 从 py310deps 清单里剔除「必须由目标 env 提供 / 绝不能遮蔽」的包
SRC=$B/requirements_py310deps.txt
[ -s "$SRC" ] || SRC=/tmp/alfworld_py310deps_requirements.txt
REQ=$B/requirements_env_common.txt
grep -vE '^(numpy|torch|torchvision)==' "$SRC" | sort -u > "$REQ"
cat "$B/requirements_offline_extra.txt" 2>/dev/null | grep -v '^#' | grep -v '^$' \
  | grep -vE '^(numpy|torch|torchvision)==' >> "$REQ"
sort -u -o "$REQ" "$REQ"

echo "########## requirements_env_common.txt ($(wc -l < $REQ)) ##########"
cat "$REQ"

echo
echo "########## 下载 cp312 wheel ##########"
FAIL=$MISS/failed_312.txt
: > "$FAIL"
ok=0; bad=0
while read -r spec; do
  [ -z "$spec" ] && continue
  if "$PY" -m pip download -d "$W312" --only-binary=:all: --no-deps \
       --python-version 312 --implementation cp --abi cp312 "$spec" \
       >/dev/null 2>>"$MISS/dl_312.stderr"; then
    ok=$((ok+1))
  else
    bad=$((bad+1)); echo "$spec" | tee -a "$FAIL"
  fi
done < "$REQ"
echo "cp312: ok=$ok fail=$bad"
echo "wheels312 count: $(ls "$W312"/*.whl 2>/dev/null | wc -l)"
echo "--- 缺 cp312 wheel 的包 ---"; cat "$FAIL"

echo
echo "########## 校验：cp311(已有 wheels/) 与 cp312 覆盖对比 ##########"
for spec in $(cat "$REQ"); do
  p=${spec%%==*}
  h311=$(ls "$B/wheels"/${p}-*cp311*.whl "$B/wheels"/${p}-*none-any.whl 2>/dev/null | head -1)
  h312=$(ls "$W312"/${p}-*cp312*.whl "$W312"/${p}-*none-any.whl 2>/dev/null | head -1)
  [ -z "$h311" ] && echo "  cp311 MISS: $p"
  [ -z "$h312" ] && echo "  cp312 MISS: $p"
done
echo "########## DONE step11 ##########"
