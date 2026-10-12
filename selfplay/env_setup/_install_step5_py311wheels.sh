#!/bin/bash
# Step5: 用一个「装在 ~/ztwang 里的独立 CPython 3.11」把 PyPI 上没有 wheel 的
#        3 个包编成 cp311 wheel（三期离线可用）。
# 不写 ~/miniconda3/envs 下的任何东西；所有产物都在 ~/ztwang/selfplay/third_party/。
set -u
B=$HOME/ztwang/selfplay/third_party
D=$B/py311
PY=$HOME/miniconda3/envs/thinktool/bin/python
TMP=$B/_tmp
mkdir -p "$TMP"

cd "$TMP"
if [ ! -s pbs.json ]; then
  curl -sSL -m 180 https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest -o pbs.json
fi
URL=$("$PY" - <<'EOF'
import json
d = json.load(open("pbs.json"))
for a in d.get("assets", []):
    n = a["name"]
    if n.startswith("cpython-3.11.") and n.endswith("-x86_64-unknown-linux-gnu-install_only.tar.gz"):
        print(a["browser_download_url"])
        break
EOF
)
echo "PBS_URL=$URL"
[ -n "$URL" ] || { echo "!!! 没有找到 3.11 install_only 资产"; exit 1; }

F=$(basename "$URL")
if [ ! -s "$F" ]; then curl -sSL -m 900 -o "$F" "$URL"; fi
ls -la "$F"

if [ ! -x "$D/bin/python3" ]; then
  rm -rf "$B/_pbsx"; mkdir -p "$B/_pbsx"
  tar -xzf "$F" -C "$B/_pbsx"
  rm -rf "$D"
  mv "$B/_pbsx/python" "$D"
  rm -rf "$B/_pbsx"
fi
"$D/bin/python3" -V
"$D/bin/python3" -m pip -q install -U pip setuptools wheel

echo "########## build cp311 wheels ##########"
"$D/bin/python3" -m pip wheel --no-deps --no-cache-dir -w "$B/wheels" \
  fast_downward_textworld==20.6.4 gym==0.26.2 jericho==3.3.1

echo "########## built files ##########"
ls -la "$B/wheels" | grep -Ei 'gym-0.26|jericho|fast_downward' || true

echo "########## verify import on 3.11 ##########"
"$D/bin/python3" -c "import gym, jericho; print('gym', gym.__version__, '| jericho', jericho.__version__)" || echo "!!! import verify failed"
"$D/bin/python3" -c "import sys; sys.path.insert(0,'$B/wheels'); print('ok')" >/dev/null

echo "########## regenerate manifest ##########"
bash "$HOME/ztwang/selfplay/env_setup/_install_step4_sdists.sh" 2>&1 | tail -3

echo "########## DONE step5 ##########"
