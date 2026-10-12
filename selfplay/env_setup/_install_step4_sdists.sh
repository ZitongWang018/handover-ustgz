#!/bin/bash
# Step4: 补齐 3 个「PyPI 上没有 wheel」的 sdist，并生成 wheel 清单。
# 只写 ~/ztwang/selfplay/third_party/。
set -u
B=$HOME/ztwang/selfplay/third_party
WS=$B/wheels_sdist
PY=$HOME/miniconda3/envs/thinktool/bin/python
mkdir -p "$WS"
cd "$WS"

echo "########## fetch sdists from PyPI JSON API ##########"
for spec in fast_downward_textworld==20.6.4 gym==0.26.2 jericho==3.3.1; do
  pkg=${spec%%==*}
  ver=${spec##*==}
  url=$("$PY" - "$pkg" "$ver" <<'EOF'
import json, sys, urllib.request
n, v = sys.argv[1], sys.argv[2]
d = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{n}/{v}/json"))
cands = [u for u in d["urls"] if u["packagetype"] == "sdist"]
print(cands[0]["url"] if cands else "")
EOF
)
  echo "GET $spec  <-  $url"
  [ -n "$url" ] && curl -sSL -O "$url"
done
ls -la "$WS"

echo "########## manifest ##########"
{
  echo "# ALFWorld-on-thinktool wheel set  ($(date -Iseconds))"
  echo "# wheels/       : cp310 + cp311 + py3-none-any wheels"
  echo "# wheels_sdist/ : 仅 3 个 PyPI 上没有 wheel 的包（sdist）"
  echo
  echo "## wheels/"
  ( cd "$B/wheels" && sha256sum * | sort -k2 )
  echo
  echo "## wheels_sdist/"
  ( cd "$B/wheels_sdist" && sha256sum * | sort -k2 )
} > "$B/WHEELS_MANIFEST.txt"
wc -l "$B/WHEELS_MANIFEST.txt"

echo "########## DONE step4 ##########"
