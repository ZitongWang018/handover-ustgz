#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
config_path="${1:-$project_dir/configs/local.json}"
python_bin="${SWE_RL_PYTHON:-python3}"
"$python_bin" "$project_dir/run.py" bootstrap
cache_root="$("$python_bin" - "$project_dir" "$config_path" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from launcher import check_workspace, paths, read_config
c = read_config(sys.argv[2]); check_workspace(c)
print(paths(c)['cache_root'])
PY
)"
export UV_CACHE_DIR="$cache_root/uv"
export UV_PYTHON_INSTALL_DIR="$cache_root/uv-python"
export TMPDIR="$cache_root/tmp"
mkdir -p "$TMPDIR"
cd "$project_dir/.deps/harbor"
# Isolated Harbor environment; no install into the server's global Python.
uv sync --frozen --extra huggingface --python 3.12
printf 'Harbor ready. Use %s/.deps/harbor/.venv/bin/python for prepare-data and serve.\n' "$project_dir"
