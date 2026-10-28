#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
config_path="${SWE_RL_CONFIG:-$project_dir/configs/local.json}"
python_bin="${SWE_RL_PYTHON:-python3}"
mapfile -t settings < <("$python_bin" - "$project_dir" "$config_path" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from launcher import check_workspace, paths, read_config
c = read_config(sys.argv[2]); check_workspace(c)
for value in (str(Path(c['workspace_root']).resolve()), c['runtime']['image'], c['runtime']['gpus'], str(paths(c)['cache_root'])):
    print(value)
PY
)
if [[ ${#settings[@]} != 4 ]]; then
  echo 'Could not read configuration' >&2
  exit 1
fi
workspace_root="${settings[0]}"
image="${settings[1]}"
devices="${settings[2]}"
cache_root="${settings[3]}"
mkdir -p "$cache_root/tmp" "$cache_root/torch_extensions"
# Record the immutable identity actually used. If absent, pull the configured image.
if ! docker image inspect "$image" >/dev/null 2>&1; then
  docker pull "$image"
fi
image_id="$(docker image inspect --format '{{.Id}}' "$image")"
docker_args=(run --rm --gpus "\"device=$devices\"" --network host --ipc host
  --shm-size 32g --ulimit memlock=-1 --ulimit stack=67108864
  --mount "type=bind,source=$workspace_root,target=$workspace_root"
  --workdir "$project_dir"
  --env "SWE_RL_IMAGE_ID=$image_id"
  --env "TMPDIR=$cache_root/tmp"
  --env "XDG_CACHE_HOME=$cache_root"
  --env "TORCH_EXTENSIONS_DIR=$cache_root/torch_extensions")
# Pass only explicitly supported secret/network settings; never write their values to disk.
for key in HARBOR_ADMIN_SECRET NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_DEBUG; do
  if [[ -v "$key" ]]; then docker_args+=(--env "$key"); fi
done
# Each invocation gets a private container PID namespace. No --privileged,
# host PID namespace, docker socket, global process cleanup, or latest git pull.
exec docker "${docker_args[@]}" "$image_id" python3 "$project_dir/run.py" "$@" --config "$config_path"
