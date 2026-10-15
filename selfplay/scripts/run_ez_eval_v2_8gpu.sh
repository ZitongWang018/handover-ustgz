#!/usr/bin/env bash
set -euo pipefail

usage() { echo "usage: $0 --base PATH --config PATH --out-root DIR (--base-only | --checkpoint V3_CKPT) [--split valid_unseen|valid_seen] [--backend hf|vllm]" >&2; exit 2; }
BASE= CONFIG= OUT_ROOT= CHECKPOINT= SPLIT=valid_unseen BACKEND=vllm BASE_ONLY=0
while (($#)); do
  case "$1" in
    --base) BASE="$2"; shift 2;; --config) CONFIG="$2"; shift 2;;
    --out-root) OUT_ROOT="$2"; shift 2;; --checkpoint) CHECKPOINT="$2"; shift 2;;
    --base-only) BASE_ONLY=1; shift;; --split) SPLIT="$2"; shift 2;;
    --backend) BACKEND="$2"; shift 2;; *) usage;;
  esac
done
[[ -n "$BASE" && -n "$CONFIG" && -n "$OUT_ROOT" ]] || usage
if [[ "$BASE_ONLY" == 1 && -n "$CHECKPOINT" ]] || [[ "$BASE_ONLY" == 0 && -z "$CHECKPOINT" ]]; then usage; fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${SKR_PYTHON:-python}"
RUN_ID="ezv2-$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_DIR="$OUT_ROOT/$RUN_ID"
mkdir -p "$OUT_ROOT"
mkdir "$RUN_DIR" || { echo "refusing to overwrite $RUN_DIR" >&2; exit 2; }
SUITE="$RUN_DIR/suite.json"
"$PYTHON_BIN" selfplay/src/ez_eval_v2.py prepare-suite --config "$CONFIG" --split "$SPLIT" --shards 8 --output "$SUITE"
N_SHARDS="$("$PYTHON_BIN" -c 'import json,sys; print(len(json.load(open(sys.argv[1],encoding="utf-8"))["shards"]))' "$SUITE")"
VISIBLE_GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
case ",${VISIBLE_GPU_IDS}," in *,,*) echo "CUDA_VISIBLE_DEVICES contains an empty GPU ID" >&2; exit 2;; esac
IFS=, read -r -a GPU_MAP <<< "$VISIBLE_GPU_IDS"
if ((${#GPU_MAP[@]} < N_SHARDS)); then echo "need $N_SHARDS visible GPU IDs, found ${#GPU_MAP[@]}" >&2; exit 2; fi
for ((i=0; i<${#GPU_MAP[@]}; i++)); do
  for ((j=0; j<i; j++)); do
    if [[ "${GPU_MAP[$i]}" == "${GPU_MAP[$j]}" ]]; then echo "duplicate CUDA_VISIBLE_DEVICES entry: ${GPU_MAP[$i]}" >&2; exit 2; fi
  done
done
ARM=adapter
if [[ "$BASE_ONLY" == 1 ]]; then ARM=base; fi
CONTRACT="$RUN_DIR/$ARM-contract.json"
PREP=("$PYTHON_BIN" selfplay/src/ez_eval_v2.py prepare-run --suite-manifest "$SUITE" --config "$CONFIG" \
  --base "$BASE" --backend "$BACKEND" --run-id "$RUN_ID" --results-dir "$RUN_DIR" \
  --output "$CONTRACT")
if [[ "$BASE_ONLY" == 1 ]]; then PREP+=(--base-only); else PREP+=(--checkpoint "$CHECKPOINT"); fi
"${PREP[@]}"
PIDS=(); FAILED=0
cleanup() { for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup INT TERM
for ((rank=0; rank<N_SHARDS; rank++)); do
  CUDA_VISIBLE_DEVICES="${GPU_MAP[$rank]}" "$PYTHON_BIN" selfplay/src/ez_eval_v2.py run --run-contract "$CONTRACT" \
    --rank "$rank" --world-size "$N_SHARDS" --output "$RUN_DIR/rank$rank.json" \
    >"$RUN_DIR/$ARM-rank$rank.log" 2>&1 &
  PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do if ! wait "$pid"; then FAILED=1; fi; done
trap - INT TERM
if ((FAILED)); then echo "one or more shards failed; results retained and merge skipped" >&2; exit 1; fi
"$PYTHON_BIN" selfplay/src/ez_eval_v2.py merge --expected-manifest "$CONTRACT" \
  --output "$RUN_DIR/$ARM-merged.json"
echo "complete: $RUN_DIR/$ARM-merged.json"
