#!/usr/bin/env python
"""Print the current runner's weighted Pass@1 and six-type macro separately."""
import json
import sys
from pathlib import Path


def summarize(path):
    with Path(path).open(encoding="utf-8") as f:
        result = json.load(f)
    if result.get("profile") != "upstream-grpo-validation" or "metrics" not in result:
        raise ValueError(f"{path}: expected merged output from the current code-aligned runner")
    p1 = result["metrics"]["pass_at_1"]
    print(f"=== {Path(path).name} run={result.get('run_id')} style={result.get('style')}")
    print(f"slots={result.get('n_slots')} split={result.get('split')} protocol={result.get('protocol_status')}")
    print(f"Pass@1 slot-weighted SR: {p1['episode_weighted_sr'] * 100:.2f}%")
    macro = p1.get("six_type_macro_sr")
    if macro is None:
        print(f"Six-type macro: unavailable; missing={', '.join(p1.get('missing_task_types', []))}")
    else:
        print(f"Six-type macro SR: {macro * 100:.2f}%")
    for task_type, sr in sorted(p1.get("per_type_sr", {}).items()):
        print(f"  {task_type}: {sr * 100:.2f}% (n={p1['per_type_slot_counts'][task_type]})")
    return p1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python skillrise/summarize.py skillrise/out/<run>/merged.json [...]")
    for filename in sys.argv[1:]:
        summarize(filename)
