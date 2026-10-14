"""Strict shard merge regressions; all fixtures are local and CPU-only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from ez_eval_merge import merge_shards


def _write_shard(path, *, run_id="run-1", task_id="task-0", sample_count=2,
                 offset=0, select="head", split="eval_out_of_distribution",
                 decode_temp=1.0):
    rows = [{"task_id": task_id, "task_type": "type-a", "sample": s,
             "won": bool(s)} for s in range(sample_count)]
    report = {
        "schema_version": 2,
        "split": split,
        "select": select,
        "offset": offset,
        "expected_task_ids": [task_id],
        "sample_rows": rows,
        "decode_effective": {"temperature": decode_temp},
        "protocol_manifest": {"schema_version": 2, "evaluation_run_id": run_id,
                               "samples_per_task": sample_count,
                               "decode": {"temperature": decode_temp}},
    }
    path.write_text(json.dumps(report), encoding="utf-8")


def _expected(paths, *, sample_count=2, task_count=2, type_count=1):
    return {"run_id": "run-1", "samples_per_task": sample_count,
            "required_task_count": task_count, "required_task_type_count": type_count,
            "split": "eval_out_of_distribution",
            "expected_shards": [{"path": str(p), "offset": i, "n": 1}
                                for i, p in enumerate(paths)]}


def test_merge_requires_declared_complete_compatible_shards(tmp_path):
    p0, p1 = tmp_path / "s0.json", tmp_path / "s1.json"
    _write_shard(p0, task_id="task-0", offset=0)
    _write_shard(p1, task_id="task-1", offset=1)
    result = merge_shards(_expected([p0, p1]))
    assert result["summary"]["n_episodes"] == 4
    assert result["summary"]["sample_suite_success_rates"] == [0.0, 1.0]

    missing = _expected([p0, tmp_path / "missing.json"])
    with pytest.raises(ValueError, match="missing expected shard"):
        merge_shards(missing)

    _write_shard(p1, task_id="task-0", offset=1)
    with pytest.raises(ValueError, match="duplicate task"):
        merge_shards(_expected([p0, p1]))


@pytest.mark.parametrize("mutations, message", [
    ({"run_id": "old"}, "wrong-run"),
    ({"sample_count": 1}, "samples_per_task"),
    ({"select": "shuffle"}, "selection"),
    ({"split": "train"}, "split"),
    ({"decode_temp": 0.0}, "incompatible protocol"),
])
def test_merge_rejects_stale_or_incompatible_shards(tmp_path, mutations, message):
    p0, p1 = tmp_path / "s0.json", tmp_path / "s1.json"
    _write_shard(p0, task_id="task-0", offset=0)
    _write_shard(p1, task_id="task-1", offset=1, **mutations)
    with pytest.raises(ValueError, match=message):
        merge_shards(_expected([p0, p1]))


def test_merge_rejects_row_outside_its_declared_shard(tmp_path):
    p0, p1 = tmp_path / "s0.json", tmp_path / "s1.json"
    _write_shard(p0, task_id="task-1", offset=0)
    _write_shard(p1, task_id="task-1", offset=1)
    with pytest.raises(ValueError, match="duplicate task"):
        merge_shards(_expected([p0, p1]))
