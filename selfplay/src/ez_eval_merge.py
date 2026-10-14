#!/usr/bin/env python
"""Strict CPU-only merger for a declared set of EZ evaluation shards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ez_eval_protocol import aggregate_complete_evaluation


def merge_shards(expected_manifest: dict) -> dict:
    shard_specs = expected_manifest.get("expected_shards")
    if not isinstance(shard_specs, list) or not shard_specs:
        raise ValueError("expected_shards must be a non-empty list")
    expected_files = [str(Path(spec["path"]).resolve()) for spec in shard_specs]
    if len(expected_files) != len(set(expected_files)):
        raise ValueError("expected shard file list contains duplicates")
    samples_per_task = int(expected_manifest.get("samples_per_task", 0))
    required_tasks = int(expected_manifest.get("required_task_count", 0))
    required_types = int(expected_manifest.get("required_task_type_count", 0))
    expected_split = expected_manifest.get("split", "eval_out_of_distribution")
    run_id = expected_manifest.get("run_id")
    if samples_per_task <= 0 or required_tasks <= 0 or required_types <= 0:
        raise ValueError("manifest must declare positive task/sample/type counts")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("manifest must declare a non-empty run_id")

    rows = []
    expected_ids = []
    base_manifest = None
    base_decode_effective = None
    base_decode = None
    shard_summaries = []
    for path_text in expected_files:
        path = Path(path_text)
        if not path.is_file():
            raise ValueError(f"missing expected shard: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 2:
            raise ValueError(f"shard has missing or unsupported result schema: {path}")
        manifest = data.get("protocol_manifest")
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 2:
            raise ValueError(f"shard has missing or unsupported manifest schema: {path}")
        if manifest.get("evaluation_run_id") != expected_manifest.get("run_id"):
            raise ValueError(f"stale or wrong-run shard manifest: {path}")
        if manifest.get("samples_per_task") != samples_per_task:
            raise ValueError(f"wrong samples_per_task in {path}")
        if data.get("split") != expected_split:
            raise ValueError(f"wrong split in {path}")
        spec = next((item for item in shard_specs
                     if str(Path(item["path"]).resolve()) == str(path)), None)
        if spec is None or len(data.get("expected_task_ids", [])) != int(spec.get("n", -1)):
            raise ValueError(f"shard task count does not match its declared spec: {path}")
        if data.get("select") != "head" or data.get("offset") != int(spec.get("offset", -1)):
            raise ValueError(f"shard selection does not match its declared offset: {path}")
        if base_manifest is None:
            base_manifest = manifest
            base_decode_effective = data.get("decode_effective")
            base_decode = manifest.get("decode")
        elif manifest != base_manifest or data.get("decode_effective") != base_decode_effective:
            raise ValueError(f"incompatible protocol manifest or effective decode in {path}")
        shard_ids = data.get("expected_task_ids")
        if not isinstance(shard_ids, list) or len(shard_ids) != len(set(shard_ids)):
            raise ValueError(f"invalid expected_task_ids in {path}")
        overlap = set(expected_ids).intersection(shard_ids)
        if overlap:
            raise ValueError(f"duplicate task across shards: {sorted(overlap)[:3]}")
        expected_ids.extend(str(task_id) for task_id in shard_ids)
        shard_rows = data.get("sample_rows")
        if not isinstance(shard_rows, list):
            raise ValueError(f"missing sample_rows in {path}")
        shard_type_by_id = {}
        for row in shard_rows:
            task_id = str(row.get("task_id", ""))
            if task_id not in shard_ids:
                raise ValueError(f"row task is outside this shard's declared IDs: {task_id}")
            task_type = str(row.get("task_type", ""))
            if task_id in shard_type_by_id and shard_type_by_id[task_id] != task_type:
                raise ValueError(f"task type differs within shard for {task_id}")
            shard_type_by_id[task_id] = task_type
        if set(shard_type_by_id) != set(shard_ids):
            raise ValueError(f"shard rows do not cover declared task ids: {path}")
        aggregate_complete_evaluation(shard_rows, shard_type_by_id, samples_per_task)
        rows.extend(shard_rows)
        shard_summaries.append({"path": str(path), "task_count": len(shard_ids),
                                "episode_count": len(shard_rows)})

    if len(expected_ids) != required_tasks:
        raise ValueError(f"expected {required_tasks} tasks, shard manifests declare {len(expected_ids)}")
    expected_types = {}
    row_types = {}
    for row in rows:
        task_id = str(row.get("task_id", ""))
        task_type = str(row.get("task_type", ""))
        if task_id not in expected_ids:
            raise ValueError(f"unexpected task in shard rows: {task_id}")
        if task_id in row_types and row_types[task_id] != task_type:
            raise ValueError(f"task type differs across rows for {task_id}")
        row_types[task_id] = task_type
    if set(row_types) != set(expected_ids):
        raise ValueError("some manifest-declared task ids have no sample rows")
    expected_types.update(row_types)
    summary = aggregate_complete_evaluation(rows, expected_types, samples_per_task)
    if summary["n_episodes"] != required_tasks * samples_per_task:
        raise ValueError("merged evaluation did not produce the required task×sample count")
    if summary["n_task_types"] != required_types:
        raise ValueError(f"expected {required_types} task types, got {summary['n_task_types']}")
    return {
        "schema_version": 2,
        "protocol_manifest": base_manifest,
        "decode_effective": base_decode_effective,
        "summary": summary,
        "shard_count": len(shard_summaries),
        "shards": shard_summaries,
        "sample_rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expected-manifest", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    expected_path = Path(args.expected_manifest)
    result = merge_shards(json.loads(expected_path.read_text(encoding="utf-8")))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    s = result["summary"]
    print(f"merged tasks={s['n_tasks']} episodes={s['n_episodes']} "
          f"episode_weighted_SR={s['episode_weighted_sr']:.4f}; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
