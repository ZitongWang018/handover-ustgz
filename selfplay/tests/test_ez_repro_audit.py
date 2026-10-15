from __future__ import annotations

import json
import ast
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "selfplay" / "scripts"))
import ez_repro_audit as audit_module  # noqa: E402


def _fixture(input_dir: Path, *, all_one_task_type: bool = False):
    input_dir.mkdir(parents=True, exist_ok=True)
    games = []
    for idx in range(91):
        task_type = audit_module.TASK_TYPES[0] if all_one_task_type else audit_module.TASK_TYPES[idx % 6]
        path = f"/fixtures/valid_seen/{task_type}-object-{idx}/trial/game.tw-pddl"
        multiplicity = 3 if idx < 6 else 2 if idx < 31 else 1
        games.extend([path] * multiplicity)
    assert len(games) == audit_module.SLOT_COUNT

    slots = [{"slot": i, "worker_seed": 1000 + i, "game_file": game,
              "game_id": game, "task_type": audit_module._task_type_from_game(game)}
             for i, game in enumerate(games)]
    selection = {
        "schema_version": 1,
        "profile": "upstream-grpo-validation",
        "source_commit": "fixture-source-commit",
        "split": "eval_in_distribution",
        "slot_count": 128,
        "slots": slots,
    }
    (input_dir / "selection_128.json").write_text(json.dumps(selection), encoding="utf-8")

    for style in audit_module.STYLES:
        rows = []
        wins_by_type = Counter()
        for seed in audit_module.EXPECTED_SEEDS:
            for slot, game in enumerate(games):
                won = (slot + seed) % 5 == 0
                task_type = audit_module._task_type_from_game(game)
                wins_by_type[task_type] += int(won)
                rows.append({"game": game, "split": "eval_in_distribution", "style": style,
                             "sample": 0, "seed": seed, "won": won})
        total_won = sum(row["won"] for row in rows)
        raw = {"summary": {"style": style, "split": "eval_in_distribution",
                            "n_episodes": len(rows), "n_won": total_won,
                            "sr": total_won / len(rows),
                            "per_type_wins": {k: v for k, v in wins_by_type.items() if v}},
               "rows": rows}
        (input_dir / f"{style}_128_seeds01234.json").write_text(json.dumps(raw), encoding="utf-8")
    return selection


def test_real_handover_raw_files_have_complete_multiset_and_expected_statistics():
    report = audit_module.audit(ROOT)
    scope = report["evaluation_scope"]
    assert scope["selection_unique_games"] == 91
    assert scope["selection_repeated_games"] == 31
    assert scope["selection_max_multiplicity"] == 3
    react = report["results"]["react"]
    assert react["complete_grid"] is True
    assert (react["episodes"], react["won"]) == (640, 195)
    assert react["same_seed_repeated_game_groups"] == 155
    assert react["same_seed_repeated_game_mixed_outcome_groups"] == 26
    assert react["seed_weighted_sr_mean_pct"] == pytest.approx(30.46875)
    assert react["seed_weighted_sr_sample_sd_pp_ddof1"] == pytest.approx(2.762135864)
    zero = report["results"]["zero_shot"]
    assert (zero["episodes"], zero["won"]) == (640, 125)
    assert zero["same_seed_repeated_game_mixed_outcome_groups"] == 19
    assert zero["seed_six_type_macro_mean_pct"] == pytest.approx(20.26963777)
    assert set(report["input_hashes_sha256"]) >= {
        "selection_128.json", "react_128_seeds01234.json", "zero_shot_128_seeds01234.json"}
    assert all(not Path(value).is_absolute() for value in report["input_hashes_sha256"])
    assert report["selection_source_commit"] == "2b38da3ad7b64414305d39609484803b07d3b9b0"
    assert any("token ids" in item for item in report["raw_schema_provenance_gaps"])


def test_audit_accepts_valid_fixture_and_marks_missing_type_macro_null(tmp_path):
    input_dir = tmp_path / "raw"
    _fixture(input_dir, all_one_task_type=True)
    report = audit_module.audit(tmp_path, input_dir)
    assert report["results"]["react"]["complete_grid"] is True
    assert report["results"]["react"]["macro_missing_seed_count"] == 5
    assert report["results"]["react"]["seed_six_type_macro_mean_pct"] is None
    assert all(seed["six_type_macro_sr_pct"] is None for seed in report["results"]["react"]["seeds"])
    raw_path = input_dir / "react_128_seeds01234.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["summary"]["per_type_wins"][audit_module.TASK_TYPES[1]] = 0
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    assert audit_module.audit(tmp_path, input_dir)["results"]["react"]["complete_grid"] is True


def test_audit_accepts_relative_repo_root(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _fixture(repo / "handover/skillrise_prompting_20261015/raw")
    monkeypatch.chdir(tmp_path)
    report = audit_module.audit(Path("repo"))
    assert report["evaluation_scope"]["episodes_per_style"] == 640


def test_audit_rejects_missing_seed_and_duplicate_extra_seed_rows(tmp_path):
    input_dir = tmp_path / "raw"
    _fixture(input_dir)
    path = input_dir / "react_128_seeds01234.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["rows"] = [row for row in raw["rows"] if row["seed"] != 4]
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(audit_module.AuditError, match="missing seed groups"):
        audit_module.audit(tmp_path, input_dir)

    _fixture(input_dir)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["rows"].extend([dict(row) for row in raw["rows"] if row["seed"] == 0])
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(audit_module.AuditError, match="has 256 rows"):
        audit_module.audit(tmp_path, input_dir)


def test_audit_rejects_wrong_game_multiset_and_won_type(tmp_path):
    input_dir = tmp_path / "raw"
    _fixture(input_dir)
    path = input_dir / "zero_shot_128_seeds01234.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    seed_zero = [i for i, row in enumerate(raw["rows"]) if row["seed"] == 0]
    selection = json.loads((input_dir / "selection_128.json").read_text(encoding="utf-8"))
    raw["rows"][seed_zero[0]]["game"] = selection["slots"][3]["game_file"]
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(audit_module.AuditError, match="game multiset mismatch"):
        audit_module.audit(tmp_path, input_dir)

    _fixture(input_dir)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["rows"][0]["won"] = "False"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(audit_module.AuditError, match="won must be a JSON boolean"):
        audit_module.audit(tmp_path, input_dir)


def test_audit_rejects_task_type_inconsistent_with_selection(tmp_path):
    input_dir = tmp_path / "raw"
    selection = _fixture(input_dir)
    selection["slots"][0]["task_type"] = audit_module.TASK_TYPES[-1]
    (input_dir / "selection_128.json").write_text(json.dumps(selection), encoding="utf-8")
    with pytest.raises(audit_module.AuditError, match="task_type disagrees"):
        audit_module.audit(tmp_path, input_dir)


def test_explicit_output_is_atomically_written_as_portable_json(tmp_path):
    input_dir = tmp_path / "raw"
    _fixture(input_dir)
    output = tmp_path / "result.json"
    assert audit_module.main(["--repo-root", str(tmp_path), "--input-dir", str(input_dir),
                              "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["audit_schema_version"] == 1
    assert all(not Path(name).is_absolute() for name in report["input_hashes_sha256"])
    assert not list(tmp_path.glob(".result.json.*.tmp"))

    protected = input_dir / "react_128_seeds01234.json"
    before = protected.read_bytes()
    assert audit_module.main(["--repo-root", str(tmp_path), "--input-dir", str(input_dir),
                              "--output", str(protected)]) == 2
    assert protected.read_bytes() == before


def test_audit_utility_is_stdlib_only_and_does_not_launch_commands():
    source = (ROOT / "selfplay" / "scripts" / "ez_repro_audit.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else (
                node.func.attr if isinstance(node.func, ast.Attribute) else "")
            assert name not in {"system", "popen", "run", "Popen", "call", "check_call", "check_output"}
    assert imports <= {"argparse", "collections", "hashlib", "json", "math", "os", "statistics",
                       "sys", "tempfile", "pathlib", "typing", "__future__"}
    assert not imports.intersection({"torch", "transformers", "alfworld", "textworld", "subprocess"})
