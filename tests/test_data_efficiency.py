"""Tests for the data-efficiency aggregation on synthetic result files."""

import json

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from vlm_opd.analysis.data_efficiency import (
    collect,
    crossover,
    markdown_table,
    result_path,
    write_report,
)


def _write(path, acc, seed):
    rng = np.random.default_rng(seed)
    recs = [{"id": f"test_{i:05d}", "correct": bool(rng.random() < acc)} for i in range(200)]
    path.write_text(json.dumps({"records": recs}))


def test_collect_and_report(tmp_path):
    budgets = [300, 900, 3000]
    for b, sft_acc, opd_acc in [(300, 0.70, 0.78), (900, 0.78, 0.82), (3000, 0.83, 0.84)]:
        _write(result_path(tmp_path, "sft", b), sft_acc, seed=b)
        _write(result_path(tmp_path, "opd", b), opd_acc, seed=b + 1)
    base = tmp_path / "base.json"; _write(base, 0.66, seed=7)
    teacher = tmp_path / "teacher.json"; _write(teacher, 0.85, seed=8)

    table = collect(tmp_path, budgets, base, teacher, n_boot=300)
    assert [p["budget"] for p in table["points"]] == budgets
    assert all(p["sft"] and p["opd"] and p["opd_minus_sft"] for p in table["points"])
    assert table["baseline"]["accuracy"] < table["teacher"]["accuracy"]
    md = markdown_table(table)
    assert md.count("|") > 20 and "teacher" in md
    cross = crossover(table)
    assert cross is not None and cross["opd_budget"] in budgets
    report = write_report(table, tmp_path)
    assert report.exists() and (tmp_path / "data_efficiency.png").exists() and (tmp_path / "data_efficiency.json").exists()


def test_collect_handles_missing_points(tmp_path):
    _write(result_path(tmp_path, "sft", 300), 0.7, seed=1)
    table = collect(tmp_path, [300, 900], n_boot=100)
    assert table["points"][0]["opd"] is None and "opd_minus_sft" not in table["points"][0]
    assert table["points"][1]["sft"] is None
    assert crossover(table) is None
    assert "pending" in markdown_table(table)


def _write_named(path, acc, seed, n=200):
    rng = np.random.default_rng(seed)
    recs = [{"id": f"test_{i:05d}", "correct": bool(rng.random() < acc)} for i in range(n)]
    path.write_text(json.dumps({"records": recs}))


def test_find_runs_mixes_legacy_and_seeded(tmp_path):
    from vlm_opd.analysis.data_efficiency import find_runs

    _write_named(tmp_path / "eval_sft_q300.json", 0.78, 42)                       # notebook-era, seed 42
    _write_named(tmp_path / "eval_sft_chartqa_human_q300_s1.json", 0.77, 1)
    _write_named(tmp_path / "eval_opd_chartqa_human_q300_s1.json", 0.83, 1)
    _write_named(tmp_path / "eval_opd_chartqa_human_q300_s2.json", 0.84, 2)
    _write_named(tmp_path / "eval_opd_geometry_q300_s1.json", 0.5, 3)              # other task, ignored
    runs = find_runs(tmp_path, "chartqa_human")
    assert sorted(runs[("sft", 300)]) == [1, 42]
    assert sorted(runs[("opd", 300)]) == [1, 2]
    assert ("opd", 300) in runs and all("geometry" not in str(p) for f in runs.values() for p in f.values())


def test_collect_seeded_pools_and_pairs_by_seed(tmp_path):
    from vlm_opd.analysis.data_efficiency import collect_seeded, markdown_table_seeded

    for s in (1, 2, 3):
        _write_named(tmp_path / f"eval_sft_chartqa_human_q300_s{s}.json", 0.75, 10 + s)
        _write_named(tmp_path / f"eval_opd_chartqa_human_q300_s{s}.json", 0.85, 20 + s)
    _write_named(tmp_path / "eval_sft_chartqa_human_q900_s1.json", 0.80, 31)
    table = collect_seeded(tmp_path, "chartqa_human", [300, 900], n_boot=300)
    p300 = table["points"][0]
    assert p300["sft"]["seeds"] == [1, 2, 3] and p300["sft"]["n"] == 600
    assert 0.7 < p300["sft"]["accuracy"] < 0.8 and p300["sft"]["std_across_seeds"] >= 0
    assert p300["opd_minus_sft"]["seeds"] == [1, 2, 3] and p300["opd_minus_sft"]["delta"] > 0.05
    assert table["points"][1]["opd"] is None and "opd_minus_sft" not in table["points"][1]
    md = markdown_table_seeded(table)
    assert "n=3" in md and "over 3 seed(s)" in md and "pending" in md
