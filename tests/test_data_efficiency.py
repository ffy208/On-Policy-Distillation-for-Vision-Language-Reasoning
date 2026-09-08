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
