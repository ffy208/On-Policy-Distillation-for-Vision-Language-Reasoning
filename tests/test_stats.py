"""Tests for bootstrap statistics on per-question correctness."""

import json

import numpy as np
import pytest

from vlm_opd.analysis.stats import bootstrap_ci, compare_results, paired_bootstrap


def test_bootstrap_ci_brackets_accuracy():
    c = np.array([1] * 70 + [0] * 30)
    r = bootstrap_ci(c, n_boot=2000)
    assert r["accuracy"] == pytest.approx(0.7)
    assert r["ci_low"] < 0.7 < r["ci_high"]
    assert 0.6 < r["ci_low"] and r["ci_high"] < 0.8


def test_paired_bootstrap_detects_consistent_gain():
    rng = np.random.default_rng(0)
    b = (rng.random(500) < 0.7).astype(int)
    a = b.copy()
    flip = rng.choice(np.where(b == 0)[0], size=40, replace=False)  # a fixes 40 of b's errors
    a[flip] = 1
    r = paired_bootstrap(a, b, n_boot=2000)
    assert r["delta"] == pytest.approx(0.08)
    assert r["ci_low"] > 0 and r["p_delta_le_0"] < 0.01
    assert r["a_only_correct"] == 40 and r["b_only_correct"] == 0


def test_paired_bootstrap_no_difference():
    rng = np.random.default_rng(1)
    a = (rng.random(300) < 0.7).astype(int)
    r = paired_bootstrap(a, a.copy(), n_boot=500)
    assert r["delta"] == 0.0 and r["ci_low"] == 0.0 and r["ci_high"] == 0.0


def test_compare_results_requires_matching_ids(tmp_path):
    def write(name, recs):
        p = tmp_path / name
        p.write_text(json.dumps({"records": recs}))
        return p

    a = write("a.json", [{"id": "q1", "correct": True}, {"id": "q2", "correct": False}])
    b = write("b.json", [{"id": "q2", "correct": True}, {"id": "q1", "correct": False}])  # different order is fine
    r = compare_results(a, b, n_boot=200)
    assert r["n"] == 2 and r["delta"] == 0.0
    c = write("c.json", [{"id": "q1", "correct": True}, {"id": "q3", "correct": False}])
    with pytest.raises(ValueError):
        compare_results(a, c, n_boot=10)
