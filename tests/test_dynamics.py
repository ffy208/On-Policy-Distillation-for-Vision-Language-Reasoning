import json

import numpy as np

from vlm_opd.analysis.dynamics import (
    _rank,
    _spearman,
    load_log,
    parse_run_name,
    residual_vs_gap,
    trajectories,
)


def _row(step, mass, eos, kl):
    return {"step": step, "kl": kl, "eos_rate": eos, "format_rate": eos,
            "role_kl_mass": {"answer": mass, "chart_value": 0.5 - mass, "arithmetic": 0.1, "text": 0.4},
            "role_concentration": {"answer": mass * 10, "chart_value": 1.0, "arithmetic": 1.0, "text": 1.0}}


def test_load_log_keeps_last_record_per_step_and_parses_names(tmp_path):
    p = tmp_path / "opd_geometry3k_q300_s1.jsonl"
    rows = [_row(1, 0.3, 0.5, 0.4), _row(2, 0.2, 0.5, 0.3), _row(2, 0.25, 0.6, 0.35)]  # step 2 logged twice (resume)
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    log = load_log(p)
    assert [r["step"] for r in log] == [1, 2] and log[1]["kl"] == 0.35
    assert parse_run_name("opd-term_geometry3k_q900_s3") == {"method": "opd-term", "task": "geometry3k", "budget": 900, "seed": 3}
    assert parse_run_name("opd_human_q300") == {"method": "opd", "task": "chartqa_human", "budget": 300, "seed": 42}
    assert parse_run_name("sft_geometry3k_q900_s3") is None


def test_rank_and_spearman():
    assert _rank(np.array([10.0, 30.0, 20.0, 20.0])).tolist() == [1.0, 4.0, 2.5, 2.5]
    assert abs(_spearman(np.array([1.0, 2, 3, 4]), np.array([10.0, 20, 30, 40])) - 1.0) < 1e-12
    assert abs(_spearman(np.array([1.0, 2, 3, 4]), np.array([4.0, 3, 2, 1])) + 1.0) < 1e-12


def test_trajectories_and_residual_correlation(tmp_path):
    logs, results = {}, {}
    for seed, residual in enumerate([0.05, 0.10, 0.20, 0.30, 0.40], start=1):
        p = tmp_path / f"opd_chartqa_human_q300_s{seed}.jsonl"
        p.write_text("\n".join(json.dumps(_row(s, 0.5 - residual, 1.0, 0.2)) for s in range(1, 51)) + "\n")
        logs[(300, seed)] = p
        r = tmp_path / f"eval_opd_chartqa_human_q300_s{seed}.json"
        acc = 0.9 - residual  # a larger residual goes with a larger gap
        r.write_text(json.dumps({"accuracy": acc, "records": []}))
        results[(300, seed)] = r
    traj = trajectories(logs, window=10)
    assert traj["n_runs"] == 5 and traj["steps"] == [10, 20, 30, 40, 50]
    assert abs(traj["mass"]["chart_value"]["mean"][0] - 0.21) < 1e-9  # mean of the residuals
    rg = residual_vs_gap(logs, results, teacher_acc=0.95, final_window=20, n_boot=200)
    assert len(rg["points"]) == 5
    assert rg["correlations"]["residual_mass"]["spearman"] > 0.99
