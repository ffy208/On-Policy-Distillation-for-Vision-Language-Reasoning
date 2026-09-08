"""Uncertainty for accuracy comparisons: bootstrap confidence intervals and paired bootstrap deltas.

All functions take per-question correctness (0/1) so that comparisons between two systems on the
same test set are paired, which is far tighter than comparing two independent intervals.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_correctness(result_json: str | Path) -> tuple[list[str], np.ndarray]:
    """Read an evaluate.py result file and return (question ids, 0/1 correctness in id order)."""
    data = json.loads(Path(result_json).read_text())
    records = sorted(data["records"], key=lambda r: r["id"])
    return [r["id"] for r in records], np.array([int(bool(r["correct"])) for r in records])


def bootstrap_ci(correct: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 42) -> dict[str, float]:
    """Percentile bootstrap interval for accuracy."""
    rng = np.random.default_rng(seed)
    n = len(correct)
    idx = rng.integers(0, n, size=(n_boot, n))
    accs = correct[idx].mean(axis=1)
    return {
        "accuracy": float(correct.mean()),
        "ci_low": float(np.quantile(accs, alpha / 2)),
        "ci_high": float(np.quantile(accs, 1 - alpha / 2)),
        "n": int(n),
    }


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 42) -> dict[str, Any]:
    """Paired bootstrap for accuracy(a) - accuracy(b) on the same questions.

    Also reports the sign-test style counts (a right & b wrong, a wrong & b right) and a
    one-sided bootstrap p-value for the delta being <= 0.
    """
    if a.shape != b.shape:
        raise ValueError("paired comparison needs the same questions in the same order")
    rng = np.random.default_rng(seed)
    n = len(a)
    diff = a.astype(float) - b.astype(float)
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = diff[idx].mean(axis=1)
    return {
        "delta": float(diff.mean()),
        "ci_low": float(np.quantile(deltas, alpha / 2)),
        "ci_high": float(np.quantile(deltas, 1 - alpha / 2)),
        "p_delta_le_0": float((deltas <= 0).mean()),
        "a_only_correct": int(((a == 1) & (b == 0)).sum()),
        "b_only_correct": int(((a == 0) & (b == 1)).sum()),
        "n": int(n),
    }


def compare_results(result_a: str | Path, result_b: str | Path, **kw) -> dict[str, Any]:
    """Paired comparison of two evaluate.py result files; ids must match exactly."""
    ids_a, a = load_correctness(result_a)
    ids_b, b = load_correctness(result_b)
    if ids_a != ids_b:
        raise ValueError("result files cover different question ids")
    return paired_bootstrap(a, b, **kw)


def summarize(result_json: str | Path, **kw) -> dict[str, float]:
    """Accuracy with a bootstrap interval for one result file."""
    _, c = load_correctness(result_json)
    return bootstrap_ci(c, **kw)
