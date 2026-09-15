"""Feedback dynamics across OPD runs: where the teacher's KL lands over training, and whether the residual
predicts the remaining gap to the teacher.

Inputs are the per-step `opd_log.jsonl` files the trainer pushes to each checkpoint repo (one row per step
with `role_kl_mass`, `role_concentration`, `eos_rate`, `format_rate`, `kl`) and the evaluation json of the
same run. Two outputs per task:

- trajectories: mean and s.d. across runs of the KL-mass share and concentration per token role, in windows of
  `window` steps (figure 2 of the paper plan);
- residual vs gap: per run, the mean over the final window of a residual statistic (default: KL-mass share on
  numeric-value tokens, `chart_value`) against the remaining accuracy gap to the teacher, with Spearman and
  Pearson correlations and a bootstrap interval over runs (figure 3).

    python -m vlm_opd.analysis.dynamics --task chartqa_human
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from ..experiment import baseline_result_name, load_tasks

ROLES = ("answer", "chart_value", "arithmetic", "text")
_RUN = re.compile(r"^(?P<method>opd(?:-[a-z0-9]+)?)_(?P<task>.+?)_q(?P<budget>\d+)_s(?P<seed>\d+)$")
_LEGACY = re.compile(r"^opd_human_q(?P<budget>\d+)$")  # notebook-era ChartQA seed-42 runs


def load_log(path: str | Path) -> list[dict[str, Any]]:
    """Rows of one run in step order. A resumed run appends its steps again; the last record per step wins."""
    by_step: dict[int, dict[str, Any]] = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            by_step[int(r["step"])] = r
    return [by_step[s] for s in sorted(by_step)]


def parse_run_name(name: str, legacy_seed: int = 42) -> dict[str, Any] | None:
    m = _RUN.match(name)
    if m:
        return {"method": m["method"], "task": m["task"], "budget": int(m["budget"]), "seed": int(m["seed"])}
    m = _LEGACY.match(name)
    if m:
        return {"method": "opd", "task": "chartqa_human", "budget": int(m["budget"]), "seed": legacy_seed}
    return None


def find_logs(log_dir: str | Path, task: str, method: str = "opd") -> dict[tuple[int, int], Path]:
    """(budget, seed) -> log path for one task and method."""
    out: dict[tuple[int, int], Path] = {}
    for p in sorted(Path(log_dir).glob("*.jsonl")):
        info = parse_run_name(p.stem)
        if info and info["task"] == task and info["method"] == method:
            out[(info["budget"], info["seed"])] = p
    return out


def windowed(rows: list[dict[str, Any]], key: str, role: str | None, window: int) -> list[float]:
    """Mean of rows[key][role] (or rows[key]) over consecutive windows of `window` steps."""
    vals = [(r[key][role] if role else r[key]) for r in rows if key in r and (role is None or role in r[key])]
    if not vals:
        return []
    return [float(np.mean(vals[i:i + window])) for i in range(0, len(vals), window)]


def trajectories(logs: dict[tuple[int, int], Path], window: int = 10) -> dict[str, Any]:
    """Across-run mean and s.d. per window of KL-mass share and concentration for every role, plus EOS and KL."""
    per_run = [load_log(p) for p in logs.values()]
    per_run = [r for r in per_run if r and "role_kl_mass" in r[-1]]
    if not per_run:
        return {"n_runs": 0}
    n_win = min(len(windowed(r, "kl", None, window)) for r in per_run)

    def agg(key, role):
        m = np.array([windowed(r, key, role, window)[:n_win] for r in per_run])
        return {"mean": m.mean(0).tolist(), "sd": m.std(0, ddof=1).tolist() if len(m) > 1 else [0.0] * n_win}

    out: dict[str, Any] = {"n_runs": len(per_run), "window": window, "steps": [window * (i + 1) for i in range(n_win)],
                           "kl": agg("kl", None), "eos_rate": agg("eos_rate", None), "format_rate": agg("format_rate", None),
                           "mass": {}, "concentration": {}}
    for role in ROLES:
        out["mass"][role] = agg("role_kl_mass", role)
        out["concentration"][role] = agg("role_concentration", role)
    return out


def _rank(v: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank), as Spearman's correlation requires."""
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), dtype=float)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rank(x), _rank(y))


def _bootstrap_corr(x: np.ndarray, y: np.ndarray, n_boot: int = 10000, seed: int = 0) -> dict[str, float]:
    """Spearman and Pearson correlation with a percentile bootstrap interval over runs for Spearman."""
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(idx.tolist())) < 3:
            continue
        boots.append(_spearman(x[idx], y[idx]))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"spearman": _spearman(x, y), "pearson": _pearson(x, y), "spearman_ci_low": float(lo),
            "spearman_ci_high": float(hi), "n": n}


def residual_vs_gap(logs: dict[tuple[int, int], Path], results: dict[tuple[int, int], Path], teacher_acc: float,
                    residual_role: str = "chart_value", final_window: int = 25, n_boot: int = 10000) -> dict[str, Any]:
    """Per run: final-window residual statistics and the remaining gap to the teacher; correlations across runs."""
    points = []
    for key, log_path in logs.items():
        if key not in results:
            continue
        rows = [r for r in load_log(log_path) if "role_kl_mass" in r]
        if len(rows) < final_window:
            continue
        tail = rows[-final_window:]
        acc = json.loads(Path(results[key]).read_text())["accuracy"]
        points.append({
            "budget": key[0], "seed": key[1], "accuracy": acc, "gap_to_teacher": teacher_acc - acc,
            "residual_mass": float(np.mean([r["role_kl_mass"].get(residual_role, 0.0) for r in tail])),
            "residual_concentration": float(np.mean([r["role_concentration"].get(residual_role, 0.0) for r in tail])),
            "answer_mass": float(np.mean([r["role_kl_mass"].get("answer", 0.0) for r in tail])),
            "final_kl": float(np.mean([r["kl"] for r in tail])),
            "final_eos_rate": float(np.mean([r["eos_rate"] for r in tail])),
        })
    out: dict[str, Any] = {"residual_role": residual_role, "final_window": final_window, "teacher_accuracy": teacher_acc,
                           "points": points, "correlations": {}}
    if len(points) >= 4:
        gap = np.array([p["gap_to_teacher"] for p in points])
        for stat in ("residual_mass", "residual_concentration", "answer_mass", "final_kl", "final_eos_rate"):
            out["correlations"][stat] = _bootstrap_corr(np.array([p[stat] for p in points]), gap, n_boot)
    return out


def plot_trajectories(traj: dict[str, Any], out_png: str | Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    steps = traj["steps"]
    for role, color in zip(ROLES, ("#d62728", "#ff7f0e", "#2ca02c", "#7f7f7f")):
        c = traj["concentration"][role]
        axes[0].plot(steps, c["mean"], marker="o", ms=3, color=color, label=role)
        axes[0].fill_between(steps, np.array(c["mean"]) - np.array(c["sd"]), np.array(c["mean"]) + np.array(c["sd"]), color=color, alpha=0.15)
    axes[0].axhline(1.0, ls=":", color="black", lw=1)
    axes[0].set_xlabel("OPD step"); axes[0].set_ylabel("KL concentration (mass share / token share)")
    axes[0].set_title(f"{title}: where the teacher's KL lands (n={traj['n_runs']} runs)", fontsize=9)
    axes[0].legend(fontsize=8)
    axes[1].plot(steps, traj["kl"]["mean"], marker="o", ms=3, color="black", label="mean per-token KL")
    ax2 = axes[1].twinx()
    ax2.plot(steps, traj["eos_rate"]["mean"], marker="s", ms=3, color="#1f77b4", label="rollout EOS rate")
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("EOS rate", color="#1f77b4")
    axes[1].set_xlabel("OPD step"); axes[1].set_ylabel("KL")
    axes[1].set_title("KL and termination over training", fontsize=9)
    h1, l1 = axes[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    axes[1].legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_residual_vs_gap(rg: dict[str, Any], out_png: str | Path, title: str, stat: str = "residual_mass") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    pts = rg["points"]
    budgets = sorted({p["budget"] for p in pts})
    cmap = plt.get_cmap("viridis", max(len(budgets), 2))
    for i, b in enumerate(budgets):
        sel = [p for p in pts if p["budget"] == b]
        ax.scatter([p[stat] for p in sel], [p["gap_to_teacher"] for p in sel], color=cmap(i), label=f"{b} questions", s=36)
    c = rg["correlations"].get(stat)
    if c:
        ax.set_title(f"{title}: Spearman {c['spearman']:+.2f} [{c['spearman_ci_low']:+.2f}, {c['spearman_ci_high']:+.2f}], n={c['n']}", fontsize=9)
    ax.set_xlabel(f"final-window {stat.replace('_', ' ')} ({rg['residual_role']})")
    ax.set_ylabel("remaining gap to teacher (accuracy)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    import argparse

    from .data_efficiency import find_runs
    from .stats import summarize

    parser = argparse.ArgumentParser(description="Feedback dynamics across OPD runs of one task")
    parser.add_argument("--task", default="chartqa_human")
    parser.add_argument("--method", default="opd")
    parser.add_argument("--log-dir", default="outputs/opd_logs")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--final-window", type=int, default=25)
    parser.add_argument("--residual-role", default="chart_value")
    args = parser.parse_args()

    task_cfg = load_tasks()["tasks"][args.task]
    logs = find_logs(args.log_dir, args.task, args.method)
    runs = find_runs(args.out_dir, args.task)
    results = {(budget, seed): path for (method, budget), files in runs.items() if method == args.method for seed, path in files.items()}
    teacher_json = Path(args.out_dir) / baseline_result_name(args.task, task_cfg, "teacher")
    teacher_acc = summarize(teacher_json, n_boot=100)["accuracy"] if teacher_json.exists() else float("nan")

    traj = trajectories(logs, args.window)
    rg = residual_vs_gap(logs, results, teacher_acc, args.residual_role, args.final_window)
    out = Path(args.out_dir)
    (out / f"dynamics_{args.task}_{args.method}.json").write_text(json.dumps({"trajectories": traj, "residual_vs_gap": rg}, indent=2))
    if traj.get("n_runs"):
        plot_trajectories(traj, out / f"dynamics_{args.task}_{args.method}.png", args.task)
    if rg["points"]:
        plot_residual_vs_gap(rg, out / f"residual_vs_gap_{args.task}_{args.method}.png", args.task)
    print(f"{args.task} / {args.method}: {traj.get('n_runs', 0)} runs with role logs, {len(rg['points'])} runs with results")
    if traj.get("n_runs"):
        first, last = 0, -1
        for role in ROLES:
            c = traj["concentration"][role]["mean"]
            print(f"  concentration {role:12s} first window {c[first]:.2f} -> last window {c[last]:.2f}")
        print(f"  eos rate {traj['eos_rate']['mean'][0]:.2f} -> {traj['eos_rate']['mean'][-1]:.2f}; kl {traj['kl']['mean'][0]:.3f} -> {traj['kl']['mean'][-1]:.3f}")
    for stat, c in rg["correlations"].items():
        print(f"  gap ~ {stat:24s} spearman {c['spearman']:+.2f} [{c['spearman_ci_low']:+.2f}, {c['spearman_ci_high']:+.2f}]  pearson {c['pearson']:+.2f}  (n={c['n']})")


if __name__ == "__main__":
    main()
