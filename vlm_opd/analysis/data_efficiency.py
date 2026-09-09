"""Aggregate data-efficiency results (accuracy vs number of training questions) into a table and a figure."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .stats import compare_results, summarize

METHODS = ("sft", "opd")


def result_path(out_dir: str | Path, method: str, budget: int) -> Path:
    """Canonical location of an evaluate.py result for one (method, budget) point."""
    return Path(out_dir) / f"eval_{method}_q{budget}.json"


def collect(out_dir: str | Path, budgets: list[int], baseline_json: str | Path | None = None,
            teacher_json: str | Path | None = None, n_boot: int = 10000) -> dict[str, Any]:
    """Read every available result and compute per-point intervals plus paired OPD-vs-SFT deltas."""
    points: list[dict[str, Any]] = []
    for budget in budgets:
        row: dict[str, Any] = {"budget": budget}
        for method in METHODS:
            path = result_path(out_dir, method, budget)
            row[method] = summarize(path, n_boot=n_boot) if path.exists() else None
        if row["sft"] and row["opd"]:
            row["opd_minus_sft"] = compare_results(result_path(out_dir, "opd", budget), result_path(out_dir, "sft", budget), n_boot=n_boot)
        points.append(row)
    table: dict[str, Any] = {"points": points}
    if baseline_json and Path(baseline_json).exists():
        table["baseline"] = summarize(baseline_json, n_boot=n_boot)
    if teacher_json and Path(teacher_json).exists():
        table["teacher"] = summarize(teacher_json, n_boot=n_boot)
    return table


def crossover(table: dict[str, Any]) -> dict[str, Any] | None:
    """Smallest OPD budget whose accuracy interval reaches the full-data SFT accuracy (data-efficiency claim)."""
    full = max(p["budget"] for p in table["points"])
    full_sft = next((p["sft"] for p in table["points"] if p["budget"] == full), None)
    if not full_sft:
        return None
    for p in sorted(table["points"], key=lambda r: r["budget"]):
        if p["opd"] and p["opd"]["ci_high"] >= full_sft["accuracy"]:
            return {"opd_budget": p["budget"], "opd_accuracy": p["opd"]["accuracy"], "full_sft_accuracy": full_sft["accuracy"],
                    "fraction_of_data": p["budget"] / full}
    return None


def markdown_table(table: dict[str, Any]) -> str:
    """Render the results as a Markdown table with 95% intervals."""
    def fmt(r):
        return "pending" if not r else f"{r['accuracy']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"

    lines = ["| Questions | SFT | OPD | OPD - SFT (paired 95% CI) |", "|---|---|---|---|"]
    for p in table["points"]:
        d = p.get("opd_minus_sft")
        delta = f"{d['delta']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]" if d else "pending"
        lines.append(f"| {p['budget']} | {fmt(p['sft'])} | {fmt(p['opd'])} | {delta} |")
    if table.get("baseline"):
        lines.append(f"| 0 (zero-shot) | {fmt(table['baseline'])} | same | |")
    if table.get("teacher"):
        lines.append(f"| teacher | {fmt(table['teacher'])} | | |")
    return "\n".join(lines)


def plot(table: dict[str, Any], out_png: str | Path) -> Path:
    """Accuracy vs training questions with interval bands, plus baseline and teacher reference lines."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for method, label in (("sft", "SFT (supervised distillation)"), ("opd", "OPD (on-policy distillation)")):
        pts = [(p["budget"], p[method]) for p in table["points"] if p[method]]
        if not pts:
            continue
        xs = [b for b, _ in pts]
        ys = [r["accuracy"] for _, r in pts]
        lo = [r["accuracy"] - r["ci_low"] for _, r in pts]
        hi = [r["ci_high"] - r["accuracy"] for _, r in pts]
        ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, label=label)
    if table.get("baseline"):
        ax.axhline(table["baseline"]["accuracy"], ls=":", color="gray", label="student zero-shot")
    if table.get("teacher"):
        ax.axhline(table["teacher"]["accuracy"], ls="--", color="black", label="teacher zero-shot")
    ax.set_xscale("log")
    ax.set_xticks([p["budget"] for p in table["points"]])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([p["budget"] for p in table["points"]], [str(p["budget"]) for p in table["points"]])
    ax.set_xlabel("training questions")
    ax.set_ylabel("relaxed accuracy (500 test questions)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = Path(out_png)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def write_report(table: dict[str, Any], out_dir: str | Path) -> Path:
    """Save the table json, the markdown table, and the figure; return the markdown path."""
    out_dir = Path(out_dir)
    (out_dir / "data_efficiency.json").write_text(json.dumps(table, indent=2))
    md = markdown_table(table)
    cross = crossover(table)
    if cross:
        md += (f"\n\nSmallest OPD budget whose 95% interval reaches full-data SFT ({cross['full_sft_accuracy']:.3f}): "
               f"{cross['opd_budget']} questions ({cross['fraction_of_data']:.0%} of the data), OPD accuracy {cross['opd_accuracy']:.3f}.")
    (out_dir / "data_efficiency.md").write_text(md + "\n")
    plot(table, out_dir / "data_efficiency.png")
    return out_dir / "data_efficiency.md"
