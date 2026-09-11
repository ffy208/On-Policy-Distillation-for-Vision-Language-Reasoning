"""Out-of-distribution table: every trained model of a task scored on each OOD test set.

For each OOD set the table lists the zero-shot student and teacher, then SFT and OPD at every budget
(mean over seeds, interval pooled over questions, paired OPD - SFT over the shared seeds), and the
in-distribution accuracy of the same models for reference, so the robustness-under-shift question
("does OPD's lead grow, shrink, or vanish off-distribution?") is answered per set.

    python -m vlm_opd.analysis.ood --task chartqa_human --budgets 100 300 900 3000
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..experiment import baseline_result_name, load_tasks
from .data_efficiency import SMOKE_SEEDS, _paired_over_seeds, _pooled
from .stats import summarize

_SEEDED = re.compile(r"^eval_(?P<method>sft|opd)_(?P<task>.+?)_q(?P<budget>\d+)_s(?P<seed>\d+)_on_(?P<set>.+)\.json$")
_LEGACY = re.compile(r"^eval_(?P<method>sft|opd)_q(?P<budget>\d+)_on_(?P<set>.+)\.json$")


def find_ood_runs(out_dir: str | Path, task: str, ood_set: str, legacy_seed: int = 42,
                  exclude_seeds: tuple[int, ...] = SMOKE_SEEDS) -> dict[tuple[str, int], dict[int, Path]]:
    """Map (method, budget) -> {seed: result file} for one task on one OOD set."""
    runs: dict[tuple[str, int], dict[int, Path]] = {}
    for p in Path(out_dir).glob("eval_*_on_*.json"):
        m = _SEEDED.match(p.name)
        if m and m["task"] == task and m["set"] == ood_set:
            if int(m["seed"]) not in exclude_seeds:
                runs.setdefault((m["method"], int(m["budget"])), {})[int(m["seed"])] = p
            continue
        m = _LEGACY.match(p.name)
        if m and task == "chartqa_human" and m["set"] == ood_set:
            runs.setdefault((m["method"], int(m["budget"])), {}).setdefault(legacy_seed, p)
    return runs


def collect_ood(out_dir: str | Path, task: str, budgets: list[int], tasks_file: str | Path | None = None,
                n_boot: int = 10000) -> dict[str, Any]:
    """One block per OOD set of the task: zero-shot rows plus SFT/OPD rows per budget."""
    from .data_efficiency import find_runs

    cfg = load_tasks(tasks_file) if tasks_file else load_tasks()
    task_cfg = cfg["tasks"][task]
    out = Path(out_dir)
    id_runs = find_runs(out, task)
    blocks: list[dict[str, Any]] = []
    for ood_repo in task_cfg.get("ood_eval", []):
        ood_set = ood_repo.split("/")[-1]
        runs = find_ood_runs(out, task, ood_set)
        block: dict[str, Any] = {"set": ood_set, "zeroshot": {}, "points": []}
        for role in ("student", "teacher"):
            f = out / baseline_result_name(task, task_cfg, role, ood_repo)
            if f.exists():
                block["zeroshot"][role] = summarize(f, n_boot=n_boot)
        for budget in budgets:
            row: dict[str, Any] = {"budget": budget}
            for method in ("sft", "opd"):
                files = runs.get((method, budget))
                row[method] = _pooled(files, n_boot) if files else None
                id_files = id_runs.get((method, budget))
                row[f"{method}_id"] = _pooled(id_files, n_boot)["accuracy"] if id_files else None
            if runs.get(("opd", budget)) and runs.get(("sft", budget)):
                row["opd_minus_sft"] = _paired_over_seeds(runs[("opd", budget)], runs[("sft", budget)], n_boot)
            block["points"].append(row)
        blocks.append(block)
    return {"task": task, "blocks": blocks}


def markdown_ood(table: dict[str, Any]) -> str:
    """One markdown table per OOD set."""
    def fmt(r: dict[str, Any] | None, id_acc: float | None = None) -> str:
        if not r:
            return "pending"
        seeds = f" (n={len(r['seeds'])})" if "seeds" in r and len(r["seeds"]) > 1 else ""
        ref = f", ID {id_acc:.3f}" if id_acc is not None else ""
        return f"{r['accuracy']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]{seeds}{ref}"

    parts = []
    for b in table["blocks"]:
        lines = [f"### {b['set']}", "", "| Model | Accuracy on the OOD set (95% CI, seeds, in-distribution accuracy) | OPD - SFT (paired) |", "|---|---|---|"]
        for role in ("student", "teacher"):
            if role in b["zeroshot"]:
                lines.append(f"| {role} zero-shot | {fmt(b['zeroshot'][role])} | |")
        for p in b["points"]:
            d = p.get("opd_minus_sft")
            delta = f"{d['delta']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] over {len(d['seeds'])} seed(s)" if d else "pending"
            lines.append(f"| SFT, {p['budget']} questions | {fmt(p['sft'], p['sft_id'])} | |")
            lines.append(f"| OPD, {p['budget']} questions | {fmt(p['opd'], p['opd_id'])} | {delta} |")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate OOD evaluations of one task's models")
    parser.add_argument("--task", default="chartqa_human")
    parser.add_argument("--budgets", type=int, nargs="+", default=[100, 300, 900, 3000])
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()
    table = collect_ood(args.out_dir, args.task, args.budgets, n_boot=args.n_boot)
    out = Path(args.out_dir)
    (out / f"ood_{args.task}.json").write_text(json.dumps(table, indent=2))
    md = markdown_ood(table)
    (out / f"ood_{args.task}.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
