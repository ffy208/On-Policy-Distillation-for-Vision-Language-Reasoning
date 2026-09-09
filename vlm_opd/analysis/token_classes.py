"""Classify generated tokens into coarse reasoning roles and aggregate per-token teacher feedback by class.

Classes (heuristic, documented in the README):
- answer:      tokens on the final `Answer:` line
- arithmetic:  digit or operator tokens on lines that contain an arithmetic cue (operator or word such as
               "sum", "average", "difference"); operands and results are not separated
- chart_value: digit tokens on lines without an arithmetic cue, i.e. values read off the chart
- text:        everything else (connectives, words, punctuation, whitespace)
"""

from __future__ import annotations

import re
from typing import Any

from ..common import ANSWER_PREFIX

CLASSES = ("chart_value", "arithmetic", "text", "answer")

_ARITH_CUE = re.compile(
    r"[+*/×÷=]|(?<=\d)\s*-\s*(?=\d)|\b(sum|total|average|mean|difference|subtract|add(ed|ing)?|divid|multipl|"
    r"ratio|minus|plus|times|increase|decrease|change|calculat|compute)\w*\b",
    re.IGNORECASE,
)
_DIGIT = re.compile(r"\d")
_OPERATOR = re.compile(r"^[\s]*[+\-*/×÷=%]+[\s]*$")


def token_spans(pieces: list[str]) -> list[tuple[int, int]]:
    """Character span of each decoded token piece in the concatenated text."""
    spans, pos = [], 0
    for p in pieces:
        spans.append((pos, pos + len(p)))
        pos += len(p)
    return spans


def line_classes(text: str) -> list[tuple[int, int, str]]:
    """Split text into lines and label each line as 'answer', 'arithmetic', or 'other' with its char span."""
    out, pos = [], 0
    lines = text.split("\n")
    answer_idx = None
    for i, line in enumerate(lines):
        if line.strip().lstrip("*").lower().startswith(ANSWER_PREFIX.lower()):
            answer_idx = i
    for i, line in enumerate(lines):
        end = pos + len(line) + (1 if i < len(lines) - 1 else 0)
        if i == answer_idx:
            label = "answer"
        elif _ARITH_CUE.search(line):
            label = "arithmetic"
        else:
            label = "other"
        out.append((pos, end, label))
        pos = end
    return out


def classify_tokens(pieces: list[str]) -> list[str]:
    """Assign one class per decoded token piece."""
    text = "".join(pieces)
    lines = line_classes(text)
    classes = []
    for (start, end), piece in zip(token_spans(pieces), pieces):
        mid = min(max(start, 0), max(len(text) - 1, 0))
        line_label = next((lab for s, e, lab in lines if s <= mid < e), "other")
        has_digit = bool(_DIGIT.search(piece))
        is_operator = bool(_OPERATOR.match(piece)) and piece.strip() != ""
        if line_label == "answer":
            classes.append("answer")
        elif line_label == "arithmetic" and (has_digit or is_operator):
            classes.append("arithmetic")
        elif has_digit:
            classes.append("chart_value")
        else:
            classes.append("text")
    return classes


def aggregate(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-class token share, KL mass share, mean KL, mean log-ratio, and concentration (mass share / token share).

    Each example needs `pieces` (decoded tokens), `kl` (per-token KL), and optionally `logratio`.
    """
    totals = {c: {"n_tokens": 0, "kl_sum": 0.0, "logratio_sum": 0.0} for c in CLASSES}
    for ex in examples:
        classes = ex.get("classes") or classify_tokens(ex["pieces"])
        for i, c in enumerate(classes):
            totals[c]["n_tokens"] += 1
            totals[c]["kl_sum"] += float(ex["kl"][i])
            if "logratio" in ex:
                totals[c]["logratio_sum"] += float(ex["logratio"][i])
    n_all = sum(t["n_tokens"] for t in totals.values())
    kl_all = sum(t["kl_sum"] for t in totals.values())
    out: dict[str, Any] = {"n_examples": len(examples), "n_tokens": n_all, "kl_total": kl_all, "classes": {}}
    for c, t in totals.items():
        share = t["n_tokens"] / n_all if n_all else 0.0
        mass = t["kl_sum"] / kl_all if kl_all else 0.0
        out["classes"][c] = {
            "n_tokens": t["n_tokens"],
            "token_share": share,
            "kl_mass_share": mass,
            "kl_mean": t["kl_sum"] / t["n_tokens"] if t["n_tokens"] else 0.0,
            "logratio_mean": t["logratio_sum"] / t["n_tokens"] if t["n_tokens"] else 0.0,
            "concentration": mass / share if share else 0.0,
        }
    return out


def markdown_table(stats: dict[str, Any]) -> str:
    """Render the per-class statistics as a Markdown table."""
    lines = ["| Token class | Tokens | Token share | KL mass share | Mean KL | Concentration |", "|---|---|---|---|---|---|"]
    for c in CLASSES:
        r = stats["classes"][c]
        lines.append(f"| {c} | {r['n_tokens']} | {r['token_share']:.1%} | {r['kl_mass_share']:.1%} | {r['kl_mean']:.3f} | {r['concentration']:.2f} |")
    return "\n".join(lines)
