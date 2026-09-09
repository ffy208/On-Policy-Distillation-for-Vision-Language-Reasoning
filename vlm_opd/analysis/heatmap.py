"""Render per-token teacher feedback as a text heatmap and per-class bar charts (matplotlib, headless)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .token_classes import CLASSES, classify_tokens

CLASS_COLORS = {"chart_value": "#1f77b4", "arithmetic": "#ff7f0e", "text": "#7f7f7f", "answer": "#2ca02c"}


def _wrap_tokens(pieces: list[str], width: int) -> list[list[tuple[int, str]]]:
    """Lay out token pieces on monospace rows of at most `width` characters; newlines force a new row."""
    rows: list[list[tuple[int, str]]] = [[]]
    col = 0
    for i, piece in enumerate(pieces):
        parts = piece.split("\n")
        for j, part in enumerate(parts):
            if j > 0:
                rows.append([]); col = 0
            if not part:
                continue
            if col + len(part) > width and col > 0:
                rows.append([]); col = 0
            rows[-1].append((i, part))
            col += len(part)
    return rows


def render_token_heatmap(
    pieces: list[str],
    values: list[float],
    out_png: str | Path,
    title: str = "",
    width: int = 90,
    vmax: float | None = None,
    classes: list[str] | None = None,
) -> Path:
    """Draw tokens in monospace with background intensity proportional to `values` (e.g. per-token KL).

    Token class is shown as a thin coloured underline so role and feedback strength are visible together.
    """
    classes = classes or classify_tokens(pieces)
    vmax = vmax or (max(values) if values else 1.0) or 1.0
    rows = _wrap_tokens(pieces, width)
    char_w, row_h = 0.115, 0.42
    fig_w = max(6.0, width * char_w + 0.6)
    fig_h = max(1.5, len(rows) * row_h + (0.8 if title else 0.4))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, width); ax.set_ylim(0, len(rows)); ax.axis("off")
    cmap = plt.get_cmap("Reds")
    for r, row in enumerate(rows):
        y = len(rows) - 1 - r
        col = 0
        for idx, part in row:
            v = min(values[idx] / vmax, 1.0)
            ax.add_patch(Rectangle((col, y + 0.08), len(part), 0.84, color=cmap(0.15 + 0.85 * v), lw=0))
            ax.add_patch(Rectangle((col, y + 0.02), len(part), 0.06, color=CLASS_COLORS[classes[idx]], lw=0))
            ax.text(col, y + 0.5, part, family="monospace", fontsize=9, va="center", ha="left")
            col += len(part)
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    handles = [Rectangle((0, 0), 1, 1, color=CLASS_COLORS[c]) for c in CLASSES]
    ax.legend(handles, CLASSES, loc="lower right", fontsize=7, ncol=len(CLASSES), frameon=False, bbox_to_anchor=(1.0, -0.02))
    fig.tight_layout()
    out_png = Path(out_png)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def render_class_bars(stats: dict[str, Any], out_png: str | Path, title: str = "") -> Path:
    """Two bars per class: share of tokens vs share of total KL mass, plus the concentration ratio on top."""
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    xs = range(len(CLASSES))
    shares = [stats["classes"][c]["token_share"] for c in CLASSES]
    masses = [stats["classes"][c]["kl_mass_share"] for c in CLASSES]
    w = 0.38
    ax.bar([x - w / 2 for x in xs], shares, w, label="share of tokens", color="#bbbbbb")
    ax.bar([x + w / 2 for x in xs], masses, w, label="share of teacher feedback (KL mass)",
           color=[CLASS_COLORS[c] for c in CLASSES])
    for x, c in zip(xs, CLASSES):
        r = stats["classes"][c]["concentration"]
        ax.text(x, max(shares[x], masses[x]) + 0.01, f"x{r:.2f}", ha="center", fontsize=8)
    ax.set_xticks(list(xs)); ax.set_xticklabels(CLASSES)
    ax.set_ylabel("share")
    ax.set_ylim(0, max(max(shares), max(masses)) * 1.25 + 0.02)
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_png = Path(out_png)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png
