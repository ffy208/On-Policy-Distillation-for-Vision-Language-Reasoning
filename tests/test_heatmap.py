"""Rendering smoke tests for the token heatmap and class bar chart."""

import pytest

pytest.importorskip("matplotlib")

from vlm_opd.analysis.heatmap import render_class_bars, render_token_heatmap
from vlm_opd.analysis.token_classes import aggregate


def test_render_heatmap_and_bars(tmp_path):
    pieces = ["The", " bar", " for", " 2019", " is", " 45", ".", "\n", "Sum", ":", " 45", " +", " 30", " =", " 75", "\n", "Answer", ":", " 75"]
    kl = [0.05, 0.02, 0.01, 0.6, 0.02, 0.9, 0.01, 0.0, 0.1, 0.05, 0.3, 0.2, 0.4, 0.1, 1.2, 0.0, 0.1, 0.1, 1.5]
    png = render_token_heatmap(pieces, kl, tmp_path / "hm.png", title="example", width=30)
    assert png.exists() and png.stat().st_size > 1000
    stats = aggregate([{"pieces": pieces, "kl": kl}])
    bars = render_class_bars(stats, tmp_path / "bars.png", title="classes")
    assert bars.exists() and bars.stat().st_size > 1000
