"""Tests for token role classification and per-class aggregation."""

from vlm_opd.analysis.token_classes import (
    CLASSES,
    aggregate,
    classify_tokens,
    line_classes,
    markdown_table,
)


def test_line_classes_labels_answer_and_arithmetic():
    text = "The bar for 2019 is 45.\nSum: 45 + 30 = 75\nAnswer: 75"
    labels = [lab for _, _, lab in line_classes(text)]
    assert labels == ["other", "arithmetic", "answer"]


def test_classify_tokens_by_role():
    pieces = ["The", " bar", " for", " 2019", " is", " 45", ".", "\n", "Sum", ":", " 45", " +", " 30", " =", " 75", "\n", "Answer", ":", " 75"]
    classes = classify_tokens(pieces)
    assert len(classes) == len(pieces)
    assert classes[pieces.index(" 2019")] == "chart_value"
    assert classes[pieces.index(" 45")] == "chart_value"          # first line: read from the chart
    assert classes[pieces.index(" +")] == "arithmetic"
    assert classes[pieces.index(" 30")] == "arithmetic"           # operand on an arithmetic line
    assert classes[pieces.index("Sum")] == "text"                 # word on an arithmetic line stays text
    assert classes[-1] == "answer" and classes[-3] == "answer"
    assert classes[0] == "text"


def test_aggregate_shares_and_concentration():
    pieces = ["a", " 1", "\n", "Answer", ":", " 1"]
    ex = {"pieces": pieces, "kl": [0.1, 0.9, 0.0, 0.2, 0.2, 1.6], "logratio": [0, 1, 0, 0, 0, 2]}
    stats = aggregate([ex])
    assert stats["n_tokens"] == 6 and abs(stats["kl_total"] - 3.0) < 1e-9
    shares = sum(stats["classes"][c]["token_share"] for c in CLASSES)
    masses = sum(stats["classes"][c]["kl_mass_share"] for c in CLASSES)
    assert abs(shares - 1) < 1e-9 and abs(masses - 1) < 1e-9
    ans = stats["classes"]["answer"]
    assert ans["n_tokens"] == 3 and abs(ans["kl_mass_share"] - 2.0 / 3.0) < 1e-9
    assert ans["concentration"] > 1 and stats["classes"]["text"]["concentration"] < 1
    assert stats["classes"]["chart_value"]["logratio_mean"] == 1.0
    assert "| answer |" in markdown_table(stats)
