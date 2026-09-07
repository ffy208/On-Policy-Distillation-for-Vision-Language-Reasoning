"""Unit tests for common.py: answer parsing, relaxed accuracy, prompt construction, image resizing."""

from PIL import Image

from vlm_opd.common import (
    ANSWER_PREFIX,
    build_messages,
    build_prompt_text,
    image_pixel_bounds,
    parse_answer,
    parse_number,
    relaxed_accuracy,
    resize_image,
    score_predictions,
)

# ---------------- parse_answer ----------------


def test_parse_answer_basic():
    text = "The bar for 2019 is 45.\nAnswer: 45"
    assert parse_answer(text) == "45"


def test_parse_answer_takes_last_occurrence():
    text = "Answer: 10\nWait, let me recheck.\nAnswer: 12"
    assert parse_answer(text) == "12"


def test_parse_answer_strips_markdown_and_period():
    assert parse_answer("**Answer:** 3.5.") == "3.5"
    assert parse_answer("answer: Yes.") == "Yes"


def test_parse_answer_missing_returns_none():
    assert parse_answer("I think the value is 45.") is None
    assert parse_answer("") is None
    assert parse_answer("Answer:") is None


# ---------------- parse_number ----------------


def test_parse_number_formats():
    assert parse_number("1,234") == 1234.0
    assert parse_number("45%") == 45.0
    assert parse_number("$12.5") == 12.5
    assert parse_number("-3") == -3.0
    assert parse_number("12.5 million") == 12.5
    assert parse_number("Germany") is None
    assert parse_number("12 and 13") is None


# ---------------- relaxed_accuracy ----------------


def test_relaxed_numeric_within_tolerance():
    assert relaxed_accuracy("104", "100")  # 4% error
    assert relaxed_accuracy("95.5", "100")  # 4.5%
    assert not relaxed_accuracy("106", "100")  # 6% exceeds tolerance
    assert relaxed_accuracy("45", "45%")
    assert relaxed_accuracy("1,000", "1000")


def test_relaxed_zero_gold():
    assert relaxed_accuracy("0", "0")
    assert not relaxed_accuracy("0.1", "0")


def test_relaxed_text_match():
    assert relaxed_accuracy("Germany", "germany")
    assert relaxed_accuracy("  United   States ", "United States")
    assert relaxed_accuracy("Yes.", "yes")
    assert not relaxed_accuracy("Germany", "France")


def test_relaxed_none_pred_is_wrong():
    assert not relaxed_accuracy(None, "45")


def test_score_predictions_aggregates():
    outputs = ["Answer: 45", "Answer: Germany", "no format here", "Answer: 60"]
    golds = ["45", "germany", "12", "50"]
    res = score_predictions(outputs, golds)
    assert res["n"] == 4
    assert res["accuracy"] == 0.5
    assert res["format_rate"] == 0.75
    assert [r["correct"] for r in res["records"]] == [True, True, False, False]


# ---------------- prompt ----------------


def test_prompt_contains_question_and_answer_prefix():
    text = build_prompt_text("What is the value for 2020?")
    assert "What is the value for 2020?" in text
    assert ANSWER_PREFIX in text


def test_build_messages_structure():
    msgs = build_messages("Q?")
    assert msgs[0]["role"] == "user"
    types = [c["type"] for c in msgs[0]["content"]]
    assert types == ["image", "text"]


def test_privileged_prompt_includes_answer():
    text = build_prompt_text("Q?", answer="42")
    assert "42" in text


# ---------------- image ----------------


def test_resize_image_shrinks_long_side():
    img = Image.new("RGB", (1600, 800))
    out = resize_image(img, max_side=768)
    assert max(out.size) == 768
    assert out.size == (768, 384)


def test_resize_image_keeps_small_image():
    img = Image.new("RGBA", (400, 300))
    out = resize_image(img, max_side=768)
    assert out.size == (400, 300)
    assert out.mode == "RGB"


def test_image_pixel_bounds_matches_max_side():
    bounds = image_pixel_bounds()
    # A 768x768 image must fit within max_pixels, otherwise the processor would downscale it again
    assert 768 * 768 <= bounds["max_pixels"]
    assert bounds["min_pixels"] < bounds["max_pixels"]
