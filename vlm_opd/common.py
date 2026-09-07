"""Logic shared by evaluation and training: prompt template, answer parsing, relaxed
accuracy scoring, and dataset loading.

Convention: every stage (teacher generation, SFT, OPD, evaluation) must import from here so
that there is exactly one implementation.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Sequence
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

DEFAULT_SEED: int = 42

STUDENT_MODEL: str = "Qwen/Qwen3-VL-2B-Instruct"
TEACHER_MODEL: str = "Qwen/Qwen3-VL-8B-Instruct"
TEACHER_MODEL_SMALL: str = "Qwen/Qwen3-VL-4B-Instruct"  # fallback teacher when GPU memory is tight

# Image preprocessing: longer side capped at 768 pixels
IMAGE_MAX_SIDE: int = 768
# Qwen3-VL uses 16-pixel patches with 2x2 spatial merging, so one visual token covers 32x32 pixels
PIXELS_PER_IMAGE_TOKEN: int = 32 * 32
MIN_IMAGE_TOKENS: int = 300
MAX_IMAGE_TOKENS: int = 600

# Fixed prefix for the final answer; every prompt and parser depends on it
ANSWER_PREFIX: str = "Answer:"

# Unified prompt template: reason first, then give the answer on the last line as `Answer: xxx`
PROMPT_TEMPLATE: str = (
    "Look at the chart and answer the question below.\n"
    "Reason step by step: first read the relevant values from the chart, "
    "then do any calculation needed. "
    "Keep the reasoning concise.\n"
    "On the last line, write the final answer in exactly this format:\n"
    f"{ANSWER_PREFIX} <your answer>\n\n"
    "Question: {question}"
)

# Privileged-information template seen by the teacher side in self-distillation (Stage 5)
PRIVILEGED_PROMPT_TEMPLATE: str = (
    PROMPT_TEMPLATE
    + "\n\nThe correct final answer is known to be: {answer}. "
    "Write out the reasoning that leads to it, and finish with the same "
    f"`{ANSWER_PREFIX} <your answer>` line."
)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_prompt_text(question: str, answer: str | None = None) -> str:
    """Build the text part of the prompt.

    Args:
        question: The question text.
        answer: If given, use the privileged template (teacher side of self-distillation).

    Returns:
        The filled prompt text (without the image placeholder).
    """
    if answer is None:
        return PROMPT_TEMPLATE.format(question=question.strip())
    return PRIVILEGED_PROMPT_TEMPLATE.format(question=question.strip(), answer=answer.strip())


def build_messages(question: str, answer: str | None = None) -> list[dict[str, Any]]:
    """Build the messages structure expected by the Qwen3-VL chat template (one image + text).

    The image itself is not placed in the messages; it is passed separately to the
    processor / vLLM. Only an `{"type": "image"}` placeholder goes here.

    Args:
        question: The question text.
        answer: If given, build the privileged prompt for the self-distillation teacher.

    Returns:
        A messages list that can be passed directly to `processor.apply_chat_template`.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": build_prompt_text(question, answer)},
            ],
        }
    ]


def image_pixel_bounds() -> dict[str, int]:
    """Return `min_pixels` / `max_pixels` for the Qwen3-VL processor, keeping image tokens in [300, 600]."""
    return {
        "min_pixels": MIN_IMAGE_TOKENS * PIXELS_PER_IMAGE_TOKEN,
        "max_pixels": MAX_IMAGE_TOKENS * PIXELS_PER_IMAGE_TOKEN,
    }


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------


def resize_image(image: Image.Image, max_side: int = IMAGE_MAX_SIDE) -> Image.Image:
    """Scale the image so its longer side is at most `max_side`, keeping aspect ratio.

    Images that are already small enough are returned unchanged (converted to RGB).
    """
    image = image.convert("RGB")
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.LANCZOS)


# ---------------------------------------------------------------------------
# Answer parsing and scoring
# ---------------------------------------------------------------------------

_ANSWER_LINE_RE = re.compile(rf"^\s*\**{re.escape(ANSWER_PREFIX)}\**\s*(.*?)\s*$", re.IGNORECASE)
_NUMBER_RE = re.compile(r"^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$")


def parse_answer(text: str) -> str | None:
    """Parse the answer from the last `Answer: xxx` line of a model output.

    Uses the last matching `Answer:` line (models occasionally write it more than once).
    Returns None when no such line exists; callers should count None as wrong instead of
    falling back to full-text matching, which would inflate accuracy for models that ignore
    the required format.
    """
    if not text:
        return None
    matched: str | None = None
    for line in text.splitlines():
        m = _ANSWER_LINE_RE.match(line)
        if m:
            matched = m.group(1)
    if matched is None:
        return None
    # Strip markdown bold markers and trailing periods
    matched = matched.strip().strip("*").strip()
    matched = matched.rstrip(".")
    return matched or None


def normalize_text(text: str) -> str:
    """Normalize a text answer: lowercase, trim, collapse whitespace, drop quotes and trailing period."""
    text = text.strip().lower()
    text = text.strip("\"'`")
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(".").strip()


def parse_number(text: str) -> float | None:
    """Best-effort numeric parsing supporting `1,234`, `45%`, `$12.5`, `12.5 million`, etc.

    Returns None when the string is not numeric.
    """
    s = text.strip().lower()
    s = s.replace(",", "").replace("$", "").replace("%", "").strip()
    # Drop common unit words (whole words only)
    s = re.sub(r"\b(million|billion|thousand|percent|years?|people|units?)\b", "", s).strip()
    if _NUMBER_RE.match(s):
        try:
            return float(s)
        except ValueError:
            return None
    return None


def relaxed_accuracy(pred: str | None, gold: str, tolerance: float = 0.05) -> bool:
    """ChartQA relaxed accuracy.

    - Numeric answers: correct if relative error is within `tolerance` (default 5%);
      a gold value of 0 requires the prediction to be exactly 0.
    - Text answers: exact match after normalization (case- and whitespace-insensitive).
    - A None prediction (no parsable answer) is always wrong.
    """
    if pred is None:
        return False
    gold_num = parse_number(gold)
    pred_num = parse_number(pred)
    if gold_num is not None and pred_num is not None:
        if gold_num == 0:
            return pred_num == 0
        return abs(pred_num - gold_num) / abs(gold_num) <= tolerance
    return normalize_text(pred) == normalize_text(gold)


def score_predictions(
    outputs: Sequence[str], golds: Sequence[str], tolerance: float = 0.05
) -> dict[str, Any]:
    """Score a batch; return accuracy, format-compliance rate, and per-sample records."""
    if len(outputs) != len(golds):
        raise ValueError(f"outputs ({len(outputs)}) and golds ({len(golds)}) differ in length")
    records: list[dict[str, Any]] = []
    n_correct = 0
    n_parsed = 0
    for out, gold in zip(outputs, golds):
        pred = parse_answer(out)
        correct = relaxed_accuracy(pred, gold, tolerance)
        n_correct += int(correct)
        n_parsed += int(pred is not None)
        records.append({"pred": pred, "gold": gold, "correct": correct})
    n = len(golds)
    return {
        "n": n,
        "accuracy": n_correct / n if n else 0.0,
        "format_rate": n_parsed / n if n else 0.0,
        "records": records,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def get_hf_token() -> str | None:
    """Read the HuggingFace token: Colab Secrets (`HF_TOKEN`) first, then environment variables.

    The token is never hard-coded.
    """
    try:
        from google.colab import userdata  # type: ignore
    except ImportError:
        userdata = None  # not running in Colab
    if userdata is not None:
        try:
            token = userdata.get("HF_TOKEN")
        except Exception as e:  # noqa: BLE001 - Colab raises SecretNotFoundError / NotebookAccessError;
            # in a subprocess `userdata.get` raises AttributeError because there is no frontend; fall back to env.
            logger.debug("Colab secret HF_TOKEN not readable here (%s); falling back to env", type(e).__name__)
            token = None
        if token:
            return token
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def load_hub_dataset(repo_id: str, split: str, token: str | None = None):
    """Load the sampled dataset from a private Hub repo (fields: id, image, question, answer).

    Args:
        repo_id: Dataset repo, e.g. "username/vlm_opd_chartqa".
        split: "train" or "test".
        token: HF token; when None, `get_hf_token()` is used.
    """
    from datasets import load_dataset

    token = token or get_hf_token()
    logger.info("Loading dataset %s [%s] from the Hub", repo_id, split)
    return load_dataset(repo_id, split=split, token=token)
