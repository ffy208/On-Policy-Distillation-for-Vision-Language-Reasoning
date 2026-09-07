"""Tests for the SFT collator using the real Qwen3-VL processor on a synthetic image.

Requires the processor files (downloaded once from the Hub, ~10 MB); skipped when unavailable.
"""

import pytest
from PIL import Image

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from vlm_opd.collate import (
    END_OF_TURN,
    IGNORE_INDEX,
    SFTCollator,
    collate_examples,
    encode_example,
    encode_prompt,
)
from vlm_opd.common import STUDENT_MODEL, image_pixel_bounds


@pytest.fixture(scope="module")
def processor():
    from transformers import AutoProcessor

    try:
        return AutoProcessor.from_pretrained(STUDENT_MODEL, **image_pixel_bounds())
    except Exception as e:  # noqa: BLE001 - offline machines skip
        pytest.skip(f"processor unavailable: {type(e).__name__}")


def _img(w=768, h=512):
    return Image.new("RGB", (w, h), color=(200, 30, 30))


def test_prompt_image_tokens_within_budget(processor):
    enc = encode_prompt(processor, _img(), "What is the value for 2020?")
    n_img = int((enc["input_ids"] == processor.image_token_id).sum())
    assert 300 <= n_img <= 600
    assert enc["image_grid_thw"].shape == (1, 3)


def test_labels_mask_prompt_and_supervise_response(processor):
    response = "Step 1: read 45.\nAnswer: 45"
    ex = encode_example(processor, _img(), "Q?", response)
    n = ex["input_ids"].shape[0]
    plen = int(ex["prompt_len"])
    assert ex["labels"].shape[0] == n and ex["attention_mask"].sum() == n
    # everything in the prompt (incl. all image tokens) is ignored
    assert (ex["labels"][:plen] == IGNORE_INDEX).all()
    img_pos = ex["input_ids"] == processor.image_token_id
    assert (ex["labels"][img_pos] == IGNORE_INDEX).all()
    # the response tokens are supervised and decode back to the response + end-of-turn
    sup = ex["labels"][plen:]
    assert (sup != IGNORE_INDEX).all()
    decoded = processor.tokenizer.decode(sup)
    assert decoded.startswith(response) and END_OF_TURN in decoded
    # the prompt ids are exactly the prompt-only encoding (no boundary re-tokenization)
    prompt_only = encode_prompt(processor, _img(), "Q?")["input_ids"]
    assert torch.equal(ex["input_ids"][:plen], prompt_only)


def test_max_length_truncates_only_response(processor):
    plen = encode_prompt(processor, _img(), "Q?")["input_ids"].shape[0]
    ex = encode_example(processor, _img(), "Q?", "word " * 200, max_length=plen + 5)
    assert ex["input_ids"].shape[0] == plen + 5
    assert int(ex["prompt_len"]) == plen
    with pytest.raises(ValueError):
        encode_example(processor, _img(), "Q?", "x", max_length=plen - 1)


def test_collate_pads_and_packs_images(processor):
    a = encode_example(processor, _img(768, 512), "short?", "Answer: 1")
    b = encode_example(processor, _img(512, 768), "a much longer question about the chart?", "Some reasoning first.\nAnswer: 2")
    batch = collate_examples([a, b], processor.tokenizer.pad_token_id)
    n = max(a["input_ids"].shape[0], b["input_ids"].shape[0])
    assert batch["input_ids"].shape == (2, n)
    assert batch["labels"].shape == (2, n) and batch["attention_mask"].shape == (2, n)
    # padded positions: pad id, attention 0, label ignored
    short = a if a["input_ids"].shape[0] < b["input_ids"].shape[0] else b
    idx = 0 if short is a else 1
    L = short["input_ids"].shape[0]
    assert (batch["input_ids"][idx, L:] == processor.tokenizer.pad_token_id).all()
    assert (batch["attention_mask"][idx, L:] == 0).all()
    assert (batch["labels"][idx, L:] == IGNORE_INDEX).all()
    # two images packed along dim 0
    assert batch["image_grid_thw"].shape == (2, 3)
    assert batch["pixel_values"].shape[0] == a["pixel_values"].shape[0] + b["pixel_values"].shape[0]
    if "mm_token_type_ids" in batch:
        assert batch["mm_token_type_ids"].shape == (2, n)


def test_sft_collator_on_raw_rows(processor):
    rows = [
        {"image": _img(), "question": "Q1?", "response": "Answer: 1"},
        {"image": _img(640, 640), "question": "Q2?", "response": "Answer: 2"},
    ]
    batch = SFTCollator(processor, max_length=2048)(rows)
    assert set(batch) >= {"input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"}
    assert batch["input_ids"].shape[0] == 2
