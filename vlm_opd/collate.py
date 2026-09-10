"""Turn (image, question, response) rows into supervised training tensors for Qwen3-VL.

Design:
- The prompt (chat header + image tokens + question + assistant header) is encoded with the
  processor so image placeholder tokens are expanded correctly.
- The response is tokenized separately with the plain tokenizer and appended, followed by the
  end-of-turn token. Concatenating ids instead of tokenizing the joined string guarantees the
  prompt/response boundary is exact, so labels for every prompt and image position are -100 and
  only response tokens (plus the end-of-turn token) are supervised.
- The same `encode_prompt` is reused by the OPD trainer to build rollout inputs.
"""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from .common import DEFAULT_PROMPT_STYLE, build_messages

IGNORE_INDEX = -100
END_OF_TURN = "<|im_end|>"


def prompt_text(processor, question: str, answer: str | None = None, style: str = DEFAULT_PROMPT_STYLE) -> str:
    """Chat-template text for the user turn plus the assistant header (generation prompt)."""
    return processor.apply_chat_template(
        build_messages(question, answer, style), tokenize=False, add_generation_prompt=True
    )


def encode_prompt(processor, image: Image.Image, question: str, answer: str | None = None,
                  style: str = DEFAULT_PROMPT_STYLE) -> dict[str, torch.Tensor]:
    """Encode one prompt (single image) without padding. Returns 1-D `input_ids` plus vision tensors."""
    enc = processor(text=[prompt_text(processor, question, answer, style)], images=[image], return_tensors="pt")
    out: dict[str, torch.Tensor] = {
        "input_ids": enc["input_ids"][0],
        "pixel_values": enc["pixel_values"],
        "image_grid_thw": enc["image_grid_thw"],
    }
    if "mm_token_type_ids" in enc:
        out["mm_token_type_ids"] = enc["mm_token_type_ids"][0]
    return out


def encode_response(processor, response: str) -> torch.Tensor:
    """Tokenize the assistant response and close the turn with `<|im_end|>`."""
    text = response.strip() + END_OF_TURN + "\n"
    return processor.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]


def encode_example(
    processor,
    image: Image.Image,
    question: str,
    response: str,
    max_length: int | None = None,
    style: str = DEFAULT_PROMPT_STYLE,
) -> dict[str, torch.Tensor]:
    """Build one unpadded SFT example with labels masked on the prompt.

    Args:
        max_length: If set, the response is truncated so the total length fits; the prompt is never cut.
        style: Prompt wording (see `common.PROMPT_STYLES`).
    """
    prompt = encode_prompt(processor, image, question, style=style)
    prompt_ids = prompt["input_ids"]
    resp_ids = encode_response(processor, response)
    if max_length is not None:
        budget = max_length - prompt_ids.shape[0]
        if budget <= 0:
            raise ValueError(f"prompt alone has {prompt_ids.shape[0]} tokens, exceeding max_length={max_length}")
        resp_ids = resp_ids[:budget]

    input_ids = torch.cat([prompt_ids, resp_ids])
    labels = torch.cat([torch.full_like(prompt_ids, IGNORE_INDEX), resp_ids])
    example: dict[str, torch.Tensor] = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": torch.ones_like(input_ids),
        "pixel_values": prompt["pixel_values"],
        "image_grid_thw": prompt["image_grid_thw"],
        "prompt_len": torch.tensor(prompt_ids.shape[0]),
    }
    if "mm_token_type_ids" in prompt:
        example["mm_token_type_ids"] = torch.cat(
            [prompt["mm_token_type_ids"], torch.zeros_like(resp_ids)]
        )
    return example


def _pad_1d(seqs: list[torch.Tensor], value: int) -> torch.Tensor:
    """Right-pad a list of 1-D tensors into a 2-D batch."""
    max_len = max(s.shape[0] for s in seqs)
    out = torch.full((len(seqs), max_len), value, dtype=seqs[0].dtype)
    for i, s in enumerate(seqs):
        out[i, : s.shape[0]] = s
    return out


def collate_examples(examples: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, Any]:
    """Right-pad a list of encoded examples into a model-ready batch."""
    batch: dict[str, Any] = {
        "input_ids": _pad_1d([e["input_ids"] for e in examples], pad_token_id),
        "attention_mask": _pad_1d([e["attention_mask"] for e in examples], 0),
        "labels": _pad_1d([e["labels"] for e in examples], IGNORE_INDEX),
        # Qwen3-VL packs image patches along dim 0 and describes them with image_grid_thw
        "pixel_values": torch.cat([e["pixel_values"] for e in examples], dim=0),
        "image_grid_thw": torch.cat([e["image_grid_thw"] for e in examples], dim=0),
    }
    if "mm_token_type_ids" in examples[0]:
        batch["mm_token_type_ids"] = _pad_1d([e["mm_token_type_ids"] for e in examples], 0)
    return batch


class SFTCollator:
    """Data collator for the HF Trainer: raw dataset rows -> padded supervised batch."""

    def __init__(self, processor, max_length: int | None = 2048, response_key: str = "response",
                 prompt_style: str = DEFAULT_PROMPT_STYLE):
        self.processor = processor
        self.max_length = max_length
        self.response_key = response_key
        self.prompt_style = prompt_style
        self.pad_token_id = processor.tokenizer.pad_token_id

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        examples = [
            encode_example(self.processor, r["image"], r["question"], r[self.response_key], self.max_length, self.prompt_style)
            for r in rows
        ]
        batch = collate_examples(examples, self.pad_token_id)
        return batch
