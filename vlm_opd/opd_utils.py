"""Pure tensor helpers for on-policy distillation: rollout inputs, generation masks, KL, batching.

Kept free of model loading so everything here is unit-testable on CPU with a tiny model.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .collate import prompt_text
from .common import DEFAULT_PROMPT_STYLE

KL_DIRECTIONS = ("reverse", "forward")


def build_rollout_inputs(processor, rows: list[dict[str, Any]], style: str = DEFAULT_PROMPT_STYLE) -> dict[str, torch.Tensor]:
    """Left-padded prompt batch so every prompt ends at the same position and generation appends cleanly."""
    tok = processor.tokenizer
    old_side = tok.padding_side
    tok.padding_side = "left"
    try:
        enc = processor(
            text=[prompt_text(processor, r["question"], style=style) for r in rows],
            images=[r["image"] for r in rows],
            padding=True,
            return_tensors="pt",
        )
    finally:
        tok.padding_side = old_side
    return dict(enc)


def generation_mask(sequences: torch.Tensor, prompt_len: int, eos_id: int, pad_id: int) -> torch.Tensor:
    """Boolean mask over the generated span: True for tokens up to and including the first EOS.

    `sequences` is the full (B, P+G) output of `generate`; the returned mask has shape (B, G).
    Padding that `generate` appends after a finished sequence is excluded, as is anything after
    the first EOS. A sequence that never emitted EOS (hit max_new_tokens) is fully valid.
    """
    gen = sequences[:, prompt_len:]
    is_eos = gen == eos_id
    eos_before = is_eos.long().cumsum(dim=1) - is_eos.long()  # EOS tokens strictly before each position
    return (eos_before == 0) & (gen != pad_id)


def full_attention_mask(prompt_attention: torch.Tensor, gen_mask: torch.Tensor) -> torch.Tensor:
    """Attention mask for prompt + generation: prompt mask as is, generated tokens valid up to EOS."""
    return torch.cat([prompt_attention.to(gen_mask.dtype).long(), gen_mask.long()], dim=1)


def logits_positions(prompt_len: int, gen_len: int, device=None) -> torch.Tensor:
    """Indices whose logits predict the generated tokens: logits at t predict token t+1."""
    return torch.arange(prompt_len - 1, prompt_len + gen_len - 1, device=device)


def per_token_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    direction: str = "reverse",
    chunk: int = 64,
) -> torch.Tensor:
    """Per-position KL between full-vocabulary distributions, computed in fp32 chunks over positions.

    reverse: KL(student || teacher) = sum_v p_s (log p_s - log p_t)  (mode seeking; the OPD default)
    forward: KL(teacher || student) = sum_v p_t (log p_t - log p_s)
    Returns a (B, G) fp32 tensor with gradients flowing through the student logits only.
    """
    if direction not in KL_DIRECTIONS:
        raise ValueError(f"direction must be one of {KL_DIRECTIONS}, got {direction!r}")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(f"shape mismatch {tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}")
    outs = []
    for start in range(0, student_logits.shape[1], chunk):
        ls = F.log_softmax(student_logits[:, start : start + chunk].float(), dim=-1)
        lt = F.log_softmax(teacher_logits[:, start : start + chunk].float(), dim=-1).detach()
        if direction == "reverse":
            kl = (ls.exp() * (ls - lt)).sum(-1)
        else:
            kl = (lt.exp() * (lt - ls)).sum(-1)
        outs.append(kl)
    return torch.cat(outs, dim=1)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of `values` over positions where `mask` is True (safe when the mask is empty)."""
    m = mask.to(values.dtype)
    return (values * m).sum() / m.sum().clamp_min(1.0)


def patches_per_image(image_grid_thw: torch.Tensor) -> torch.Tensor:
    """Number of pixel_values rows each image occupies (t * h * w patches before spatial merging)."""
    return image_grid_thw.prod(dim=1)


def slice_vision(pixel_values: torch.Tensor, image_grid_thw: torch.Tensor, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the packed vision tensors for images [start, end) of a one-image-per-row batch."""
    counts = patches_per_image(image_grid_thw)
    offsets = torch.cat([torch.zeros(1, dtype=counts.dtype, device=counts.device), counts.cumsum(0)])
    return pixel_values[offsets[start] : offsets[end]], image_grid_thw[start:end]


def micro_batches(batch_size: int, micro: int) -> list[tuple[int, int]]:
    """Split [0, batch_size) into consecutive [start, end) ranges of at most `micro` rows."""
    if micro <= 0:
        raise ValueError("micro must be positive")
    return [(s, min(s + micro, batch_size)) for s in range(0, batch_size, micro)]
