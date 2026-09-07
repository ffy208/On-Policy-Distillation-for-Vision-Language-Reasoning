"""Load the Qwen3-VL student for training: frozen vision tower, LoRA on the language model only."""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

# LoRA goes on every linear projection of the language model decoder layers and nowhere else.
# Vision blocks live under `visual` (attention `qkv`/`proj`, MLP `linear_fc1`/`linear_fc2`) and
# the projector under `visual.merger`; none of them match this pattern.
LORA_TARGET_REGEX = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def load_student(
    model_id: str,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict | None = "auto",
    attn_implementation: str = "sdpa",
    token: str | None = None,
):
    """Load a Qwen3-VL model for conditional generation."""
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        token=token,
    )
    return model


def freeze_all(model) -> None:
    """Freeze every parameter (LoRA adapters added afterwards are the only trainable weights)."""
    for p in model.parameters():
        p.requires_grad_(False)


def apply_lora(model, r: int = 64, alpha: int | None = None, dropout: float = 0.05):
    """Wrap the model with LoRA adapters on the language-model projections."""
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha if alpha is not None else 2 * r,
        lora_dropout=dropout,
        bias="none",
        target_modules=LORA_TARGET_REGEX,
    )
    return get_peft_model(model, cfg)


def load_student_with_lora(
    model_id: str,
    lora_r: int = 64,
    lora_alpha: int | None = None,
    lora_dropout: float = 0.05,
    gradient_checkpointing: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict | None = "auto",
    token: str | None = None,
):
    """Student ready for training: base frozen (vision tower included), LoRA on the LM, grad checkpointing on."""
    model = load_student(model_id, dtype=dtype, device_map=device_map, token=token)
    freeze_all(model)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = apply_lora(model, lora_r, lora_alpha, lora_dropout)
    summary = trainable_summary(model)
    logger.info(
        "Trainable params: %s / %s (%.2f%%)",
        f"{summary['trainable']:,}", f"{summary['total']:,}", summary["trainable_pct"],
    )
    return model


def trainable_summary(model) -> dict[str, Any]:
    """Count trainable vs total parameters and check that no vision parameter is trainable."""
    trainable = total = 0
    vision_trainable: list[str] = []
    for name, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            if "visual" in name:
                vision_trainable.append(name)
    return {
        "trainable": trainable,
        "total": total,
        "trainable_pct": 100.0 * trainable / max(1, total),
        "vision_trainable": vision_trainable,
    }
