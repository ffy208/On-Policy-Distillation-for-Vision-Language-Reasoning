"""Core package for the VLM On-Policy Distillation project."""

from .common import (
    ANSWER_PREFIX,
    DEFAULT_SEED,
    STUDENT_MODEL,
    TEACHER_MODEL,
    build_messages,
    build_prompt_text,
    load_hub_dataset,
    parse_answer,
    relaxed_accuracy,
    resize_image,
    score_predictions,
)

__all__ = [
    "ANSWER_PREFIX",
    "DEFAULT_SEED",
    "STUDENT_MODEL",
    "TEACHER_MODEL",
    "build_messages",
    "build_prompt_text",
    "load_hub_dataset",
    "parse_answer",
    "relaxed_accuracy",
    "resize_image",
    "score_predictions",
]
