"""TrainingArguments construction must work on the installed transformers (4.x or 5.x API)."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from vlm_opd.sft import build_training_args


def test_build_training_args_cpu(tmp_path):
    args = build_training_args(
        tmp_path, epochs=2, learning_rate=1e-4, per_device_batch_size=4, grad_accum=4,
        warmup_ratio=0.03, seed=42, max_steps=20, bf16=False, gradient_checkpointing=False,
        dataloader_num_workers=0,
    )
    assert args.max_steps == 20
    assert args.per_device_train_batch_size * args.gradient_accumulation_steps == 16
    assert args.remove_unused_columns is False
    assert "cosine" in str(args.lr_scheduler_type).lower()
    # warmup expressed as a ratio on either API generation
    ratio = getattr(args, "warmup_ratio", None)
    assert (ratio == 0.03) or (args.warmup_steps == 0.03)
