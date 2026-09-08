"""Stage 1b: supervised distillation. Fine-tune the student with LoRA on filtered teacher solutions.

Uses the plain transformers Trainer with `SFTCollator` (explicit prompt/image masking) instead of
TRL's SFTTrainer: TRL's VLM path changes between versions and its dependencies conflict with
vLLM, whereas peft + accelerate install cleanly next to vLLM, so one environment serves
generation, training, and evaluation.

Usage:
    python -m vlm_opd.sft --data-repo ffyang/vlm_opd_sft_human --out-dir ckpt/sft_human \
        --push-adapter-repo ffyang/vlm_opd_sft_human_lora --push-merged-repo ffyang/vlm_opd_sft_human_merged
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from .common import DEFAULT_SEED, STUDENT_MODEL, get_hf_token, image_pixel_bounds, load_hub_dataset

logger = logging.getLogger(__name__)


def question_index(row_id: str) -> int:
    """Position of a question in the sampled train set, parsed from ids like `train_00042`."""
    return int(str(row_id).rsplit("_", 1)[-1])


def limit_questions(dataset, max_question_index: int | None):
    """Keep only rows whose question index is below `max_question_index` (a data budget in questions).

    The SFT dataset holds one kept teacher solution per question, in train-set order, so a budget of N
    questions is the subset with index < N, not the first N rows.
    """
    if max_question_index is None:
        return dataset
    return dataset.filter(lambda r: question_index(r["id"]) < max_question_index, desc=f"Questions < {max_question_index}")


def build_training_args(
    out_dir: str | Path,
    epochs: float = 2.0,
    learning_rate: float = 1e-4,
    per_device_batch_size: int = 4,
    grad_accum: int = 4,
    warmup_ratio: float = 0.03,
    seed: int = DEFAULT_SEED,
    max_steps: int = -1,
    logging_steps: int = 10,
    bf16: bool = True,
    gradient_checkpointing: bool = True,
    dataloader_num_workers: int = 2,
):
    """TrainingArguments for LoRA SFT, compatible with transformers 4.x and 5.x.

    transformers 5 removed `warmup_ratio`; `warmup_steps` now takes a float in [0, 1) as a ratio.
    """
    import inspect

    from transformers import TrainingArguments

    kwargs: dict[str, Any] = {
        "output_dir": str(Path(out_dir) / "trainer"),
        "num_train_epochs": epochs,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "lr_scheduler_type": "cosine",
        "bf16": bf16,
        "gradient_checkpointing": gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "logging_steps": logging_steps,
        "save_strategy": "no",
        "report_to": [],
        "remove_unused_columns": False,  # rows carry PIL images; the collator does all the work
        "dataloader_num_workers": dataloader_num_workers,
        "seed": seed,
        "optim": "adamw_torch",
    }
    if "warmup_ratio" in inspect.signature(TrainingArguments.__init__).parameters:
        kwargs["warmup_ratio"] = warmup_ratio
    else:
        kwargs["warmup_steps"] = warmup_ratio
    return TrainingArguments(**kwargs)


def train_sft(
    dataset,
    out_dir: str | Path,
    model_id: str = STUDENT_MODEL,
    epochs: float = 2.0,
    learning_rate: float = 1e-4,
    per_device_batch_size: int = 4,
    grad_accum: int = 4,
    lora_r: int = 64,
    max_length: int = 2048,
    warmup_ratio: float = 0.03,
    seed: int = DEFAULT_SEED,
    max_steps: int = -1,
    logging_steps: int = 10,
) -> dict[str, Any]:
    """Run LoRA SFT and save the adapter to `out_dir/adapter`. Returns training metrics."""
    from transformers import AutoProcessor, Trainer, set_seed

    from .collate import SFTCollator
    from .modeling import load_student_with_lora, trainable_summary

    set_seed(seed)
    token = get_hf_token()
    processor = AutoProcessor.from_pretrained(model_id, token=token, **image_pixel_bounds())
    model = load_student_with_lora(model_id, lora_r=lora_r, token=token)
    summary = trainable_summary(model)
    if summary["vision_trainable"]:
        raise RuntimeError(f"vision parameters must stay frozen, found trainable: {summary['vision_trainable'][:3]}")

    out_dir = Path(out_dir)
    args = build_training_args(
        out_dir, epochs, learning_rate, per_device_batch_size, grad_accum, warmup_ratio, seed,
        max_steps, logging_steps,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=SFTCollator(processor, max_length=max_length),
    )
    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    metrics = {
        "model": model_id,
        "n_train": len(dataset),
        "epochs": epochs,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "effective_batch_size": per_device_batch_size * grad_accum,
        "lora_r": lora_r,
        "trainable_params": summary["trainable"],
        "train_loss": result.training_loss,
        "global_steps": result.global_step,
        "elapsed_sec": round(elapsed, 1),
        "log_history": trainer.state.log_history,
    }
    (out_dir / "sft_metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("SFT done: loss=%.4f steps=%d (%.0fs) -> %s", result.training_loss, result.global_step, elapsed, adapter_dir)
    return metrics


def merge_and_save(adapter_dir: str | Path, merged_dir: str | Path, model_id: str = STUDENT_MODEL) -> Path:
    """Merge the LoRA adapter into the base weights and save a standalone model for vLLM evaluation."""
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor

    from .modeling import load_student

    token = get_hf_token()
    base = load_student(model_id, dtype=torch.bfloat16, device_map="cpu", token=token)
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    merged_dir = Path(merged_dir)
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    AutoProcessor.from_pretrained(model_id, token=token).save_pretrained(str(merged_dir))
    logger.info("Merged model saved to %s", merged_dir)
    return merged_dir


def push_dir(local_dir: str | Path, repo_id: str, message: str) -> None:
    """Upload a directory to a private Hub model repo."""
    from huggingface_hub import HfApi

    api = HfApi(token=get_hf_token())
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=str(local_dir), commit_message=message)
    logger.info("Uploaded %s -> %s", local_dir, repo_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA supervised distillation for Qwen3-VL")
    parser.add_argument("--data-repo", type=str, required=True, help="Hub dataset with a `response` column")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--model", type=str, default=STUDENT_MODEL)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-steps", type=int, default=-1, help="Stop after N optimizer steps (smoke test)")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N training rows")
    parser.add_argument("--max-question-index", type=int, default=None,
                        help="Data budget in questions: keep rows whose question index is below N")
    parser.add_argument("--push-adapter-repo", type=str, default=None)
    parser.add_argument("--push-merged-repo", type=str, default=None, help="Also merge LoRA into the base and push")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ds = load_hub_dataset(args.data_repo, args.split)
    ds = limit_questions(ds, args.max_question_index)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    logger.info("Training rows: %d", len(ds))
    train_sft(
        ds, args.out_dir, args.model, args.epochs, args.lr, args.batch, args.grad_accum,
        args.lora_r, args.max_length, seed=args.seed, max_steps=args.max_steps,
    )
    adapter_dir = Path(args.out_dir) / "adapter"
    if args.push_adapter_repo:
        push_dir(adapter_dir, args.push_adapter_repo, "SFT LoRA adapter")
    if args.push_merged_repo:
        merged_dir = merge_and_save(adapter_dir, Path(args.out_dir) / "merged", args.model)
        push_dir(merged_dir, args.push_merged_repo, "SFT merged weights")


if __name__ == "__main__":
    main()
