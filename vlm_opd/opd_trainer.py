"""Stage 2: on-policy distillation (OPD) for Qwen3-VL.

Each step:
1. The student samples one rollout per question (do_sample=True, temperature 1.0) so the
   training states are the ones the student actually reaches.
2. Prompt + rollout is scored by the frozen teacher (no grad) and by the student (with grad)
   on the same image tensors; logits are computed only for the generated span.
3. The loss is the per-token KL between the two distributions, averaged over generated
   tokens; prompt and image positions never enter the loss.
4. LoRA parameters are updated; every `ckpt_every` steps the adapter + optimizer state go to
   the Hub so a Colab disconnect can be resumed.

Usage:
    python -m vlm_opd.opd_trainer --data-repo ffyang/vlm_opd_chartqa_human \
        --ckpt-repo ffyang/vlm_opd_opd_human_ckpt --merged-repo ffyang/vlm_opd_opd_human_merged \
        --out-dir ckpt/opd_human --total-steps 500
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .collate import END_OF_TURN
from .common import (
    DEFAULT_PROMPT_STYLE,
    DEFAULT_SEED,
    PROMPT_STYLES,
    STUDENT_MODEL,
    TEACHER_MODEL,
    get_hf_token,
    image_pixel_bounds,
    load_hub_dataset,
    parse_answer,
    relaxed_accuracy,
)
from .opd_utils import (
    build_rollout_inputs,
    full_attention_mask,
    generation_mask,
    logits_positions,
    micro_batches,
    per_token_kl,
    slice_vision,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OPDConfig:
    """Immutable training configuration (mirrors configs/opd.yaml)."""

    data_repo: str
    out_dir: str
    student_model: str = STUDENT_MODEL
    teacher_model: str = TEACHER_MODEL
    ckpt_repo: str | None = None
    merged_repo: str | None = None
    merge: bool = True  # write out_dir/merged (base + LoRA) for local evaluation even without a merged_repo
    batch_size: int = 8
    micro_batch: int = 4
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    kl_direction: str = "reverse"
    learning_rate: float = 5e-5
    total_steps: int = 500
    warmup_steps: int = 20
    grad_clip: float = 1.0
    lora_r: int = 64
    ckpt_every: int = 50
    seed: int = DEFAULT_SEED
    limit: int | None = None
    resume: bool = True
    prompt_style: str = DEFAULT_PROMPT_STYLE


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def load_teacher(model_id: str, token: str | None = None, device_map: str | dict | None = "auto"):
    """Frozen bf16 teacher in eval mode."""
    from .modeling import freeze_all, load_student

    teacher = load_student(model_id, dtype=torch.bfloat16, device_map=device_map, token=token)
    freeze_all(teacher)
    teacher.eval()
    return teacher


def build_student(model_id: str, lora_r: int, resume_dir: str | Path | None = None, token: str | None = None,
                  device_map: str | dict | None = "auto"):
    """Student with LoRA: fresh adapters, or adapters loaded from a checkpoint directory when resuming."""
    from .modeling import freeze_all, load_student, load_student_with_lora

    if resume_dir is None:
        return load_student_with_lora(model_id, lora_r=lora_r, token=token, device_map=device_map)
    from peft import PeftModel

    base = load_student(model_id, dtype=torch.bfloat16, device_map=device_map, token=token)
    freeze_all(base)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    model = PeftModel.from_pretrained(base, str(resume_dir), is_trainable=True)
    logger.info("Resumed LoRA adapter from %s", resume_dir)
    return model


# ---------------------------------------------------------------------------
# One OPD step
# ---------------------------------------------------------------------------


@torch.no_grad()
def rollout(student, inputs: dict[str, torch.Tensor], cfg: OPDConfig, pad_id: int, eos_id: int, seed: int) -> torch.Tensor:
    """Sample one continuation per prompt. Returns the full (B, P+G) sequences."""
    student.eval()
    torch.manual_seed(seed)
    out = student.generate(
        **inputs,
        do_sample=True,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=0,
        max_new_tokens=cfg.max_new_tokens,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        use_cache=True,
    )
    student.train()
    return out


def opd_step(student, teacher, processor, rows: list[dict[str, Any]], cfg: OPDConfig, step_seed: int) -> dict[str, Any]:
    """Rollout, teacher/student scoring, KL loss and backward for one batch. Does not call optimizer.step()."""
    device = next(student.parameters()).device
    dtype = next(teacher.parameters()).dtype
    tok = processor.tokenizer
    pad_id, eos_id = tok.pad_token_id, tok.convert_tokens_to_ids(END_OF_TURN)

    inputs = {k: v.to(device) for k, v in build_rollout_inputs(processor, rows, cfg.prompt_style).items()}
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    batch_size, prompt_len = inputs["input_ids"].shape

    t0 = time.time()
    sequences = rollout(student, inputs, cfg, pad_id, eos_id, step_seed)
    t_rollout = time.time() - t0

    gen_len = sequences.shape[1] - prompt_len
    gen_mask = generation_mask(sequences, prompt_len, eos_id, pad_id)
    attn = full_attention_mask(inputs["attention_mask"], gen_mask)
    keep = logits_positions(prompt_len, gen_len, device=device)
    mm = inputs.get("mm_token_type_ids")
    if mm is not None:
        mm = torch.cat([mm, torch.zeros((batch_size, gen_len), dtype=mm.dtype, device=device)], dim=1)
    n_tokens = gen_mask.sum().clamp_min(1).float()

    kl_sum = torch.zeros((), device=device)
    kl_rows: list[torch.Tensor] = []  # detached per-token KL per row, for the role breakdown
    t_teacher = t_student = 0.0
    for start, end in micro_batches(batch_size, cfg.micro_batch):
        pv, grid = slice_vision(inputs["pixel_values"], inputs["image_grid_thw"], start, end)
        kwargs: dict[str, Any] = {
            "input_ids": sequences[start:end],
            "attention_mask": attn[start:end],
            "pixel_values": pv,
            "image_grid_thw": grid,
            "logits_to_keep": keep,
        }
        if mm is not None:
            kwargs["mm_token_type_ids"] = mm[start:end]
        t1 = time.time()
        with torch.no_grad():
            teacher_logits = teacher(**kwargs).logits
        t_teacher += time.time() - t1
        t2 = time.time()
        student_logits = student(**kwargs).logits
        kl = per_token_kl(student_logits, teacher_logits, cfg.kl_direction)
        mask = gen_mask[start:end]
        loss = (kl * mask).sum() / n_tokens  # token-mean over the whole batch, accumulated across micro-batches
        loss.backward()
        t_student += time.time() - t2
        kl_sum += (kl.detach() * mask).sum()
        kl_rows.extend((kl.detach() * mask).float().cpu())
        del teacher_logits, student_logits, kl, loss

    # Rollout quality signals (no gradient): length, stopping, answer format, correctness vs gold
    lengths = gen_mask.sum(dim=1)
    ended = (sequences[:, prompt_len:] == eos_id).any(dim=1)
    texts = [tok.decode(sequences[i, prompt_len:][gen_mask[i]], skip_special_tokens=True) for i in range(batch_size)]
    preds = [parse_answer(t) for t in texts]
    golds = [r.get("answer") for r in rows]
    correct = [relaxed_accuracy(p, g) for p, g in zip(preds, golds) if g is not None]
    role_stats = role_breakdown(tok, sequences[:, prompt_len:], gen_mask, kl_rows)
    return {
        "role_kl_mass": role_stats["kl_mass"],
        "role_token_share": role_stats["token_share"],
        "role_concentration": role_stats["concentration"],
        "kl": (kl_sum / n_tokens).item(),
        "n_gen_tokens": int(n_tokens.item()),
        "gen_len_mean": lengths.float().mean().item(),
        "gen_len_max": int(lengths.max().item()),
        "eos_rate": ended.float().mean().item(),
        "format_rate": sum(p is not None for p in preds) / batch_size,
        "rollout_acc": (sum(correct) / len(correct)) if correct else None,
        "prompt_len": prompt_len,
        "t_rollout": round(t_rollout, 2),
        "t_teacher": round(t_teacher, 2),
        "t_student": round(t_student, 2),
        "sample_output": texts[0][-400:],
    }


def role_breakdown(tok, gen_tokens: torch.Tensor, gen_mask: torch.Tensor, kl_rows: list[torch.Tensor]) -> dict[str, dict[str, float]]:
    """Share of teacher KL mass and of tokens per token role (answer / arithmetic / chart_value / text) for one batch.

    Uses the same classifier as the Stage 4 analysis so training logs and the offline analysis agree.
    """
    from .analysis.token_classes import CLASSES, aggregate

    examples = []
    for i in range(gen_tokens.shape[0]):
        ids = gen_tokens[i][gen_mask[i]].tolist()
        if not ids:
            continue
        examples.append({"pieces": [tok.decode([t]) for t in ids], "kl": kl_rows[i][gen_mask[i].cpu()].tolist()})
    if not examples:
        return {"kl_mass": {}, "token_share": {}, "concentration": {}}
    stats = aggregate(examples)["classes"]
    return {
        "kl_mass": {c: round(stats[c]["kl_mass_share"], 4) for c in CLASSES},
        "token_share": {c: round(stats[c]["token_share"], 4) for c in CLASSES},
        "concentration": {c: round(stats[c]["concentration"], 3) for c in CLASSES},
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _next_batch(dataset, cursor: int, epoch: int, batch_size: int, seed: int) -> tuple[list[dict[str, Any]], int, int]:
    """Deterministic shuffled sampling without replacement; reshuffles with a new seed at each epoch boundary."""
    n = len(dataset)
    if cursor + batch_size > n:
        cursor, epoch = 0, epoch + 1
    perm = np.random.default_rng(seed + epoch).permutation(n)
    idx = perm[cursor : cursor + batch_size].tolist()
    return [dataset[i] for i in idx], cursor + batch_size, epoch


def train(cfg: OPDConfig) -> dict[str, Any]:
    """Run OPD end to end: resume if possible, train, checkpoint, save the final adapter and merged weights."""
    from transformers import AutoProcessor, get_cosine_schedule_with_warmup

    from .hub_utils import load_latest, restore_train_state, save_ckpt, upload_small_file
    from .modeling import trainable_summary

    token = get_hf_token()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "opd_log.jsonl"
    (out_dir / "opd_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    dataset = load_hub_dataset(cfg.data_repo, "train", token)
    if cfg.limit:
        dataset = dataset.select(range(min(cfg.limit, len(dataset))))
    processor = AutoProcessor.from_pretrained(cfg.student_model, token=token, **image_pixel_bounds())

    resume_dir, start_step, cursor, epoch = None, 0, 0, 0
    if cfg.resume and cfg.ckpt_repo:
        found = load_latest(cfg.ckpt_repo, out_dir / "ckpt", token)
        if found:
            start_step, resume_dir = found

    teacher = load_teacher(cfg.teacher_model, token)
    student = build_student(cfg.student_model, cfg.lora_r, resume_dir, token)
    summary = trainable_summary(student)
    if summary["vision_trainable"]:
        raise RuntimeError("vision parameters must stay frozen")
    params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_steps, cfg.total_steps)
    if resume_dir is not None:
        meta = restore_train_state(resume_dir, optimizer, scheduler)
        cursor, epoch = int(meta.get("cursor", 0)), int(meta.get("epoch", 0))
        logger.info("Resuming at step %d (cursor=%d, epoch=%d)", start_step, cursor, epoch)

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    for step in range(start_step + 1, cfg.total_steps + 1):
        t_step = time.time()
        rows, cursor, epoch = _next_batch(dataset, cursor, epoch, cfg.batch_size, cfg.seed)
        metrics = opd_step(student, teacher, processor, rows, cfg, step_seed=cfg.seed * 1000 + step)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip).item()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        record = {
            "step": step,
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "grad_norm": grad_norm,
            "t_step": round(time.time() - t_step, 2),
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if torch.cuda.is_available() else None,
            **metrics,
        }
        with log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        logger.info(
            "step %d | kl %.4f | len %.0f | eos %.2f | fmt %.2f | acc %s | answer-KL %s | %.1fs (roll %.1f, teach %.1f, stud %.1f) | mem %s GB",
            step, record["kl"], record["gen_len_mean"], record["eos_rate"], record["format_rate"],
            f"{record['rollout_acc']:.2f}" if record["rollout_acc"] is not None else "n/a",
            record["role_kl_mass"].get("answer", "n/a"),
            record["t_step"], record["t_rollout"], record["t_teacher"], record["t_student"], record["peak_mem_gb"],
        )

        if cfg.ckpt_repo and (step % cfg.ckpt_every == 0 or step == cfg.total_steps):
            save_ckpt(student, optimizer, step, cfg.ckpt_repo, out_dir / "ckpt", scheduler=scheduler,
                      extra_state={"cursor": cursor, "epoch": epoch, "config": asdict(cfg)}, token=token)
            upload_small_file(log_path, cfg.ckpt_repo, path_in_repo="opd_log.jsonl", token=token)

    adapter_dir = out_dir / "adapter"
    student.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    merged_dir: Path | None = None
    if cfg.merge or cfg.merged_repo:
        del teacher
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        from .sft import merge_and_save, push_dir

        merged_dir = merge_and_save(adapter_dir, out_dir / "merged", cfg.student_model)
        if cfg.merged_repo:
            push_dir(merged_dir, cfg.merged_repo, f"OPD merged weights, step {cfg.total_steps}")
    return {"final_step": cfg.total_steps, "log_path": str(log_path), "adapter_dir": str(adapter_dir),
            "merged_dir": str(merged_dir) if merged_dir else None}


def main() -> None:
    parser = argparse.ArgumentParser(description="On-policy distillation for Qwen3-VL")
    parser.add_argument("--data-repo", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--student", default=STUDENT_MODEL)
    parser.add_argument("--teacher", default=TEACHER_MODEL)
    parser.add_argument("--ckpt-repo", default=None)
    parser.add_argument("--merged-repo", default=None, help="Push the merged model here (large: ~4 GB per run)")
    parser.add_argument("--no-merge", action="store_true", help="Skip writing out_dir/merged")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--kl-direction", choices=["reverse", "forward"], default="reverse")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--total-steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--ckpt-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N training questions (smoke)")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--prompt-style", type=str, default=DEFAULT_PROMPT_STYLE, choices=PROMPT_STYLES)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = OPDConfig(
        data_repo=args.data_repo, out_dir=args.out_dir, student_model=args.student, teacher_model=args.teacher,
        ckpt_repo=args.ckpt_repo or None, merged_repo=args.merged_repo or None, merge=not args.no_merge,
        batch_size=args.batch_size,
        micro_batch=args.micro_batch, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_p=args.top_p, kl_direction=args.kl_direction, learning_rate=args.lr, total_steps=args.total_steps,
        warmup_steps=args.warmup_steps, grad_clip=args.grad_clip, lora_r=args.lora_r, ckpt_every=args.ckpt_every,
        seed=args.seed, limit=args.limit, resume=not args.no_resume, prompt_style=args.prompt_style,
    )
    train(cfg)


if __name__ == "__main__":
    main()
