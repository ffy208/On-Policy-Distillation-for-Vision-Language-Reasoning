"""Stage 4: measure where the teacher's token-level feedback lands on the student's own rollouts.

For each test question the student samples a solution (temperature 1.0, the same regime as OPD
training). Teacher and student then score the identical sequence and, for every generated token,
we record the full-vocabulary reverse KL (student || teacher) and the log-probability ratio of the
sampled token, log p_student(y) - log p_teacher(y). Tokens are classified into reasoning roles and
the feedback mass is aggregated per role.

Usage (evaluation/training environment, one GPU):
    python -m vlm_opd.analysis.token_feedback --student Qwen/Qwen3-VL-2B-Instruct \
        --teacher Qwen/Qwen3-VL-8B-Instruct --data-repo ffyang/vlm_opd_chartqa_human \
        --n 100 --out outputs/token_feedback_baseline.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ..collate import END_OF_TURN
from ..common import (
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
from ..opd_utils import (
    build_rollout_inputs,
    full_attention_mask,
    generation_mask,
    logits_positions,
    micro_batches,
    per_token_kl,
    slice_vision,
)
from .token_classes import aggregate, classify_tokens

logger = logging.getLogger(__name__)


def _load(model_id: str, token: str | None):
    from ..modeling import freeze_all, load_student

    model = load_student(model_id, dtype=torch.bfloat16, device_map="auto", token=token)
    freeze_all(model)
    return model.eval()


@torch.no_grad()
def score_batch(student, teacher, processor, rows: list[dict[str, Any]], max_new_tokens: int, temperature: float,
                seed: int, micro: int, prompt_style: str = DEFAULT_PROMPT_STYLE) -> list[dict[str, Any]]:
    """Sample one rollout per row with the student, then score every generated token with both models."""
    device = next(student.parameters()).device
    dtype = next(teacher.parameters()).dtype
    tok = processor.tokenizer
    pad_id, eos_id = tok.pad_token_id, tok.convert_tokens_to_ids(END_OF_TURN)

    inputs = {k: v.to(device) for k, v in build_rollout_inputs(processor, rows, prompt_style).items()}
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    batch_size, prompt_len = inputs["input_ids"].shape
    torch.manual_seed(seed)
    gen_kwargs = {"do_sample": temperature > 0, "max_new_tokens": max_new_tokens, "pad_token_id": pad_id, "eos_token_id": eos_id}
    if temperature > 0:
        gen_kwargs.update(temperature=temperature, top_p=1.0, top_k=0)
    sequences = student.generate(**inputs, **gen_kwargs)

    gen_len = sequences.shape[1] - prompt_len
    gen_mask = generation_mask(sequences, prompt_len, eos_id, pad_id)
    attn = full_attention_mask(inputs["attention_mask"], gen_mask)
    keep = logits_positions(prompt_len, gen_len, device=device)
    mm = inputs.get("mm_token_type_ids")
    if mm is not None:
        mm = torch.cat([mm, torch.zeros((batch_size, gen_len), dtype=mm.dtype, device=device)], dim=1)
    targets = sequences[:, prompt_len:]

    results: list[dict[str, Any]] = []
    for start, end in micro_batches(batch_size, micro):
        pv, grid = slice_vision(inputs["pixel_values"], inputs["image_grid_thw"], start, end)
        kwargs: dict[str, Any] = {"input_ids": sequences[start:end], "attention_mask": attn[start:end],
                                  "pixel_values": pv, "image_grid_thw": grid, "logits_to_keep": keep}
        if mm is not None:
            kwargs["mm_token_type_ids"] = mm[start:end]
        s_logits = student(**kwargs).logits
        t_logits = teacher(**kwargs).logits
        kl = per_token_kl(s_logits, t_logits, "reverse")
        tgt = targets[start:end].unsqueeze(-1)
        lp_s = F.log_softmax(s_logits.float(), dim=-1).gather(-1, tgt).squeeze(-1)
        lp_t = F.log_softmax(t_logits.float(), dim=-1).gather(-1, tgt).squeeze(-1)
        logratio = lp_s - lp_t
        for b in range(end - start):
            m = gen_mask[start + b]
            ids = targets[start + b][m].tolist()
            pieces = [tok.decode([i]) for i in ids]
            text = tok.decode(ids, skip_special_tokens=True)
            row = rows[start + b]
            pred = parse_answer(text)
            results.append({
                "id": row["id"], "question": row["question"], "gold": row["answer"],
                "pred": pred, "correct": relaxed_accuracy(pred, row["answer"]),
                "n_tokens": int(m.sum()), "pieces": pieces,
                "kl": [round(v, 5) for v in kl[b][m].tolist()],
                "logratio": [round(v, 5) for v in logratio[b][m].tolist()],
                "logp_student": [round(v, 5) for v in lp_s[b][m].tolist()],
                "logp_teacher": [round(v, 5) for v in lp_t[b][m].tolist()],
                "classes": classify_tokens(pieces),
            })
        del s_logits, t_logits
    return results


def run(student_id: str, teacher_id: str, data_repo: str, n: int, out_path: str | Path, split: str = "test",
        batch_size: int = 8, micro: int = 4, max_new_tokens: int = 512, temperature: float = 1.0,
        seed: int = DEFAULT_SEED, prompt_style: str = DEFAULT_PROMPT_STYLE) -> dict[str, Any]:
    """Score the first `n` questions of a split and write per-token records (jsonl) plus class statistics (json)."""
    from transformers import AutoProcessor

    token = get_hf_token()
    ds = load_hub_dataset(data_repo, split, token)
    ds = ds.select(range(min(n, len(ds))))
    processor = AutoProcessor.from_pretrained(student_id, token=token, **image_pixel_bounds())
    teacher = _load(teacher_id, token)
    student = _load(student_id, token)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with out_path.open("w") as f:
        for start in range(0, len(ds), batch_size):
            rows = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
            batch = score_batch(student, teacher, processor, rows, max_new_tokens, temperature, seed + start, micro,
                                prompt_style)
            for r in batch:
                f.write(json.dumps(r) + "\n")
            records.extend(batch)
            logger.info("scored %d / %d questions", len(records), len(ds))

    stats = aggregate(records)
    stats.update({"student": student_id, "teacher": teacher_id, "data_repo": data_repo, "split": split, "n_questions": len(records),
                  "temperature": temperature, "seed": seed, "prompt_style": prompt_style,
                  "rollout_accuracy": sum(r["correct"] for r in records) / max(1, len(records)),
                  "mean_kl_per_token": stats["kl_total"] / max(1, stats["n_tokens"])})
    stats_path = out_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2))
    logger.info("class statistics -> %s", stats_path)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Token-level teacher feedback on student rollouts")
    parser.add_argument("--student", default=STUDENT_MODEL)
    parser.add_argument("--teacher", default=TEACHER_MODEL)
    parser.add_argument("--data-repo", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--prompt-style", type=str, default=DEFAULT_PROMPT_STYLE, choices=PROMPT_STYLES)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.student, args.teacher, args.data_repo, args.n, args.out, args.split, args.batch_size, args.micro_batch,
        args.max_new_tokens, args.temperature, args.seed, args.prompt_style)


if __name__ == "__main__":
    main()
