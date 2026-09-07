"""Stage 1a: sample teacher solutions for the training questions with vLLM and keep the correct ones.

Runs in the evaluation environment (vLLM). Output is a Hub dataset with the columns
{id, image, question, answer, response, n_attempts} containing only rows whose kept response
is scored correct by relaxed accuracy.

Usage:
    python -m vlm_opd.generate_teacher --model Qwen/Qwen3-VL-8B-Instruct \
        --data-repo ffyang/vlm_opd_chartqa_human --out-repo ffyang/vlm_opd_sft_human \
        --stats outputs/teacher_generation.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from .common import (
    DEFAULT_SEED,
    get_hf_token,
    image_pixel_bounds,
    load_hub_dataset,
    parse_answer,
    relaxed_accuracy,
)
from .evaluate import build_inputs, build_vllm

logger = logging.getLogger(__name__)


def sample_solutions(
    llm,
    inputs: list[dict[str, Any]],
    num_samples: int,
    temperature: float,
    max_new_tokens: int,
    seed: int,
) -> list[list[str]]:
    """Return `num_samples` sampled completions per prompt."""
    from vllm import SamplingParams

    params = SamplingParams(n=num_samples, temperature=temperature, max_tokens=max_new_tokens, seed=seed)
    outputs = llm.generate(inputs, params)
    return [[c.text for c in o.outputs] for o in outputs]


def pick_correct(candidates: list[str], gold: str) -> tuple[str | None, int]:
    """Return the first correct candidate and how many were tried before finding it."""
    for i, text in enumerate(candidates, start=1):
        if relaxed_accuracy(parse_answer(text), gold):
            return text, i
    return None, len(candidates)


def generate_dataset(
    model_id: str,
    dataset,
    num_samples: int = 1,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    seed: int = DEFAULT_SEED,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.9,
):
    """Sample teacher solutions and return (filtered dataset, stats dict)."""
    from transformers import AutoProcessor

    token = get_hf_token()
    processor = AutoProcessor.from_pretrained(model_id, token=token, **image_pixel_bounds())
    llm = build_vllm(model_id, max_model_len, gpu_memory_utilization, seed)

    inputs = build_inputs(processor, dataset)
    t0 = time.time()
    samples = sample_solutions(llm, inputs, num_samples, temperature, max_new_tokens, seed)
    elapsed = time.time() - t0

    responses: list[str | None] = []
    attempts: list[int] = []
    for cands, ex in zip(samples, dataset):
        resp, n = pick_correct(cands, ex["answer"])
        responses.append(resp)
        attempts.append(n)

    ds = dataset.add_column("response", responses).add_column("n_attempts", attempts)
    kept = ds.filter(lambda r: r["response"] is not None, desc="Keeping correct solutions")
    lengths = [len(r) for r in kept["response"]]
    stats = {
        "teacher": model_id,
        "n_questions": len(dataset),
        "n_kept": len(kept),
        "acceptance_rate": len(kept) / max(1, len(dataset)),
        "num_samples": num_samples,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "mean_response_chars": sum(lengths) / max(1, len(lengths)),
        "elapsed_sec": round(elapsed, 1),
    }
    logger.info("Teacher generation: kept %d / %d (%.1f%%) in %.1fs", len(kept), len(dataset), 100 * stats["acceptance_rate"], elapsed)
    return kept, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample teacher solutions and keep the correct ones")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data-repo", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--out-repo", type=str, default=None, help="Hub dataset repo for the filtered SFT data")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--stats", type=str, default=None, help="Where to write the generation stats json")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-mem", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=int, default=None, help="Only the first N questions (smoke test)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ds = load_hub_dataset(args.data_repo, args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    kept, stats = generate_dataset(
        args.model, ds, args.num_samples, args.temperature, args.max_new_tokens, args.seed,
        args.max_model_len, args.gpu_mem,
    )
    if args.stats:
        Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats).write_text(json.dumps(stats, indent=2))
    if args.save_dir:
        kept.save_to_disk(args.save_dir)
    if args.out_repo:
        kept.push_to_hub(args.out_repo, split="train", private=True, token=get_hf_token())
        logger.info("SFT dataset pushed to %s", args.out_repo)


if __name__ == "__main__":
    main()
