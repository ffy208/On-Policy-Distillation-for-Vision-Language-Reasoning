"""Stage 0: batch inference for Qwen3-VL with vLLM, scoring, and json output.

Note: import this module only in the evaluation environment (with vLLM installed);
do not import it in the training environment.
Usage:
    python -m vlm_opd.evaluate --model Qwen/Qwen3-VL-2B-Instruct --data-repo username/vlm_opd_chartqa \
        --out outputs/eval_student_zeroshot.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from .common import (
    DEFAULT_PROMPT_STYLE,
    DEFAULT_SEED,
    PROMPT_STYLES,
    build_messages,
    get_hf_token,
    image_pixel_bounds,
    load_hub_dataset,
    score_predictions,
)

logger = logging.getLogger(__name__)


def build_vllm(
    model_id: str,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.85,
    seed: int = DEFAULT_SEED,
    enable_lora: bool = False,
    max_lora_rank: int = 64,
):
    """Construct the vLLM engine. Image pixel bounds match the training side."""
    import os

    # vLLM forks its engine process by default and the fork fails when the parent has already touched CUDA
    # ("Cannot re-initialize CUDA in forked subprocess"); this happened on every Blackwell node on PACE while
    # the L40S nodes were fine. Spawning is always safe, so make it the default unless the caller chose.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM

    return LLM(
        model=model_id,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs=image_pixel_bounds(),
        seed=seed,
        enable_lora=enable_lora,
        max_lora_rank=max_lora_rank if enable_lora else 16,
        trust_remote_code=True,
    )


def build_inputs(processor, dataset, style: str = DEFAULT_PROMPT_STYLE) -> list[dict[str, Any]]:
    """Convert the dataset into vLLM multimodal inputs."""
    inputs: list[dict[str, Any]] = []
    for ex in dataset:
        prompt = processor.apply_chat_template(
            build_messages(ex["question"], style=style), tokenize=False, add_generation_prompt=True
        )
        inputs.append({"prompt": prompt, "multi_modal_data": {"image": ex["image"]}})
    return inputs


def generate(
    llm,
    inputs: list[dict[str, Any]],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    seed: int = DEFAULT_SEED,
    lora_path: str | None = None,
) -> list[str]:
    """Batch generation; returns one text output per sample."""
    from vllm import SamplingParams

    params = SamplingParams(temperature=temperature, max_tokens=max_new_tokens, seed=seed)
    lora_request = None
    if lora_path is not None:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest("adapter", 1, lora_path)
    outputs = llm.generate(inputs, params, lora_request=lora_request)
    return [o.outputs[0].text for o in outputs]


def run_eval(
    model_id: str,
    dataset,
    out_path: str | Path,
    lora_path: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.85,
    seed: int = DEFAULT_SEED,
    tag: str | None = None,
    llm=None,
    prompt_style: str = DEFAULT_PROMPT_STYLE,
) -> dict[str, Any]:
    """Full evaluation pipeline: build engine -> build inputs -> generate -> score -> write json.

    Args:
        model_id: Base model id.
        dataset: Dataset with image / question / answer fields.
        out_path: Path of the result json.
        lora_path: Optional LoRA adapter directory (for evaluating SFT / OPD students).
        tag: Label stored in the result, e.g. "student_zeroshot".
        llm: A pre-built vLLM engine (reuse to avoid reloading).

    Returns:
        Dict with accuracy, format_rate, and per-sample records.
    """
    from transformers import AutoProcessor

    token = get_hf_token()
    processor = AutoProcessor.from_pretrained(model_id, token=token, **image_pixel_bounds())
    if llm is None:
        llm = build_vllm(
            model_id, max_model_len, gpu_memory_utilization, seed, enable_lora=lora_path is not None
        )

    inputs = build_inputs(processor, dataset, prompt_style)
    t0 = time.time()
    texts = generate(llm, inputs, max_new_tokens, temperature, seed, lora_path)
    elapsed = time.time() - t0

    golds = [ex["answer"] for ex in dataset]
    scored = score_predictions(texts, golds)
    for rec, ex, text in zip(scored["records"], dataset, texts):
        rec["id"] = ex["id"]
        rec["question"] = ex["question"]
        rec["output"] = text

    result: dict[str, Any] = {
        "tag": tag or Path(out_path).stem,
        "model": model_id,
        "lora_path": lora_path,
        "n": scored["n"],
        "accuracy": scored["accuracy"],
        "format_rate": scored["format_rate"],
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "seed": seed,
        "prompt_style": prompt_style,
        "elapsed_sec": round(elapsed, 1),
        "mean_output_chars": sum(len(t) for t in texts) / max(1, len(texts)),
        "records": scored["records"],
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    logger.info(
        "[%s] acc=%.4f format=%.4f n=%d (%.1fs) -> %s",
        result["tag"], result["accuracy"], result["format_rate"], result["n"], elapsed, out_path,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM batch evaluation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data-repo", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    # vLLM needs room for the image tokens, the prompt, and the whole generation; grow the context with the budget
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=None, help="Default: 3584 + max_new_tokens")
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N rows (smoke test)")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--prompt-style", type=str, default=DEFAULT_PROMPT_STYLE, choices=PROMPT_STYLES)
    args = parser.parse_args()
    if args.max_model_len is None:
        args.max_model_len = 3584 + args.max_new_tokens

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ds = load_hub_dataset(args.data_repo, args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    run_eval(
        args.model, ds, args.out, args.lora_path, args.max_new_tokens, args.temperature,
        args.max_model_len, args.gpu_mem, args.seed, args.tag, prompt_style=args.prompt_style,
    )


if __name__ == "__main__":
    main()
