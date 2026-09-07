"""Stage 0: download ChartQA, sample with a fixed seed, resize images, unify fields, and push
to a private Hub dataset repo.

Usage (CLI):
    python -m vlm_opd.prepare --repo-id username/vlm_opd_chartqa --n-train 3000 --n-test 500
    python -m vlm_opd.prepare --n-train 100 --n-test 50 --save-dir data/smoke   # local smoke test, no Hub push
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

from datasets import Dataset, DatasetDict, Features, Value, load_dataset
from datasets import Image as HFImage

from .common import DEFAULT_SEED, IMAGE_MAX_SIDE, get_hf_token, resize_image

logger = logging.getLogger(__name__)

SOURCE_DATASET = "HuggingFaceM4/ChartQA"
# ChartQA labels each question as human-written or machine-generated. Machine questions are
# templated and easy (the 2B student already scores ~0.84 zero-shot on a random mix), so the
# main experiments use human questions only to leave headroom between student and teacher.
QUESTION_SOURCES = ("all", "human", "machine")

# Unified schema
UNIFIED_FEATURES = Features(
    {
        "id": Value("string"),
        "image": HFImage(),
        "question": Value("string"),
        "answer": Value("string"),
    }
)


def _to_unified(example: dict[str, Any], idx: int, prefix: str, max_side: int) -> dict[str, Any]:
    """Convert one HuggingFaceM4/ChartQA row into {id, image, question, answer}.

    Source fields: image, query, label (list[str]), human_or_machine.
    """
    label = example["label"]
    answer = label[0] if isinstance(label, (list, tuple)) else label
    return {
        "id": f"{prefix}_{idx:05d}",
        "image": resize_image(example["image"], max_side),
        "question": str(example["query"]).strip(),
        "answer": str(answer).strip(),
    }


def filter_question_source(source: Dataset, question_source: str) -> Dataset:
    """Keep only human-written or machine-generated questions (or everything for "all")."""
    if question_source not in QUESTION_SOURCES:
        raise ValueError(f"question_source must be one of {QUESTION_SOURCES}, got {question_source!r}")
    if question_source == "all":
        return source
    label_names = source.features["human_or_machine"].names
    target = label_names.index(question_source)
    return source.filter(lambda ex: ex["human_or_machine"] == target, desc=f"Keeping {question_source} questions")


def sample_split(
    source: Dataset,
    n: int,
    seed: int,
    prefix: str,
    max_side: int = IMAGE_MAX_SIDE,
    question_source: str = "all",
) -> Dataset:
    """Filter by question source, shuffle with a fixed seed, take the first n rows, resize, unify fields."""
    source = filter_question_source(source, question_source)
    n = min(n, len(source))
    subset = source.shuffle(seed=seed).select(range(n))
    unified = subset.map(
        _to_unified,
        with_indices=True,
        fn_kwargs={"prefix": prefix, "max_side": max_side},
        remove_columns=subset.column_names,
        features=UNIFIED_FEATURES,
        desc=f"Processing {prefix}",
    )
    return unified


def build_dataset(
    n_train: int,
    n_test: int,
    seed: int = DEFAULT_SEED,
    max_side: int = IMAGE_MAX_SIDE,
    source_dataset: str = SOURCE_DATASET,
    token: str | None = None,
    question_source: str = "all",
) -> DatasetDict:
    """Build the sampled train / test splits from the original ChartQA."""
    token = token or get_hf_token()
    logger.info("Downloading source dataset %s", source_dataset)
    raw = load_dataset(source_dataset, token=token)
    train = sample_split(raw["train"], n_train, seed, "train", max_side, question_source)
    test = sample_split(raw["test"], n_test, seed, "test", max_side, question_source)
    logger.info("Sampling done (%s questions): train=%d, test=%d", question_source, len(train), len(test))
    return DatasetDict({"train": train, "test": test})


def push(ds: DatasetDict, repo_id: str, token: str | None = None) -> None:
    """Push to a private Hub dataset repo."""
    token = token or get_hf_token()
    ds.push_to_hub(repo_id, private=True, token=token)
    logger.info("Dataset pushed to %s", repo_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="ChartQA sampling and preprocessing")
    parser.add_argument("--repo-id", type=str, default=None, help="Private Hub dataset repo")
    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-side", type=int, default=IMAGE_MAX_SIDE)
    parser.add_argument("--source", type=str, default=SOURCE_DATASET)
    parser.add_argument("--save-dir", type=str, default=None, help="Also save to a local directory")
    parser.add_argument(
        "--question-source", type=str, default="all", choices=QUESTION_SOURCES,
        help="Keep only human-written or machine-generated questions",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ds = build_dataset(
        args.n_train, args.n_test, args.seed, args.max_side, args.source,
        question_source=args.question_source,
    )
    if args.save_dir:
        ds.save_to_disk(args.save_dir)
        logger.info("Dataset saved locally to %s", args.save_dir)
    if args.repo_id:
        push(ds, args.repo_id)
    if not args.save_dir and not args.repo_id:
        logger.warning("Neither --repo-id nor --save-dir given; nothing was written")


if __name__ == "__main__":
    main()
