"""Sample a source dataset with a fixed seed, resize images, unify fields, and push to a private Hub
dataset repo. Every task and OOD test set in the study goes through this module so downstream code
only ever sees the schema {id, image, question, answer}.

Sources (see `SOURCES`): chartqa (train/test), geometry3k (train/test), charxiv and chartqapro
(test only, used as out-of-distribution chart evaluations).

Usage (CLI):
    python -m vlm_opd.prepare --source chartqa --question-source human --repo-id ffyang/vlm_opd_chartqa_human
    python -m vlm_opd.prepare --source geometry3k --n-train 2101 --n-test 500 --repo-id ffyang/vlm_opd_geometry3k
    python -m vlm_opd.prepare --source charxiv --n-test 500 --repo-id ffyang/vlm_opd_ood_charxiv
    python -m vlm_opd.prepare --source chartqa --n-train 100 --n-test 50 --save-dir data/smoke   # local, no push
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

# Source registry: Hub id, which raw splits map to our train/test, and the per-row converter.
# Converters return None for rows to drop (e.g. answer types the scorer cannot judge).
SOURCES: dict[str, dict[str, Any]] = {
    "chartqa": {"hub": "HuggingFaceM4/ChartQA", "train": "train", "test": "test", "convert": "_convert_chartqa"},
    "geometry3k": {"hub": "hiyouga/geometry3k", "train": "train", "test": "test", "convert": "_convert_geometry3k"},
    # CharXiv: only the validation split carries answers; reasoning questions with numeric answers
    # (answer type 3 = number in chart, 4 = number in general) are scorable with relaxed accuracy.
    "charxiv": {"hub": "princeton-nlp/CharXiv", "train": None, "test": "validation", "convert": "_convert_charxiv"},
    # ChartQAPro: keep single-turn factoid questions; Question/Answer are lists (multi-turn for other types).
    "chartqapro": {"hub": "ahmed-masry/ChartQAPro", "train": None, "test": "test", "convert": "_convert_chartqapro"},
}

# Unified schema
UNIFIED_FEATURES = Features(
    {
        "id": Value("string"),
        "image": HFImage(),
        "question": Value("string"),
        "answer": Value("string"),
    }
)


def _convert_chartqa(example: dict[str, Any]) -> dict[str, Any] | None:
    """HuggingFaceM4/ChartQA: image, query, label (list[str]), human_or_machine."""
    label = example["label"]
    answer = label[0] if isinstance(label, (list, tuple)) else label
    return {"image": example["image"], "question": str(example["query"]).strip(), "answer": str(answer).strip()}


def _convert_geometry3k(example: dict[str, Any]) -> dict[str, Any] | None:
    """hiyouga/geometry3k: images (list with one image), problem (starts with an `<image>` tag), answer."""
    images = example["images"]
    if not images:
        return None
    question = str(example["problem"]).replace("<image>", " ").strip()
    return {"image": images[0], "question": question, "answer": str(example["answer"]).strip()}


def _convert_charxiv(example: dict[str, Any]) -> dict[str, Any] | None:
    """princeton-nlp/CharXiv validation: reasoning_q / reasoning_a with numeric answer types only."""
    if example.get("reasoning_a") is None or int(example.get("reasoning_a_type") or 0) not in (3, 4):
        return None
    return {"image": example["image"], "question": str(example["reasoning_q"]).strip(), "answer": str(example["reasoning_a"]).strip()}


def _convert_chartqapro(example: dict[str, Any]) -> dict[str, Any] | None:
    """ahmed-masry/ChartQAPro test: Question / Answer are lists; keep single-turn Factoid rows."""
    if example.get("Question Type") != "Factoid":
        return None
    qs, ans = example["Question"], example["Answer"]
    if not qs or not ans or len(qs) != 1:
        return None
    image = example["image"]
    if isinstance(image, (bytes, bytearray)):
        import io

        from PIL import Image as PILImage

        image = PILImage.open(io.BytesIO(image))
    elif isinstance(image, dict) and "bytes" in image:
        import io

        from PIL import Image as PILImage

        image = PILImage.open(io.BytesIO(image["bytes"]))
    return {"image": image, "question": str(qs[0]).strip(), "answer": str(ans[0]).strip()}


CONVERTERS = {name: globals()[cfg["convert"]] for name, cfg in SOURCES.items()}


def _to_unified(example: dict[str, Any], idx: int, prefix: str, max_side: int, source: str = "chartqa") -> dict[str, Any]:
    """Convert one raw row into {id, image, question, answer} with the image resized."""
    row = CONVERTERS[source](example)
    if row is None:
        raise ValueError("row was filtered; call `convert_rows` instead of `_to_unified` on unfiltered data")
    return {"id": f"{prefix}_{idx:05d}", "image": resize_image(row["image"], max_side), "question": row["question"], "answer": row["answer"]}


def convert_rows(raw: Dataset, source: str, prefix: str, max_side: int) -> Dataset:
    """Apply the source converter row by row (dropping filtered rows) and assign sequential ids."""
    rows: list[dict[str, Any]] = []
    for ex in raw:
        row = CONVERTERS[source](ex)
        if row is None:
            continue
        rows.append({"id": f"{prefix}_{len(rows):05d}", "image": resize_image(row["image"], max_side),
                     "question": row["question"], "answer": row["answer"]})
    return Dataset.from_list(rows, features=UNIFIED_FEATURES)


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
    source_name: str = "chartqa",
) -> Dataset:
    """Filter (ChartQA question source), shuffle with a fixed seed, take the first n rows, convert and resize.

    For sources whose converter drops rows, more than n rows are drawn so that n survive when possible.
    """
    if source_name == "chartqa":
        source = filter_question_source(source, question_source)
    shuffled = source.shuffle(seed=seed)
    take = min(len(shuffled), n if source_name in ("chartqa", "geometry3k") else max(n * 4, n + 200))
    unified = convert_rows(shuffled.select(range(take)), source_name, prefix, max_side)
    return unified.select(range(min(n, len(unified))))


def build_dataset(
    n_train: int,
    n_test: int,
    seed: int = DEFAULT_SEED,
    max_side: int = IMAGE_MAX_SIDE,
    source_dataset: str | None = None,
    token: str | None = None,
    question_source: str = "all",
    source: str = "chartqa",
) -> DatasetDict:
    """Build the sampled train / test splits for one source (test-only sources yield just `test`)."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {tuple(SOURCES)}, got {source!r}")
    cfg = SOURCES[source]
    token = token or get_hf_token()
    hub = source_dataset or cfg["hub"]
    logger.info("Downloading source dataset %s", hub)
    raw = load_dataset(hub, token=token)
    out: dict[str, Dataset] = {}
    if cfg["train"] is not None and n_train > 0:
        out["train"] = sample_split(raw[cfg["train"]], n_train, seed, "train", max_side, question_source, source)
    out["test"] = sample_split(raw[cfg["test"]], n_test, seed, "test", max_side, question_source, source)
    logger.info("Sampling done for %s: %s", source, {k: len(v) for k, v in out.items()})
    return DatasetDict(out)


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
    parser.add_argument("--source", type=str, default="chartqa", choices=tuple(SOURCES), help="Source dataset adapter")
    parser.add_argument("--source-dataset", type=str, default=None, help="Override the Hub id of the source")
    parser.add_argument("--save-dir", type=str, default=None, help="Also save to a local directory")
    parser.add_argument(
        "--question-source", type=str, default="all", choices=QUESTION_SOURCES,
        help="Keep only human-written or machine-generated questions",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ds = build_dataset(
        args.n_train, args.n_test, args.seed, args.max_side, args.source_dataset,
        question_source=args.question_source, source=args.source,
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
