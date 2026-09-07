"""Offline tests for prepare.py: sampling and field unification on synthetic images."""

import pytest

pytest.importorskip("datasets")

from datasets import Dataset
from PIL import Image

from vlm_opd.prepare import sample_split


def _fake_chartqa(n: int) -> Dataset:
    rows = {
        "image": [Image.new("RGB", (1000 + i, 500)) for i in range(n)],
        "query": [f"Question {i}?" for i in range(n)],
        "label": [[str(i)] for i in range(n)],
        "human_or_machine": [0] * n,
    }
    return Dataset.from_dict(rows)


def test_sample_split_unifies_fields_and_resizes():
    src = _fake_chartqa(10)
    out = sample_split(src, n=4, seed=42, prefix="train", max_side=768)
    assert len(out) == 4
    assert set(out.column_names) == {"id", "image", "question", "answer"}
    ex = out[0]
    assert ex["id"].startswith("train_")
    assert isinstance(ex["answer"], str)
    assert max(ex["image"].size) <= 768


def test_sample_split_is_deterministic():
    src = _fake_chartqa(20)
    a = sample_split(src, 5, seed=42, prefix="t")["question"]
    b = sample_split(src, 5, seed=42, prefix="t")["question"]
    c = sample_split(src, 5, seed=7, prefix="t")["question"]
    assert a == b
    assert a != c
