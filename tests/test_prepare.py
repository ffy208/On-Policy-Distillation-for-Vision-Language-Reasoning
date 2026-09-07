"""Offline tests for prepare.py: sampling and field unification on synthetic images."""

import pytest

pytest.importorskip("datasets")

from datasets import ClassLabel, Dataset
from PIL import Image

from vlm_opd.prepare import filter_question_source, sample_split


def _fake_chartqa(n: int) -> Dataset:
    # Even indices are human-written, odd indices machine-generated (mirrors the real label order)
    rows = {
        "image": [Image.new("RGB", (1000 + i, 500)) for i in range(n)],
        "query": [f"Question {i}?" for i in range(n)],
        "label": [[str(i)] for i in range(n)],
        "human_or_machine": [i % 2 for i in range(n)],
    }
    ds = Dataset.from_dict(rows)
    return ds.cast_column("human_or_machine", ClassLabel(names=["human", "machine"]))


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


def test_filter_question_source():
    src = _fake_chartqa(10)
    human = filter_question_source(src, "human")
    machine = filter_question_source(src, "machine")
    assert len(human) == 5 and len(machine) == 5
    assert all(int(q.split()[1].rstrip("?")) % 2 == 0 for q in human["query"])
    assert len(filter_question_source(src, "all")) == 10
    with pytest.raises(ValueError):
        filter_question_source(src, "robot")


def test_sample_split_with_question_source():
    src = _fake_chartqa(20)
    out = sample_split(src, n=50, seed=42, prefix="t", question_source="human")
    assert len(out) == 10  # capped by the number of human questions
