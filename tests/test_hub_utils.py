"""Offline tests for hub_utils.py: only local save / restore logic, no network access."""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from vlm_opd.hub_utils import (
    LATEST_FILE,
    META_FILE,
    restore_train_state,
    save_ckpt,
    step_dirname,
)


class _FakePeftModel:
    """Stand-in for a PEFT model: save_pretrained writes a single small file."""

    def save_pretrained(self, path: str) -> None:
        Path(path, "adapter_config.json").write_text("{}")


def test_step_dirname_zero_padded():
    assert step_dirname(50) == "step_000050"


def test_save_and_restore_local(tmp_path):
    model = _FakePeftModel()
    param = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.AdamW([param], lr=1e-3)
    param.grad = torch.ones(2)
    opt.step()

    ckpt_dir = save_ckpt(
        model, opt, step=50, repo_id="dummy/repo", local_dir=tmp_path, push=False,
        extra_state={"seed": 42},
    )
    assert ckpt_dir == tmp_path / "step_000050"
    assert (ckpt_dir / "adapter_config.json").exists()
    assert (ckpt_dir / "optimizer.pt").exists()
    assert (tmp_path / LATEST_FILE).read_text() == "50"
    assert json.loads((ckpt_dir / META_FILE).read_text()) == {"step": 50, "seed": 42}

    new_opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=1e-3)
    meta = restore_train_state(ckpt_dir, new_opt)
    assert meta["step"] == 50
    assert new_opt.state_dict()["state"][0]["step"] == opt.state_dict()["state"][0]["step"]


def test_prune_keeps_latest_only(tmp_path):
    model = _FakePeftModel()
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    for step in (50, 100, 150):
        save_ckpt(model, opt, step, "dummy/repo", local_dir=tmp_path, push=False, keep_local=1)
    remaining = sorted(p.name for p in tmp_path.glob("step_*"))
    assert remaining == ["step_000150"]


def test_remote_steps_to_prune_keeps_newest():
    from vlm_opd.hub_utils import remote_steps_to_prune

    files = ["latest.txt", "opd_log.jsonl", "step_000025/adapter_model.safetensors", "step_000025/optimizer.pt",
             "step_000050/adapter_model.safetensors", "step_000100/adapter_model.safetensors"]
    assert remote_steps_to_prune(files, 1) == ["step_000025", "step_000050"]
    assert remote_steps_to_prune(files, 2) == ["step_000025"]
    assert remote_steps_to_prune(files, 0) == [] and remote_steps_to_prune(["latest.txt"], 1) == []
