"""Save checkpoints to the HuggingFace Hub and resume after a disconnect.

Design (see the project README, "Storage"):
- Save only the LoRA adapter weights + optimizer (and optional scheduler) state, never the base model.
- Hub repo layout:
    step_000050/adapter_model.safetensors, adapter_config.json, optimizer.pt, scheduler.pt
    step_000100/...
    latest.txt        # contains the most recent step number, e.g. "100"
- To resume, read latest.txt first and download only that step directory.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from .common import get_hf_token

logger = logging.getLogger(__name__)

LATEST_FILE = "latest.txt"
OPTIMIZER_FILE = "optimizer.pt"
SCHEDULER_FILE = "scheduler.pt"
META_FILE = "train_state.json"


def step_dirname(step: int) -> str:
    """Convert a step number into a zero-padded directory name so names sort lexically."""
    return f"step_{step:06d}"


def ensure_repo(repo_id: str, token: str | None = None) -> None:
    """Make sure the private model repo exists (create it if missing)."""
    api = HfApi(token=token or get_hf_token())
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)


def save_ckpt(
    model: Any,
    optimizer: Any,
    step: int,
    repo_id: str,
    local_dir: str | Path = "ckpt",
    scheduler: Any = None,
    extra_state: dict[str, Any] | None = None,
    push: bool = True,
    token: str | None = None,
    keep_local: int = 1,
    keep_remote: int = 1,
) -> Path:
    """Save one checkpoint locally and push it to the Hub.

    Args:
        model: PEFT-wrapped model implementing `save_pretrained` (writes only LoRA weights).
        optimizer: torch optimizer; its `state_dict()` is saved.
        step: Current training step, used for naming.
        repo_id: Private Hub repo, e.g. "username/vlm_opd_ckpt".
        local_dir: Local staging directory.
        scheduler: Optional learning-rate scheduler.
        extra_state: Extra training state to record (RNG seed, data cursor, ...) as json.
        push: When False, only save locally (unit tests / offline debugging).
        token: HF token; read automatically when None.
        keep_local: Maximum number of local step directories to keep (avoids filling Colab disk).
        keep_remote: Maximum number of step directories to keep on the Hub (0 keeps all). Resume only
            needs the latest one, and each step holds the LoRA weights plus optimizer state (~0.8 GB), so
            private-storage quota is what this protects.

    Returns:
        Path of the local checkpoint directory.
    """
    import json

    import torch

    local_dir = Path(local_dir)
    ckpt_dir = local_dir / step_dirname(step)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(ckpt_dir))
    torch.save(optimizer.state_dict(), ckpt_dir / OPTIMIZER_FILE)
    if scheduler is not None:
        torch.save(scheduler.state_dict(), ckpt_dir / SCHEDULER_FILE)
    meta = {"step": step, **(extra_state or {})}
    (ckpt_dir / META_FILE).write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    latest_path = local_dir / LATEST_FILE
    latest_path.write_text(str(step))
    logger.info("Checkpoint written locally to %s", ckpt_dir)

    if push:
        token = token or get_hf_token()
        api = HfApi(token=token)
        api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(ckpt_dir),
            path_in_repo=step_dirname(step),
            commit_message=f"checkpoint step {step}",
        )
        # Upload latest.txt separately so the pointer only moves after the step folder is complete
        api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=str(latest_path),
            path_in_repo=LATEST_FILE,
            commit_message=f"update latest -> {step}",
        )
        logger.info("Checkpoint step %d pushed to Hub repo %s", step, repo_id)
        if keep_remote > 0:
            prune_remote(repo_id, keep_remote, token=token)

    _prune_local(local_dir, keep_local)
    return ckpt_dir


def remote_steps_to_prune(files: list[str], keep: int) -> list[str]:
    """Step directories present in a repo file list, oldest first, except the newest `keep`."""
    dirs = sorted({f.split("/", 1)[0] for f in files if f.startswith("step_") and "/" in f})
    return dirs[:-keep] if keep > 0 else []


def prune_remote(repo_id: str, keep: int = 1, token: str | None = None) -> list[str]:
    """Delete all but the newest `keep` step directories from a Hub checkpoint repo. Returns what was removed."""
    api = HfApi(token=token or get_hf_token())
    try:
        old = remote_steps_to_prune(api.list_repo_files(repo_id), keep)
        for d in old:
            api.delete_folder(path_in_repo=d, repo_id=repo_id, commit_message=f"prune {d}")
    except Exception as e:  # noqa: BLE001 - pruning is housekeeping; never fail training over it
        logger.warning("Could not prune old checkpoints in %s: %s", repo_id, e)
        return []
    if old:
        logger.info("Pruned %s from %s", ", ".join(old), repo_id)
    return old


def _prune_local(local_dir: Path, keep: int) -> None:
    """Keep only the most recent `keep` local step directories."""
    step_dirs = sorted(p for p in local_dir.glob("step_*") if p.is_dir())
    for old in step_dirs[:-keep] if keep > 0 else step_dirs:
        shutil.rmtree(old, ignore_errors=True)
        logger.debug("Removed old local checkpoint %s", old)


def get_latest_step(repo_id: str, token: str | None = None) -> int | None:
    """Read the latest step recorded in latest.txt on the Hub; None if the repo/file is missing."""
    token = token or get_hf_token()
    try:
        path = hf_hub_download(repo_id, LATEST_FILE, repo_type="model", token=token)
    except Exception as e:  # noqa: BLE001 - missing repo / file both mean "no checkpoint yet"
        logger.info("Hub repo %s has no latest.txt yet (%s)", repo_id, type(e).__name__)
        return None
    text = Path(path).read_text().strip()
    return int(text) if text else None


def load_latest(
    repo_id: str,
    local_dir: str | Path = "ckpt",
    token: str | None = None,
) -> tuple[int, Path] | None:
    """Download the latest checkpoint from the Hub.

    Returns:
        `(step, local_dir)`, or None when the repo holds no checkpoint.
        The caller then loads the LoRA adapter with `PeftModel.from_pretrained(base, path)`
        and restores the optimizer with `restore_train_state(path, optimizer, scheduler)`.
    """
    token = token or get_hf_token()
    step = get_latest_step(repo_id, token)
    if step is None:
        return None
    subdir = step_dirname(step)
    local_dir = Path(local_dir)
    snapshot_download(
        repo_id,
        repo_type="model",
        token=token,
        allow_patterns=[f"{subdir}/*"],
        local_dir=str(local_dir),
    )
    ckpt_dir = local_dir / subdir
    logger.info("Fetched checkpoint step %d from the Hub -> %s", step, ckpt_dir)
    return step, ckpt_dir


def restore_train_state(
    ckpt_dir: str | Path,
    optimizer: Any,
    scheduler: Any = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Restore optimizer / scheduler state from a checkpoint directory and return train_state.json."""
    import json

    import torch

    ckpt_dir = Path(ckpt_dir)
    optimizer.load_state_dict(torch.load(ckpt_dir / OPTIMIZER_FILE, map_location=device))
    sched_path = ckpt_dir / SCHEDULER_FILE
    if scheduler is not None and sched_path.exists():
        scheduler.load_state_dict(torch.load(sched_path, map_location=device))
    meta_path = ckpt_dir / META_FILE
    meta: dict[str, Any] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return meta


def upload_small_file(
    local_path: str | Path,
    repo_id: str,
    path_in_repo: str | None = None,
    repo_type: str = "model",
    token: str | None = None,
) -> None:
    """Push a small file (evaluation json, logs) to the Hub, into a model or dataset repo."""
    local_path = Path(local_path)
    api = HfApi(token=token or get_hf_token())
    api.create_repo(repo_id, repo_type=repo_type, private=True, exist_ok=True)
    api.upload_file(
        repo_id=repo_id,
        repo_type=repo_type,
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo or f"outputs/{local_path.name}",
        commit_message=f"upload {local_path.name}",
    )
