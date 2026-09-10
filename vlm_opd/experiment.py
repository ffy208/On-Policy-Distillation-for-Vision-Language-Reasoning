"""One experiment point end to end: train (SFT or OPD) -> merge -> evaluate -> upload, idempotently.

Designed for batch schedulers: each invocation handles one (task, method, budget, seed) and skips
work whose result already exists in the Hub results repo, so a killed job can simply be resubmitted.
Training and evaluation run as child processes (same isolation the notebooks use).

Usage:
    python -m vlm_opd.experiment --task chartqa_human --method opd --budget 300 --seed 1
    python -m vlm_opd.experiment --task chartqa_human --method sft --budget 100 --seed 1 --sft-steps 20 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .common import get_hf_token

logger = logging.getLogger(__name__)

TASKS_FILE = Path(__file__).resolve().parent.parent / "configs" / "tasks.yaml"
METHODS = ("sft", "opd")


def load_tasks(path: str | Path = TASKS_FILE) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


@dataclass(frozen=True)
class PointNames:
    """Every name derived from (task, method, budget, seed)."""

    tag: str
    out_dir: str
    result_name: str
    merged_repo: str
    ckpt_repo: str | None

    @property
    def merged_dir(self) -> str:
        """Local merged model (base + LoRA) written by the trainer; used for evaluation in `local` artifact mode."""
        return f"{self.out_dir}/merged"

    @property
    def adapter_repo(self) -> str:
        """Small Hub repo holding only the final LoRA weights (SFT points; OPD keeps its adapter in ckpt_repo)."""
        return self.merged_repo.replace("_merged", "_lora")

    def ood_result_name(self, ood_repo: str) -> str:
        return self.result_name.replace(".json", f"_on_{ood_repo.split('/')[-1]}.json")


def merged_mode(defaults: dict[str, Any]) -> str:
    """Where merged models live: `local` (scratch disk; only adapters go to the Hub) or `hub` (push ~4 GB per point)."""
    mode = defaults.get("artifacts", {}).get("merged", "local")
    if mode not in ("local", "hub"):
        raise ValueError(f"artifacts.merged must be local or hub, got {mode!r}")
    return mode


def eval_model(names: PointNames, defaults: dict[str, Any]) -> str:
    """Model argument for evaluation: the local merged directory or the Hub merged repo."""
    return names.merged_dir if merged_mode(defaults) == "local" else names.merged_repo


def point_names(task: str, task_cfg: dict[str, Any], defaults: dict[str, Any], method: str, budget: int, seed: int) -> PointNames:
    """Hub repo names and result file name for one point. Seed-42 ChartQA points keep the notebook-era names."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    user = defaults["hub_user"]
    legacy = task_cfg.get("legacy_seed") == seed and task_cfg.get("legacy_prefix")
    if legacy:
        prefix = task_cfg["legacy_prefix"]
        return PointNames(
            tag=f"{method}_q{budget}",
            out_dir=f"ckpt/{method}_q{budget}",
            result_name=f"eval_{method}_q{budget}.json",
            merged_repo=f"{user}/vlm_opd_{method}_{prefix}_q{budget}_merged",
            ckpt_repo=f"{user}/vlm_opd_opd_{prefix}_q{budget}_ckpt" if method == "opd" else None,
        )
    base = f"{method}_{task}_q{budget}_s{seed}"
    return PointNames(
        tag=base,
        out_dir=f"ckpt/{base}",
        result_name=f"eval_{base}.json",
        merged_repo=f"{user}/vlm_opd_{base}_merged",
        ckpt_repo=f"{user}/vlm_opd_{base}_ckpt" if method == "opd" else None,
    )


def train_command(method: str, names: PointNames, task_cfg: dict[str, Any], defaults: dict[str, Any], budget: int, seed: int,
                  student: str, teacher: str, sft_steps: int | None, opd_steps: int | None) -> list[str]:
    """argv for the training module of one point."""
    py = sys.executable
    style = ["--prompt-style", task_cfg.get("prompt_style", "chart")]
    local = merged_mode(defaults) == "local"
    if method == "sft":
        c = defaults["sft"]
        publish = ["--merge", "--push-adapter-repo", names.adapter_repo] if local else ["--push-merged-repo", names.merged_repo]
        return [py, "-m", "vlm_opd.sft", *style, "--data-repo", task_cfg["sft_data_repo"], "--out-dir", names.out_dir,
                "--model", student, "--max-question-index", str(budget), "--max-steps", str(sft_steps or c["max_steps"]),
                "--lr", str(c["lr"]), "--batch", str(c["batch"]), "--grad-accum", str(c["grad_accum"]),
                "--lora-r", str(c["lora_r"]), "--seed", str(seed), *publish]
    c = defaults["opd"]
    publish = [] if local else ["--merged-repo", names.merged_repo]
    return [py, "-m", "vlm_opd.opd_trainer", *style, "--data-repo", task_cfg["data_repo"], "--out-dir", names.out_dir,
            "--student", student, "--teacher", teacher, "--limit", str(budget),
            "--total-steps", str(opd_steps or c["total_steps"]), "--batch-size", str(c["batch_size"]),
            "--micro-batch", str(c["micro_batch"]), "--lr", str(c["lr"]), "--ckpt-every", str(c["ckpt_every"]),
            "--kl-direction", c["kl_direction"], "--seed", str(seed),
            "--ckpt-repo", names.ckpt_repo or "", *publish]


def eval_command(model_repo: str, data_repo: str, out_path: str | Path, tag: str, seed: int, gpu_mem: float,
                 prompt_style: str = "chart") -> list[str]:
    return [sys.executable, "-m", "vlm_opd.evaluate", "--prompt-style", prompt_style, "--model", model_repo,
            "--data-repo", data_repo, "--split", "test", "--out", str(out_path), "--tag", tag, "--seed", str(seed),
            "--gpu-mem", str(gpu_mem)]


def result_on_hub(result_repo: str, name: str, token: str | None) -> bool:
    """True when outputs/<name> exists in the results repo."""
    from huggingface_hub import HfApi

    try:
        return f"outputs/{name}" in HfApi(token=token).list_repo_files(result_repo)
    except Exception as e:  # noqa: BLE001 - missing repo means nothing is done yet
        logger.info("results repo %s not readable (%s); assuming empty", result_repo, type(e).__name__)
        return False


def run(cmd: list[str], log_path: Path) -> None:
    """Run a child process, streaming its output to stdout and to a log file; raise on failure."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("$ %s", " ".join(cmd))
    with log_path.open("a") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:  # type: ignore[union-attr]
            log.write(line)
            sys.stdout.write(line)
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"{cmd[2]} failed with exit {rc}; log at {log_path}")


def run_point(task: str, method: str, budget: int, seed: int, tasks_file: str | Path = TASKS_FILE, student: str | None = None,
              teacher: str | None = None, sft_steps: int | None = None, opd_steps: int | None = None, dry_run: bool = False,
              skip_ood: bool = False) -> dict[str, Any]:
    """Train, evaluate in distribution, evaluate on every OOD set, upload each result; skip what already exists."""
    from .hub_utils import upload_small_file

    cfg = load_tasks(tasks_file)
    defaults, task_cfg = cfg["defaults"], cfg["tasks"][task]
    student = student or task_cfg.get("student", defaults["student"])
    teacher = teacher or task_cfg.get("teacher", defaults["teacher"])
    names = point_names(task, task_cfg, defaults, method, budget, seed)
    result_repo = defaults["result_repo"]
    token = None if dry_run else get_hf_token()
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    plan: dict[str, Any] = {"names": names.__dict__, "commands": [], "skipped": []}

    def maybe(name: str, cmd: list[str], log_tag: str) -> None:
        if not dry_run and result_on_hub(result_repo, name, token):
            plan["skipped"].append(name)
            logger.info("skip %s: already on the Hub", name)
            return
        plan["commands"].append(cmd)
        if dry_run:
            return
        run(cmd, Path("logs") / f"{log_tag}.log")
        upload_small_file(out_dir / name, result_repo, token=token)

    model = eval_model(names, defaults)
    ood_sets = [] if skip_ood else list(task_cfg.get("ood_eval", []))
    id_done = not dry_run and result_on_hub(result_repo, names.result_name, token)
    ood_missing = [r for r in ood_sets if dry_run or not result_on_hub(result_repo, names.ood_result_name(r), token)]
    # In local mode the merged model lives on this machine's disk; if it is gone but an evaluation is still
    # needed, run the trainer again (OPD resumes from its final Hub checkpoint and only merges; SFT retrains).
    need_model = (not id_done) or bool(ood_missing)
    model_missing = merged_mode(defaults) == "local" and not Path(names.merged_dir).exists()
    if not dry_run and id_done and not need_model:
        plan["skipped"].append(names.result_name)
        logger.info("skip %s: every result already on the Hub", names.tag)
        return plan

    if dry_run or not id_done or model_missing:
        cmd_train = train_command(method, names, task_cfg, defaults, budget, seed, student, teacher, sft_steps, opd_steps)
        plan["commands"].append(cmd_train)
        if not dry_run:
            run(cmd_train, Path("logs") / f"train_{names.tag}.log")
    if dry_run or not id_done:
        maybe(names.result_name, eval_command(model, task_cfg["data_repo"], out_dir / names.result_name, names.tag, seed,
                                              defaults["eval"]["gpu_mem"], task_cfg.get("prompt_style", "chart")),
              f"eval_{names.tag}")
    else:
        plan["skipped"].append(names.result_name)

    for ood_repo in ood_missing:
        name = names.ood_result_name(ood_repo)
        maybe(name, eval_command(model, ood_repo, out_dir / name, f"{names.tag}_on_{ood_repo.split('/')[-1]}",
                                 seed, defaults["eval"]["gpu_mem"], task_cfg.get("prompt_style", "chart")),
              f"eval_{names.tag}_ood")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one (task, method, budget, seed) experiment point")
    parser.add_argument("--task", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--budget", type=int, required=True, help="Number of training questions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--student", default=None)
    parser.add_argument("--teacher", default=None)
    parser.add_argument("--sft-steps", type=int, default=None, help="Override SFT optimizer steps (smoke tests)")
    parser.add_argument("--opd-steps", type=int, default=None, help="Override OPD steps (smoke tests)")
    parser.add_argument("--tasks-file", default=str(TASKS_FILE))
    parser.add_argument("--skip-ood", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running anything")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    plan = run_point(args.task, args.method, args.budget, args.seed, args.tasks_file, args.student, args.teacher,
                     args.sft_steps, args.opd_steps, args.dry_run, args.skip_ood)
    if args.dry_run:
        for cmd in plan["commands"]:
            print(" ".join(cmd))
    if plan["skipped"]:
        print("skipped:", ", ".join(plan["skipped"]))
    os.makedirs("outputs", exist_ok=True)


if __name__ == "__main__":
    main()
