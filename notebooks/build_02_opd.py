"""Generate notebooks/02_opd.ipynb (edit this script, then run it from the repository root)."""

import json


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


cells = [
md("""# 02 · On-policy distillation (OPD)

The student samples its own rollouts (temperature 1.0), the frozen 8B teacher scores every generated token, and the
per-token KL between the two distributions is minimized on the LoRA parameters. Prompt and image positions never enter
the loss. Checkpoints (LoRA + optimizer) go to the Hub every `CKPT_EVERY` steps and the run resumes automatically after a
disconnect: just re-run this notebook.

Budget on an A100 40 GB (measured in the smoke run): 33 s per step at batch 8 with rollout taking about 90% of it and
peak memory 27 GB. Rollout is latency-bound, so the full run uses batch 16 for nearly double the throughput; 150 steps
(2400 rollouts) take roughly 1.5 to 2 hours. Colab sessions may end before that; the resume logic exists for exactly this reason.
Start with `SMOKE = True` (100 questions, 20 steps, checkpoint every 10) to confirm loss decreases and memory is stable."""),

code("""# ==================== Config ====================
REPO_URL  = "https://github.com/ffy208/On-Policy-Distillation-for-Vision-Language-Reasoning.git"
REPO_DIR  = "/content/vlm_opd_repo"
QUESTION_SOURCE = "human"
DATA_REPO   = f"ffyang/vlm_opd_chartqa_{QUESTION_SOURCE}"
RESULT_REPO = "ffyang/vlm_opd_results"

SMOKE = True
SEED  = 42
RUN_NAME    = f"opd_{QUESTION_SOURCE}" + ("_smoke" if SMOKE else "")
CKPT_REPO   = f"ffyang/vlm_opd_{RUN_NAME}_ckpt"
MERGED_REPO = f"ffyang/vlm_opd_{RUN_NAME}_merged"
TOTAL_STEPS = 20 if SMOKE else 150     # 150 steps x batch 16 = 2400 rollouts (same sample budget as 300 x 8)
CKPT_EVERY  = 10 if SMOKE else 25
LIMIT       = 100 if SMOKE else None
BATCH_SIZE, MICRO_BATCH = (8, 4) if SMOKE else (16, 4)   # rollout time is latency-bound, so batch 16 nearly doubles throughput
LR = 5e-5
KL_DIRECTION = "reverse"     # "forward" for the ablation
suffix = "_smoke" if SMOKE else ""

import os
os.environ["HF_HOME"] = "/content/hf_cache"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)"""),

md("## 1. Install dependencies"),

code("""!pip install -q -U "vllm>=0.8" "transformers>=4.57" "datasets>=3.0" "huggingface_hub>=0.26" "peft>=0.13" "accelerate>=1.0" qwen-vl-utils pyyaml > /content/pip_install.log 2>&1; echo "pip exit code: $?"; tail -n 15 /content/pip_install.log
!pip uninstall -y -q torchaudio 2>/dev/null; echo "torchaudio removed"
!pip uninstall -y -q torchao 2>/dev/null; echo "torchao removed\""""),

code("""# Pillow repair. Colab can end up with a mix of Pillow 11 and 12 files in one directory
# (ImportError: cannot import name '_Ink' from 'PIL._typing'). Two subtleties:
#  - a plain force-reinstall does not clear the stale files and Colab may pin pillow via PIP_CONSTRAINT;
#  - the Colab kernel imports PIL at startup, so after repairing the files on disk the kernel still holds the old
#    modules in memory and must be restarted once. This cell is idempotent: it skips when Pillow is healthy, and
#    restarts the runtime automatically after a repair. After the restart, run this cell again (it will skip), then continue.
import glob, os, shutil, subprocess, sys, sysconfig, time

def pillow_healthy() -> bool:
    chk = subprocess.run([sys.executable, "-c", "import PIL, PIL.ImageText; assert int(PIL.__version__.split('.')[0]) >= 12; print(PIL.__version__)"],
                         capture_output=True, text=True)
    print("pillow check:", (chk.stdout or chk.stderr).strip())
    return chk.returncode == 0

if pillow_healthy():
    print("Pillow OK, no repair needed")
else:
    site = sysconfig.get_paths()["purelib"]
    print("PIP_CONSTRAINT:", os.environ.get("PIP_CONSTRAINT"))
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "pillow"], check=False)
    for p in glob.glob(os.path.join(site, "PIL")) + glob.glob(os.path.join(site, "[Pp]illow*")):
        shutil.rmtree(p, ignore_errors=True); print("removed", p)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "pillow>=12"], check=True,
                   env={**os.environ, "PIP_CONSTRAINT": ""})
    assert pillow_healthy(), "Pillow is still broken after reinstall"
    print("Pillow repaired on disk. The kernel still holds the old PIL modules, so the runtime restarts now.")
    print("This is expected, not a crash. When it comes back, run the cells from the top again: the install is instant,")
    print("this cell reports 'Pillow OK', and everything continues.")
    time.sleep(2)
    try:
        get_ipython().kernel.do_shutdown(restart=True)   # clean Jupyter restart, no 'session crashed' dialog
    except Exception:
        os.kill(os.getpid(), 9)"""),

code("""import sys
from PIL import Image
import torch, transformers, vllm, peft
from transformers import AutoProcessor
from peft.import_utils import is_torchao_available
assert not is_torchao_available()
GPU_NAME = torch.cuda.get_device_name(0); GPU_GB = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU: {GPU_NAME} ({GPU_GB:.1f} GB) | torch {torch.__version__} | transformers {transformers.__version__} | peft {peft.__version__}")
assert GPU_GB >= 30 and torch.cuda.is_bf16_supported(), "OPD needs an A100 40 GB: teacher 8B + student + activations\""""),

md("## 2. Clone the repository, log in"),

code("""import os, sys
if not os.path.isdir(REPO_DIR):
    !git clone {REPO_URL} {REPO_DIR}
else:
    !cd {REPO_DIR} && git pull
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)
from vlm_opd import common, hub_utils
from huggingface_hub import login
token = common.get_hf_token(); assert token, "Add HF_TOKEN to Colab Secrets"
login(token=token, add_to_git_credential=False)
os.environ["HF_TOKEN"] = token"""),

code("""import json, subprocess, sys
from pathlib import Path
OUT_DIR = Path("outputs"); OUT_DIR.mkdir(exist_ok=True)

def run_module(module: str, args: list, tag: str) -> None:
    # Run `python -m <module> <args>` in a child process, streaming output and saving a log.
    log_path = OUT_DIR / f"{tag}.log"
    cmd = [sys.executable, "-m", module, *map(str, args)]
    print("$", " ".join(cmd))
    with log_path.open("a") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            log.write(line); sys.stdout.write(line)
        rc = proc.wait()
    if rc != 0:
        tail = "".join(log_path.read_text().splitlines(keepends=True)[-40:])
        raise RuntimeError(f"{module} failed (exit {rc}); log: {log_path}\\n--- log tail ---\\n{tail}")"""),

md("## 3. Train\n\nRe-running this cell after a disconnect resumes from the latest Hub checkpoint. The per-step log is `ckpt/<run>/opd_log.jsonl` and is also uploaded with every checkpoint."),

code("""opd_args = ["--data-repo", DATA_REPO, "--out-dir", f"ckpt/{RUN_NAME}",
            "--ckpt-repo", CKPT_REPO, "--merged-repo", MERGED_REPO,
            "--batch-size", BATCH_SIZE, "--micro-batch", MICRO_BATCH, "--lr", LR,
            "--total-steps", TOTAL_STEPS, "--ckpt-every", CKPT_EVERY, "--kl-direction", KL_DIRECTION, "--seed", SEED]
if LIMIT:
    opd_args += ["--limit", LIMIT]
run_module("vlm_opd.opd_trainer", opd_args, f"train_{RUN_NAME}")"""),

code("""# Training curves from the jsonl log
log = [json.loads(l) for l in Path(f"ckpt/{RUN_NAME}/opd_log.jsonl").read_text().splitlines()]
print(f"{len(log)} steps | first kl {log[0]['kl']:.4f} -> last kl {log[-1]['kl']:.4f}")
print(f"gen_len first {log[0]['gen_len_mean']:.0f} -> last {log[-1]['gen_len_mean']:.0f} | eos_rate last {log[-1]['eos_rate']:.2f}")
print(f"mean step time {sum(r['t_step'] for r in log)/len(log):.1f}s "
      f"(rollout {sum(r['t_rollout'] for r in log)/len(log):.1f}, teacher {sum(r['t_teacher'] for r in log)/len(log):.1f}, "
      f"student {sum(r['t_student'] for r in log)/len(log):.1f}) | peak mem {max(r['peak_mem_gb'] or 0 for r in log)} GB")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
steps = [r["step"] for r in log]
ax[0].plot(steps, [r["kl"] for r in log]); ax[0].set_title("per-token KL"); ax[0].set_xlabel("step")
ax[1].plot(steps, [r["gen_len_mean"] for r in log]); ax[1].set_title("mean rollout length"); ax[1].set_xlabel("step")
ax[2].plot(steps, [r["rollout_acc"] for r in log]); ax[2].set_title("rollout accuracy (train)"); ax[2].set_xlabel("step")
plt.tight_layout(); plt.savefig(OUT_DIR / f"opd_curves{suffix}.png", dpi=120); plt.show()"""),

md("## 4. Evaluate the OPD student"),

code("""eval_out = OUT_DIR / f"eval_opd_student{suffix}.json"
run_module("vlm_opd.evaluate", ["--model", MERGED_REPO, "--data-repo", DATA_REPO, "--split", "test",
                                 "--out", eval_out, "--tag", "opd_student", "--seed", SEED, "--gpu-mem", 0.85], f"eval_{RUN_NAME}")
opd_res = json.loads(eval_out.read_text())
print(f"OPD student acc = {opd_res['accuracy']:.4f}  format = {opd_res['format_rate']:.4f}  n = {opd_res['n']}")
print(opd_res["records"][0]["output"][:600])"""),

md("## 5. Summary and upload"),

code("""summary = {
    "gpu": GPU_NAME, "seed": SEED, "question_source": QUESTION_SOURCE, "smoke": SMOKE, "kl_direction": KL_DIRECTION,
    "total_steps": len(log), "batch_size": BATCH_SIZE, "lr": LR,
    "kl_first": log[0]["kl"], "kl_last": log[-1]["kl"],
    "gen_len_first": log[0]["gen_len_mean"], "gen_len_last": log[-1]["gen_len_mean"],
    "mean_step_sec": sum(r["t_step"] for r in log) / len(log), "peak_mem_gb": max(r["peak_mem_gb"] or 0 for r in log),
    "opd_student_accuracy": opd_res["accuracy"], "opd_student_format_rate": opd_res["format_rate"],
}
print(json.dumps(summary, indent=2))
(OUT_DIR / f"summary_stage2{suffix}.json").write_text(json.dumps(summary, indent=2))
for name in (f"eval_opd_student{suffix}.json", f"summary_stage2{suffix}.json", f"opd_curves{suffix}.png"):
    hub_utils.upload_small_file(OUT_DIR / name, RESULT_REPO, token=token)
print("Results pushed to", RESULT_REPO)"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "A100"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
with open("notebooks/02_opd.ipynb", "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("notebook written,", len(cells), "cells")
