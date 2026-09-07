"""Generate notebooks/00_setup_and_eval.ipynb from a script (avoids hand-editing JSON).

Usage: run `python notebooks/build_00_setup_and_eval.py` from the repository root.
"""

import json


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


cells = [
md("""# 00 · Setup, data sampling, and zero-shot evaluation

This notebook is only a driver; all core logic lives in `vlm_opd/*.py` in the repository. It does three things:

1. Install the evaluation dependencies (vLLM) and read `HF_TOKEN` from Colab Secrets
2. Sample ChartQA with a fixed seed, resize images, and push to a private Hub dataset repo
3. Run a 500-sample zero-shot evaluation for the student (Qwen3-VL-2B) and the teacher (Qwen3-VL-8B) with vLLM, writing json to `outputs/`

> Before running, edit `REPO_URL` and `DATA_REPO` in the config cell below.
> Start with `SMOKE = True` (100 train / 50 test) to exercise the whole pipeline, then switch to `False` for the full run."""),

code("""# ==================== Config ====================
REPO_URL  = "https://github.com/ffy208/On-Policy-Distillation-for-Vision-Language-Reasoning.git"
REPO_DIR  = "/content/vlm_opd_repo"
DATA_REPO = "ffyang/vlm_opd_chartqa"   # private Hub dataset repo
RESULT_REPO = "ffyang/vlm_opd_results"  # optional: also push evaluation json to the Hub

SMOKE = True                 # True: 100/50 rows smoke test; False: 3000/500 full scale
N_TRAIN = 100 if SMOKE else 3000
N_TEST  = 50  if SMOKE else 500
SEED    = 42

import os
os.environ["HF_HOME"] = "/content/hf_cache"      # model cache on local disk, not Drive
os.makedirs(os.environ["HF_HOME"], exist_ok=True)"""),

md("## 1. Install dependencies\n\nThe evaluation environment installs only vLLM and its matching transformers version, not TRL / PEFT (the training notebook installs those separately)."),

code("""# Install output is written to a log; only the tail is shown. Read the log if anything below fails to import.
!pip install -q -U "vllm>=0.8" "transformers>=4.57" "datasets>=3.0" "huggingface_hub>=0.26" qwen-vl-utils pillow pyyaml > /content/pip_install.log 2>&1; echo "pip exit code: $?"; tail -n 15 /content/pip_install.log

# vLLM upgrades torch (and torchvision) to a newer CUDA build, but Colab's preinstalled torchaudio stays on the
# old CUDA build. transformers imports torchaudio when loading any processor and torchaudio then fails its CUDA
# version check. We never use audio, so remove it instead of trying to match versions.
!pip uninstall -y -q torchaudio 2>/dev/null; echo "torchaudio removed"
"""),

code("""# Sanity check: these imports must succeed in the same interpreter that the evaluation subprocess will use.
import sys
print("python:", sys.executable, sys.version.split()[0])
import torch, transformers, vllm
from transformers import AutoProcessor  # pulls in processing_utils; this is where a stale torchaudio would fail
print("torch", torch.__version__, "(cuda", torch.version.cuda, ") | transformers", transformers.__version__, "| vllm", vllm.__version__)
print("cuda available:", torch.cuda.is_available())
GPU_NAME = torch.cuda.get_device_name(0)
GPU_GB = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU: {GPU_NAME} ({GPU_GB:.1f} GB)")
if GPU_GB < 30:
    print("WARNING: this is not a 40 GB A100. The 8B teacher does not fit; the notebook will fall back to the 4B teacher.")
    print("         For the real experiments switch to an A100 via Runtime > Change runtime type.")
if not torch.cuda.is_bf16_supported():
    print("WARNING: no bf16 support on this GPU; vLLM falls back to fp16 for evaluation.")"""),

md("## 2. Clone the repository and import `vlm_opd`"),

code("""import os, sys
if not os.path.isdir(REPO_DIR):
    !git clone {REPO_URL} {REPO_DIR}
else:
    !cd {REPO_DIR} && git pull
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

import importlib, vlm_opd.common, vlm_opd.prepare, vlm_opd.evaluate, vlm_opd.hub_utils
for m in (vlm_opd.common, vlm_opd.prepare, vlm_opd.evaluate, vlm_opd.hub_utils):
    importlib.reload(m)   # re-run this cell after editing a .py file
from vlm_opd import common, prepare, evaluate, hub_utils
print("Prompt template preview:\\n", common.build_prompt_text("What is the value for 2020?"))"""),

md("## 3. Read the HF token (stored in Colab Secrets as `HF_TOKEN`, never hard-coded)"),

code("""from huggingface_hub import login
token = common.get_hf_token()
assert token, "Add HF_TOKEN to Colab Secrets and grant this notebook access"
login(token=token, add_to_git_credential=False)
print("HF login OK")"""),

md("## 4. Sample the data and push to the Hub\n\nSkip this cell if the data has already been pushed; load it from the Hub instead."),

code("""import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ds = prepare.build_dataset(n_train=N_TRAIN, n_test=N_TEST, seed=SEED, token=token)
print(ds)
ex = ds["test"][0]
print(ex["id"], ex["image"].size, "|", ex["question"], "->", ex["answer"])
display(ex["image"])
prepare.push(ds, DATA_REPO, token=token)"""),

code("""# Load the test split back from the Hub to confirm it is readable
test_ds = common.load_hub_dataset(DATA_REPO, "test", token=token)
print(test_ds)"""),

md("""## 5. Zero-shot evaluation

Each model is evaluated in its own subprocess (`python -m vlm_opd.evaluate`), so GPU memory is fully
released when the process exits. This avoids relying on vLLM engine cleanup inside the running kernel.
The subprocess reads the token from the `HF_TOKEN` environment variable.
The first download of the 8B teacher is about 16 GB and takes a few minutes."""),

code("""import json, os, subprocess, sys
from pathlib import Path

OUT_DIR = Path("outputs"); OUT_DIR.mkdir(exist_ok=True)
suffix = "_smoke" if SMOKE else ""
os.environ["HF_TOKEN"] = token   # only in this process's environment for the subprocess; never written to disk

def eval_in_subprocess(model_id: str, tag: str, gpu_mem: float = 0.85, limit: int | None = None) -> dict:
    \"\"\"Run vlm_opd.evaluate in a child process, streaming its output and saving a full log.

    On failure the last lines of the log are printed so the real error is visible here,
    instead of only the CalledProcessError from the wrapper.
    \"\"\"
    out = OUT_DIR / f"eval_{tag}{suffix}.json"
    log_path = OUT_DIR / f"eval_{tag}{suffix}.log"
    cmd = [sys.executable, "-m", "vlm_opd.evaluate",
           "--model", model_id, "--data-repo", DATA_REPO, "--split", "test",
           "--out", str(out), "--tag", tag, "--seed", str(SEED), "--gpu-mem", str(gpu_mem)]
    if limit:
        cmd += ["--limit", str(limit)]
    print("$", " ".join(cmd))
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            log.write(line)
            sys.stdout.write(line)
        rc = proc.wait()
    if rc != 0:
        tail = "".join(log_path.read_text().splitlines(keepends=True)[-40:])
        raise RuntimeError(f"evaluation subprocess failed (exit {rc}); full log: {log_path}\\n--- log tail ---\\n{tail}")
    res = json.loads(out.read_text())
    print(f"[{tag}] acc = {res['accuracy']:.4f}  format = {res['format_rate']:.4f}  n = {res['n']}  ({res['elapsed_sec']}s)")
    return res"""),

code("""# 5a. Student zero-shot (baseline 0)
student_res = eval_in_subprocess(common.STUDENT_MODEL, "student_zeroshot")
print(student_res["records"][0]["output"][:600])"""),

code("""# 5b. Teacher zero-shot (teacher upper bound).
# The 8B teacher needs ~16 GB of weights alone, so on GPUs under 30 GB (e.g. a 15 GB T4) fall back to the 4B teacher.
TEACHER_FOR_THIS_GPU = common.TEACHER_MODEL if GPU_GB >= 30 else common.TEACHER_MODEL_SMALL
print("teacher:", TEACHER_FOR_THIS_GPU)
teacher_res = eval_in_subprocess(TEACHER_FOR_THIS_GPU, "teacher_zeroshot", gpu_mem=0.9)
print(teacher_res["records"][0]["output"][:600])"""),

md("## 6. Summarize and save\n\nIf teacher zero-shot accuracy is below 60%, switch to Geometry3K as described in the project plan."),

code("""import json
summary = {
    "n_test": len(test_ds),
    "seed": SEED,
    "gpu": GPU_NAME,
    "student_model": common.STUDENT_MODEL,
    "teacher_model": TEACHER_FOR_THIS_GPU,
    "student_zeroshot": student_res["accuracy"],
    "teacher_zeroshot": teacher_res["accuracy"],
    "student_format_rate": student_res["format_rate"],
    "teacher_format_rate": teacher_res["format_rate"],
}
print(json.dumps(summary, indent=2))
(OUT_DIR / f"summary_stage0{suffix}.json").write_text(json.dumps(summary, indent=2))
if summary["teacher_zeroshot"] < 0.60:
    print("WARNING: teacher zero-shot accuracy is below 60%; consider switching to Geometry3K")"""),

code("""# Optional: push the evaluation json to a private Hub repo so it survives a session disconnect
for name in (f"eval_student_zeroshot{suffix}.json", f"eval_teacher_zeroshot{suffix}.json", f"summary_stage0{suffix}.json"):
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
with open("notebooks/00_setup_and_eval.ipynb", "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("notebook written,", len(cells), "cells")
