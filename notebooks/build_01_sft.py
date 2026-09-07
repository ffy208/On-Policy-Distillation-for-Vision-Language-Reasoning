"""Generate notebooks/01_sft.ipynb (edit this script, then run it from the repository root)."""

import json


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


cells = [
md("""# 01 · Supervised distillation baseline (SFT)

Three steps, each in its own subprocess so GPU memory is released between them:

1. **Teacher generation** (vLLM): the 8B teacher answers every training question once at temperature 0.7; only correct solutions are kept and pushed to a Hub dataset.
2. **LoRA SFT** (transformers Trainer + peft): rank 64 on the language model, vision tower frozen, 2 epochs, lr 1e-4. The adapter and a merged copy of the weights are pushed to the Hub.
3. **Evaluation** (vLLM): the merged student on the 500-row test split, same scorer as Stage 0.

peft and accelerate install next to vLLM without conflicts, so no runtime restart is needed between steps.
Start with `SMOKE = True` (100 questions, 20 optimizer steps), then switch to `False`."""),

code("""# ==================== Config ====================
REPO_URL  = "https://github.com/ffy208/On-Policy-Distillation-for-Vision-Language-Reasoning.git"
REPO_DIR  = "/content/vlm_opd_repo"
QUESTION_SOURCE = "human"
DATA_REPO   = f"ffyang/vlm_opd_chartqa_{QUESTION_SOURCE}"        # from notebook 00
SFT_DATA_REPO = f"ffyang/vlm_opd_sft_{QUESTION_SOURCE}"          # teacher solutions (written here)
ADAPTER_REPO  = f"ffyang/vlm_opd_sft_{QUESTION_SOURCE}_lora"
MERGED_REPO   = f"ffyang/vlm_opd_sft_{QUESTION_SOURCE}_merged"
RESULT_REPO   = "ffyang/vlm_opd_results"

SMOKE = True
SEED  = 42
GEN_LIMIT  = 100 if SMOKE else None    # questions sent to the teacher
MAX_STEPS  = 20  if SMOKE else -1      # optimizer steps (-1 = full 2 epochs)
suffix = "_smoke" if SMOKE else ""

import os
os.environ["HF_HOME"] = "/content/hf_cache"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)"""),

md("## 1. Install dependencies"),

code("""!pip install -q -U "vllm>=0.8" "transformers>=4.57" "datasets>=3.0" "huggingface_hub>=0.26" "peft>=0.13" "accelerate>=1.0" qwen-vl-utils pyyaml > /content/pip_install.log 2>&1; echo "pip exit code: $?"; tail -n 15 /content/pip_install.log
!pip uninstall -y -q torchaudio 2>/dev/null; echo "torchaudio removed\""""),

code("""# Pillow repair. Colab ends up with a mix of Pillow 11 and 12 files in one directory
# (ImportError: cannot import name '_Ink' from 'PIL._typing'); a plain force-reinstall does not clear it and Colab may pin
# pillow through PIP_CONSTRAINT. Remove every trace, install Pillow 12 ignoring constraints, verify in a fresh subprocess.
import glob, os, shutil, subprocess, sys, sysconfig
site = sysconfig.get_paths()["purelib"]
print("PIP_CONSTRAINT:", os.environ.get("PIP_CONSTRAINT"))
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "pillow"], check=False)
for p in glob.glob(os.path.join(site, "PIL")) + glob.glob(os.path.join(site, "[Pp]illow*")):
    shutil.rmtree(p, ignore_errors=True); print("removed", p)
env = {**os.environ, "PIP_CONSTRAINT": ""}
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "pillow>=12"], check=True, env=env)
chk = subprocess.run([sys.executable, "-c", "import PIL, PIL.ImageText; print('pillow', PIL.__version__, 'ok')"],
                     capture_output=True, text=True)
print(chk.stdout or chk.stderr)
assert chk.returncode == 0, "Pillow is still broken, see the message above\""""),

code("""import sys
print("python:", sys.executable, sys.version.split()[0])
from PIL import Image
import torch, transformers, vllm, peft, accelerate
from transformers import AutoProcessor
print("torch", torch.__version__, "(cuda", torch.version.cuda, ") | transformers", transformers.__version__,
      "| vllm", vllm.__version__, "| peft", peft.__version__)
GPU_NAME = torch.cuda.get_device_name(0); GPU_GB = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU: {GPU_NAME} ({GPU_GB:.1f} GB)")
assert GPU_GB >= 30, "Stage 1 with the 8B teacher needs an A100 40 GB (Runtime > Change runtime type)"
assert torch.cuda.is_bf16_supported(), "bf16 training requires an Ampere or newer GPU\""""),

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
os.environ["HF_TOKEN"] = token   # for the subprocesses; never written to disk"""),

code("""import json, subprocess, sys
from pathlib import Path
OUT_DIR = Path("outputs"); OUT_DIR.mkdir(exist_ok=True)

def run_module(module: str, args: list, tag: str) -> None:
    # Run `python -m <module> <args>` in a child process, streaming output and saving a log.
    log_path = OUT_DIR / f"{tag}{suffix}.log"
    cmd = [sys.executable, "-m", module, *map(str, args)]
    print("$", " ".join(cmd))
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            log.write(line); sys.stdout.write(line)
        rc = proc.wait()
    if rc != 0:
        tail = "".join(log_path.read_text().splitlines(keepends=True)[-40:])
        raise RuntimeError(f"{module} failed (exit {rc}); log: {log_path}\\n--- log tail ---\\n{tail}")"""),

md("## 3. Teacher generation\n\nThe 8B teacher answers each training question once (temperature 0.7). Solutions whose final answer is wrong are dropped. Typical acceptance is 60 to 75 percent."),

code("""gen_args = ["--model", common.TEACHER_MODEL, "--data-repo", DATA_REPO, "--split", "train",
            "--out-repo", SFT_DATA_REPO, "--stats", OUT_DIR / f"teacher_generation{suffix}.json",
            "--num-samples", 1, "--temperature", 0.7, "--seed", SEED, "--gpu-mem", 0.9]
if GEN_LIMIT:
    gen_args += ["--limit", GEN_LIMIT]
run_module("vlm_opd.generate_teacher", gen_args, "teacher_generation")
gen_stats = json.loads((OUT_DIR / f"teacher_generation{suffix}.json").read_text())
print(json.dumps(gen_stats, indent=2))"""),

md("## 4. LoRA SFT\n\nEffective batch 16 (4 x 4 accumulation), lr 1e-4 cosine, 2 epochs. On an A100 the full run on ~2000 kept solutions takes roughly 20 to 30 minutes. The adapter and the merged weights are pushed to the Hub."),

code("""sft_args = ["--data-repo", SFT_DATA_REPO, "--out-dir", f"ckpt/sft_{QUESTION_SOURCE}{suffix}",
            "--epochs", 2, "--lr", 1e-4, "--batch", 4, "--grad-accum", 4, "--lora-r", 64,
            "--seed", SEED, "--max-steps", MAX_STEPS,
            "--push-adapter-repo", ADAPTER_REPO + suffix, "--push-merged-repo", MERGED_REPO + suffix]
run_module("vlm_opd.sft", sft_args, "sft_train")
sft_metrics = json.loads(Path(f"ckpt/sft_{QUESTION_SOURCE}{suffix}/sft_metrics.json").read_text())
print({k: v for k, v in sft_metrics.items() if k != "log_history"})
losses = [(h["step"], h["loss"]) for h in sft_metrics["log_history"] if "loss" in h]
print("loss curve:", losses)"""),

md("## 5. Evaluate the SFT student\n\nSame scorer and test split as Stage 0, using the merged weights as a regular vLLM model."),

code("""eval_out = OUT_DIR / f"eval_sft_student{suffix}.json"
run_module("vlm_opd.evaluate", ["--model", MERGED_REPO + suffix, "--data-repo", DATA_REPO, "--split", "test",
                                 "--out", eval_out, "--tag", "sft_student", "--seed", SEED, "--gpu-mem", 0.85], "eval_sft")
sft_res = json.loads(eval_out.read_text())
print(f"SFT student acc = {sft_res['accuracy']:.4f}  format = {sft_res['format_rate']:.4f}  n = {sft_res['n']}")
print(sft_res["records"][0]["output"][:600])"""),

md("## 6. Summary and upload"),

code("""summary = {
    "gpu": GPU_NAME, "seed": SEED, "question_source": QUESTION_SOURCE, "smoke": SMOKE,
    "teacher_acceptance_rate": gen_stats["acceptance_rate"], "n_sft_rows": gen_stats["n_kept"],
    "sft_train_loss": sft_metrics["train_loss"], "sft_steps": sft_metrics["global_steps"],
    "sft_student_accuracy": sft_res["accuracy"], "sft_student_format_rate": sft_res["format_rate"],
}
print(json.dumps(summary, indent=2))
(OUT_DIR / f"summary_stage1{suffix}.json").write_text(json.dumps(summary, indent=2))
for name in (f"teacher_generation{suffix}.json", f"eval_sft_student{suffix}.json", f"summary_stage1{suffix}.json"):
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
with open("notebooks/01_sft.ipynb", "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("notebook written,", len(cells), "cells")
