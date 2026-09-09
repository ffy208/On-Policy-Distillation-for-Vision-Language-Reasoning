"""Generate notebooks/04_token_feedback.ipynb (edit this script, then run it from the repository root)."""

import json


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


cells = [
md("""# 04 · Where does the teacher's token-level feedback land?

The student samples a solution for each of 100 test questions (temperature 1.0, the same regime as OPD training).
Teacher and student then score the identical sequence; for every generated token we record the reverse KL
(student || teacher) and the log-probability ratio of the sampled token. Tokens are classified as `chart_value`
(numbers read from the chart), `arithmetic` (numbers and operators on lines with an arithmetic cue), `answer`
(the final `Answer:` line) or `text` (everything else), and feedback mass is aggregated per class.

Two runs: the zero-shot student (what OPD starts from) and the OPD-300 student (what remains after training).
About 20 GPU minutes in total."""),

code("""# ==================== Config ====================
REPO_URL  = "https://github.com/ffy208/On-Policy-Distillation-for-Vision-Language-Reasoning.git"
REPO_DIR  = "/content/vlm_opd_repo"
QUESTION_SOURCE = "human"
DATA_REPO   = f"ffyang/vlm_opd_chartqa_{QUESTION_SOURCE}"
RESULT_REPO = "ffyang/vlm_opd_results"
N_QUESTIONS = 100
SEED = 42
STUDENTS = {
    "baseline": "Qwen/Qwen3-VL-2B-Instruct",
    "opd_q300": f"ffyang/vlm_opd_opd_{QUESTION_SOURCE}_q300_merged",
}

import os
os.environ["HF_HOME"] = "/content/hf_cache"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)"""),

md("## 1. Install dependencies"),

code("""!pip install -q -U "transformers>=4.57" "datasets>=3.0" "huggingface_hub>=0.26" "peft>=0.13" "accelerate>=1.0" qwen-vl-utils pyyaml matplotlib > /content/pip_install.log 2>&1; echo "pip exit code: $?"; tail -n 15 /content/pip_install.log
!pip uninstall -y -q torchaudio 2>/dev/null; echo "torchaudio removed"
!pip uninstall -y -q torchao 2>/dev/null; echo "torchao removed\""""),

code("""# Pillow repair (see README, Colab issues). Idempotent; restarts the runtime once if a repair was needed.
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
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "pillow"], check=False)
    for p in glob.glob(os.path.join(site, "PIL")) + glob.glob(os.path.join(site, "[Pp]illow*")):
        shutil.rmtree(p, ignore_errors=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "pillow>=12"], check=True, env={**os.environ, "PIP_CONSTRAINT": ""})
    assert pillow_healthy(), "Pillow is still broken after reinstall"
    print("Pillow repaired on disk; the runtime restarts now (expected, not a crash). Re-run from the top afterwards.")
    time.sleep(2)
    try:
        get_ipython().kernel.do_shutdown(restart=True)
    except Exception:
        os.kill(os.getpid(), 9)"""),

code("""import sys
from PIL import Image
import torch, transformers
from transformers import AutoProcessor
GPU_NAME = torch.cuda.get_device_name(0); GPU_GB = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU: {GPU_NAME} ({GPU_GB:.1f} GB) | torch {torch.__version__} | transformers {transformers.__version__}")
assert GPU_GB >= 30, "teacher 8B + student need an A100 40 GB\""""),

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
os.environ["HF_TOKEN"] = token

import json, subprocess
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

md("## 3. Score rollouts for each student"),

code("""for name, model_id in STUDENTS.items():
    out = OUT_DIR / f"token_feedback_{name}.jsonl"
    if out.exists():
        print("exists, skipping", out); continue
    run_module("vlm_opd.analysis.token_feedback",
               ["--student", model_id, "--teacher", common.TEACHER_MODEL, "--data-repo", DATA_REPO,
                "--n", N_QUESTIONS, "--out", out, "--seed", SEED, "--batch-size", 8, "--micro-batch", 4],
               f"token_feedback_{name}")
    hub_utils.upload_small_file(out.with_suffix(".stats.json"), RESULT_REPO, token=token)"""),

md("## 4. Tables, bar charts, heatmaps"),

code("""from IPython.display import Image as IPImage, Markdown, display
from vlm_opd.analysis.heatmap import render_class_bars, render_token_heatmap
from vlm_opd.analysis.token_classes import CLASSES, markdown_table

for name in STUDENTS:
    stats = json.loads((OUT_DIR / f"token_feedback_{name}.stats.json").read_text())
    display(Markdown(f"### {name}: {stats['n_questions']} questions, {stats['n_tokens']} tokens, "
                     f"mean KL/token {stats['mean_kl_per_token']:.3f}, rollout accuracy {stats['rollout_accuracy']:.3f}\\n\\n" + markdown_table(stats)))
    bars = render_class_bars(stats, OUT_DIR / f"token_classes_{name}.png", title=f"Teacher feedback by token class ({name})")
    display(IPImage(str(bars)))
    hub_utils.upload_small_file(bars, RESULT_REPO, token=token)"""),

code("""# Heatmaps: the two baseline rollouts with the highest total KL, plus the same questions after OPD
records = {name: [json.loads(l) for l in (OUT_DIR / f"token_feedback_{name}.jsonl").read_text().splitlines()] for name in STUDENTS}
base = sorted(records["baseline"], key=lambda r: -sum(r["kl"]))[:2]
vmax = max(max(r["kl"]) for r in base)
for r in base:
    png = render_token_heatmap(r["pieces"], r["kl"], OUT_DIR / f"heatmap_baseline_{r['id']}.png",
                               title=f"baseline student | {r['id']} | gold={r['gold']} pred={r['pred']} | per-token reverse KL", vmax=vmax)
    display(IPImage(str(png))); hub_utils.upload_small_file(png, RESULT_REPO, token=token)
    match = next((x for x in records.get("opd_q300", []) if x["id"] == r["id"]), None)
    if match:
        png2 = render_token_heatmap(match["pieces"], match["kl"], OUT_DIR / f"heatmap_opd_q300_{r['id']}.png",
                                    title=f"OPD-300 student | {r['id']} | gold={match['gold']} pred={match['pred']} | same colour scale", vmax=vmax)
        display(IPImage(str(png2))); hub_utils.upload_small_file(png2, RESULT_REPO, token=token)"""),

md("## 4b. Digit-position breakdown\n\nNumbers are tokenized digit by digit. This compares the KL on the first token of each number with its later tokens."),

code("""from vlm_opd.analysis.token_classes import digit_position_stats
digit_stats = {}
for name in STUDENTS:
    recs = [json.loads(l) for l in (OUT_DIR / f"token_feedback_{name}.jsonl").read_text().splitlines()]
    d = digit_position_stats(recs); digit_stats[name] = d
    print(f"{name}: {d['numbers']} numbers | first-token KL {d['first_token_kl_mean']:.3f} vs later {d['later_token_kl_mean']:.3f} "
          f"(ratio {d['first_to_later_ratio']:.1f}x) | first tokens are {d['first_token_share']:.0%} of number tokens but carry "
          f"{d['first_token_mass_share']:.0%} of number KL")
(OUT_DIR / "digit_position_stats.json").write_text(json.dumps(digit_stats, indent=2))
hub_utils.upload_small_file(OUT_DIR / "digit_position_stats.json", RESULT_REPO, token=token)
for name in STUDENTS:  # keep the raw per-token records too; they are small
    hub_utils.upload_small_file(OUT_DIR / f"token_feedback_{name}.jsonl", RESULT_REPO, token=token)"""),

md("## 5. Summary"),

code("""summary = {}
for name in STUDENTS:
    s = json.loads((OUT_DIR / f"token_feedback_{name}.stats.json").read_text())
    summary[name] = {"mean_kl_per_token": s["mean_kl_per_token"], "rollout_accuracy": s["rollout_accuracy"], "n_tokens": s["n_tokens"],
                     "digit_first_to_later_ratio": digit_stats[name]["first_to_later_ratio"],
                     "digit_first_token_mass_share": digit_stats[name]["first_token_mass_share"],
                     **{f"{c}_concentration": s["classes"][c]["concentration"] for c in CLASSES},
                     **{f"{c}_kl_mass_share": s["classes"][c]["kl_mass_share"] for c in CLASSES}}
print(json.dumps(summary, indent=2))
(OUT_DIR / "summary_stage4.json").write_text(json.dumps(summary, indent=2))
hub_utils.upload_small_file(OUT_DIR / "summary_stage4.json", RESULT_REPO, token=token)
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
with open("notebooks/04_token_feedback.ipynb", "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("notebook written,", len(cells), "cells")
