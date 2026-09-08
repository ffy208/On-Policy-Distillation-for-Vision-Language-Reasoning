"""Generate notebooks/03_data_efficiency.ipynb (edit this script, then run it from the repository root)."""

import json


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


cells = [
md("""# 03 · Data-efficiency curve: SFT vs OPD at 300 / 900 / 3000 questions

Design (see `configs/data_efficiency.yaml`):

- Both methods train on the first N questions of the same sampled train set, N in {300, 900, 3000}.
- SFT uses the same number of optimizer steps at every budget (302, which is 2 epochs at 3000); smaller budgets simply
  see more epochs. OPD uses 150 steps of batch 16 at every budget. With update counts fixed, data quantity is the only variable.
- Every model is evaluated on the same 500-question test set; OPD vs SFT differences use a paired bootstrap.

Total GPU time is roughly 6 hours on an A100. Sessions will end before that: **just re-run the whole notebook**. Every
completed point is skipped (its result json is on the Hub), and an interrupted OPD run resumes from its latest checkpoint.
The 3000-question SFT point reuses the Stage 1 full run."""),

code("""# ==================== Config ====================
REPO_URL  = "https://github.com/ffy208/On-Policy-Distillation-for-Vision-Language-Reasoning.git"
REPO_DIR  = "/content/vlm_opd_repo"
QUESTION_SOURCE = "human"
DATA_REPO     = f"ffyang/vlm_opd_chartqa_{QUESTION_SOURCE}"
SFT_DATA_REPO = f"ffyang/vlm_opd_sft_{QUESTION_SOURCE}"
RESULT_REPO   = "ffyang/vlm_opd_results"

BUDGETS   = [300, 900, 3000]
SFT_STEPS = 302
OPD_STEPS, OPD_BATCH, OPD_MICRO, OPD_CKPT_EVERY = 150, 16, 4, 25
SEED = 42

import os
os.environ["HF_HOME"] = "/content/hf_cache"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)"""),

md("## 1. Install dependencies"),

code("""!pip install -q -U "vllm>=0.8" "transformers>=4.57" "datasets>=3.0" "huggingface_hub>=0.26" "peft>=0.13" "accelerate>=1.0" qwen-vl-utils pyyaml matplotlib > /content/pip_install.log 2>&1; echo "pip exit code: $?"; tail -n 15 /content/pip_install.log
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
assert GPU_GB >= 30 and torch.cuda.is_bf16_supported(), "Stage 3 needs an A100 40 GB\""""),

md("## 2. Clone the repository, log in, helpers"),

code("""import os, sys
if not os.path.isdir(REPO_DIR):
    !git clone {REPO_URL} {REPO_DIR}
else:
    !cd {REPO_DIR} && git pull
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)
from vlm_opd import common, hub_utils
from huggingface_hub import login, hf_hub_download
login(token=common.get_hf_token(), add_to_git_credential=False)
token = common.get_hf_token(); os.environ["HF_TOKEN"] = token"""),

code("""import json, shutil, subprocess, sys
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
        raise RuntimeError(f"{module} failed (exit {rc}); log: {log_path}\\n--- log tail ---\\n{tail}")

def fetch_result(name: str) -> Path | None:
    # Local copy first, then the Hub results repo; None when the point has not been produced yet.
    local = OUT_DIR / name
    if local.exists():
        return local
    try:
        got = hf_hub_download(RESULT_REPO, f"outputs/{name}", repo_type="model", token=token)
        shutil.copy(got, local); return local
    except Exception:
        return None

def result_name(method: str, budget: int) -> str:
    return f"eval_{method}_q{budget}.json"

def evaluate_and_upload(model_repo: str, method: str, budget: int) -> dict:
    out = OUT_DIR / result_name(method, budget)
    run_module("vlm_opd.evaluate", ["--model", model_repo, "--data-repo", DATA_REPO, "--split", "test", "--out", out,
                                     "--tag", f"{method}_q{budget}", "--seed", SEED, "--gpu-mem", 0.85], f"eval_{method}_q{budget}")
    hub_utils.upload_small_file(out, RESULT_REPO, token=token)
    res = json.loads(out.read_text())
    print(f"[{method} q{budget}] acc = {res['accuracy']:.4f}  format = {res['format_rate']:.4f}")
    return res"""),

md("## 3. Reuse the Stage 1 full SFT run as the 3000-question SFT point"),

code("""if fetch_result(result_name("sft", 3000)) is None:
    src = fetch_result("eval_sft_student.json")
    assert src is not None, "Stage 1 full-run result eval_sft_student.json not found on the Hub; run notebook 01 with SMOKE=False first"
    shutil.copy(src, OUT_DIR / result_name("sft", 3000))
    hub_utils.upload_small_file(OUT_DIR / result_name("sft", 3000), RESULT_REPO, token=token)
    print("copied Stage 1 SFT result as the q3000 SFT point")
else:
    print("q3000 SFT point already present")"""),

md("## 4. Train and evaluate every missing point\n\nCheapest first. Re-running this cell after a disconnect continues where it stopped."),

code("""def sft_point(budget: int) -> None:
    merged = f"ffyang/vlm_opd_sft_{QUESTION_SOURCE}_q{budget}_merged"
    run_module("vlm_opd.sft", ["--data-repo", SFT_DATA_REPO, "--out-dir", f"ckpt/sft_q{budget}",
                               "--max-question-index", budget, "--max-steps", SFT_STEPS,
                               "--lr", 1e-4, "--batch", 4, "--grad-accum", 4, "--lora-r", 64, "--seed", SEED,
                               "--push-merged-repo", merged], f"train_sft_q{budget}")
    evaluate_and_upload(merged, "sft", budget)

def opd_point(budget: int) -> None:
    ckpt = f"ffyang/vlm_opd_opd_{QUESTION_SOURCE}_q{budget}_ckpt"
    merged = f"ffyang/vlm_opd_opd_{QUESTION_SOURCE}_q{budget}_merged"
    run_module("vlm_opd.opd_trainer", ["--data-repo", DATA_REPO, "--out-dir", f"ckpt/opd_q{budget}",
                                       "--limit", budget, "--total-steps", OPD_STEPS, "--batch-size", OPD_BATCH,
                                       "--micro-batch", OPD_MICRO, "--ckpt-every", OPD_CKPT_EVERY, "--lr", 5e-5,
                                       "--kl-direction", "reverse", "--seed", SEED,
                                       "--ckpt-repo", ckpt, "--merged-repo", merged], f"train_opd_q{budget}")
    log = Path(f"ckpt/opd_q{budget}/opd_log.jsonl")
    if log.exists():
        hub_utils.upload_small_file(log, RESULT_REPO, path_in_repo=f"outputs/opd_log_q{budget}.jsonl", token=token)
    evaluate_and_upload(merged, "opd", budget)

PLAN = [(300, "sft"), (300, "opd"), (900, "sft"), (900, "opd"), (3000, "sft"), (3000, "opd")]
for budget, method in PLAN:
    if fetch_result(result_name(method, budget)) is not None:
        print(f"skip {method} q{budget}: result already exists"); continue
    print(f"\\n===== {method.upper()} with {budget} questions =====")
    (sft_point if method == "sft" else opd_point)(budget)"""),

md("## 5. Curve, table, paired comparison"),

code("""from vlm_opd.analysis.data_efficiency import collect, crossover, write_report
from IPython.display import Image as IPImage, Markdown, display

baseline = fetch_result("eval_student_zeroshot.json")
teacher = fetch_result("eval_teacher_zeroshot.json")
table = collect(OUT_DIR, BUDGETS, baseline_json=baseline, teacher_json=teacher, n_boot=10000)
report_md = write_report(table, OUT_DIR)
display(Markdown(report_md.read_text()))
display(IPImage(str(OUT_DIR / "data_efficiency.png")))
print("crossover:", json.dumps(crossover(table), indent=2))
for name in ("data_efficiency.json", "data_efficiency.md", "data_efficiency.png"):
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
with open("notebooks/03_data_efficiency.ipynb", "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("notebook written,", len(cells), "cells")
