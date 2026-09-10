# Running the experiments on PACE (Slurm)

Everything the Colab notebooks do is also a plain `python -m vlm_opd.*` command, so the study runs
on a cluster without notebooks. One Slurm job = one experiment point (task, method, budget, seed).
Results go to the private Hub results repo; a resubmitted job skips finished work and resumes OPD
from its latest Hub checkpoint.

## One-time setup

```bash
git clone https://github.com/ffy208/On-Policy-Distillation-for-Vision-Language-Reasoning.git
cd On-Policy-Distillation-for-Vision-Language-Reasoning
bash slurm/setup_env.sh                 # uv venv + vllm/transformers/peft; HF_HOME on scratch (works from any clone location)
echo 'hf_...' > ~/.hf_token && chmod 600 ~/.hf_token
```

The scripts load `python/3.12.5` and `cuda/12.6.1` and do not set a partition: Slurm uses the account's
default (`coc-gpu` here; the `ice-gpu` H100/H200 nodes are not open to this account and jobs sent there sit
in `ReqNodeNotAvail`). `coc-gpu` offers 8 A100 and 32 L40S (48 GB). The default GRES is `gpu:l40s:1` (the larger pool);
override with `--gres=gpu:a100:1` on the sbatch line or `GPU=a100 slurm/sweep.sh ...`. The `ice-bw-gpu`
RTX PRO 6000 Blackwell nodes (97 GB per card) are also open to the account and are reached with
`--gres=gpu:rtx_pro_6000_blackwell:1`; they fit a 32B teacher on one card. GRES type names are
lowercase and memory is requested per GPU (`--mem-per-gpu`), as PACE expects. Always `sbatch` from the repository root: the job uses `SLURM_SUBMIT_DIR` to find the code.
Set `SCRATCH` and `HF_HOME` in `~/.bashrc` so models download to scratch, not to the home quota:

```bash
export SCRATCH=/storage/ice1/<path>/<user>
export HF_HOME=$SCRATCH/hf_cache
```

## Smoke test (about 15 minutes on one A100)

Measured 2026-09-09 on one L40S: 9 min 40 s wall time end to end (environment, 5 OPD steps on 100 questions,
merge, vLLM evaluation of 500 questions in 20 s); accuracy 0.754 after 5 steps versus 0.668 zero-shot.
Resubmitting the same point afterwards exits in 7 s because the result is already on the Hub.

```bash
sbatch slurm/smoke.sbatch          # 5 OPD steps on 100 questions, seed 99, 45-minute walltime
tail -f logs/vlmopd-smoke-<jobid>.out
```

Walltimes are deliberately short (smoke 45 min, OPD points 2.5 h, SFT points 1 h): on a busy
partition short jobs backfill into gaps that long jobs cannot use. A pending reason of
`ReqNodeNotAvail, May be reserved for other job` with free `mixed` nodes usually means exactly that.

A seed other than 42 produces fresh Hub repo names (`ffyang/vlm_opd_opd_chartqa_human_q100_s99_*`) and
does not touch the notebook-era results. Delete those Hub repos afterwards or keep them as a record.

## The first real block (go/no-go for the headline)

```bash
slurm/sweep.sh chartqa_human "100" "42" "sft opd"          # the 100-question point
slurm/sweep.sh chartqa_human "100 300" "1 2 3" "sft opd"   # three seeds at 100 and 300
```

Twelve OPD jobs of about 1.6 h and twelve SFT jobs of about 20 min; with a few A100s in parallel
this is an afternoon. Then, on any machine with the token:

```bash
python scripts/fetch_results.py            # pulls every result json and figure from the Hub
```

Then aggregate every seed found (notebook-era seed-42 files included):

```bash
python -m vlm_opd.analysis.data_efficiency --task chartqa_human --budgets 100 300 900 3000
```

This writes `outputs/data_efficiency_chartqa_human.{json,md,png}` with per-seed accuracies, the
across-seed mean and standard deviation, a bootstrap interval pooled over questions of all seeds,
and the OPD minus SFT paired bootstrap computed over the seeds both methods share.

## Bad GPUs

One L40S on `atl1-1-03-004-23-0` raised `cudaErrorECCUncorrectable` at device setup and killed four jobs
across two submissions (2026-09-10). Every job script now touches the GPU first; if that fails, the job adds
its node to its own `ExcNodeList`, requeues itself (`--requeue`), and exits 0, so the point simply starts
again elsewhere (OPD resumes from its Hub checkpoint). To exclude a node by hand for a whole sweep:
`export SBATCH_EXCLUDE=<node>` before `slurm/sweep.sh`. Report the node to pace-support@oit.gatech.edu.

## What goes to the Hub

A free Hub account has 100 GB of private storage, and the notebook-era policy (push every merged model, keep
every checkpoint step) hit that cap after about twenty points: every later push failed with HTTP 400 and the
jobs died after training. The runner now uses `artifacts: {merged: local}` in `configs/tasks.yaml`:

- merged models (base + LoRA, 4.3 GB) are written to `ckpt/<point>/merged` on the cluster disk and evaluated
  from there; nothing that large is pushed;
- SFT points push only their LoRA adapter (`..._lora`, 0.3 GB); OPD points keep their adapter in the
  checkpoint repo, which is pruned to the latest step after every push (0.8 GB per run);
- results (small json) go to the results repo as before.

If a merged model is missing locally when an evaluation is still needed, the runner re-invokes the trainer:
OPD resumes from its final checkpoint and only merges, SFT retrains (12 minutes). `python scripts/hub_storage.py`
reports storage per repo; `--prune-ckpts`, `--squash`, and `--delete-matching smoke` reclaim space (dry run without
`--yes`). The quota counts every LFS file in a repo's git history, so deleting files or folders frees nothing
until the history is squashed; the runner squashes a checkpoint repo after each prune. Released storage shows
up on the billing page with a delay: after deleting and squashing, the page read 47.6 GB while the files in the
remaining repos summed to 9.9 GB (2026-09-10), a constant 38 GB gap that only the Hub can clear.

## Adding a task or an OOD test set

`vlm_opd/prepare.py` has one adapter per source (`SOURCES`): `chartqa`, `geometry3k` (train + test),
`charxiv` and `chartqapro` (test only, chart questions with numeric answers, used as out-of-distribution
evaluations for the ChartQA-trained models). Each adapter maps the raw fields onto `{id, image, question, answer}`
and drops rows the relaxed-accuracy scorer cannot judge. One job builds a source end to end:

```bash
sbatch --export=ALL,SOURCE=geometry3k slurm/prepare_data.sbatch   # dataset + verified teacher solutions (SFT data)
sbatch --export=ALL,SOURCE=charxiv    slurm/prepare_data.sbatch   # OOD test set
sbatch --export=ALL,SOURCE=chartqapro slurm/prepare_data.sbatch
```

Geometry3K answers are expressions such as `2 \sqrt { 5 }` or `\frac { 26 } { 3 }`; the scorer evaluates them
to a number (no exponents allowed, so the evaluator cannot be made to hang) and applies the same 5 percent
tolerance. Its task entry in `configs/tasks.yaml` sets `prompt_style: geometry`, which every command
(SFT, OPD, evaluation, teacher generation) receives as `--prompt-style`; the two wordings differ only in
the task description, the `Answer:` format instruction is shared.

For a new source: add a converter and a `SOURCES` entry in `prepare.py`, a task entry (or an `ood_eval`
list item) in `configs/tasks.yaml`, and, for a training task, generate the SFT data with
`python -m vlm_opd.generate_teacher --prompt-style <style>`. Every point then also produces
`eval_<point>_on_<set>.json` for each OOD set of its task.

## Job sizing

Measured on an A100 40 GB with the defaults in `configs/tasks.yaml`: OPD 150 steps x batch 16
takes about 90 min at 27 GB peak; SFT 302 steps takes 12 min; each evaluation about 3 min of vLLM
startup plus under a minute of decoding. The 8B teacher download (16 GB) happens once into
`HF_HOME`.
