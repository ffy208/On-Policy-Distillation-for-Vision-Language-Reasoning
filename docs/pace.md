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
override with `--gres=gpu:a100:1` on the sbatch line or `GPU=a100 slurm/sweep.sh ...`. GRES type names are
lowercase and memory is requested per GPU (`--mem-per-gpu`), as PACE expects. Always `sbatch` from the repository root: the job uses `SLURM_SUBMIT_DIR` to find the code.
Set `SCRATCH` and `HF_HOME` in `~/.bashrc` so models download to scratch, not to the home quota:

```bash
export SCRATCH=/storage/ice1/<path>/<user>
export HF_HOME=$SCRATCH/hf_cache
```

## Smoke test (about 15 minutes on one A100)

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
