# On-Policy Distillation for Vision-Language Reasoning

Transfer on-policy distillation (OPD) from text-only LLMs to vision-language models (VLMs), compare it against supervised distillation (SFT) on ChartQA, and analyze which kinds of tokens receive the strongest token-level teacher feedback.

Student: `Qwen/Qwen3-VL-2B-Instruct` (LoRA rank 64, language model only)
Teacher: `Qwen/Qwen3-VL-8B-Instruct` (fall back to 4B first if GPU memory is tight)

## Why OPD: from behavior cloning to DAgger to OPD

1. **Supervised distillation is behavior cloning.** Fine-tuning the student on teacher outputs only teaches it on states the teacher visited.
2. **Behavior cloning suffers from compounding errors.** Once the student drifts at inference time it enters states never seen in training, errors accumulate, and regret grows quadratically in the sequence length T, O(T²ε). Long chain-of-thought reasoning has a large T, so the problem is most severe there.
3. **DAgger's fix.** Let the student roll out on its own and query the expert for the correct action on the states the student actually reaches. Regret drops to O(Tε). The cost is online expert queries.
4. **A teacher LLM can be queried online for free.** The student generates a sequence, the teacher provides a distribution over every generated token, and we minimize the per-token KL. This is OPD (Agarwal et al. 2024).
5. **Transfer to VLMs.** OPD only acts on the output distribution over text tokens; the image is just part of the prefix, so the method transfers directly. Engineering details that change: image token positions must be masked out of the KL, teacher and student must share one image tensor and one processor pass, and image resolution must be capped to control memory.

## Three questions to answer

1. On multimodal reasoning, is OPD stronger than SFT?
2. Is OPD more data-efficient? (10% / 30% / 100% data-efficiency curve)
3. Does token-level teacher feedback concentrate on chart-reading numeric tokens or on reasoning-step tokens? (Unique to the VLM setting.)

## Running on Colab

Open the latest notebook directly from GitHub (Colab keeps its own copy, so reopen this link after every push that touches the notebook):

- Stage 0: https://colab.research.google.com/github/ffy208/On-Policy-Distillation-for-Vision-Language-Reasoning/blob/main/notebooks/00_setup_and_eval.ipynb
- Stage 1: https://colab.research.google.com/github/ffy208/On-Policy-Distillation-for-Vision-Language-Reasoning/blob/main/notebooks/01_sft.ipynb
- Stage 2: https://colab.research.google.com/github/ffy208/On-Policy-Distillation-for-Vision-Language-Reasoning/blob/main/notebooks/02_opd.ipynb
- Stage 3: https://colab.research.google.com/github/ffy208/On-Policy-Distillation-for-Vision-Language-Reasoning/blob/main/notebooks/03_data_efficiency.ipynb
- Stage 4: https://colab.research.google.com/github/ffy208/On-Policy-Distillation-for-Vision-Language-Reasoning/blob/main/notebooks/04_token_feedback.ipynb

Known Colab environment issue: installing vLLM upgrades torch to a newer CUDA build while the preinstalled torchaudio stays on the old one, and transformers then fails to import any processor. The install cell removes torchaudio for this reason; if you see `PyTorch and TorchAudio were compiled with different CUDA versions`, run `pip uninstall -y torchaudio` and retry.

A second Colab issue: the Pillow directory ends up with a mix of Pillow 11 and 12 files (`ImportError: cannot import name '_Ink' from 'PIL._typing'`), and a plain force-reinstall does not always clear it. Each notebook has an idempotent repair cell: it checks Pillow in a fresh subprocess, and if broken deletes the `PIL` directory, installs Pillow 12 with `PIP_CONSTRAINT` unset, verifies the import, and then restarts the runtime automatically. The restart is required because the Colab kernel imports PIL at startup, so the old modules stay in memory after the files on disk are fixed. After the restart, run the repair cell again (it reports OK and does nothing) and continue.

A third Colab issue: peft checks the installed `torchao` when injecting LoRA and rejects Colab's preinstalled torchao 0.10 (`Found an incompatible version of torchao ... only versions above 0.16.0 are supported`). We never quantize, so the Stage 1 install cell uninstalls torchao; peft then skips the check.

## Repository layout

```
vlm_opd/
  common.py         prompt template, Answer: parsing, relaxed accuracy, data loading (single implementation; other modules import it)
  hub_utils.py      checkpoint push to the HF Hub and resume after disconnect
  prepare.py        ChartQA sampling, image resizing, push to a Hub dataset repo
  evaluate.py       vLLM batch inference + scoring -> json
  generate_teacher.py (Stage 1) sample teacher solutions with vLLM, keep the correct ones, push as SFT data
  collate.py        (Stage 1/2) prompt encoding and supervised batches with prompt/image positions masked
  modeling.py       (Stage 1/2) student loader: frozen base + vision, LoRA on language-model projections only
  sft.py            (Stage 1) LoRA SFT with the transformers Trainer, merge + push to Hub
  opd_utils.py      (Stage 2) rollout inputs, generation masks, per-token KL, vision-batch slicing (pure tensors)
  opd_trainer.py    (Stage 2) on-policy distillation loop: rollout -> teacher/student scoring -> KL -> LoRA update
  analysis/stats.py (Stage 3) bootstrap intervals and paired bootstrap deltas on per-question correctness
  analysis/data_efficiency.py (Stage 3) accuracy-vs-questions table, crossover point, figure, report
  analysis/token_feedback.py (Stage 4) student rollouts scored by both models: per-token reverse KL and log-ratio
  analysis/token_classes.py  (Stage 4) token roles (chart_value / arithmetic / text / answer) and per-class aggregation
  analysis/heatmap.py        (Stage 4) token heatmap and per-class bar chart rendering
notebooks/
  00_setup_and_eval.ipynb    install deps, sample data, zero-shot evaluation
  01_sft.ipynb               teacher generation -> LoRA SFT -> evaluation
  02_opd.ipynb               on-policy distillation with Hub checkpoints and automatic resume
  03_data_efficiency.ipynb   SFT vs OPD at 300 / 900 / 3000 questions; skips finished points, resumes OPD
  04_token_feedback.ipynb    where teacher feedback lands, before and after OPD
  build_*.py                 generate the notebooks above (edit the script, then regenerate)
configs/            yaml configs
docs/resume_log.md  measurable results per stage in XYZ form, with the metrics still to capture
outputs/            metric json files, figures
tests/              offline unit tests
```

## Fixed conventions (shared by every stage)

| Item | Value |
|------|-------|
| Random seed | 42 (passed explicitly to sampling, generation, and training) |
| Train / test size | 3000 sampled from the ChartQA train split, 500 from the test split, human-written questions only (`question_source: human`; the machine-generated questions are templated and leave almost no headroom between student and teacher) |
| Images | longer side ≤ 768 px; processor `min_pixels` / `max_pixels` keep image tokens between 300 and 600 |
| Prompt | `PROMPT_TEMPLATE` in `vlm_opd/common.py`: reason first, last line `Answer: xxx` |
| Scoring | relaxed accuracy: numeric answers within 5% relative error; text answers exact after normalization; a missing `Answer:` line counts as wrong |
| Evaluation decoding | greedy, `max_new_tokens=512` |
| OPD rollout | `do_sample=True`, temperature 1.0, `max_new_tokens=512` |

## Storage

- Model weights never go to Drive; every session downloads them from the Hub into `/content/hf_cache` (`HF_HOME`).
- The HF token lives in Colab Secrets (`HF_TOKEN`) and is read through `common.get_hf_token()`.
- Checkpoints go to a private Hub model repo: LoRA weights + optimizer state only, every 50 steps; `latest.txt` records the latest step, and `hub_utils.load_latest()` resumes after a disconnect.
- The sampled dataset goes to a private Hub dataset repo with the unified schema `{id, image, question, answer}`.

## Stage 1 design notes

- The teacher answers each training question once at temperature 0.7; only solutions whose final answer scores correct are kept (`generate_teacher.py`). Set `--num-samples k` to rejection-sample k candidates per question.
- SFT uses the plain transformers `Trainer` with `SFTCollator` rather than TRL's `SFTTrainer`. The collator encodes the prompt (with the image) through the processor and appends the separately tokenized response plus `<|im_end|>`, so labels are exactly -100 on every prompt and image position. TRL's VLM path changes between releases and its dependencies clash with vLLM; peft + accelerate install next to vLLM, so one Colab environment runs generation, training, and evaluation with no restarts.
- LoRA (rank 64, alpha 128) targets only the language-model projections (`q/k/v/o/gate/up/down_proj` under `language_model`). The vision tower and projector stay frozen; `trainable_summary()` asserts this at startup.
- After training the adapter is merged into the base weights and pushed as a standalone model, so evaluation reuses `evaluate.py` unchanged instead of relying on vLLM's LoRA support for Qwen3-VL.

## Stage 2 design notes

- Rollouts are sampled with `do_sample=True`, temperature 1.0, `top_k=0`, `top_p=1.0`, never greedy: the whole point of on-policy training is covering the states the student actually reaches. Prompts are left-padded so every generation starts at the same index.
- Teacher and student score the identical token sequence and the identical `pixel_values`; the vision encoder runs once per model. Logits are computed only for the generated span via `logits_to_keep`, which keeps the two vocabulary-sized tensors at about 1.2 GB each for batch 8 x 512 tokens.
- The loss is the exact full-vocabulary per-token KL, `reverse` = KL(student || teacher) by default (`--kl-direction forward` for the ablation), averaged over generated tokens up to and including the first `<|im_end|>`. Prompt, image, and post-EOS padding positions are masked out. The KL is computed in fp32 chunks of 64 positions; a batch is processed in micro-batches with gradient accumulation so peak memory stays bounded.
- Every step logs per-token KL, mean and max rollout length, EOS rate, `Answer:` format rate, rollout accuracy against gold, time split into rollout / teacher forward / student forward-backward, peak GPU memory, learning rate, and gradient norm to `opd_log.jsonl`. These are the quantities the resume log needs.
- Every `ckpt_every` steps the LoRA adapter, optimizer, scheduler, data cursor, and epoch go to the Hub with a `latest.txt` pointer; re-running the notebook resumes from there. Data order is a seeded permutation per epoch and rollout sampling is seeded per step, so a resumed run follows the same trajectory.

## Stage 3 design notes

- Budgets are the first 300, 900, and 3000 questions of the sampled train set; SFT and OPD see identical questions. For SFT the budget is applied to the kept teacher solutions by question index (`--max-question-index`), so the 300-question point trains on about 240 solutions.
- Update counts are held fixed across budgets so that data quantity is the only variable: SFT runs 302 optimizer steps at every budget (2 epochs at 3000, more epochs at smaller budgets); OPD runs 150 steps of batch 16 (2400 rollouts) at every budget. Rollout time is latency-bound, which is why batch 16 replaced the batch 8 / 300-step plan after the smoke run.
- All six models are scored on the same 500-question test set. Each point carries a percentile bootstrap interval; OPD minus SFT at each budget uses a paired bootstrap over questions, which is much tighter than comparing two independent intervals. The reported data-efficiency claim is the smallest OPD budget whose interval reaches the full-data SFT accuracy.
- The notebook is restart-safe: finished points are detected from the Hub results repo and skipped, and an interrupted OPD run resumes from its latest checkpoint.

## Stage 4 design notes

- The student samples one solution per test question at temperature 1.0, the same regime as OPD training, so the feedback is measured on states the student actually visits. Teacher and student score the identical sequence; per token we record the full-vocabulary reverse KL (the OPD training signal) and the sampled-token log-ratio log p_student - log p_teacher (positive where the student is over-confident relative to the teacher).
- Token roles are heuristic and documented in `analysis/token_classes.py`: the final `Answer:` line is `answer`; digit and operator tokens on lines with an arithmetic cue (an operator or a word such as sum, average, difference) are `arithmetic`; digit tokens on other lines are `chart_value` (values read off the chart); everything else is `text`. Operands and results on arithmetic lines are not separated.
- The statistic reported per class is concentration = share of total KL mass / share of tokens. A class above 1 receives more teacher feedback than its length alone would predict.
- The same analysis is run on the OPD-300 student to show what feedback remains after training; heatmaps for the same questions use the same colour scale.

## Verified external assumptions (2026-09-07)

- `HuggingFaceM4/ChartQA` has fields `image / query / label(list[str]) / human_or_machine`, with 28299 train and 2500 test rows. `prepare.py` uses `label[0]` as the answer.
- The Qwen3-VL processor is `Qwen2VLImageProcessorFast` with patch 16 and merge 2, so one visual token covers 32x32 pixels. Both transformers and vLLM accept `min_pixels` / `max_pixels` (mapped internally to `size.shortest_edge` / `size.longest_edge`), so `common.image_pixel_bounds()` serves both `AutoProcessor.from_pretrained` and vLLM's `mm_processor_kwargs`.
- In the evaluation notebook each model runs in its own subprocess, so GPU memory is released when the process exits. This avoids loading the teacher while the student engine is still resident in the same kernel.

## Local development

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
# hub_utils tests need torch (CPU build is enough)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
pytest
```

Evaluation (vLLM) and training (TRL + PEFT) dependencies are installed in separate notebooks. Locally only the GPU-free unit tests run.

## Results

### Stage 0 smoke test (2026-09-07, Tesla T4, 50 test rows, random human+machine mix)

| Model | Accuracy | Format rate |
|-------|----------|-------------|
| Student Qwen3-VL-2B zero-shot | 0.84 | 1.00 |
| Teacher Qwen3-VL-4B zero-shot (8B does not fit on a T4) | 0.90 | 0.98 |

The prompt format works, but the student-teacher gap on the random mix is too small to separate OPD from SFT within the noise of a 500-row test set. The main experiments therefore use human-written questions only.

### Stage 0 baseline (2026-09-07, A100 40 GB, 500 human-written test questions, seed 42)

| Model | Accuracy | Format rate |
|-------|----------|-------------|
| Student Qwen3-VL-2B zero-shot (baseline 0) | 0.668 | 0.960 |
| Teacher Qwen3-VL-8B zero-shot (teacher upper bound) | 0.844 | 0.992 |
| SFT student, smoke run (82 solutions, 20 steps) | 0.780 | 0.982 |
| SFT student, full run (2402 solutions, 302 steps, 2 epochs) | 0.834 | 0.988 |
| OPD student, smoke run (100 questions, 20 steps, batch 8, reverse KL) | 0.792 | 0.966 |

### Stage 3 data-efficiency curve (2026-09-08, A100, 500 human-written test questions, seed 42)

| Training questions | SFT (302 steps) | OPD (150 steps x batch 16) | OPD - SFT, paired 95% CI |
|---|---|---|---|
| 300 (10%) | 0.778 [0.742, 0.814] | 0.836 [0.804, 0.868] | +0.058 [+0.028, +0.088] |
| 900 (30%) | 0.808 [0.774, 0.842] | 0.820 [0.786, 0.854] | +0.012 [-0.016, +0.040] |
| 3000 (100%) | 0.834 [0.802, 0.866] | 0.824 [0.790, 0.858] | -0.010 [-0.036, +0.016] |
| 0 (zero-shot) | 0.668 [0.626, 0.710] | | |
| teacher 8B | 0.844 [0.812, 0.876] | | |

OPD with 300 questions (10% of the data) matches full-data SFT and sits within noise of the teacher. At 900 and 3000 questions both methods saturate at the teacher ceiling and the paired differences are within noise. Caveats: single seed; the 3000-question OPD run covers each question less than once (2400 rollouts), so it is undertrained relative to SFT's 2 epochs.

Student-teacher gap: 17.6 points. Evaluation of 500 rows takes 12 s (2B) and 36 s (8B) with vLLM on the A100.

### Stage 4 token-level feedback (2026-09-09, 100 test questions, rollouts sampled at temperature 1.0)

| Token class | Zero-shot student: tokens -> KL mass (concentration) | OPD-300 student: concentration |
|---|---|---|
| answer | 7.3% -> 19.5% (2.67x) | 0.38x |
| text | 74.8% -> 73.9% (0.99x) | 1.24x |
| chart_value | 7.4% -> 2.9% (0.39x) | 0.73x |
| arithmetic | 10.5% -> 3.7% (0.36x) | 0.42x |
| mean KL per token | 0.489 | 0.235 |

Before training, the teacher's feedback concentrates on the final answer and on discourse/format tokens, not on chart-reading digits; numbers are tokenized digit by digit and the disagreement sits on the first digit of a number, which per-token averaging dilutes. After OPD the answer-line feedback almost vanishes and the residual feedback shifts toward chart values, i.e. what remains to learn is perception rather than answer selection or format.

## Stage progress

- [x] Stage 0: repository layout, `common.py`, `hub_utils.py`, `prepare.py`, `evaluate.py`, `00_setup_and_eval.ipynb`
- [x] Stage 1: SFT baseline 0.834 (teacher 0.844, zero-shot 0.668); 80.1% teacher acceptance over 3000 questions
- [x] Stage 2 code smoke-tested on the A100: 33 s/step at batch 8, peak 27 GB, KL 0.42 -> 0.26 in 20 steps, OPD smoke student 0.792
- [x] Stage 3: OPD matches full-data SFT with 10% of the questions (+5.8 points over SFT at 300 questions, paired CI excludes 0)
- [x] Stage 4: teacher feedback concentrates on the answer line (2.67x) before OPD; after OPD-300 it drops to 0.38x and mean KL halves
- [ ] Stage 5 (optional): self-distillation (SDPO)
