# Resume Log: On-Policy Distillation for Vision-Language Reasoning

Purpose: this project will appear on a resume as evidence of hands-on experience with on-policy
distillation (OPD) for vision-language models. Every entry below follows the XYZ format,
"Accomplished [X] as measured by [Y], by doing [Z]", and every Y is a number that can be traced to
a json file on the Hub (`ffyang/vlm_opd_results`), a commit, or a test run. Keep this file updated
at the end of each stage; add candidate resume bullets under "Resume bullets" once a stage's numbers
are final.

Hardware: Google Colab, NVIDIA A100-SXM4-40GB. Models: student Qwen3-VL-2B-Instruct, teacher
Qwen3-VL-8B-Instruct. Task: ChartQA, human-written questions, 3000 train / 500 test, seed 42.

## Completed entries

### Stage 0 (2026-09-07): infrastructure and baselines

- Accomplished a reproducible evaluation pipeline for Qwen3-VL on ChartQA, as measured by a 500-question evaluation completing in 12.2 s for the 2B student and 36.2 s for the 8B teacher (versus the 3 to 5 minutes budgeted in the project plan), by building a vLLM batch-inference module with a shared relaxed-accuracy scorer and running each model in an isolated subprocess.
- Accomplished a task setup with meaningful distillation headroom, as measured by widening the student-teacher accuracy gap from 6 points (0.84 vs 0.90, 50-question smoke test on a random ChartQA mix) to 17.6 points (0.668 vs 0.844, 500 human-written questions), by adding a question-source filter to the data pipeline and re-sampling only human-written ChartQA questions.
- Accomplished baseline measurements for the whole study, as measured by student zero-shot accuracy 0.668 (format compliance 0.960) and teacher zero-shot accuracy 0.844 (format compliance 0.992) on the fixed 500-question test set, by designing a single reasoning prompt with a fixed `Answer:` line and a parser that counts missing answers as errors.
- Accomplished a Hub-only storage design that survives Colab disconnects, as measured by zero data loss across three session terminations during Stage 0, by pushing the sampled dataset, all evaluation json files, and (for training) LoRA checkpoints with a `latest.txt` pointer to private HuggingFace repos instead of Google Drive.
- Accomplished a tested shared code base for the study, as measured by 32 passing unit tests (answer parsing, relaxed accuracy, deterministic sampling, checkpoint save/restore, prompt masking with the real Qwen3-VL processor, LoRA targeting, and a forward/backward pass on a tiny Qwen3-VL), by putting every piece of evaluation/training-shared logic in one module and covering it offline.
- Accomplished a stable Colab environment for vLLM plus training, as measured by resolving three environment failures (torchaudio CUDA 12.8 vs torch CUDA 13.0 import crash, a mixed-version Pillow install, and an 8B-teacher OOM on a 15 GB T4) so that the pipeline ran end to end on the A100, by adding install-time repairs, an import sanity cell, and GPU-memory-aware teacher selection to the notebook.

### Stage 1 (completed 2026-09-08)

- Accomplished an end-to-end supervised-distillation pipeline for a VLM (teacher sampling -> filtering -> LoRA SFT -> merge -> vLLM evaluation), as measured by a 100-question smoke run in which the 8B teacher's solutions were kept at 82% acceptance (82/100), a rank-64 LoRA with 69.7M trainable parameters (3.17% of 2.2B) trained 20 optimizer steps in 47 s (2.3 s/step, 6.9 samples/s, effective batch 16) with training loss falling from 0.323 to 0.137, and the merged weights loading in vLLM without modification, by sampling teacher solutions with vLLM, filtering by answer correctness, fine-tuning only language-model projections with explicit masking of prompt and image tokens, and merging the adapter into the base weights.
- Accomplished a first supervised-distillation gain on the fixed 500-question test set, as measured by student accuracy rising from 0.668 to 0.780 (+11.2 points) and format compliance from 0.960 to 0.982 after only 82 teacher solutions and 20 optimizer steps (smoke configuration, 2026-09-08), by fine-tuning the rank-64 LoRA on answer-verified teacher reasoning traces.
- Accomplished a supervised-distillation baseline within 1 point of the teacher, as measured by student accuracy 0.834 (format compliance 0.988) on the 500-question test set versus 0.668 zero-shot (+16.6 points) and 0.844 for the 8B teacher (full run, 2026-09-08), by keeping 2402 of 3000 teacher solutions (80.1% acceptance) and training the rank-64 LoRA for 302 optimizer steps (2 epochs, effective batch 16, final training loss 0.191).
- Design consequence recorded 2026-09-08: with SFT at 0.834 and the teacher at 0.844, full-data OPD-vs-SFT differences fall inside the ~3 to 4 point confidence interval of a 500-question test set. The data-efficiency curve (300 / 900 / 3000 questions) therefore becomes the primary comparison, with an optional out-of-distribution evaluation as a secondary axis.
- Accomplished a Colab environment that runs vLLM inference and peft training in one runtime with no restarts, as measured by four resolved preinstalled-package conflicts (torchaudio CUDA mismatch, mixed Pillow 11/12 files, torchao 0.10 rejected by peft, and an 8B-teacher OOM on a 15 GB T4) plus one transformers 5 API change (`warmup_ratio` removed), by adding install-time repairs, import-time assertions, GPU-memory-aware model selection, and version-adaptive TrainingArguments construction.

### Stage 2 and 3 (completed 2026-09-08)

- Accomplished a working on-policy distillation loop for a VLM on a single 40 GB GPU, as measured by a 20-step smoke run (batch 8 rollouts, 100 questions, reverse KL) with per-token KL falling from 0.421 to 0.262, peak GPU memory 26.97 GB with the 8B teacher and 2B student co-resident, 33.0 s per step on average (rollout 17 to 46 s, teacher forward about 0.5 s, student forward-backward about 1.8 s), checkpoints pushed to the Hub at steps 10 and 20, and the merged student scoring 0.792 accuracy (format 0.966) on the 500-question test set versus 0.668 zero-shot, by sampling student rollouts at temperature 1.0, scoring only the generated span with `logits_to_keep`, computing the full-vocabulary KL in fp32 chunks, and updating a rank-64 LoRA.
- Observation for the length-collapse question: mean rollout length grew from 121 to 215 tokens over 20 steps (teacher responses are longer than the student's), so the risk in this setup is lengthening toward the 512-token cap rather than collapse; EOS rate stayed at 0.75 to 1.00 per batch.
- Accomplished disconnect-safe OPD training on Colab, as measured by the 300-question run resuming automatically from the Hub checkpoint at step 75 after a session loss (adapter, optimizer, scheduler, and data cursor restored) and completing steps 76 to 150 with no repeated or skipped batches, by checkpointing every 25 steps with a `latest.txt` pointer and a seeded data permutation.
- Measured OPD throughput at batch 16 (2026-09-08, 300-question run): 35 s per step on average (range 18 to 53 s; rollout 15 to 47 s, teacher forward 1.0 s, student forward-backward 3.5 s), peak memory 27.08 GB (unchanged from batch 8), so doubling the batch doubled training tokens per second at no memory cost. Late-run per-token KL 0.10 to 0.28 (mean about 0.17), mean rollout length 80 to 186 tokens (no drift toward the 512 cap), EOS and format rates at 1.00.
- Accomplished a 10x data-efficiency result for on-policy distillation on a VLM, as measured by OPD trained on 300 questions (10% of the data) reaching 0.836 accuracy [95% CI 0.804, 0.868] on the 500-question test set, matching full-data SFT on 3000 questions (0.834 [0.802, 0.866]) and sitting within noise of the 8B teacher (0.844), by sampling 2400 student rollouts over the 300 questions and minimizing the per-token reverse KL to the teacher.
- Accomplished a statistically significant low-data win for OPD over supervised distillation, as measured by a paired-bootstrap difference of +5.8 points [+2.8, +8.8] at 300 questions (OPD 0.836 vs SFT 0.778, same questions, same number of optimizer updates), by holding update counts fixed across methods so data quantity is the only variable.
- Documented the saturation regime honestly: at 900 questions the paired difference is +1.2 [-1.6, +4.0] and at 3000 questions -1.0 [-3.6, +1.6], both within noise, because both methods converge to the teacher ceiling of 0.844 with a 500-question test set. Caveats recorded: single seed; the 3000-question OPD run covers each question less than once (2400 rollouts), so it is undertrained relative to SFT's 2 epochs.

### Stage 4 (completed 2026-09-09)

- Accomplished a token-level account of where on-policy teacher feedback lands on a VLM, as measured over 9456 generated tokens from 100 test-question rollouts of the zero-shot student: the final-answer line holds 7.3% of tokens but 19.5% of the teacher's KL mass (concentration 2.67x), reasoning text is proportional (0.99x), and chart-reading numbers and arithmetic tokens receive less than their share (0.39x and 0.36x), by scoring each student rollout with both models and aggregating the full-vocabulary reverse KL by heuristic token role.
- Accomplished a before/after comparison that shows what OPD changes, as measured by the OPD-300 student's mean per-token KL falling from 0.489 to 0.235 (11956 tokens), answer-line feedback collapsing from 19.5% to 2.2% of KL mass (2.67x to 0.38x), and the residual feedback shifting toward chart values (0.39x to 0.73x), by re-running the same analysis on the trained student with identical questions and colour scale.
- Method observation recorded: numbers are tokenized into single digits and the teacher-student disagreement on a number sits almost entirely on its first digit (visible in the heatmaps), so per-token averaging dilutes number-level feedback; a digit-position breakdown is reported alongside the class table. Caveat: token roles are heuristic; the `text` class mixes structural and reasoning tokens.

### Cluster port (2026-09-09, PACE ICE)

- Accomplished a notebook-free port of the whole study to a Slurm cluster, as measured by one Slurm job running environment setup, 5 OPD steps on 100 questions, LoRA merge, Hub push, and a 500-question vLLM evaluation end to end in 9 min 40 s on a single L40S (accuracy 0.754 after 5 steps versus 0.668 zero-shot), by turning each stage into a `python -m vlm_opd.*` command driven by a task registry (`configs/tasks.yaml`) and an idempotent experiment runner that names every artifact by (task, method, budget, seed).
- Accomplished restart safety at the job level, as measured by a resubmitted job for a finished point exiting in 7 s after checking the Hub results repo instead of retraining, by checking result presence on the Hub before every training or evaluation command.
- Accomplished a second task and two out-of-distribution test sets without touching the training code, as measured by 63 passing tests covering four dataset converters (ChartQA, Geometry3K, CharXiv, ChartQAPro), two prompt styles, and a math-aware scorer that evaluates radical, fraction, and pi expressions with the same 5 percent tolerance, by adding a per-source converter registry to the data preparation module and threading a prompt style through every command from the task registry.

### Multi-seed replication on PACE (2026-09-10)

- Accomplished a seed-robust version of the low-data result, as measured by three seeds at 100 and 300 training questions (18 new training runs on L40S GPUs) giving OPD 0.810 (sd 0.010) versus SFT 0.793 (sd 0.022) at 100 questions and OPD 0.827 (sd 0.008) versus SFT 0.799 (sd 0.015) at 300, with paired bootstrap deltas of +2.5 [+0.7, +4.2] and +2.9 [+1.3, +4.6] points over the shared seeds, by running the sweep through the idempotent experiment runner and aggregating with `python -m vlm_opd.analysis.data_efficiency` (intervals pooled over questions across seeds, deltas paired by seed and question).
- Recorded the correction honestly: the original single-seed +5.8 at 300 questions was the most favourable of four SFT seeds (0.778 against 0.800 to 0.814 for the others); the multi-seed estimate is about half that but still excludes zero, and OPD's across-seed spread is half of SFT's. OPD on 100 questions (0.810) matches SFT on 900 (0.808), so the roughly 10x data-efficiency statement now rests on two budgets and three seeds.
- Accomplished a storage policy fix after a silent failure mode, as measured by 14 parallel jobs all dying at their first Hub push once private storage reached 99.5 GB of the 100 GB free quota (HTTP 400), then completing after the change, by keeping 4.3 GB merged models on the cluster disk, pushing only 0.3 GB LoRA adapters, pruning checkpoint repos to their latest step, and adding a storage report/reclaim script (99.5 GB to 59.7 GB).
- Diagnosed one unrelated hardware failure: two jobs on the same node failed at device setup with `cudaErrorECCUncorrectable`; excluded the node via `SBATCH_EXCLUDE` and reported it.

## Metrics to capture in later stages

Fill each of these with the exact number and the json file it comes from.

Stage 1, SFT baseline
- Teacher acceptance rate (kept / sampled) and mean response length.
- SFT student accuracy and format rate on the 500-question test set; delta versus the 0.668 baseline.
- Training cost: optimizer steps, wall-clock minutes, effective batch size, trainable parameter count and share.

Stage 2, OPD
- OPD student accuracy and format rate; delta versus baseline 0 and versus SFT.
- Per-step cost: seconds per step split into rollout, teacher forward, student backward; peak GPU memory.
- Per-token KL at start and end of training; mean rollout length over training (length collapse check).
- Number of steps and number of distinct training questions seen.

Stage 3, data efficiency
- Accuracy of SFT and OPD at 300, 900, 3000 training questions; the smallest OPD data size that matches SFT at 3000.

Stage 4, token-level feedback analysis
- Share of total teacher-student log-probability-ratio mass on each token class (chart-reading numeric, arithmetic, connective, final answer).
- Number of test questions and tokens analyzed.

Stage 5 (optional), self-distillation
- Accuracy, mean output length over training, and whether length collapse occurs (length at step 0 vs final).

## Resume bullets (final, 2026-09-09)

Header line:

**On-Policy Distillation for Vision-Language Reasoning** — Independent research, Sept 2026. Python, PyTorch, transformers, PEFT/LoRA, vLLM, HuggingFace Hub. Code: https://github.com/ffy208/On-Policy-Distillation-for-Vision-Language-Reasoning · Page: https://ffy208.github.io/On-Policy-Distillation-for-Vision-Language-Reasoning/

Research-facing bullets:

- Built an on-policy distillation pipeline for a vision-language model (Qwen3-VL 2B student, 8B teacher) on ChartQA, training a LoRA student on full-vocabulary per-token KL to the teacher over the student's own sampled rollouts, with Hub-checkpointed, disconnect-safe training on a single 40 GB A100.
- Raised student accuracy from 0.668 to 0.827 (3 seeds) on 500 held-out human-written questions using only 300 training questions, matching supervised distillation trained on 3,000 (0.834) and beating it by 2.9 points at equal data (paired bootstrap 95% CI +1.3 to +4.6, three seeds); OPD on 100 questions matches SFT on 900.
- Showed with a token-level analysis of 21K generated tokens that the teacher's feedback concentrates on the final-answer line (2.7x its token share) rather than chart-reading digits (0.4x), and that OPD closes that gap (0.4x) while halving mean per-token KL, independently corroborating the motivation of 2026 visual re-weighting methods.

Engineering-facing alternative for the third bullet:

- Cut evaluation to 12 s per 500 questions with vLLM (versus a planned 5 min), profiled rollout as 90% of OPD step time and doubled throughput at constant 27 GB by scaling the rollout batch, and shipped 1.5K lines of tested Python (53 CPU tests, CI) with restart-safe Colab notebooks.

Guidance: describe it as independent research and a replication plus one measurement, never as a paper; be ready to explain every number (0.668 zero-shot, 0.834 full-data SFT, 0.827 OPD-300 over 3 seeds, 0.844 teacher, +2.9 paired delta over seeds, and that the first single-seed run showed +5.8) and to say unprompted that 900 and 3000 are still single-seed; if space is tight keep the second bullet.
