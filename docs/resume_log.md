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

### Stage 2 (smoke run passed 2026-09-08, full runs pending)

- Accomplished a working on-policy distillation loop for a VLM on a single 40 GB GPU, as measured by a 20-step smoke run (batch 8 rollouts, 100 questions, reverse KL) with per-token KL falling from 0.421 to 0.262, peak GPU memory 26.97 GB with the 8B teacher and 2B student co-resident, 33.0 s per step on average (rollout 17 to 46 s, teacher forward about 0.5 s, student forward-backward about 1.8 s), checkpoints pushed to the Hub at steps 10 and 20, and the merged student scoring 0.792 accuracy (format 0.966) on the 500-question test set versus 0.668 zero-shot, by sampling student rollouts at temperature 1.0, scoring only the generated span with `logits_to_keep`, computing the full-vocabulary KL in fp32 chunks, and updating a rank-64 LoRA.
- Observation for the length-collapse question: mean rollout length grew from 121 to 215 tokens over 20 steps (teacher responses are longer than the student's), so the risk in this setup is lengthening toward the 512-token cap rather than collapse; EOS rate stayed at 0.75 to 1.00 per batch.
- Accomplished disconnect-safe OPD training on Colab, as measured by the 300-question run resuming automatically from the Hub checkpoint at step 75 after a session loss (adapter, optimizer, scheduler, and data cursor restored) and completing steps 76 to 150 with no repeated or skipped batches, by checkpointing every 25 steps with a `latest.txt` pointer and a seeded data permutation.
- Measured OPD throughput at batch 16 (2026-09-08, 300-question run): 35 s per step on average (range 18 to 53 s; rollout 15 to 47 s, teacher forward 1.0 s, student forward-backward 3.5 s), peak memory 27.08 GB (unchanged from batch 8), so doubling the batch doubled training tokens per second at no memory cost. Late-run per-token KL 0.10 to 0.28 (mean about 0.17), mean rollout length 80 to 186 tokens (no drift toward the 512 cap), EOS and format rates at 1.00.
- [pending] OPD accuracy at 300 / 900 / 3000 questions and the paired comparison against SFT.

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

## Resume bullets (draft, update when numbers are final)

- Built an on-policy distillation pipeline for vision-language models (Qwen3-VL 2B student, 8B teacher) on ChartQA, [pending: improving student accuracy from 0.668 to X and outperforming supervised distillation by Y points on 500 held-out human-written questions].
- [pending: data-efficiency claim, e.g. matched full-data SFT accuracy with N% of the training questions].
- [pending: analysis claim, e.g. showed that X% of token-level teacher feedback concentrates on chart-reading numeric tokens].
