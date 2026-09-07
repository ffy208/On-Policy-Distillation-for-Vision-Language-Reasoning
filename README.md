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

Known Colab environment issue: installing vLLM upgrades torch to a newer CUDA build while the preinstalled torchaudio stays on the old one, and transformers then fails to import any processor. The install cell removes torchaudio for this reason; if you see `PyTorch and TorchAudio were compiled with different CUDA versions`, run `pip uninstall -y torchaudio` and retry.

A second Colab issue: upgrading Pillow in place can leave a mix of two versions on disk (`ImportError: cannot import name '_Ink' from 'PIL._typing'`). The install cell reinstalls Pillow cleanly; if you hit this, run `pip install --force-reinstall --no-deps pillow`, then restart the runtime.

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
  opd_trainer.py    (Stage 2) on-policy distillation training loop
  analysis/         (Stage 4) token heatmaps
notebooks/
  00_setup_and_eval.ipynb    install deps, sample data, zero-shot evaluation
  01_sft.ipynb               teacher generation -> LoRA SFT -> evaluation
  build_*.py                 generate the notebooks above (edit the script, then regenerate)
configs/            yaml configs
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

Student-teacher gap: 17.6 points. Evaluation of 500 rows takes 12 s (2B) and 36 s (8B) with vLLM on the A100.

## Stage progress

- [x] Stage 0: repository layout, `common.py`, `hub_utils.py`, `prepare.py`, `evaluate.py`, `00_setup_and_eval.ipynb`
- [x] Stage 1 code: `generate_teacher.py`, `collate.py`, `modeling.py`, `sft.py`, `01_sft.ipynb` (smoke run on Colab pending)
- [ ] Stage 2: OPD training loop
- [ ] Stage 3: data-efficiency curve
- [ ] Stage 4: token-level feedback visualization
- [ ] Stage 5 (optional): self-distillation (SDPO)
