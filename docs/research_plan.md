# From project to paper: research plan (draft, 2026-09-09)

This note turns the finished study into a paper-sized direction. It records the gap analysis
against the 2026 literature, the candidate contributions, the experiment plan with compute
estimates, and the venue and timeline options. It is a plan, not a commitment; revise as results
arrive.

## 1. Where the field is (Sept 2026)

Method papers on on-policy distillation (OPD) for vision-language models all start from the same
observation, that vanilla OPD under-serves visual grounding, and change the objective:

| Paper | Diagnosis | Fix | Cost |
|---|---|---|---|
| VA-OPD (2605.21924) | on vision-critical tokens the student's prediction barely changes with or without fine visual detail, while the teacher's does | teacher-relative counterfactual "visual advantage" per token; separate KL for high-VA and low-VA tokens; rollout re-weighting | 2 teacher forwards per rollout |
| Visual Gradient Steering (2606.00564, ICML 2026 spotlight) | gradients of the language-prior and visual-grounding components are nearly orthogonal | steer the update toward the grounding component | modest |
| FP-OPD (2608.01263) | teacher corrections can depend on visual distinctions a compact student cannot represent | project the teacher-student gap onto the student's visual tangent space under the Fisher metric | perturbation forwards |
| Vision-OPD (2605.18740) | regional-to-global perception gap: crops answer better than full images | self-distillation from crop-conditioned teacher | crops |
| Perception Before Supervision (2608.09931) | counterfactual blind spots | self-contained visual distillation | counterfactuals |
| VOLD (2510.23497) | LLM-to-VLM reasoning transfer needs a cold start | cold-start alignment then OPD | training stages |

What none of them do:

1. **Characterize the vanilla objective's regime.** They report benchmark gains at one data scale. Nobody maps *when* vanilla OPD beats SFT (data budget, distribution shift, capacity gap) and when it does not. Our data-efficiency curve is the only such measurement we found, and it is one task, one seed.
2. **Measure what gets learned first.** Our token-role analysis shows OPD removes answer-line and format disagreement first and leaves chart-value disagreement last. None of the papers report feedback dynamics over training; VA-OPD reports the static concentration of VA in a token minority, which is consistent with our finding and complementary to it.
3. **Test the DAgger prediction that matters.** The argument for OPD is compounding error under drift. The clean test is distribution shift at evaluation: train on one chart distribution, test on others. The 2026 papers evaluate in-distribution.

## 2. Candidate contributions (pick the first; the others are fallbacks)

**A. An empirical study: when and why on-policy distillation beats supervised distillation for VLMs.**
Three claims, each with a controlled experiment:

- *Sample efficiency.* The 10x crossover replicates across tasks and student sizes. Curve budgets {100, 300, 900, 3000}, fixed update counts, paired bootstrap, 3 seeds at the crossover budget.
- *Robustness under shift.* OPD's advantage over SFT grows out of distribution. Train on ChartQA human questions; evaluate on held-out chart benchmarks (ChartQA-Pro / CharXiv / PlotQA / ChartBench subsets) with the same scorer. Prediction from theory: SFT overfits teacher traces on the training distribution; OPD trains on student-reached states and degrades less.
- *Feedback dynamics.* Log the teacher's KL by token role at every step. Hypothesis: answer and format disagreement collapse within the first tens of steps, perception (chart values, first digits of numbers) is the persistent residual, and the residual's share predicts the remaining gap to the teacher. This gives the re-weighting papers a *when*, not just a *where*.

Positioning: complementary to VA-OPD / VGS / FP-OPD. They change the objective; this paper characterizes the regime of the plain objective and shows what the modified objectives are needed for. Fits a rigorous empirical venue.

**B. A cheap method riding on A's finding.** Role-aware or first-digit-aware KL weighting (upweight `chart_value` tokens and the first token of each number). Zero extra forwards, unlike VA-OPD. Only worth writing up if it closes a measurable part of the perception residual; otherwise it is an ablation inside A.

**C. Staged distillation.** If A's dynamics claim holds, a two-stage recipe (OPD for format and answer selection, then a perception-targeted objective) could beat either alone. Higher risk; only after A.

## 3. Experiment plan for A

Reuse everything in this repository; the additions are data adapters, OOD evaluation, per-step role logging, and seeds.

| Block | Runs | GPU hours (A100) |
|---|---|---|
| ChartQA, 2B student: add 100-question point, 3 seeds at 100 and 300 for SFT and OPD | 10 OPD, 10 SFT | 20 |
| Second task (Geometry3K or a document/infographic VQA subset), 2B student: budgets {100, 300, 900, 3000}, 1 seed, plus 3 seeds at the crossover | 7 OPD, 7 SFT | 14 |
| Third task, same design | 7 OPD, 7 SFT | 14 |
| 4B student on ChartQA, budgets {300, 900, 3000} | 3 OPD, 3 SFT | 10 |
| OOD evaluation of every model on 3 to 4 held-out chart benchmarks | evaluation only | 6 |
| Feedback dynamics: per-step role-KL logging (free), plus token-feedback analysis at 5 checkpoints per run for 6 runs | analysis | 4 |
| Method B ablation on ChartQA at 300 questions, 3 seeds | 3 OPD | 6 |
| **Total** | | **~75 to 90** |

Colab Pro units do not cover this. Options: Georgia Tech PACE (already used for the Soccer-Twos project), or rented A100s (~$1.5 to 2 per hour, so roughly $150 to $200). PACE is the obvious choice; the code is notebook-independent (every stage is a `python -m vlm_opd.*` command) so it moves to a Slurm script directly.

Code additions, in order:

1. `prepare.py`: dataset adapters (`--source` already exists; add per-source field mapping and scorer choice, multiple-choice scoring for Geometry3K).
2. `opd_trainer.py`: per-step KL share by token role using `analysis/token_classes.py` on the rollouts (cheap: tokens are already decoded for the accuracy signal). Log to `opd_log.jsonl`.
3. `evaluate.py`: `--data-repo` for OOD sets; a script to build the OOD test sets with the same unified schema.
4. `03_data_efficiency` generalized to a Slurm-friendly runner with a task argument; seeds as a loop.
5. `opd_utils.per_token_kl`: optional per-token weights for method B.

## 4. Venue and timeline

- **TMLR** (rolling, no deadline): the best fit for a careful controlled study; reviewers reward rigor and honest scope over novelty. Target submission in 10 to 12 weeks.
- **ICLR 2027 workshops** (deadlines around Feb 2027) or **CVPR 2027** (deadline ~Nov 2026; tight, would need the runs done by late October).
- **ACL / EMNLP 2027 findings** if the writing leans toward analysis.

Timeline at roughly 10 hours per week:

| Weeks | Milestone |
|---|---|
| 1 to 2 | Data adapters for two more tasks, OOD test sets, per-step role logging; smoke everything on Colab |
| 3 to 6 | Runs on PACE; seeds; OOD evaluation; keep `docs/resume_log.md` style records |
| 7 to 8 | Analysis: curves with intervals, OOD deltas, dynamics plots; decide whether method B is in or out |
| 9 to 10 | Draft with `ml-paper-writing`; internal review; arXiv + TMLR submission |

## 5. Risks and how to read them early

- *The crossover does not replicate on another task.* Then the paper becomes "when it does and does not", still publishable if the OOD and dynamics claims hold; check after the second task's 300-question point (week 3).
- *OOD shows no OPD advantage.* Then the DAgger argument does not translate at this scale; report it. Decide after ChartQA-trained models are scored on the first OOD set (week 3).
- *Seed variance swallows the 300-question gap.* Then the headline weakens to the crossover statement; check after the first 3 seeds (week 3).
- *A mentor.* An independent study or a faculty co-author at Georgia Tech raises the credibility of the venue submission and unlocks compute. Approach someone in the second week with the current project page as the pitch.

## 6. What to do first

1. Ask for PACE allocation and a faculty mentor with the project page.
2. Add per-step role-KL logging (one afternoon); every future run then produces the dynamics data for free.
3. Run the 100-question ChartQA point and 3 seeds at 300 on whatever GPU is available; this decides how strong the headline is before any new task is built.

## 7. Collision check for a self-distillation variant (2026-09-09)

Asked: could on-policy *self*-distillation (same 2B model as teacher with privileged information) be the paper? Prior-work check:

| Idea | Already done by | Date |
|---|---|---|
| Resolution as privilege (teacher sees full-res, student low-res) | RP-OPSD (2607.24447), Qwen3.5-9B, +5.45% rel., 1.78x faster than OPSD | Jul 2026 |
| Crops as privilege | Vision-OPD (2605.18740) | May 2026 |
| Answer-revealing vs answer-free hints for multimodal | PTD-PO (2606.07000): argues against answer-revealing, uses spatial + textual hints | Jun 2026 |
| Balancing several privilege types | AVSD (2605.20643), text math/code | May 2026 |
| Evidence-grounded self-teacher + token weighting by visual reliance | Video-OPSD (2608.27065) | Aug 2026 |
| Visual cues as recoverable privilege | ViCuR (2606.05718) | Jun 2026 |
| Augmented views, no privilege | S2VOPD (2608.14144) | Aug 2026 |
| Length-collapse mechanism of OPSD | One Symptom, Three Levers (2608.25936) | Aug 2026 |
| Which teacher tokens are reliable (position) | 2605.21606; Position bias of OPD (2606.22600) | May/Jun 2026 |

Verdict: every ingredient of "answer vs visual privilege for a VLM self-teacher" exists as a separate paper, and the cadence is roughly one new OPSD-for-VLM paper every two to three weeks. A method paper here is very likely to be scooped or judged incremental. Self-distillation variants should enter the work only as *conditions* of a study, not as the contribution.

What the check did **not** find for VLMs: an OPD-vs-SFT data-budget crossover study, an out-of-distribution (chart-to-chart shift) comparison, or feedback-allocation dynamics by token role over training. For text LLMs, Rethinking OPD (2604.13016) and Position Bias (2606.22600) cover dynamics partially. The regime-study plan in sections 2 to 4 therefore stands, with one addition:

- **Diagnostic framing.** Use the token-role feedback tool as the paper's lens: measure where the teacher's signal lands and how it moves over training for vanilla OPD, an answer-privileged self-teacher (SDPO-style), and a resolution-privileged self-teacher (RP-OPSD reimplementation, cheap on our infrastructure). This turns the self-distillation interest into three rows of one table instead of a competing method, and it is the kind of comparison the method papers do not report.
