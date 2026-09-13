# 能不能投 ICLR 2027：判断、缺口与 12 天关键路径

写于 2026-09-13。基于仓库当前状态（ChartQA 4 seeds、OOD 负结果、token-role 反馈分析）和
`docs/research_plan.md` 的撞车检查。本文只回答一个问题：**现在这份工作能不能变成一篇能投
ICLR 2027 主会的论文，代价是什么。**

## 0. 时间窗（先看这个）

| 事项 | 日期（AoE = UTC-12） | 距今 |
|---|---|---|
| 摘要注册（强制，逾期不能投正文） | 2026-09-18 | 5 天 |
| 正文截止 | 2026-09-25 | 12 天 |
| 评审放出 | 2026-11-05 | |
| 最终决定 | 2026-12-16 | |
| 会议 | 2027-04-26 至 30，加州 | |

摘要注册是免费的期权：注册了可以不投，没注册则正文无法提交。**无论最终投不投，9-18 之前
先把标题摘要注册掉。**

## 1. 硬性门槛：互惠评审（这是真正的 go/no-go）

ICLR 2027 新规：每篇投稿必须有**至少一位作者注册评审 3 篇论文**，且该作者需"有资格评审"
——即在 ICLR / NeurIPS / ICML / UAI / AISTATS / JMLR / TMLR / ACL / EMNLP / EACL / NAACL /
CL / TACL 等列表venue 上有**至少一篇已接收的主会论文**（workshop paper、tiny paper、
position paper、blog 不算），资格以摘要截止日前已接收的论文为准。

含义：**单人第一次投稿、没有上述已接收论文，无法满足这条，会被 desk reject。**
`research_plan.md` 里"第二周找一个 mentor"这一项，现在不是加分项，而是提交前置条件。

今天必须做的一件事：拿项目主页去找一位有资格的合作者（GT 里做 VLM / 蒸馏 / RL 的教授或
博后均可）。同时读 ICLR 2027 的 co-authorship 新规和 AI 使用披露新规——这个仓库是高度
agent 协作完成的，披露口径要提前想好，别在 rebuttal 时被问。

如果 5 天内找不到合格作者，ICLR 主会这条路今年直接关闭，跳到第 6 节的备选。

## 2. 现在手里有什么

- ChartQA（human-written）2B 学生 / 8B 教师，SFT vs OPD 的数据预算曲线 {100, 300, 900, 3000}，
  100 和 300 各 4 seeds，固定更新步数，paired bootstrap：+1.8 [+0.4, +3.4] 和
  +2.7 [+1.3, +4.2]；OPD 的跨 seed 标准差是 SFT 的一半。
- 数据效率：OPD 用 10% 数据 ≈ SFT 用全量。
- token-role 反馈测量：训练前教师 KL 在答案行上集中 2.67x，OPD 后降到 0.38x，残余转向
  chart_value；每 token 平均 KL 从 0.489 降到 0.235。
- OOD（CharXiv / ChartQAPro，4 seeds）：**负结果**，两种方法相对 zero-shot 都只涨 2-5 分，
  OPD 的优势基本落在噪声里（唯一显著是 CharXiv@100 的 +3.0）。
- 工程：notebook-free 的 Slurm runner、幂等实验编排、断点续训、单测、PACE 上 32 张 L40S
  可用（另有 8 张 A100 和 97 GB 的 Blackwell 节点，能放 32B 教师）。

一句话：**测量做得比绝大多数同类 arXiv 稿子干净，但它现在是"一个任务、一个规模的实证研究"。**

## 3. 直说：现在这个形态投主会会被怎么打

可预期的评审意见，几乎一定会出现：

1. **单任务。** 只有 ChartQA，结论无法排除"这是 ChartQA 的性质"。
2. **规模单点。** 2B 学生 / 8B 教师 / LoRA，没有规模轴，审稿人会说 conclusions may not scale。
3. **数据效率不是新发现。** GKD (Agarwal et al. 2024) 已经有这个结论，"在 VLM 上复现"本身
   不够 ICLR 主会的 novelty bar。
4. **没有方法。** 2026 年 VLM-OPD 已经是密集赛道（VA-OPD、Visual Gradient Steering 是
   ICML 2026 spotlight、FP-OPD、VOLD、PTD-PO、RP-OPSD…，另有 70+ 篇的 OPD 综述和统一框架
   EasyOPD）。纯 characterization 在这种密度下会被判 incremental。
5. **不和已发表方法比。** 没有任何 VLM-OPD 方法基线。
6. **主打结论之一是负结果。** 诚实是加分，但如果论文的重量落在负结果上，AC 很难给 accept。

所以：**不改造框架，就是投个几乎必拒的稿子。** 但改造的材料已经在手里了。

## 4. 能过线的论文形态：诊断 → 免费修正

ICLR 喜欢的形状不是"我们测量了 X"，而是"我们测量到 X，X 预测了 Y，按 Y 做一个零成本的改动
就能拿到本来要花两倍算力才能拿到的收益"。这个项目正好卡在这个形状上，因为
Stage 4 的测量**已经**指出 vanilla OPD 的反馈分配错位（答案行 2.67x、chart_value 0.39x）。

建议的论文骨架（暂定标题：*Answer First, Perception Last: Feedback Allocation in On-Policy
Distillation of Vision-Language Models*）：

| Claim | 内容 | 状态 |
|---|---|---|
| C1 规制 | 固定更新预算下，OPD 只在低数据区优于 SFT，约 10x 数据效率；跨两个任务、4 seeds、paired bootstrap | ChartQA 有；**Geometry3K 缺** |
| C2 动力学 | 教师 KL 的分配随训练迁移：答案行与格式分歧在最初几十步内塌掉，感知（chart_value、数字首 token）是持久残余；训练末期的感知残余占比**预测**学生到教师的剩余差距 | 每步 role-KL 日志已经在跑，**跨 run 的相关性分析缺** |
| C3 修正 | 由 C2 推出的 role-weighted KL（上权 chart_value 与每个数字的首 token），**零额外 forward**；对比 VA-OPD 式 counterfactual 重加权（需 2 次 teacher forward） | **代码与实验全缺**（`opd_utils.per_token_kl` 的权重接口在 plan 里列为待做） |
| C4 边界 | 分布漂移下 OPD 优势消失（含教师本身在 OOD 上只有 0.41 的诊断）；on-policy 优势在此规模上是 in-distribution 现象 | **已有**，作为 scope 一节而非主结论 |
| C5 规模轴 | 4B 学生、32B 教师各一档，证明结论不是单点 artifact | **缺**，但便宜 |

C3 是这篇论文能不能进主会的**唯一支点**：如果"免费的角色加权" ≈ "昂贵的视觉优势重加权"，
这是一个 ICLR 会买的结论（对整条 VLM-OPD 赛道说"你们的收益里有多少其实不需要额外 forward"）。
如果 C3 拿不到区间排除零的提升，就没有主会稿，只有一篇很好的 TMLR/CVPR 研究。

## 5. 算力与 12 天关键路径

按仓库实测吞吐（L40S 13.5 s/step，OPD 150 步；SFT 302 步约 12 min；vLLM 评测约 5 min 含引擎
启动；geometry 的 1536 token 预算按 34 s/step 估）算出的增量预算，见
`iclr_run_budget.csv`：

| 块 | OPD runs | SFT runs | GPU-h |
|---|---|---|---|
| Geometry3K 教师采样 + SFT 数据 | | | 0.5 |
| Geometry3K 曲线 100/300/900 × 4 seeds | 12 | 12 | 23.4 |
| Geometry3K 3000q 单 seed | 1 | 1 | 1.9 |
| 角色加权 KL，ChartQA 100/300 × 4 seeds | 8 | | 5.8 |
| 角色加权 KL，Geometry3K 100/300 × 4 seeds | 8 | | 12.7 |
| VA-OPD 式重加权基线，ChartQA 300 × 4 seeds | 4 | | 3.3 |
| 4B 学生，ChartQA 100/300 × 2 seeds | 4 | 4 | 5.7 |
| 32B 教师，ChartQA 300 × 3 seeds（Blackwell） | 3 | | 2.6 |
| 新模型的 OOD 评测补齐 | | | 3.3 |
| **合计** | **40** | **17** | **≈59** |

**算力不是瓶颈。** 59 GPU-h，在 32 张 L40S 上以并行度 8 跑是 7-8 小时墙钟（不含排队）。
瓶颈是三件事：C3 的代码、排队与坏节点（已经吃过 ECC 的亏）、以及**写作**。

关键路径（今天起）：

| 日期 | 动作 | 不做就死 |
|---|---|---|
| 9-13 今天 | 发合作者邀约（2-3 人）；注册 OpenReview；读 co-authorship / AI-use / 互惠评审三份新规；今晚提交 Geometry3K 教师采样作业 | 是 |
| 9-14 | 在 `opd_utils.per_token_kl` 上加 per-token 权重 + 从 `token_classes` 生成角色权重；单测 + 5 步 smoke；提交 Geometry3K 曲线 sweep（24 runs） | 是 |
| 9-15 | 提交 ChartQA 角色加权 sweep（8 runs）；实现 VA-OPD 式基线（teacher 两次 forward 的 counterfactual 版本） | 是 |
| 9-16 | Geometry3K 角色加权 sweep；写每步 role-KL 的聚合分析（C2 的相关性）；开写 Setup / Method | |
| 9-17 | 规模轴（4B 学生、32B 教师）；写 Intro + Abstract | |
| **9-18** | **注册摘要（AoE 截止）** | 是 |
| 9-19 至 20 | 所有 run 收尾；5 张图定稿；**C3 决策门**：两个任务上区间是否排除零 | 是 |
| 9-21 至 24 | 全文、related work（10+ 篇 2026 年工作要认真处理）、附录、reproducibility statement；找合作者读一遍 | |
| 9-25 | 提交 | |

图的配额（5 张，正文 10 页）：(1) 两任务的数据效率曲线含区间；(2) role-KL 随步数的迁移
（答案/格式 vs chart_value）；(3) 感知残余 vs 到教师的剩余差距的散点 + 相关性；
(4) vanilla / 角色加权 / VA-OPD 式 的对比条形图（含额外 forward 成本标注）；
(5) token 热力图（已有）。

## 6. 决策门与备选

**9-20 的门**：如果 (a) 合格合作者已落定，且 (b) C3 在两个任务上区间排除零 —— 投 ICLR 主会。
否则不要硬投；把同一批实验投向：

- **CVPR 2027**（截止约 2026-11 初）：多一个半月，能补硬漂移的 OOD 设计（训一个图表族、
  测另一个族，且教师在两边都强），chart/VLM 实证工作在 CVPR 的接受度好于纯 characterization 在 ICLR。
- **TMLR**（滚动，无截止）：`research_plan.md` 原本的选择，也是**期望值最高**的一个。审稿标准
  是"claim 是否被证据支持"而非 novelty，负结果和 scope 诚实在这里是加分。10-12 周的节奏能把
  三个任务、规模轴、方法 B 全做扎实。而且 TMLR 主会论文本身就能解掉明年 ICLR 的互惠评审资格。
- **ICLR 2027 workshops**（截止约 2027-02）：现在这个形态直接就能投，曝光快，但不算主会论文。
- **ARR / EMNLP 2027**：如果写作重心落在 analysis 上。

我的建议：**今天开始按 ICLR 路线跑（成本主要是 8 天的密集工作，实验本身很便宜），9-18 注册摘要
保留期权，9-20 按门槛决定投 ICLR 还是转 CVPR/TMLR。** 不要在这 12 天里再加第三个任务或
两阶段方法 C——范围纪律比多一个结果重要。

## 7. 这 12 天不要做的事

- 不要再加任务（第三个任务是 CVPR/TMLR 版本的事）。
- 不要重构 runner。
- 不要追求 900/3000 的全 seed 覆盖：C1 的重量在 100/300，大预算点保持单 seed 并在正文说明。
- 不要把负结果藏起来。审稿人会自己算 OOD 表，藏了更糟；把它写成 scope 一节。
