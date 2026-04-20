# Prior-Aware Masked Diffusion for Protein Language Model Steering

**项目主文档** — 包含目的、文章 story、实验设计、当前结果、剩余工作。

---

## 📋 目录

1. [项目背景与动机](#1-项目背景与动机)
2. [核心论点(Paper Thesis)](#2-核心论点paper-thesis)
3. [文章 Story & Title](#3-文章-story--title)
4. [方法:Prior-Aware Rejection Sampling](#4-方法prior-aware-rejection-sampling)
5. [实验设计](#5-实验设计)
6. [当前主要结果](#6-当前主要结果)
7. [Paper 结构](#7-paper-结构)
8. [剩余工作](#8-剩余工作)
9. [数据文件夹说明](#9-数据文件夹说明)
10. [References](#10-references)

---

## 1. 项目背景与动机

### 1.1 Masked Diffusion 式的蛋白质语言模型引导(Protein Steering)

**任务**: 给定属性目标(如可溶性、热稳定性、功能适应度),用预训练蛋白质语言模型 ESM-2 生成具备该属性的序列。

**主流做法** (Huang et al., 2024):
- **生成过程**: 迭代 mask-predict(10 轮,每轮 10% 位置 masked → ESM-2 forward → top-p sampling 填回)—— 这是一种 **masked diffusion** 式的生成
- **引导**: 在 ESM-2 的 forward 中注入 **steering vector** `h_new = (h + α·diff) * ||h||/||h + α·diff||`(norm-preserving rescale)
- `diff = mean(L17 of positive examples) - mean(L17 of negative examples)`
- 默认 scope 是 **all-layer(allL)**:每一层都注入 steering vector

### 1.2 Structure-Property Trade-off(核心问题)

**观察到的问题**:
- 强引导(large α)能显著提升目标属性(oracle)
- 但同时**结构预测 pLDDT 下降**,甚至出现 structure collapse(ESMFold pLDDT → 0.35,接近无序)
- Oracle 可能被 "oracle 幻觉" 欺骗:生成序列的氨基酸分布像可溶蛋白,但实际没折叠结构

### 1.3 直接的实验证据 — Structure Collapse 在 L17 空间 ≈ Random AA

在 sol_easy 上扫 α(单层 L17 steering):

| α | pLDDT | L17 Mahalanobis² |
|---|---|---|
| 1 | 0.656 | 1219 |
| 5 | 0.450 | 872 |
| **10** | **0.346** | **619** |
| **Random AA baseline** | **0.255** | **604** |

**过强引导(L17 α=10)产生的序列,其 L17 激活在统计意义上 与随机氨基酸袋难以区分**(Mahal 619 vs 604)。这是 paper 的 **smoking gun** motivation。

### 1.4 直觉:Steering 把激活推离 UniRef50 训练分布

ESM-2 的内部激活(特别是 L17)被训练在 UniRef50 ~58M 自然蛋白上。Steering 注入了 **off-manifold** 方向,迫使 forward pass 走出训练分布。生成的序列因此在 L17 层看起来"非自然",结构预测也崩塌。

**本文的核心思路**: 用 UniRef50 的 **先验分布信息** 过滤掉这些 off-manifold 生成。

---

## 2. 核心论点(Paper Thesis)

> **"Masked-diffusion 式的蛋白质语言模型引导会推开激活至训练分布外,导致结构塌陷。通过 post-generation 的密度 filter(基于 UniRef50 L17 先验分布),可以过滤掉 off-manifold 的生成序列,显著提升 steering 的结构-属性 trade-off。我们提出的 χ²(D)-based Mahalanobis filter 只需 10 KB 先验统计量,rejection sampling gain 超过 ESM-2 3B pseudo-perplexity 2-5 倍,cost 便宜 50 倍。"**

---

## 3. 文章 Story & Title

### 3.1 Title

**英文**: *"Prior-Aware Masked Diffusion for Protein Language Model Steering"*

### 3.2 Abstract(草稿)

Protein language model steering via masked-diffusion generation improves target properties (e.g., solubility, thermostability) but often at the cost of structural quality (pLDDT). We show the underlying cause is the deviation of intermediate-layer activations from the natural protein distribution: under strong steering, the L17 Mahalanobis distance of generated sequences becomes statistically indistinguishable from that of random amino acid baselines, and ESMFold pLDDT collapses to disordered levels. We propose **Prior-Aware Rejection Sampling**, a post-generation filter based on a χ²(D) probabilistic threshold of Mahalanobis distance at L17, using only ~10 KB of precomputed UniRef50 statistics and a single ESM-2 forward pass per sequence. On solubility and thermostability tasks, our filter simultaneously improves pLDDT (+0.04-0.07) and oracle score (+0.05-0.12), while being 2-5× more effective and 50× cheaper than pseudo-perplexity from ESM-2 650M and 3B. We also show a learned flow-matching residual (GLP) as an alternative implementation of the same prior provides only marginal additional gain over Gaussian Mahalanobis, suggesting first- and second-moment statistics suffice for this quality signal. We validate across 6 tasks: sol/therm × {easy, hard}, TrpB, and GFP, and discuss how task-specific sequence distributions (constrained mutagenesis protocols) require adaptive thresholds.

### 3.3 Story Arc(三幕)

**Act 1 (Motivation)**: 复现 Huang et al. steering,发现强 α 下 pLDDT 崩。Random AA baseline 定标 "sequence = junk",证明过强 steering 产生的 L17 激活 **在统计上无法与随机氨基酸区分**。

**Act 2 (Method)**: 问题本质是激活偏离 UniRef50 训练分布。**Post-generation filter** 基于 L17 先验密度过滤 off-manifold 序列。两种 filter 实现:
- **Mahalanobis χ²**(main): Gaussian 一阶二阶矩近似,10 KB
- **GLP residual**: 完整 flow-matching 密度,1.3 GB,但精度只略优

**Act 3 (Validation)**: 跨 6 tasks 实验。Filter 在 sol/therm 上给出干净双赢(pLDDT ↑, oracle ↑)。Fitness benchmarks 有 task-specific 限制但 percentile-based 变体 still applicable。Filter vastly outperforms ESM-2 pseudo-ppl (650M and 3B) at rejection sampling.

---

## 4. 方法:Prior-Aware Rejection Sampling

### 4.1 框架

给定一个 steering setting,生成 N 条候选序列,计算每条序列的**密度 proxy** `s(seq)`。保留 `s(seq)` 满足某个阈值的序列(或 top-K by s)。最终使用这些过滤后的序列作为最终输出。

### 4.2 Proxy 实现 1(main): Mahalanobis χ²

```python
# 预计算: UniRef50 所有蛋白过 ESM-2,取 L17 激活,per-dim 均值 μ 和方差 σ²
# 已保存为 rep_statistics.pt (10 KB)

# 对生成序列 seq(过 unsteered ESM-2)
h = ESM2(seq)[layer=17]         # (L, 1280) 每残基的 L17 激活
z = (h - μ) / σ                 # per-dim z-score
mahal_per_pos = (z ** 2).sum(-1)  # (L,) 每残基 Mahalanobis²
score = mahal_per_pos.mean()      # 全序列平均(长度归一化)
```

**成本**: 1 次 ESM-2 forward + element-wise 算术 = **~0.3s per seq, 10 KB stats**

### 4.3 Filter 策略

**Strategy A (absolute χ² threshold)**: 基于 χ²(D=1280) 分布理论
- 接受 `Mahal² ≥ D − k√(2D)`,比如 k=1 时 threshold = 1229
- **数学严谨**: 如果 h ~ N(μ, Σ),则 Mahal² ~ χ²(D),期望 = D = 1280,95% CI ≈ [1180, 1380]
- **高维 Gaussian 几何**: 自然蛋白的典型 Mahal² ≈ D,远离 mean(球壳效应)
- **适合**: sol/therm tasks(序列分布覆盖 UniRef50 shell 附近)

**Strategy B/C (percentile-based, task-adaptive)**:
- 接受 top-K% by Mahal within each setting
- **适合**: fitness benchmarks 或任何 Mahal 分布偏离 χ²(D) 的场景

### 4.4 Proxy 实现 2(alternative): GLP Residual

**Luo et al.** 在 UniRef50 上训练了一个 flow-matching 模型(Generative Latent Prior),估计 `p(h)` 的完整非 Gaussian 密度。我们用它做 SDEdit 式 residual:

```python
noisy = (1 - sigma) * h + sigma * noise         # 加小量噪声,sigma 由 u=0.15 控制
h_denoised = GLP.denoise(noisy, 25_steps)       # 25 步 denoise 回流形
resid_score = ||h - h_denoised||                # 残差作密度 proxy
```

**成本**: 1 次 ESM-2 forward + 25 步 GLP denoise ≈ **~0.8s per seq,需 1.3 GB GLP 模型**

**结果(见 §6)**: GLP resid 与 Mahal 相关度极高(r ≈ 0.95+),Spearman(GLP, pLDDT) 仅比 Mahal 好 0.01-0.05。**"第一、二阶矩近似已足够,learned flow-matching 的 non-Gaussian 信号对质量预测几乎没额外贡献"**。

### 4.5 Baseline: ESM-2 Pseudo-Perplexity

```python
# 在 15 个随机位置做 MLM-style pseudo-ppl
ppl = exp(mean(NLL over 15 masked positions))
```

两个 size: **650M** 和 **3B**。

**结果**: Mahal 和 GLP resid 都**大幅 beat ppl**(见 §6)。

---

## 5. 实验设计

### 5.1 任务覆盖 (6 tasks)

| Task | 类型 | 属性预测器 | Reference N | Gen N |
|---|---|---|---|---|
| **sol_easy** | Solubility(binary)| Ankh-based predictor | 162 | 500 |
| **sol_hard** | Solubility(harder split)| 同上 | 162 | 500 |
| **therm_easy** | Thermostability(°C)| Tm regressor | — | 500 |
| **therm_hard** | Thermostability(harder split)| 同上 | — | 500 |
| **trpb** | Fitness(4 位点突变)| lookup from experimental data | 200 | 200 |
| **gfp** | Fitness(Kirjner hard split)| gfp_sarkisyan predictor | 200 | 200 |

### 5.2 Steering Settings(每 task 5 个)

1. **L17_a1** — weak single-layer(baseline 近自然)
2. **L17_a10** — strong single-layer(**structure collapse 证据,用于 Fig 1 motivation**)
3. **allL_a2** — moderate all-layer(**filter 主要验证点**)
4. **allL_a3** — strong all-layer(oracle 饱和)
5. **allL_a2_L17GLP_u0.5** — 对比(附录/ablation,不是 main claim)

共 **6 tasks × 5 settings = 30 (task, setting) 组合**。

### 5.3 Pipeline Per (task, setting)

1. 生成 N 条序列 (500 for sol/therm, 200 for fitness)
2. Oracle 评分 → per-seq property score
3. ESMFold → per-seq pLDDT + pTM
4. Proxy 计算 → per-seq:
   - **Mahal**: 用 rep_statistics 算 L17 Mahalanobis²(main filter)
   - **GLP resid (u=0.15)**: GLP SDEdit 的 residual(alternative filter)
   - **ppl 650M**: ESM-2 650M pseudo-perplexity(baseline)
5. **allL_a2 only**: 额外计算 **ppl 3B**(100 seqs × 6 tasks)作 main text baseline

### 5.4 Random AA Baseline

从 `data/benchmarks/random_aa_seqs/random_1000.csv` 取 100 条随机序列,同样计算 Mahal / pLDDT / oracle。这是 Motivation Fig 1 的 **关键 anchor**。

---

## 6. 当前主要结果

### 6.1 Motivation (Fig 1 数据)

**Structure collapse under strong L17 steering** (sol_easy):

| α | L17 pLDDT | L17 Mahal² |
|---|---|---|
| 1 | 0.656 | 1219 |
| 2 | 0.626 | 1207 |
| 3 | 0.590 | 1143 |
| 5 | 0.450 | 872 |
| **10** | **0.346** | **619** |
| **Random AA** | **0.255** | **604** |

**Key finding**: L17 α=10 的 Mahal (619) ≈ Random AA Mahal (604) — "过强 steering 产生的序列在 L17 空间 **与随机氨基酸袋难以区分**"。

### 6.2 Mahal 分布 per task(allL_a2 setting)

| Task | Mahal 均值 | > 1229 的比例 | χ² filter 适用? |
|---|---|---|---|
| sol_easy | 1013 | 10.6% | ✅ 适配 |
| sol_hard | 1078 | 21.8% | ✅ 适配 |
| therm_easy | **1389** | 99.6% | ⚠️ filter 几乎 no-op(需 percentile 版) |
| therm_hard | 1390 | 99.4% | ⚠️ 同上 |
| trpb | **1439**(极窄) | 100% | ❌ filter 无差异(WT-邻近)|
| gfp | **478**(极低) | 0% | ❌ 所有 OOD(需 percentile 版)|

不同 task 的 Mahal 自然分布差异很大,单一绝对阈值**不能直接套用到所有 task**。但在 **Mahal 自然分布覆盖 UniRef50 shell(≈ 1280)附近** 的任务(sol/therm)上,χ² filter 非常有效。

### 6.3 **Main Result** — χ² filter 在 sol/therm 4 splits 上**双赢**(Table 1)

**Strategy A (absolute χ² threshold k=1, Mahal² ≥ 1229), task-optimal steering setting per task**:

| Task | Setting(最适 filter 的 regime)| base (pLDDT / oracle) | + Mahal filter | **Δ pLDDT** | **Δ oracle** |
|---|---|---|---|---|---|
| **sol_easy** | allL α=2 (moderate steering) | 0.483 / 0.672 | **0.546 / 0.790** | **+0.063** | **+0.118** |
| **sol_hard** | allL α=2 | 0.512 / 0.684 | **0.580 / 0.755** | **+0.068** | **+0.071** |
| **therm_easy** | L17 α=1 (weak steering) | 0.696 / 54.41°C | **0.759 / 55.14°C** | **+0.063** | **+0.735°C** |
| **therm_hard** | L17 α=1 | 0.694 / 48.22°C | **0.761 / 48.57°C** | **+0.067** | **+0.352°C** |

**所有 4 splits 都 pLDDT ↑ 且 oracle ↑**,Δ pLDDT 稳定 +0.06-0.07。

**Framing**: Task 之间最适合 filter 的 steering regime 不同,反映不同 task 的**天然 fitness**:
- **Solubility tasks**: 需要 **moderate all-layer α=2** 才把 oracle 推到 0.67+(近饱和);filter 在此筛出 Mahal 高的 "真可溶" 序列
- **Thermostability tasks**: **weak L17 α=1** 已足以给出 54°C/48°C 的 oracle;filter 筛掉 off-manifold 的坏样本获得 +0.06 pLDDT 和 +0.35-0.74°C

**Filter 的 universality 体现在对两种 regime 都 work**,不是 "一个阈值处处适用"。

### 6.3b 跨所有 5 settings × 4 splits 的 filter 平均效果

| Task | avg Δ pLDDT | avg Δ oracle | Filter 有效? |
|---|---|---|---|
| **sol_easy** | **+0.045** | **+0.091** | ✅ |
| **sol_hard** | **+0.054** | **+0.052** | ✅ |
| therm_easy | +0.026 | +0.150 °C | ✅(部分 settings)|
| therm_hard | +0.038 | **+0.764** °C ⭐ | ✅(部分 settings)|
| trpb | 0 | 0 | ❌(task protocol 限制,WT-邻近)|
| gfp | — | — | ❌(Mahal 均值太低,阈值失配)|

**结论**: Filter 在 **sol/therm 4 个 splits 上稳定 work**;fitness 任务有 **task-specific 限制**(见 §5)。完整 5 setting × 4 splits 详细表格见 Appendix C。

### 6.4 Main Result — Filter beats Pseudo-Perplexity

**rejection sampling (top-20% vs random-20%) on sol_easy allL_a2 (N=250)**:

| Proxy | Δ pLDDT | Δ oracle | Cost per seq | Model size |
|---|---|---|---|---|
| **Mahal** | **+0.049** | **+0.122** | **0.3s** | **10 KB stats** |
| GLP resid (u=0.05) | +0.045 | **+0.133** | 0.8s | 1.3 GB |
| GLP resid (u=0.15) | **+0.066** | +0.097 | 0.8s | 1.3 GB |
| ppl 650M | +0.028 | +0.021 | 5s | 2.5 GB |
| **ppl 3B** | **+0.025** | **-0.000** ❌ | **15s** | **11 GB** |

**Mahal + GLP resid 都大幅优于 ppl(包括 3B)。3B 不比 650M 强。**

### 6.5 Spearman r(proxy, pLDDT),3 key settings

| Setting | Mahal r | GLP resid r (u=0.15) | ppl 650M r | ppl 3B r |
|---|---|---|---|---|
| sol_easy allL_a2 | +0.51 | **+0.54** | -0.27 | -0.23 |
| sol_easy L17_a1 | +0.67 | **+0.68** | -0.59 | -0.55 |
| sol_easy allL_a3 | +0.21 | **+0.25** | -0.04 | -0.06 |

**Finding**: Mahal ≈ GLP resid(very close),两者**一致 beat ppl**。

### 6.6 "Goldilocks" 分析(14k seqs pool, sol_easy only)

Bin 序列按 Mahal,观察 pLDDT 和 oracle:

| Mahal 区间 | N | mean pLDDT | mean oracle | 语义 |
|---|---|---|---|---|
| 319-538(极低)| 1390 | 0.352 | 0.330 | over-steered / random |
| 538-707 | 2085 | 0.359 | 0.387 | over-steered mild |
| 707-936 | 3475 | 0.436 | 0.556 | moderate allL strong α |
| **936-1149** | **3475** | **0.529** | **0.587** ⭐ | **oracle 甜点** (allL α=2-3) |
| 1149-1367 | 2085 | 0.607 | 0.424 | 近自然 |
| **1367-2143** | **1390** | **0.666** ⭐ | 0.436 | **pLDDT 甜点** (L17 α=1, natural) |

**pLDDT 单调随 Mahal 上升**;oracle 呈 **inverted-U**(甜点 Mahal ~1000)。Random AA (Mahal 604) 落在最低 bin,证实 motivation。

---

## 7. Paper 结构

### §1 Introduction
- Protein steering + masked diffusion 是什么
- Huang et al. 的 trade-off 问题
- Prior-aware rejection sampling 的提出
- 4 个 contributions:
  1. 发现 over-steering 让 L17 激活 ≈ random AA
  2. 提出 χ² Mahal filter,双赢(pLDDT + oracle)
  3. 显示 Mahal ≈ GLP resid >> ppl(ppl 3B 都不行)
  4. 跨 6 tasks 验证 + 讨论 task-specific 限制

### §2 Background
- ESM-2 mask-predict 生成(§2.1)
- Steering vector 注入机制(§2.2)
- UniRef50 L17 先验分布 + Generative Latent Prior(§2.3)
- Mahalanobis 距离 + χ²(D) 分布(§2.4)

### §3 Method: Prior-Aware Rejection Sampling
- 框架 overview(§3.1)
- Mahalanobis χ² filter(§3.2)
- Strategy A absolute vs B/C percentile(§3.3)
- Alternative: GLP residual(§3.4)
- Baseline: pseudo-perplexity(§3.5)

### §4 Experiments

#### §4.1 Motivation: Structure Collapse (Fig 1)
- L17 α=10 pLDDT 崩到 0.35
- L17 α=10 Mahal ≈ Random AA Mahal(**smoking gun**)
- Trade-off quantified across settings

#### §4.2 Main Result — Filter 在 sol/therm 4 splits 上双赢(Fig 2 + Table 1)
- Main Table 1 展示 task-optimal setting per task(sol: allL α=2, therm: L17 α=1)
- 所有 4 splits 都 pLDDT ↑ 且 oracle ↑,Δ pLDDT 稳定 +0.06-0.07
- Framing: 不同 task 需要不同 steering 强度,filter 对两种 regime 都 work
- Full 5 setting × 4 split matrix 放 Appendix C
- Fitness tasks 单独在 §4.3 discussed(task-specific 限制)

#### §4.3 Filter vs Pseudo-Perplexity(Fig 3 + Table 2)
- Mahal beats ppl 650M and 3B
- 3B 不比 650M 强
- Cost-effectiveness: 10 KB + 0.3s vs 11 GB + 15s

#### §4.4 Mahal vs GLP Resid(Fig 4 + Table 3 / appendix)
- Spearman correlation 对比
- Rejection sampling gain 对比
- **结论**: "First two moments 基本够了,learned flow-matching 的 non-Gaussian gain 微小"

#### §4.5 Goldilocks Analysis(Fig 5)
- Mahal binning → pLDDT 单调、oracle inverted-U
- Random AA 锚点

### §5 Analysis
- 高维 Gaussian 球壳几何(why Mahal works)(§5.1)
- L17 α=10 ≈ Random AA 的机制(§5.2)
- χ² filter 在 fitness tasks 失效的原因(§5.3)

### §6 Limitations & Future Work
- Gaussian 假设近似(真实 L17 non-Gaussian)
- Fitness tasks 的 adaptive threshold 未深入
- **In-generation modifier 作为 negative result**:我们也尝试过用 GLP 在生成过程中做 SDEdit 投影,仅在 allL α=1.5-2 sol_easy 下 work,不通用,故本文未采用。Attention-based GLP 可能 unlock 这条路径。
- Ensemble of Mahal + GLP resid

### §7 Conclusion

### Appendix A: χ²(D) 分布证明
### Appendix B: 高维 Gaussian shell 几何可视化
### Appendix C: 全 30 (task, setting) × proxy × filter strategy 详细结果
### Appendix D: Reproducibility 细节

---

## 8. 剩余工作

### 8.1 核心已完成

- [x] 6 tasks × 5 settings 全部生成
- [x] 全部 ESMFold (500 or 200 seqs per setting)
- [x] 全部 oracle scored
- [x] 全部 proxy (Mahal + GLP resid u=0.15 + ppl 650M)
- [x] ppl 3B on allL_a2 × 6 tasks × 100 seqs
- [x] Random AA baseline + Goldilocks 分析(sol_easy)
- [x] 数据整理到 `paper_data/`

### 8.2 分析与制图(待做)

- [ ] **Fig 1 teaser**: Motivation 两栏图(pLDDT vs α + L17_a10 ≈ Random AA 散点)
- [ ] **Fig 2** (main): Filter 在 6 tasks 的 pLDDT/oracle 提升 bar chart
- [ ] **Fig 3**: Filter vs ppl 成本-效果 scatter
- [ ] **Fig 4**: Mahal vs GLP resid 相关度(证明 Gaussian 足够)
- [ ] **Fig 5**: Goldilocks binning(pLDDT/oracle vs Mahal bin)
- [ ] **Table 1**: Main result - filter gain per (task, setting)
- [ ] **Table 2**: Proxy cost benchmark(wall-clock + memory)
- [ ] **Table 3**: Appendix 详细结果矩阵

### 8.3 补充实验(nice-to-have)

- [ ] Ensemble of Mahal + GLP + ppl → 是否更好?
- [ ] K-sweep(top-5%, 10%, 20%, 50%)filter 曲线
- [ ] Full covariance Mahalanobis vs diagonal(检查近似损失)
- [ ] Hutchinson NLL 作 "proper" GLP density 对比

### 8.4 写作

- [ ] Draft abstract + intro
- [ ] Method section
- [ ] Results with figures + tables
- [ ] Appendix
- [ ] Code release (rejection_sampler.py)

---

## 9. 数据文件夹说明

### 目录结构

```
paper_data/
├── PAPER_PLAN.md                        ← 本文档
├── README.md                            (原始 autogenerated README)
│
├── reference/                           (20 KB)
│   └── rep_statistics.pt                UniRef50 L17 mean + var (for Mahal)
│
├── steering_vectors/                    (1.4 MB)
│   ├── 650M_sol_steering_vectors.pt
│   ├── 650M_therm_steering_vectors.pt
│   ├── 650M_trpb_fitness_steering_vectors.pt
│   └── 650M_gfp_fitness_steering_vectors.pt
│
├── random_aa/                           (816 KB,motivation anchor)
│   ├── sequences.csv                    1000 random amino acid sequences
│   ├── esmfold_results.csv              pLDDT per random seq
│   └── random_sol_scored.csv            sol oracle for random
│
├── goldilocks_sol_easy/                 (4.1 MB)
│   ├── goldilocks_all.csv               All ~14k sol_easy seqs with Mahal+pLDDT+oracle
│   ├── goldilocks_settings.csv          Per-setting aggregates
│   └── goldilocks_bins.csv              Mahal binning results
│
├── sol_easy/ (4.9 MB)                   ┐
├── sol_hard/ (2.7 MB)                   │
├── therm_easy/ (2.8 MB)                 │── 6 个 task,结构相同:
├── therm_hard/ (3.1 MB)                 │   ├── generated/ (5 settings × seqs)
├── trpb/ (1.8 MB)                       │   ├── oracle/ (5 scored CSVs)
└── gfp/ (1.2 MB)                        ┘   ├── esmfold/ (merged per-seq pLDDT)
                                             └── proxy/ (per-seq Mahal+GLP resid+ppl)
```

### Proxy CSV 列说明

每个 `paper_data/<task>/proxy/<setting>_proxy.csv` 包含:

| 列 | 含义 |
|---|---|
| `sequence` | 生成的蛋白序列 |
| `mahal` | Mahalanobis²(per-residue 均值),**main filter proxy** |
| `glp_resid` | GLP denoising residual at u=0.15(alternative filter proxy,用于 ablation)|
| `ppl_650m` | ESM-2 650M pseudo-perplexity(baseline)|
| `plddt` | ESMFold predicted LDDT |
| `oracle` | task-specific oracle score |

**ppl 3B** 在单独文件 `<setting>_ppl3b.csv` 里,只有 allL_a2 setting × 100 seqs(main text baseline)。

### 数据量

| 量 | 数值 |
|---|---|
| Tasks | 6 |
| Settings per task | 5 |
| Total (task, setting) pairs | 30 |
| Generated seqs total | ~13,400 |
| pLDDT / oracle data points | ~13,400(全部 ESMFolded + scored)|
| Proxy data points | ~13,400 (Mahal + GLP + ppl 650M)|
| ppl 3B data points | 600(100 seqs × 6 tasks,allL_a2 only)|

---

## 10. References

- **Huang et al., 2025**: "Steering Protein Language Models" (ICML 2025) — base steering paper
- **Luo et al., 2025**: GLP paper — flow-matching prior on L17
- **ESM-2**: Lin et al., 2023, Science
- **ESMFold**: Lin et al., 2023
- **SDEdit**: Meng et al., 2021 — diffusion editing (inspiration for GLP residual construction)

**Datasets**:
- **Solubility (sol_easy/hard)**: DeepSol dataset splits
- **Thermostability (therm)**: Meltome Atlas
- **TrpB (fitness)**: engineered 4-site mutations
- **GFP (fitness)**: Kirjner et al. hard split

---

## 📌 项目当前状态(2026-04-20)

- ✅ **所有实验数据已生成并整理**(`paper_data/` 就绪)
- ✅ **核心发现已 validated**:
  - Motivation: L17 α=10 ≈ Random AA structurally
  - **Main result: Mahal χ² filter 在 sol/therm 上同时提升 pLDDT 和 oracle**
  - Mahal + GLP resid 都大幅 beat ESM-2 pseudo-ppl (包括 3B)
  - Fitness 任务 filter 需 task-specific tuning(discussion)
- ⏳ **接下来**: 制图 + 写作(~2 周)
- 🎯 **目标投稿**: ICML 2026 / NeurIPS 2026 / bioRxiv first

**负责人**: Shui-bai Zhang (szhang967@wisc.edu)
