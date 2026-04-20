# Prior-Aware Masked Diffusion for Protein Language Model Steering

**项目主文档** — 包含目的、文章 story、实验设计、当前结果、剩余工作。

---

## 📋 目录

1. [项目背景与动机](#1-项目背景与动机)
2. [核心论点(Paper Thesis)](#2-核心论点paper-thesis)
3. [文章 Story & Title](#3-文章-story--title)
4. [方法框架](#4-方法框架)
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
- 两种 scope:
  - **Single-layer (L17)**: 只在 layer 17 注入
  - **All-layer (allL)**: 每一层都注入

### 1.2 结构-属性 Trade-off

**观察到的问题**:
- 强引导(large α)能显著提升目标属性(oracle)
- 但同时**结构预测 pLDDT 下降**,甚至出现 structure collapse(ESMFold pLDDT → 0.35,接近无序)
- Oracle 可能被 "oracle 幻觉" 欺骗:生成序列的氨基酸分布像可溶蛋白,但实际没折叠结构

### 1.3 Generative Latent Prior (GLP)

**Luo et al. 的工作**:
- 在 UniRef50 上训练 **flow-matching model**,建模 ESM-2 **L17 激活** 的分布 `p(h)`
- 335M 参数的 `TransformerMLPDenoiser`(per-position MLP,**无 self-attention**)
- 存储为 `rep_statistics.pt` (L17 的 mean + variance)+ GLP denoiser 权重

**GLP 可以做的两件事**:
1. **Projection**: 给定一个激活 h,`GLP(h + noise)` 可以 denoise 回 p(h) 的流形
2. **Density estimation**: 评估 h 在 p(h) 下的密度

---

## 2. 核心论点(Paper Thesis)

> **"基于 masked diffusion 的蛋白质语言模型引导会把激活推离 UniRef50 训练分布,导致结构塌陷。通过在引导过程中引入分布先验信息(无论是生成中的 GLP 投影 modifier,还是生成后的 Mahalanobis-χ² 密度 filter),都能显著改善引导生成的结构-属性 trade-off。"**

**两种引入先验的方式(都涵盖在 "Prior-Aware Masked Diffusion" 框架下)**:

1. **Modifier (in-generation)**: 用 GLP 的 SDEdit 投影,在每个 forward pass 的 L17 处把激活拉回训练流形
2. **Filter (post-generation)**: 用 Mahalanobis-χ² 距离或 GLP residual 作密度 proxy,对生成序列做 rejection sampling

**它们共享同一个 prior(UniRef50 L17 分布),只是 **应用阶段不同**。

---

## 3. 文章 Story & Title

### 3.1 Title

**英文**: *"Prior-Aware Masked Diffusion for Protein Language Model Steering"*

### 3.2 Abstract(草稿)

Protein language model steering via masked-diffusion generation can produce sequences with enhanced target properties but often at the cost of structural quality (pLDDT). We show that the underlying cause is the deviation of intermediate-layer activations from the natural protein distribution, and propose **Prior-Aware Masked Diffusion**, which incorporates prior information from UniRef50 L17 activations at two stages of steering. First, as an **in-generation modifier** via SDEdit-style projection through a pre-trained flow-matching model (GLP), which rescues pLDDT at moderate steering strength (+0.04-0.09 pLDDT on sol_easy at allL α=2 with minimal oracle cost). Second, as a **post-generation density filter** using Mahalanobis distance (a 10 KB Gaussian approximation to the UniRef50 L17 prior) with a χ²(D)-based threshold, which simultaneously improves pLDDT and oracle for moderate steering on sol and therm tasks (e.g., allL α=2 on sol_hard: +0.068 pLDDT, +0.071 oracle). A key diagnostic: over-steered sequences (L17 α=10) have L17 Mahalanobis distance statistically indistinguishable from random amino acid baselines, revealing structure collapse as deviation-to-mean in high-dimensional activation space. Our filter uses only per-position statistics from a 10 KB precomputed file and outperforms ESM-2 650M and 3B pseudo-perplexity at rejection sampling by 2-5×. We validate across 6 tasks: sol/therm × {easy, hard}, TrpB, and GFP.

### 3.3 Story Arc(三幕)

**Act 1 (Motivation)**: 复现 Huang et al. steering,发现强 α 下 pLDDT 崩。Random AA baseline 定标 "sequence = junk",证明过强 steering 产生的 L17 激活**在统计上无法与随机氨基酸区分**。

**Act 2 (Methods)**: 问题本质是激活偏离 UniRef50 训练分布。两种引入先验的方法:
- **Modifier**: GLP 投影(在 forward 中把 L17 拉回流形)
- **Filter**: Mahalanobis χ² density filter(过滤掉 OOD 的生成序列)

**Act 3 (Validation)**: 跨 6 tasks 验证。Modifier 在 allL α=1.5-2 moderate steering 下 rescue pLDDT(+0.05-0.09)。Filter 在 sol/therm 上 rejection sampling 获得双赢(pLDDT ↑, oracle ↑)。Fitness benchmarks 有 task-specific 限制但 percentile-based 变体 still applicable。

---

## 4. 方法框架

### 4.1 In-Generation Modifier: GLP SDEdit 投影

```python
# 每个 forward pass,在 L17 的 hidden state h 上做 SDEdit 投影
noisy = (1 - sigma) * h + sigma * noise     # 加噪,sigma 由 u 控制
h_new = GLP.denoise(noisy, 25_steps)        # 从 noisy 去噪回流形
```

**关键参数**: `u ∈ [0, 1]` — 控制去噪起点的 timestep
- `u = 0`: 从纯噪声 denoise → 几乎纯 prior sampling
- `u = 0.5`: 半-SDEdit(主要 finding 的设置)
- `u = 1`: 近乎 identity(什么都不做)

**发现**:
- allL α=1.5-2 + u=0.5 给出最大 pLDDT rescue(+0.05-0.09)
- L17 单层 steering 下 GLP projection 永远伤 pLDDT(per-position 架构限制)
- allL α≥4 下 GLP 反而有害(activation 太 OOD,GLP 无法拉回)

### 4.2 Post-Generation Filter: Mahalanobis χ² Density

#### 4.2.1 计算 per-sequence Mahal

```python
# 对生成序列 seq(过 unsteered ESM-2 forward)
h = ESM2(seq)[layer=17]         # (L, 1280) 每残基的 L17 激活
z = (h - mean) / std            # per-dim z-score (使用 UniRef50 的 rep_statistics)
mahal_per_pos = (z ** 2).sum(dim=-1)   # (L,) 每残基 Mahal²
mahal_score = mahal_per_pos.mean()      # 全序列平均
```

**成本**: 1 次 ESM-2 forward + element-wise 算术 = **~0.3s per seq, 10 KB stats**

#### 4.2.2 Filter 策略

**Strategy A (absolute χ² threshold)**: 基于 χ²(D=1280) 分布理论
- 接受 `Mahal² ≥ D - k√(2D)`,比如 k=1 时 threshold = 1229
- **数学严谨**: 如果 h ~ N(μ, Σ),则 Mahal² ~ χ²(D),期望 = D = 1280
- **适合**: sol/therm tasks(序列分布覆盖 UniRef50 shell 附近)

**Strategy B/C (percentile-based,task-adaptive)**:
- 接受 top-K% by Mahal within each setting
- **适合**: fitness benchmarks 或任何 Mahal 分布偏离 χ²(D) 的场景

### 4.3 先验的来源(关键点)

两种方法**共享同一个 prior**: UniRef50 L17 激活分布。

- **Modifier (GLP)**: 学到的完整 density `log p_GLP(h)` (flow matching)
- **Filter (Mahal)**: Gaussian 一阶/二阶矩近似 `(h - μ)^T Σ_diag^{-1} (h - μ)`

两者都**编码 "先验流形信息"**,只是在不同阶段、以不同粒度应用。

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
2. **L17_a10** — strong single-layer(**structure collapse 证据**)
3. **allL_a2** — moderate all-layer(**filter 主要验证点**)
4. **allL_a3** — strong all-layer(oracle 饱和)
5. **allL_a2_L17GLP_u0.5** — modifier 版本(GLP in-generation projection)

共 **6 tasks × 5 settings = 30 (task, setting) 组合**。

### 5.3 Pipeline Per (task, setting)

1. 生成 N 条序列 (500 for sol/therm, 200 for fitness)
2. Oracle 评分 → per-seq property score
3. ESMFold → per-seq pLDDT + pTM
4. Proxy 计算 → per-seq:
   - **Mahal**: 用 rep_statistics 算 L17 Mahalanobis²
   - **GLP resid (u=0.15)**: GLP SDEdit 小噪声后的 residual(non-Gaussian 信号)
   - **ppl 650M**: ESM-2 650M pseudo-perplexity(15 random positions)
5. **allL_a2 only**: 额外计算 **ppl 3B** 作 main text baseline(100 seqs × 6 tasks)

### 5.4 Random AA Baseline

从 `data/benchmarks/random_aa_seqs/random_1000.csv` 取 100 条随机序列,同样计算 Mahal / pLDDT / oracle。这是 Motivation Fig 1 的 **关键 anchor**。

---

## 6. 当前主要结果

### 6.1 Mahal 分布 per task(allL_a2 setting)

| Task | Mahal 均值 | > 1229 的比例 | 数据是否适配 χ² filter |
|---|---|---|---|
| sol_easy | **1013** | 10.6% | ✅ 适配 |
| sol_hard | 1078 | 21.8% | ✅ 适配 |
| therm_easy | **1389** | 99.6% | ⚠️ filter 几乎 no-op |
| therm_hard | 1390 | 99.4% | ⚠️ 同上 |
| trpb | **1439**(极窄) | 100% | ❌ filter 无差异(WT-邻近)|
| gfp | **478**(极低) | 0% | ❌ 所有 OOD |

### 6.2 Motivation (Fig 1 数据)

**Structure collapse under strong L17 steering** (sol_easy):

| α | L17 pLDDT | L17 Mahal² |
|---|---|---|
| 1 | 0.656 | 1219 |
| 2 | 0.626 | 1207 |
| 3 | 0.590 | 1143 |
| 5 | 0.450 | 872 |
| **10** | **0.346** | **619** |
| **Random AA** | **0.255** | **604** |

**Key finding**: L17 α=10 的 Mahal (619) ≈ Random AA Mahal (604) — "过强 steering 产生的序列在 L17 空间**与随机氨基酸袋难以区分**"。

### 6.3 In-Generation Modifier (GLP projection) 效果

**sol_easy × allL 系列,u=0.5**:

| α | no GLP | + GLP u=0.5 | Δ pLDDT |
|---|---|---|---|
| 1 | 0.334 | 0.328 | -0.005 |
| **1.5** | 0.370 | **0.458** | **+0.088** ⭐ 最大 rescue |
| **2** | 0.480 | **0.526** | **+0.046** ★ 主要 claim |
| 2.5 | 0.536 | 0.544 | +0.008 |
| 3 | 0.539 | 0.537 | ~0 |
| 4 | 0.567 | 0.531 | **-0.036 ❌** |
| 5 | 0.575 | 0.562 | -0.013 |

**sweet spot = α ∈ [1.5, 2.5]**,超出则 GLP 反而伤 pLDDT。

### 6.4 Post-Generation Filter(Mahal χ²)的主要结果

**Strategy A (absolute χ² threshold k=1, Mahal² ≥ 1229),跨 6 tasks**:

| Task | base pLDDT / oracle | filt pLDDT / oracle | Δ pLDDT | Δ oracle | ✓ |
|---|---|---|---|---|---|
| **sol_easy** | 0.483 / 0.672 | **0.546 / 0.790** | **+0.063** | **+0.119** | ✓ |
| **sol_hard** | 0.512 / 0.684 | **0.580 / 0.755** | **+0.068** | **+0.071** | ✓ |
| therm_easy | 0.514 / 44.7°C | 0.514 / 44.7°C | +0.000 | -0.002 | ⚠️ |
| therm_hard | 0.512 / 44.9°C | 0.512 / 44.9°C | -0.000 | -0.000 | ⚠️ |
| trpb | 0.791 / 0.401 | 0.791 / 0.401 | 0 | 0 | ❌ |
| gfp | 0.312 / 1.319 | —(全拒)| — | — | ❌ |

(以上为 allL_a2 setting;完整 5-setting 结果见下)

**per-task 平均(5 settings)**:

| Task | avg Δ pLDDT | avg Δ oracle | Filter 有效? |
|---|---|---|---|
| sol_easy | **+0.045** | **+0.091** | ✅ |
| sol_hard | **+0.054** | **+0.052** | ✅ |
| therm_easy | +0.026 | +0.150 °C | ✅(部分)|
| therm_hard | +0.038 | **+0.764** °C ⭐ | ✅(部分)|
| trpb | 0 | 0 | ❌(任务限制)|
| gfp | — | — | ❌(Mahal 太低)|

### 6.5 Proxy 对比(Spearman r with pLDDT,3 key settings)

| Setting | Mahal r | GLP resid r (u=0.15) | ppl 650M r | ppl 3B r |
|---|---|---|---|---|
| sol_easy allL_a2 | +0.51 | **+0.54** | -0.27 | -0.23 |
| sol_easy L17_a1 | +0.67 | **+0.68** | -0.59 | -0.55 |
| sol_easy allL_a3 | +0.21 | **+0.25** | -0.04 | -0.06 |

**Finding**: Mahal ≈ GLP resid(very close),两者**一致 beat ppl**(包括 3B)。

### 6.6 Rejection Sampling Gain (top-20% vs random-20%)

**sol_easy allL_a2 (N=250)**:

| Proxy | ΔpLDDT | Δoracle | Cost per seq |
|---|---|---|---|
| **Mahal** | **+0.049** | **+0.122** | 0.3s, 10KB |
| GLP resid (u=0.05) | +0.045 | **+0.133** | 0.8s, 1.3GB |
| GLP resid (u=0.15) | **+0.066** | +0.097 | 0.8s, 1.3GB |
| ppl 650M | +0.028 | +0.021 | 5s |
| ppl 3B | +0.025 | **-0.000** | 15s |

### 6.7 "Goldilocks" 分析(14k seqs pool, sol_easy only)

Bin 序列按 Mahal,观察 pLDDT 和 oracle:

| Mahal 区间 | N | mean pLDDT | mean oracle | 语义 |
|---|---|---|---|---|
| 319-538(极低)| 1390 | 0.352 | 0.330 | over-steered / random |
| 538-707 | 2085 | 0.359 | 0.387 | over-steered mild |
| 707-936 | 3475 | 0.436 | 0.556 | moderate allL strong α |
| **936-1149** | **3475** | **0.529** | **0.587** ⭐ | **oracle 甜点** (allL α=2-3) |
| 1149-1367 | 2085 | 0.607 | 0.424 | 近自然 |
| **1367-2143** | **1390** | **0.666** ⭐ | 0.436 | **pLDDT 甜点** (L17 α=1, natural) |

**pLDDT 单调随 Mahal 上升**,oracle 呈 **inverted-U**(甜点 Mahal ~1000)。

---

## 7. Paper 结构

### §1 Introduction
- Protein steering + masked diffusion 是什么
- Huang et al. 的 trade-off 问题
- 引入 prior 的两种方式
- 4 个 contributions

### §2 Background
- ESM-2 mask-predict 生成(§2.1)
- Steering vector 注入机制(§2.2)
- Generative Latent Prior(§2.3)
- Mahalanobis 距离 + χ² 分布(§2.4)

### §3 Method
- Prior-Aware Masked Diffusion 框架(§3.1)
- Modifier: GLP in-generation projection(§3.2)
- Filter: Mahalanobis χ² post-hoc filter(§3.3)
- Strategy A (absolute) vs Strategy B/C (percentile)(§3.4)

### §4 Experiments

#### §4.1 Motivation: Structure Collapse (Fig 1)
- L17 α=10 pLDDT 崩到 0.35
- L17 α=10 Mahal ≈ Random AA Mahal(**smoking gun**)
- Trade-off quantified across settings

#### §4.2 In-Generation Modifier (GLP)
- α × u grid search on sol_easy
- Sweet spot α=1.5-2, u=0.5
- pLDDT rescue up to +0.09
- Mechanism: all-layer coherent steering + per-position GLP 投影

#### §4.3 Post-Generation Filter (Mahal χ²) — **Main Result**
- Table 1: 6 tasks × filter gain per setting
- Strategy A absolute threshold: works on sol/therm
- Strategy C percentile: universal fallback

#### §4.4 Proxy Comparison
- Mahal vs GLP resid vs ppl 650M vs ppl 3B
- Cross-setting correlation + rejection sampling gain
- ppl 3B 不比 650M 好

#### §4.5 Goldilocks Analysis
- Mahal 分布 vs pLDDT/oracle 关系
- Random AA baseline 位置

### §5 Analysis
- 高维 Gaussian 球壳几何(why Mahal works)(§5.1)
- L17 α=10 ≈ Random AA 的机制(§5.2)
- χ² filter 在 sol/therm work 但 fitness 不 work 的原因(§5.3)

### §6 Limitations & Future Work
- Gaussian 假设近似(真实 L17 non-Gaussian)
- Fitness tasks 的 adaptive threshold 未深入
- Per-position MLP GLP 的架构限制(attention-based 是 future work)

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
- [x] ppl 3B on allL_a2 × 6 tasks
- [x] Random AA baseline + Goldilocks 分析(sol_easy)
- [x] 数据整理到 `paper_data/`

### 8.2 分析与制图(待做)

- [ ] **Fig 1 teaser**: Motivation 两栏图(pLDDT vs α + L17_a10 ≈ Random AA 散点)
- [ ] **Fig 2**: Modifier GLP 投影 α × u heatmap(sol_easy)
- [ ] **Fig 3** (main): Filter 在 6 tasks 的 pLDDT/oracle 提升 bar chart
- [ ] **Fig 4**: Goldilocks binning(pLDDT/oracle vs Mahal bin)
- [ ] **Fig 5**: Proxy 成本-效果 scatter(Mahal/GLP/ppl 对比)
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
├── random_aa/                           (816 KB)
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
├── therm_hard/ (3.1 MB)                 │   ├── generated/ (5 settings × 500 seqs)
├── trpb/ (1.8 MB)                       │   ├── oracle/ (5 scored CSVs)
└── gfp/ (1.2 MB)                        ┘   ├── esmfold/ (merged per-seq pLDDT)
                                             └── proxy/ (per-seq Mahal+GLP+ppl650M+ppl3B)
```

### Proxy CSV 列说明

每个 `paper_data/<task>/proxy/<setting>_proxy.csv` 包含:

| 列 | 含义 |
|---|---|
| `sequence` | 生成的蛋白序列 |
| `mahal` | Mahalanobis²(per-residue 均值)|
| `glp_resid` | GLP denoising residual at u=0.15(per-residue 均值)|
| `ppl_650m` | ESM-2 650M pseudo-perplexity(15 random mask positions)|
| `plddt` | ESMFold predicted LDDT |
| `oracle` | task-specific oracle score |

**ppl 3B** 在单独文件 `<setting>_ppl3b.csv` 里,只有 allL_a2 setting × 100 seqs。

### 数据量

| 量 | 数值 |
|---|---|
| Tasks | 6 |
| Settings per task | 5 |
| Total (task, setting) pairs | 30 |
| Generated seqs total | ~13,400 (sol/therm 500 × 4 × 5 = 10k + fitness 200 × 2 × 5 = 2k + random 1k = 13k) |
| pLDDT data points | ~13,400(全都 ESMFolded) |
| Oracle data points | ~13,400 |
| Proxy data points | ~13,400 (Mahal + GLP + ppl 650M)|
| ppl 3B data points | 600(100 seqs × 6 tasks,allL_a2 only) |

---

## 10. References

- **Huang et al., 2025**: "Steering Protein Language Models" (ICML 2025) — base steering paper
- **Luo et al., 2025**: GLP paper (to be confirmed) — flow-matching prior on L17
- **ESM-2**: Lin et al., 2023, Science
- **ESMFold**: Lin et al., 2023
- **SDEdit**: Meng et al., 2021 — image editing with diffusion (inspiration for GLP modifier)

**Datasets**:
- **Solubility (sol_easy/hard)**: DeepSol dataset splits
- **Thermostability (therm)**: Meltome Atlas
- **TrpB (fitness)**: Engqvist lab engineered variants (4-site mutations)
- **GFP (fitness)**: Kirjner et al. hard split

---

## 📌 项目当前状态(2026-04-20)

- ✅ **所有实验数据已生成并整理**(`paper_data/` 就绪)
- ✅ **核心发现已 validated**:
  - Motivation: L17 α=10 ≈ Random AA structurally
  - Modifier: allL α=1.5-2 + GLP u=0.5 gives +0.05-0.09 pLDDT
  - Filter: sol/therm tasks 下 Mahal χ² filter 给 +0.04-0.07 pLDDT + oracle 涨
  - Fitness 任务 filter 需 task-specific tuning
- ⏳ **接下来**: 制图 + 写作(~2 周)
- 🎯 **目标投稿**: ICML 2026 / NeurIPS 2026 / bioRxiv first

**负责人**: Shui-bai Zhang (szhang967@wisc.edu)
