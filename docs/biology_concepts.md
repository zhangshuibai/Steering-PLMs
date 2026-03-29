# Biology Concepts in Protein Engineering & Steering PLMs

> Sources:
> - Paper1: Yang et al., "Steering Generative Models with Experimental Data for Protein Fitness Optimization" (arXiv:2505.15093, NeurIPS 2025)
> - Paper2: Peng et al., "Planner Aware Path Learning in Diffusion Language Models Training" (OpenReview:lAlI5FuIf7)
> - Repo: Steering-PLMs codebase

---

## 一、氨基酸与蛋白质基础

### 1. Amino Acids（氨基酸）

**出现**: Paper1 | Paper2 | Repo

蛋白质的基本构建单元，共 20 种标准氨基酸（standard amino acids）：A (Ala), C (Cys), D (Asp), E (Glu), F (Phe), G (Gly), H (His), I (Ile), K (Lys), L (Leu), M (Met), N (Asn), P (Pro), Q (Gln), R (Arg), S (Ser), T (Thr), V (Val), W (Trp), Y (Tyr)。每种氨基酸具有不同的化学性质（极性 polarity、电荷 charge、疏水性 hydrophobicity 等）。蛋白质序列就是氨基酸的线性排列（linear chain），一个长度为 M 的蛋白质有 20^M 种可能的序列组合。

- Paper1 中：设计空间（design space）= 20^M，TrpB 有 15 个可变残基 → 20^15 ≈ 3.3×10^19 种可能
- Paper2 中：词表 V = 20 种氨基酸 + 特殊 token（mask, pad 等）
- Repo 中：ESM2 的 token indices 4:24 对应 20 种标准氨基酸

### 2. Residue / Residue Position（残基 / 残基位置）

**出现**: Paper1 | Paper2 | Repo

蛋白质链中的单个氨基酸称为一个 residue（残基），因为氨基酸聚合（polymerization）时脱水缩合，保留的是氨基酸的"残留部分"。残基位置（residue position）用编号标识（如 TrpB 的第 117、118、119 位）。蛋白质工程中，选择哪些残基位置进行突变（mutation site selection）是关键决策。

- Paper1：TrpB 只允许 15 个特定残基位置变化（targeted residues）；CreiLOV 和 GB1 全长可变
- Repo：`steering_esm3_optimization.py` 中用余弦相似度（cosine similarity）选择突变位点

### 3. Protein Sequence（蛋白质序列）

**出现**: Paper1 | Paper2 | Repo

氨基酸的线性排列顺序（primary structure），决定了蛋白质的三维结构（3D structure）和功能（function）。"序列 → 结构 → 功能"是分子生物学中心法则（central dogma of molecular biology）的延伸。序列长度从几十到上千个氨基酸不等。

- Paper1：TrpB = 389 AA, CreiLOV = 119 AA, GB1 = 56 AA
- Repo：处理范围 30–1022 AA（ESM2 输入限制）

### 4. Mutation / Variant（突变 / 变体）

**出现**: Paper1 | Repo

将蛋白质序列中某个位置的氨基酸替换为另一种，称为突变（mutation）。每次替换产生一个变体（variant）。突变是蛋白质工程的基本操作。

- **Point mutation（点突变）**: 改变单个残基
- **Combinatorial mutation（组合突变）**: 同时改变多个残基（Paper1 中 TrpB 同时变 15 个位点）
- **Insertions / Deletions (InDels，插入/缺失)**: 插入或删除残基，改变序列长度（Paper1 提到但未考虑）

### 5. Protein Structure（蛋白质结构）

**出现**: Paper2 | Repo (ESM3)

蛋白质折叠（protein folding）成特定的三维结构才能行使功能。蛋白质结构分为四个层级：

- **Primary structure（一级结构）**: 氨基酸序列
- **Secondary structure（二级结构, SS8）**: α-helix（α-螺旋）、β-sheet（β-折叠）等局部结构模式。ESM3 使用 8 类分类（8-class secondary structure）
- **Tertiary structure（三级结构）**: 整条多肽链的三维折叠
- **Quaternary structure（四级结构）**: 多条多肽链的组装（subunit assembly）

---

## 二、蛋白质性质与功能

### 6. Protein Fitness（蛋白质适应度）

**出现**: Paper1

对蛋白质某种所需功能的量化度量（quantitative measure of a desired property）。Fitness 是一个通用术语，可以指任何可测量的蛋白质性质：活性（activity）、稳定性（stability）、荧光（fluorescence）等。Fitness 越高 = 蛋白质在目标任务上表现越好。

- 核心挑战：设计空间巨大（20^M），但实验只能测量 10²–10³ 个序列的 fitness

### 7. Fitness Landscape（适应度景观）

**出现**: Paper1

将每个可能的蛋白质序列映射到其 fitness 值，形成高维"景观"。像一个地形图——有"山峰"（high fitness，功能最优序列）和"山谷"（low fitness，功能丧失序列）。蛋白质工程的目标就是在这个景观中找到最高峰（global optimum）。

- 景观通常是 rugged（崎岖的），存在大量 local optima（局部最优），global optimum 极难找到

### 8. Enzyme Activity（酶活性）

**出现**: Paper1

酶（enzyme）催化化学反应的能力。TrpB（tryptophan synthase β-subunit，色氨酸合成酶 β 亚基）催化色氨酸（tryptophan）的合成反应。活性通常用反应速率（reaction rate）、转化率（conversion rate）、kcat/Km 等指标衡量。

- **Non-native activities（非天然活性）**: 通过工程化使酶催化自然界中不存在的反应（Frances Arnold 的研究方向，2018 年诺贝尔化学奖）

### 9. Fluorescence（荧光）

**出现**: Paper1

某些蛋白质（如 GFP 家族，Green Fluorescent Protein）能吸收特定波长的光（excitation）并发射另一波长的光（emission）。CreiLOV 是一种 LOV 域荧光蛋白（LOV domain fluorescent protein），优化目标是增强其荧光强度（fluorescence intensity）。荧光蛋白在细胞生物学中广泛用作报告基因（reporter）和标记工具（labeling tool）。

### 10. Binding Affinity（结合亲和力）

**出现**: Paper1

蛋白质与其靶标分子（target molecule）——其他蛋白质、DNA、小分子（small molecule）等——结合的强度。通常用解离常数 Kd (dissociation constant) 衡量：Kd 越小 = 结合越强。GB1 (Protein G B1 domain) 是一个 IgG-binding 蛋白，优化目标是增强其与 IgG 抗体（immunoglobulin G）的结合能力。

### 11. Stability（稳定性）/ Thermostability（热稳定性）

**出现**: Paper1 | Repo

蛋白质维持其折叠结构（folded state）的能力：

- **Thermostability（热稳定性）**: 在高温下保持结构和功能的能力，用熔点温度 **Tm (melting temperature, °C)** 衡量。Tm 越高 = 蛋白质越耐热、越稳定
- **Thermodynamic stability（热力学稳定性）**: 折叠态（folded state）与展开态（unfolded state）的自由能差 ΔG (Gibbs free energy change)

Repo 中：Meltome Atlas 数据，Tm 范围 30–98°C，positive ≥ 70°C, negative ≤ 50°C

### 12. Solubility（溶解度）

**出现**: Repo

蛋白质在水溶液（aqueous solution）中溶解并保持可溶态（soluble state）的能力。不可溶的蛋白质会聚集（aggregate）形成包涵体（inclusion bodies），失去功能。

- Repo 中：二分类（binary classification）——soluble（可溶）≥ 0.5 vs insoluble（不可溶）< 0.5
- 数据来源：DeepSol 数据库
- 工业应用：重组蛋白表达（recombinant protein expression）中溶解度是关键瓶颈

### 13. Foldability（可折叠性）

**出现**: Paper2

蛋白质序列能否折叠成稳定的三维结构。Paper2 中定义为：**pLDDT > 70 且 pTM > 0.5** 的序列比例。

- 这是一个计算指标（computational metric），通过 ESMFold 预测结构后评估
- PAPL 方法将 foldability 从 48% 提升到 59%（相对提升 40%）

### 14. LOV Domain（LOV 结构域）

**出现**: Paper1

Light-Oxygen-Voltage domain，一类感光蛋白质结构域（photosensory protein domain）。CreiLOV 来自莱茵衣藻（*Chlamydomonas reinhardtii*）的 LOV 域蛋白，能结合 FMN（flavin mononucleotide，黄素单核苷酸）辅因子（cofactor）产生荧光。

### 15. SASA (Solvent Accessible Surface Area，溶剂可及表面积)

**出现**: Repo (ESM3)

蛋白质表面能被水分子（water molecule / solvent）接触到的面积（单位 Å²）。反映残基的暴露/埋藏程度（exposure / burial）：

- 高 SASA = 残基暴露在蛋白质表面（surface-exposed），亲水残基（hydrophilic residues）倾向于此
- 低 SASA = 残基埋在蛋白质内部（buried in the core），疏水残基（hydrophobic residues）倾向于此

ESM3 将 SASA 离散化为 token，作为多模态输入（multimodal input）之一。

---

## 三、实验蛋白质系统

### 16. TrpB (Tryptophan Synthase β-subunit，色氨酸合成酶 β 亚基)

**出现**: Paper1

| 属性 | 值 |
|------|-----|
| 长度 | 389 AA |
| 可变残基数 | 15 (targeted residues) |
| 优化目标 | Enzyme activity（酶活性） |
| MSA 大小 | 5.7 × 10⁴ |
| 训练 fitness 数 | 75,618 |
| 测试 fitness 数 | 23,313 |
| 数据来源 | Johnston et al., 2024 |

色氨酸合成酶（tryptophan synthase）是一个 αββα 四聚体（tetramer），催化吲哚（indole）和 L-丝氨酸（L-serine）缩合生成 L-色氨酸（L-tryptophan）的反应。β 亚基负责最后的缩合步骤。

### 17. CreiLOV (LOV Fluorescent Protein，LOV 荧光蛋白)

**出现**: Paper1

| 属性 | 值 |
|------|-----|
| 长度 | 119 AA |
| 可变残基数 | 119 (全长) |
| 优化目标 | Fluorescence（荧光强度） |
| MSA 大小 | 3.7 × 10⁵ |
| 训练 fitness 数 | 6,842 |
| 测试 fitness 数 | 2,401 |
| 数据来源 | Chen et al., 2023c |

### 18. GB1 (Protein G B1 Domain，蛋白G B1结构域)

**出现**: Paper1

| 属性 | 值 |
|------|-----|
| 长度 | 56 AA |
| 可变残基数 | 56 (全长) |
| 优化目标 | Binding affinity（结合亲和力） |
| MSA 大小 | 126 |
| 训练 fitness 数 | 3.9 × 10⁶ |
| 测试 fitness 数 | 9.6 × 10⁴ |
| 数据来源 | Olson et al., 2014 |

Protein G 是链球菌（*Streptococcus*）表面的免疫球蛋白结合蛋白（immunoglobulin-binding protein）。B1 domain 是其最小的独立折叠单元，常用作蛋白质工程的模型系统（model system）。

---

## 四、进化与序列分析

### 19. Multiple Sequence Alignment (MSA，多序列比对)

**出现**: Paper1

将多个同源蛋白质序列（homologous protein sequences）对齐排列，使得进化上对应的残基位置（evolutionarily corresponding positions）对齐。MSA 揭示了哪些位置保守（conserved，功能重要）、哪些位置可变（variable，可耐受突变）。

- Paper1 中：MSA 作为生成模型的训练数据，代表特定蛋白质家族的自然序列分布（natural sequence distribution）
- TrpB 的 MSA 有 5.7 万条序列

### 20. Homologous Sequences（同源序列）

**出现**: Paper1

来自共同祖先（common ancestor）的蛋白质序列。通过进化（evolution）产生分化（divergence），但共享相似的序列（sequence similarity）、结构（structural similarity）和功能（functional similarity）。MSA 就是由同源序列构建的。

- **Orthologs（直系同源）**: 物种分化产生的同源基因
- **Paralogs（旁系同源）**: 基因复制产生的同源基因

### 21. Evolutionary Likelihood（进化似然）

**出现**: Paper1

一个序列在自然蛋白质分布（natural protein distribution）中的概率。Paper1 的关键生物学洞察：

> **进化似然高的序列往往功能也好**——自然选择（natural selection）已经"筛选"过了，经过数十亿年进化存活下来的序列大概率是功能性的（functional）。

这为使用生成先验模型（generative prior）采样蛋白质提供了理论基础。

### 22. Protein Family（蛋白质家族）

**出现**: Paper1

共享共同进化起源（common evolutionary origin）、相似序列和结构的一组蛋白质。例如所有色氨酸合成酶（tryptophan synthases）构成一个蛋白质家族。Paper1 中每个蛋白质的 MSA 就代表其所属家族的序列多样性。

- 蛋白质家族通常由 Pfam、InterPro 等数据库定义和分类

### 23. Sequence Identity（序列一致性）

**出现**: Repo

两条序列中相同氨基酸位置的比例（proportion of identical residues at aligned positions）。用于衡量序列相似度（sequence similarity）：

- **90% identity**: 非常相似，CD-HIT 用此阈值做同数据集内去冗余（redundancy removal）
- **50% identity**: UniRef50 的聚类阈值（clustering threshold）
- **30% identity**: 跨数据集过滤阈值（cross-dataset filtering），Repo 中用于防止训练/测试数据泄漏（data leakage）
- **<25% identity**: 通常认为序列不再有可检测的同源性（homology），进入"暮光区"（twilight zone）

### 24. UniRef50

**出现**: Paper2 | Repo

UniProt Reference Clusters at 50% identity。将 UniProt 数据库中的蛋白质序列按 50% 序列一致性聚类，每个 cluster 选一条代表序列（representative sequence）。

- 规模：约 5800 万条序列（58M sequences）
- Repo 中：GLP 训练数据来源（4M 条用于 650M 模型，全量用于 3B 模型）
- Paper2 中：DLM-150M 在 UniRef50 上预训练

---

## 五、蛋白质工程方法

### 25. Directed Evolution（定向进化）

**出现**: Paper1

模拟自然进化（natural evolution）来工程化蛋白质的实验方法（Frances Arnold 因此获 2018 年诺贝尔化学奖）：

1. 从已知蛋白质（parent / wild-type）出发
2. 随机突变（random mutagenesis）产生变体库（variant library）
3. 筛选（screening）测量每个变体的 fitness
4. 选择最优变体（fittest variant）作为下一轮起点（parent for next round）
5. 重复以上步骤

局限：每轮通常只积累 1 个有益突变（beneficial mutation），本质上是局部搜索（local search），效率低。

### 26. Wet-lab Assays（湿实验测定）

**出现**: Paper1

在实验室中实际测量蛋白质功能的实验。与 dry-lab / in silico（干实验/计算模拟）相对。

- **Low-throughput（低通量）**: 每轮只能测 10²–10³ 条序列
- **High-throughput screening (HTS，高通量筛选)**: 自动化平台可测 10⁴–10⁶ 条序列，但成本高
- **Deep mutational scanning (DMS，深度突变扫描)**: 系统性地测试所有单点突变的 fitness
- 湿实验成本高、耗时长，是蛋白质工程的主要瓶颈（bottleneck）

### 27. Screening（筛选）

**出现**: Paper1

从大量蛋白质变体中测量和筛选出具有目标功能的变体。筛选通量（screening throughput）决定了每轮能评估多少个变体。

- **Activity-based screening（基于活性的筛选）**: 直接测量酶催化活性
- **Fluorescence-based screening（基于荧光的筛选）**: 用荧光信号作为 readout（如 FACS，flow cytometry）
- **Binding-based screening（基于结合的筛选）**: 用 display 技术（phage display, yeast display）筛选高亲和力变体

### 28. Oracle（神谕模型 / 适应度模拟器）

**出现**: Paper1 | Repo

用大量实验数据训练的计算模型（computational surrogate），模拟真实实验测量（wet-lab measurement）。在计算研究中代替昂贵的湿实验来评估序列 fitness。

- Paper1：用大量真实 fitness 数据训练 oracle（f(x) → y），然后在实验中用 oracle 代替湿实验评估生成序列
- Repo：`sol_predictor_final.pt`（solubility oracle，溶解度神谕）和 `therm_predictor_nocdhit.pt`（thermostability oracle，热稳定性神谕）
- Oracle 架构（Repo）：ESM2-650M mean-pooled features → Linear(1280,1280) → GELU → LayerNorm → Linear(1280,1) → Sigmoid/Linear

---

## 六、结构预测与评估指标

### 29. pLDDT (predicted Local Distance Difference Test，预测局部距离差异测试)

**出现**: Paper2 | Repo (ESM3)

AlphaFold / ESMFold 输出的**逐残基置信度分数**（per-residue confidence score），范围 0–100：

| 范围 | 含义 |
|------|------|
| > 90 | Very high confidence（可信的原子级精度） |
| 70–90 | Confident（主链结构可靠） |
| 50–70 | Low confidence（仅折叠拓扑可参考） |
| < 50 | Very low（可能是无序区域 disordered region） |

Paper2 中：PAPL 生成的蛋白质平均 pLDDT = 81.48（confident 级别）

### 30. pTM (predicted Template Modeling score，预测模板建模分数)

**出现**: Paper2

预测的 TM-score（Template Modeling score），衡量整体结构预测的可靠性（0–1）：

- > 0.5：整体拓扑结构（topology）可能正确
- > 0.7：高置信度的结构预测
- Paper2 中：PAPL 达到 pTM = 0.72

TM-score 本身是结构比较指标，衡量两个三维结构的相似程度，与蛋白质长度归一化。

### 31. pAE (predicted Aligned Error，预测对齐误差)

**出现**: Paper2

预测的对齐误差（单位 Å），衡量结构中任意两个残基之间相对位置预测的准确性（pairwise positional accuracy）。值越低越好。

- Paper2 中：PAPL 达到 pAE = 8.97 Å（所有模型中最低/最好）
- 对于多结构域蛋白质（multi-domain protein），pAE 可以揭示哪些结构域之间的相对位置是可靠的

### 32. ESMFold

**出现**: Paper2

Meta 开发的蛋白质结构预测工具（protein structure prediction tool），基于 ESM2 语言模型。与 AlphaFold 不同，ESMFold **不需要 MSA 输入**，直接从单条序列预测三维结构（single-sequence structure prediction），速度远快于 AlphaFold。

- Paper2 中：用 ESMFold 评估生成序列的 foldability（批量评估数千条序列）
- 输出：三维坐标 + pLDDT + pTM + pAE

---

## 七、序列处理与分析工具

### 33. CD-HIT (Cluster Database at High Identity with Tolerance)

**出现**: Repo

蛋白质 / 核酸序列聚类工具（sequence clustering tool）。将相似序列归为一组（cluster），只保留代表序列（representative sequence），用于：

- **去冗余（redundancy removal）**: 避免训练集中的高度相似序列导致过拟合（overfitting）
- **防止数据泄漏（data leakage prevention）**: 确保训练集和测试集没有高相似度序列

Repo 中的使用：
- `cd-hit -c 0.9`：90% identity 阈值做集内去冗余
- `cd-hit-2d -c 0.3`：30% identity 阈值做跨集过滤

### 34. FASTA Format（FASTA 格式）

**出现**: Repo

蛋白质 / 核酸序列的标准文本文件格式（standard sequence format）：

```
>sequence_id description
MKTVRQERLKSIVRILERSKEPVSGAQ...
```

- `>` 开头的行是序列标题（header），包含序列 ID 和描述
- 后续行是序列本身（单字母氨基酸代码）
- Repo 中：`uniref50.fasta.gz`（gzip 压缩的 FASTA 文件，5800 万条序列）

### 35. SCRATCH-1D

**出现**: Repo (DeepSol)

蛋白质序列分析工具套件（protein sequence analysis toolkit），可预测：
- Secondary structure（二级结构）
- Solvent accessibility（溶剂可及性）
- Disordered regions（无序区域）

DeepSol 管线（pipeline）中用于提取蛋白质的生物特征（biological features）作为溶解度预测的输入。

---

## 八、蛋白质生成与评估指标

### 36. Pseudo-Perplexity (pPPL，伪困惑度)

**出现**: Repo

衡量生成蛋白质序列"自然度"（naturalness）的指标。使用预训练 PLM 逐个掩码（mask）每个位置，计算模型恢复原始氨基酸的概率：

$$\text{pPPL} = \exp\left(-\frac{1}{L} \sum_{i=1}^{L} \log P(x_i \mid x_{-i})\right)$$

- 越低 = 序列越像自然蛋白质（模型越"不惊讶"，即序列符合进化先验）
- 越高 = 序列越不自然（可能无法折叠 / 无功能）
- Repo 中用 ESM2-650M 或 ESM2-3B 计算
- 与 NLP 中 perplexity 的区别：蛋白质用的是双向上下文（bidirectional context），而非自回归（autoregressive）

### 37. Token Entropy（token 熵）

**出现**: Paper2

衡量生成序列中氨基酸组成多样性（amino acid composition diversity）的指标。

- 高熵（high entropy）= 氨基酸分布均匀，序列使用了多种氨基酸
- 低熵（low entropy）= 某些氨基酸被过度使用，序列组成单调
- Paper2 中 PAPL 的 entropy = 3.12（接近自然蛋白质水平，最大理论值 log₂(20) ≈ 4.32）

### 38. Sequence Diversity（序列多样性）

**出现**: Paper1 | Paper2

生成序列之间的差异程度。通常用 pairwise sequence identity（两两序列一致性）的补数衡量：diversity = 1 − mean(pairwise identity)。

- 高多样性（> 90%）表明模型没有 mode collapse（模式坍塌，即只生成少数几种序列）
- Paper2 中：PAPL 的 diversity = 91.73%
- Paper1 中：steering 强度越大（guidance strength↑），多样性越低 → fitness vs diversity trade-off

### 39. Masking Strategy（掩码策略）

**出现**: Paper1 | Paper2 | Repo

在蛋白质序列中随机选择位置替换为 [MASK] token，让模型预测被掩盖的氨基酸。这是 BERT-style masked language model（如 ESM2）和 masked diffusion language model（MDLM）的核心训练/推理方式。

- Repo 中：mask_ratio = 0.1，10 轮迭代（iterative rounds）覆盖全序列
- Paper2 中：MDLM 用 absorbing state masking 作为前向扩散（forward diffusion）的噪声方式
- Paper1 中：MDLM 是最强的 discrete diffusion 类型之一

### 40. Mean Pooling（均值池化）

**出现**: Repo

将蛋白质中每个氨基酸的向量表示（token-level representation）取平均，得到固定长度的**序列级表示**（sequence-level representation）。去除 BOS/EOS 等特殊 token 后对所有残基表示取平均。

- Repo 中：ESM2 提取的 1280 维（650M）或 2560 维（3B）表示
- 用于训练 downstream property predictor（下游性质预测器）
- 替代方案：CLS token representation、attention-weighted pooling 等

---

## 九、预训练蛋白质模型

### 41. ESM2 (Evolutionary Scale Modeling 2)

**出现**: Paper1 | Paper2 | Repo

Meta AI 开发的蛋白质掩码语言模型（masked language model），在 UniRef 数据库上预训练。ESM 系列是目前最广泛使用的蛋白质语言模型之一。

| 版本 | 参数量 | 层数 | 隐藏维度 |
|------|--------|------|----------|
| ESM2-150M | 150M | 30 | 640 |
| ESM2-650M | 650M | 33 | 1280 |
| ESM2-3B | 2.84B | 36 | 2560 |

- Repo 中：650M 是主力模型（特征提取、steering 向量计算、pPPL 评估）；3B 是大规模版本
- Paper1 中：ESM embeddings 用于 continuous diffusion 中的 latent space
- Paper2 中：ESM3 作为 baseline

### 42. ESM3

**出现**: Repo

Meta 的下一代**多模态蛋白质模型**（multimodal protein model），同时处理多种蛋白质信息：

| 模态 | Token 类型 | 说明 |
|------|-----------|------|
| Sequence | amino acid tokens | 序列信息 |
| Structure | coordinate tokens | 三维结构坐标 (B, L, 3, 3) |
| Secondary Structure | SS8 tokens | 8 类二级结构 |
| SASA | SASA tokens | 溶剂可及表面积 |
| Function | InterPro tokens | 功能域注释 |
| pLDDT | confidence tokens | 结构置信度 |

Repo 中用于多模态 steering 和 sequence optimization。

### 43. ProGen2

**出现**: Paper1 | Paper2

Salesforce 开发的**自回归蛋白质语言模型**（autoregressive protein language model），从左到右逐个生成氨基酸。

- Paper1 中：ProGen2-small (151M params) 作为 ARLM + DPO 的基础模型
- Paper2 中：ProGen2-medium / large 作为 baseline（foldability 12.75% / 11.87%）

### 44. EvoDiff

**出现**: Paper1 | Paper2

微软开发的**蛋白质序列扩散模型**（protein sequence diffusion model），在进化数据上预训练。

- Paper1 中：用 EvoDiff 38M-Uniform 的预训练权重初始化 D3PM 模型
- Paper2 中：作为 baseline（foldability 仅 0.43%，表现最差）

### 45. DPLM (Diffusion Protein Language Model，扩散蛋白质语言模型)

**出现**: Paper1 | Paper2

基于掩码扩散（masked diffusion）的蛋白质语言模型。

- Paper2 中：DPLM-650M 是最强 baseline 之一（foldability 49.14%）
- Paper1 中：提到可以 finetune DPLM 作为 MDLM 使用

### 46. ProLLaMA

**出现**: Repo

基于 LLaMA 架构的蛋白质语言模型（protein LLM），将蛋白质序列视为"蛋白质语言"进行自回归生成。Repo 中用于提取 head-wise 和 MLP-wise 的 steering 向量。

---

## 十、数据库与数据来源

### 47. Meltome Atlas

**出现**: Repo

大规模蛋白质热稳定性实验数据库，通过 **Thermal Proteomics Profiling (TPP，热蛋白质组学分析)** 技术测量了多个物种中蛋白质的熔点温度 Tm。TPP 通过逐步升温并检测蛋白质沉淀来测定每个蛋白质的 Tm。

- 数据包含：蛋白质序列、Tm (°C)、物种（organism）、测量统计
- Repo 中用于训练热稳定性 oracle

### 48. DeepSol

**出现**: Repo

蛋白质溶解度预测数据库与工具，包含 62,478 条训练序列，带有二分类标签（soluble / insoluble）和生物特征（biological features）。

- 来源于大肠杆菌（*E. coli*）重组表达的实验数据
- 包含序列特征 + SCRATCH-1D 预测的结构特征

### 49. InterPro

**出现**: Repo (ESM3)

蛋白质功能域数据库（protein domain database），整合了多个蛋白质家族/功能域数据库：

- **Pfam**: 蛋白质家族的隐马尔可夫模型（HMM）
- **PROSITE**: 蛋白质功能位点的序列模式（motifs）
- **PRINTS**: 蛋白质指纹图谱
- 其他多个成员数据库

ESM3 使用 InterPro 注释作为功能 token 输入，实现功能条件生成（function-conditioned generation）。

---

## 十一、概念交叉总结表

| # | 概念 | 英文 | Paper1 | Paper2 | Repo |
|---|------|------|:------:|:------:|:----:|
| 1 | 氨基酸 | Amino Acids | ✓ | ✓ | ✓ |
| 2 | 残基/残基位置 | Residue / Residue Position | ✓ | ✓ | ✓ |
| 3 | 蛋白质序列 | Protein Sequence | ✓ | ✓ | ✓ |
| 4 | 突变/变体 | Mutation / Variant | ✓ | | ✓ |
| 5 | 蛋白质结构 | Protein Structure | | ✓ | ✓ |
| 6 | 蛋白质适应度 | Protein Fitness | ✓ | | |
| 7 | 适应度景观 | Fitness Landscape | ✓ | | |
| 8 | 酶活性 | Enzyme Activity | ✓ | | |
| 9 | 荧光 | Fluorescence | ✓ | | |
| 10 | 结合亲和力 | Binding Affinity | ✓ | | |
| 11 | 稳定性/热稳定性 | Stability / Thermostability | ✓ | | ✓ |
| 12 | 溶解度 | Solubility | | | ✓ |
| 13 | 可折叠性 | Foldability | | ✓ | |
| 14 | LOV 结构域 | LOV Domain | ✓ | | |
| 15 | 溶剂可及表面积 | SASA | | | ✓ |
| 16 | TrpB | Tryptophan Synthase β-subunit | ✓ | | |
| 17 | CreiLOV | LOV Fluorescent Protein | ✓ | | |
| 18 | GB1 | Protein G B1 Domain | ✓ | | |
| 19 | 多序列比对 | MSA | ✓ | | |
| 20 | 同源序列 | Homologous Sequences | ✓ | | |
| 21 | 进化似然 | Evolutionary Likelihood | ✓ | | |
| 22 | 蛋白质家族 | Protein Family | ✓ | | |
| 23 | 序列一致性 | Sequence Identity | | | ✓ |
| 24 | UniRef50 | UniRef50 | | ✓ | ✓ |
| 25 | 定向进化 | Directed Evolution | ✓ | | |
| 26 | 湿实验测定 | Wet-lab Assays | ✓ | | |
| 27 | 筛选 | Screening | ✓ | | |
| 28 | 神谕模型 | Oracle | ✓ | | ✓ |
| 29 | pLDDT | predicted Local Distance Difference Test | | ✓ | ✓ |
| 30 | pTM | predicted Template Modeling score | | ✓ | |
| 31 | pAE | predicted Aligned Error | | ✓ | |
| 32 | ESMFold | ESMFold | | ✓ | |
| 33 | CD-HIT | Cluster Database at High Identity with Tolerance | | | ✓ |
| 34 | FASTA 格式 | FASTA Format | | | ✓ |
| 35 | SCRATCH-1D | SCRATCH-1D | | | ✓ |
| 36 | 伪困惑度 | Pseudo-Perplexity (pPPL) | | | ✓ |
| 37 | token 熵 | Token Entropy | | ✓ | |
| 38 | 序列多样性 | Sequence Diversity | ✓ | ✓ | |
| 39 | 掩码策略 | Masking Strategy | ✓ | ✓ | ✓ |
| 40 | 均值池化 | Mean Pooling | | | ✓ |
| 41 | ESM2 | Evolutionary Scale Modeling 2 | ✓ | ✓ | ✓ |
| 42 | ESM3 | ESM3 | | ✓ | ✓ |
| 43 | ProGen2 | ProGen2 | ✓ | ✓ | |
| 44 | EvoDiff | EvoDiff | ✓ | ✓ | |
| 45 | DPLM | Diffusion Protein Language Model | ✓ | ✓ | |
| 46 | ProLLaMA | ProLLaMA | | | ✓ |
| 47 | Meltome Atlas | Meltome Atlas | | | ✓ |
| 48 | DeepSol | DeepSol | | | ✓ |
| 49 | InterPro | InterPro | | | ✓ |
