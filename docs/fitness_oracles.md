# Fitness Oracles

本文档说明本仓库新增的 3 个 wet-lab fitness oracle 是如何训练、使用和验证的：

- `GFP Sarkisyan`
- `CreiLOV`
- `TrpB`

说明：raw benchmark、processed CSV 和 ESM2 feature cache 都比较大，默认按本地产物管理，不直接提交到仓库；已训练好的 oracle 权重、配置和测试预测结果会保留在仓库中。

## 1. 文件位置

- 数据准备脚本：
  - [`scripts/benchmarks/prepare_gfp_sarkisyan.py`](../scripts/benchmarks/prepare_gfp_sarkisyan.py)
  - [`scripts/benchmarks/prepare_creilov.py`](../scripts/benchmarks/prepare_creilov.py)
  - [`scripts/benchmarks/prepare_trpb.py`](../scripts/benchmarks/prepare_trpb.py)
- 通用训练/评估代码：
  - [`evaluation/oracle/fitness/common.py`](../evaluation/oracle/fitness/common.py)
  - [`evaluation/oracle/fitness/train_fitness_predictor.py`](../evaluation/oracle/fitness/train_fitness_predictor.py)
  - [`evaluation/oracle/fitness/evaluate_fitness_oracle.py`](../evaluation/oracle/fitness/evaluate_fitness_oracle.py)
- 已训练好的权重：
  - [`evaluation/oracle/gfp_sarkisyan/`](../evaluation/oracle/gfp_sarkisyan/)
  - [`evaluation/oracle/creilov/`](../evaluation/oracle/creilov/)
  - [`evaluation/oracle/trpb/`](../evaluation/oracle/trpb/)
- ESM2 特征缓存：
  - `saved_predictors/gfp_sarkisyan_features/`
  - `saved_predictors/creilov_features/`
  - `saved_predictors/trpb_features/`

## 2. 训练范式

三者使用同一套训练范式：

- embedding model：冻结的 `ESM2-650M` (`esm2_t33_650M_UR50D`)
- 序列表征：取最后 1 层 token embedding，对去掉 BOS/EOS 的 token 做 mean pooling
- 回归头：`lm_head`，即 `Linear -> GELU -> LayerNorm -> Linear`
- label transform：训练前对 `fitness` 做 `z-score`
- loss：`MSELoss`
- optimizer：`AdamW(lr=1e-4, weight_decay=1e-2)`
- early stopping：按验证集 `Spearman rho` 选 best checkpoint，`patience=20`，最多 `200` epoch
- batch size：
  - feature extraction：`8`（TrpB 用 `4`）
  - head training：`256`

注意：ESM2 只用于提特征，不做 finetune。真正训练的参数只有回归头。

## 3. 数据处理

所有数据在进入 ESM2 前，都被还原为全长蛋白序列，并统一保存为：

`sequence, fitness, split, dataset, wt_sequence, num_mutations, mutated_positions`

### 3.1 GFP Sarkisyan

- raw 文件：
  - `data/benchmarks/raw/gfp_sarkisyan/amino_acid_genotypes_to_brightness.tsv`
  - `data/benchmarks/raw/gfp_sarkisyan/avGFP_reference_sequence.fa`
- 原始表给的是 `aaMutations`，例如 `SA108D:SN144D`
- 处理方式：从 avGFP WT 全长序列出发，将 mutation token 打回全长 mutant sequence
- 训练标签：`medianBrightness`
- 额外处理：去掉包含 stop 的序列；按 `sequence` 去重；用 mutation-position-group split 划分 train/val/test

### 3.2 CreiLOV

- raw 文件：
  - `data/benchmarks/raw/creilov/sb2c00662_si_001.xlsx`
  - `data/benchmarks/raw/creilov/sb2c00662_si_002.xlsx`
- 实际训练使用 `si_002`，因为它是组合库主表；`si_001` 主要是单突变表
- 原始表第一列是 HGVS 风格变体，例如 `p.Arg5Asp, p.Thr7Ser`
- 处理方式：从 119 aa 的 CreiLOV WT 全长序列出发，逐个 token 还原 full-length mutant sequence
- 训练标签：默认取 `mean_log`
- 额外处理：去掉 stop；按 `sequence` 去重；用 mutation-position-group split 划分

### 3.3 TrpB

- raw 文件：`data/benchmarks/raw/trpb/data.zip`
- 实际使用的成员：
  - `data/figure_data/4-site_merged_replicates/20230827/four-site_simplified_AA_data.csv`
  - `data/ftmlde_data/tm9d8s_AAs.fasta`
- 原始表中的 `AAs` 不是全长序列，而是 4 个可变位点 `183, 184, 227, 228` 的氨基酸组合
- 处理方式：将这 4-mer 打回 `Tm9D8s` WT 全长序列，得到 full-length mutant sequence
- 训练标签：`fitness`
- 额外处理：去掉 stop；按 `sequence` 去重；用 mutation-position-group split 划分

重要 caveat：当前训练的是 Johnston release 里的 **4-site TrpB 子集**，不是文档中提到的 full 15-site TrpB benchmark。如果后续要对齐论文级 benchmark，需要重新定义 TrpB 数据接入方式。

## 4. 训练命令

先准备数据：

```bash
python scripts/benchmarks/prepare_gfp_sarkisyan.py
python scripts/benchmarks/prepare_creilov.py
python scripts/benchmarks/prepare_trpb.py
```

再训练 oracle：

```bash
python evaluation/oracle/gfp_sarkisyan/train_gfp_sarkisyan_predictor.py
python evaluation/oracle/creilov/train_creilov_predictor.py
python evaluation/oracle/trpb/train_trpb_predictor.py
```

也可以直接跑总 pipeline：

```bash
bash scripts/pipelines/run_fitness_oracles.sh
```

## 5. 如何使用

对任意带 `sequence` 列的 CSV 打分：

```bash
python evaluation/oracle/fitness/evaluate_fitness_oracle.py \
  --input_csv your_sequences.csv \
  --predictor_path evaluation/oracle/creilov/creilov_predictor_final.pt \
  --sequence_col sequence \
  --label_col fitness
```

如果传了 `--label_col`，脚本会同时输出 `MAE / RMSE / R2 / Pearson / Spearman / top-k enrichment`。

## 6. 训练效果

### GFP Sarkisyan

- validation `Spearman = 0.7649`
- test `Spearman = 0.7702`
- test `Pearson = 0.8068`
- test `top-5% enrichment = 3.93x`

### CreiLOV

- validation `Spearman = 0.9782`
- test `Spearman = 0.9771`
- test `Pearson = 0.9806`
- test `top-5% enrichment = 14.12x`

### TrpB

- validation `Spearman = 0.3435`
- test `Spearman = 0.4031`
- test `Pearson = 0.8215`
- test `top-5% enrichment = 15.01x`

## 7. 结果解读与 sanity checks

- 三个模型保存后的 checkpoint 已重新加载验证过，重算出的 test metrics 与配置文件中的记录一致。
- 训练代码会显式检查 `train/val/test` 之间是否有完全相同的 `sequence` overlap。
- `CreiLOV` 的分数显著高于最初预期，当前更合理的解释是：它是一个仅覆盖 15 个可变位点、每个位点可选氨基酸很少的组合库，因此任务本身比全长自由突变更受限。
- `TrpB` 没有达到最初设想的 `Spearman >= 0.65`。当前结果更像是“高值筛选可用，但全局排序一般”。同时要注意：现在的 TrpB split 只建立在 4-site 子库上，test 仅包含很少的 mutation-position groups，因此这个结果不能直接当作 full TrpB benchmark 的结论。
