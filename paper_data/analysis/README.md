# Analysis Scripts

所有分析脚本。按 `01_`, `02_`, ... 编号表示**建议运行顺序**。

## 环境要求

Python 3.10 + conda env `steering`(ESM-2 + GLP)和 `esmfold`(ESMFold)。从 repo 根目录运行所有脚本:

```bash
cd /data/szhang967/Steering-PLMs
python paper_data/analysis/<script>.py
```

## 脚本清单

### 数据产生阶段(per-sequence proxy)

| 脚本 | 作用 | 输入 | 输出 |
|---|---|---|---|
| `01_compute_proxies.py` | 每条序列的 Mahal + GLP resid (u=0.15) + ppl 650M | generated + oracle + ESMFold CSVs | `<task>/proxy/<setting>_proxy.csv` |
| `02_compute_ppl_3b.py` | 每条序列的 ESM-2 3B pseudo-perplexity(100 seqs subset, main-text baseline) | generated CSVs | `<task>/proxy/<setting>_ppl3b.csv` |

### 分析阶段(聚合 + 统计)

| 脚本 | 作用 | 输出 |
|---|---|---|
| `03_goldilocks_analysis.py` | 把 14k+ sol_easy 序列按 Mahal 分 bin,观察 pLDDT/oracle 分布 | `goldilocks_sol_easy/{all,settings,bins}.csv` |
| `04_filter_chisq.py` | χ² filter Strategy A 的 k-sweep + threshold 分析 | stdout |
| `05_filter_task_optimal.py` | Table 1 主结果: 每 task 最适合 filter 的 setting 下的 gain | stdout |
| `06_filter_percentile.py` | Strategy B/C 百分位-based filter,适应 task-specific Mahal 分布(for fitness tasks) | stdout |
| `07_rejection_sampling.py` | Top-K by proxy (Mahal/GLP/ppl 650M/ppl 3B) rejection sampling 效果对比 | stdout |
| `08_random_aa_baseline.py` | Random AA motivation baseline (§4.1):Mahal ≈ 604, pLDDT ≈ 0.26 | stdout + `random_aa/proxy.csv` |
| `09_proxy_u_sweep.py` | GLP resid 在不同 u 值下的 correlation,找最佳 u | stdout |
| `10_filter_within_setting.py` | Per-setting within-setting filter 分析(strict winners 找寻) | stdout |

## 建议运行顺序

**重复已发表结果 (Reproducibility)**:
```bash
# Stage 1: per-seq proxies (若 paper_data/<task>/proxy/ 未就绪)
python paper_data/analysis/01_compute_proxies.py --split sol_easy --settings "L17_a1 L17_a10 allL_a2 allL_a3 allL_a2_L17GLP_u0.5"
# ...(同样跑 sol_hard, therm_easy, therm_hard, trpb, gfp)

python paper_data/analysis/02_compute_ppl_3b.py --settings "allL_a2" --tasks "sol_easy sol_hard therm_easy therm_hard trpb gfp"

# Stage 2: 出 Table 1 main result
python paper_data/analysis/05_filter_task_optimal.py

# Stage 3: 出 baseline 和 Motivation
python paper_data/analysis/08_random_aa_baseline.py
python paper_data/analysis/07_rejection_sampling.py

# Stage 4: Appendix / ablations
python paper_data/analysis/03_goldilocks_analysis.py
python paper_data/analysis/04_filter_chisq.py
python paper_data/analysis/06_filter_percentile.py
python paper_data/analysis/09_proxy_u_sweep.py
python paper_data/analysis/10_filter_within_setting.py
```

## 依赖的 external 文件

- `utils/esm2_utils.py` — ESM-2 加载
- `scripts/glp_deviation/generate_alpha.py` — GLP projection function
- `generative_latent_prior/runs/glp-esm2-650m-layer17-d6/rep_statistics.pt` — UniRef50 L17 stats
- `generative_latent_prior/runs/glp-esm2-650m-layer17-d6/` — GLP checkpoint

## 注意事项

- 部分脚本硬编码了 `/tmp/*.csv` 路径(如 `goldilocks_analysis.py` 的 output path),因为原始是开发阶段的临时脚本。最终 paper-ready 版会改写为 `paper_data/` 相对路径。
- 某些脚本调用函数时需要 `paper_data/<task>/proxy/` 目录已存在,运行前请确认。
- ESMFold 在另一个 conda env (`esmfold`),使用 `generate_alpha.py` 等需要 `steering` env。
