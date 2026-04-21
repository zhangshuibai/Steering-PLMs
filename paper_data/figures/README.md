# Figures

Paper 图,按 Main text + Appendix 分类。

## 计划中的图

| Fig | 内容 | 输入数据 | 生成脚本(待写)|
|---|---|---|---|
| 1 | Motivation: pLDDT vs α + L17_a10 ≈ Random AA | `goldilocks_sol_easy/`, `random_aa/` | `make_fig1_motivation.py` |
| 2 | Main: Filter gain 在 4 splits | `<task>/master.csv` × 4 | `make_fig2_main_result.py` |
| 3 | Filter vs ppl 成本-效果 | Table 1 + cost benchmark | `make_fig3_cost_effectiveness.py` |
| 4 | Mahal vs GLP resid correlation | `master.csv` pooled | `make_fig4_mahal_vs_glp.py` |
| 5 | Goldilocks bin | `goldilocks_sol_easy/goldilocks_bins.csv` | `make_fig5_goldilocks.py` |

制图脚本建议放 `paper_data/analysis/` 下,输出 `.pdf` / `.png` 到本目录。
