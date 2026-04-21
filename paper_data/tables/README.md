# Tables

Paper 表,CSV + LaTeX 两版。

## 计划中的表

| Table | 内容 | 来源 |
|---|---|---|
| 1 (main) | χ² filter gain, 4 sol/therm splits, task-optimal settings | `analysis/05_filter_task_optimal.py` |
| 2 | Proxy cost benchmark(wall-clock + memory)| 待实测 |
| 3 (appendix) | 全 30 (task, setting) matrix | `analysis/05_filter_task_optimal.py` with full output |

每个 table 导出 `.csv`(机器可读)+ `.tex`(LaTeX fragment,paper 用)。
