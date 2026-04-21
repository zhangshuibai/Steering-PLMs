"""Re-analyze with percentile-based Mahal filter (adaptive per task).
Also correctly merge oracle for fitness tasks.
Compare: χ² absolute threshold vs percentile-based.
"""
import pandas as pd, numpy as np, os
from scipy.stats import spearmanr

TASKS = ['sol_easy', 'sol_hard', 'therm_easy', 'therm_hard', 'trpb', 'gfp']
SETTINGS = ['L17_a1', 'L17_a10', 'allL_a2', 'allL_a3', 'allL_a2_L17GLP_u0.5']

def load_task_setting(task, setting):
    f = f'paper_data/{task}/proxy/{setting}_proxy.csv'
    if not os.path.exists(f): return None
    df = pd.read_csv(f).dropna(subset=['mahal', 'plddt'])
    # Re-merge oracle for fitness tasks where column name differs
    if df['oracle'].isna().all():
        sf = f'paper_data/{task}/oracle/{setting}_scored.csv'
        if os.path.exists(sf):
            scored = pd.read_csv(sf)
            col = None
            for c in ['pred_prob', 'pred_tm', 'lookup_fitness', 'oracle_fitness', 'pred_score']:
                if c in scored.columns: col = c; break
            if col:
                s2o = dict(zip(scored['sequence'], scored[col]))
                df['oracle'] = df['sequence'].map(s2o)
    return df

def run_filter(df, mask_fn, label):
    acc = df[mask_fn(df)]
    if len(acc) < 5: return None
    base_p = df['plddt'].mean()
    base_o = df[df['oracle'].notna()]['oracle'].mean()
    p = acc['plddt'].mean()
    o = acc[acc['oracle'].notna()]['oracle'].mean() if acc['oracle'].notna().any() else np.nan
    return {
        'N': len(df), 'filt_N': len(acc), 'rate': 100*len(acc)/len(df),
        'base_p': base_p, 'base_o': base_o,
        'filt_p': p, 'filt_o': o,
        'dp': p - base_p,
        'do': o - base_o if not (np.isnan(o) or np.isnan(base_o)) else np.nan,
    }

print('=== Filter Strategy Comparison ===\n')
print('Strategy A: χ² absolute threshold Mahal² ≥ D - √(2D) = 1229')
print('Strategy B: top 80% by Mahal within each setting (percentile-based)')
print('Strategy C: top 50% by Mahal within each setting')
print()
print(f'{"Task":<12} {"Setting":<25} {"base p/o":>14} {"A: p/o ↑":>16} {"B80: p/o ↑":>16} {"C50: p/o ↑":>16}')
print('-' * 105)

summary = {task: {'A': [], 'B': [], 'C': []} for task in TASKS}
for task in TASKS:
    for setting in SETTINGS:
        df = load_task_setting(task, setting)
        if df is None or len(df) < 50: continue
        base_p = df['plddt'].mean()
        base_o = df[df['oracle'].notna()]['oracle'].mean()
        base_str = f'{base_p:.3f}/{base_o:.3f}' if not np.isnan(base_o) else f'{base_p:.3f}/—'

        rA = run_filter(df, lambda d: d['mahal'] >= 1229, 'A')
        q_80 = df['mahal'].quantile(0.20)  # top 80% = above 20th percentile
        q_50 = df['mahal'].quantile(0.50)
        rB = run_filter(df, lambda d, q=q_80: d['mahal'] >= q, 'B')
        rC = run_filter(df, lambda d, q=q_50: d['mahal'] >= q, 'C')

        def fmt(r):
            if r is None: return f'{"(few)":>16}'
            do = f'{r["do"]:+.3f}' if not np.isnan(r['do']) else '  —  '
            return f'{r["dp"]:>+6.3f}/{do:>7}'

        print(f'{task:<12} {setting:<25} {base_str:>14} {fmt(rA):>16} {fmt(rB):>16} {fmt(rC):>16}')

        for lbl, r in [('A', rA), ('B', rB), ('C', rC)]:
            if r is not None:
                summary[task][lbl].append((r['dp'], r['do'] if not np.isnan(r['do']) else 0))
    print()

print('\n=== PER-TASK SUMMARY: avg Δ across 5 settings ===')
print(f'{"Task":<12} | {"A (abs χ²)":>20} | {"B (top 80%)":>20} | {"C (top 50%)":>20}')
print(f'{"":12} | {"ΔpLDDT":>8} {"Δoracle":>10} | {"ΔpLDDT":>8} {"Δoracle":>10} | {"ΔpLDDT":>8} {"Δoracle":>10}')
print('-' * 90)
for task in TASKS:
    row = [task]
    for lbl in ['A', 'B', 'C']:
        items = summary[task][lbl]
        if not items:
            row.append(f'{"—":>8} {"—":>10}')
        else:
            avg_dp = np.mean([x[0] for x in items])
            avg_do = np.mean([x[1] for x in items])
            row.append(f'{avg_dp:>+8.3f} {avg_do:>+10.3f}')
    print(f'{task:<12} | {row[1]:>20} | {row[2]:>20} | {row[3]:>20}')
