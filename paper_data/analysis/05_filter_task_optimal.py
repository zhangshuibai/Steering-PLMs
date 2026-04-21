"""Analyze whether prior-based filtering works across all 6 tasks.
For each (task, setting): apply χ² Mahal filter (k=1, Mahal² ≥ D - √(2D)) and
compute pLDDT + oracle improvement vs all-steered baseline.
"""
import pandas as pd, numpy as np, os
from scipy.stats import spearmanr

D = 1280
sqrt2D = np.sqrt(2 * D)
THRESHOLD = D - 1 * sqrt2D  # k=1 → Mahal² ≥ 1229

TASKS = ['sol_easy', 'sol_hard', 'therm_easy', 'therm_hard', 'trpb', 'gfp']
SETTINGS = ['L17_a1', 'L17_a10', 'allL_a2', 'allL_a3', 'allL_a2_L17GLP_u0.5']

print(f'{"Task":<12} {"Setting":<25} {"N":>4} {"base_pLDDT":>10} {"base_oracle":>11} {"filt_N":>6} {"rate%":>6} {"filt_pLDDT":>10} {"ΔpLDDT":>8} {"filt_oracle":>11} {"Δoracle":>8} {"✓":>2}')
print('-' * 140)

winners_count = {task: 0 for task in TASKS}
tasks_proxy_corr = {}

for task in TASKS:
    task_corrs = []
    for setting in SETTINGS:
        proxy_csv = f'paper_data/{task}/proxy/{setting}_proxy.csv'
        if not os.path.exists(proxy_csv):
            print(f'{task:<12} {setting:<25}  (no file)')
            continue
        df = pd.read_csv(proxy_csv).dropna(subset=['mahal', 'plddt'])
        if len(df) < 50:
            continue
        base_p = df['plddt'].mean()
        # Oracle may have NaN for some settings
        orc_df = df[df['oracle'].notna()]
        base_o = orc_df['oracle'].mean() if len(orc_df) > 0 else np.nan

        # Apply χ² filter
        mask = df['mahal'] >= THRESHOLD
        filt = df[mask]
        n = len(filt)
        if n < 5:
            mark = '❌ too few'
            print(f'{task:<12} {setting:<25} {len(df):>4} {base_p:>10.3f} {base_o:>11.3f} {n:>6} {100*n/len(df):>5.1f}% {"-":>10} {"-":>8} {"-":>11} {"-":>8}  {mark}')
            continue
        rate = 100 * n / len(df)
        filt_p = filt['plddt'].mean()
        filt_o = filt[filt['oracle'].notna()]['oracle'].mean() if filt['oracle'].notna().any() else np.nan
        dp = filt_p - base_p
        do = filt_o - base_o if not (np.isnan(filt_o) or np.isnan(base_o)) else np.nan

        # Mark if "win" (pLDDT up AND oracle not too much down)
        win = dp > 0 and (np.isnan(do) or do > -0.05)
        mark = '✓' if win else ' '
        if win:
            winners_count[task] += 1

        do_str = f'{do:+.3f}' if not np.isnan(do) else '   —  '
        filt_o_str = f'{filt_o:.3f}' if not np.isnan(filt_o) else '  —  '
        base_o_str = f'{base_o:.3f}' if not np.isnan(base_o) else '  —  '
        print(f'{task:<12} {setting:<25} {len(df):>4} {base_p:>10.3f} {base_o_str:>11} {n:>6} {rate:>5.1f}% {filt_p:>10.3f} {dp:>+8.3f} {filt_o_str:>11} {do_str:>8}  {mark}')

        # Also compute Spearman correlations
        r_mahal_p = spearmanr(df['mahal'], df['plddt']).correlation
        r_glp_p = spearmanr(df['glp_resid'], df['plddt']).correlation
        r_ppl_p = spearmanr(df['ppl_650m'], df['plddt']).correlation if df['ppl_650m'].notna().any() else np.nan
        task_corrs.append({
            'setting': setting,
            'r_mahal': r_mahal_p,
            'r_glp': r_glp_p,
            'r_ppl': r_ppl_p,
        })
    tasks_proxy_corr[task] = task_corrs
    print()

print('\n' + '=' * 80)
print('SUMMARY: Filter wins (ΔpLDDT > 0 AND Δoracle > -0.05) per task')
print('=' * 80)
for task in TASKS:
    print(f'  {task:<12}: {winners_count[task]}/5 settings show clean filter wins')

print('\n' + '=' * 80)
print('SUMMARY: Spearman r(proxy, pLDDT) averaged across settings per task')
print('=' * 80)
print(f'{"Task":<12} {"r(Mahal)":>10} {"r(GLP)":>10} {"r(ppl)":>10} {"Mahal wins ppl?":>17}')
print('-' * 65)
for task in TASKS:
    corrs = tasks_proxy_corr.get(task, [])
    if not corrs: continue
    r_m = np.nanmean([c['r_mahal'] for c in corrs])
    r_g = np.nanmean([c['r_glp'] for c in corrs])
    r_p = np.nanmean([c['r_ppl'] for c in corrs])
    win = 'Yes ✓' if abs(r_m) > abs(r_p) else 'No ✗'
    print(f'{task:<12} {r_m:>+10.3f} {r_g:>+10.3f} {r_p:>+10.3f} {win:>17}')
