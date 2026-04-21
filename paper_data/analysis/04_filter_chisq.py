"""Test χ²-style Mahal filter at different k values.
Two policies:
  A) Two-sided: accept if Mahal² ∈ [D - k√(2D), D + k√(2D)]
  B) One-sided (lower only): accept if Mahal² ≥ D - k√(2D)
     (since pLDDT is monotonic in Mahal, upper bound hurts)

Compute: accept rate, mean pLDDT (accepted), mean oracle (accepted), for each k.
Compare against random baseline.
"""
import pandas as pd, numpy as np

df = pd.read_csv('/tmp/goldilocks_all.csv')
# Filter to seqs with all proxies
df = df.dropna(subset=['mahal', 'plddt']).copy()
print(f'Total seqs: {len(df)}  |  with oracle: {df["oracle"].notna().sum()}')

D = 1280
sqrt2D = np.sqrt(2 * D)  # ≈ 50.6
print(f'D = {D},  sqrt(2D) = {sqrt2D:.1f}')

# Random baselines
mean_plddt_all = df['plddt'].mean()
mean_oracle_all = df[df['oracle'].notna()]['oracle'].mean()
print(f'\nFull pool baseline:  pLDDT = {mean_plddt_all:.3f}, oracle = {mean_oracle_all:.3f} (N={len(df)})')

ks = [0.5, 1, 1.5, 2, 3, 5, 7, 10, 15, 20]
print(f'\n=== Policy A: Two-sided  Mahal² ∈ [D - k√2D, D + k√2D] ===')
print(f'{"k":>5} {"lo":>6} {"hi":>6} {"N":>6} {"accept%":>8} {"mean pLDDT":>11} {"ΔpLDDT":>8} {"mean oracle":>12} {"Δoracle":>9}')
print('-' * 85)
for k in ks:
    lo, hi = D - k * sqrt2D, D + k * sqrt2D
    mask = (df['mahal'] >= lo) & (df['mahal'] <= hi)
    acc = df[mask]
    n = len(acc)
    rate = n / len(df) * 100
    p = acc['plddt'].mean()
    dp = p - mean_plddt_all
    acc_with_o = acc[acc['oracle'].notna()]
    o = acc_with_o['oracle'].mean() if len(acc_with_o) else float('nan')
    do = o - mean_oracle_all if not np.isnan(o) else float('nan')
    print(f'{k:>5.1f} {lo:>6.0f} {hi:>6.0f} {n:>6} {rate:>7.1f}% {p:>11.3f} {dp:>+8.3f} {o:>12.3f} {do:>+9.3f}')

print(f'\n=== Policy B: One-sided lower bound  Mahal² ≥ D - k√2D ===')
print(f'{"k":>5} {"lo":>6} {"N":>6} {"accept%":>8} {"mean pLDDT":>11} {"ΔpLDDT":>8} {"mean oracle":>12} {"Δoracle":>9}')
print('-' * 82)
for k in ks:
    lo = D - k * sqrt2D
    mask = df['mahal'] >= lo
    acc = df[mask]
    n = len(acc)
    rate = n / len(df) * 100
    p = acc['plddt'].mean()
    dp = p - mean_plddt_all
    acc_with_o = acc[acc['oracle'].notna()]
    o = acc_with_o['oracle'].mean() if len(acc_with_o) else float('nan')
    do = o - mean_oracle_all if not np.isnan(o) else float('nan')
    print(f'{k:>5.1f} {lo:>6.0f} {n:>6} {rate:>7.1f}% {p:>11.3f} {dp:>+8.3f} {o:>12.3f} {do:>+9.3f}')

# Within-setting analysis: apply filter within each setting, average gain
print(f'\n=== Policy B applied within each setting (avg over settings) ===')
print(f'{"k":>5} {"avg accept%":>12} {"avg ΔpLDDT":>12} {"avg Δoracle":>13}')
print('-' * 50)
for k in ks:
    lo = D - k * sqrt2D
    gains_p = []
    gains_o = []
    rates = []
    for group, gdf in df.groupby('group'):
        if len(gdf) < 50: continue
        mask = gdf['mahal'] >= lo
        if mask.sum() < 10:
            gains_p.append(np.nan); gains_o.append(np.nan); rates.append(0)
            continue
        base_p = gdf['plddt'].mean()
        acc_p = gdf[mask]['plddt'].mean()
        gains_p.append(acc_p - base_p)
        rates.append(mask.mean() * 100)
        if gdf['oracle'].notna().sum() > 0:
            base_o = gdf[gdf['oracle'].notna()]['oracle'].mean()
            acc_o_df = gdf[mask & gdf['oracle'].notna()]
            if len(acc_o_df) > 0:
                gains_o.append(acc_o_df['oracle'].mean() - base_o)
    print(f'{k:>5.1f} {np.nanmean(rates):>11.1f}% {np.nanmean(gains_p):>+12.3f} {np.nanmean(gains_o):>+13.3f}')

print('\n[done]')
