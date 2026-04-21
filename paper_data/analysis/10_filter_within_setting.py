"""Within-setting filter test:
For each high-oracle setting, apply χ² Mahal filter and check if:
  - pLDDT of filtered > pLDDT of all-steered
  - oracle of filtered >= oracle of input (reference natural = 0.199)
  - oracle drop vs all-steered acceptable (not too big)
"""
import pandas as pd, numpy as np

df = pd.read_csv('/tmp/goldilocks_all.csv').dropna(subset=['mahal','plddt','oracle'])

# Reference natural sol oracle (from paper summary)
REF_ORACLE = 0.199

# Select "high-oracle but low-pLDDT" settings
HIGH_ORACLE_SETTINGS = [
    'sol_easy_allL_a3',
    'sol_easy_allL_a3_L17GLP_u0.5',
    'sol_easy_allL_a2.8_L17GLP_u0.5',
    'sol_easy_allL_a2.5_L17GLP_u0.5',
    'sol_easy_allL_a2_L17GLP_u0.5',
    'sol_easy_allL_a4',
    'sol_easy_allL_a5',
    'sol_easy_allL_a10',
    'sol_easy_allL_a4_L17GLP_u0.5',
    'sol_easy_allL_a5_L17GLP_u0.5',
]

# Filter strategies to test
D = 1280
sqrt2D = np.sqrt(2 * D)
FILTERS = [
    ('none (all)',              None),
    ('B k=0.5 lower',            lambda m: m >= D - 0.5*sqrt2D),      # ≥1255
    ('B k=1 lower',              lambda m: m >= D - 1.0*sqrt2D),      # ≥1229
    ('B k=2 lower',              lambda m: m >= D - 2.0*sqrt2D),      # ≥1179
    ('B k=3 lower',              lambda m: m >= D - 3.0*sqrt2D),      # ≥1128
    ('Asym [1255, 1432]',        lambda m: (m >= 1255) & (m <= 1432)),
    ('top-20% by Mahal',         'top20'),
    ('top-50% by Mahal',         'top50'),
]

print(f'Reference natural (input) oracle: {REF_ORACLE}')
print()
for setting in HIGH_ORACLE_SETTINGS:
    sub = df[df['group'] == setting]
    if len(sub) < 50:
        continue
    all_p = sub['plddt'].mean()
    all_o = sub['oracle'].mean()
    print(f'\n=== {setting.replace("sol_easy_", "")} (N={len(sub)}) ===')
    print(f'    All-steered baseline: pLDDT={all_p:.3f}, oracle={all_o:.3f}')
    print(f'    {"filter":<22} {"N":>4} {"accept%":>8} {"pLDDT":>6} {"ΔpLDDT":>7} {"oracle":>6} {"Δo vs all":>9} {"vs ref":>7}')
    for name, f in FILTERS:
        if f is None:
            acc = sub
        elif f == 'top20':
            acc = sub.nlargest(max(len(sub)//5, 1), 'mahal')
        elif f == 'top50':
            acc = sub.nlargest(max(len(sub)//2, 1), 'mahal')
        else:
            acc = sub[f(sub['mahal'])]
        if len(acc) == 0:
            continue
        p = acc['plddt'].mean()
        o = acc['oracle'].mean()
        dp = p - all_p
        do_all = o - all_o
        do_ref = o - REF_ORACLE
        marker = ''
        if p > all_p and o > REF_ORACLE:
            marker = ' ✓'
        print(f'    {name:<22} {len(acc):>4} {100*len(acc)/len(sub):>7.1f}% {p:>6.3f} {dp:>+7.3f} {o:>6.3f} {do_all:>+9.3f} {do_ref:>+7.3f}{marker}')
