"""Compute rejection sampling gains for BOTH pLDDT and oracle (within-setting)."""
import pandas as pd, numpy as np, os

SETTINGS = {
    'allL_a2':  ('sol_easy_allL_a2_full500', 'allL_a2_scored.csv'),
    'L17_a1':   ('sol_easy_L17_a1_full500', 'L17_a1_scored.csv'),
    'allL_a3':  ('sol_easy_allL_a3_full500', 'allL_a3_scored.csv'),
}
EVAL = 'new-results/glp_deviation/sol_easy_alpha/_eval'

def analyze(short, esm_group, scored_name):
    # Load full-500 ESMFold + per-seq proxy + oracle
    esm_csv = f'{EVAL}/esmfold_full500_{short}.csv'
    esm_df = pd.read_csv(esm_csv).dropna(subset=['plddt']).reset_index(drop=True)
    # Same subsample as D-3
    esm_df = esm_df.sample(n=250, random_state=42).reset_index(drop=True)
    # Join proxies
    proxy_df = pd.read_csv(f'/tmp/d3_{short}.csv').reset_index(drop=True)
    assert len(proxy_df) == len(esm_df), f'{len(proxy_df)} vs {len(esm_df)}'
    df = pd.concat([esm_df[['sequence', 'plddt']].reset_index(drop=True),
                    proxy_df], axis=1)
    # Drop duplicate plddt column
    df = df.loc[:, ~df.columns.duplicated()]
    # Load oracle (per-seq)
    scored = pd.read_csv(f'{EVAL}/{scored_name}')
    # Match by sequence
    seq_to_prob = dict(zip(scored['sequence'], scored['pred_prob']))
    df['oracle'] = df['sequence'].map(seq_to_prob)
    df = df.dropna(subset=['oracle'])

    K = len(df) // 5  # top 20%
    np.random.seed(42)
    rand_idx = np.random.choice(len(df), K, replace=False)
    random_p = df.iloc[rand_idx]['plddt'].mean()
    random_o = df.iloc[rand_idx]['oracle'].mean()

    proxies = {
        'Mahal':       ('mahal', True),     # higher = better
        'GLP u=0.05':  ('glp_u0.05', True),
        'GLP u=0.15':  ('glp_u0.15', True),
        'ppl 650M':    ('ppl_650m', False), # lower = better
        'ppl 3B':      ('ppl_3b', False),
    }
    print(f'\n=== {short}: N={len(df)}, K(top-20%)={K} ===')
    print(f'  Random-K: pLDDT={random_p:.3f}  oracle={random_o:.3f}')
    print(f'  {"proxy":<13} {"top pLDDT":>11} {"ΔpLDDT":>8} {"top oracle":>12} {"Δoracle":>9}')
    print('  ' + '-' * 60)
    for name, (col, high_is_good) in proxies.items():
        if high_is_good:
            top = df.nlargest(K, col)
        else:
            top = df.nsmallest(K, col)
        tp = top['plddt'].mean()
        to = top['oracle'].mean()
        print(f'  {name:<13} {tp:>11.3f} {tp-random_p:>+8.3f} {to:>12.3f} {to-random_o:>+9.3f}')
    return df

for s, (esm_g, sc) in SETTINGS.items():
    try:
        analyze(s, esm_g, sc)
    except Exception as e:
        print(f'{s}: ERROR {e}')
