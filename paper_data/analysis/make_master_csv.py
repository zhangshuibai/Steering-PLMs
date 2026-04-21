"""Create a single master CSV per task merging all proxies + metrics.
Writes: paper_data/<task>/master.csv with columns
    [task, setting, sequence, plddt, oracle, mahal, glp_resid, ppl_650m, ppl_3b]
"""
import pandas as pd, os

TASKS = ['sol_easy', 'sol_hard', 'therm_easy', 'therm_hard', 'trpb', 'gfp']
SETTINGS = ['L17_a1', 'L17_a10', 'allL_a2', 'allL_a3', 'allL_a2_L17GLP_u0.5']

for task in TASKS:
    rows = []
    for setting in SETTINGS:
        proxy_csv = f'paper_data/{task}/proxy/{setting}_proxy.csv'
        if not os.path.exists(proxy_csv):
            print(f'  skip {task}/{setting} (no proxy)'); continue
        df = pd.read_csv(proxy_csv)
        df['task'] = task
        df['setting'] = setting
        # Merge ppl_3b if exists
        ppl3b_csv = f'paper_data/{task}/proxy/{setting}_ppl3b.csv'
        if os.path.exists(ppl3b_csv):
            p3b = pd.read_csv(ppl3b_csv)
            s2p = dict(zip(p3b['sequence'], p3b['ppl_3b']))
            df['ppl_3b'] = df['sequence'].map(s2p)
        else:
            df['ppl_3b'] = float('nan')
        rows.append(df)
    if not rows:
        print(f'[{task}] no data'); continue
    master = pd.concat(rows, ignore_index=True)
    # Reorder cols
    col_order = ['task', 'setting', 'sequence', 'plddt', 'oracle',
                 'mahal', 'glp_resid', 'ppl_650m', 'ppl_3b']
    master = master[[c for c in col_order if c in master.columns]]
    out = f'paper_data/{task}/master.csv'
    master.to_csv(out, index=False)
    print(f'[{task}] wrote {out}  ({len(master)} rows, {len(master["setting"].unique())} settings)')

print('\nDone. To load any task:')
print('  import pandas as pd')
print('  df = pd.read_csv("paper_data/sol_easy/master.csv")')
