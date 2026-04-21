"""Compute ESM-2 3B pseudo-perplexity on key (task, setting) pairs.
Runs on 100 seqs per (task, setting) for efficiency.
"""
import argparse, os, sys, torch, pandas as pd, numpy as np
sys.path.insert(0, '/data/szhang967/Steering-PLMs')
from utils.esm2_utils import load_esm2_model

DEVICE = 'cuda:0'
N_SEQS = 100
N_POSITIONS = 15

def load_3b():
    return load_esm2_model('3B', device=DEVICE)

@torch.no_grad()
def pppl(seq, model, alphabet):
    bc = alphabet.get_batch_converter()
    mask_idx = alphabet.mask_idx
    _, _, tokens = bc([('protein', seq[:1022])])
    tokens = tokens.to(DEVICE)
    L = tokens.shape[-1] - 2
    rng = np.random.RandomState(42)
    positions = rng.choice(L, min(L, N_POSITIONS), replace=False)
    nll = 0.0
    for pos in positions:
        mt = tokens.clone()
        orig = mt[0, pos + 1].item()
        mt[0, pos + 1] = mask_idx
        om = model(mt)
        lp = torch.log_softmax(om['logits'][0, pos + 1], dim=-1)
        nll += -lp[orig].item()
    return float(np.exp(nll / len(positions)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--settings', required=True)
    ap.add_argument('--tasks', required=True)
    args = ap.parse_args()

    print('Loading 3B...')
    model, alphabet = load_3b()
    print('Loaded.')

    for task in args.tasks.split():
        for setting in args.settings.split():
            if task in ['sol_easy', 'sol_hard', 'therm_easy', 'therm_hard']:
                gen_csv = f'new-results/glp_deviation/{task}_alpha/{setting}.csv'
                outdir = f'new-results/glp_deviation/{task}_alpha/_eval/paper_proxy'
            else:
                gen_csv = f'new-results/glp_deviation/phase_b/{task}_paper/{setting}.csv'
                outdir = f'new-results/glp_deviation/phase_b/{task}_paper/_eval/paper_proxy'
            if not os.path.exists(gen_csv):
                print(f'skip {task}/{setting}'); continue
            os.makedirs(outdir, exist_ok=True)
            out_csv = f'{outdir}/{setting}_ppl3b.csv'
            if os.path.exists(out_csv) and pd.read_csv(out_csv).shape[0] >= N_SEQS:
                print(f'skip (already done) {task}/{setting}'); continue
            df = pd.read_csv(gen_csv).head(N_SEQS)
            if 'sequence' not in df.columns:
                df = df.rename(columns={df.columns[0]: 'sequence'})
            rows = []
            for i, row in df.iterrows():
                try:
                    p = pppl(row['sequence'], model, alphabet)
                except Exception as e:
                    print(f'  err {i}: {e}'); continue
                rows.append({'sequence': row['sequence'], 'ppl_3b': p})
                if (i + 1) % 20 == 0:
                    print(f'  {task}/{setting}: {i+1}/{len(df)}', flush=True)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f'[{task}/{setting}] wrote {out_csv} ({len(rows)} rows)')

if __name__ == '__main__':
    main()
