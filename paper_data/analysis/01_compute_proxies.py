"""Compute per-sequence proxies (Mahal, GLP resid u=0.15, ppl 650M) for a split
across multiple settings. Output: one CSV per setting with
[sequence, mahal, glp_resid, ppl_650m, plddt, oracle] joined.
"""
import argparse, os, sys, torch, pandas as pd, numpy as np, glob
sys.path.insert(0, '/data/szhang967/Steering-PLMs')
sys.path.insert(0, '/data/szhang967/Steering-PLMs/generative_latent_prior')
from utils.esm2_utils import load_esm2_model
from scripts.glp_deviation.generate_alpha import load_glp, build_glp_projection_fn_interior

DEVICE = 'cuda:0'
GLP_PATH = 'generative_latent_prior/runs/glp-esm2-650m-layer17-d6'
rep_stats = torch.load(f'{GLP_PATH}/rep_statistics.pt', map_location='cpu', weights_only=False)
rep_mean = rep_stats['mean'].float().squeeze().to(DEVICE)
rep_std = torch.sqrt(rep_stats['var'].float().squeeze().to(DEVICE))

model, alphabet = load_esm2_model('650M', device=DEVICE)
bc = alphabet.get_batch_converter()
glp = load_glp(GLP_PATH, device=DEVICE)
glp_gen = torch.Generator(device=DEVICE); glp_gen.manual_seed(0)
glp_fn = build_glp_projection_fn_interior(glp, u=0.15, num_timesteps=25, noise_generator=glp_gen)

@torch.no_grad()
def compute(seq):
    _, _, tokens = bc([('protein', seq[:1022])])
    tokens = tokens.to(DEVICE)
    out = model(tokens, repr_layers=[17])
    h17 = out['representations'][17]
    h_interior = h17[0, 1:-1]
    z = (h_interior - rep_mean) / rep_std
    mahal = (z ** 2).sum(dim=-1).mean().item()
    h_esm = h17[0].unsqueeze(1)
    glp_gen.manual_seed(0)
    projected = glp_fn(h_esm)
    resid = (h_esm[1:-1] - projected[1:-1]).norm(dim=-1).mean().item()
    # ppl 650M via 15 random masks
    L = h_interior.shape[0]
    np_rng = np.random.RandomState(42)
    positions = np_rng.choice(L, min(L, 15), replace=False)
    nll = 0.0
    mask_idx = alphabet.mask_idx
    for pos in positions:
        mt = tokens.clone()
        orig = mt[0, pos + 1].item()
        mt[0, pos + 1] = mask_idx
        om = model(mt)
        lp = torch.log_softmax(om['logits'][0, pos + 1], dim=-1)
        nll += -lp[orig].item()
    ppl = float(np.exp(nll / len(positions)))
    return mahal, resid, ppl

def process_csv(gen_csv, esm_csv, scored_csv, out_csv):
    gen = pd.read_csv(gen_csv)
    if 'sequence' not in gen.columns:
        gen = gen.rename(columns={gen.columns[0]: 'sequence'})
    esm = pd.read_csv(esm_csv) if os.path.exists(esm_csv) else None
    scored = pd.read_csv(scored_csv) if os.path.exists(scored_csv) else None
    rows = []
    for i, row in gen.iterrows():
        seq = row['sequence']
        try:
            m, r, p = compute(seq)
        except Exception as e:
            print(f'  skip seq {i}: {e}')
            continue
        pl = np.nan
        orc = np.nan
        if esm is not None:
            em = esm[esm['sequence'] == seq]
            if len(em) > 0: pl = em['plddt'].iloc[0]
        if scored is not None:
            sm = scored[scored['sequence'] == seq]
            if len(sm) > 0:
                col = 'pred_prob' if 'pred_prob' in scored.columns else 'pred_tm' if 'pred_tm' in scored.columns else 'score'
                if col in scored.columns: orc = sm[col].iloc[0]
        rows.append({'sequence': seq, 'mahal': m, 'glp_resid': r, 'ppl_650m': p, 'plddt': pl, 'oracle': orc})
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(gen)}', flush=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f'  wrote {out_csv} ({len(rows)} rows)')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default=None, help='sol_easy/sol_hard/therm_easy/therm_hard')
    ap.add_argument('--fitness', default=None, help='trpb/gfp')
    ap.add_argument('--settings', required=True)
    args = ap.parse_args()
    settings = args.settings.split()
    if args.split:
        root = f'new-results/glp_deviation/{args.split}_alpha'
        outdir = f'{root}/_eval/paper_proxy'
        # Aggregate ESMFold: look for full500_stage1 first, then esmfold_results
        esm_candidates = [
            f'{root}/_eval/esmfold_full500_stage1.csv',
            f'{root}/_eval/esmfold_full500_allL_a2.csv',
            f'{root}/_eval/esmfold_full500_L17_a1.csv',
            f'{root}/_eval/esmfold_full500_allL_a3.csv',
        ]
        # Merge all ESMFold sources
        all_esm = [pd.read_csv(f) for f in glob.glob(f'{root}/_eval/esmfold*.csv') if os.path.exists(f)]
        if all_esm:
            merged = pd.concat(all_esm, ignore_index=True).drop_duplicates(subset=['sequence'])
            os.makedirs(outdir, exist_ok=True)
            merged.to_csv(f'{outdir}/_merged_esmfold.csv', index=False)
        for s in settings:
            gen_csv = f'{root}/{s}.csv'
            if not os.path.exists(gen_csv):
                print(f'skip {s} (no gen)'); continue
            esm_csv = f'{outdir}/_merged_esmfold.csv'
            scored_csv = f'{root}/_eval/{s}_scored.csv'
            out_csv = f'{outdir}/{s}_proxy.csv'
            print(f'\n[{args.split} / {s}]')
            os.makedirs(outdir, exist_ok=True)
            process_csv(gen_csv, esm_csv, scored_csv, out_csv)
    elif args.fitness:
        root = f'new-results/glp_deviation/phase_b/{args.fitness}_paper'
        outdir = f'{root}/_eval/paper_proxy'
        os.makedirs(outdir, exist_ok=True)
        esm_csv = f'{root}/_eval/esmfold_paper.csv'
        for s in settings:
            gen_csv = f'{root}/{s}.csv'
            if not os.path.exists(gen_csv):
                print(f'skip {s}'); continue
            scored_csv = f'{root}/_eval/{s}_scored.csv'
            out_csv = f'{outdir}/{s}_proxy.csv'
            print(f'\n[{args.fitness} / {s}]')
            process_csv(gen_csv, esm_csv, scored_csv, out_csv)
    else:
        print('must provide --split or --fitness'); sys.exit(1)

if __name__ == '__main__':
    main()
