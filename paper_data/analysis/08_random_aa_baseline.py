"""Test Mahal and GLP resid on random amino acid baseline sequences.
Compare to the Goldilocks curve: do random seqs have LOW Mahal + LOW pLDDT
(supporting monotonic story), or HIGH Mahal + LOW pLDDT (breaking it)?
"""
import sys, os, torch, pandas as pd, numpy as np
sys.path.insert(0, '/data/szhang967/Steering-PLMs')
sys.path.insert(0, '/data/szhang967/Steering-PLMs/generative_latent_prior')

from utils.esm2_utils import load_esm2_model
from scripts.glp_deviation.generate_alpha import load_glp, build_glp_projection_fn_interior

DEVICE = 'cuda:0'
GLP_PATH = 'generative_latent_prior/runs/glp-esm2-650m-layer17-d6'
rep_stats = torch.load(f'{GLP_PATH}/rep_statistics.pt', map_location='cpu', weights_only=False)
rep_mean = rep_stats['mean'].float().squeeze().to(DEVICE)
rep_std  = torch.sqrt(rep_stats['var'].float().squeeze().to(DEVICE))

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
    return mahal, resid

# Load random sequences (N=100 subsample)
rand_df = pd.read_csv('data/benchmarks/random_aa_seqs/random_1000.csv')
print(f'Random seqs: {len(rand_df)}, columns: {list(rand_df.columns)}')
print(f'First seq: {rand_df.iloc[0, 0][:60]}...')
rand_df = rand_df.head(100)

# Load existing ESMFold for random
esm_rand = pd.read_csv('new-results/fixed_layer/random_baseline/esmfold_results.csv')
print(f'Random ESMFold: {len(esm_rand)} rows, cols: {list(esm_rand.columns)}')
# Join by sequence
rand_merged = rand_df.merge(esm_rand[['sequence','plddt']] if 'plddt' in esm_rand.columns else esm_rand,
                             left_on=rand_df.columns[0], right_on='sequence', how='inner')
print(f'Merged: {len(rand_merged)} seqs with pLDDT')

# Compute Mahal and GLP resid
print('Computing proxies on random sequences...')
mahals, resids = [], []
for i, row in rand_merged.iterrows():
    m, r = compute(row['sequence'])
    mahals.append(m); resids.append(r)

rand_merged['mahal'] = mahals
rand_merged['glp_resid'] = resids

print(f'\n=== Random AA sequences ===')
print(f'  N={len(rand_merged)}')
print(f'  mean pLDDT:     {rand_merged["plddt"].mean():.3f}  ±  {rand_merged["plddt"].std():.3f}')
print(f'  mean Mahal:     {rand_merged["mahal"].mean():.0f}  ±  {rand_merged["mahal"].std():.0f}')
print(f'  mean GLP resid: {rand_merged["glp_resid"].mean():.1f}  ±  {rand_merged["glp_resid"].std():.1f}')

# Compare to Goldilocks bins
print(f'\n=== Where do random seqs land in the Goldilocks bins? ===')
bins_df = pd.read_csv('/tmp/goldilocks_bins.csv')
print(bins_df.to_string(index=False))

# Find which bin the random mean falls into
mahal_ranges = [(b['mahal_lo'], b['mahal_hi']) for _, b in bins_df.iterrows()]
rand_mahal = rand_merged['mahal'].mean()
for i, (lo, hi) in enumerate(mahal_ranges):
    if lo <= rand_mahal <= hi:
        print(f'Random seqs mean Mahal ({rand_mahal:.0f}) falls in bin {i}')
        print(f'  That bin has pLDDT={bins_df.iloc[i].mean_plddt:.3f}, oracle={bins_df.iloc[i].mean_oracle:.3f}')
        break
else:
    print(f'Random seqs Mahal {rand_mahal:.0f} outside all bins')

rand_merged.to_csv('/tmp/random_aa_proxy.csv', index=False)
print(f'\n[done] /tmp/random_aa_proxy.csv')
