"""Goldilocks analysis: compute per-sequence Mahalanobis and GLP residual for
all ESMFolded sequences across all settings, then look for sweet-spot pattern
(non-monotonic relationship with pLDDT/oracle).

Hypothesis: higher Mahal/GLP correlates with higher pLDDT+oracle up to a point,
then reverses (too far from distribution → collapsed structure).
"""
import sys, os, torch, pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
sys.path.insert(0, '/data/szhang967/Steering-PLMs')
sys.path.insert(0, '/data/szhang967/Steering-PLMs/generative_latent_prior')

from utils.esm2_utils import load_esm2_model
from scripts.glp_deviation.generate_alpha import load_glp, build_glp_projection_fn_interior

DEVICE = 'cuda:0'
GLP_PATH = 'generative_latent_prior/runs/glp-esm2-650m-layer17-d6'
GLP_STATS = f'{GLP_PATH}/rep_statistics.pt'

print('[load] GLP stats + ESM-2 + GLP model')
rep_stats = torch.load(GLP_STATS, map_location='cpu', weights_only=False)
rep_mean = rep_stats['mean'].float().squeeze().to(DEVICE)
rep_var = rep_stats['var'].float().squeeze().to(DEVICE)
rep_std = torch.sqrt(rep_var)
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
    glp_resid = (h_esm[1:-1] - projected[1:-1]).norm(dim=-1).mean().item()
    return mahal, glp_resid

# ===== Collect data across all settings with ESMFold results =====
SPLIT = 'sol_easy'
esm_dir = f'new-results/glp_deviation/{SPLIT}_alpha/_eval'
all_esm = []
for f in glob.glob(f'{esm_dir}/esmfold*.csv'):
    all_esm.append(pd.read_csv(f))
esm_df = pd.concat(all_esm, ignore_index=True).drop_duplicates(subset=['group', 'seq_idx'])
esm_df = esm_df.dropna(subset=['plddt'])
print(f'[data] {len(esm_df)} total ESMFolded sequences across settings')
print(f'       {esm_df["group"].nunique()} unique settings')

# Add oracle per seq via sequence matching (look up each setting's scored csv)
def load_oracle(label):
    setting = label.replace(f'{SPLIT}_', '')
    scored_path = f'{esm_dir}/{setting}_scored.csv'
    if not os.path.exists(scored_path):
        return {}
    scored = pd.read_csv(scored_path)
    return dict(zip(scored['sequence'], scored['pred_prob']))

esm_df['oracle'] = np.nan
for label in esm_df['group'].unique():
    seq_to_oracle = load_oracle(label)
    mask = esm_df['group'] == label
    esm_df.loc[mask, 'oracle'] = esm_df.loc[mask, 'sequence'].map(seq_to_oracle)

# Compute Mahal + GLP resid for each sequence
print('[compute] Mahal + GLP resid per sequence...')
mahals, resids = [], []
for i, row in esm_df.iterrows():
    try:
        m, r = compute(row['sequence'])
    except Exception as e:
        m, r = np.nan, np.nan
    mahals.append(m); resids.append(r)
    if (i + 1) % 200 == 0:
        print(f'  {i+1}/{len(esm_df)}')

esm_df['mahal'] = mahals
esm_df['glp_resid'] = resids
esm_df.to_csv('/tmp/goldilocks_all.csv', index=False)

# ===== Setting-level summary =====
print('\n=== Setting-level summary ===')
summ = esm_df.groupby('group').agg(
    n=('plddt', 'count'),
    mean_plddt=('plddt', 'mean'),
    mean_oracle=('oracle', 'mean'),
    mean_mahal=('mahal', 'mean'),
    mean_glp_resid=('glp_resid', 'mean'),
).reset_index()
summ = summ.sort_values('mean_mahal')
print(summ.to_string(index=False))
summ.to_csv('/tmp/goldilocks_settings.csv', index=False)

# ===== Correlation cross-setting =====
print('\n=== Cross-setting correlations ===')
r_mp = spearmanr(summ['mean_mahal'], summ['mean_plddt']).correlation
r_mo = spearmanr(summ['mean_mahal'], summ['mean_oracle']).correlation
r_gp = spearmanr(summ['mean_glp_resid'], summ['mean_plddt']).correlation
r_go = spearmanr(summ['mean_glp_resid'], summ['mean_oracle']).correlation
print(f'  r(setting mean Mahal,     setting mean pLDDT):  {r_mp:+.3f}')
print(f'  r(setting mean Mahal,     setting mean oracle): {r_mo:+.3f}')
print(f'  r(setting mean GLP resid, setting mean pLDDT):  {r_gp:+.3f}')
print(f'  r(setting mean GLP resid, setting mean oracle): {r_go:+.3f}')

# ===== Goldilocks binning =====
print('\n=== Goldilocks binning: bin by Mahal, compute mean pLDDT/oracle per bin ===')
# pool all sequences, bin by Mahal
bins = np.quantile(esm_df['mahal'].dropna(), [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
esm_df['mahal_bin'] = pd.cut(esm_df['mahal'], bins=bins, include_lowest=True)
bin_summ = esm_df.groupby('mahal_bin', observed=True).agg(
    n=('plddt', 'count'),
    mahal_lo=('mahal', 'min'),
    mahal_hi=('mahal', 'max'),
    mean_plddt=('plddt', 'mean'),
    mean_oracle=('oracle', 'mean'),
).reset_index()
print(bin_summ.to_string(index=False))
bin_summ.to_csv('/tmp/goldilocks_bins.csv', index=False)

print('\n[done] /tmp/goldilocks_all.csv, /tmp/goldilocks_settings.csv, /tmp/goldilocks_bins.csv')
