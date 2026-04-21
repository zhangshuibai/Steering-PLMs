"""Phase D-3: Rejection sampling comparison of 3 proxies:
  - GLP denoising residual
  - ESM2 pseudo-perplexity (pppl)
  - Mahalanobis at L17

For each proxy: sort 500 seqs by that proxy, take top-100 and bottom-100,
compute mean pLDDT, compare to random-100 baseline.
"""
import sys, os, torch, pandas as pd, numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, '/data/szhang967/Steering-PLMs')
sys.path.insert(0, '/data/szhang967/Steering-PLMs/generative_latent_prior')

from utils.esm2_utils import load_esm2_model
from scripts.glp_deviation.generate_alpha import load_glp, build_glp_projection_fn_interior

DEVICE = 'cuda:0'
GLP_PATH = 'generative_latent_prior/runs/glp-esm2-650m-layer17-d6'
GLP_STATS = f'{GLP_PATH}/rep_statistics.pt'

# Key settings for proxy comparison
SETTINGS = {
    'allL_a2': 'sol_easy_allL_a2_full500',
    'L17_a1': 'sol_easy_L17_a1_full500',
    'allL_a3': 'sol_easy_allL_a3_full500',
}

print(f'[{pd.Timestamp.now()}] Loading models...')
rep_stats = torch.load(GLP_STATS, map_location='cpu', weights_only=False)
rep_mean = rep_stats['mean'].float().squeeze().to(DEVICE)
rep_var = rep_stats['var'].float().squeeze().to(DEVICE)
rep_std = torch.sqrt(rep_var)

model, alphabet = load_esm2_model('650M', device=DEVICE)
bc = alphabet.get_batch_converter()
glp = load_glp(GLP_PATH, device=DEVICE)

# Also load ESM-2 3B for a fairer ppl comparison
print(f'[{pd.Timestamp.now()}] Loading ESM-2 3B…')
try:
    model_3b, alphabet_3b = load_esm2_model('3B', device=DEVICE)
    bc_3b = alphabet_3b.get_batch_converter()
    HAS_3B = True
    print('  loaded 3B')
except Exception as e:
    print(f'  3B load failed: {e}')
    HAS_3B = False

glp_noise_gen = torch.Generator(device=DEVICE); glp_noise_gen.manual_seed(0)
glp_fn = build_glp_projection_fn_interior(
    glp, u=0.05, num_timesteps=25, noise_generator=glp_noise_gen
)

GLP_U_SWEEP = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
glp_fns = {}
for u in GLP_U_SWEEP:
    g = torch.Generator(device=DEVICE); g.manual_seed(0)
    glp_fns[u] = build_glp_projection_fn_interior(glp, u=u, num_timesteps=25, noise_generator=g)

@torch.no_grad()
def compute_proxies(seq):
    """Return dict with mahal, glp_resid_{u}, pppl for a sequence."""
    _, _, tokens = bc([('protein', seq[:1022])])
    tokens = tokens.to(DEVICE)
    T = tokens.shape[-1]
    out = model(tokens, repr_layers=[17])
    h17 = out['representations'][17]  # (1, T, 1280)
    h_interior = h17[0, 1:-1]         # (L, 1280)
    L = h_interior.shape[0]

    # Mahalanobis mean per residue
    z = (h_interior - rep_mean) / rep_std
    mahal = (z ** 2).sum(dim=-1).mean().item()

    # GLP denoising residuals at multiple u values
    h_esm_format = h17[0].unsqueeze(1)  # (T, 1, 1280) for glp_fn
    resid_by_u = {}
    for u in GLP_U_SWEEP:
        # Reset generator for reproducibility
        gen = glp_fns[u].__closure__[0].cell_contents if hasattr(glp_fns[u], '__closure__') else None
        # Simpler: call and compute residual
        projected = glp_fns[u](h_esm_format)
        resid_by_u[u] = (h_esm_format[1:-1] - projected[1:-1]).norm(dim=-1).mean().item()
    resid = resid_by_u[0.05]  # keep legacy field

    # Pseudo-perplexity via MLM at each position
    mask_idx = alphabet.mask_idx
    nll_sum = 0.0
    # Batch-friendly: mask one position at a time; average NLL
    # For efficiency, estimate on a random subset of positions
    n_positions_to_sample = min(L, 15)  # reduced from 30 for overnight speed
    positions = np.random.RandomState(42).choice(L, n_positions_to_sample, replace=False)
    for pos in positions:
        masked_tokens = tokens.clone()
        original = masked_tokens[0, pos + 1].item()
        masked_tokens[0, pos + 1] = mask_idx
        out_m = model(masked_tokens)
        logits = out_m['logits'][0, pos + 1]
        log_probs = torch.log_softmax(logits, dim=-1)
        nll_sum += -log_probs[original].item()
    pppl = np.exp(nll_sum / n_positions_to_sample)

    # ESM-2 3B pseudo-ppl (same sampled positions)
    pppl_3b = float('nan')
    if HAS_3B:
        _, _, tok3b = bc_3b([('protein', seq[:1022])])
        tok3b = tok3b.to(DEVICE)
        mask3b = alphabet_3b.mask_idx
        nll3b = 0.0
        for pos in positions:
            mt = tok3b.clone()
            orig = mt[0, pos + 1].item()
            mt[0, pos + 1] = mask3b
            out3b = model_3b(mt)
            lp = torch.log_softmax(out3b['logits'][0, pos + 1], dim=-1)
            nll3b += -lp[orig].item()
        pppl_3b = np.exp(nll3b / n_positions_to_sample)

    out_dict = {'mahal': mahal, 'glp_resid': resid, 'ppl_650m': pppl, 'ppl_3b': pppl_3b}
    for u, v in resid_by_u.items():
        out_dict[f'glp_u{u}'] = v
    return out_dict

# ============================================================
# Main analysis
# ============================================================
results = []
for short, esm_group in SETTINGS.items():
    gen_csv = f'new-results/glp_deviation/sol_easy_alpha/{short}.csv'
    esm_csv = f'new-results/glp_deviation/sol_easy_alpha/_eval/esmfold_full500_{short}.csv'
    if not os.path.exists(esm_csv):
        # fall back to 100-seq ESMFold
        print(f'Full 500 ESMFold not found for {short}, using 100-seq version')
        esm_base = 'new-results/glp_deviation/sol_easy_alpha/_eval/esmfold_results.csv'
        esm_df = pd.read_csv(esm_base)
        esm_df = esm_df[esm_df['group'] == f'sol_easy_{short}'].head(100)
    else:
        esm_df = pd.read_csv(esm_csv)
    esm_df = esm_df.dropna(subset=['plddt']).reset_index(drop=True)
    # Subsample for overnight speed: limit to 250 seqs per setting
    if len(esm_df) > 250:
        esm_df = esm_df.sample(n=250, random_state=42).reset_index(drop=True)
    print(f'\n=== {short}: {len(esm_df)} seqs with pLDDT (subsampled) ===')

    rows = []
    for i, row in esm_df.iterrows():
        try:
            proxies = compute_proxies(row['sequence'])
            proxies['plddt'] = row['plddt']
            rows.append(proxies)
        except Exception as e:
            print(f'  skip seq: {e}')
        if (i + 1) % 100 == 0:
            print(f'  processed {i+1}/{len(esm_df)}')
    df = pd.DataFrame(rows)
    df.to_csv(f'/tmp/d3_{short}.csv', index=False)

    # Correlations
    r_m = spearmanr(df['mahal'], df['plddt']).correlation
    r_p = spearmanr(df['ppl_650m'], df['plddt']).correlation
    r_p3 = spearmanr(df['ppl_3b'], df['plddt']).correlation if df['ppl_3b'].notna().any() else float('nan')
    print(f'  Spearman r: Mahal={r_m:+.3f}  ppl_650M={r_p:+.3f}  ppl_3B={r_p3:+.3f}')
    print('  GLP resid at various u:')
    for u in GLP_U_SWEEP:
        col = f'glp_u{u}'
        if col in df.columns:
            r = spearmanr(df[col], df['plddt']).correlation
            marker = ' ← beats Mahal' if abs(r) > abs(r_m) else ''
            print(f'    u={u}: r={r:+.3f}{marker}')
    r_g = spearmanr(df['glp_u0.05'], df['plddt']).correlation if 'glp_u0.05' in df.columns else float('nan')

    # Rejection sampling: pick top-20% by each proxy, compute mean pLDDT
    K = max(20, len(df) // 5)  # top ~20%
    rand_100 = df.sample(K, random_state=42)['plddt'].mean()
    rand_100_std = df.sample(K, random_state=42)['plddt'].std()

    # Higher Mahal/GLP_resid = better pLDDT (positive correlation)
    # Lower ppl = better pLDDT (negative correlation expected)
    # Pick accordingly:
    top_m = df.nlargest(K, 'mahal')['plddt'].mean()
    top_g = df.nlargest(K, 'glp_resid')['plddt'].mean()
    top_p = df.nsmallest(K, 'ppl_650m')['plddt'].mean()
    top_p3 = df.nsmallest(K, 'ppl_3b')['plddt'].mean() if df['ppl_3b'].notna().any() else float('nan')

    print(f'  Rejection sampling (top-{K} vs random-{K}):')
    print(f'    Random-{K} pLDDT: {rand_100:.3f} ± {rand_100_std:.3f}')
    print(f'    Top-{K} Mahal       : {top_m:.3f}  (Δ={top_m - rand_100:+.3f})')
    print(f'    Top-{K} GLP_resid   : {top_g:.3f}  (Δ={top_g - rand_100:+.3f})')
    print(f'    Low-ppl 650M top-{K}: {top_p:.3f}  (Δ={top_p - rand_100:+.3f})')
    print(f'    Low-ppl 3B   top-{K}: {top_p3:.3f}  (Δ={top_p3 - rand_100:+.3f})')

    results.append({
        'setting': short, 'n_seqs': len(df),
        'mean_plddt': df['plddt'].mean(),
        'r_mahal': r_m, 'r_glp_resid': r_g,
        'r_ppl_650m': r_p, 'r_ppl_3b': r_p3,
        'rejsamp_top_mahal': top_m, 'rejsamp_top_glp': top_g,
        'rejsamp_low_ppl_650m': top_p, 'rejsamp_low_ppl_3b': top_p3,
        'rejsamp_random': rand_100,
        'gain_mahal': top_m - rand_100,
        'gain_glp': top_g - rand_100,
        'gain_ppl_650m': top_p - rand_100,
        'gain_ppl_3b': top_p3 - rand_100,
    })

summary = pd.DataFrame(results)
summary.to_csv('/tmp/d3_summary.csv', index=False)
print('\n=== SUMMARY ===')
print(summary.to_string())
print(f'\nRaw data saved to /tmp/d3_*.csv')
