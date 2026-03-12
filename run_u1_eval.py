"""Quick eval for u=1.0 across all step values."""
import os, sys, types, json, torch, numpy as np, pandas as pd
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'generative_latent_prior'))

from steering_with_glp import (
    build_glp_projection_fn, generate_with_glp, steering_forward_with_glp,
    evaluate_sol, PropertyPredictor,
)
from evaluate_ppl import compute_pseudo_perplexity_multi_gpu
from generative_latent_prior.glp.denoiser import GLP
from utils.esm2_utils import load_esm2_model

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def main():
    mp.set_start_method('spawn', force=True)
    device = 'cuda:0'
    output_dir = 'results/comprehensive_eval/glp'
    os.makedirs(output_dir, exist_ok=True)

    u = 1.0
    steps_list = [25, 50, 100, 200, 400]
    results = []

    # Phase 1: Load models + generate + sol eval
    ref_seqs = pd.read_csv('data/sol_easy.csv')['sequence'].tolist()
    model_650m, alphabet = load_esm2_model("650M", device=device)
    model_650m.steering_forward_glp = types.MethodType(steering_forward_with_glp, model_650m)

    pos_sv, neg_sv = torch.load('saved_steering_vectors/650M_sol_steering_vectors.pt')
    steering_vectors_all = (pos_sv - neg_sv).to(device)

    from omegaconf import OmegaConf
    glp_config = OmegaConf.load('generative_latent_prior/runs/glp-esm2-650m-layer17-d6/config.yaml')
    OmegaConf.resolve(glp_config)
    glp_config.glp_kwargs.normalizer_config.rep_statistic = 'generative_latent_prior/runs/glp-esm2-650m-layer17-d6/rep_statistics.pt'
    glp_model = GLP(**glp_config.glp_kwargs)
    glp_model.to(device)
    glp_model.load_pretrained('generative_latent_prior/runs/glp-esm2-650m-layer17-d6', name='final')
    glp_model.eval()

    predictor = PropertyPredictor(embed_dim=1280)
    ckpt = torch.load('saved_predictors/sol_predictor_final.pt', map_location='cpu')
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        predictor.load_state_dict(ckpt['model_state_dict'])
    else:
        predictor.load_state_dict(ckpt)
    predictor = predictor.to(device).eval()

    gen_params = {'mask_ratio': 0.1, 'temperature': 1.0, 'top_p': 0.9}

    for steps in steps_list:
        csv_path = os.path.join(output_dir, f'u{u}_steps{steps}.csv')
        method = f'L17+GLP(u={u},s={steps})'

        if os.path.exists(csv_path) and len(pd.read_csv(csv_path)) >= 100:
            seqs = pd.read_csv(csv_path)['sequence'].tolist()
            print(f"{method}: CACHED")
        else:
            print(f"{method}: generating...")
            glp_project_fn = build_glp_projection_fn(glp_model, u=u, num_timesteps=steps)
            seqs = generate_with_glp(
                ref_seqs, model_650m, alphabet, steering_vectors_all,
                glp_project_fn, 17, device, 100, gen_params
            )
            pd.DataFrame({'sequence': seqs}).to_csv(csv_path, index=False)

        mean_prob, sol_ratio, probs = evaluate_sol(seqs, model_650m, alphabet, predictor, device)
        print(f"  sol={sol_ratio*100:.1f}%")
        results.append({'method': method, 'csv_path': csv_path, 'sol_ratio': float(sol_ratio),
                        'sol_mean_prob': float(mean_prob), 'n_seqs': len(seqs), 'u': u, 'steps': steps})

    # Phase 2: pPPL
    del model_650m, predictor, glp_model, steering_vectors_all
    torch.cuda.empty_cache()

    print("\nEvaluating pPPL...")
    for res in results:
        seqs = pd.read_csv(res['csv_path'])['sequence'].tolist()
        ppls = compute_pseudo_perplexity_multi_gpu(seqs, '3B', [0,1,2,3], 32)
        ppls_arr = np.array(ppls)
        res['ppl_mean'] = float(ppls_arr.mean())
        res['ppl_median'] = float(np.median(ppls_arr))
        res['ppl_std'] = float(ppls_arr.std())
        print(f"  {res['method']}: sol={res['sol_ratio']*100:.1f}%, pPPL={res['ppl_mean']:.4f}")

    # Summary
    print(f"\n{'Method':<30} | {'Sol %':>7} | {'pPPL mean':>10} | {'pPPL med':>10}")
    print("-" * 65)
    for r in results:
        print(f"{r['method']:<30} | {r['sol_ratio']*100:>6.1f}% | {r['ppl_mean']:>10.4f} | {r['ppl_median']:>10.4f}")

    with open('results/comprehensive_eval/u1_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/comprehensive_eval/u1_summary.json")

if __name__ == "__main__":
    main()
