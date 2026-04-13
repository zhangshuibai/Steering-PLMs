"""
Evaluate pseudo-perplexity (pPPL) of protein sequences using ESM2.

pPPL measures how "natural" a protein sequence looks to a pretrained language model.
For each position, we mask it and measure the model's log-probability of the true token.
pPPL = exp(-1/L * Σ log P(x_i | x_\i))

Lower pPPL = more natural/protein-like sequence.

Supports multi-GPU parallel evaluation via --gpu_ids.

Usage:
    python evaluate_ppl.py \
        --input_csvs results/ESM2_gen_steering_sol_easy.csv data/sol_easy.csv \
        --labels "Steering" "Reference" \
        --model 3B --gpu_ids 0 1 4 5
"""

import argparse
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import esm
import torch.multiprocessing as mp
from functools import partial


def load_esm2_model(model_size, device):
    """Load ESM2 model by size string."""
    model_map = {
        "150M": "esm2_t30_150M_UR50D",
        "650M": "esm2_t33_650M_UR50D",
        "3B": "esm2_t36_3B_UR50D",
    }
    model_name = model_map[model_size]
    print(f"Loading {model_name} on {device}...")
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.to(device)
    model.eval()
    return model, alphabet


def compute_ppl_single_seq(seq, model, alphabet, device, batch_masks=32, max_len=1022):
    """Compute pPPL for a single sequence."""
    batch_converter = alphabet.get_batch_converter()
    mask_idx = alphabet.mask_idx

    seq = seq[:max_len]
    L = len(seq)

    data = [("protein", seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    original_tokens = tokens.clone()

    positions = list(range(1, L + 1))
    log_probs_sum = 0.0

    for start in range(0, len(positions), batch_masks):
        batch_pos = positions[start:start + batch_masks]
        bs = len(batch_pos)

        masked_tokens = original_tokens.repeat(bs, 1)
        for j, pos in enumerate(batch_pos):
            masked_tokens[j, pos] = mask_idx

        with torch.no_grad():
            results = model(masked_tokens)
        logits = results["logits"]

        for j, pos in enumerate(batch_pos):
            true_token = original_tokens[0, pos].item()
            log_p = torch.log_softmax(logits[j, pos], dim=-1)[true_token].item()
            log_probs_sum += log_p

    ppl = np.exp(-log_probs_sum / L)
    return ppl


def worker_fn(gpu_id, model_size, seq_indices, seqs, batch_masks, result_dict):
    """Worker function for multi-GPU evaluation."""
    device = f"cuda:{gpu_id}"
    model, alphabet = load_esm2_model(model_size, device)

    for idx in tqdm(seq_indices, desc=f"GPU {gpu_id}", position=gpu_id):
        ppl = compute_ppl_single_seq(seqs[idx], model, alphabet, device, batch_masks)
        result_dict[idx] = ppl


def compute_pseudo_perplexity(seqs, model, alphabet, device, batch_masks=32, max_len=1022):
    """Single-GPU pPPL computation."""
    ppls = []
    for seq_i, seq in enumerate(tqdm(seqs, desc="Computing pPPL")):
        ppl = compute_ppl_single_seq(seq, model, alphabet, device, batch_masks, max_len)
        ppls.append(ppl)
        if (seq_i + 1) % 20 == 0:
            print(f"  [{seq_i+1}/{len(seqs)}] running mean pPPL = {np.mean(ppls):.4f}")
    return ppls


def compute_pseudo_perplexity_multi_gpu(seqs, model_size, gpu_ids, batch_masks=32):
    """Multi-GPU pPPL computation using multiprocessing."""
    n_seqs = len(seqs)
    n_gpus = len(gpu_ids)

    # Split sequence indices across GPUs
    chunks = []
    chunk_size = (n_seqs + n_gpus - 1) // n_gpus
    for i in range(n_gpus):
        start = i * chunk_size
        end = min(start + chunk_size, n_seqs)
        if start < end:
            chunks.append(list(range(start, end)))

    print(f"  Distributing {n_seqs} sequences across {len(chunks)} GPUs: {gpu_ids[:len(chunks)]}")
    for i, chunk in enumerate(chunks):
        print(f"    GPU {gpu_ids[i]}: {len(chunk)} sequences")

    # Use multiprocessing with shared dict
    manager = mp.Manager()
    result_dict = manager.dict()

    processes = []
    for i, chunk in enumerate(chunks):
        p = mp.Process(
            target=worker_fn,
            args=(gpu_ids[i], model_size, chunk, seqs, batch_masks, result_dict)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Collect results in order
    ppls = [result_dict[i] for i in range(n_seqs)]
    return ppls


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csvs', type=str, nargs='+', required=True,
                        help='CSV files with sequences (column: sequence)')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                        help='Labels for each CSV (for display). Default: filenames')
    parser.add_argument('--model', type=str, default='3B', choices=['150M', '650M', '3B'],
                        help='ESM2 model size for evaluation (default: 3B)')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0],
                        help='GPU IDs for parallel evaluation (default: [0])')
    parser.add_argument('--batch_masks', type=int, default=32,
                        help='Number of masked positions per forward pass')
    parser.add_argument('--max_seqs', type=int, default=None,
                        help='Max sequences to evaluate per CSV (for quick testing)')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Save per-sequence pPPL to CSV')
    args = parser.parse_args()

    if args.labels is None:
        args.labels = [os.path.splitext(os.path.basename(f))[0] for f in args.input_csvs]
    assert len(args.labels) == len(args.input_csvs), "Number of labels must match number of input CSVs"

    multi_gpu = len(args.gpu_ids) > 1

    # For single GPU, load model once
    if not multi_gpu:
        device = f"cuda:{args.gpu_ids[0]}"
        model, alphabet = load_esm2_model(args.model, device)

    all_results = []

    for csv_path, label in zip(args.input_csvs, args.labels):
        print(f"\n{'='*60}")
        print(f"Evaluating: {label} ({csv_path})")
        print(f"{'='*60}")

        df = pd.read_csv(csv_path)
        seqs = df['sequence'].tolist()
        if args.max_seqs is not None:
            seqs = seqs[:args.max_seqs]
        print(f"  {len(seqs)} sequences")

        if multi_gpu:
            ppls = compute_pseudo_perplexity_multi_gpu(
                seqs, args.model, args.gpu_ids, args.batch_masks
            )
        else:
            ppls = compute_pseudo_perplexity(
                seqs, model, alphabet, device, args.batch_masks
            )

        ppls_arr = np.array(ppls)
        log_ppls = np.log(ppls_arr)

        print(f"\n  Results for [{label}]:")
        print(f"    pPPL  mean ± std:   {ppls_arr.mean():.4f} ± {ppls_arr.std():.4f}")
        print(f"    pPPL  median:       {np.median(ppls_arr):.4f}")
        print(f"    pPPL  min/max:      {ppls_arr.min():.4f} / {ppls_arr.max():.4f}")
        print(f"    log(pPPL) mean±std: {log_ppls.mean():.4f} ± {log_ppls.std():.4f}")
        print(f"    Avg seq length:     {np.mean([len(s) for s in seqs]):.1f}")

        for i, (s, p) in enumerate(zip(seqs, ppls)):
            all_results.append({
                'group': label,
                'seq_idx': i,
                'sequence': s,
                'length': len(s),
                'ppl': p,
                'log_ppl': np.log(p),
            })

    # Summary comparison
    print(f"\n{'='*60}")
    print(f"Summary Comparison (ESM2-{args.model} pPPL)")
    print(f"{'='*60}")
    print(f"{'Group':<25} {'N':>5} {'pPPL mean':>12} {'pPPL med':>12} {'log(pPPL)':>12}")
    print(f"{'-'*66}")
    results_df = pd.DataFrame(all_results)
    for label in args.labels:
        grp = results_df[results_df['group'] == label]
        print(f"{label:<25} {len(grp):>5} {grp['ppl'].mean():>12.4f} {grp['ppl'].median():>12.4f} {grp['log_ppl'].mean():>12.4f}")

    # Save results
    if args.output_csv is None:
        args.output_csv = f"results/ppl_eval_{args.model}.csv"
    results_df.to_csv(args.output_csv, index=False)
    print(f"\nPer-sequence results saved to {args.output_csv}")
    print(f"{'='*60}")
