"""
Shared components for evaluation: PropertyPredictor model and ESM2 feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import esm


class PropertyPredictor(nn.Module):
    """RobertaLMHead-style: Linear -> GELU -> LayerNorm -> Linear(1)."""
    def __init__(self, embed_dim=1280):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.dense(x)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.out_proj(x)
        return x.squeeze(-1)


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


def extract_features(seqs, model, alphabet, device, batch_size=8, max_len=1022):
    """Extract mean-pooled last-layer representations from ESM2."""
    batch_converter = alphabet.get_batch_converter()
    n_layers = model.num_layers
    all_features = []

    n_batches = (len(seqs) + batch_size - 1) // batch_size
    for start in tqdm(range(0, len(seqs), batch_size), total=n_batches, desc="Extracting features"):
        batch_seqs = seqs[start:start + batch_size]
        batch_seqs = [s[:max_len] for s in batch_seqs]
        data = [("protein", s) for s in batch_seqs]
        _, _, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[n_layers])

        for i, seq_len in enumerate(batch_lens):
            rep = results["representations"][n_layers][i, 1:seq_len - 1].mean(0).cpu()
            all_features.append(rep)

    return torch.stack(all_features)


def load_predictor(predictor_path, embed_dim=1280, device='cpu'):
    """Load a trained PropertyPredictor from checkpoint."""
    predictor = PropertyPredictor(embed_dim=embed_dim)
    ckpt = torch.load(predictor_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        predictor.load_state_dict(ckpt['model_state_dict'])
    else:
        predictor.load_state_dict(ckpt)
    predictor = predictor.to(device)
    predictor.eval()
    return predictor


def evaluate_sol(seqs, esm_model, alphabet, predictor, device):
    """Evaluate solubility: returns (mean_prob, sol_ratio, probs_array)."""
    features = extract_features(seqs, esm_model, alphabet, device)
    with torch.no_grad():
        scores = predictor(features.to(device)).cpu()
    probs = torch.sigmoid(scores).numpy()
    labels = (probs >= 0.5).astype(int)
    return probs.mean(), labels.mean(), probs
