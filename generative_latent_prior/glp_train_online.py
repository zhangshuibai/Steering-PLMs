"""
Online GLP Training: ESM2 generates activations on-the-fly, no disk storage needed.

Supports both single-GPU and multi-GPU modes:
  - Single GPU: ESM2 + GLP on same GPU
  - Multi GPU:  ESM2 workers on multiple GPUs produce activations via Queue,
                GLP trains on one GPU consuming from Queue (~linear speedup)

Usage:
    # Single GPU
    CUDA_VISIBLE_DEVICES=4 python generative_latent_prior/glp_train_online.py \
        --config generative_latent_prior/configs/glp-esm2-3b-layer18-d6-online.yaml

    # Multi GPU (4x speedup)
    python generative_latent_prior/glp_train_online.py \
        --config generative_latent_prior/configs/glp-esm2-3b-layer18-d6-online.yaml \
        --gpu_ids 4 5 6 7

    # Evaluate existing checkpoint
    python generative_latent_prior/glp_train_online.py \
        --eval_existing generative_latent_prior/runs/glp-esm2-650m-layer17-d6 \
        --esm_model_size 650M --extract_layer 17 --gpu_ids 5
"""

import argparse
import logging
import math
import os
import json
import sys
import time
import queue as queue_module
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from tqdm import tqdm

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_esm2_activations import read_fasta_sequences, make_dynamic_batches
from glp.denoiser import Normalizer, GLP
from glp import flow_matching
from glp_train import (
    run_evaluation,
    save_checkpoint,
    cosine_scheduler_with_warmup,
    linear_scheduler_with_warmup,
    cosine_scheduler,
    linear_scheduler,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================
#   ESM2 Model Loading
# ==========================

ESM_MODEL_REGISTRY = {
    "650M": ("esm2_t33_650M_UR50D", 1280),
    "3B":   ("esm2_t36_3B_UR50D",   2560),
}


def load_esm_model(model_size, device):
    """Load frozen ESM2 model."""
    import esm

    if model_size not in ESM_MODEL_REGISTRY:
        raise ValueError(f"Unknown ESM2 model size: {model_size}. Choose from {list(ESM_MODEL_REGISTRY.keys())}")

    model_name, hidden_dim = ESM_MODEL_REGISTRY[model_size]
    logger.info(f"Loading ESM2-{model_size} ({model_name}, hidden_dim={hidden_dim})...")

    model, alphabet = getattr(esm.pretrained, model_name)()
    model.to(device).eval()
    model.token_dropout = False
    for p in model.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"ESM2-{model_size} loaded: {n_params/1e9:.2f}B params, "
                f"GPU mem: {torch.cuda.memory_allocated(device)/1e9:.1f} GB")
    return model, alphabet, hidden_dim


# ==========================
#   Online Statistics
# ==========================

@torch.no_grad()
def compute_online_statistics(esm_model, alphabet, sequences, layer,
                               device, max_tokens, hidden_dim, n_tokens=100000):
    """
    Compute mean and var of ESM2 activations online from a subset of sequences.
    Returns rep_statistics dict compatible with Normalizer.
    """
    batch_converter = alphabet.get_batch_converter()
    batches = make_dynamic_batches(sequences, max_tokens)

    running_sum = torch.zeros(hidden_dim, dtype=torch.float64, device=device)
    running_sq_sum = torch.zeros(hidden_dim, dtype=torch.float64, device=device)
    total_tokens = 0

    logger.info(f"Computing rep_statistics online (target: {n_tokens} tokens)...")
    pbar = tqdm(batches, desc="Computing statistics")
    for batch_seqs in pbar:
        data = [("protein", s) for s in batch_seqs]
        _, _, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to(device)

        results = esm_model(batch_tokens, repr_layers=[layer])
        reps = results["representations"][layer]  # (B, T, dim)

        for i, seq_len in enumerate(batch_lens):
            acts = reps[i, 1:seq_len - 1].double()  # exclude BOS/EOS
            running_sum += acts.sum(0)
            running_sq_sum += (acts ** 2).sum(0)
            total_tokens += acts.shape[0]

        pbar.set_postfix(tokens=total_tokens)
        if total_tokens >= n_tokens:
            break
    pbar.close()

    mean = running_sum / total_tokens
    var = running_sq_sum / total_tokens - mean ** 2

    logger.info(f"Statistics computed from {total_tokens:,} tokens. "
                f"Mean range: [{mean.min():.4f}, {mean.max():.4f}], "
                f"Var range: [{var.min():.4f}, {var.max():.4f}]")

    return {
        "mean": mean.float().unsqueeze(0).cpu(),  # (1, dim)
        "var": var.float().unsqueeze(0).cpu(),     # (1, dim)
    }


# ==========================
#   Online Activation Dataset (Single GPU)
# ==========================

class OnlineActivationDataset(torch.utils.data.IterableDataset):
    """
    Streaming dataset that generates ESM2 activations on-the-fly.
    Used in single-GPU mode.
    """
    def __init__(self, sequences, esm_model, alphabet, layer,
                 batch_size=8192, max_tokens=4096, hidden_dim=1280,
                 device='cuda:0', seed=42):
        self.sequences = sequences
        self.esm_model = esm_model
        self.alphabet = alphabet
        self.layer = layer
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.hidden_dim = hidden_dim
        self.device = device
        self.seed = seed
        self.batch_converter = alphabet.get_batch_converter()

        # Estimate total tokens for progress tracking
        self.est_total_tokens = sum(len(s) for s in sequences)
        self.n_batches_est = self.est_total_tokens // batch_size
        logger.info(f"OnlineActivationDataset: {len(sequences):,} sequences, "
                     f"~{self.est_total_tokens:,} tokens, "
                     f"~{self.n_batches_est} batches of {batch_size}")

    def __len__(self):
        return self.n_batches_est

    def __iter__(self):
        rng = np.random.RandomState(self.seed)

        # Shuffle sequences
        perm = rng.permutation(len(self.sequences))
        shuffled_seqs = [self.sequences[i] for i in perm]

        # Dynamic batching
        esm_batches = make_dynamic_batches(shuffled_seqs, self.max_tokens)
        # Shuffle batch order too (dynamic batching sorts by length)
        batch_order = rng.permutation(len(esm_batches))

        # Token buffer
        buffer = []
        buffer_size = 0

        for bi in batch_order:
            batch_seqs = esm_batches[bi]
            data = [("protein", s) for s in batch_seqs]
            _, _, batch_tokens = self.batch_converter(data)
            batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
            batch_tokens = batch_tokens.to(self.device)

            with torch.no_grad():
                results = self.esm_model(batch_tokens, repr_layers=[self.layer])
            reps = results["representations"][self.layer]  # (B, T, dim)

            for i, seq_len in enumerate(batch_lens):
                acts = reps[i, 1:seq_len - 1].float()  # (L, dim)
                buffer.append(acts)
                buffer_size += acts.shape[0]

            # Yield full batches from buffer
            while buffer_size >= self.batch_size:
                all_acts = torch.cat(buffer, dim=0)
                perm_idx = torch.randperm(all_acts.shape[0], device=self.device)
                all_acts = all_acts[perm_idx]

                yield all_acts[:self.batch_size].unsqueeze(1)  # (B, 1, dim)

                remainder = all_acts[self.batch_size:]
                if remainder.shape[0] > 0:
                    buffer = [remainder]
                    buffer_size = remainder.shape[0]
                else:
                    buffer = []
                    buffer_size = 0

        # Yield last partial batch if at least half a batch
        if buffer_size >= self.batch_size // 2:
            all_acts = torch.cat(buffer, dim=0)
            perm_idx = torch.randperm(all_acts.shape[0], device=self.device)
            all_acts = all_acts[perm_idx]
            yield all_acts.unsqueeze(1)


# ==========================
#   Multi-GPU ESM2 Worker
# ==========================

def esm_worker(worker_id, gpu_id, esm_model_size, sequences, layer,
               max_tokens, batch_size, hidden_dim, result_queue,
               n_epochs, seed):
    """
    Worker process: load ESM2 on one GPU, produce activation batches into Queue.
    Each batch is a CPU tensor (B, 1, dim) to avoid cross-GPU transfer issues.
    Sends None sentinel when done.
    """
    import esm

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    model_name, _ = ESM_MODEL_REGISTRY[esm_model_size]
    print(f"[ESM Worker {worker_id}] Loading {esm_model_size} on GPU {gpu_id}, "
          f"{len(sequences):,} sequences...", flush=True)

    model, alphabet = getattr(esm.pretrained, model_name)()
    model.to(device).eval()
    model.token_dropout = False
    for p in model.parameters():
        p.requires_grad = False

    batch_converter = alphabet.get_batch_converter()
    mem_gb = torch.cuda.memory_allocated(device) / 1e9
    print(f"[ESM Worker {worker_id}] Ready on GPU {gpu_id}. Mem: {mem_gb:.1f} GB", flush=True)

    total_batches_produced = 0

    for epoch in range(n_epochs):
        rng = np.random.RandomState(seed + epoch * 1000 + worker_id)
        perm = rng.permutation(len(sequences))
        shuffled = [sequences[i] for i in perm]

        esm_batches = make_dynamic_batches(shuffled, max_tokens)
        batch_order = rng.permutation(len(esm_batches))

        buffer = []
        buffer_size = 0

        for bi in batch_order:
            batch_seqs = esm_batches[bi]
            data = [("protein", s) for s in batch_seqs]
            _, _, batch_tokens = batch_converter(data)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[layer])
            reps = results["representations"][layer]

            for i, seq_len in enumerate(batch_lens):
                acts = reps[i, 1:seq_len - 1].float().cpu()  # to CPU for queue
                buffer.append(acts)
                buffer_size += acts.shape[0]

            while buffer_size >= batch_size:
                all_acts = torch.cat(buffer, dim=0)
                idx = torch.randperm(all_acts.shape[0])
                all_acts = all_acts[idx]

                result_queue.put(all_acts[:batch_size].unsqueeze(1))  # (B, 1, dim) CPU
                total_batches_produced += 1

                remainder = all_acts[batch_size:]
                buffer = [remainder] if remainder.shape[0] > 0 else []
                buffer_size = remainder.shape[0]

        # Discard last partial batch (avoids inconsistent batch sizes
        # that would mess up gradient_accumulation_steps loss weighting)
        if buffer_size > 0:
            print(f"[ESM Worker {worker_id}] Epoch {epoch+1}: discarded "
                  f"{buffer_size} remainder tokens", flush=True)

    result_queue.put(None)  # sentinel
    print(f"[ESM Worker {worker_id}] Done. Produced {total_batches_produced} batches.", flush=True)


# ==========================
#   Online Eval Buffer
# ==========================

@torch.no_grad()
def build_eval_buffer_online(esm_model, alphabet, sequences, layer,
                              normalizer, device, hidden_dim,
                              n_samples=8192, max_tokens=4096):
    """
    Build eval buffer by running ESM2 on a random subset of sequences.
    Returns normalized latents (n_samples, 1, dim) and raw (n_samples, 1, dim).
    """
    batch_converter = alphabet.get_batch_converter()

    rng = np.random.RandomState(12345)
    subset_idx = rng.permutation(len(sequences))
    subset_seqs = [sequences[i] for i in subset_idx]

    batches = make_dynamic_batches(subset_seqs, max_tokens)

    acts_list = []
    total = 0

    logger.info(f"Building eval buffer online (target: {n_samples} tokens)...")
    for batch_seqs in tqdm(batches, desc="Eval buffer"):
        data = [("protein", s) for s in batch_seqs]
        _, _, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to(device)

        results = esm_model(batch_tokens, repr_layers=[layer])
        reps = results["representations"][layer]

        for i, seq_len in enumerate(batch_lens):
            acts = reps[i, 1:seq_len - 1].float()
            acts_list.append(acts.cpu())
            total += acts.shape[0]

        if total >= n_samples:
            break

    all_acts = torch.cat(acts_list, dim=0)[:n_samples]
    raw = all_acts.unsqueeze(1).to(device)
    normalized = normalizer.normalize(raw)

    logger.info(f"Eval buffer: {raw.shape[0]} samples")
    return normalized, raw


# ==========================
#   Training Step (shared between single/multi GPU)
# ==========================

def do_train_step(model, batch, config, optimizer, scheduler,
                  train_steps, num_gradient_steps, device,
                  pbar, wandb_run, eval_latents, output_path):
    """
    Execute one training sub-step (forward + backward).
    Returns updated (train_steps, num_gradient_steps).
    """
    latents = model.normalizer.normalize(batch.to(device))

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=config.use_bf16):
        outputs = model(latents=latents)
        loss = outputs.loss

    loss = loss / config.gradient_accumulation_steps
    loss.backward()
    train_steps += 1

    if train_steps % config.gradient_accumulation_steps == 0:
        num_gradient_steps += 1

        if config.gradient_clipping_threshold > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clipping_threshold
            )

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        pbar.update(1)
        pbar.set_description(
            f"step {num_gradient_steps} "
            f"(loss: {loss.detach().float():.4f})"
        )

        if num_gradient_steps % config.log_every_n_steps == 0:
            avg_loss = loss.detach().item()
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/step": num_gradient_steps,
                        "train/loss": avg_loss,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                    },
                    step=num_gradient_steps,
                )

        # Periodic evaluation
        if num_gradient_steps % config.eval_every_n_steps == 0:
            logger.info(f"Running evaluation at step {num_gradient_steps}...")
            model.eval()
            eval_metrics = run_evaluation(model, eval_latents, config, device)
            model.train()

            log_parts = [f"Step {num_gradient_steps}"]
            for k, v in eval_metrics.items():
                log_parts.append(f"{k}={v:.4f}")
            logger.info(" | ".join(log_parts))

            if wandb_run is not None:
                wandb_run.log(eval_metrics, step=num_gradient_steps)

        # Periodic save
        if config.save_every_n_steps and num_gradient_steps % config.save_every_n_steps == 0:
            save_checkpoint(model, output_path, f"step_{num_gradient_steps}",
                            optimizer, scheduler, save_opt_state=config.save_opt_state)

    return train_steps, num_gradient_steps


# ==========================
#   Config
# ==========================

@dataclass
class OnlineTrainConfig:
    # ESM2 settings
    esm_model_size: str = "3B"
    fasta_path: str = ""
    extract_layer: int = 18
    max_tokens: int = 4096
    max_seqs: Optional[int] = None
    min_len: int = 30
    max_len: int = 1022
    stat_n_tokens: int = 100000

    # Multi-GPU
    gpu_ids: Optional[List[int]] = None  # e.g. [4,5,6,7]

    # model
    model_name: str = ""
    glp_kwargs: Optional[Any] = None

    # training
    use_bf16: bool = True
    num_epochs: int = 1
    batch_size: int = 8192
    learning_rate: float = 5e-5
    lr_scheduler: Optional[dict] = None
    gradient_accumulation_steps: int = 1
    gradient_clipping_threshold: float = 1.0

    # logging and saving
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 2000
    eval_n_samples: int = 8192
    eval_num_timesteps: int = 20
    save_every_n_steps: Optional[int] = None
    save_epochs: Optional[List[int]] = None
    save_opt_state: bool = False
    output_path: Optional[str] = None
    run_name: Optional[str] = None

    # wandb
    wandb_enabled: bool = False
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None


# ==========================
#   Main Training
# ==========================

def train_online(config, device="cuda:0", gpu_ids=None):
    """
    Main online training loop.
    If gpu_ids has multiple GPUs, uses multi-GPU ESM2 workers for ~linear speedup.
    """
    multi_gpu = gpu_ids is not None and len(gpu_ids) > 1
    if multi_gpu:
        glp_device = f"cuda:{gpu_ids[0]}"
        device = glp_device
        logger.info(f"Multi-GPU mode: GLP on GPU {gpu_ids[0]}, "
                     f"ESM2 workers on GPUs {gpu_ids}")
    elif gpu_ids is not None and len(gpu_ids) == 1:
        device = f"cuda:{gpu_ids[0]}"

    # Setup output
    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_path / "config.yaml")
    logger.info(f"Output: {output_path}")

    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    # 1. Load ESM2 on first GPU (for stats + eval buffer)
    esm_model, alphabet, hidden_dim = load_esm_model(config.esm_model_size, device)

    # 2. Read sequences
    sequences = read_fasta_sequences(
        config.fasta_path,
        max_seqs=config.max_seqs,
        min_len=config.min_len,
        max_len=config.max_len,
    )

    # 3. Compute or load rep_statistics
    rep_stat_path = output_path / "rep_statistics.pt"
    if rep_stat_path.exists():
        logger.info(f"Loading existing rep_statistics from {rep_stat_path}")
        rep_stats = torch.load(rep_stat_path, map_location="cpu")
    else:
        rep_stats = compute_online_statistics(
            esm_model, alphabet, sequences, config.extract_layer,
            device, config.max_tokens, hidden_dim,
            n_tokens=config.stat_n_tokens,
        )
        torch.save(rep_stats, rep_stat_path)
        logger.info(f"Saved rep_statistics to {rep_stat_path}")

    # 4. Initialize GLP
    glp_kwargs = OmegaConf.to_container(config.glp_kwargs, resolve=True)
    glp_kwargs["normalizer_config"]["rep_statistic"] = str(rep_stat_path)
    model = GLP(**glp_kwargs)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"GLP param count: {n_params:,} ({n_params/1e6:.1f}M)")

    # 5. Init wandb
    wandb_run = None
    if config.wandb_enabled:
        import wandb
        wandb_run = wandb.init(
            entity=config.wandb_entity,
            project=config.wandb_project,
            name=config.wandb_run_name or config.run_name,
            config=OmegaConf.to_container(config),
        )

    # 6. Build eval buffer online (using single ESM2)
    eval_latents, _ = build_eval_buffer_online(
        esm_model, alphabet, sequences, config.extract_layer,
        model.normalizer, device, hidden_dim,
        n_samples=config.eval_n_samples,
        max_tokens=config.max_tokens,
    )
    logger.info(f"Eval buffer: {eval_latents.shape}")

    # 7. Estimate epoch size for scheduler
    est_total_tokens = sum(len(s) for s in sequences)
    effective_batch_size = config.batch_size // config.gradient_accumulation_steps
    epoch_size = est_total_tokens // effective_batch_size
    total_num_steps = config.num_epochs * (epoch_size // config.gradient_accumulation_steps)
    logger.info(f"Estimated: ~{est_total_tokens:,} tokens, "
                f"~{epoch_size} batches/epoch, "
                f"~{total_num_steps} gradient steps total")

    # 8. Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    if config.lr_scheduler is None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=partial(
                eval(config.lr_scheduler["scheduler_cls"]),
                warmup_steps=config.lr_scheduler["warmup_ratio"] * total_num_steps,
                max_steps=total_num_steps,
                initial_factor=config.lr_scheduler["initial_factor"],
                final_factor=config.lr_scheduler["final_factor"],
            )
        )

    # 9. Pre-training evaluation (baseline)
    logger.info("Running evaluation before training (step 0)...")
    model.eval()
    eval_metrics_0 = run_evaluation(model, eval_latents, config, device)
    log_parts = ["Step 0 (before training)"]
    for k, v in eval_metrics_0.items():
        log_parts.append(f"{k}={v:.4f}")
    logger.info(" | ".join(log_parts))
    if wandb_run is not None:
        wandb_run.log(eval_metrics_0, step=0)

    # 10. Training
    if multi_gpu:
        # Free ESM2 from main process BEFORE spawning workers
        # (workers load their own copies; keeping it here wastes GPU memory)
        del esm_model, alphabet
        torch.cuda.empty_cache()
        logger.info(f"Freed ESM2 from main process. GPU mem: "
                     f"{torch.cuda.memory_allocated(device)/1e9:.1f} GB")

        _train_multi_gpu(
            model, config, optimizer, scheduler, eval_latents,
            sequences, hidden_dim, gpu_ids,
            effective_batch_size, epoch_size, total_num_steps,
            output_path, device, wandb_run,
        )
    else:
        _train_single_gpu(
            model, config, optimizer, scheduler, eval_latents,
            esm_model, alphabet, sequences, hidden_dim,
            effective_batch_size, epoch_size,
            output_path, device, wandb_run,
        )

    # Final evaluation
    logger.info("Final evaluation...")
    model.eval()
    final_metrics = run_evaluation(model, eval_latents, config, device)
    log_parts = ["FINAL"]
    for k, v in final_metrics.items():
        log_parts.append(f"{k}={v:.4f}")
    logger.info(" | ".join(log_parts))

    with open(output_path / "eval_results.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    if wandb_run is not None:
        wandb_run.log(final_metrics)
        import wandb
        wandb.finish()

    logger.info("Training complete!")


def _train_single_gpu(model, config, optimizer, scheduler, eval_latents,
                      esm_model, alphabet, sequences, hidden_dim,
                      effective_batch_size, epoch_size,
                      output_path, device, wandb_run):
    """Single-GPU training: ESM2 + GLP on same GPU."""
    dataset = OnlineActivationDataset(
        sequences, esm_model, alphabet, config.extract_layer,
        batch_size=effective_batch_size,
        max_tokens=config.max_tokens,
        hidden_dim=hidden_dim,
        device=device,
    )

    train_steps = 0
    num_gradient_steps = 0

    for epoch in range(config.num_epochs):
        model.train()
        gradient_steps_in_epoch = epoch_size // config.gradient_accumulation_steps
        pbar = tqdm(total=gradient_steps_in_epoch, desc=f"Epoch {epoch+1}", dynamic_ncols=True)

        for step, batch in enumerate(dataset):
            train_steps, num_gradient_steps = do_train_step(
                model, batch, config, optimizer, scheduler,
                train_steps, num_gradient_steps, device,
                pbar, wandb_run, eval_latents, output_path,
            )

        pbar.close()

        if config.save_epochs and (epoch + 1) in set(config.save_epochs):
            save_checkpoint(model, output_path / "checkpoints", f"epoch_{epoch + 1}")
        save_checkpoint(model, output_path, "final", optimizer, scheduler,
                        save_opt_state=config.save_opt_state)


def _train_multi_gpu(model, config, optimizer, scheduler, eval_latents,
                     sequences, hidden_dim, gpu_ids,
                     effective_batch_size, epoch_size, total_num_steps,
                     output_path, device, wandb_run):
    """
    Multi-GPU training: spawn ESM2 workers on each GPU, GLP trains on device.
    Workers produce activation batches via mp.Queue → main process trains GLP.
    ESM2 must be freed from main process before calling this function.
    """
    n_workers = len(gpu_ids)

    # Split sequences across workers (interleave for load balance)
    seqs_sorted = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
    chunks = [[] for _ in range(n_workers)]
    for i, idx in enumerate(seqs_sorted):
        chunks[i % n_workers].append(sequences[idx])
    for i, chunk in enumerate(chunks):
        total_toks = sum(len(s) for s in chunk)
        logger.info(f"  Worker {i} (GPU {gpu_ids[i]}): {len(chunk):,} seqs, ~{total_toks:,} tokens")

    # Create queue and spawn workers
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue(maxsize=n_workers * 8)

    workers = []
    for i, gpu_id in enumerate(gpu_ids):
        p = ctx.Process(
            target=esm_worker,
            args=(i, gpu_id, config.esm_model_size, chunks[i], config.extract_layer,
                  config.max_tokens, effective_batch_size, hidden_dim,
                  result_queue, config.num_epochs, 42),
            daemon=True,
        )
        p.start()
        workers.append(p)
        logger.info(f"  Spawned ESM worker {i} on GPU {gpu_id} (PID {p.pid})")

    # Training loop: consume from queue
    model.train()
    train_steps = 0
    num_gradient_steps = 0
    done_count = 0
    gradient_steps_total = total_num_steps

    pbar = tqdm(total=gradient_steps_total, desc="Training (multi-GPU)", dynamic_ncols=True)

    while done_count < n_workers:
        try:
            batch = result_queue.get(timeout=600)  # 10 min timeout
        except queue_module.Empty:
            # Check if workers are still alive
            alive = sum(1 for w in workers if w.is_alive())
            if alive == 0:
                logger.warning("All workers exited but not all sentinels received!")
                break
            logger.warning(f"Queue timeout. {alive}/{n_workers} workers still alive. Retrying...")
            continue

        if batch is None:
            done_count += 1
            logger.info(f"Worker finished ({done_count}/{n_workers} done)")
            continue

        # batch: (B, 1, dim) CPU tensor → move to GLP device
        train_steps, num_gradient_steps = do_train_step(
            model, batch, config, optimizer, scheduler,
            train_steps, num_gradient_steps, device,
            pbar, wandb_run, eval_latents, output_path,
        )

    pbar.close()

    # Wait for all workers to finish
    for w in workers:
        w.join(timeout=30)
        if w.is_alive():
            logger.warning(f"Worker PID {w.pid} didn't exit cleanly, terminating...")
            w.terminate()

    # Save
    save_checkpoint(model, output_path, "final", optimizer, scheduler,
                    save_opt_state=config.save_opt_state)


# ==========================
#   Eval Existing Checkpoint
# ==========================

def eval_existing(checkpoint_dir, esm_model_size, extract_layer, fasta_path,
                  device="cuda:0", max_tokens=4096, max_seqs=None,
                  eval_n_samples=8192, eval_num_timesteps=20):
    """
    Evaluate an existing GLP checkpoint using online-generated eval buffer.
    """
    checkpoint_dir = Path(checkpoint_dir)
    logger.info(f"Evaluating existing checkpoint: {checkpoint_dir}")

    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    esm_model, alphabet, hidden_dim = load_esm_model(esm_model_size, device)
    sequences = read_fasta_sequences(fasta_path, max_seqs=max_seqs)

    config = OmegaConf.load(checkpoint_dir / "config.yaml")
    OmegaConf.set_struct(config, False)

    rep_stat_path = checkpoint_dir / "rep_statistics.pt"
    if not rep_stat_path.exists():
        rep_stats = compute_online_statistics(
            esm_model, alphabet, sequences, extract_layer,
            device, max_tokens, hidden_dim,
        )
        torch.save(rep_stats, rep_stat_path)

    glp_kwargs = OmegaConf.to_container(config.glp_kwargs, resolve=True)
    glp_kwargs["normalizer_config"]["rep_statistic"] = str(rep_stat_path)
    model = GLP(**glp_kwargs)
    model.to(device)
    model.load_pretrained(checkpoint_dir, name="final")
    logger.info(f"Loaded GLP checkpoint from {checkpoint_dir}")

    eval_latents, _ = build_eval_buffer_online(
        esm_model, alphabet, sequences, extract_layer,
        model.normalizer, device, hidden_dim,
        n_samples=eval_n_samples,
        max_tokens=max_tokens,
    )

    model.eval()
    eval_config = OmegaConf.create({
        "eval_num_timesteps": eval_num_timesteps,
        "eval_n_samples": eval_n_samples,
    })
    metrics = run_evaluation(model, eval_latents, eval_config, device)

    logger.info("=" * 60)
    logger.info(f"Evaluation Results for {checkpoint_dir}")
    logger.info("=" * 60)
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.6f}")
    logger.info("=" * 60)

    results_path = checkpoint_dir / "eval_results_online.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    return metrics


# ==========================
#   CLI Entry Point
# ==========================

def main():
    parser = argparse.ArgumentParser(description="Online GLP Training with ESM2")
    parser.add_argument("--config", type=str, default=None, help="Config YAML file")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--gpu_ids", type=int, nargs='+', default=None,
                        help="GPU IDs for multi-GPU mode. E.g.: --gpu_ids 4 5 6 7")

    # Eval existing mode
    parser.add_argument("--eval_existing", type=str, default=None,
                        help="Path to existing GLP checkpoint dir to evaluate")
    parser.add_argument("--esm_model_size", type=str, default=None)
    parser.add_argument("--extract_layer", type=int, default=None)
    parser.add_argument("--fasta_path", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_seqs", type=int, default=None)

    args = parser.parse_args()

    if args.eval_existing:
        eval_device = f"cuda:{args.gpu_ids[0]}" if args.gpu_ids else args.device
        eval_existing(
            checkpoint_dir=args.eval_existing,
            esm_model_size=args.esm_model_size or "650M",
            extract_layer=args.extract_layer or 17,
            fasta_path=args.fasta_path or "data/uniref50/uniref50.fasta.gz",
            device=eval_device,
            max_tokens=args.max_tokens,
            max_seqs=args.max_seqs,
        )
    else:
        assert args.config is not None, "Must provide --config for training"

        config_base = OmegaConf.structured(OnlineTrainConfig())
        OmegaConf.set_struct(config_base, False)
        config_file = OmegaConf.load(args.config)
        config = OmegaConf.merge(config_base, config_file)

        # CLI overrides
        if args.esm_model_size:
            config.esm_model_size = args.esm_model_size
        if args.extract_layer:
            config.extract_layer = args.extract_layer
        if args.fasta_path:
            config.fasta_path = args.fasta_path
        if args.max_seqs:
            config.max_seqs = args.max_seqs
        if args.gpu_ids:
            config.gpu_ids = args.gpu_ids

        train_online(config, device=args.device, gpu_ids=args.gpu_ids)


if __name__ == "__main__":
    main()
