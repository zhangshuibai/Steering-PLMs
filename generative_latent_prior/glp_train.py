import logging
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch
from datasets import Dataset
from omegaconf import ListConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from glp.denoiser import Normalizer, GLP
from glp import flow_matching
from glp.utils_acts import MemmapReader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainConfig:
    # model
    model_name: str = ""
    glp_kwargs: Optional[Any] = None
    # data
    shuffle: bool = True
    train_dataset: str = ""
    rep_statistic: str = ""
    # training
    use_bf16: bool = True
    num_epochs: int = 1
    epoch_size: Optional[int] = None
    batch_size: int = 4096
    learning_rate: float = 5e-5
    lr_scheduler: Optional[dict] = None
    gradient_accumulation_steps: int = 1
    gradient_clipping_threshold: float = 1
    # logging and saving
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 1000
    eval_n_samples: int = 8192
    eval_num_timesteps: int = 20
    save_every_n_steps: Optional[int] = None
    save_epochs: Optional[List[int]] = None
    save_opt_state: bool = False
    output_path: Optional[Path] = None
    # wandb
    wandb_enabled: bool = False
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

class ActDataset(Dataset):
    def __init__(self, reader: MemmapReader | list[MemmapReader]):
        reader = [reader] if not isinstance(reader, (list, ListConfig)) else reader
        self.reader = reader

    def __len__(self):
        return len(self.reader[0])

    def __getitem__(self, idx):
        batch = {}
        # handle multi_layer model
        # folders should be of the form layer_<idx>
        # also need to set multi_layer_n_layers in glp_kwargs
        # for this to actually be used by denoiser
        layer_match = re.search(r"layer_(\d+)", str(self.reader[0].data_dir))
        if layer_match:
            batch["layer_idx"] = int(layer_match.group(1))
        # prepare latents
        # latents should be saved as (dim,)
        latents = [
            torch.tensor(reader[idx])[None, :]
            for r, reader in enumerate(self.reader)
        ]
        # handle special multi-reader case
        # e.g., concat different features from different readers
        # not currently used but useful for conditional modeling
        latents = torch.cat(latents, dim=-1)
        # handle data saved in half rather than full precision
        latents = latents.view(torch.bfloat16) if latents.dtype == torch.int16 else latents
        latents = latents.float()
        batch["activations"] = latents
        return batch


class PartFileDataset(Dataset):
    """
    直接读取 extract_esm2_activations.py 产生的 worker part 文件,
    跳过 memmap merge. 每个 part 文件是 (N, dim) 的 numpy array.

    目录结构:
        data_dir/
            worker_0/part_0000.npy, part_0001.npy, ...
            worker_1/...
            ...
    """
    def __init__(self, data_dir):
        import bisect
        data_dir = Path(data_dir)
        self.mmaps = []
        self.cumulative_sizes = [0]

        worker_dirs = sorted(data_dir.glob("worker_*"))
        for wd in worker_dirs:
            n_parts = int(np.load(wd / "n_parts.npy")[0])
            for p in range(n_parts):
                part_path = wd / f"part_{p:04d}.npy"
                mm = np.load(part_path, mmap_mode='r')
                self.mmaps.append(mm)
                self.cumulative_sizes.append(self.cumulative_sizes[-1] + mm.shape[0])

        self.total_size = self.cumulative_sizes[-1]
        logger.info(f"PartFileDataset: {len(self.mmaps)} parts, "
                     f"{self.total_size:,} activations from {data_dir}")

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        import bisect
        part_idx = bisect.bisect_right(self.cumulative_sizes, idx) - 1
        local_idx = idx - self.cumulative_sizes[part_idx]
        row = self.mmaps[part_idx][local_idx]
        latents = torch.tensor(np.array(row), dtype=torch.float32)[None, :]
        return {"activations": latents}


class StreamingPartDataset(torch.utils.data.IterableDataset):
    """
    流式读取 part 文件: 每次加载一个 part 到内存, shuffle 后直接 yield 整个 batch.
    避免逐样本 yield 的 Python 开销, 大幅提升数据加载速度.

    关键优化: 直接 yield (batch_size, 1, dim) 的 tensor, 跳过 DataLoader 的 collation.
    """
    def __init__(self, data_dir, batch_size=4096, seed=42):
        data_dir = Path(data_dir)
        self.part_files = []
        self.total_size = 0

        worker_dirs = sorted(data_dir.glob("worker_*"))
        for wd in worker_dirs:
            n_parts = int(np.load(wd / "n_parts.npy")[0])
            for p in range(n_parts):
                part_path = wd / f"part_{p:04d}.npy"
                self.part_files.append(part_path)
                mm = np.load(part_path, mmap_mode='r')
                self.total_size += mm.shape[0]

        self.batch_size = batch_size
        self.seed = seed
        self.n_batches = self.total_size // batch_size
        logger.info(f"StreamingPartDataset: {len(self.part_files)} parts, "
                     f"{self.total_size:,} activations, "
                     f"{self.n_batches} batches of {batch_size} from {data_dir}")

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        rng = np.random.RandomState(self.seed)
        part_order = rng.permutation(len(self.part_files))

        # 跨 part 的余量缓冲区
        remainder = None

        for pi in part_order:
            data = np.load(self.part_files[pi])  # (N, 1280)
            # part 内 shuffle
            indices = rng.permutation(data.shape[0])
            data = data[indices]

            # 如果有上一个 part 的余量, 拼接到前面
            if remainder is not None:
                data = np.concatenate([remainder, data], axis=0)
                remainder = None

            # 切成 batch_size 大小的块, 直接 yield tensor
            n = data.shape[0]
            n_full_batches = n // self.batch_size
            for i in range(n_full_batches):
                start = i * self.batch_size
                end = start + self.batch_size
                batch = torch.from_numpy(data[start:end].copy()).float()  # (B, 1280)
                yield batch.unsqueeze(1)  # (B, 1, 1280)

            # 余量留给下一个 part
            leftover = n - n_full_batches * self.batch_size
            if leftover > 0:
                remainder = data[n_full_batches * self.batch_size:]

class ActivationCollator:
    def __init__(self, normalizer: Normalizer):
        self.normalizer = normalizer

    @torch.no_grad()
    def __call__(self, rows):
        batch = {}
        # handle multi_layer model
        if 'layer_idx' in rows[0]:
            layer_idx = torch.tensor([row['layer_idx'] for row in rows], dtype=torch.long)
            batch['layer_idx'] = layer_idx
        else:
            layer_idx = None
        # prepare latents
        latents = torch.stack([row['activations'] for row in rows], dim=0)
        batch['latents'] = self.normalizer.normalize(latents, layer_idx=layer_idx)
        return batch

def load_activation_dataset(
    dataset_paths: str | list[str],
    batch_size: int = 4096,
):
    dataset_paths = [dataset_paths] if isinstance(dataset_paths, str) else dataset_paths
    datasets = []
    for path in dataset_paths:
        path = Path(path)
        worker_dirs = sorted(path.glob("worker_*"))
        if worker_dirs and (worker_dirs[0] / "n_parts.npy").exists():
            logger.info(f"Detected part-file format at {path}, using streaming loader")
            dataset = StreamingPartDataset(path, batch_size=batch_size)
        else:
            dtype_path = path / "dtype.txt"
            dtype = np.dtype(dtype_path.read_text().strip().replace('np.', ''))
            reader = MemmapReader(path, dtype)
            dataset = ActDataset(reader=reader)
        datasets.append(dataset)
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)

def get_activation_dataloader(
    dataset,
    batch_size: int,
    normalizer: Normalizer,
    shuffle: bool = True,
):
    if isinstance(dataset, StreamingPartDataset):
        # StreamingPartDataset 已经 yield 预组好的 batch tensor,
        # 用 batch_size=None 关闭 DataLoader 的自动 batching
        return DataLoader(
            dataset,
            batch_size=None,  # 不自动 batch, 直接用 dataset yield 的
            collate_fn=None,
            num_workers=0,
            pin_memory=True,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            collate_fn=ActivationCollator(normalizer),
            num_workers=0,
            pin_memory=False,
        )

def linear_scheduler(step, max_steps, initial_factor, final_factor):
    alpha = step / max_steps
    return alpha * final_factor + (1 - alpha) * initial_factor

def linear_scheduler_with_warmup(step, *, warmup_steps, max_steps, initial_factor, final_factor):
    if step < warmup_steps:
        return linear_scheduler(step, warmup_steps, initial_factor, 1.0)
    elif step >= max_steps:
        return final_factor
    else:
        return linear_scheduler(step - warmup_steps, max_steps - warmup_steps, 1.0, final_factor)

def cosine_scheduler(step, max_steps, initial_factor, final_factor):
    alpha = step / max_steps
    cosine_out = 0.5 * (1 + math.cos(math.pi * alpha))
    return final_factor + (initial_factor - final_factor) * cosine_out

def cosine_scheduler_with_warmup(step, *, warmup_steps, max_steps, initial_factor, final_factor):
    if step < warmup_steps:
        return linear_scheduler(step, warmup_steps, initial_factor, 1.0)
    elif step >= max_steps:
        return final_factor
    else:
        return cosine_scheduler(step - warmup_steps, max_steps - warmup_steps, 1.0, final_factor)

# ==========================
#   Evaluation Metrics
# ==========================

def load_eval_buffer(data_dir, n_samples, normalizer, device):
    """
    从所有 worker 的所有 part 文件中均匀采样 n_samples 个样本作为 eval buffer.
    每个 part 取相同数量的样本, 确保覆盖数据分布的多样性.
    返回 normalized latents (n_samples, 1, dim).
    """
    data_dir = Path(data_dir)
    worker_dirs = sorted(data_dir.glob("worker_*"))

    # 收集所有 part 文件路径
    all_parts = []
    for wd in worker_dirs:
        n_parts = int(np.load(wd / "n_parts.npy")[0])
        for p in range(n_parts):
            all_parts.append(wd / f"part_{p:04d}.npy")

    # 每个 part 取 samples_per_part 个 (从末尾取, 避免和训练早期重叠)
    samples_per_part = max(1, n_samples // len(all_parts))
    rng = np.random.RandomState(seed=12345)

    chunks = []
    total = 0
    for part_path in all_parts:
        mm = np.load(part_path, mmap_mode='r')
        n_rows = mm.shape[0]
        k = min(samples_per_part, n_rows)
        # 随机选取 k 个索引
        indices = rng.choice(n_rows, size=k, replace=False)
        chunks.append(np.array(mm[indices]))
        total += k
        if total >= n_samples:
            break

    data = np.concatenate(chunks, axis=0)[:n_samples]  # (n_samples, 1280)
    raw = torch.tensor(data, dtype=torch.float32).unsqueeze(1)  # (N, 1, dim)
    normalized = normalizer.normalize(raw.to(device))
    logger.info(f"Eval buffer: {data.shape[0]} samples from {len(chunks)} parts "
                 f"({samples_per_part} per part)")
    return normalized, raw.to(device)


@torch.no_grad()
def compute_mmd(x, y, sigma=1.0):
    """
    MMD (Maximum Mean Discrepancy) with RBF kernel.
    x, y: (N, dim) tensors.
    """
    def rbf_kernel(a, b, sigma):
        # (N, M)
        dist = torch.cdist(a, b, p=2).pow(2)
        return torch.exp(-dist / (2 * sigma ** 2))

    xx = rbf_kernel(x, x, sigma).mean()
    yy = rbf_kernel(y, y, sigma).mean()
    xy = rbf_kernel(x, y, sigma).mean()
    return (xx + yy - 2 * xy).item()


def compute_fid(real, fake):
    """
    FID (Fréchet Distance) 直接在激活空间计算.
    实现来自 GLP repo 的 script_eval.py (参考 clean-fid, 用 scipy.linalg.sqrtm).
    real, fake: (N, dim) tensors.
    """
    from scipy import linalg
    real_np, fake_np = real.cpu().numpy(), fake.cpu().numpy()
    mu1, sig1 = np.mean(real_np, axis=0), np.cov(real_np, rowvar=False)
    mu2, sig2 = np.mean(fake_np, axis=0), np.cov(fake_np, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sig1.dot(sig2), disp=False)
    if not np.isfinite(covmean).all():
        eps = 1e-6
        offset = np.eye(sig1.shape[0]) * eps
        covmean = linalg.sqrtm((sig1 + offset).dot(sig2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sig1) + np.trace(sig2) - 2 * np.trace(covmean))


@torch.no_grad()
def compute_nll_hutchinson(model, latents, num_timesteps=100, n_hutchinson=5):
    """
    用 Hutchinson trace estimator 近似 flow matching 的 NLL.

    NLL = -log p(x) ≈ 0.5 ||x||² + ∫₀¹ Tr(∂v/∂z) dt

    其中 Tr(∂v/∂z) 通过 Hutchinson estimator 近似:
        Tr(J) ≈ E_ε[ε^T J ε]  where ε ~ N(0, I)

    Args:
        model: GLP model
        latents: (N, 1, dim) normalized latents
        num_timesteps: ODE 积分步数
        n_hutchinson: Hutchinson 估计的随机向量数
    Returns:
        mean NLL (scalar)
    """
    device = latents.device
    batch_size = latents.shape[0]
    dim = latents.shape[-1]

    # 从 t=0 (data) 到 t=1 (noise) 的 ODE
    # log p(x) = log p(z_1) - ∫₀¹ Tr(∂v/∂z_t) dt
    # log p(z_1) = -0.5 * ||z_1||² - 0.5 * dim * log(2π)  (标准高斯)
    # NLL = -log p(x) = 0.5 * ||z_1||² + 0.5 * dim * log(2π) + ∫₀¹ Tr(∂v/∂z_t) dt

    dt = 1.0 / num_timesteps
    z_t = latents.clone()  # (N, 1, dim)
    log_det_sum = torch.zeros(batch_size, device=device)

    model.scheduler.set_timesteps(model.scheduler.config.num_train_timesteps)

    for step in range(num_timesteps):
        t = step * dt  # 从 0 → 1 (ODE: data → noise)
        # scheduler.timesteps 是降序: index=0 → sigma=1(noise), index=last → sigma=0(data)
        # ODE time t=0 对应 sigma=0(data), t=1 对应 sigma=1(noise)
        # 所以要用 (1-t) 映射: t=0 → index=last → sigma=0, t=1 → index=0 → sigma=1
        t_tensor = torch.full((batch_size,), 1.0 - t, device=device)
        indices = (t_tensor * (len(model.scheduler.timesteps) - 1)).long().clamp(0, len(model.scheduler.timesteps) - 1)
        timesteps = model.scheduler.timesteps.to(device)[indices]
        timesteps = timesteps[:, None, None]

        # Hutchinson trace estimator
        trace_est = torch.zeros(batch_size, device=device)
        for _ in range(n_hutchinson):
            eps = torch.randn_like(z_t)  # (N, 1, dim)
            z_t_input = z_t.detach().requires_grad_(True)

            with torch.enable_grad():
                v = model.denoiser(latents=z_t_input, timesteps=timesteps)
                # 计算 ε^T (∂v/∂z) ε via vjp
                vjp = torch.autograd.grad(
                    outputs=v, inputs=z_t_input,
                    grad_outputs=eps,
                    create_graph=False, retain_graph=False
                )[0]
            # ε^T J ε: 对每个 sample 分别求和 (flatten seq+dim → sum)
            trace_est += (vjp * eps).flatten(start_dim=1).sum(dim=-1)  # (N,)

        trace_est = trace_est / n_hutchinson
        log_det_sum += trace_est * dt

        # Euler 积分: z_{t+dt} = z_t + v(z_t, t) * dt
        with torch.no_grad():
            v = model.denoiser(latents=z_t, timesteps=timesteps)
            z_t = z_t + v * dt

    # z_1 ≈ z_t, 计算 log p(z_1) under N(0,I)
    # log p(z_1) under N(0,I): 对每个 sample 分别计算
    log_pz1 = -0.5 * z_t.pow(2).flatten(start_dim=1).sum(dim=-1) - 0.5 * dim * math.log(2 * math.pi)  # (N,)

    # log p(x) = log p(z_1) - ∫ Tr(∂v/∂z) dt
    log_px = log_pz1 - log_det_sum
    nll = -log_px  # (N,)

    return nll.mean().item(), nll.std().item()


@torch.no_grad()
def run_evaluation(model, eval_latents, config, device):
    """
    运行完整评估: NLL, FID, MMD.
    eval_latents: (N, 1, dim) normalized real latents.
    """
    n_eval = eval_latents.shape[0]
    dim = eval_latents.shape[-1]
    metrics = {}

    # ---- 0. Eval loss: 在 eval 数据上计算与 train 相同的 MSE loss ----
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(latents=eval_latents)
    metrics['eval/loss'] = outputs.loss.item()

    # ---- 1. 用 GLP 从纯噪声生成 fake samples ----
    noise = torch.randn(n_eval, 1, dim, device=device)
    generated = flow_matching.sample(
        model, noise,
        num_timesteps=config.eval_num_timesteps,
    )  # (N, 1, dim)

    real_flat = eval_latents.squeeze(1)   # (N, dim)
    fake_flat = generated.squeeze(1)      # (N, dim)

    # ---- 2. FID ----
    # 高维 FID 需要降维, 用 PCA 到 128 维
    combined = torch.cat([real_flat, fake_flat], dim=0)  # (2N, dim)
    mean = combined.mean(dim=0)
    combined_centered = combined - mean
    # SVD 降维
    U, S, Vh = torch.linalg.svd(combined_centered, full_matrices=False)
    proj = Vh[:128]  # (128, dim)
    real_proj = (real_flat - mean) @ proj.T  # (N, 128)
    fake_proj = (fake_flat - mean) @ proj.T  # (N, 128)
    metrics['eval/fid'] = compute_fid(real_proj, fake_proj)

    # ---- 3. MMD ----
    # 用 PCA 降维后的数据, 自适应 sigma
    with torch.no_grad():
        dists = torch.cdist(real_proj[:500], real_proj[:500], p=2)
        sigma = dists.median().item()
    metrics['eval/mmd'] = compute_mmd(real_proj[:2048], fake_proj[:2048], sigma=sigma)

    # ---- 4. NLL (Hutchinson) ----
    # NLL 计算较慢, 用少量样本 + 少步数
    nll_samples = min(256, n_eval)
    nll_mean, nll_std = compute_nll_hutchinson(
        model, eval_latents[:nll_samples],
        num_timesteps=20, n_hutchinson=1
    )
    metrics['eval/nll_mean'] = nll_mean
    metrics['eval/nll_std'] = nll_std

    # ---- 5. 基础统计: 生成样本 vs 真实样本的 mean/std 偏差 ----
    metrics['eval/gen_mean_err'] = (fake_flat.mean(dim=0) - real_flat.mean(dim=0)).abs().mean().item()
    metrics['eval/gen_std_err'] = (fake_flat.std(dim=0) - real_flat.std(dim=0)).abs().mean().item()

    return metrics


def main(device="cuda:0"):
    config_base = OmegaConf.structured(TrainConfig())
    OmegaConf.set_struct(config_base, False)
    config_cli = OmegaConf.from_cli()
    config_path = config_cli.pop("config", None)
    config_file = OmegaConf.load(config_path) if config_path else OmegaConf.create()
    config = OmegaConf.merge(config_base, config_file, config_cli)

    # setup output path
    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving checkpoints to {output_path}")
    OmegaConf.save(config, output_path / "config.yaml")

    # wait for rep_statistic from producer
    rep_statistic = config.glp_kwargs.get("normalizer_config", {}).get("rep_statistic")
    if rep_statistic:
        if os.path.exists(rep_statistic):
            logger.info(f"Waiting for rep_statistic {rep_statistic}...")
            while not os.path.exists(rep_statistic):
                time.sleep(5)

    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    logger.info(f"Config: {config}")

    # init wandb
    wandb_run = None
    if config.wandb_enabled:
        import wandb
        wandb_run = wandb.init(
            entity=config.wandb_entity,
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=OmegaConf.to_container(config),
        )

    # load model
    model = GLP(**config.glp_kwargs)
    model.to(device)
    logger.info(f"Model param count: {sum(p.numel() for p in model.parameters())}")

    # load dataset
    train_dataset = load_activation_dataset(config.train_dataset, batch_size=config.batch_size)
    train_dataloader = get_activation_dataloader(
        dataset=train_dataset,
        batch_size=config.batch_size // config.gradient_accumulation_steps,
        normalizer=model.normalizer,
        shuffle=config.shuffle,
    )

    # 加载 eval buffer (从数据集最后一个 part 取样本)
    eval_latents, _ = load_eval_buffer(
        config.train_dataset, config.eval_n_samples, model.normalizer, device
    )
    logger.info(f"Eval buffer loaded: {eval_latents.shape}")

    # setup optimizer and scheduler
    epoch_size = (config.epoch_size // config.batch_size) if config.epoch_size else len(train_dataloader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    if config.lr_scheduler is None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1)
    else:
        total_num_steps = config.num_epochs * (epoch_size // config.gradient_accumulation_steps)
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

    # 训练前评估 (作为 step=0 的 baseline)
    logger.info("Running evaluation before training (step 0)...")
    model.eval()
    eval_metrics_0 = run_evaluation(model, eval_latents, config, device)
    log_parts = ["Step 0 (before training)"]
    for k, v in eval_metrics_0.items():
        log_parts.append(f"{k}={v:.4f}")
    logger.info(" | ".join(log_parts))
    if wandb_run is not None:
        wandb_run.log(eval_metrics_0, step=0)

    # training loop
    train_steps = 0
    num_gradient_steps = 0

    for epoch in range(config.num_epochs):
        model.train()
        gradient_steps_in_epoch = epoch_size // config.gradient_accumulation_steps
        pbar = tqdm(
            total=gradient_steps_in_epoch,
            desc=f"Training Epoch: {epoch + 1}",
            dynamic_ncols=True,
        )
        is_streaming = isinstance(train_dataset, StreamingPartDataset)
        for step, batch in enumerate(train_dataloader):
            # StreamingPartDataset yield 的是原始 tensor (B, 1, dim), 需要手动 normalize
            # 非 streaming 的 DataLoader 已经在 collator 里做了 normalize, yield dict
            if is_streaming:
                latents = model.normalizer.normalize(batch.to(device))
                batch = {"latents": latents}
            else:
                batch = {k: v.to(device) if v is not None else None for k, v in batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=config.use_bf16):
                outputs = model(**batch)
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
                    f"Epoch: {epoch + 1}/{config.num_epochs}, "
                    f"batch {step + 1}/{epoch_size} "
                    f"(loss: {loss.detach().float():.4f})"
                )

                if num_gradient_steps % config.log_every_n_steps == 0:
                    avg_loss = loss.detach().item()
                    # epoch 进度: 线性增长 0 → 1
                    epoch_progress = epoch + (step + 1) / (gradient_steps_in_epoch * config.gradient_accumulation_steps)
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/epoch": epoch_progress,
                                "train/step": num_gradient_steps,
                                "train/loss": avg_loss,
                                "train/learning_rate": scheduler.get_last_lr()[0],
                            },
                            step=num_gradient_steps
                        )

                # ---- 定期评估 ----
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

                if config.save_every_n_steps and num_gradient_steps % config.save_every_n_steps == 0:
                    save_checkpoint(model, output_path, f"step_{num_gradient_steps}", optimizer, scheduler, save_opt_state=config.save_opt_state)

            if step >= gradient_steps_in_epoch * config.gradient_accumulation_steps:
                break

        pbar.close()

        # save epoch checkpoint
        if config.save_epochs and (epoch + 1) in set(config.save_epochs):
            save_checkpoint(model, output_path / "checkpoints", f"epoch_{epoch + 1}")

        # always save latest checkpoint
        save_checkpoint(model, output_path, "final", optimizer, scheduler, save_opt_state=config.save_opt_state)

    # 训练结束后最终评估
    logger.info("Final evaluation...")
    model.eval()
    final_metrics = run_evaluation(model, eval_latents, config, device)
    log_parts = ["FINAL"]
    for k, v in final_metrics.items():
        log_parts.append(f"{k}={v:.4f}")
    logger.info(" | ".join(log_parts))
    if wandb_run is not None:
        wandb_run.log(final_metrics, step=num_gradient_steps)
        wandb.finish()

def save_checkpoint(model, output_path, checkpoint_name, optimizer=None, scheduler=None, save_opt_state=False):
    model.save_pretrained(path=output_path, name=checkpoint_name)
    logger.info(f"Model saved to {output_path}/{checkpoint_name}")
    if save_opt_state:
        if optimizer is not None:
            torch.save(optimizer.state_dict(), output_path / "optimizer_state.pt")
        if scheduler is not None:
            torch.save(scheduler.state_dict(), output_path / "scheduler_state.pt")

if __name__ == "__main__":
    main()
