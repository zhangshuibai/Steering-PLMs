"""
Extract ESM2-650M Layer 17 activations for GLP training.

优化策略:
  1. 按序列长度排序 → 同一 batch 内长度接近, 最小化 padding 浪费
  2. 动态 batch: 按 max_tokens 控制 (而非固定 batch_size), 充分利用 GPU 显存
  3. 流式写入磁盘: 不在内存中累积所有激活, 避免 OOM
  4. 在线统计: running mean/var, 不需要两次遍历

Usage:
    python extract_esm2_activations.py \
        --fasta data/uniref50/uniref50.fasta.gz \
        --output_dir data/esm2_650m_layer17_uniref50 \
        --layer 17 --max_seqs 4000000 \
        --max_tokens 8192 \
        --gpu_ids 0 1 2 3
"""

import argparse
import gzip
import os
import sys
import numpy as np
import torch
import torch.multiprocessing as mp
from pathlib import Path
from tqdm import tqdm

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def read_fasta_sequences(fasta_path, max_seqs=None, min_len=30, max_len=1022):
    """Read sequences from FASTA file (supports .gz)."""
    open_fn = gzip.open if fasta_path.endswith('.gz') else open
    sequences = []
    current_seq = []

    print(f"Reading sequences from {fasta_path}...")
    with open_fn(fasta_path, 'rt') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    seq = ''.join(current_seq)
                    if min_len <= len(seq) <= max_len:
                        sequences.append(seq)
                    if max_seqs and len(sequences) >= max_seqs:
                        break
                current_seq = []
            else:
                current_seq.append(line)

        # Last sequence
        if current_seq and (max_seqs is None or len(sequences) < max_seqs):
            seq = ''.join(current_seq)
            if min_len <= len(seq) <= max_len:
                sequences.append(seq)

    print(f"  Read {len(sequences)} sequences (len {min_len}-{max_len})")
    return sequences


def make_dynamic_batches(seqs, max_tokens):
    """
    按长度排序后, 动态分 batch, 使每个 batch 的 total_tokens ≈ max_tokens.
    这样短序列可以组成大 batch, 长序列组成小 batch, 充分利用 GPU.
    """
    # 按长度排序 (保留原始索引用于打乱)
    indexed_seqs = sorted(enumerate(seqs), key=lambda x: len(x[1]))

    batches = []
    current_batch = []
    current_max_len = 0

    for idx, seq in indexed_seqs:
        seq_len = len(seq) + 2  # +2 for BOS/EOS tokens
        new_max_len = max(current_max_len, seq_len)
        new_total = new_max_len * (len(current_batch) + 1)

        if current_batch and new_total > max_tokens:
            batches.append([s for _, s in current_batch])
            current_batch = [(idx, seq)]
            current_max_len = seq_len
        else:
            current_batch.append((idx, seq))
            current_max_len = new_max_len

    if current_batch:
        batches.append([s for _, s in current_batch])

    return batches


def extract_worker(gpu_id, seqs, layer, output_dir, max_tokens, worker_id, n_workers, save_interval):
    """Worker: extract Layer activations on one GPU with streaming save."""
    import esm

    device = f"cuda:{gpu_id}"
    print(f"[Worker {worker_id}] Loading ESM2-650M on {device}, {len(seqs)} sequences...")

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()
    model.token_dropout = False
    batch_converter = alphabet.get_batch_converter()

    # Output paths
    worker_dir = Path(output_dir) / f"worker_{worker_id}"
    worker_dir.mkdir(parents=True, exist_ok=True)

    # Dynamic batching
    batches = make_dynamic_batches(seqs, max_tokens)
    batch_sizes = [len(b) for b in batches]
    print(f"[Worker {worker_id}] {len(batches)} batches "
          f"(batch_size range: {min(batch_sizes)}-{max(batch_sizes)}, "
          f"median: {sorted(batch_sizes)[len(batch_sizes)//2]})")

    # Running statistics (online)
    running_sum = np.zeros(1280, dtype=np.float64)
    running_sq_sum = np.zeros(1280, dtype=np.float64)
    total_tokens = 0

    # Streaming buffer
    act_buffer = []
    part_idx = 0

    def flush_buffer():
        nonlocal act_buffer, part_idx
        if not act_buffer:
            return
        arr = np.concatenate(act_buffer, axis=0)  # (N, 1280)
        np.save(worker_dir / f"part_{part_idx:04d}.npy", arr)
        part_idx += 1
        act_buffer = []

    for batch_seqs in tqdm(batches, desc=f"GPU {gpu_id}", position=worker_id):
        data = [("protein", s) for s in batch_seqs]
        _, _, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[layer])

        reps = results["representations"][layer]  # (B, T, 1280)

        for i, seq_len in enumerate(batch_lens):
            # Token-level activations, excluding BOS and EOS
            token_acts = reps[i, 1:seq_len - 1].cpu().float().numpy()  # (L, 1280)
            act_buffer.append(token_acts)

            # Online statistics
            running_sum += token_acts.sum(axis=0).astype(np.float64)
            running_sq_sum += (token_acts.astype(np.float64) ** 2).sum(axis=0)
            total_tokens += token_acts.shape[0]

        # Periodically flush to disk (every save_interval batches)
        if len(act_buffer) >= save_interval:
            flush_buffer()

    # Final flush
    flush_buffer()

    # Save statistics
    np.save(worker_dir / "running_sum.npy", running_sum)
    np.save(worker_dir / "running_sq_sum.npy", running_sq_sum)
    np.save(worker_dir / "total_tokens.npy", np.array([total_tokens]))
    np.save(worker_dir / "n_parts.npy", np.array([part_idx]))
    print(f"[Worker {worker_id}] Done. {total_tokens} activations in {part_idx} parts.")


def merge_and_write_memmap(output_dir, n_workers, d_input=1280):
    """
    Merge worker outputs into GLP-compatible memmap format.
    直接把 part 文件 rename 为 data 文件, 构建 indices, 避免逐条写入.
    """
    output_dir = Path(output_dir)

    # Gather statistics
    total_sum = np.zeros(d_input, dtype=np.float64)
    total_sq_sum = np.zeros(d_input, dtype=np.float64)
    total_tokens = 0

    for w in range(n_workers):
        worker_dir = output_dir / f"worker_{w}"
        total_sum += np.load(worker_dir / "running_sum.npy")
        total_sq_sum += np.load(worker_dir / "running_sq_sum.npy")
        total_tokens += int(np.load(worker_dir / "total_tokens.npy")[0])

    mean = total_sum / total_tokens
    var = total_sq_sum / total_tokens - mean ** 2

    print(f"Total activations: {total_tokens:,}")
    print(f"Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"Var  range: [{var.min():.4f}, {var.max():.4f}]")

    # Save rep_statistics.pt
    rep_stats = {
        "mean": torch.tensor(mean, dtype=torch.float32).unsqueeze(0),   # (1, 1280)
        "var": torch.tensor(var, dtype=torch.float32).unsqueeze(0),     # (1, 1280)
    }
    torch.save(rep_stats, output_dir / "rep_statistics.pt")
    print(f"Saved rep_statistics.pt")

    # Save dtype.txt
    (output_dir / "dtype.txt").write_text("float32")

    # 直接转换 part 文件为 data memmap 文件 + 构建 indices
    # 每个 part 文件是 (N, 1280) 的 numpy array
    # GLP 的 MemmapReader 期望:
    #   data_XXXX.npy: 扁平化的 1D memmap
    #   data_indices.npy: (num_samples, 3) → (file_idx, start_idx, end_idx)
    print("Converting part files to memmap format...")
    all_indices = []  # (file_idx, start, end)
    data_file_idx = 0

    for w in tqdm(range(n_workers), desc="  Merge workers"):
        worker_dir = output_dir / f"worker_{w}"
        n_parts = int(np.load(worker_dir / "n_parts.npy")[0])

        for p in range(n_parts):
            part_path = worker_dir / f"part_{p:04d}.npy"
            acts = np.load(part_path)  # (N, 1280)
            n_acts = acts.shape[0]

            # Write as flat memmap
            flat = acts.reshape(-1)  # (N * 1280,)
            data_path = output_dir / f"data_{data_file_idx:04d}.npy"
            mm = np.memmap(data_path, mode='w+', dtype=np.float32, shape=flat.shape[0])
            mm[:] = flat
            mm.flush()
            del mm

            # Build indices: each activation is (file_idx, start, end) in the flat array
            for i in range(n_acts):
                start = i * d_input
                end = start + d_input
                all_indices.append((data_file_idx, start, end))

            data_file_idx += 1

    # Save indices
    indices_arr = np.array(all_indices, dtype=np.uint64)
    np.save(output_dir / "data_indices.npy", indices_arr)
    print(f"Memmap: {data_file_idx} data files, {len(all_indices):,} activations")

    # Clean up worker dirs
    import shutil
    for w in range(n_workers):
        shutil.rmtree(output_dir / f"worker_{w}")
    print("Cleaned up worker directories.")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--layer', type=int, default=17)
    parser.add_argument('--max_seqs', type=int, default=4_000_000)
    parser.add_argument('--min_len', type=int, default=30)
    parser.add_argument('--max_len', type=int, default=1022)
    parser.add_argument('--max_tokens', type=int, default=8192,
                        help='Max tokens per batch (dynamic batching). '
                             'H200 141GB: use 8192-16384 for ESM2-650M')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--save_interval', type=int, default=5000,
                        help='Flush buffer to disk every N sequences')
    parser.add_argument('--merge_only', action='store_true')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_gpus = len(args.gpu_ids)

    if not args.merge_only:
        # Read sequences
        seqs = read_fasta_sequences(args.fasta, args.max_seqs, args.min_len, args.max_len)

        # Sort globally by length, then interleave across GPUs for load balance
        seqs_sorted = sorted(seqs, key=len)
        chunks = [[] for _ in range(n_gpus)]
        for i, seq in enumerate(seqs_sorted):
            chunks[i % n_gpus].append(seq)

        print(f"\nExtracting Layer {args.layer} activations with {n_gpus} GPUs")
        print(f"  max_tokens per batch: {args.max_tokens}")
        for i, chunk in enumerate(chunks):
            lens = [len(s) for s in chunk]
            print(f"  GPU {args.gpu_ids[i]}: {len(chunk)} seqs "
                  f"(len range: {min(lens)}-{max(lens)}, mean: {sum(lens)/len(lens):.0f})")

        # Launch workers
        processes = []
        for i, chunk in enumerate(chunks):
            p = mp.Process(
                target=extract_worker,
                args=(args.gpu_ids[i], chunk, args.layer, args.output_dir,
                      args.max_tokens, i, n_gpus, args.save_interval)
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        print("\nAll workers done. Merging...")

    # Merge into memmap format
    merge_and_write_memmap(args.output_dir, n_gpus)
    print("Done!")
