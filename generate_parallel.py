"""
generate_parallel.py
====================
Parallel dataset generation using all available CPU cores.
Same output format as generate_dataset.py but ~4x faster on 4 cores.

Usage:
    python generate_parallel.py --n_samples 100000 --num_species 4 \
        --out datasets/train_100k.h5 --workers 4
"""

import multiprocessing as mp
import numpy as np
import secrets
import os
import sys
import argparse

from deepGTA_torch.kinetic_models import enumerate_viable_models
from deepGTA_torch.data_generator import generate_single_sample


def _worker(args):
    """Generate a chunk of samples in a single worker process."""
    chunk_idx, class_ids, cfg = args
    viable_ids = enumerate_viable_models(cfg["num_species"])
    num_classes = len(viable_ids)
    X_buf = np.zeros((len(class_ids), 1, 256, 64), dtype=np.float32)
    y_buf = np.zeros(len(class_ids), dtype=np.int64)
    for i, c_id in enumerate(class_ids):
        np.random.seed(secrets.randbits(32))
        X, label = generate_single_sample(
            viable_ids=viable_ids,
            num_species=cfg["num_species"],
            num_classes=num_classes,
            data_t=cfg["data_t"],
            range_irf=cfg["range_irf"],
            x_points=64,
            y_points=256,
            max_gauss=6,
            s_m=1.0,
            t_0=cfg["t_0"],
            class_id=int(c_id),
        )
        X_buf[i] = X
        y_buf[i] = label
    return chunk_idx, X_buf, y_buf


def main():
    import h5py
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description="Parallel dataset generation for DeepSKAN")
    parser.add_argument("--n_samples",   type=int, default=100000,
                        help="Total number of samples to generate")
    parser.add_argument("--num_species", type=int, default=4,
                        help="Number of kinetic species (4 → 21 classes, 5 → 103 classes)")
    parser.add_argument("--out",         type=str, required=True,
                        help="Output HDF5 file path")
    parser.add_argument("--workers",     type=int, default=mp.cpu_count(),
                        help="Number of parallel worker processes (default: all CPUs)")
    parser.add_argument("--chunk",       type=int, default=512,
                        help="Samples per worker chunk")
    args = parser.parse_args()

    viable_ids = enumerate_viable_models(args.num_species)
    num_classes = len(viable_ids)
    print(f"num_species={args.num_species}  num_classes={num_classes}")
    print(f"Generating {args.n_samples:,} samples with {args.workers} workers -> {args.out}")

    # Balanced class index
    base = args.n_samples // num_classes
    extra = args.n_samples % num_classes
    index = np.repeat(np.arange(num_classes), base)
    index = np.concatenate([index, np.random.choice(num_classes, extra, replace=False)])
    np.random.shuffle(index)

    cfg = {
        "num_species": args.num_species,
        "data_t": np.logspace(-1, 4, 256),
        "range_irf": [0.05, 0.5],
        "t_0": np.array([1.0, 5000.0]),
    }

    # Split into chunks for parallel processing
    chunks = [(i, index[i:i + args.chunk], cfg)
              for i in range(0, args.n_samples, args.chunk)]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    with h5py.File(args.out, "w") as f:
        ds_X = f.create_dataset("X", shape=(args.n_samples, 1, 256, 64),
                                dtype="float32", chunks=(256, 1, 256, 64),
                                compression="lzf")
        ds_y = f.create_dataset("y", shape=(args.n_samples,),
                                dtype="int64", chunks=(256,))
        f.attrs["num_species"] = args.num_species
        f.attrs["num_classes"] = num_classes
        f.attrs["n_samples"] = args.n_samples

        # Generate all chunks in parallel
        with mp.Pool(args.workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(_worker, chunks),
                total=len(chunks),
                desc="Generating",
            ))

        # Write results to HDF5
        for chunk_idx, X_buf, y_buf in results:
            end = min(chunk_idx + len(y_buf), args.n_samples)
            ds_X[chunk_idx:end] = X_buf[:end - chunk_idx]
            ds_y[chunk_idx:end] = y_buf[:end - chunk_idx]

    size_mb = os.path.getsize(args.out) / 1024 / 1024
    print(f"Done. {size_mb:.1f} MB -> {args.out}")


if __name__ == "__main__":
    main()
