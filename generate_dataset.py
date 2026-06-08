"""
generate_dataset.py
===================
FIX #1: Pre-generate a large synthetic TA dataset and save it to disk.

Why: solve_ivp() inside __getitem__() costs ~0.1-0.3s per sample.
     With epoch_size=65536 and batch_size=16, that is 4096 batches × ~3s = ~3.4 hours
     just in data generation per epoch.
     Pre-generating once lets training load tensors from disk at full I/O speed.

Output format
-------------
HDF5 file (requires h5py):
    /X       float32  (N, 1, 256, 64)   normalised TA maps
    /y       int64    (N,)               class labels  0 … num_classes-1
    attrs:   num_species, num_classes, n_samples

Usage
-----
# Generate 500 000 samples for 4-species training (recommended for Kaggle/Colab)
python generate_dataset.py --n_samples 500000 --num_species 4 --out datasets/train_500k.h5

# Smaller validation set
python generate_dataset.py --n_samples 65536 --num_species 4 --out datasets/val_64k.h5

# On Colab / Kaggle, point --out at the persistent path:
#   Colab:  /content/drive/MyDrive/deepSKAN/datasets/train_500k.h5
#   Kaggle: /kaggle/working/deepSKAN/datasets/train_500k.h5

Then train with:
    python train.py --dataset datasets/train_500k.h5 --val_dataset datasets/val_64k.h5
"""

import argparse
import os
import secrets
import numpy as np
from tqdm import tqdm

# Optional dependency — fail early with a helpful message
try:
    import h5py
except ImportError:
    raise ImportError(
        "h5py is required for dataset pre-generation.\n"
        "Install it with:  pip install h5py"
    )

from deepGTA_torch.kinetic_models import enumerate_viable_models
from deepGTA_torch.data_generator import generate_single_sample


def generate(args):
    viable_ids  = enumerate_viable_models(args.num_species)
    num_classes = len(viable_ids)
    print(f"[generate] num_species={args.num_species}  num_classes={num_classes}")
    print(f"[generate] Generating {args.n_samples:,} samples → {args.out}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # ── Build a balanced class index ─────────────────────────────────────────
    # FIX #10: guarantee each class appears equally often
    base   = args.n_samples // num_classes
    extra  = args.n_samples  % num_classes
    index  = np.repeat(np.arange(num_classes), base)
    index  = np.concatenate([index, np.random.choice(num_classes, size=extra, replace=False)])
    np.random.shuffle(index)

    # ── Generator config (mirrors SyntheticTADataset defaults) ───────────────
    data_t    = np.logspace(-1, 4, 256)
    range_irf = [0.05, 0.5]
    x_points  = 64
    y_points  = 256
    max_gauss = 6
    s_m       = 1.0
    t_0       = np.array([1.0, 5000.0])

    # ── Write HDF5 ────────────────────────────────────────────────────────────
    with h5py.File(args.out, 'w') as f:
        ds_X = f.create_dataset(
            'X', shape=(args.n_samples, 1, y_points, x_points),
            dtype='float32',
            chunks=(256, 1, y_points, x_points),   # 256-sample chunks for fast sequential reads
            compression='lzf',                      # fast, lossless; no quality loss
        )
        ds_y = f.create_dataset(
            'y', shape=(args.n_samples,),
            dtype='int64',
            chunks=(256,),
        )

        # Store metadata as HDF5 attributes
        f.attrs['num_species']  = args.num_species
        f.attrs['num_classes']  = num_classes
        f.attrs['n_samples']    = args.n_samples

        # ── Generate in batches to allow tqdm progress ────────────────────
        batch_size = args.write_batch
        for start in tqdm(range(0, args.n_samples, batch_size), desc='Generating'):
            end   = min(start + batch_size, args.n_samples)
            batch = end - start

            X_buf = np.zeros((batch, 1, y_points, x_points), dtype=np.float32)
            y_buf = np.zeros(batch, dtype=np.int64)

            for i in range(batch):
                np.random.seed(secrets.randbits(32))
                c_id   = int(index[start + i])
                X, label = generate_single_sample(
                    viable_ids  = viable_ids,
                    num_species = args.num_species,
                    num_classes = num_classes,
                    data_t      = data_t,
                    range_irf   = range_irf,
                    x_points    = x_points,
                    y_points    = y_points,
                    max_gauss   = max_gauss,
                    s_m         = s_m,
                    t_0         = t_0,
                    class_id    = c_id,
                    debug_mode  = False,
                )
                X_buf[i] = X
                y_buf[i] = label

            ds_X[start:end] = X_buf
            ds_y[start:end] = y_buf

    size_mb = os.path.getsize(args.out) / 1024 / 1024
    print(f"[generate] Done. File size: {size_mb:.1f} MB  →  {args.out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-generate DeepSKAN training dataset')
    parser.add_argument('--n_samples',   type=int, default=500_000,
                        help='Total number of samples to generate (default: 500000)')
    parser.add_argument('--num_species', type=int, default=5,
                        help='Number of kinetic species (default: 5 → 103 classes)')
    parser.add_argument('--out',         type=str, default='datasets/train_500k.h5',
                        help='Output HDF5 file path')
    parser.add_argument('--write_batch', type=int, default=1024,
                        help='How many samples to buffer before writing to disk (default: 1024)')
    args = parser.parse_args()
    generate(args)
