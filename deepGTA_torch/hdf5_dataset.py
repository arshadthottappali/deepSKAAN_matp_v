"""
hdf5_dataset.py
===============
FIX #1: PyTorch Dataset that reads from a pre-generated HDF5 file.

Replaces on-the-fly solve_ivp() generation during training with fast
tensor reads from disk. Typical speedup: 20-50x per batch.

HDF5 layout expected (produced by generate_dataset.py):
    /X       float32  (N, 1, 256, 64)
    /y       int64    (N,)
    attrs:   num_species, num_classes, n_samples
"""

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import h5py
except ImportError:
    raise ImportError(
        "h5py is required to use HDF5 datasets.\n"
        "Install with:  pip install h5py"
    )


class HDF5TADataset(Dataset):
    """
    Load pre-generated TA samples from an HDF5 file.

    The file is opened lazily (once per worker process) to be compatible
    with PyTorch's multi-process DataLoader.

    Args:
        h5_path:   Path to the .h5 file produced by generate_dataset.py.
        augment:   If True, apply lightweight on-the-fly augmentations:
                     - random horizontal flip (wavelength axis)
                     - random Gaussian noise  (σ up to 0.02)
                   These are cheap (no ODE solving) and improve robustness.
    """

    def __init__(self, h5_path: str, augment: bool = False):
        self.h5_path = h5_path
        self.augment = augment
        self._file   = None   # opened lazily per worker

        # Read metadata without keeping the file open
        with h5py.File(h5_path, 'r') as f:
            self.n_samples   = int(f.attrs['n_samples'])
            self.num_classes = int(f.attrs['num_classes'])
            self.num_species = int(f.attrs['num_species'])

        print(
            f"[HDF5TADataset] Loaded {self.h5_path}  "
            f"n={self.n_samples:,}  classes={self.num_classes}  "
            f"augment={augment}"
        )

    def __len__(self):
        return self.n_samples

    def _open(self):
        """Open HDF5 file once per worker (lazy init)."""
        if self._file is None:
            self._file = h5py.File(self.h5_path, 'r')

    def __getitem__(self, idx):
        self._open()

        X = self._file['X'][idx]   # (1, 256, 64)  float32
        y = int(self._file['y'][idx])

        if self.augment:
            # Lightweight augmentation — no ODE, just array ops
            # 1. Random horizontal flip on wavelength axis
            if np.random.rand() > 0.5:
                X = X[:, :, ::-1].copy()
            # 2. Tiny additive Gaussian noise
            noise_sigma = np.random.uniform(0, 0.02)
            X = (X + np.random.normal(0, noise_sigma, X.shape)).astype(np.float32)
            # Re-normalise so values stay in [-1, 1]
            abs_max = np.max(np.abs(X))
            if abs_max > 0:
                X = X / abs_max

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def __del__(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
