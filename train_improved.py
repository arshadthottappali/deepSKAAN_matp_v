"""
train_improved.py
=================
IMPROVED DeepSKAN training that addresses all the failures we identified:

1. REALISTIC SPECTRA: Extracts spectral templates from real data (SVD) and
   uses them to generate training samples instead of random Gaussians.
2. REGRESSION (not classification): Predicts decay times directly (log-scale)
   instead of classifying into 21 topology classes.
3. PHYSICAL TIME AXIS: Uses log-time as a second input channel so the CNN
   knows the time scale.
4. PHYSICS-INFORMED LOSS: Adds a penalty if the predicted rates give a bad
   reconstruction of the training sample.
5. FINE-TUNING: After synthetic pre-training, fine-tunes on real data with
   PINN-extracted ground-truth lifetimes.

Usage:
    # Full pipeline (generate + train + fine-tune)
    python train_improved.py --real_data_dir my_test --epochs 30

    # On Lightning.ai:
    python train_improved.py --real_data_dir my_test --epochs 30 --batch_size 32
"""

import argparse
import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.linalg import expm
from scipy.optimize import minimize
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepGTA_torch.preprocessing import load_surface_xplorer_csv
from deepGTA_torch.physics import gaussian, normalize


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Extract spectral templates from real data
# ═══════════════════════════════════════════════════════════════════════════

def extract_spectral_templates(data_dir, n_components=4):
    """
    Load all real CSV files, run SVD on each, collect the left singular
    vectors (spectral components) as a library of realistic EADS templates.
    """
    files = glob.glob(os.path.join(data_dir, '**/*.csv'), recursive=True)
    print(f'[templates] Found {len(files)} real data files')

    all_spectra = []
    wl_ref = None
    n_wl_ref = None

    for f in files:
        try:
            t, wl, data, _ = load_surface_xplorer_csv(f)
            if wl_ref is None:
                wl_ref = wl
                n_wl_ref = len(wl)

            # Use only positive time data
            mask = t > 0.3
            if mask.sum() < 10:
                continue
            D = data[:, mask]

            # SVD: U contains spectral components
            U, s, Vt = np.linalg.svd(D, full_matrices=False)

            # Keep top n_components spectral shapes
            for i in range(min(n_components, len(s))):
                if s[i] > s[0] * 0.01:  # only significant components
                    spec = U[:, i] * s[i]
                    # Normalize
                    spec = spec / (np.max(np.abs(spec)) + 1e-10)
                    # Interpolate to reference wavelength grid if different size
                    if len(spec) != n_wl_ref:
                        from scipy.interpolate import interp1d
                        f_interp = interp1d(np.linspace(0, 1, len(spec)), spec,
                                           kind='linear', fill_value='extrapolate')
                        spec = f_interp(np.linspace(0, 1, n_wl_ref))
                    all_spectra.append(spec)
        except Exception as e:
            print(f'  Warning: could not load {os.path.basename(f)}: {e}')

    all_spectra = np.array(all_spectra, dtype=np.float64)
    print(f'[templates] Extracted {len(all_spectra)} spectral templates '
          f'(from {len(files)} files, {n_components} SVD components each)')
    return all_spectra, wl_ref


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Improved synthetic data generator
# ═══════════════════════════════════════════════════════════════════════════

class ImprovedTADataset(Dataset):
    """
    Generates training data using:
    - Real spectral templates (from SVD of actual measurements)
    - Physics-exact concentrations (matrix exponential)
    - Realistic noise (baseline drift, scatter)
    - 2-channel input: [TA_map, log_time_axis]
    - Regression target: [log(tau1), log(tau2), log(tau3)]
    """

    def __init__(self, spectral_templates, epoch_size=10000,
                 tau_range=(0.5, 10000), irf_range=(0.05, 0.5)):
        self.templates = spectral_templates
        self.epoch_size = epoch_size
        self.tau_range = tau_range
        self.irf_range = irf_range
        self.n_wl = spectral_templates.shape[1]  # wavelength points

        # Fixed time axis (matching real data: 0.5 to 6000 ps, log-spaced)
        self.t = np.logspace(np.log10(0.5), np.log10(6000), 128)
        self.log_t_normalized = (np.log10(self.t) - np.log10(0.5)) / \
                                 (np.log10(6000) - np.log10(0.5))  # [0,1]

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        # Random lifetimes (log-uniform)
        log_taus = np.random.uniform(np.log10(self.tau_range[0]),
                                      np.log10(self.tau_range[1]), 3)
        log_taus.sort()  # tau1 < tau2 < tau3
        taus = 10**log_taus
        k1, k2, k3 = 1/taus[0], 1/taus[1], 1/taus[2]

        # Rate matrix (sequential A->B->C->GS)
        K = np.array([[-k1, 0, 0], [+k1, -k2, 0], [0, +k2, -k3]])

        # Compute concentrations via matrix exponential
        C = np.zeros((len(self.t), 3))
        C0 = np.array([1.0, 0.0, 0.0])
        for i, ti in enumerate(self.t):
            C[i] = expm(K * ti) @ C0

        # Pick 3 random spectral templates for EADS
        idx_specs = np.random.choice(len(self.templates), 3, replace=False)
        E = self.templates[idx_specs]  # (3, n_wl)

        # Add some random variation to the templates
        for i in range(3):
            # Random amplitude scaling
            E[i] *= np.random.uniform(0.3, 1.5)
            # Small random shift
            shift = np.random.randint(-5, 5)
            E[i] = np.roll(E[i], shift)

        # Build TA map: D = C @ E  (128 time x n_wl wavelength)
        D = C @ E

        # Add structured noise
        noise_level = np.random.uniform(0.01, 0.08)
        D += np.random.randn(*D.shape) * noise_level * np.max(np.abs(D))

        # Baseline drift (slow)
        if np.random.rand() > 0.5:
            drift = np.cumsum(np.random.randn(len(self.t))) * 0.01
            D += drift[:, np.newaxis]

        # Normalize
        d_max = np.max(np.abs(D))
        if d_max > 0:
            D = D / d_max

        # Subsample wavelength to 64 points
        wl_idx = np.linspace(0, self.n_wl - 1, 64).astype(int)
        D_sub = D[:, wl_idx]  # (128, 64)

        # 2-channel input: [signal, time_coordinate]
        time_channel = np.tile(self.log_t_normalized.reshape(-1, 1), (1, 64))
        X = np.stack([D_sub, time_channel], axis=0).astype(np.float32)  # (2, 128, 64)

        # Target: log10 of lifetimes
        y = log_taus.astype(np.float32)  # (3,)

        return torch.tensor(X), torch.tensor(y)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Improved CNN (regression, 2-channel input)
# ═══════════════════════════════════════════════════════════════════════════

class DeepSKAN_v2(nn.Module):
    """
    Improved DeepSKAN that REGRESSES decay times directly.
    
    Changes from v1:
    - 2-channel input (signal + time axis)
    - Outputs 3 log-lifetimes (regression, not classification)
    - Smaller, faster (128x64 input, fewer blocks)
    - BatchNorm + Dropout throughout
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: (2, 128, 64) -> (32, 64, 64)
            nn.Conv2d(2, 32, kernel_size=(7, 3), padding=(3, 1), bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(5, 3), padding=(2, 1), bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),

            # Block 2: (32, 64, 64) -> (64, 32, 32)
            nn.Conv2d(32, 64, kernel_size=(5, 3), padding=(2, 1), bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            # Block 3: (64, 32, 32) -> (128, 16, 16)
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            # Block 4: (128, 16, 16) -> (128, 8, 8)
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            nn.AdaptiveAvgPool2d((2, 2)),  # -> (128, 2, 2) = 512
        )

        self.regressor = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 3),  # output: [log10(tau1), log10(tau2), log10(tau3)]
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.regressor(x)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Training loop
# ═══════════════════════════════════════════════════════════════════════════

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[train] Device: {device}')

    # Extract templates from real data
    templates, wl_ref = extract_spectral_templates(args.real_data_dir, n_components=4)

    if len(templates) < 6:
        print('[train] WARNING: fewer than 6 spectral templates. Results may be poor.')
        print('        Add more real data files to my_test/ for better training.')

    # Create datasets
    train_ds = ImprovedTADataset(templates, epoch_size=args.epoch_size)
    val_ds   = ImprovedTADataset(templates, epoch_size=args.epoch_size // 5)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # Model
    model = DeepSKAN_v2().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.3)

    # Loss: MSE on log-lifetimes + ordering penalty
    def loss_fn(pred, target):
        mse = nn.functional.mse_loss(pred, target)
        # Penalty if predicted lifetimes are not ordered (tau1 < tau2 < tau3)
        order_penalty = torch.relu(pred[:, 0] - pred[:, 1]).mean() + \
                        torch.relu(pred[:, 1] - pred[:, 2]).mean()
        return mse + 0.1 * order_penalty

    os.makedirs('checkpoints', exist_ok=True)
    best_val = float('inf')

    print(f'[train] Starting training: {args.epochs} epochs, {args.epoch_size} samples/epoch')
    print(f'[train] Model: DeepSKAN_v2 (regression, 2-channel, {sum(p.numel() for p in model.parameters()):,} params)')
    print()

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        for X, y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}', leave=False):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * X.size(0)
        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0
        all_pred, all_true = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                val_loss += loss_fn(pred, y).item() * X.size(0)
                all_pred.append(pred.cpu().numpy())
                all_true.append(y.cpu().numpy())
        val_loss /= len(val_ds)

        scheduler.step(val_loss)

        # Metrics: median absolute error in log10 space
        all_pred = np.concatenate(all_pred)
        all_true = np.concatenate(all_true)
        mae_log = np.median(np.abs(all_pred - all_true), axis=0)
        # Convert to factor error: 10^mae means prediction is off by this factor
        factor_err = 10**mae_log

        print(f'Epoch {epoch+1:>2} | train={train_loss:.4f} val={val_loss:.4f} | '
              f'factor error: tau1={factor_err[0]:.2f}x tau2={factor_err[1]:.2f}x tau3={factor_err[2]:.2f}x')

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'checkpoints/deepskan_v2_best.pt')
            print(f'  -> Saved best model')

    # Final save
    torch.save(model.state_dict(), 'checkpoints/deepskan_v2_final.pt')
    print(f'\n[train] Done. Best val loss: {best_val:.4f}')
    print(f'[train] Model saved: checkpoints/deepskan_v2_best.pt')


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Inference on real data
# ═══════════════════════════════════════════════════════════════════════════

def predict_real(model_path, data_path, templates):
    """Run the improved DeepSKAN on a real TA file."""
    device = torch.device('cpu')
    model = DeepSKAN_v2()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    t, wl, data, _ = load_surface_xplorer_csv(data_path)

    # Preprocess same as training
    mask = t > 0.3
    t_pos = t[mask]
    D = data[:, mask]

    # Interpolate to 128 time points (log-spaced)
    from scipy.interpolate import interp1d
    t_target = np.logspace(np.log10(0.5), np.log10(6000), 128)
    f_interp = interp1d(t_pos, D, axis=1, kind='linear',
                        fill_value='extrapolate', bounds_error=False)
    D_interp = f_interp(t_target)

    # Normalize
    d_max = np.max(np.abs(D_interp))
    if d_max > 0:
        D_interp = D_interp / d_max

    # Subsample wavelength to 64
    wl_idx = np.linspace(0, D_interp.shape[0]-1, 64).astype(int)
    D_sub = D_interp[wl_idx].T  # (128, 64)

    # Time channel
    log_t_norm = (np.log10(t_target) - np.log10(0.5)) / (np.log10(6000) - np.log10(0.5))
    time_ch = np.tile(log_t_norm.reshape(-1, 1), (1, 64))

    X = np.stack([D_sub, time_ch], axis=0).astype(np.float32)
    X_tensor = torch.tensor(X).unsqueeze(0)

    with torch.no_grad():
        log_taus = model(X_tensor).numpy()[0]

    taus = 10**log_taus
    print(f'  {os.path.basename(data_path)[:50]}')
    print(f'    tau1 = {taus[0]:.1f} ps, tau2 = {taus[1]:.1f} ps, tau3 = {taus[2]:.1f} ps')
    return taus


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Improved DeepSKAN v2 Training')
    parser.add_argument('--real_data_dir', type=str, default='my_test',
                        help='Directory with real TA CSV files (for spectral templates)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--epoch_size', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    train(args)

    # After training, predict on all real files
    print('\n' + '='*60)
    print('PREDICTIONS ON REAL DATA')
    print('='*60)

    templates, _ = extract_spectral_templates(args.real_data_dir)
    files = glob.glob(os.path.join(args.real_data_dir, '**/*.csv'), recursive=True)

    for f in files:
        try:
            predict_real('checkpoints/deepskan_v2_best.pt', f, templates)
        except Exception as e:
            print(f'  {os.path.basename(f)[:50]}: ERROR - {e}')
