"""
train.py
========
Train DeepSKAN.

Two dataset modes:
  1. On-the-fly (default):  generates samples using solve_ivp during training.
                            Slow but requires no pre-generation step.
  2. HDF5 (recommended):   loads a pre-generated dataset from disk.
                            Use generate_dataset.py first, then pass --dataset.

Platform auto-detection:
  - Colab  → saves checkpoints to Google Drive if mounted
  - Kaggle → saves to /kaggle/working/deepSKAN/checkpoints/
  - Local  → uses --save_path as-is

Quick-start examples
--------------------
# Debug: verify shapes and loss before committing to a full run
python train.py --debug_shapes

# Fast training from pre-generated data (recommended)
python generate_dataset.py --n_samples 500000 --out datasets/train_500k.h5
python generate_dataset.py --n_samples 65536  --out datasets/val_64k.h5
python train.py --dataset datasets/train_500k.h5 --val_dataset datasets/val_64k.h5

# On-the-fly (slow, no pre-generation needed)
python train.py --epoch_size 65536 --epochs 64

# Resume after disconnect
python train.py --dataset datasets/train_500k.h5 --resume checkpoints/best_model_latest.pt
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepGTA_torch.model import DeepSKAN
from deepGTA_torch.data_generator import SyntheticTADataset


# ── Platform helpers ──────────────────────────────────────────────────────────

def _detect_platform() -> str:
    try:
        import google.colab  # noqa: F401
        return 'colab'
    except ImportError:
        pass
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
        return 'kaggle'
    return 'local'


def _resolve_save_path(requested: str, platform: str) -> str:
    """Reroute checkpoint path to persistent storage on Colab / Kaggle."""
    if platform == 'colab' and os.path.isdir('/content/drive/MyDrive'):
        base = '/content/drive/MyDrive/deepSKAN/checkpoints'
        path = os.path.join(base, os.path.basename(requested))
        print(f'[train] Colab + Drive detected → {path}')
        return path

    if platform == 'kaggle':
        base = '/kaggle/working/deepSKAN/checkpoints'
        path = os.path.join(base, os.path.basename(requested))
        print(f'[train] Kaggle detected → {path}')
        print('[train] NOTE: /kaggle/working is NOT persisted. Download before session ends.')
        return path

    return requested


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(path, epoch, model, optimizer, scheduler,
                    best_val_loss, num_classes, num_species):
    """FIX #12: full checkpoint — model + optimizer + scheduler + metadata."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss':        best_val_loss,
        'num_classes':          num_classes,
        'num_species':          num_species,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    """Return (start_epoch, best_val_loss) after restoring all states."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    print(f'[train] Resumed from epoch {ckpt["epoch"]+1}, '
          f'best_val_loss={ckpt["best_val_loss"]:.4f}')
    return ckpt['epoch'] + 1, ckpt['best_val_loss']


# ── Dataset factory ───────────────────────────────────────────────────────────

def build_datasets(args):
    """
    FIX #1: if --dataset is given, use the fast HDF5 loader.
    Otherwise fall back to on-the-fly generation.
    """
    if args.dataset:
        from deepGTA_torch.hdf5_dataset import HDF5TADataset
        train_ds = HDF5TADataset(args.dataset,     augment=True)
        val_path = args.val_dataset or args.dataset   # reuse train set for val if not given
        val_ds   = HDF5TADataset(val_path,          augment=False)
        print(f'[train] HDF5 mode  train={args.dataset}  val={val_path}')
    else:
        train_ds = SyntheticTADataset(epoch_size=args.epoch_size,          num_species=args.num_species)
        val_ds   = SyntheticTADataset(epoch_size=args.epoch_size // 8,     num_species=args.num_species)
        print(f'[train] On-the-fly mode  epoch_size={args.epoch_size}')

    return train_ds, val_ds


# ── Main training function ────────────────────────────────────────────────────

def train():
    parser = argparse.ArgumentParser(description='Train DeepSKAN PyTorch')
    # Data
    parser.add_argument('--dataset',      type=str,   default=None,
                        help='Path to pre-generated HDF5 training dataset (recommended)')
    parser.add_argument('--val_dataset',  type=str,   default=None,
                        help='Path to pre-generated HDF5 validation dataset')
    parser.add_argument('--epoch_size',   type=int,   default=65536,
                        help='Samples per epoch in on-the-fly mode (ignored with --dataset)')
    parser.add_argument('--num_species',  type=int,   default=5)
    # Training
    parser.add_argument('--epochs',       type=int,   default=64)
    parser.add_argument('--batch_size',   type=int,   default=32,
                        help='Batch size (default 32; increase to 64-128 with HDF5)')
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--dropout',      type=float, default=0.3,
                        help='Dropout probability in FC layers (FIX #6)')
    # Checkpointing
    parser.add_argument('--save_path',    type=str,   default='checkpoints/best_model.pt')
    parser.add_argument('--resume',       type=str,   default=None,
                        help='Explicit checkpoint path to resume from')
    # Utility
    parser.add_argument('--debug_shapes', action='store_true',
                        help='FIX #11: print shapes/loss for one batch then exit')
    parser.add_argument('--num_workers',  type=int,   default=2,
                        help='DataLoader worker processes (0 = main process)')
    args = parser.parse_args()

    platform  = _detect_platform()
    save_path = _resolve_save_path(args.save_path, platform)
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[train] platform={platform}  device={device}')

    # ── Datasets & loaders ────────────────────────────────────────────────────
    train_ds, val_ds = build_datasets(args)
    num_classes      = train_ds.num_classes

    # More workers are safe with HDF5 (lazy file open per worker)
    nw = args.num_workers
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=nw, pin_memory=True)

    print(f'[train] num_classes={num_classes}  batches/epoch={len(train_loader)}')

    # ── Model / criterion / optimizer / scheduler ─────────────────────────────
    model     = DeepSKAN(num_classes=num_classes, dropout_p=args.dropout).to(device)

    # Multi-GPU support: use DataParallel if more than one GPU is available
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f'[train] Using {torch.cuda.device_count()} GPUs (DataParallel)')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3
    )

    # FIX #13: mixed-precision training — free ~2x speedup on modern GPUs
    use_amp = device.type == 'cuda'
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)
    if use_amp:
        print('[train] Mixed precision (AMP) enabled.')

    start_epoch   = 0
    best_val_loss = float('inf')

    # ── Auto-resume ───────────────────────────────────────────────────────────
    resume_path = args.resume or save_path.replace('.pt', '_latest.pt')
    if os.path.isfile(resume_path):
        start_epoch, best_val_loss = load_checkpoint(
            resume_path, model, optimizer, scheduler, device
        )

    # ── FIX #11: shape / loss diagnostic ─────────────────────────────────────
    if args.debug_shapes:
        inputs, targets = next(iter(train_loader))
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
        print('=== DEBUG SHAPES ===')
        print(f'  inputs.shape  : {inputs.shape}  (expected [B, 1, 256, 64])')
        print(f'  outputs.shape : {outputs.shape} (expected [B, {num_classes}])')
        print(f'  targets.shape : {targets.shape} (expected [B])')
        print(f'  targets range : {targets.min().item()} … {targets.max().item()}  '
              f'(expected 0 … {num_classes-1})')
        import math
        print(f'  loss.item()   : {loss.item():.4f}  '
              f'(random baseline ≈ {math.log(num_classes):.2f})')
        print('====================')
        return

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = train_correct = total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # FIX #13: autocast for mixed precision
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(inputs)
                loss    = criterion(outputs, targets)

            scaler.scale(loss).backward()

            # FIX #8: gradient clipping prevents exploding gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            train_loss    += loss.item() * inputs.size(0)
            _, predicted   = outputs.max(1)
            total         += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'loss': f'{train_loss/total:.4f}',
                'acc':  f'{100.*train_correct/total:.2f}%',
            })

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = val_correct = val_total = 0

        with torch.no_grad():
            pbar_v = tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Val]  ')
            for inputs, targets in pbar_v:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(inputs)
                    loss    = criterion(outputs, targets)

                val_loss    += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total   += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

                pbar_v.set_postfix({
                    'loss': f'{val_loss/val_total:.4f}',
                    'acc':  f'{100.*val_correct/val_total:.2f}%',
                })

        val_loss  /= val_total
        val_acc    = 100. * val_correct  / val_total
        train_acc  = 100. * train_correct / total

        print(
            f'Epoch {epoch+1:>3} | '
            f'Train  loss={train_loss/total:.4f}  acc={train_acc:.2f}% | '
            f'Val    loss={val_loss:.4f}  acc={val_acc:.2f}%'
        )

        scheduler.step(val_loss)

        # ── Checkpoint: best model ────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                save_path, epoch, model, optimizer, scheduler,
                best_val_loss, num_classes, args.num_species,
            )
            print(f'  ✓ New best → {save_path}')

        # ── Checkpoint: latest (always) — safe resume point ──────────────────
        latest_path = save_path.replace('.pt', '_latest.pt')
        save_checkpoint(
            latest_path, epoch, model, optimizer, scheduler,
            best_val_loss, num_classes, args.num_species,
        )


if __name__ == '__main__':
    train()
