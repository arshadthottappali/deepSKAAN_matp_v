"""
setup_env.py
============
Run this once at the top of your Colab or Kaggle notebook to install
dependencies and (on Colab) mount Google Drive for persistent checkpoints.

Usage in a notebook cell:
    %run setup_env.py
or:
    exec(open("setup_env.py").read())
"""

import sys
import os
import subprocess

# ── Detect platform ──────────────────────────────────────────────────────────
def _is_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False

def _is_kaggle():
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None

PLATFORM = "colab" if _is_colab() else ("kaggle" if _is_kaggle() else "local")
print(f"[setup_env] Detected platform: {PLATFORM}")

# ── Install Python dependencies ───────────────────────────────────────────────
def install_deps():
    packages = [
        "torch>=2.0",
        "numpy>=1.24",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "networkx>=3.0",
        "tqdm>=4.65",
        "h5py>=3.8",
    ]
    print("[setup_env] Installing packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])
    print("[setup_env] Packages installed.")

install_deps()

# ── Google Drive mount (Colab only) ──────────────────────────────────────────
DRIVE_ROOT = None

if PLATFORM == "colab":
    print("[setup_env] Mounting Google Drive...")
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    DRIVE_ROOT = "/content/drive/MyDrive/deepSKAN"
    os.makedirs(DRIVE_ROOT, exist_ok=True)
    print(f"[setup_env] Drive mounted. DeepSKAN root: {DRIVE_ROOT}")

# ── Kaggle: output directory ──────────────────────────────────────────────────
elif PLATFORM == "kaggle":
    # Kaggle persists /kaggle/working between sessions within the same kernel.
    # For cross-session persistence, save outputs and download manually, or use
    # the Kaggle Datasets API to upload/download checkpoints.
    DRIVE_ROOT = "/kaggle/working/deepSKAN"
    os.makedirs(DRIVE_ROOT, exist_ok=True)
    print(f"[setup_env] Kaggle working dir: {DRIVE_ROOT}")
    print("[setup_env] NOTE: /kaggle/working is NOT persisted across sessions.")
    print("           Download checkpoints via: kaggle datasets version -p /kaggle/working")

else:
    DRIVE_ROOT = "checkpoints"
    os.makedirs(DRIVE_ROOT, exist_ok=True)

# ── Expose paths for the rest of the notebook ────────────────────────────────
CHECKPOINT_DIR = os.path.join(DRIVE_ROOT, "checkpoints")
DATASET_DIR    = os.path.join(DRIVE_ROOT, "datasets")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DATASET_DIR,    exist_ok=True)

print(f"[setup_env] Checkpoint dir : {CHECKPOINT_DIR}")
print(f"[setup_env] Dataset dir    : {DATASET_DIR}")
print("[setup_env] Done. Import CHECKPOINT_DIR and DATASET_DIR from this module.")
