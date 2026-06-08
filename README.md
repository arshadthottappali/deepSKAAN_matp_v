# DeepSKAN

**Deep Spectroscopy Kinetic Analysis Network** — automated kinetic model
classification for transient absorption (TA) spectroscopy using a deep
residual CNN, followed by Global and Target Analysis fitting.

> If this package contributes to a scientific publication, please cite:
> **Kollenz et al., J. Phys. Chem. B 2020** — https://doi.org/10.1021/acs.jpcb.0c04299

---

## How it works

Traditional TA analysis requires an expert to manually guess a kinetic model
(which excited states exist and how they interconvert), then fit a system of ODEs
to the data — a slow, subjective process.

DeepSKAN treats the 2D TA map (time × wavelength) as a grayscale image and uses
a ResNet-style CNN to classify it into one of **103 physically distinct kinetic
topologies** (for 4 species). The CNN prediction then guides a classical
Global/Target Analysis fit to extract decay times and Species-Associated
Difference Spectra (SADS).

```
CSV file  →  preprocess  →  CNN classifier  →  kinetic topology
                                                      ↓
                                           Global Analysis (decay times)
                                                      ↓
                                           Target Analysis (SADS + concentrations)
                                                      ↓
                                              result plots
```

---

## Requirements

- **Python 3.7+** (3.10 / 3.11 recommended)
- PyTorch ≥ 2.0
- numpy ≥ 1.24
- scipy ≥ 1.10
- matplotlib ≥ 3.7
- networkx ≥ 3.0
- tqdm ≥ 4.65
- h5py ≥ 3.8  *(only needed for pre-generated datasets)*

See `requirements_modern.txt` for the full pinned list.

---

## Installation

```bash
pip install .
# or for development
pip install -e .
```

---

## Project structure

```
deepSKAN/
├── deepGTA_torch/          # Main PyTorch package
│   ├── model.py            # DeepSKAN ResNet architecture
│   ├── kinetic_models.py   # Kinetic model enumeration and ODE matrices
│   ├── physics.py          # ODE solver, IRF convolution, normalization
│   ├── data_generator.py   # On-the-fly synthetic TA dataset
│   ├── hdf5_dataset.py     # Fast HDF5 dataset loader
│   ├── preprocessing.py    # Load and preprocess real TA CSV files
│   ├── analyzer.py         # DLAnalyzer (CNN) + GTAnalyzer (GA/TA fitting)
│   ├── plotting.py         # All result visualisations
│   └── gradcam.py          # Grad-CAM explanation maps
│
├── train.py                # Train the CNN
├── analyze.py              # Full analysis pipeline on a real CSV
├── evaluate.py             # Benchmark / screen real data files
├── generate_dataset.py     # Pre-generate HDF5 training dataset
├── setup_env.py            # One-click Colab / Kaggle environment setup
│
├── my_test/                # Example real TA data files (CSV)
├── examples/               # Legacy usage examples
└── deepGTA/                # Legacy Keras/TF implementation (reference only)
```

---

## Quick start

### 1 — Environment setup (Colab / Kaggle)

Add this cell at the top of your notebook:

```python
# Upload or clone the repo first, then:
%run setup_env.py
# Exposes: CHECKPOINT_DIR, DATASET_DIR
```

`setup_env.py` auto-detects the platform, installs dependencies, and mounts
Google Drive (Colab) or sets up `/kaggle/working` (Kaggle) for persistent storage.

### 2 — Generate training data *(recommended, do once)*

Pre-generating avoids running `solve_ivp()` inside every training batch, which
is the main training bottleneck (~58 min/epoch on-the-fly vs ~minutes from disk).

```bash
# 500 000 training samples  (~30-60 min, ~2-4 GB)
python generate_dataset.py \
    --n_samples 500000 \
    --num_species 4 \
    --out datasets/train_500k.h5

# 65 536 validation samples
python generate_dataset.py \
    --n_samples 65536 \
    --num_species 4 \
    --out datasets/val_64k.h5
```

On Colab, point `--out` at your Drive:
```bash
python generate_dataset.py \
    --n_samples 500000 \
    --out /content/drive/MyDrive/deepSKAN/datasets/train_500k.h5
```

### 3 — Train

```bash
# Fast — from pre-generated HDF5 (recommended)
python train.py \
    --dataset     datasets/train_500k.h5 \
    --val_dataset datasets/val_64k.h5 \
    --epochs 64 \
    --batch_size 32

# Slow — on-the-fly generation (no pre-generation needed)
python train.py --epoch_size 65536 --epochs 64

# Verify shapes and loss sanity before a full run
python train.py --dataset datasets/train_500k.h5 --debug_shapes

# Resume after disconnect (auto-detects *_latest.pt checkpoint)
python train.py --dataset datasets/train_500k.h5
```

**Platform notes:**
- On **Colab + Drive**: checkpoints are automatically saved to
  `/content/drive/MyDrive/deepSKAN/checkpoints/` if Drive is mounted.
- On **Kaggle**: checkpoints go to `/kaggle/working/deepSKAN/checkpoints/`.
  Download them before the session ends — `/kaggle/working` is not persisted.
- **Kaggle is recommended over Colab Free** for long training runs
  (30 h/week free P100 GPU vs unpredictable Colab session limits).

Training saves two checkpoints per epoch:
- `best_model.pt` — best validation loss seen so far
- `best_model_latest.pt` — end of most recent epoch (safe resume point)

Both include model weights, optimizer state, scheduler state, and metadata
so training can fully resume after a disconnect.

### 4 — Analyse real data

```bash
python analyze.py \
    --data  my_test/sample.csv \
    --model checkpoints/best_model.pt \
    --out_dir results/

# Quick CNN-only screen (skip Global/Target Analysis)
python analyze.py \
    --data my_test/sample.csv \
    --model checkpoints/best_model.pt \
    --predict_only

# Include Grad-CAM explanation overlay
python analyze.py \
    --data my_test/sample.csv \
    --model checkpoints/best_model.pt \
    --gradcam

# Show top-5 alternative kinetic models and lower confidence threshold
python analyze.py \
    --data my_test/sample.csv \
    --model checkpoints/best_model.pt \
    --top_k 5 \
    --min_confidence 0.60
```

**Output files** (all saved to `--out_dir`):

| File | Description |
|---|---|
| `*_raw.png` | Raw TA heatmap with diverging colour scale |
| `*_kinetic_model.png` | Predicted kinetic topology as a directed graph |
| `*_confidence.png` | Bar chart of top-10 CNN class probabilities |
| `*_gradcam.png` | Grad-CAM overlay showing which TA regions drove the prediction (`--gradcam`) |
| `*_transients.png` | Species concentration profiles from Target Analysis |
| `*_sads.png` | Species-Associated Difference Spectra |

### 5 — Evaluate / benchmark

```bash
# Screen all CSVs in a folder (no ground truth needed)
python evaluate.py \
    --model   checkpoints/best_model.pt \
    --data    my_test/ \
    --out_dir results/eval/ \
    --gradcam

# Accuracy benchmark against known labels
python evaluate.py \
    --model  checkpoints/best_model.pt \
    --data   my_test/ \
    --labels my_test/labels.json \
    --mode   accuracy
```

`labels.json` format:
```json
{
    "sample1.csv": 42,
    "sample2.csv": 17
}
```

Reports top-1 accuracy, top-3 accuracy, and a per-class breakdown.

---

## Data format

DeepSKAN reads **Surface Xplorer CSV** files. The expected layout is:

```
,  t_0,   t_1,   t_2,  ...   (time axis, ps — first row)
wl_0, v00,  v01,  v02,  ...
wl_1, v10,  v11,  v12,  ...
...
```

Where `wl_N` are wavelengths (nm) and `t_N` are delay times (ps).
Metadata lines beginning with a letter are parsed as key: value pairs and ignored during analysis.

---

## Model architecture

DeepSKAN is a 7-block residual CNN:

```
Input (1 × 256 × 64)  — normalised TA map: 256 time pts × 64 wavelength pts
  ↓ GaussianNoise (σ=0.01, training only)
  ↓ ResBlock 1  — 64 ch,  kernel (15×1)  + MaxPool (2×1)   [time-domain features]
  ↓ ResBlock 2  — 128 ch, kernel (11×1)  + MaxPool (2×1)
  ↓ ResBlock 3  — 128 ch, kernel (9×3)   + MaxPool (2×2)   [mixed features]
  ↓ ResBlock 4  — 128 ch, kernel (9×3)   + MaxPool (2×2)
  ↓ ResBlock 5  — 256 ch, kernel (5×5)   + MaxPool (2×2)   [spatial features]
  ↓ ResBlock 6  — 256 ch, kernel (3×3)   + MaxPool (2×2)
  ↓ ResBlock 7  — 128 ch, kernel (3×3)
  ↓ AdaptiveAvgPool2d (2×4)  →  flatten to 1024
  ↓ FC: 1024 → 512 → 512 → 256 → num_classes
  ↓ Softmax → class probabilities
```

Each residual block uses `Conv → BatchNorm → ReLU` internally.
Dropout (p=0.3 default) is applied before each FC layer.
The asymmetric early kernels (tall, narrow) are physically motivated — TA dynamics
unfold along the time axis across many orders of magnitude, while spectral features
are comparatively compact.

---

## Kinetic models

For `num_species=4`, there are **103 viable, non-isomorphic kinetic topologies**
after filtering out:
- Models where some states can never be populated after laser excitation
- Models with more than 2 species feeding into one state (overbranching)
- Isomorphic duplicates (relabelling of middle species)

Each model is encoded as a binary adjacency matrix stored as an integer bitmask.
The CNN outputs a probability distribution over all 103 classes.

---

## Training data generation

The model is trained entirely on **synthetic data** — no real labelled TA spectra
are needed. For each sample:

1. A random kinetic topology is selected (balanced across all 103 classes)
2. Random rate constants are sampled (log-uniform, 1 ps – 5 ns lifetimes)
3. The ODE system is solved with `scipy.integrate.solve_ivp` (RK45, relaxed tolerances)
4. Random Species-Associated Difference Spectra (SADS) are generated as sums of Gaussians
5. The 2D TA map is assembled: `TA = Σ (SADS_i × concentration_i) − bleach`
6. Structured experimental noise is added (baseline drift, scatter spike, coherent artefact)
7. The map is normalised to `[−1, 1]` using the absolute maximum

---

## Colab / Kaggle notebook workflow

```python
# Cell 1 — setup
%run setup_env.py            # installs deps, mounts Drive, exports paths

# Cell 2 — clone / upload repo (if not done)
# !git clone https://github.com/your-fork/deepSKAN

# Cell 3 — generate dataset (run once, takes ~30–60 min)
import subprocess
subprocess.run([
    "python", "generate_dataset.py",
    "--n_samples", "500000",
    "--out", f"{DATASET_DIR}/train_500k.h5",
])

# Cell 4 — train
subprocess.run([
    "python", "train.py",
    "--dataset",     f"{DATASET_DIR}/train_500k.h5",
    "--val_dataset", f"{DATASET_DIR}/val_64k.h5",
    "--save_path",   f"{CHECKPOINT_DIR}/best_model.pt",
    "--epochs", "64",
])

# Cell 5 — analyse
subprocess.run([
    "python", "analyze.py",
    "--data",  "my_test/sample.csv",
    "--model", f"{CHECKPOINT_DIR}/best_model.pt",
    "--gradcam",
])
```

> **Kaggle tip:** After training, save your checkpoint as a Kaggle Dataset output so it
> persists across sessions:
> ```bash
> cp /kaggle/working/deepSKAN/checkpoints/best_model.pt /kaggle/working/
> ```
> Then download it from the kernel's Output tab.

---

## Key improvements (vs original)

| Area | Change |
|---|---|
| **Training speed** | Pre-generated HDF5 dataset eliminates `solve_ivp` from every batch |
| **ODE solver** | Relaxed tolerances (1e-6) for training, tight (1e-10) for real-data fitting |
| **Normalisation** | Uses `max(|data|)` — correct for negative TA signals (ground-state bleach) |
| **Model** | BatchNorm in all residual blocks, Dropout before FC layers, AdaptiveAvgPool |
| **Training** | Mixed-precision (AMP), gradient clipping, full checkpoint with resume |
| **Class balance** | Balanced sampling — all 103 classes equally represented per epoch |
| **Structured noise** | Synthetic training data includes baseline drift, scatter spikes, coherent artefacts |
| **GA optimiser** | Differential evolution replaces random-restart Monte Carlo |
| **TA ranking** | Decay permutations ranked by true fit residual, not amplitude heuristic |
| **Confidence** | Low-confidence warning + top-k alternatives when CNN is uncertain |
| **Grad-CAM** | Saliency overlay shows which TA regions drove the CNN decision |
| **Evaluation** | `evaluate.py` screens real data files; accuracy mode with ground-truth labels |
| **Input validation** | Clear error messages for bad CSV format or mismatched model species count |
| **Platform support** | Auto-detects Colab / Kaggle; routes checkpoints to persistent storage |

---

## Citation

```bibtex
@article{kollenz2020deepskan,
  title   = {Excited State Kinetics of Ruthenium(II) Polypyridyl Complexes},
  author  = {Kollenz, Philipp and others},
  journal = {J. Phys. Chem. B},
  year    = {2020},
  doi     = {10.1021/acs.jpcb.0c04299}
}
```

---

## License

See `LICENSE` for details.
