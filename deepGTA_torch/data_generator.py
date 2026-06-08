"""
data_generator.py
=================
On-the-fly synthetic TA dataset.

Changes vs original:
  FIX #10 — Balanced class sampling: each class gets exactly the same number
             of training samples per epoch, eliminating random under-representation
             of rare classes in the 103-class set.
  BUG FIX — preselect_k() was a closure over a mutable K_ that was modified
             inside the loop; now takes K_ as an argument.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import secrets
from scipy.interpolate import interp1d

from .kinetic_models import enumerate_viable_models, generate_matrix
from .physics import solve_kinetics, gaussian, normalize


def _preselect_k(K_: np.ndarray, num_species: int) -> bool:
    """
    Return False if the rate matrix K_ is degenerate:
      - a state drains much faster than it fills (ratio > 10)
      - a minor pathway rate is less than 10% of the dominant rate into a state
    BUG FIX: extracted from __getitem__ so it doesn't close over a mutable K_.
    """
    K_check = K_.copy()
    for i in range(num_species):
        K_check[i, i] = 0
    for i in range(num_species):
        col_sum = np.sum(K_check[:, i])
        row_sum = np.sum(K_check[i, :])
        if col_sum > row_sum * 10 and i > 0:
            return False
        for j in range(num_species):
            if K_check[i, j] > 0:
                col_max = np.amax(K_check[:, j])
                if K_check[i, j] < col_max * 0.1:
                    return False
    return True


def _add_structured_noise(
    X: np.ndarray,
    y_points: int,
    x_points: int,
    data_t: np.ndarray,
) -> np.ndarray:
    """
    FIX #9: Add structured noise that mimics real TA experimental artefacts.

    Three independent noise sources (each randomly enabled / scaled):

    1. Baseline drift  — slow wavelength-independent offset that drifts over time.
                         Simulates laser power fluctuation or slow detector drift.
    2. Scatter spike   — a narrow vertical stripe at a random wavelength, strongest
                         near time-zero.  Represents pump-scatter contamination.
    3. Coherent artefact — a broad Gaussian feature centred at time-zero that
                         decays rapidly.  Present in real TA near t=0 due to
                         non-resonant solvent response.

    All contributions are small relative to the signal so the kinetics still
    dominate, but large enough that the model learns to ignore them.
    """
    X_noisy = X.copy()   # (1, y_points, x_points)

    # ── 1. Baseline drift ──────────────────────────────────────────────────
    if np.random.rand() < 0.5:
        drift_amp   = np.random.uniform(0.01, 0.05)
        # Smooth random walk along time axis
        drift       = np.cumsum(np.random.randn(y_points)) * drift_amp / y_points
        drift      -= drift.mean()
        X_noisy[0] += drift.reshape(y_points, 1)   # same for all wavelengths

    # ── 2. Scatter spike ───────────────────────────────────────────────────
    if np.random.rand() < 0.4:
        spike_col   = np.random.randint(0, x_points)
        spike_amp   = np.random.uniform(0.02, 0.15)
        # Spike decays exponentially from time-zero
        t0_idx      = np.argmin(np.abs(data_t - data_t[0]))
        decay_tau   = np.random.uniform(1, 10)           # ps
        spike_time  = spike_amp * np.exp(-(data_t - data_t[t0_idx]) / decay_tau)
        spike_time  = np.clip(spike_time, 0, None)
        X_noisy[0, :, spike_col] += spike_time

    # ── 3. Coherent artefact near time-zero ───────────────────────────────
    if np.random.rand() < 0.4:
        ca_amp      = np.random.uniform(0.03, 0.12)
        ca_width    = np.random.uniform(0.05, 0.3)       # ps (sigma)
        ca_centre   = np.random.uniform(-0.5, 0.5)       # ps (around t=0)
        # Gaussian in time, uniform across wavelength
        ca          = ca_amp * np.exp(-0.5 * ((data_t - ca_centre) / ca_width) ** 2)
        X_noisy[0] += ca.reshape(y_points, 1)

    # Re-normalise so values stay in [-1, 1] after noise
    abs_max = np.max(np.abs(X_noisy))
    if abs_max > 0:
        X_noisy = X_noisy / abs_max

    return X_noisy.astype(np.float32)


def generate_single_sample(
    viable_ids: list,
    num_species: int,
    num_classes: int,
    data_t: np.ndarray,
    range_irf: list,
    x_points: int,
    y_points: int,
    max_gauss: int,
    s_m: float,
    t_0: np.ndarray,
    class_id: int = None,
    debug_mode: bool = False,
    add_structured_noise: bool = True,
):
    """
    Generate one synthetic TA sample.
    Extracted as a standalone function so it can be reused by the
    pre-generation script (generate_dataset.py) without importing the Dataset.

    Args:
        add_structured_noise: If True (default), apply FIX #9 structured noise
                              (baseline drift, scatter spike, coherent artefact).

    Returns:
        (X, c_id) normally, or (X, c_id, K, data_s, plot_c, sigma_irf) in debug mode.
    """
    data_x = np.arange(0, x_points)
    data_s = np.zeros((num_species, x_points))

    noise_lvl = np.random.uniform(0, 0.05)

    # ── Build species-associated difference spectra (SADS) ────────────────
    for i in range(num_species):
        num_gauss = np.random.randint(1, max_gauss + 1)

        amp    = np.random.uniform(0.5, 1)
        center = np.random.uniform(0, x_points)
        sigma  = np.exp(np.random.uniform(np.log(s_m), np.log(x_points / 4)))
        g = gaussian(data_x, center, sigma)
        if np.sum(g) > 0:
            g /= np.sum(g)
        g *= amp
        data_s[i] += g

        for _ in range(num_gauss - 1):
            amp    = np.random.uniform(0.1, 1)
            center = np.random.uniform(0, x_points)
            sigma  = np.exp(np.random.uniform(np.log(s_m), np.log(x_points)))
            g = gaussian(data_x, center, sigma)
            if np.sum(g) > 0:
                g /= np.sum(g)
            g *= amp
            data_s[i] += g

        data_s[i] += np.random.normal(0, noise_lvl / 64, x_points)

    # ── Pick kinetic model ────────────────────────────────────────────────
    c_id = class_id if class_id is not None else np.random.randint(0, num_classes)
    k_id = viable_ids[c_id]

    # ── Solve ODE (retry until physically meaningful) ─────────────────────
    found_solvable = False
    K = None
    step_c = step_t = None

    while not found_solvable:
        K  = generate_matrix(k_id, num_species, t_0[0], t_0[1])
        K_ = K.copy()

        if not _preselect_k(K_, num_species):
            continue

        sigma_irf = np.random.uniform(*range_irf)
        # Relaxed tolerances (tight_tol=False) — fine for synthetic data
        step_c, step_t = solve_kinetics(data_t, K, sigma_irf, interpolate=False)

        found_solvable = True
        for i in range(num_species):
            if np.sum(K[:, i]) < 0:          # species has a decay channel
                if np.amax(step_c[:, i]) < 0.01:
                    found_solvable = False

    # ── Interpolate concentration profiles onto fixed time grid ───────────
    plot_c = np.zeros((y_points, num_species))
    for i in range(num_species):
        f = interp1d(
            step_t, step_c[:, i], kind='cubic',
            bounds_error=False,
            fill_value=(step_c[0, i], step_c[-1, i]),
            assume_sorted=True,
        )
        plot_c[:, i] = f(data_t)

    # ── Build 2-D TA map ──────────────────────────────────────────────────
    data_pure = np.zeros((y_points, x_points))
    for i in range(num_species):
        data_pure += data_s[i] * plot_c[:, i].reshape((y_points, 1))

    # Ground state bleach (negative contribution from last species = ground state)
    data_pure -= data_s[-1] * np.ones((y_points, 1))

    data_norm = normalize(data_pure)
    X = data_norm.reshape((1, y_points, x_points)).astype(np.float32)

    # FIX #9: add structured experimental noise to close the sim-to-real gap
    if add_structured_noise:
        X = _add_structured_noise(X, y_points, x_points, data_t)

    if debug_mode:
        return X, c_id, K, data_s, plot_c, sigma_irf
    return X, c_id


class SyntheticTADataset(Dataset):
    """
    Generates synthetic TA samples on the fly.

    FIX #10 — Balanced class sampling:
        When class_id is None (normal training), each epoch is divided evenly
        across all num_classes so every class appears exactly
        (epoch_size // num_classes) times, with the remainder filled randomly.
        This eliminates the statistical under-representation of rarer topology
        classes that pure random sampling produces.
    """

    def __init__(
        self,
        epoch_size:  int,
        num_species: int  = 5,
        debug_mode:  bool = False,
        class_id:    int  = None,
    ):
        self.epoch_size  = epoch_size
        self.num_species = num_species
        self.debug_mode  = debug_mode
        self.class_id    = class_id

        self.viable_ids  = enumerate_viable_models(num_species)
        self.num_classes = len(self.viable_ids)

        # Time axis: 0.1 ps → 10 000 ps (log-spaced, 256 points)
        self.data_t    = np.logspace(-1, 4, 256)
        self.range_irf = [0.05, 0.5]   # IRF sigma: 50–500 fs in ps
        self.x_points  = 64
        self.y_points  = 256
        self.max_gauss = 6
        self.s_m       = 1
        self.t_0       = np.array([1.0, 5000.0])  # lifetime bounds in ps

        # FIX #10: pre-build a balanced class index for this epoch
        self._class_index = self._build_balanced_index()

    # ── Balanced index ────────────────────────────────────────────────────────

    def _build_balanced_index(self) -> np.ndarray:
        """
        Return an array of length epoch_size where each class appears as
        evenly as possible.  Shuffled so the DataLoader sees mixed classes
        within each batch.
        """
        if self.class_id is not None:
            # Fixed class mode (debug / eval) — no balancing needed
            return np.full(self.epoch_size, self.class_id, dtype=np.int64)

        base_count = self.epoch_size // self.num_classes
        remainder  = self.epoch_size  % self.num_classes

        index = np.repeat(np.arange(self.num_classes), base_count)
        # Fill the remainder by sampling without replacement from class ids
        extra = np.random.choice(self.num_classes, size=remainder, replace=False)
        index = np.concatenate([index, extra])
        np.random.shuffle(index)
        return index.astype(np.int64)

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        # Use a fresh random seed per sample (important for DataLoader workers)
        np.random.seed(secrets.randbits(32))

        c_id = int(self._class_index[idx])

        result = generate_single_sample(
            viable_ids  = self.viable_ids,
            num_species = self.num_species,
            num_classes = self.num_classes,
            data_t      = self.data_t,
            range_irf   = self.range_irf,
            x_points    = self.x_points,
            y_points    = self.y_points,
            max_gauss   = self.max_gauss,
            s_m         = self.s_m,
            t_0         = self.t_0,
            class_id    = c_id,
            debug_mode  = self.debug_mode,
            add_structured_noise = True,   # FIX #9: enabled by default
        )

        if self.debug_mode:
            X, c_id, K, data_s, plot_c, sigma_irf = result
            return torch.tensor(X), torch.tensor(c_id, dtype=torch.long), K, data_s, plot_c, sigma_irf

        X, c_id = result
        return torch.tensor(X), torch.tensor(c_id, dtype=torch.long)
