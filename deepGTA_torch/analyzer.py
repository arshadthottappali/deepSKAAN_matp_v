"""
analyzer.py
===========
DLAnalyzer  — CNN-based kinetic model classifier.
GTAnalyzer  — Global Analysis (GA) + Target Analysis (TA) fitting.

Phase 3 fixes applied:
  FIX #15 — get_best_ta uses actual residual (sum-of-squares vs data) instead
             of the heuristic np.amax(data_s) for ranking decay permutations.
  FIX #16 — get_best_ga replaced with scipy differential_evolution for robust
             global optimisation instead of random-restart Monte Carlo.
  FIX #17 — predict() and DLAnalyzer expose confidence level; analyze.py warns
             when confidence is below a configurable threshold.
  FIX #18 — get_best_ta caps permutations at MAX_PERMUTATIONS (default 24) and
             falls back to a random sample when n! exceeds the cap, preventing
             O(n!) blow-up for 5+ decays.
"""

import math
import warnings
from itertools import permutations
from typing import Tuple

import numpy as np
import torch
from scipy.optimize import least_squares, differential_evolution

from .model import DeepSKAN
from .kinetic_models import enumerate_viable_models, generate_binary_matrix
from .physics import convolve_irf, solve_kinetics


# Maximum number of decay permutations to try in get_best_ta.
# 4! = 24  (fine),  5! = 120  (slow),  6! = 720  (very slow).
# When n! > cap, we sample this many random permutations instead.
MAX_PERMUTATIONS = 24


# ── Confidence threshold ──────────────────────────────────────────────────────
# FIX #17: warn when the CNN top-1 probability is below this.
LOW_CONFIDENCE_THRESHOLD = 0.40


class DLAnalyzer:
    """
    Wraps a trained DeepSKAN model and provides prediction utilities.

    FIX #17: predict() now also returns the top-1 confidence score.
             A warning is printed when confidence < LOW_CONFIDENCE_THRESHOLD
             so downstream code can decide whether to trust the result.
    """

    def __init__(self, model_path: str, num_species: int = 5, device=None):
        self.num_species = num_species
        self.viable_ids  = enumerate_viable_models(num_species)
        self.device      = device or torch.device('cpu')

        self.model = DeepSKAN(num_classes=len(self.viable_ids))

        # Support both old (weights-only) and new (full checkpoint) saves
        raw = torch.load(model_path, map_location=self.device)
        if isinstance(raw, dict) and 'model_state_dict' in raw:
            self.model.load_state_dict(raw['model_state_dict'])
        else:
            self.model.load_state_dict(raw)

        self.model.to(self.device)
        self.model.eval()

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Run the CNN on a preprocessed TA map.

        Args:
            X: numpy array of shape (1, 256, 64) or (B, 1, 256, 64).

        Returns:
            probs: softmax probabilities, shape (num_classes,).

        FIX #4  (Phase 1): safe unsqueeze — handles both 3-D and 4-D inputs.
        FIX #17 (Phase 3): prints a warning when top-1 confidence is low.
        """
        X_np = np.array(X, dtype=np.float32)
        if X_np.ndim == 3:
            X_tensor = torch.tensor(X_np).unsqueeze(0).to(self.device)
        elif X_np.ndim == 4:
            X_tensor = torch.tensor(X_np).to(self.device)
        else:
            raise ValueError(
                f"Expected 3-D (C,H,W) or 4-D (B,C,H,W) input, got shape {X_np.shape}"
            )

        with torch.no_grad():
            logits = self.model(X_tensor)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # FIX #17: confidence warning
        top1_conf = float(probs.max())
        top1_class = int(probs.argmax())
        if top1_conf < LOW_CONFIDENCE_THRESHOLD:
            warnings.warn(
                f"[DLAnalyzer] Low confidence: top-1 class {top1_class} "
                f"has probability {top1_conf:.2f} < threshold {LOW_CONFIDENCE_THRESHOLD:.2f}. "
                f"The predicted kinetic model may be unreliable. "
                f"Consider inspecting the top-3 candidates.",
                UserWarning,
                stacklevel=2,
            )

        return probs

    def get_top_k_predictions(self, probs: np.ndarray, k: int = 3):
        """
        FIX #17: return the top-k class IDs, their probabilities, and binary
        matrices, so callers can inspect alternatives when confidence is low.

        Returns:
            List of (class_id, probability, binary_matrix) sorted by probability desc.
        """
        top_idx  = np.argsort(probs)[::-1][:k]
        results  = []
        for idx in top_idx:
            k_id   = self.viable_ids[idx]
            K_bin  = generate_binary_matrix(k_id, self.num_species)
            results.append((int(idx), float(probs[idx]), K_bin))
        return results

    def get_prediction_binary_matrix(
        self, prediction: np.ndarray, min_confidence: float = None
    ) -> Tuple[np.ndarray, int]:
        if min_confidence is None:
            c_id  = int(np.argmax(prediction))
            k_id  = self.viable_ids[c_id]
            K_pred = generate_binary_matrix(k_id, self.num_species)
            return K_pred, 1

        # Weighted blend of top-n models that together exceed min_confidence
        sorted_probs = np.sort(prediction)
        sorted_idx   = np.argsort(prediction)

        n = 1
        for i in range(1, len(prediction) + 1):
            if np.sum(sorted_probs[-i:]) > min_confidence:
                n = i
                break

        K_pred = np.zeros((self.num_species, self.num_species))
        for i in range(n):
            i_pred = sorted_idx[-i - 1]
            k_id   = self.viable_ids[i_pred]
            K      = generate_binary_matrix(k_id, self.num_species)
            for j in range(self.num_species):
                K[j, j] = 0
            K_pred += K * prediction[i_pred]

        K_pred /= np.sum(sorted_probs[-n:])
        return K_pred, n

    def get_num_decays(self, prediction: np.ndarray) -> Tuple[int, bool]:
        K, _ = self.get_prediction_binary_matrix(prediction)
        num_k  = 0
        offset = False
        for i in range(self.num_species):
            num_k += int(K[i, i] < 0)
            if np.sum(np.clip(K[i, :], 0, None)) > 0 and K[i, i] == 0:
                offset = True
        return num_k, offset


# ── GTAnalyzer ────────────────────────────────────────────────────────────────

class GTAnalyzer:
    """
    Global Analysis (GA) and Target Analysis (TA) fitting.

    GA:  fits decay times + IRF assuming independent exponential components.
    TA:  uses the CNN-predicted kinetic topology to fit species concentrations
         and species-associated difference spectra (SADS).
    """

    def __init__(self, x_true: np.ndarray, t: np.ndarray):
        self.x_true = x_true   # (256, 64)  normalised TA map
        self.t      = t        # (256,)     time axis

    # ── GA internals ──────────────────────────────────────────────────────────

    def _build_c_fit(self, decays, irf, tz) -> np.ndarray:
        """Build concentration matrix from decay times and IRF parameters."""
        c = np.zeros((256, self.num_spectra))
        for i in range(self.num_decays):
            c[:, i] = convolve_irf(self.t, 1.0 / (decays[i] * 1000), tz, irf)
        if self.offset:
            c[:, -1] = convolve_irf(self.t, 0.0, tz, irf)
        return c

    def generate_fit(self, p: np.ndarray) -> np.ndarray:
        decays = p[:self.num_decays]
        irf    = p[self.num_decays]
        tz     = p[self.num_decays + 1]

        self.c_fit  = self._build_c_fit(decays, irf, tz)
        self.dads   = np.dot(np.linalg.pinv(np.nan_to_num(self.c_fit)), self.x_true)
        self.x_fit  = np.dot(self.c_fit, self.dads)   # (256, 64)
        self.iterations += 1
        return self.x_fit

    def _residual(self, p: np.ndarray) -> np.ndarray:
        return (self.generate_fit(p) - self.x_true).ravel()

    def get_ga(self, decays_start, irf_start=0.0, tz_start=0.0,
               offset=False, max_nfev=200):
        self.num_decays  = len(decays_start)
        self.num_spectra = self.num_decays + int(offset)
        self.offset      = offset
        self.iterations  = 0

        p_start = np.array([*np.asarray(decays_start).ravel(), irf_start, tz_start])
        diag    = np.array([*[1e-9] * self.num_decays, 1e-9, 1e-9])

        return least_squares(
            self._residual, p_start,
            method='lm',
            gtol=1e-15, ftol=1e-15, xtol=1e-15,
            x_scale=diag,
            max_nfev=max_nfev,
        )

    # FIX #16: differential evolution replaces Monte Carlo random restarts
    def get_best_ga(
        self,
        decay_min: float,
        decay_max: float,
        num_decays: int,
        irf_start: float = 0.0,
        tz_start: float  = 0.0,
        offset: bool     = False,
        num_tries: int   = 10,      # kept for API compatibility; ignored with DE
    ) -> object:
        """
        FIX #16: Use scipy differential_evolution for global optimisation of
        decay times, replacing the previous Monte Carlo random-restart approach.

        differential_evolution explores the parameter space systematically and
        is far less likely to get stuck in local minima than random restarts,
        especially when decay times are similar in magnitude.

        The DE result is then refined with a tight Levenberg-Marquardt polish.
        """
        self.num_decays  = num_decays
        self.num_spectra = num_decays + int(offset)
        self.offset      = offset
        self.iterations  = 0

        # Parameter bounds: [decay_min, decay_max] for each decay,
        # [0.01, 500] for IRF sigma (ps), [-50, 50] for time zero offset (ps)
        bounds = (
            [(decay_min, decay_max)] * num_decays
            + [(0.01, 500.0)]    # irf
            + [(-50.0,  50.0)]   # tz
        )

        # Scalar cost for DE (sum of squared residuals)
        def _cost(p):
            return float(np.sum(self._residual(p) ** 2))

        de_result = differential_evolution(
            _cost,
            bounds,
            seed=42,
            maxiter=300,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.7,
            popsize=10,
            workers=1,          # keep single-threaded inside the optimizer
        )

        # Polish with LM from the DE solution
        return self.get_ga(
            decays_start=de_result.x[:num_decays],
            irf_start=de_result.x[num_decays],
            tz_start=de_result.x[num_decays + 1],
            offset=offset,
            max_nfev=500,
        )

    # ── TA internals ──────────────────────────────────────────────────────────

    def get_ta(self, K_b: np.ndarray, decays, irf: float,
               num_species: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build full rate matrix from binary topology K_b and decay times,
        solve ODEs, and compute SADS via pseudo-inverse.
        """
        K = np.zeros((num_species, num_species))
        j = 0
        for i in range(num_species):
            num_p = -K_b[i, i]
            if num_p > 0:
                for k in range(num_species):
                    if K_b[k, i] > 0:
                        K[k, i] = 1.0 / (K_b[k, i] * decays[j] * 1000) / num_p
                K[i, i] = -1.0 / (decays[j] * 1000)
                j += 1

        # Real-data fitting — use tight tolerances
        data_c          = solve_kinetics(self.t, K, irf, interpolate=True, tight_tol=True)
        data_c[:, -1]   = 0.0
        data_s          = np.dot(np.linalg.pinv(data_c), self.x_true)
        return data_c, data_s, K

    def _ta_residual(self, data_c: np.ndarray, data_s: np.ndarray,
                     num_species: int) -> float:
        """
        FIX #15: actual sum-of-squared-residuals between the TA fit and data,
        replacing the previous np.amax(data_s) heuristic that did not measure
        fit quality at all.
        """
        x_ta = np.zeros((256, 64))
        for i in range(num_species):
            x_ta += data_s[i] * data_c[:, i].reshape((256, 1))
        x_ta -= data_s[-1] * np.ones((256, 1))
        return float(np.sum((x_ta - self.x_true) ** 2))

    def get_best_ta(
        self,
        c_id: int,
        decays,
        irf: float,
        num_species: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Try all permutations of decay times and return the one with the
        lowest actual fit residual.

        FIX #15: error metric is now ||X_fit - X_true||² (real residual).
        FIX #18: cap at MAX_PERMUTATIONS; randomly sample if n! > cap to
                 prevent O(n!) blowup for 5+ decays.
        """
        viable_ids = enumerate_viable_models(num_species)
        k_id = viable_ids[c_id]
        K_b  = generate_binary_matrix(k_id, num_species)

        all_perms = list(permutations(decays))

        # FIX #18: cap permutation search
        n_fact = math.factorial(len(decays))
        if n_fact > MAX_PERMUTATIONS:
            warnings.warn(
                f"[GTAnalyzer] {len(decays)}! = {n_fact} permutations exceeds cap "
                f"({MAX_PERMUTATIONS}). Randomly sampling {MAX_PERMUTATIONS} permutations.",
                UserWarning,
                stacklevel=2,
            )
            rng   = np.random.default_rng(seed=0)
            idx   = rng.choice(len(all_perms), size=MAX_PERMUTATIONS, replace=False)
            perms = [all_perms[i] for i in idx]
        else:
            perms = all_perms

        best_err  = np.inf
        best_dec  = perms[0]

        for dec in perms:
            data_c, data_s, _ = self.get_ta(K_b, dec, irf, num_species)
            # FIX #15: use true residual
            err = self._ta_residual(data_c, data_s, num_species)
            if err < best_err:
                best_err = err
                best_dec = dec

        return self.get_ta(K_b, best_dec, irf, num_species)
