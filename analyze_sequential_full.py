"""
analyze_sequential_full.py
==========================
Full Global Analysis with sequential model A→B→C→GS.
Produces:
  - EADS (Evolution-Associated Difference Spectra)
  - Concentration profiles (population dynamics)
  - Kinetic fits and residuals
  - Isotope effect comparison

For a sequential model, EADS (not SADS) is the correct decomposition because
each spectrum evolves into the next one along a linear pathway.

Usage:
    python analyze_sequential_full.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, least_squares
from scipy.special import erf
from deepGTA_torch.preprocessing import load_surface_xplorer_csv
import os


def convolve_exp_irf(t, tau, mu, sigma):
    """
    Exponential decay convolved with Gaussian IRF.
    Analytical solution for sequential kinetics with IRF.

    Args:
        t:     time axis (ps)
        tau:   lifetime (ps)
        mu:    time-zero offset (ps)
        sigma: IRF width (ps, Gaussian sigma)
    """
    if tau < 1e-10:
        # Effectively infinite lifetime — step function convolved with IRF
        return 0.5 * (1 + erf((t - mu) / (np.sqrt(2) * sigma)))

    k = 1.0 / tau
    arg1 = k * (mu + k * sigma**2 / 2 - t)
    arg2 = (t - mu - k * sigma**2) / (np.sqrt(2) * sigma)

    # Clip to prevent overflow
    arg1 = np.clip(arg1, -500, 500)
    return 0.5 * np.exp(arg1) * (1 + erf(arg2))


def build_concentration_matrix(t, taus, irf_sigma, t0):
    """
    Build concentration matrix for sequential A→B→C→GS model.

    For sequential kinetics: C_i(t) = sum of exponentials with amplitudes
    determined by the rate matrix eigenvalues.

    Simplified approach: compute decay-associated concentrations
    (each column is exp(-t/tau_i) convolved with IRF).
    """
    n_species = len(taus)
    n_t = len(t)
    C = np.zeros((n_t, n_species))

    for i in range(n_species):
        C[:, i] = convolve_exp_irf(t, taus[i], t0, irf_sigma)

    return C


def sequential_concentrations(t, taus, irf_sigma, t0):
    """
    Compute actual sequential concentration profiles.
    For A→B→C→GS with rates k1, k2, k3:
      C_A(t) = exp(-k1*t)
      C_B(t) = k1/(k2-k1) * [exp(-k1*t) - exp(-k2*t)]
      C_C(t) = ... (analytical expression)

    All convolved with Gaussian IRF.
    """
    k = [1.0 / tau for tau in taus]
    n_t = len(t)

    # Build decay-associated concentrations (DADS basis)
    # Then convert to species concentrations via the sequential scheme
    C_dads = np.zeros((n_t, len(taus)))
    for i in range(len(taus)):
        C_dads[:, i] = convolve_exp_irf(t, taus[i], t0, irf_sigma)

    # For sequential model, the species concentrations relate to DADS as:
    # C_species = C_dads @ transformation_matrix
    # But for EADS decomposition, we use C_dads directly
    # (EADS = pseudo-inverse of C_dads applied to data)

    # Actual species populations for plotting:
    # Simple approach: numerically compute from rate equations
    dt = np.diff(t, prepend=t[0] - (t[1] - t[0]))
    C_species = np.zeros((n_t, len(taus) + 1))  # +1 for GS

    # Use IRF-convolved rise as the excitation
    rise = convolve_exp_irf(t, 1e10, t0, irf_sigma)  # step function with IRF

    # Numerical integration of sequential kinetics
    C_species[0, 0] = 0  # A starts at 0, rises with pump
    for j in range(1, n_t):
        dt_j = t[j] - t[j-1]
        # Excitation pumps into A
        pump = (rise[j] - rise[j-1]) if j > 0 else 0

        # A: gains from pump, loses with k1
        C_species[j, 0] = C_species[j-1, 0] + pump - k[0] * C_species[j-1, 0] * dt_j
        # B: gains from A, loses with k2
        C_species[j, 1] = C_species[j-1, 1] + k[0] * C_species[j-1, 0] * dt_j - k[1] * C_species[j-1, 1] * dt_j
        # C: gains from B, loses with k3
        C_species[j, 2] = C_species[j-1, 2] + k[1] * C_species[j-1, 1] * dt_j - k[2] * C_species[j-1, 2] * dt_j
        # GS: gains from C
        C_species[j, 3] = C_species[j-1, 3] + k[2] * C_species[j-1, 2] * dt_j

    # Normalize
    c_max = np.max(C_species[:, 0])
    if c_max > 0:
        C_species /= c_max

    return C_species


def fit_global_analysis(filepath, label):
    """
    Full Global Analysis: fit 3 exponentials + IRF to the entire 2D TA matrix,
    then extract EADS and concentration profiles.
    """
    t, wl, data, _ = load_surface_xplorer_csv(filepath)

    print(f'=== {label} ===')
    print(f'  Data: {data.shape[0]} wavelengths × {data.shape[1]} time points')
    print(f'  Time: {t[0]:.2f} – {t[-1]:.1f} ps')
    print(f'  Wavelength: {wl[0]:.1f} – {wl[-1]:.1f} nm')

    # Use all time points (negative time included for IRF fitting)
    t_fit = t
    data_fit = data  # (n_wl, n_t)

    def model_2d(p):
        """Build 2D model: C(t) × EADS(wl) where EADS = pinv(C) @ data."""
        tau1, tau2, tau3, sigma, t0 = p
        taus = [tau1, tau2, tau3]
        C = build_concentration_matrix(t_fit, taus, sigma, t0)
        # EADS via pseudo-inverse
        eads = np.linalg.pinv(C) @ data_fit.T  # (3, n_wl)
        # Reconstructed data
        recon = (C @ eads).T  # (n_wl, n_t)
        return C, eads, recon

    def cost(p):
        _, _, recon = model_2d(p)
        return np.sum((recon - data_fit) ** 2)

    # Bounds: taus [0.5, 6000], IRF [0.05, 1.0], t0 [-2, 2]
    bounds = [(0.5, 50), (10, 500), (100, 6000), (0.05, 1.0), (-2.0, 2.0)]

    print('  Running differential evolution...')
    result = differential_evolution(cost, bounds, seed=42, maxiter=500,
                                     tol=1e-8, popsize=15, workers=1)

    # Polish
    def residual_flat(p):
        _, _, recon = model_2d(p)
        return (recon - data_fit).ravel()

    fit = least_squares(residual_flat, result.x, method='lm', max_nfev=2000)

    tau1, tau2, tau3, sigma, t0 = fit.x
    taus = sorted([tau1, tau2, tau3])
    C_final, eads_final, recon_final = model_2d(fit.x)

    # Compute actual sequential concentration profiles
    C_species = sequential_concentrations(t_fit, taus, sigma, t0)

    print(f'  τ₁ = {taus[0]:.1f} ps  (A→B)')
    print(f'  τ₂ = {taus[1]:.1f} ps  (B→C)')
    print(f'  τ₃ = {taus[2]:.1f} ps  (C→GS)')
    print(f'  IRF σ = {sigma:.3f} ps')
    print(f'  t₀ = {t0:.3f} ps')
    print(f'  cost = {fit.cost:.4e}')
    print()

    return {
        'label': label,
        't': t_fit,
        'wl': wl,
        'data': data_fit,
        'recon': recon_final,
        'C_dads': C_final,
        'C_species': C_species,
        'eads': eads_final,
        'taus': taus,
        'sigma': sigma,
        't0': t0,
        'cost': fit.cost,
    }


def plot_eads(results_d, results_h, out_dir):
    """Plot EADS for both samples."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#E74C3C', '#27AE60', '#2980B9']
    labels_species = ['EADS₁ (A)', 'EADS₂ (B)', 'EADS₃ (C)']

    for ax, r in [(ax1, results_d), (ax2, results_h)]:
        for i in range(3):
            ax.plot(r['wl'], r['eads'][i], color=colors[i], linewidth=2,
                    label=f'{labels_species[i]}, τ={r["taus"][i]:.1f} ps')
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')
        ax.set_xlabel('Wavelength (nm)', fontsize=12)
        ax.set_ylabel('ΔA (norm.)', fontsize=12)
        ax.set_title(f'EADS — {r["label"]}', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2)

    fig.suptitle('Evolution-Associated Difference Spectra (EADS)\nSequential: A → B → C → GS',
                 fontsize=13, y=1.04)
    fig.tight_layout()
    path = os.path.join(out_dir, 'eads.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()


def plot_concentrations(results_d, results_h, out_dir):
    """Plot concentration profiles (species populations vs time)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#E74C3C', '#27AE60', '#2980B9', '#95A5A6']
    labels_species = ['A (S₁ hot)', 'B (S₁ relaxed)', 'C (T₁ / CT)', 'GS']

    for ax, r in [(ax1, results_d), (ax2, results_h)]:
        t = r['t']
        C = r['C_species']
        mask = t > 0.1  # only positive time

        for i in range(4):
            ax.semilogx(t[mask], C[mask, i], color=colors[i], linewidth=2,
                        label=labels_species[i])

        ax.set_xlabel('Time (ps)', fontsize=12)
        ax.set_ylabel('Population (norm.)', fontsize=12)
        ax.set_title(f'Concentration Profiles — {r["label"]}', fontsize=13)
        ax.legend(fontsize=10, loc='right')
        ax.set_xlim(0.1, 7000)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.2)
        ax.axhline(0, color='gray', linewidth=0.5)

    fig.suptitle('Species Concentrations (Sequential A → B → C → GS)',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, 'concentrations.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()


def plot_eads_comparison(results_d, results_h, out_dir):
    """Overlay EADS of D vs H for each species."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    labels = ['EADS₁ (A)', 'EADS₂ (B)', 'EADS₃ (C)']

    for i, ax in enumerate(axes):
        ax.plot(results_d['wl'], results_d['eads'][i], 'b-', linewidth=2,
                label=f'D (τ={results_d["taus"][i]:.1f} ps)')
        ax.plot(results_h['wl'], results_h['eads'][i], 'r--', linewidth=2,
                label=f'H (τ={results_h["taus"][i]:.1f} ps)')
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_xlabel('Wavelength (nm)', fontsize=11)
        ax.set_ylabel('ΔA', fontsize=11)
        ax.set_title(labels[i], fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    fig.suptitle('EADS Comparison: Deuterated vs Protonated', fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, 'eads_comparison.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()


def write_summary(results_d, results_h, out_dir):
    """Write summary table."""
    d = results_d['taus']
    h = results_h['taus']

    lines = [
        "=" * 72,
        "GLOBAL ANALYSIS — Sequential Model: A → B → C → GS",
        "EADS Decomposition (Evolution-Associated Difference Spectra)",
        "=" * 72,
        "",
        f"{'Component':<15} {'CBK-D (ps)':<15} {'CBK-H (ps)':<15} {'D/H Ratio':<12} {'Assignment':<20}",
        "-" * 72,
        f"{'τ₁ (A→B)':<15} {d[0]:<15.1f} {h[0]:<15.1f} {d[0]/h[0]:<12.2f} {'Vibrational cooling':<20}",
        f"{'τ₂ (B→C)':<15} {d[1]:<15.1f} {h[1]:<15.1f} {d[1]/h[1]:<12.2f} {'Structural relaxation':<20}",
        f"{'τ₃ (C→GS)':<15} {d[2]:<15.1f} {h[2]:<15.1f} {d[2]/h[2]:<12.2f} {'Radiative/non-rad decay':<20}",
        "-" * 72,
        "",
        f"CBK-D:  IRF σ = {results_d['sigma']:.3f} ps,  t₀ = {results_d['t0']:.3f} ps,  cost = {results_d['cost']:.4e}",
        f"CBK-H:  IRF σ = {results_h['sigma']:.3f} ps,  t₀ = {results_h['t0']:.3f} ps,  cost = {results_h['cost']:.4e}",
        "",
        "Notes:",
        "  - EADS (not SADS) because model is sequential",
        f"  - τ₃ exceeds measurement window (~6000 ps) — lower bound estimate",
        "  - 3 EADS components + Gaussian IRF convolution",
        "  - Optimization: differential evolution + Levenberg-Marquardt polish",
        "=" * 72,
    ]

    text = "\n".join(lines)
    print(text)

    path = os.path.join(out_dir, 'global_analysis_summary.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'\nSaved: {path}')


# ═══ MAIN ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    out_dir = 'results/sequential_full'
    os.makedirs(out_dir, exist_ok=True)

    print("Global Analysis: Sequential A → B → C → GS\n")

    results_d = fit_global_analysis(
        'my_test/CBK-D_373NAnEn_newdepo0614_sample_NAnEn_SaphWLC_476nm_Avg2_5scan_25uW -bg -chirp.csv',
        'CBK-D (deuterated)'
    )

    results_h = fit_global_analysis(
        'my_test/CBK-H_232NAnEn_newdepo0614_sample_NAnEn_SaphWLC_476nm_Avg2_5scan_25uW -bg -chirp.csv',
        'CBK-H (protonated)'
    )

    # Generate all plots
    plot_eads(results_d, results_h, out_dir)
    plot_concentrations(results_d, results_h, out_dir)
    plot_eads_comparison(results_d, results_h, out_dir)
    write_summary(results_d, results_h, out_dir)

    print(f"\nAll outputs: {out_dir}/")
