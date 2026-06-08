"""
analyze_sequential.py
=====================
Fit sequential A→B→C→GS model to TA data using SVD + 3 exponentials.
Generates publication-quality plots and a summary table.

Usage:
    python analyze_sequential.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, least_squares
from deepGTA_torch.preprocessing import load_surface_xplorer_csv
import os


def fit_sequential(filepath, label):
    """Fit 3 exponentials to SVD-extracted kinetic trace."""
    t, wl, data, _ = load_surface_xplorer_csv(filepath)

    # Use only positive time data
    pos_mask = t > 0.5
    t_pos = t[pos_mask]
    data_pos = data[:, pos_mask]

    # SVD to extract dominant kinetics
    U, s, Vt = np.linalg.svd(data_pos, full_matrices=False)
    trace = Vt[0, :]
    trace = trace / np.max(np.abs(trace))

    print(f'=== {label} ===')
    print(f'  Time: {t_pos[0]:.2f} – {t_pos[-1]:.1f} ps ({len(t_pos)} pts)')
    print(f'  Wavelength: {wl[0]:.1f} – {wl[-1]:.1f} nm ({len(wl)} pts)')
    print(f'  SVD singular values: {s[0]:.3f}, {s[1]:.4f}, {s[2]:.4f}')

    # 3-exponential model
    def model(p, t):
        a1, tau1, a2, tau2, a3, tau3 = p
        return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2) + a3 * np.exp(-t / tau3)

    def residual(p):
        return model(p, t_pos) - trace

    def cost(p):
        return np.sum(residual(p) ** 2)

    # Global optimization
    bounds = [(-2, 2), (1, 100), (-2, 2), (50, 1000), (-2, 2), (500, 6000)]
    result = differential_evolution(cost, bounds, seed=42, maxiter=1000,
                                     tol=1e-10, popsize=20, workers=1)

    # Polish with LM
    fit = least_squares(residual, result.x, method='lm', max_nfev=5000)

    a1, tau1, a2, tau2, a3, tau3 = fit.x
    taus = sorted([abs(tau1), abs(tau2), abs(tau3)])
    amps = [a1, a2, a3]

    print(f'  τ₁ = {taus[0]:.1f} ps  (A→B)')
    print(f'  τ₂ = {taus[1]:.1f} ps  (B→C)')
    print(f'  τ₃ = {taus[2]:.1f} ps  (C→GS)')
    print(f'  cost = {fit.cost:.4e}')
    print()

    return {
        'label': label,
        't': t_pos,
        'trace': trace,
        'fit_trace': model(fit.x, t_pos),
        'taus': taus,
        'amps': amps,
        'fit_params': fit.x,
        'cost': fit.cost,
        't_full': t,
        'wl': wl,
        'data': data,
    }


def plot_kinetic_fits(results_d, results_h, out_dir):
    """Plot kinetic traces + fits for both samples."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, r in [(ax1, results_d), (ax2, results_h)]:
        ax.semilogx(r['t'], r['trace'], 'ko', markersize=2, alpha=0.5, label='Data (SVD)')
        ax.semilogx(r['t'], r['fit_trace'], 'r-', linewidth=2, label='Fit (3 exp)')

        # Plot individual components
        p = r['fit_params']
        a1, tau1, a2, tau2, a3, tau3 = p
        ax.semilogx(r['t'], a1 * np.exp(-r['t'] / tau1), 'b--', alpha=0.6,
                     label=f'τ₁={abs(tau1):.1f} ps')
        ax.semilogx(r['t'], a2 * np.exp(-r['t'] / tau2), 'g--', alpha=0.6,
                     label=f'τ₂={abs(tau2):.1f} ps')
        ax.semilogx(r['t'], a3 * np.exp(-r['t'] / tau3), 'm--', alpha=0.6,
                     label=f'τ₃={abs(tau3):.1f} ps')

        ax.set_xlabel('Time (ps)', fontsize=12)
        ax.set_ylabel('Normalized ΔA (SVD component 1)', fontsize=11)
        ax.set_title(r['label'], fontsize=13)
        ax.legend(fontsize=9, loc='upper right')
        ax.set_xlim(0.5, 7000)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')
        ax.grid(True, alpha=0.2)

    fig.suptitle('Sequential Fit: A → B → C → GS', fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, 'kinetic_fits.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()


def plot_residuals(results_d, results_h, out_dir):
    """Plot fit residuals."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 3))

    for ax, r in [(ax1, results_d), (ax2, results_h)]:
        resid = r['trace'] - r['fit_trace']
        ax.semilogx(r['t'], resid, 'k-', linewidth=0.8)
        ax.axhline(0, color='red', linewidth=0.5, linestyle='--')
        ax.set_xlabel('Time (ps)', fontsize=11)
        ax.set_ylabel('Residual', fontsize=11)
        ax.set_title(r['label'], fontsize=12)
        ax.set_xlim(0.5, 7000)
        ax.grid(True, alpha=0.2)

    fig.suptitle('Fit Residuals', fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, 'residuals.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()


def plot_ta_comparison(results_d, results_h, out_dir):
    """Plot raw TA heatmaps side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, r in [(ax1, results_d), (ax2, results_h)]:
        t_full = r['t_full']
        wl = r['wl']
        data = r['data']

        # Only plot positive times
        mask = t_full > 0
        t_plot = t_full[mask]
        d_plot = data[:, mask]

        v = np.max(np.abs(d_plot)) * 0.8
        im = ax.pcolormesh(t_plot, wl, d_plot, cmap='RdBu_r',
                           vmin=-v, vmax=v, shading='auto')
        ax.set_xscale('log')
        ax.set_xlabel('Time (ps)', fontsize=12)
        ax.set_ylabel('Wavelength (nm)', fontsize=12)
        ax.set_title(r['label'], fontsize=13)
        fig.colorbar(im, ax=ax, label='ΔA')

    fig.suptitle('Transient Absorption Maps', fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, 'ta_heatmaps.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()


def plot_isotope_comparison(results_d, results_h, out_dir):
    """Bar chart comparing decay times D vs H."""
    d_taus = results_d['taus']
    h_taus = results_h['taus']

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(3)
    width = 0.35

    bars_d = ax.bar(x - width/2, d_taus, width, label='CBK-D (deuterated)',
                     color='#2980B9', edgecolor='white')
    bars_h = ax.bar(x + width/2, h_taus, width, label='CBK-H (protonated)',
                     color='#E74C3C', edgecolor='white')

    ax.set_xlabel('Decay Component', fontsize=12)
    ax.set_ylabel('Lifetime (ps)', fontsize=12)
    ax.set_title('Kinetic Isotope Effect: D vs H', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(['τ₁ (A→B)', 'τ₂ (B→C)', 'τ₃ (C→GS)'], fontsize=11)
    ax.legend(fontsize=11)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.2, axis='y')

    # Annotate with ratio
    for i in range(3):
        ratio = d_taus[i] / h_taus[i] if h_taus[i] > 0 else 0
        y_pos = max(d_taus[i], h_taus[i]) * 1.3
        ax.text(x[i], y_pos, f'D/H = {ratio:.2f}', ha='center', fontsize=10,
                fontweight='bold')

    fig.tight_layout()
    path = os.path.join(out_dir, 'isotope_effect.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()


def write_summary_table(results_d, results_h, out_dir):
    """Write a text summary table."""
    d_taus = results_d['taus']
    h_taus = results_h['taus']

    lines = [
        "=" * 70,
        "SEQUENTIAL FIT RESULTS: A → B → C → GS (3 Exponentials)",
        "=" * 70,
        "",
        f"{'Component':<15} {'CBK-D (ps)':<15} {'CBK-H (ps)':<15} {'D/H Ratio':<12}",
        "-" * 57,
        f"{'τ₁ (A→B)':<15} {d_taus[0]:<15.1f} {h_taus[0]:<15.1f} {d_taus[0]/h_taus[0]:<12.2f}",
        f"{'τ₂ (B→C)':<15} {d_taus[1]:<15.1f} {h_taus[1]:<15.1f} {d_taus[1]/h_taus[1]:<12.2f}",
        f"{'τ₃ (C→GS)':<15} {d_taus[2]:<15.1f} {h_taus[2]:<15.1f} {d_taus[2]/h_taus[2]:<12.2f}",
        "-" * 57,
        "",
        f"CBK-D fit cost: {results_d['cost']:.4e}",
        f"CBK-H fit cost: {results_h['cost']:.4e}",
        "",
        "Notes:",
        f"  - τ₃ > measurement window ({results_d['t'][-1]:.0f} ps) — extrapolated",
        "  - Model: sum of 3 exponentials fitted to 1st SVD component",
        "  - Global optimization: differential evolution + LM polish",
        "=" * 70,
    ]

    text = "\n".join(lines)
    print(text)

    path = os.path.join(out_dir, 'fit_summary.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'\nSaved: {path}')


# ═══ MAIN ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    out_dir = 'results/sequential'
    os.makedirs(out_dir, exist_ok=True)

    print("Fitting sequential model: A → B → C → GS\n")

    results_d = fit_sequential(
        'my_test/CBK-D_373NAnEn_newdepo0614_sample_NAnEn_SaphWLC_476nm_Avg2_5scan_25uW -bg -chirp.csv',
        'CBK-D (deuterated)'
    )

    results_h = fit_sequential(
        'my_test/CBK-H_232NAnEn_newdepo0614_sample_NAnEn_SaphWLC_476nm_Avg2_5scan_25uW -bg -chirp.csv',
        'CBK-H (protonated)'
    )

    # Generate all plots
    plot_ta_comparison(results_d, results_h, out_dir)
    plot_kinetic_fits(results_d, results_h, out_dir)
    plot_residuals(results_d, results_h, out_dir)
    plot_isotope_comparison(results_d, results_h, out_dir)
    write_summary_table(results_d, results_h, out_dir)

    print(f"\nAll outputs saved to: {out_dir}/")
