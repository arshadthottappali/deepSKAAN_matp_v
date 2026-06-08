"""
STEP 1 — Understand the data and the physics
=============================================
Before building any PINN, we need to understand:
  1. What our data looks like
  2. What equations describe the physics (the kinetic ODEs)
  3. What we want to extract (decay times)

This script just LOADS and VISUALIZES — no ML yet.
Run it, read the comments, look at the output plot.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
# Add parent directory to path so we can import deepGTA_torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepGTA_torch.preprocessing import load_surface_xplorer_csv


# ════════════════════════════════════════════════════════════════════════════
# THE PHYSICS — what are we modeling?
# ════════════════════════════════════════════════════════════════════════════
#
# We have a sequential relaxation:   A → B → C → Ground State
#
# Each arrow has a rate constant k (units: 1/time).
# The lifetime tau = 1/k.
#
# The populations evolve according to coupled differential equations:
#
#   dA/dt = -k1 * A                  (A only decays)
#   dB/dt = +k1 * A - k2 * B         (B fed by A, decays to C)
#   dC/dt = +k2 * B - k3 * C         (C fed by B, decays to GS)
#   dGS/dt = +k3 * C                 (ground state fills up)
#
# At t=0 (after the laser pulse): A=1, B=C=GS=0
#
# The MEASURED signal is:
#   ΔA(wavelength, time) = A(t)·ε_A(λ) + B(t)·ε_B(λ) + C(t)·ε_C(λ)
#
# where ε_i(λ) are the species spectra (EADS).
#
# GOAL: find k1, k2, k3 (the decay rates) and the spectra ε_i(λ).
# ════════════════════════════════════════════════════════════════════════════


def explore_data(filepath, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    t, wl, data, meta = load_surface_xplorer_csv(filepath)

    print(f"  Data matrix shape: {data.shape}  (wavelengths × time points)")
    print(f"  Time axis:       {t[0]:.2f} to {t[-1]:.1f} ps   ({len(t)} points)")
    print(f"  Wavelength axis: {wl[0]:.1f} to {wl[-1]:.1f} nm   ({len(wl)} points)")
    print(f"  Signal range:    {data.min():.4f} to {data.max():.4f}")

    # Negative time = before the laser pulse arrives (baseline)
    n_neg = np.sum(t < 0)
    print(f"  Negative-time points (pre-pulse baseline): {n_neg}")

    return t, wl, data


def plot_data_overview(t, wl, data, label, ax_map, ax_kinetic, ax_spectrum):
    """Show the 3 ways to look at TA data."""

    # 1. The full 2D map (time vs wavelength)
    mask = t > 0
    v = np.percentile(np.abs(data), 99)
    im = ax_map.pcolormesh(t[mask], wl, data[:, mask], cmap='RdBu_r',
                           vmin=-v, vmax=v, shading='auto')
    ax_map.set_xscale('log')
    ax_map.set_xlabel('Time (ps)')
    ax_map.set_ylabel('Wavelength (nm)')
    ax_map.set_title(f'{label}\n2D TA Map')

    # 2. Kinetic traces — pick a few wavelengths, plot signal vs time
    #    This shows the DECAY we want to fit
    wl_picks = [int(len(wl)*0.25), int(len(wl)*0.5), int(len(wl)*0.75)]
    for idx in wl_picks:
        ax_kinetic.semilogx(t[mask], data[idx, mask], label=f'{wl[idx]:.0f} nm')
    ax_kinetic.set_xlabel('Time (ps)')
    ax_kinetic.set_ylabel('ΔA')
    ax_kinetic.set_title('Kinetic traces\n(what decays over time)')
    ax_kinetic.legend(fontsize=8)
    ax_kinetic.axhline(0, color='gray', lw=0.5)

    # 3. Spectra — pick a few times, plot signal vs wavelength
    #    This shows the SPECTRA of the species
    t_pos = t[mask]
    data_pos = data[:, mask]
    time_picks = [int(len(t_pos)*0.05), int(len(t_pos)*0.3), int(len(t_pos)*0.9)]
    for idx in time_picks:
        ax_spectrum.plot(wl, data_pos[:, idx], label=f'{t_pos[idx]:.0f} ps')
    ax_spectrum.set_xlabel('Wavelength (nm)')
    ax_spectrum.set_ylabel('ΔA')
    ax_spectrum.set_title('Spectra at different times\n(species fingerprints)')
    ax_spectrum.legend(fontsize=8)
    ax_spectrum.axhline(0, color='gray', lw=0.5)


if __name__ == '__main__':
    import os
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(ROOT, 'pinn_learning/output'), exist_ok=True)

    files = [
        (os.path.join(ROOT, 'my_test/CBK-D_373NAnEn_newdepo0614_sample_NAnEn_SaphWLC_476nm_Avg2_5scan_25uW -bg -chirp.csv'), 'CBK-D'),
        (os.path.join(ROOT, 'my_test/CBK-H_232NAnEn_newdepo0614_sample_NAnEn_SaphWLC_476nm_Avg2_5scan_25uW -bg -chirp.csv'), 'CBK-H'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for row, (filepath, label) in enumerate(files):
        t, wl, data = explore_data(filepath, label)
        plot_data_overview(t, wl, data, label,
                           axes[row, 0], axes[row, 1], axes[row, 2])

    fig.suptitle('STEP 1: Understanding TA Data — 3 Views of the Same Dataset',
                 fontsize=14)
    fig.tight_layout()
    out = os.path.join(ROOT, 'pinn_learning/output/step1_data_overview.png')
    fig.savefig(out, dpi=150)
    print(f"\n\nSaved: {out}")
    print("\nLook at the plot. Notice:")
    print("  - Left:   the full 2D map (color = signal strength)")
    print("  - Middle: kinetic traces — these DECAY, and we fit the decay times")
    print("  - Right:  spectra change shape over time = species interconverting")
