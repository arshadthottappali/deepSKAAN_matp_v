"""
plotting.py
===========
Visualisation utilities for DeepSKAN results.

Phase 3 additions:
  - plot_confidence_bar(): FIX #17 — bar chart of top-N CNN class probabilities
    so the user can see at a glance how certain the prediction is.
  - Improved plot_ta_heatmap(): proper axis labels (time / wavelength), log time axis.
  - Improved plot_transients(): log time axis, proper labels.
  - Improved plot_sads(): wavelength pixel axis label.
"""

import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for scripts and notebooks
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx


# ── TA heatmap ────────────────────────────────────────────────────────────────

def plot_ta_heatmap(
    X: np.ndarray,
    title: str = 'TA Data',
    t: np.ndarray = None,
    wl: np.ndarray = None,
) -> plt.Figure:
    """
    Plot a TA map as a filled contour heatmap.

    Args:
        X:     2-D array (time × wavelength).  If t/wl not given, pixel indices used.
        title: Figure title.
        t:     Optional time axis (ps).
        wl:    Optional wavelength axis (nm).
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    x_vals = wl  if wl  is not None else np.arange(X.shape[1])
    y_vals = t   if t   is not None else np.arange(X.shape[0])

    # Symmetric colour scale centred on zero (diverging ΔA map)
    v_abs = np.nanmax(np.abs(X))
    v_abs = v_abs if v_abs > 0 else 1.0
    levels = np.linspace(-v_abs, v_abs, 64)

    c = ax.contourf(x_vals, y_vals, X, levels=levels, cmap='RdBu_r')
    fig.colorbar(c, ax=ax, label='ΔA (norm.)')

    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Wavelength (nm)' if wl is not None else 'Wavelength (pixels)')
    ax.set_ylabel('Time (ps)'       if t  is not None else 'Time (pixels)')

    if t is not None and np.all(t > 0):
        ax.set_yscale('log')

    fig.tight_layout()
    return fig


# ── Kinetic model graph ───────────────────────────────────────────────────────

def plot_kinetic_model(K_pred: np.ndarray, num_species: int = 4) -> plt.Figure:
    """
    Draw the predicted kinetic connectivity as a directed graph.
    Edge labels show the fractional weight of each pathway.
    """
    K_binary = np.zeros((num_species, num_species))
    edge_labels = {}

    for i in range(num_species):
        for j in range(num_species):
            if K_pred[i, j] > 0.5:
                K_binary[i, j] = 1
                edge_labels[(j, i)] = f"{K_pred[i, j]:.2f}"   # (from, to) in networkx

    fig, ax = plt.subplots(figsize=(5, 5))
    G   = nx.from_numpy_array(K_binary.T, create_using=nx.DiGraph())
    pos = nx.circular_layout(G)
    node_labels = {i: chr(65 + i) for i in range(num_species)}

    node_colors = ['#AED6F1' if i < num_species - 1 else '#A9DFBF'
                   for i in range(num_species)]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, font_color='black', ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='#2C3E50', arrows=True,
                           arrowsize=20, width=2, ax=ax,
                           connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_size=8, ax=ax)

    ax.set_title('Predicted Kinetic Model', fontsize=13)
    ax.axis('off')
    fig.tight_layout()
    return fig


# ── Concentration transients ──────────────────────────────────────────────────

def plot_transients(
    plot_c: np.ndarray,
    num_species: int = 4,
    t: np.ndarray = None,
) -> plt.Figure:
    """
    Plot species concentration profiles vs time.

    Args:
        plot_c:      (256, num_species) concentration matrix.
        num_species: total number of species (last one = ground state, excluded).
        t:           Optional time axis (ps) for proper x-axis labelling.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    x_vals = t if t is not None else np.arange(plot_c.shape[0])
    colors = plt.cm.tab10(np.linspace(0, 1, num_species - 1))

    for i in range(num_species - 1):   # exclude ground state
        ax.plot(x_vals, plot_c[:, i], label=f'Species {chr(65 + i)}',
                color=colors[i], linewidth=2)

    ax.set_xlabel('Time (ps)' if t is not None else 'Time (index)')
    ax.set_ylabel('Population (norm.)')
    ax.set_title('Concentration Profiles (Target Analysis)')
    ax.legend(frameon=True, fontsize=10)

    if t is not None and np.all(t > 0):
        ax.set_xscale('log')

    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    fig.tight_layout()
    return fig


# ── SADS ──────────────────────────────────────────────────────────────────────

def plot_sads(sads: np.ndarray, num_species: int = 4,
              wl: np.ndarray = None) -> plt.Figure:
    """
    Plot Species-Associated Difference Spectra.

    Args:
        sads:        (num_species, wavelength_points) array.
        num_species: number of species (including ground state).
        wl:          Optional wavelength axis (nm).
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    x_vals = wl if wl is not None else np.arange(sads.shape[1])
    colors = plt.cm.tab10(np.linspace(0, 1, num_species))

    for i in range(num_species):
        ax.plot(x_vals, sads[i], label=f'Species {chr(65 + i)}',
                color=colors[i], linewidth=2)

    ax.set_xlabel('Wavelength (nm)' if wl is not None else 'Wavelength (pixels)')
    ax.set_ylabel('ΔA (norm.)')
    ax.set_title('Species-Associated Difference Spectra (SADS)')
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.legend(frameon=True, fontsize=10)
    fig.tight_layout()
    return fig


# ── Confidence bar chart (FIX #17) ───────────────────────────────────────────

def plot_confidence_bar(probs: np.ndarray, top_n: int = 10) -> plt.Figure:
    """
    FIX #17: Bar chart of the top-N CNN class probabilities.

    Gives the researcher an immediate visual of:
      - how confident the prediction is
      - whether there are close runner-up classes that merit inspection

    Args:
        probs: (num_classes,) softmax probability array.
        top_n: how many top classes to show (default 10).
    """
    top_n   = min(top_n, len(probs))
    top_idx = np.argsort(probs)[::-1][:top_n]
    top_p   = probs[top_idx]

    # Colour bars: green if top-1, orange if close (>0.10), grey otherwise
    threshold_close = 0.10
    colors = []
    for i, p in enumerate(top_p):
        if i == 0:
            colors.append('#27AE60')    # green  — top-1
        elif p > threshold_close:
            colors.append('#E67E22')    # orange — notable alternative
        else:
            colors.append('#AEB6BF')    # grey   — minor

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(top_n), top_p * 100, color=colors, edgecolor='white', linewidth=0.8)

    ax.set_xticks(range(top_n))
    ax.set_xticklabels([f"Class {c}" for c in top_idx], rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Probability (%)')
    ax.set_title(f'CNN Prediction Confidence — Top {top_n} Classes')
    ax.set_ylim(0, max(top_p * 100) * 1.20)

    # Annotate bars with exact percentage
    for bar, p in zip(bars, top_p):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{p*100:.1f}%',
            ha='center', va='bottom', fontsize=8,
        )

    fig.tight_layout()
    return fig
