"""
build_poster.py
==============
Assemble the complete ML4MatSci poster as a single A0 portrait image.
Combines all figures, text sections, title, and footer into one PNG.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import os

ROOT = 'c:/Users/jamal/Downloads/deepSKAN_public-master/deepSKAN_public-master'
RES = os.path.join(ROOT, 'results')
POSTER = os.path.join(ROOT, 'poster')

# File paths
CBK_D = "CBK-D_373NAnEn_newdepo0614_sample_NAnEn_SaphWLC_476nm_Avg2_5scan_25uW -bg -chirp"
CBK_H = "CBK-H_232NAnEn_newdepo0614_sample_NAnEn_SaphWLC_476nm_Avg2_5scan_25uW -bg -chirp"

FIGS = {
    'raw_d':       os.path.join(RES, 'CBK-D', f'{CBK_D}_raw.png'),
    'raw_h':       os.path.join(RES, 'CBK-H', f'{CBK_H}_raw.png'),
    'conf_d':      os.path.join(RES, 'CBK-D', f'{CBK_D}_confidence.png'),
    'conf_h':      os.path.join(RES, 'CBK-H', f'{CBK_H}_confidence.png'),
    'training':    os.path.join(RES, 'training_curve.png'),
    'eads':        os.path.join(RES, 'sequential_full', 'eads.png'),
    'conc':        os.path.join(RES, 'sequential_full', 'concentrations.png'),
    'eads_comp':   os.path.join(RES, 'sequential_full', 'eads_comparison.png'),
    'isotope':     os.path.join(RES, 'sequential', 'isotope_effect.png'),
    'scheme':      os.path.join(POSTER, 'kinetic_scheme.png'),
    'pipeline':    os.path.join(POSTER, 'pipeline_overview.png'),
    'comparison':  os.path.join(POSTER, 'comparison_table.png'),
}

# Colors
C_PRIMARY = '#1A5276'
C_ACCENT = '#E67E22'
C_BG = '#FFFFFF'
C_BOX = '#EBF5FB'
C_BOX_BORDER = '#2980B9'
C_FAIL = '#FADBD8'
C_SUCCESS = '#D5F5E3'


def add_image(ax, path, title=None):
    """Add an image to an axis."""
    if os.path.exists(path):
        img = mpimg.imread(path)
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, f'[missing:\n{os.path.basename(path)}]',
                ha='center', va='center', fontsize=6, color='red')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', color=C_PRIMARY, pad=4)


def section_header(ax, num, text):
    """Draw a numbered section header bar."""
    ax.axis('off')
    rect = mpatches.FancyBboxPatch((0.0, 0.0), 1.0, 1.0,
                                    boxstyle="round,pad=0.02",
                                    facecolor=C_PRIMARY, edgecolor='none',
                                    transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(0.015, 0.5, f'{num}', fontsize=20, fontweight='bold',
            color=C_ACCENT, va='center', ha='left', transform=ax.transAxes)
    ax.text(0.08, 0.5, text, fontsize=14, fontweight='bold',
            color='white', va='center', ha='left', transform=ax.transAxes)


def text_box(ax, text, facecolor=C_BOX, fontsize=9):
    """Draw a text box."""
    ax.axis('off')
    ax.text(0.02, 0.98, text, fontsize=fontsize, va='top', ha='left',
            transform=ax.transAxes, wrap=True,
            bbox=dict(boxstyle='round,pad=0.5', facecolor=facecolor,
                     edgecolor=C_BOX_BORDER, linewidth=1))


# ═══ BUILD POSTER ═══════════════════════════════════════════════════════════

# A0 portrait: 841 × 1189 mm. At 100 dpi → ~33 × 47 inches
fig = plt.figure(figsize=(24, 34), facecolor=C_BG)
gs = GridSpec(100, 12, figure=fig, hspace=0.4, wspace=0.3,
              left=0.03, right=0.97, top=0.985, bottom=0.015)

# ─── TITLE BAND ───────────────────────────────────────────────────────────
ax_title = fig.add_subplot(gs[0:6, :])
ax_title.axis('off')
title_rect = mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.01",
                                      facecolor=C_PRIMARY, edgecolor='none',
                                      transform=ax_title.transAxes)
ax_title.add_patch(title_rect)
ax_title.text(0.5, 0.70,
              'Challenges in Interpreting Ultrafast Spectroscopy Data:',
              ha='center', va='center', fontsize=26, fontweight='bold',
              color='white', transform=ax_title.transAxes)
ax_title.text(0.5, 0.45,
              'A Case for Machine Learning in Materials Research',
              ha='center', va='center', fontsize=24, fontweight='bold',
              color=C_ACCENT, transform=ax_title.transAxes)
ax_title.text(0.5, 0.18,
              'Muhammed Arshad Thottappali  •  Institute of Macromolecular Chemistry, Czech Academy of Sciences',
              ha='center', va='center', fontsize=14, color='white',
              transform=ax_title.transAxes)
ax_title.text(0.5, 0.04,
              'ML4MatSci 2nd PhD Summer School  •  Sarajevo, Bosnia and Herzegovina  •  10–12 June 2026',
              ha='center', va='center', fontsize=11, color='#AED6F1', style='italic',
              transform=ax_title.transAxes)

# ─── SECTION 1: THE PROBLEM ───────────────────────────────────────────────
ax = fig.add_subplot(gs[7:9, 0:6]); section_header(ax, '1', 'The Problem: Why TA Analysis is Hard')

ax = fig.add_subplot(gs[9:18, 0:6])
text_box(ax,
    "Transient Absorption (TA) spectroscopy probes photo-induced dynamics on\n"
    "fs–ns timescales. Raw data = 2D matrix ΔA(wavelength, time).\n\n"
    "KEY CHALLENGES:\n"
    "•  Model ambiguity — many kinetic models fit the same data\n"
    "•  Overlapping, broad excited-state spectra\n"
    "•  High dimensionality (~300 λ × 250 time = 75,000 points)\n"
    "•  Expert bottleneck — manual model guessing takes HOURS per sample\n"
    "    and is subjective\n\n"
    "OUR SYSTEM:\n"
    "•  Two isotopic variants (H- and D-substituted molecular material)\n"
    "•  Known sequential relaxation:  A → B → C → Ground State\n"
    "•  Question: Can ML automate the model-selection step?",
    fontsize=11)

# ─── SECTION 2: APPROACH ──────────────────────────────────────────────────
ax = fig.add_subplot(gs[7:9, 6:12]); section_header(ax, '2', 'The Approach: DeepSKAN (CNN)')

ax = fig.add_subplot(gs[9:14, 6:12])
text_box(ax,
    "CORE IDEA: Treat the 2D TA map as a grayscale IMAGE and train a\n"
    "Convolutional Neural Network (CNN) to classify the kinetic topology.\n\n"
    "•  7-block ResNet, asymmetric kernels (time vs wavelength)\n"
    "•  BatchNorm + Dropout regularization\n"
    "•  Input 256×64 → output 21 kinetic-model classes (4 species)\n"
    "•  Trained ENTIRELY on synthetic physics-simulated data\n"
    "    (no labeled experimental data needed)",
    fontsize=11)

ax = fig.add_subplot(gs[14:18, 6:12]); add_image(ax, FIGS['pipeline'])

# ─── SECTION 2b: training curve ───────────────────────────────────────────
ax = fig.add_subplot(gs[19:27, 0:6]); add_image(ax, FIGS['training'], 'Training: 72% val accuracy (21 classes)')

ax = fig.add_subplot(gs[19:27, 6:12]); add_image(ax, FIGS['scheme'], 'Known Kinetic Scheme')

# ─── SECTION 3: CNN ON REAL DATA ──────────────────────────────────────────
ax = fig.add_subplot(gs[28:30, 0:12]); section_header(ax, '3', 'Applying DeepSKAN to Real Experimental Data')

ax = fig.add_subplot(gs[30:37, 0:4]); add_image(ax, FIGS['conf_d'], 'CBK-D: Class 19, 40% (branching — WRONG)')
ax = fig.add_subplot(gs[30:37, 4:8]); add_image(ax, FIGS['conf_h'], 'CBK-H: Class 6, 59% (sequential — close!)')

ax = fig.add_subplot(gs[30:37, 8:12])
text_box(ax,
    "CNN RESULTS:\n\n"
    "•  CBK-H → Class 6 (A→B→C)\n"
    "   sequential, almost correct!\n\n"
    "•  CBK-D → Class 19 (branching)\n"
    "   WRONG (we know it's sequential)\n\n"
    "•  Confidence LOW (40–59%)\n"
    "   → model honestly reports\n"
    "      its own uncertainty\n\n"
    "•  Automated GA gave\n"
    "   τ₁=0.1 ps, τ₂=0.0 ps\n"
    "   → NONSENSICAL",
    facecolor=C_FAIL, fontsize=10)

# ─── SECTION 4: DIRECT GA ─────────────────────────────────────────────────
ax = fig.add_subplot(gs[38:40, 0:12]); section_header(ax, '4', 'Direct Global Analysis (Known Sequential Model)')

ax = fig.add_subplot(gs[40:48, 0:6]); add_image(ax, FIGS['eads'], 'EADS (Evolution-Associated Difference Spectra)')
ax = fig.add_subplot(gs[40:48, 6:12]); add_image(ax, FIGS['conc'], 'Concentration Profiles (A→B→C→GS)')

ax = fig.add_subplot(gs[49:56, 0:6]); add_image(ax, FIGS['isotope'], 'Kinetic Isotope Effect (D vs H)')

ax = fig.add_subplot(gs[49:56, 6:12])
text_box(ax,
    "DIRECT METHOD (SVD + 3-exp + IRF, differential evolution):\n\n"
    "  Component      CBK-D       CBK-H      D/H\n"
    "  ───────────────────────────────────────────\n"
    "  τ₁ (A→B)      9.4 ps      13.8 ps    0.69\n"
    "  τ₂ (B→C)      91.8 ps     173.9 ps   0.53\n"
    "  τ₃ (C→GS)     >6 ns       >6 ns      ~2.5\n"
    "  ───────────────────────────────────────────\n\n"
    "•  IRF ≈ 70 fs (matches laser system)\n"
    "•  Fit residuals ~10⁻⁴ (excellent)\n"
    "•  Normal isotope effect on τ₃ (slow decay)\n"
    "•  Inverse isotope effect on τ₁, τ₂\n"
    "   → suggests tunneling / mode coupling",
    facecolor=C_SUCCESS, fontsize=11)

# ─── SECTION 5: WHY IT FAILED ─────────────────────────────────────────────
ax = fig.add_subplot(gs[57:59, 0:12]); section_header(ax, '5', 'Why Did DeepSKAN Fail? — The Sim-to-Real Gap')

ax = fig.add_subplot(gs[59:67, 0:6]); add_image(ax, FIGS['comparison'])

ax = fig.add_subplot(gs[59:67, 6:12])
text_box(ax,
    "ROOT CAUSES:\n\n"
    "1.  PREPROCESSING destroyed information —\n"
    "    interpolation to 256×64 + normalization\n"
    "    removed amplitude & time-scale the fitter needs\n\n"
    "2.  TRAINING DATA too simple — random Gaussian\n"
    "    spectra ≠ real molecular spectra\n\n"
    "3.  LOW CONFIDENCE = honest uncertainty\n"
    "    (a feature, not a bug)\n\n"
    "4.  21 CLASSES inherently ambiguous —\n"
    "    several topologies give similar TA maps\n\n"
    "FIXES: realistic noise, real-spectra templates,\n"
    "separate classification from quantitative fitting,\n"
    "hybrid CNN-suggest → expert-validate workflow",
    facecolor=C_FAIL, fontsize=10)

# ─── SECTION 6: CONCLUSIONS ───────────────────────────────────────────────
ax = fig.add_subplot(gs[68:70, 0:12]); section_header(ax, '6', 'Conclusions: Where ML Helps and Where It Does Not')

ax = fig.add_subplot(gs[70:78, 0:6])
text_box(ax,
    "ML / DL IS USEFUL FOR:\n\n"
    "✓  Automated screening of many samples\n"
    "✓  First-pass model suggestion when no\n"
    "    prior knowledge exists\n"
    "✓  Flagging unusual / unexpected kinetics\n"
    "✓  Reducing human bias in model selection",
    facecolor=C_SUCCESS, fontsize=11)

ax = fig.add_subplot(gs[70:78, 6:12])
text_box(ax,
    "ML / DL is NOT (yet) a replacement for:\n\n"
    "✗  Quantitative parameter extraction\n"
    "✗  Systems where the model is already known\n"
    "✗  Low-confidence predictions without\n"
    "    expert validation\n"
    "✗  Data with a large sim-to-real gap",
    facecolor=C_FAIL, fontsize=11)

# ─── IDEAL WORKFLOW ───────────────────────────────────────────────────────
ax = fig.add_subplot(gs[79:84, 0:12])
text_box(ax,
    "THE IDEAL HYBRID WORKFLOW:\n\n"
    "   Real TA data  →  CNN screens topology (fast, automated)  →  Expert validates  →\n"
    "   Classical Global Analysis with constrained model  →  EADS + concentrations + decay times (publication-quality)",
    facecolor='#FCF3CF', fontsize=12)

# ─── EADS COMPARISON (extra figure) ───────────────────────────────────────
ax = fig.add_subplot(gs[85:94, 0:12]); add_image(ax, FIGS['eads_comp'], 'EADS: Deuterated vs Protonated — Direct Comparison')

# ─── FOOTER ───────────────────────────────────────────────────────────────
ax_footer = fig.add_subplot(gs[95:100, :])
ax_footer.axis('off')
footer_rect = mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.01",
                                       facecolor=C_PRIMARY, edgecolor='none',
                                       transform=ax_footer.transAxes)
ax_footer.add_patch(footer_rect)
ax_footer.text(0.02, 0.5,
    "References:  Kollenz et al., J. Phys. Chem. B (2020)  •  van Stokkum et al., BBA (2004)  •  Selvaraju et al., ICCV (2017)",
    fontsize=10, color='white', va='center', ha='left', transform=ax_footer.transAxes)
ax_footer.text(0.98, 0.5,
    "Framework: PyTorch  •  Training: T4 GPU, 50k synthetic samples",
    fontsize=10, color='#AED6F1', va='center', ha='right', transform=ax_footer.transAxes)

# ─── SAVE ─────────────────────────────────────────────────────────────────
out_path = os.path.join(POSTER, 'POSTER_full.png')
fig.savefig(out_path, dpi=120, bbox_inches='tight', facecolor=C_BG)
print(f'Saved poster: {out_path}')

# Also save a PDF (vector, best for printing)
out_pdf = os.path.join(POSTER, 'POSTER_full.pdf')
fig.savefig(out_pdf, bbox_inches='tight', facecolor=C_BG)
print(f'Saved poster: {out_pdf}')
plt.close()
