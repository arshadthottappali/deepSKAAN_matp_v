"""
Generate A0 portrait poster as PPTX.
A0 = 841 x 1189 mm = 33.1 x 46.8 inches
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Cm, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)
OUT = os.path.join(ROOT, 'POSTER_A0.pptx')

# A0 portrait dimensions
WIDTH = Inches(33.1)
HEIGHT = Inches(46.8)

# Colors
DARK_BLUE = RGBColor(0x1A, 0x52, 0x76)
LIGHT_BLUE = RGBColor(0xEB, 0xF5, 0xFB)
ORANGE = RGBColor(0xE6, 0x7E, 0x22)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RED = RGBColor(0xE7, 0x4C, 0x3C)
GREEN = RGBColor(0x27, 0xAE, 0x60)
GRAY = RGBColor(0x5D, 0x6D, 0x7E)
LIGHT_RED = RGBColor(0xFA, 0xDB, 0xD8)
LIGHT_GREEN = RGBColor(0xD5, 0xF5, 0xE3)
LIGHT_ORANGE = RGBColor(0xFC, 0xF3, 0xCF)

prs = Presentation()
prs.slide_width = WIDTH
prs.slide_height = HEIGHT

slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank


def add_box(left, top, width, height, fill_color=None, border_color=None):
    """Add a rectangle shape."""
    shape = slide.shapes.add_shape(1, left, top, width, height)
    if fill_color:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_color
    else:
        shape.fill.background()
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(2)
    else:
        shape.line.fill.background()
    return shape


def add_text(left, top, width, height, lines, bg_color=None):
    """Add text box. lines = [(text, size, bold, color, align), ...]"""
    if bg_color:
        add_box(left, top, width, height, fill_color=bg_color, border_color=DARK_BLUE)
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(lines):
        text, size, bold, color = item[:4]
        align = item[4] if len(item) > 4 else PP_ALIGN.LEFT
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.font.size = Pt(size)
        p.font.bold = bold
        p.alignment = align
        if color:
            p.font.color.rgb = color
    return tb


def add_image(left, top, height, filename):
    """Add image if it exists."""
    path = os.path.join(PROJECT, filename)
    if os.path.exists(path):
        slide.shapes.add_picture(path, left, top, height=height)
        return True
    else:
        print(f'  WARNING: image not found: {filename}')
        return False


# Dimensions (in inches, A0 = 33.1 x 46.8)
# Margins
M = 0.8  # margin
COL_W = (33.1 - 3*M) / 2   # two columns
LEFT_COL = Inches(M)
RIGHT_COL = Inches(M + COL_W + M)
COL = Inches(COL_W)
FULL_W = Inches(33.1 - 2*M)

# ══════════════════════════════════════════════════════════════════════════
# TITLE BAND
# ══════════════════════════════════════════════════════════════════════════
add_box(Inches(0), Inches(0), WIDTH, Inches(4.5), fill_color=DARK_BLUE)

add_text(Inches(1), Inches(0.4), Inches(31), Inches(2.0), [
    ("Challenges in Interpreting Ultrafast Spectroscopy Data:", 52, True, WHITE, PP_ALIGN.CENTER),
    ("A Case for Machine Learning in Materials Research", 48, True, ORANGE, PP_ALIGN.CENTER),
])

add_text(Inches(1), Inches(2.8), Inches(31), Inches(1.5), [
    ("Muhammed Arshad Thottappali", 32, True, WHITE, PP_ALIGN.CENTER),
    ("Institute of Macromolecular Chemistry, Czech Academy of Sciences, Prague", 24, False, RGBColor(0xAE, 0xD6, 0xF1), PP_ALIGN.CENTER),
    ("ML4MatSci 2nd PhD Summer School  \u2022  Sarajevo  \u2022  June 10\u201312, 2026", 20, False, RGBColor(0xAE, 0xD6, 0xF1), PP_ALIGN.CENTER),
])

# ══════════════════════════════════════════════════════════════════════════
# SECTION 1: THE PROBLEM (left column)
# ══════════════════════════════════════════════════════════════════════════
Y = 5.0

add_box(LEFT_COL, Inches(Y), COL, Inches(1.0), fill_color=DARK_BLUE)
add_text(LEFT_COL, Inches(Y+0.1), COL, Inches(0.8), [
    ("1. THE CHALLENGE: Ultrafast Spectroscopy Data", 28, True, WHITE, PP_ALIGN.CENTER),
])

Y += 1.2
add_text(LEFT_COL, Inches(Y), COL, Inches(5.0), [
    ("Transient Absorption (TA) spectroscopy probes photo-induced", 20, False, None),
    ("dynamics on femtosecond\u2013nanosecond timescales.", 20, False, None),
    ("", 10, False, None),
    ("Key Challenges:", 22, True, RED),
    ("\u2022  Model ambiguity: many kinetic models fit the same data", 19, False, None),
    ("\u2022  Overlapping, broad excited-state spectral features", 19, False, None),
    ("\u2022  Expert bottleneck: manual model guessing takes hours", 19, False, None),
    ("\u2022  Subjective \u2014 different experts may choose different models", 19, False, None),
    ("", 10, False, None),
    ("Our System:", 22, True, DARK_BLUE),
    ("\u2022  H/D-substituted porphyrin complex (CBK-H, CBK-D)", 19, False, None),
    ("\u2022  Excitation 476 nm, white-light continuum probe (430\u2013800 nm)", 19, False, None),
    ("\u2022  Sequential relaxation: A \u2192 B \u2192 C \u2192 Ground State", 19, True, None),
], bg_color=LIGHT_BLUE)

Y += 5.5
add_image(LEFT_COL, Inches(Y), Inches(5.5), 'results/sequential/ta_heatmaps.png')

# ══════════════════════════════════════════════════════════════════════════
# SECTION 2: GLOBAL ANALYSIS RESULTS (right column top)
# ══════════════════════════════════════════════════════════════════════════
Y = 5.0

add_box(RIGHT_COL, Inches(Y), COL, Inches(1.0), fill_color=DARK_BLUE)
add_text(RIGHT_COL, Inches(Y+0.1), COL, Inches(0.8), [
    ("2. GLOBAL ANALYSIS: Forced Sequential Model", 28, True, WHITE, PP_ALIGN.CENTER),
])

Y += 1.2
add_text(RIGHT_COL, Inches(Y), COL, Inches(5.5), [
    ("Model: A \u2192 B \u2192 C \u2192 GS (3 exponentials + Gaussian IRF)", 20, True, DARK_BLUE),
    ("Method: Differential evolution + Levenberg-Marquardt polish", 18, False, GRAY),
    ("", 10, False, None),
    ("Results (PINN, full 2D fit):", 20, True, GREEN),
    ("", 8, False, None),
    ("  Component     CBK-D        CBK-H        D/H Ratio", 18, False, None),
    ("  \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500", 18, False, None),
    ("  \u03c4\u2081 (A\u2192B)       9.6 ps       13.4 ps      0.72", 18, True, None),
    ("  \u03c4\u2082 (B\u2192C)       92.6 ps      166.8 ps     0.56", 18, True, None),
    ("  \u03c4\u2083 (C\u2192GS)      >6 ns        >6 ns        2.35", 18, True, None),
    ("", 10, False, None),
    ("ANOMALOUS ISOTOPE EFFECT:", 22, True, RED),
    ("\u03c4\u2081, \u03c4\u2082: Deuterium is FASTER (inverse KIE)", 20, True, RED),
    ("Expected: D should relax MORE SLOWLY", 18, False, GRAY),
    ("\u2192 Suggests tunneling or vibrational mode coupling", 18, False, None),
], bg_color=LIGHT_GREEN)

Y += 6.0
add_image(RIGHT_COL, Inches(Y), Inches(5.0), 'results/sequential_full/eads.png')

# ══════════════════════════════════════════════════════════════════════════
# SECTION 3: DeepSKAN CNN (left column middle)
# ══════════════════════════════════════════════════════════════════════════
Y = 17.5

add_box(LEFT_COL, Inches(Y), COL, Inches(1.0), fill_color=DARK_BLUE)
add_text(LEFT_COL, Inches(Y+0.1), COL, Inches(0.8), [
    ("3. ML APPROACH: DeepSKAN (CNN Classification)", 28, True, WHITE, PP_ALIGN.CENTER),
])

Y += 1.2
add_text(LEFT_COL, Inches(Y), COL, Inches(5.5), [
    ("DeepSKAN (Kollenz et al., J. Phys. Chem. B, 2020)", 20, True, DARK_BLUE),
    ("DOI: 10.1021/acs.jpcb.0c04299", 16, False, GRAY),
    ("", 8, False, None),
    ("Core Idea:", 20, True, None),
    ("Treat the 2D TA map as a grayscale IMAGE and use a", 18, False, None),
    ("CNN (ResNet) to CLASSIFY the kinetic topology.", 18, False, None),
    ("", 8, False, None),
    ("Our Implementation:", 20, True, DARK_BLUE),
    ("\u2022  7-block ResNet + BatchNorm + Dropout", 18, False, None),
    ("\u2022  Input: 256\u00d764 normalized TA map", 18, False, None),
    ("\u2022  Output: probability over 21 kinetic model classes", 18, False, None),
    ("\u2022  Trained on 100k synthetic physics-simulated samples", 18, False, None),
    ("\u2022  Validation accuracy: 72% (T4 GPU, 40 epochs)", 18, False, None),
    ("", 8, False, None),
    ("Pipeline:", 20, True, None),
    ("  TA map \u2192 CNN \u2192 topology class \u2192 GA/TA fitting \u2192 \u03c4 values", 18, True, None),
])

Y += 6.0
add_image(LEFT_COL, Inches(Y), Inches(4.5), 'results/training_curve.png')

# ══════════════════════════════════════════════════════════════════════════
# SECTION 4: CNN PREDICTIONS ON REAL DATA (right column middle)
# ══════════════════════════════════════════════════════════════════════════
Y = 17.5

add_box(RIGHT_COL, Inches(Y), COL, Inches(1.0), fill_color=DARK_BLUE)
add_text(RIGHT_COL, Inches(Y+0.1), COL, Inches(0.8), [
    ("4. CNN PREDICTIONS ON REAL DATA", 28, True, WHITE, PP_ALIGN.CENTER),
])

Y += 1.2
add_text(RIGHT_COL, Inches(Y), COL, Inches(3.5), [
    ("Predictions:", 22, True, DARK_BLUE),
    ("", 8, False, None),
    ("  CBK-H: Class 6 (A\u2192B\u2192C, sequential)", 20, True, GREEN),
    ("          Confidence: 59% \u2014 CLOSE to correct!", 18, False, GREEN),
    ("", 8, False, None),
    ("  CBK-D: Class 19 (branching: A\u2192B, A\u2192C, B\u2192C, B\u2192D)", 20, True, RED),
    ("          Confidence: 40% \u2014 WRONG (we know it's sequential)", 18, False, RED),
    ("", 8, False, None),
    ("Automated GA with CNN-predicted model:", 18, False, None),
    ("  \u03c4\u2081 = 0.1 ps, \u03c4\u2082 = 0.0 ps  \u2190 NONSENSICAL", 20, True, RED),
], bg_color=LIGHT_RED)

Y += 4.0
# Confidence charts
add_image(RIGHT_COL, Inches(Y), Inches(4.0),
    'results/CBK-D/CBK-D_373NAnEn_newdepo0614_sample_NAnEn_SaphWLC_476nm_Avg2_5scan_25uW -bg -chirp_confidence.png')

Y += 4.3
add_image(RIGHT_COL, Inches(Y), Inches(4.0),
    'results/CBK-H/CBK-H_232NAnEn_newdepo0614_sample_NAnEn_SaphWLC_476nm_Avg2_5scan_25uW -bg -chirp_confidence.png')

# ══════════════════════════════════════════════════════════════════════════
# SECTION 5: WHY IT FAILED (full width)
# ══════════════════════════════════════════════════════════════════════════
Y = 30.5

add_box(LEFT_COL, Inches(Y), FULL_W, Inches(1.0), fill_color=DARK_BLUE)
add_text(LEFT_COL, Inches(Y+0.1), FULL_W, Inches(0.8), [
    ("5. WHY DID THE CNN FAIL? \u2014 The Sim-to-Real Gap", 28, True, WHITE, PP_ALIGN.CENTER),
])

Y += 1.2
add_text(LEFT_COL, Inches(Y), Inches(15), Inches(5.0), [
    ("Root Causes:", 22, True, RED),
    ("", 8, False, None),
    ("1. Synthetic training spectra (random Gaussians) \u2260 real molecular spectra", 19, False, None),
    ("2. Preprocessing (normalize to 256\u00d764) destroyed time-scale information", 19, False, None),
    ("3. Classification \u2192 wrong model \u2192 GA fitting collapses to \u03c4=0", 19, False, None),
    ("4. Low confidence (40%) correctly flags uncertainty \u2014 a feature, not a bug", 19, False, None),
    ("", 10, False, None),
    ("What Would Fix It:", 22, True, GREEN),
    ("", 8, False, None),
    ("\u2022  Train on realistic spectra (from SVD of real data) \u2014 tested, works!", 19, False, None),
    ("\u2022  Regression (predict \u03c4 directly) instead of classification", 19, False, None),
    ("\u2022  Physics-informed approach (PINN) \u2014 embed ODEs in the loss function", 19, False, None),
    ("\u2022  Agentic AI: LLM reads literature \u2192 proposes model \u2192 fits data", 19, False, None),
], bg_color=LIGHT_ORANGE)

add_image(RIGHT_COL, Inches(Y), Inches(5.0), 'poster/comparison_table.png')

# ══════════════════════════════════════════════════════════════════════════
# SECTION 6: CONCLUSIONS + FUTURE (full width)
# ══════════════════════════════════════════════════════════════════════════
Y = 37.0

add_box(LEFT_COL, Inches(Y), FULL_W, Inches(1.0), fill_color=DARK_BLUE)
add_text(LEFT_COL, Inches(Y+0.1), FULL_W, Inches(0.8), [
    ("6. CONCLUSIONS + FUTURE DIRECTIONS", 28, True, WHITE, PP_ALIGN.CENTER),
])

Y += 1.2
add_text(LEFT_COL, Inches(Y), Inches(15), Inches(5.5), [
    ("ML IS Useful For:", 22, True, GREEN),
    ("\u2713  Automated screening of many samples (combinatorial libraries)", 19, False, None),
    ("\u2713  First-pass model suggestion when no prior knowledge exists", 19, False, None),
    ("\u2713  Flagging unusual/unexpected kinetics", 19, False, None),
    ("\u2713  Reducing human bias in model selection", 19, False, None),
    ("", 10, False, None),
    ("ML is NOT (yet) a Replacement For:", 22, True, RED),
    ("\u2717  Quantitative parameter extraction (when model is known)", 19, False, None),
    ("\u2717  Systems with strong sim-to-real gap", 19, False, None),
    ("\u2717  Low-confidence predictions without expert validation", 19, False, None),
    ("", 10, False, None),
    ("FUTURE: Agentic AI for Spectroscopy", 24, True, ORANGE),
    ("\u2022  LLM agent reads molecular structure + published literature", 19, False, None),
    ("\u2022  Proposes kinetic model with chemical assignments", 19, False, None),
    ("\u2022  Fits data with physics-constrained method (PINN)", 19, False, None),
    ("\u2022  No training data needed, no sim-to-real gap", 19, False, None),
    ("\u2022  Combines domain expertise with automated fitting", 19, False, None),
])

add_image(RIGHT_COL, Inches(Y), Inches(5.5), 'pinn_learning/output/PINN_isotope_effect_bar.png')

# ══════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════
Y = 44.0
add_box(Inches(0), Inches(Y), WIDTH, Inches(2.8), fill_color=DARK_BLUE)
add_text(Inches(1), Inches(Y+0.3), Inches(31), Inches(2.2), [
    ("References:", 18, True, WHITE),
    ("\u2022 Kollenz et al., J. Phys. Chem. B (2020) \u2014 DeepSKAN original     \u2022 van Stokkum et al., BBA (2004) \u2014 Global/Target Analysis", 16, False, RGBColor(0xAE, 0xD6, 0xF1)),
    ("", 8, False, None),
    ("Tools: Python, PyTorch 2.x, SciPy, Lightning.ai (T4 GPU)          Contact: arshad@imc.cas.cz", 16, False, RGBColor(0xAE, 0xD6, 0xF1)),
])

# ─── Save ─────────────────────────────────────────────────────────────────
prs.save(OUT)
size_mb = os.path.getsize(OUT) / 1024 / 1024
print(f'Saved: {OUT}')
print(f'Size: {size_mb:.1f} MB')
print('\nSections:')
print('  1. The Challenge (TA data + why it is hard)')
print('  2. Global Analysis (forced sequential model + isotope effect)')
print('  3. ML Approach: DeepSKAN CNN (training + architecture)')
print('  4. CNN Predictions (confidence charts, what it predicted)')
print('  5. Why It Failed (sim-to-real gap + fixes)')
print('  6. Conclusions + Future (agentic AI)')
