"""
Pitch slides v2 — based on Arshad's preferred narrative:
Slide 1: Real science (TA data + forced GA + anomalous isotope effect)
Slide 2: Can ML help? DeepSKAN approach + predictions + lessons
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)
OUT = os.path.join(ROOT, '11_MuhammedArshad_pitch_v2.pptx')

DARK_BLUE = RGBColor(0x1A, 0x52, 0x76)
ORANGE = RGBColor(0xE6, 0x7E, 0x22)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RED = RGBColor(0xE7, 0x4C, 0x3C)
GREEN = RGBColor(0x27, 0xAE, 0x60)
GRAY = RGBColor(0x5D, 0x6D, 0x7E)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)


def title_bar(slide, text, subtitle=""):
    shape = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.333), Inches(1.2))
    shape.fill.solid()
    shape.fill.fore_color.rgb = DARK_BLUE
    shape.line.fill.background()
    tb = slide.shapes.add_textbox(Inches(0.4), Inches(0.1), Inches(12.5), Inches(0.65))
    p = tb.text_frame.paragraphs[0]
    p.text = text
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = WHITE
    if subtitle:
        tb2 = slide.shapes.add_textbox(Inches(0.4), Inches(0.7), Inches(12.5), Inches(0.4))
        p2 = tb2.text_frame.paragraphs[0]
        p2.text = subtitle
        p2.font.size = Pt(12)
        p2.font.color.rgb = RGBColor(0xAE, 0xD6, 0xF1)


def add_text_block(slide, left, top, width, height, lines):
    """lines: list of (text, size, bold, color) tuples"""
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, (text, size, bold, color) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.font.size = Pt(size)
        p.font.bold = bold
        if color:
            p.font.color.rgb = color


# ══════════════════════════════════════════════════════════════════════════
# SLIDE 1: Real Science — TA data + Global Analysis + Isotope Effect
# ══════════════════════════════════════════════════════════════════════════
s1 = prs.slides.add_slide(prs.slide_layouts[6])
title_bar(s1,
    "Ultrafast Dynamics of H/D-Substituted Porphyrin: An Anomalous Isotope Effect",
    "Muhammed Arshad T.  |  Institute of Macromolecular Chemistry, Czech Academy of Sciences  |  ML4MatSci 2026")

# Left: description + results
add_text_block(s1, 0.4, 1.4, 5.0, 5.8, [
    ("Transient Absorption Spectroscopy", 16, True, DARK_BLUE),
    ("", 6, False, None),
    ("System: CBK porphyrin complex", 13, False, None),
    ("  • CBK-H (protonated) vs CBK-D (deuterated)", 12, False, None),
    ("  • Excitation: 476 nm, WLC probe 430-800 nm", 12, False, None),
    ("", 6, False, None),
    ("Forced Sequential Model: A \u2192 B \u2192 C \u2192 GS", 14, True, DARK_BLUE),
    ("(Global Analysis with 3 exponentials + IRF)", 11, False, GRAY),
    ("", 6, False, None),
    ("Results:", 14, True, GREEN),
    ("  Component    CBK-D      CBK-H      D/H", 11, False, None),
    ("  \u03c4\u2081 (A\u2192B)     9.6 ps     13.4 ps    0.72", 11, False, None),
    ("  \u03c4\u2082 (B\u2192C)     92.6 ps    166.8 ps   0.56", 11, False, None),
    ("  \u03c4\u2083 (C\u2192GS)    >6 ns      >6 ns      2.35", 11, False, None),
    ("", 6, False, None),
    ("ANOMALY:", 14, True, RED),
    ("\u03c4\u2081, \u03c4\u2082: D is FASTER (inverse isotope effect)", 12, True, RED),
    ("Expected: D should be SLOWER", 12, False, GRAY),
    ("\u2192 Suggests tunneling or mode-coupling", 12, False, None),
    ("", 6, False, None),
    ("But: we had to FORCE the sequential model.", 12, True, ORANGE),
    ("Can ML suggest the right model automatically?", 12, True, ORANGE),
])

# Right: figures (TA heatmaps + EADS)
# Add TA heatmap if exists
img_ta = os.path.join(PROJECT, 'results', 'sequential', 'ta_heatmaps.png')
if os.path.exists(img_ta):
    s1.shapes.add_picture(img_ta, Inches(5.6), Inches(1.4), height=Inches(2.8))

# Add EADS if exists
img_eads = os.path.join(PROJECT, 'results', 'sequential_full', 'eads.png')
if os.path.exists(img_eads):
    s1.shapes.add_picture(img_eads, Inches(5.6), Inches(4.4), height=Inches(2.8))


# ══════════════════════════════════════════════════════════════════════════
# SLIDE 2: Can ML Help? — DeepSKAN + our version + predictions
# ══════════════════════════════════════════════════════════════════════════
s2 = prs.slides.add_slide(prs.slide_layouts[6])
title_bar(s2,
    "Can ML Automate Kinetic Model Selection?",
    "From DeepSKAN (Kollenz et al. JPCB 2020) to our improved implementation")

# Left column: DeepSKAN concept
add_text_block(s2, 0.4, 1.4, 6.0, 5.8, [
    ("DeepSKAN: CNN for Kinetic Topology Classification", 14, True, DARK_BLUE),
    ("Kollenz et al., J. Phys. Chem. B (2020)", 10, False, GRAY),
    ("DOI: 10.1021/acs.jpcb.0c04299", 10, False, GRAY),
    ("", 6, False, None),
    ("Idea: Treat 2D TA map as IMAGE \u2192 classify topology", 12, False, None),
    ("", 4, False, None),
    ("  TA map (256\u00d764) \u2192 ResNet CNN \u2192 1 of 21 classes", 12, True, None),
    ("", 6, False, None),
    ("Our Implementation:", 13, True, DARK_BLUE),
    ("  \u2022 7-block ResNet + BatchNorm + Dropout", 11, False, None),
    ("  \u2022 Trained on 100k synthetic samples (T4 GPU)", 11, False, None),
    ("  \u2022 72% validation accuracy on 21 classes", 11, False, None),
    ("", 6, False, None),
    ("Predictions on Real Data:", 13, True, ORANGE),
    ("", 4, False, None),
    ("  CBK-D: Class 19 (branching), confidence 40%", 12, False, None),
    ("  CBK-H: Class 6  (sequential), confidence 59%", 12, False, None),
    ("", 6, False, None),
    ("  CBK-H prediction is CLOSE to correct!", 12, True, GREEN),
    ("  CBK-D prediction is WRONG (we know it's sequential)", 12, True, RED),
    ("", 6, False, None),
    ("Key Insight:", 13, True, DARK_BLUE),
    ("  \u2022 Low confidence = model honestly reports uncertainty", 11, False, None),
    ("  \u2022 Sim-to-real gap: synthetic training \u2260 real spectra", 11, False, None),
    ("  \u2022 Classification is hard; regression (predict \u03c4) works better", 11, False, None),
])

# Right: confidence chart + schematic
img_conf_d = os.path.join(PROJECT, 'results', 'CBK-D',
    'CBK-D_373NAnEn_newdepo0614_sample_NAnEn_SaphWLC_476nm_Avg2_5scan_25uW -bg -chirp_confidence.png')
img_conf_h = os.path.join(PROJECT, 'results', 'CBK-H',
    'CBK-H_232NAnEn_newdepo0614_sample_NAnEn_SaphWLC_476nm_Avg2_5scan_25uW -bg -chirp_confidence.png')
img_pipeline = os.path.join(ROOT, 'pipeline_overview.png')

if os.path.exists(img_conf_d):
    s2.shapes.add_picture(img_conf_d, Inches(6.5), Inches(1.3), height=Inches(2.0))
if os.path.exists(img_conf_h):
    s2.shapes.add_picture(img_conf_h, Inches(6.5), Inches(3.4), height=Inches(2.0))
if os.path.exists(img_pipeline):
    s2.shapes.add_picture(img_pipeline, Inches(6.5), Inches(5.5), height=Inches(1.7))

# Footer
add_text_block(s2, 6.5, 5.5, 6.5, 1.5, [
    ("Future: Agentic AI (LLM + literature \u2192 model suggestion)", 11, True, ORANGE),
    ("\u2192 No training data, no sim-to-real gap", 10, False, GRAY),
])


# ─── Save ─────────────────────────────────────────────────────────────────
prs.save(OUT)
print(f'Saved: {OUT}')
print(f'Size: {os.path.getsize(OUT)/1024:.0f} KB')
