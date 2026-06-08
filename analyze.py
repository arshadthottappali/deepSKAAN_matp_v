"""
analyze.py
==========
End-to-end DeepSKAN analysis pipeline:
  1. Load + preprocess real TA data (CSV)
  2. CNN predicts kinetic model
  3. Global Analysis  (GA)  — finds decay times
  4. Target Analysis  (TA)  — fits concentration profiles + SADS
  5. Save result plots

Phase 3 additions:
  FIX #17 — confidence threshold warning; top-3 alternatives printed when low.
  FIX #19 — input validation: CSV shape, time axis, species / model mismatch.

Usage:
    python analyze.py --data my_test/sample.csv --model checkpoints/best_model.pt

    # Override confidence threshold (default 0.40)
    python analyze.py --data ... --model ... --min_confidence 0.60

    # Run only CNN prediction, skip GA/TA (useful for quick screening)
    python analyze.py --data ... --model ... --predict_only
"""

import argparse
import os
import sys
import warnings

import numpy as np

from deepGTA_torch.preprocessing import load_surface_xplorer_csv, preprocess_for_cnn
from deepGTA_torch.analyzer import DLAnalyzer, GTAnalyzer, LOW_CONFIDENCE_THRESHOLD
from deepGTA_torch.plotting import (
    plot_ta_heatmap,
    plot_kinetic_model,
    plot_transients,
    plot_sads,
    plot_confidence_bar,
)
from deepGTA_torch.gradcam import GradCAM, plot_gradcam


# ── Input validation ──────────────────────────────────────────────────────────

def _validate_inputs(args, t, wl, data, dl_analyzer):
    """
    FIX #19: validate data and model compatibility before running the pipeline.
    Raises SystemExit with a clear message on failure.
    """
    errors = []

    # 1. Time axis must have positive values
    if not np.any(t > 0):
        errors.append(
            "Time axis has no positive values. "
            "Check that the CSV is in Surface Xplorer format with time in ps."
        )

    # 2. Data matrix must not be all zeros or all NaN
    if np.all(data == 0) or np.all(np.isnan(data)):
        errors.append("Data matrix is all-zero or all-NaN after loading.")

    # 3. Wavelength axis needs at least 4 points for interpolation
    if len(wl) < 4:
        errors.append(
            f"Wavelength axis has only {len(wl)} points; need at least 4."
        )

    # 4. Time axis needs at least 10 points
    if len(t) < 10:
        errors.append(
            f"Time axis has only {len(t)} points; need at least 10."
        )

    # 5. num_species in model vs argument
    if hasattr(dl_analyzer, 'num_species') and dl_analyzer.num_species != args.num_species:
        errors.append(
            f"--num_species={args.num_species} but the loaded model was trained "
            f"with num_species={dl_analyzer.num_species}. They must match."
        )

    if errors:
        print("\n[analyze] INPUT VALIDATION FAILED:")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)

    print("[analyze] Input validation passed.")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def analyze():
    parser = argparse.ArgumentParser(description="DeepSKAN analysis pipeline")
    parser.add_argument('--data',           required=True,  help='Path to CSV TA data')
    parser.add_argument('--model',          required=True,  help='Path to model checkpoint (.pt)')
    parser.add_argument('--num_species',    type=int, default=5)
    parser.add_argument('--out_dir',        default='results')
    parser.add_argument('--min_confidence', type=float, default=LOW_CONFIDENCE_THRESHOLD,
                        help=f'CNN confidence threshold for warning (default {LOW_CONFIDENCE_THRESHOLD})')
    parser.add_argument('--predict_only',   action='store_true',
                        help='Stop after CNN prediction — skip GA and TA')
    parser.add_argument('--top_k',          type=int, default=3,
                        help='Number of top kinetic models to display (default 3)')
    parser.add_argument('--gradcam',        action='store_true',
                        help='FIX #22: save Grad-CAM explanation overlay image')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(args.data))[0]

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("[analyze] Loading data...")
    t, wl, data, metadata = load_surface_xplorer_csv(args.data)
    print(f"[analyze] Data shape: {data.shape}  "
          f"t=[{t[0]:.2f}, {t[-1]:.2f}] ps  "
          f"wl=[{wl[0]:.1f}, {wl[-1]:.1f}] nm")

    # Save raw TA heatmap
    fig_raw = plot_ta_heatmap(data.T, title=f"Raw TA — {basename}")
    fig_raw.savefig(os.path.join(args.out_dir, f"{basename}_raw.png"), dpi=150)
    print(f"[analyze] Saved raw heatmap.")

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    print("[analyze] Preprocessing for CNN...")
    X_cnn = preprocess_for_cnn(t, wl, data)   # (1, 256, 64)

    # ── 3. Load model + validate ──────────────────────────────────────────────
    print("[analyze] Loading model...")
    dl_analyzer = DLAnalyzer(args.model, num_species=args.num_species)

    # FIX #19: validate inputs before doing any expensive computation
    _validate_inputs(args, t, wl, data, dl_analyzer)

    # ── 4. CNN prediction ─────────────────────────────────────────────────────
    print("[analyze] Predicting kinetic model...")
    probs    = dl_analyzer.predict(X_cnn)        # FIX #17: warns if low confidence
    class_id = int(np.argmax(probs))
    top1_conf = float(probs[class_id])

    print(f"[analyze] Top-1 prediction: class {class_id}  "
          f"confidence={top1_conf:.3f}  "
          f"{'⚠ LOW' if top1_conf < args.min_confidence else '✓ OK'}")

    # FIX #17: show top-k alternatives so user can make an informed decision
    top_k = dl_analyzer.get_top_k_predictions(probs, k=args.top_k)
    print(f"[analyze] Top-{args.top_k} predictions:")
    for rank, (cid, prob, _) in enumerate(top_k, 1):
        print(f"    {rank}. class {cid:>3d}  p={prob:.4f}")

    # Plot top-1 kinetic model graph
    K_pred, _ = dl_analyzer.get_prediction_binary_matrix(probs)
    fig_k = plot_kinetic_model(K_pred, args.num_species)
    fig_k.savefig(os.path.join(args.out_dir, f"{basename}_kinetic_model.png"), dpi=150)

    # Plot CNN confidence bar chart (top-10)
    fig_conf = plot_confidence_bar(probs, top_n=10)
    fig_conf.savefig(os.path.join(args.out_dir, f"{basename}_confidence.png"), dpi=150)

    # FIX #22: optional Grad-CAM explanation
    if args.gradcam:
        print("[analyze] Computing Grad-CAM...")
        try:
            gcam    = GradCAM(dl_analyzer.model, target_layer=dl_analyzer.model.res7)
            cam     = gcam.compute(X_cnn, class_id)
            fig_gc  = plot_gradcam(
                X_cnn[0, 0], cam,
                title=f"Grad-CAM — class {class_id}  p={top1_conf:.2f}",
            )
            fig_gc.savefig(os.path.join(args.out_dir, f"{basename}_gradcam.png"), dpi=150)
            gcam.remove_hooks()
            print(f"[analyze] Grad-CAM saved.")
        except Exception as e:
            print(f"[analyze] Grad-CAM failed: {e}")

    if args.predict_only:
        print("[analyze] --predict_only set. Stopping after CNN prediction.")
        print(f"[analyze] Results saved to {args.out_dir}/")
        return

    # ── 5. Global Analysis ────────────────────────────────────────────────────
    num_decays, offset = dl_analyzer.get_num_decays(probs)
    print(f"[analyze] Running Global Analysis  "
          f"(num_decays={num_decays}, offset={offset})...")

    t_min  = float(t[t > 0].min()) if np.any(t > 0) else 0.1
    t_min  = max(0.1, t_min)
    t_max  = max(1000.0, float(t[-1]))
    data_t = np.logspace(np.log10(t_min), np.log10(t_max), 256)

    gta = GTAnalyzer(X_cnn[0], data_t)

    # FIX #16: differential evolution — more robust than random restarts
    fit = gta.get_best_ga(
        decay_min=1.0, decay_max=t_max,
        num_decays=num_decays,
        irf_start=10.0, tz_start=0.0,
        offset=offset,
    )

    fit_decays = fit.x[:num_decays]
    fit_irf    = fit.x[num_decays]
    fit_tz     = fit.x[num_decays + 1]
    fit_cost   = float(fit.cost)

    print(f"[analyze] GA result:")
    for i, d in enumerate(fit_decays):
        print(f"    τ_{i+1} = {d:.1f} ps")
    print(f"    IRF   = {fit_irf:.2f} ps")
    print(f"    tz    = {fit_tz:.2f} ps")
    print(f"    cost  = {fit_cost:.4e}")

    # ── 6. Target Analysis ────────────────────────────────────────────────────
    print("[analyze] Running Target Analysis...")

    # FIX #15: best permutation chosen by actual fit residual
    # FIX #18: permutation count capped at MAX_PERMUTATIONS
    plot_c, sads, K_ta = gta.get_best_ta(
        class_id, fit_decays, fit_irf, num_species=args.num_species
    )

    fig_transients = plot_transients(plot_c, args.num_species)
    fig_transients.savefig(
        os.path.join(args.out_dir, f"{basename}_transients.png"), dpi=150
    )

    fig_sads = plot_sads(sads, args.num_species)
    fig_sads.savefig(
        os.path.join(args.out_dir, f"{basename}_sads.png"), dpi=150
    )

    # ── 7. Summary ────────────────────────────────────────────────────────────
    print(f"\n[analyze] ✓ Complete. Outputs saved to: {args.out_dir}/")
    print(f"    {basename}_raw.png")
    print(f"    {basename}_kinetic_model.png")
    print(f"    {basename}_confidence.png")
    if args.gradcam:
        print(f"    {basename}_gradcam.png")
    print(f"    {basename}_transients.png")
    print(f"    {basename}_sads.png")


if __name__ == '__main__':
    analyze()
