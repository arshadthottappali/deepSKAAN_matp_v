"""
evaluate.py
===========
FIX #24: Benchmark DeepSKAN on real experimental TA data.

Two evaluation modes
--------------------
1. predict  (default)
   Runs the CNN on one or more real CSV files and reports:
     - top-1 predicted class + confidence
     - top-3 alternatives
     - Grad-CAM overlay saved as PNG
   No ground truth required — useful for blind screening.

2. accuracy  (requires --labels)
   Loads a JSON file that maps CSV filename → true class index,
   then reports per-class and overall accuracy / top-3 accuracy.
   Use this when you have manually verified ground-truth assignments.

Usage
-----
# Predict on a single file
python evaluate.py --model checkpoints/best_model.pt \\
                   --data   my_test/sample.csv

# Predict on all CSVs in a folder + save Grad-CAM images
python evaluate.py --model checkpoints/best_model.pt \\
                   --data   my_test/ \\
                   --out_dir results/eval \\
                   --gradcam

# Accuracy benchmark (needs ground-truth labels JSON)
python evaluate.py --model  checkpoints/best_model.pt \\
                   --data   my_test/ \\
                   --labels my_test/labels.json \\
                   --mode   accuracy

labels.json format:
    {
        "sample1.csv": 42,
        "sample2.csv": 17
    }
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np

from deepGTA_torch.preprocessing import load_surface_xplorer_csv, preprocess_for_cnn
from deepGTA_torch.analyzer      import DLAnalyzer, LOW_CONFIDENCE_THRESHOLD
from deepGTA_torch.plotting      import plot_confidence_bar
from deepGTA_torch.gradcam       import GradCAM, plot_gradcam


# ── Helpers ───────────────────────────────────────────────────────────────────

def _collect_csv_files(path: str) -> list:
    """Return list of CSV paths — accepts a single file or a directory."""
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        files = sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith('.csv')
        )
        if not files:
            print(f"[evaluate] No CSV files found in {path}")
            sys.exit(1)
        return files
    print(f"[evaluate] --data path not found: {path}")
    sys.exit(1)


def _predict_one(dl_analyzer: DLAnalyzer, csv_path: str, num_species: int):
    """
    Load, preprocess, and run CNN on one CSV file.

    Returns:
        (X_cnn, t, wl, probs, class_id, top1_conf)
    """
    t, wl, data, _ = load_surface_xplorer_csv(csv_path)
    X_cnn           = preprocess_for_cnn(t, wl, data)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        probs = dl_analyzer.predict(X_cnn)
        low_conf = any(issubclass(x.category, UserWarning) for x in w)

    class_id  = int(np.argmax(probs))
    top1_conf = float(probs[class_id])
    return X_cnn, t, wl, probs, class_id, top1_conf, low_conf


# ── Predict mode ──────────────────────────────────────────────────────────────

def run_predict(args, dl_analyzer: DLAnalyzer, csv_files: list):
    """Screen one or more files — no ground truth required."""
    gcam = None
    if args.gradcam:
        gcam = GradCAM(dl_analyzer.model, target_layer=dl_analyzer.model.res7)

    results = []

    for csv_path in csv_files:
        basename = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"\n[evaluate] {basename}")

        X_cnn, t, wl, probs, class_id, top1_conf, low_conf = _predict_one(
            dl_analyzer, csv_path, args.num_species
        )

        flag = '⚠ LOW' if low_conf else '✓'
        print(f"  Top-1 : class {class_id:>3d}  p={top1_conf:.4f}  {flag}")

        top3 = dl_analyzer.get_top_k_predictions(probs, k=3)
        for rank, (cid, prob, _) in enumerate(top3, 1):
            print(f"  Top-{rank} : class {cid:>3d}  p={prob:.4f}")

        # Save confidence bar chart
        fig_conf = plot_confidence_bar(probs, top_n=10)
        fig_conf.savefig(
            os.path.join(args.out_dir, f"{basename}_confidence.png"), dpi=150
        )
        fig_conf.clf()

        # Optionally save Grad-CAM
        if gcam is not None:
            try:
                cam = gcam.compute(X_cnn, class_id)
                fig_gc = plot_gradcam(
                    X_cnn[0, 0], cam,
                    title=f"Grad-CAM — {basename}  class {class_id}  p={top1_conf:.2f}",
                    t=t, wl=wl,
                )
                fig_gc.savefig(
                    os.path.join(args.out_dir, f"{basename}_gradcam.png"), dpi=150
                )
                fig_gc.clf()
                print(f"  Grad-CAM saved.")
            except Exception as e:
                print(f"  Grad-CAM failed: {e}")

        results.append({
            'file':      basename,
            'class_id':  class_id,
            'conf':      top1_conf,
            'low_conf':  low_conf,
        })

    if gcam is not None:
        gcam.remove_hooks()

    # Summary
    print(f"\n[evaluate] Screened {len(results)} file(s).")
    low = [r for r in results if r['low_conf']]
    if low:
        print(f"  ⚠  {len(low)} file(s) had low confidence (<{LOW_CONFIDENCE_THRESHOLD}):")
        for r in low:
            print(f"      {r['file']}  p={r['conf']:.3f}")


# ── Accuracy mode ─────────────────────────────────────────────────────────────

def run_accuracy(args, dl_analyzer: DLAnalyzer, csv_files: list):
    """Compare predictions to ground-truth labels and report accuracy."""
    with open(args.labels) as f:
        label_map = json.load(f)   # {"filename.csv": class_id, ...}

    correct_top1 = correct_top3 = total = 0
    per_class_correct = {}
    per_class_total   = {}

    for csv_path in csv_files:
        fname    = os.path.basename(csv_path)
        if fname not in label_map:
            print(f"[evaluate] No label for {fname} — skipping.")
            continue

        true_id = int(label_map[fname])
        _, _, _, probs, pred_id, top1_conf, _ = _predict_one(
            dl_analyzer, csv_path, args.num_species
        )

        top3_ids = [cid for cid, _, _ in dl_analyzer.get_top_k_predictions(probs, k=3)]

        is_top1 = (pred_id == true_id)
        is_top3 = (true_id in top3_ids)

        correct_top1 += int(is_top1)
        correct_top3 += int(is_top3)
        total        += 1

        per_class_correct[true_id] = per_class_correct.get(true_id, 0) + int(is_top1)
        per_class_total[true_id]   = per_class_total.get(true_id, 0) + 1

        status = '✓' if is_top1 else ('~top3' if is_top3 else '✗')
        print(f"  {status}  {fname:50s}  "
              f"true={true_id:>3d}  pred={pred_id:>3d}  p={top1_conf:.3f}")

    if total == 0:
        print("[evaluate] No labelled files found.")
        return

    print(f"\n[evaluate] Results on {total} labelled file(s):")
    print(f"  Top-1 accuracy : {correct_top1}/{total} = {100*correct_top1/total:.1f}%")
    print(f"  Top-3 accuracy : {correct_top3}/{total} = {100*correct_top3/total:.1f}%")

    # Per-class breakdown
    print("\n  Per-class breakdown (classes with >1 sample):")
    for c_id in sorted(per_class_total):
        n   = per_class_total[c_id]
        acc = per_class_correct.get(c_id, 0)
        if n > 1:
            print(f"    class {c_id:>3d}: {acc}/{n} = {100*acc/n:.0f}%")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Evaluate DeepSKAN on real TA data')
    parser.add_argument('--model',       required=True,
                        help='Path to model checkpoint (.pt)')
    parser.add_argument('--data',        required=True,
                        help='Path to CSV file or folder of CSVs')
    parser.add_argument('--num_species', type=int, default=5)
    parser.add_argument('--out_dir',     default='results/eval')
    parser.add_argument('--mode',        choices=['predict', 'accuracy'],
                        default='predict',
                        help='predict = no labels needed; accuracy = needs --labels')
    parser.add_argument('--labels',      default=None,
                        help='JSON file mapping filename → true class id (accuracy mode)')
    parser.add_argument('--gradcam',     action='store_true',
                        help='Save Grad-CAM overlay images (predict mode only)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[evaluate] Loading model from {args.model}...")
    dl_analyzer = DLAnalyzer(args.model, num_species=args.num_species)

    csv_files = _collect_csv_files(args.data)
    print(f"[evaluate] Found {len(csv_files)} CSV file(s).")

    if args.mode == 'accuracy':
        if not args.labels:
            print("[evaluate] --labels is required for accuracy mode.")
            sys.exit(1)
        run_accuracy(args, dl_analyzer, csv_files)
    else:
        run_predict(args, dl_analyzer, csv_files)

    print(f"\n[evaluate] Outputs saved to: {args.out_dir}/")


if __name__ == '__main__':
    main()
