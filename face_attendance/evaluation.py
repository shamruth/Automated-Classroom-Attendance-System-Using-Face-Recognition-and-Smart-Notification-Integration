# """
# ================================================================================
#   evaluation.py — Face Recognition Model Evaluation Suite
#   --------------------------------------------------------
#   Metrics computed:
#     • Accuracy          • Precision (macro & per-class)
#     • Recall            • F1-Score  (macro & per-class)
#     • FAR  (False Acceptance Rate)
#     • FRR  (False Rejection Rate)
#     • Confusion Matrix  (saved as PNG + printed as table)
#     • Recognition Time  (per-image ms) and FPS

#   Usage
#   ─────
#   # Basic evaluation (uses students/ folder as test set)
#   python evaluation.py

#   # Use a separate labelled test folder
#   python evaluation.py --test-dir test_images/

#   # Change tolerance or model
#   python evaluation.py --tolerance 0.45 --model cnn

#   # Save all plots/reports to a custom folder
#   python evaluation.py --out-dir eval_results/

#   Test-folder layout (same as students/ enrollment layout):
#   ──────────────────────────────────────────────────────────
#   test_images/
#       S001/          ← folder name = ground-truth student ID
#           img1.jpg
#           img2.jpg
#       S002/
#           img1.jpg
#       unknown/       ← images that should NOT match anyone
#           stranger1.jpg
# ================================================================================
# """

# import cv2
# import face_recognition
# import numpy as np
# import pandas as pd
# import sqlite3
# import json
# import time
# import argparse
# import logging
# import warnings
# import sys
# from pathlib import Path
# from datetime import datetime
# from typing import Optional
# from collections import defaultdict

# # Matplotlib for plots (non-interactive backend so it works on headless servers)
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import seaborn as sns

# # Suppress harmless sklearn warnings
# warnings.filterwarnings("ignore", category=UserWarning)

# try:
#     from sklearn.metrics import (
#         accuracy_score, precision_score, recall_score,
#         f1_score, confusion_matrix, classification_report,
#         ConfusionMatrixDisplay,
#     )
# except ImportError:
#     print("ERROR: scikit-learn not found.  Run:  pip install scikit-learn matplotlib seaborn")
#     sys.exit(1)

# # ── Logging ──────────────────────────────────────────────────────────────────
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[logging.StreamHandler(), logging.FileHandler("evaluation.log")],
# )
# log = logging.getLogger(__name__)

# # ── Paths (must match face_attendance.py) ────────────────────────────────────
# BASE_DIR        = Path(__file__).parent
# STUDENTS_DIR    = BASE_DIR / "students"
# ENCODINGS_FILE  = BASE_DIR / "encodings.json"
# IMAGE_EXTS      = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# # ─────────────────────────────────────────────────────────────────────────────
# #  SECTION 1 — Load known encodings (reuse cache built during enrollment)
# # ─────────────────────────────────────────────────────────────────────────────

# def load_known_encodings(cache_path: Path = ENCODINGS_FILE) -> tuple[list, list]:
#     """Load the pre-built face encodings cache produced by face_attendance.py."""
#     if not cache_path.exists():
#         log.error("encodings.json not found. Run face_attendance.py --rebuild first.")
#         sys.exit(1)
#     data      = json.loads(cache_path.read_text())
#     encodings = [np.array(e) for e in data["encodings"]]
#     ids       = data["ids"]
#     log.info("Loaded %d known encoding(s) for %d student(s).",
#              len(ids), len(set(ids)))
#     return encodings, ids


# # ─────────────────────────────────────────────────────────────────────────────
# #  SECTION 2 — Build test dataset from a labelled folder tree
# # ─────────────────────────────────────────────────────────────────────────────

# def build_test_dataset(test_dir: Path, model: str = "hog") -> pd.DataFrame:
#     """
#     Walk test_dir.  For every image:
#       - Detect face(s)
#       - Encode the first detected face
#       - Record ground-truth label (folder name) + raw encoding

#     Returns a DataFrame with columns:
#       image_path | ground_truth | encoding | detection_time_ms
#     """
#     records = []
#     folders = sorted([d for d in test_dir.iterdir() if d.is_dir()])

#     if not folders:
#         log.error("No sub-folders found in %s", test_dir)
#         sys.exit(1)

#     log.info("Scanning test dataset in %s  (%d class folders)…", test_dir, len(folders))

#     for folder in folders:
#         gt_label = folder.name   # Ground truth = folder name (student ID or "unknown")

#         for img_path in sorted(folder.iterdir()):
#             if img_path.suffix.lower() not in IMAGE_EXTS:
#                 continue

#             # --- time the detection step ---
#             t0  = time.perf_counter()
#             img = face_recognition.load_image_file(str(img_path))
#             locs = face_recognition.face_locations(img, model=model)
#             encs = face_recognition.face_encodings(img, locs)
#             elapsed_ms = (time.perf_counter() - t0) * 1000

#             if encs:
#                 records.append({
#                     "image_path"        : str(img_path),
#                     "ground_truth"      : gt_label,
#                     "encoding"          : encs[0],        # first face only
#                     "detection_time_ms" : elapsed_ms,
#                 })
#             else:
#                 log.warning("  No face in %s — skipped.", img_path.name)

#     df = pd.DataFrame(records)
#     log.info("Test dataset: %d images with detected faces.", len(df))
#     return df


# # ─────────────────────────────────────────────────────────────────────────────
# #  SECTION 3 — Predict labels for every test image
# # ─────────────────────────────────────────────────────────────────────────────

# def predict(
#     test_df: pd.DataFrame,
#     known_encodings: list,
#     known_ids: list,
#     tolerance: float,
# ) -> pd.DataFrame:
#     """
#     For each test encoding, compute distances to all known encodings and pick
#     the nearest neighbour.  Label "unknown" if distance > tolerance.

#     Adds columns to test_df:
#       predicted | confidence_pct | match_distance | inference_time_ms
#     """
#     predictions, confidences, distances, inf_times = [], [], [], []

#     for enc in test_df["encoding"]:
#         t0 = time.perf_counter()

#         dists   = face_recognition.face_distance(known_encodings, enc)
#         best_i  = int(np.argmin(dists))
#         best_d  = float(dists[best_i])

#         elapsed = (time.perf_counter() - t0) * 1000

#         if best_d <= tolerance:
#             pred = known_ids[best_i]
#             conf = round((1.0 - best_d) * 100, 2)
#         else:
#             pred = "unknown"
#             conf = 0.0

#         predictions.append(pred)
#         confidences.append(conf)
#         distances.append(round(best_d, 4))
#         inf_times.append(elapsed)

#     result = test_df.copy()
#     result["predicted"]        = predictions
#     result["confidence_pct"]   = confidences
#     result["match_distance"]   = distances
#     result["inference_time_ms"] = inf_times
#     result["total_time_ms"]    = result["detection_time_ms"] + result["inference_time_ms"]
#     return result


# # ─────────────────────────────────────────────────────────────────────────────
# #  SECTION 4 — Compute all evaluation metrics
# # ─────────────────────────────────────────────────────────────────────────────

# def compute_metrics(df: pd.DataFrame, known_student_ids: list) -> dict:
#     """
#     Compute the full evaluation suite.

#     ── Standard Classification Metrics ──────────────────────────────────────
#     Accuracy  = (TP + TN) / Total
#                 Fraction of images correctly identified (or correctly rejected).

#     Precision = TP / (TP + FP)    [macro-averaged across classes]
#                 Of all faces the model labelled as student X, how many were
#                 actually student X?  High precision → few imposters accepted.

#     Recall    = TP / (TP + FN)    [macro-averaged across classes]
#                 Of all faces that actually belong to student X, how many did
#                 the model correctly identify?  High recall → few real students
#                 rejected.

#     F1-Score  = 2 × (Precision × Recall) / (Precision + Recall)
#                 Harmonic mean — balances precision and recall.

#     ── Biometric-Specific Metrics ───────────────────────────────────────────
#     FAR  (False Acceptance Rate)
#          = FP / (FP + TN)
#          Probability that an *unknown* person is incorrectly accepted as a
#          known student.  Critical for security — a high FAR means imposters
#          can be marked present.

#     FRR  (False Rejection Rate)
#          = FN / (FN + TP)
#          Probability that a *genuine* student is incorrectly rejected (marked
#          as Unknown).  High FRR means real students are not marked.

#     EER  (Equal Error Rate) — conceptual point where FAR == FRR.
#          Lower EER → better overall biometric system.

#     ── Timing Metrics ───────────────────────────────────────────────────────
#     Recognition Time = detection time + inference (matching) time per image.
#     FPS = 1000 / avg_total_time_ms  (equivalent frames per second).
#     """

#     y_true = list(df["ground_truth"])
#     y_pred = list(df["predicted"])

#     # ── All unique labels (union of ground-truth and predicted) ───────────────
#     all_labels = sorted(set(y_true) | set(y_pred))

#     # ── Standard metrics ──────────────────────────────────────────────────────
#     accuracy  = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred, average="macro",
#                                 zero_division=0, labels=all_labels)
#     recall    = recall_score(y_true, y_pred, average="macro",
#                              zero_division=0, labels=all_labels)
#     f1        = f1_score(y_true, y_pred, average="macro",
#                          zero_division=0, labels=all_labels)

#     per_class = classification_report(
#         y_true, y_pred, labels=all_labels,
#         zero_division=0, output_dict=True,
#     )

#     cm = confusion_matrix(y_true, y_pred, labels=all_labels)

#     # ── Biometric metrics: FAR & FRR ─────────────────────────────────────────
#     #
#     # Strategy: treat each known student individually in a 1-vs-rest scheme.
#     # For each genuine attempt (ground_truth == student_id):
#     #   • If model ALSO predicts that student  → True Positive  (correct accept)
#     #   • If model predicts "unknown" or wrong → False Negative  → counts in FRR
#     # For each impostor attempt (ground_truth != student_id, i.e. "unknown" or
#     # a different student in the test set):
#     #   • If model predicts that student        → False Positive → counts in FAR
#     #   • If model rejects (predicts "unknown") → True Negative  (correct reject)

#     total_FP = total_FN = total_TP = total_TN = 0

#     for sid in known_student_ids:
#         for gt, pred in zip(y_true, y_pred):
#             genuine  = (gt == sid)      # this image IS that student
#             accepted = (pred == sid)    # model said it IS that student

#             if genuine and accepted:
#                 total_TP += 1           # correct accept
#             elif genuine and not accepted:
#                 total_FN += 1           # genuine user rejected → FRR numerator
#             elif not genuine and accepted:
#                 total_FP += 1           # impostor accepted    → FAR numerator
#             else:
#                 total_TN += 1           # impostor correctly rejected

#     FAR = total_FP / (total_FP + total_TN) if (total_FP + total_TN) > 0 else 0.0
#     FRR = total_FN / (total_FN + total_TP) if (total_FN + total_TP) > 0 else 0.0

#     # ── Timing stats ──────────────────────────────────────────────────────────
#     avg_detection_ms  = df["detection_time_ms"].mean()
#     avg_inference_ms  = df["inference_time_ms"].mean()
#     avg_total_ms      = df["total_time_ms"].mean()
#     fps               = 1000.0 / avg_total_ms if avg_total_ms > 0 else 0.0
#     min_total_ms      = df["total_time_ms"].min()
#     max_total_ms      = df["total_time_ms"].max()

#     return {
#         # Classification
#         "accuracy"           : accuracy,
#         "precision_macro"    : precision,
#         "recall_macro"       : recall,
#         "f1_macro"           : f1,
#         # Biometric
#         "FAR"                : FAR,
#         "FRR"                : FRR,
#         "EER_approx"         : (FAR + FRR) / 2,
#         # Raw confusion counts
#         "TP"                 : total_TP,
#         "FP"                 : total_FP,
#         "TN"                 : total_TN,
#         "FN"                 : total_FN,
#         # Timing
#         "avg_detection_ms"   : avg_detection_ms,
#         "avg_inference_ms"   : avg_inference_ms,
#         "avg_total_ms"       : avg_total_ms,
#         "min_total_ms"       : min_total_ms,
#         "max_total_ms"       : max_total_ms,
#         "fps"                : fps,
#         # Per-class detail
#         "per_class_report"   : per_class,
#         "confusion_matrix"   : cm,
#         "labels"             : all_labels,
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# #  SECTION 5 — Print & save results
# # ─────────────────────────────────────────────────────────────────────────────

# DIVIDER = "=" * 72

# def print_summary(metrics: dict, tolerance: float, model: str, n_images: int):
#     """Print a formatted summary of all metrics to the terminal."""

#     print(f"\n{DIVIDER}")
#     print("  FACE RECOGNITION ATTENDANCE SYSTEM — EVALUATION REPORT")
#     print(f"  Generated : {datetime.now():%Y-%m-%d  %H:%M:%S}")
#     print(f"  Model     : {model.upper()}   Tolerance: {tolerance}   Images tested: {n_images}")
#     print(DIVIDER)

#     # ── Classification metrics ────────────────────────────────────────────────
#     print("\n  ┌─────────────────────────────────────────┐")
#     print("  │     CLASSIFICATION METRICS               │")
#     print("  ├─────────────────────────────────────────┤")
#     print(f"  │  Accuracy        : {metrics['accuracy']*100:6.2f} %             │")
#     print(f"  │  Precision (macro): {metrics['precision_macro']*100:6.2f} %            │")
#     print(f"  │  Recall    (macro): {metrics['recall_macro']*100:6.2f} %            │")
#     print(f"  │  F1-Score  (macro): {metrics['f1_macro']*100:6.2f} %            │")
#     print("  └─────────────────────────────────────────┘")

#     # ── Biometric metrics ─────────────────────────────────────────────────────
#     print("\n  ┌─────────────────────────────────────────┐")
#     print("  │     BIOMETRIC SECURITY METRICS           │")
#     print("  ├─────────────────────────────────────────┤")
#     print(f"  │  FAR  (False Acceptance Rate): {metrics['FAR']*100:6.2f} %  │")
#     print(f"  │  FRR  (False Rejection Rate) : {metrics['FRR']*100:6.2f} %  │")
#     print(f"  │  EER  (approx)               : {metrics['EER_approx']*100:6.2f} %  │")
#     print("  ├─────────────────────────────────────────┤")
#     print(f"  │  True  Positives (TP) : {metrics['TP']:5d}             │")
#     print(f"  │  True  Negatives (TN) : {metrics['TN']:5d}             │")
#     print(f"  │  False Positives (FP) : {metrics['FP']:5d}             │")
#     print(f"  │  False Negatives (FN) : {metrics['FN']:5d}             │")
#     print("  └─────────────────────────────────────────┘")

#     # ── Timing ────────────────────────────────────────────────────────────────
#     print("\n  ┌─────────────────────────────────────────┐")
#     print("  │     RECOGNITION TIME / SPEED             │")
#     print("  ├─────────────────────────────────────────┤")
#     print(f"  │  Avg Detection time  : {metrics['avg_detection_ms']:7.2f} ms        │")
#     print(f"  │  Avg Inference time  : {metrics['avg_inference_ms']:7.2f} ms        │")
#     print(f"  │  Avg Total  time     : {metrics['avg_total_ms']:7.2f} ms        │")
#     print(f"  │  Min Total  time     : {metrics['min_total_ms']:7.2f} ms        │")
#     print(f"  │  Max Total  time     : {metrics['max_total_ms']:7.2f} ms        │")
#     print(f"  │  Equivalent FPS      : {metrics['fps']:7.2f} fps       │")
#     print("  └─────────────────────────────────────────┘")

#     # ── Per-class breakdown ───────────────────────────────────────────────────
#     print("\n  PER-CLASS METRICS:")
#     print(f"  {'Student ID':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
#     print("  " + "-" * 60)
#     report = metrics["per_class_report"]
#     for label in metrics["labels"]:
#         if label not in report:
#             continue
#         r = report[label]
#         print(f"  {label:<15} {r['precision']*100:>9.2f}% {r['recall']*100:>9.2f}% "
#               f"{r['f1-score']*100:>9.2f}% {int(r['support']):>10}")
#     print(DIVIDER + "\n")


# def save_confusion_matrix(metrics: dict, out_dir: Path):
#     """Render and save the confusion matrix as a colour-coded PNG."""
#     cm     = metrics["confusion_matrix"]
#     labels = metrics["labels"]

#     fig, axes = plt.subplots(1, 2, figsize=(max(10, len(labels) * 1.4 + 2),
#                                              max(8, len(labels) * 1.2)))

#     # ── Raw counts ────────────────────────────────────────────────────────────
#     sns.heatmap(
#         cm, annot=True, fmt="d", cmap="Blues",
#         xticklabels=labels, yticklabels=labels,
#         linewidths=0.5, ax=axes[0],
#     )
#     axes[0].set_title("Confusion Matrix — Raw Counts", fontsize=13, fontweight="bold")
#     axes[0].set_xlabel("Predicted Label", fontsize=11)
#     axes[0].set_ylabel("True Label", fontsize=11)
#     axes[0].tick_params(axis="x", rotation=45)
#     axes[0].tick_params(axis="y", rotation=0)

#     # ── Normalised (row-wise recall) ──────────────────────────────────────────
#     cm_norm = cm.astype(float)
#     row_sums = cm_norm.sum(axis=1, keepdims=True)
#     row_sums[row_sums == 0] = 1          # avoid divide by zero
#     cm_norm /= row_sums

#     sns.heatmap(
#         cm_norm, annot=True, fmt=".2f", cmap="Greens",
#         xticklabels=labels, yticklabels=labels,
#         linewidths=0.5, vmin=0, vmax=1, ax=axes[1],
#     )
#     axes[1].set_title("Confusion Matrix — Normalised (Recall %)", fontsize=13, fontweight="bold")
#     axes[1].set_xlabel("Predicted Label", fontsize=11)
#     axes[1].set_ylabel("True Label", fontsize=11)
#     axes[1].tick_params(axis="x", rotation=45)
#     axes[1].tick_params(axis="y", rotation=0)

#     plt.tight_layout()
#     path = out_dir / "confusion_matrix.png"
#     plt.savefig(path, dpi=150, bbox_inches="tight")
#     plt.close()
#     log.info("Confusion matrix saved → %s", path)


# def save_metrics_bar_chart(metrics: dict, out_dir: Path):
#     """Bar chart comparing Accuracy / Precision / Recall / F1 / FAR / FRR."""
#     labels = ["Accuracy", "Precision", "Recall", "F1-Score", "FAR", "FRR"]
#     values = [
#         metrics["accuracy"],
#         metrics["precision_macro"],
#         metrics["recall_macro"],
#         metrics["f1_macro"],
#         metrics["FAR"],
#         metrics["FRR"],
#     ]
#     colours = ["#2ecc71", "#3498db", "#9b59b6", "#1abc9c", "#e74c3c", "#e67e22"]

#     fig, ax = plt.subplots(figsize=(10, 6))
#     bars = ax.bar(labels, [v * 100 for v in values], color=colours,
#                   edgecolor="white", linewidth=1.2, width=0.55)

#     for bar, val in zip(bars, values):
#         ax.text(bar.get_x() + bar.get_width() / 2,
#                 bar.get_height() + 1.0,
#                 f"{val*100:.2f}%",
#                 ha="center", va="bottom", fontsize=11, fontweight="bold")

#     ax.set_ylim(0, 115)
#     ax.set_ylabel("Percentage (%)", fontsize=12)
#     ax.set_title("Face Recognition Model — Evaluation Metrics", fontsize=14, fontweight="bold")
#     ax.axhline(y=100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
#     ax.set_facecolor("#f8f9fa")
#     fig.patch.set_facecolor("#ffffff")
#     ax.grid(axis="y", alpha=0.4, linestyle="--")
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)

#     plt.tight_layout()
#     path = out_dir / "metrics_bar_chart.png"
#     plt.savefig(path, dpi=150, bbox_inches="tight")
#     plt.close()
#     log.info("Metrics bar chart saved → %s", path)


# def save_timing_histogram(df: pd.DataFrame, out_dir: Path):
#     """Histogram of per-image total recognition times."""
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#     # Histogram
#     axes[0].hist(df["total_time_ms"], bins=20, color="#3498db",
#                  edgecolor="white", linewidth=0.8)
#     axes[0].axvline(df["total_time_ms"].mean(), color="red",
#                     linestyle="--", linewidth=1.5, label=f"Mean = {df['total_time_ms'].mean():.1f} ms")
#     axes[0].set_xlabel("Total Recognition Time (ms)", fontsize=11)
#     axes[0].set_ylabel("Image Count", fontsize=11)
#     axes[0].set_title("Recognition Time Distribution", fontsize=13, fontweight="bold")
#     axes[0].legend()
#     axes[0].set_facecolor("#f8f9fa")

#     # Stacked bar: detection vs inference
#     mean_det = df["detection_time_ms"].mean()
#     mean_inf = df["inference_time_ms"].mean()
#     axes[1].bar(["Avg Time"], [mean_det], label=f"Detection  ({mean_det:.1f} ms)",
#                 color="#e67e22", width=0.4)
#     axes[1].bar(["Avg Time"], [mean_inf], bottom=[mean_det],
#                 label=f"Inference  ({mean_inf:.1f} ms)", color="#2ecc71", width=0.4)
#     axes[1].set_ylabel("Time (ms)", fontsize=11)
#     axes[1].set_title(f"Time Breakdown\nFPS ≈ {1000/(mean_det+mean_inf):.1f}",
#                       fontsize=13, fontweight="bold")
#     axes[1].legend()
#     axes[1].set_facecolor("#f8f9fa")
#     for ax in axes:
#         ax.spines["top"].set_visible(False)
#         ax.spines["right"].set_visible(False)
#         ax.grid(axis="y", alpha=0.4, linestyle="--")

#     plt.tight_layout()
#     path = out_dir / "timing_histogram.png"
#     plt.savefig(path, dpi=150, bbox_inches="tight")
#     plt.close()
#     log.info("Timing histogram saved → %s", path)


# def save_per_class_chart(metrics: dict, out_dir: Path):
#     """Per-student Precision / Recall / F1 grouped bar chart."""
#     report = metrics["per_class_report"]
#     labels = [l for l in metrics["labels"] if l in report]

#     prec = [report[l]["precision"] * 100 for l in labels]
#     rec  = [report[l]["recall"]    * 100 for l in labels]
#     f1   = [report[l]["f1-score"]  * 100 for l in labels]

#     x   = np.arange(len(labels))
#     w   = 0.26

#     fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.2 + 2), 6))
#     ax.bar(x - w, prec, w, label="Precision", color="#3498db", edgecolor="white")
#     ax.bar(x,     rec,  w, label="Recall",    color="#2ecc71", edgecolor="white")
#     ax.bar(x + w, f1,   w, label="F1-Score",  color="#9b59b6", edgecolor="white")

#     ax.set_xticks(x)
#     ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=10)
#     ax.set_ylabel("Score (%)", fontsize=12)
#     ax.set_ylim(0, 115)
#     ax.set_title("Per-Student  Precision / Recall / F1-Score", fontsize=13, fontweight="bold")
#     ax.legend(fontsize=11)
#     ax.set_facecolor("#f8f9fa")
#     ax.grid(axis="y", alpha=0.4, linestyle="--")
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)

#     plt.tight_layout()
#     path = out_dir / "per_class_metrics.png"
#     plt.savefig(path, dpi=150, bbox_inches="tight")
#     plt.close()
#     log.info("Per-class chart saved → %s", path)


# def save_csv_report(df: pd.DataFrame, metrics: dict, out_dir: Path):
#     """Save the per-image prediction table + a summary sheet to CSV."""
#     # Per-image results (drop raw encoding — not human-readable)
#     results_df = df.drop(columns=["encoding"], errors="ignore")
#     results_df.to_csv(out_dir / "eval_per_image.csv", index=False)

#     # Summary metrics
#     summary = {
#         "Metric": [
#             "Accuracy (%)", "Precision-Macro (%)", "Recall-Macro (%)", "F1-Macro (%)",
#             "FAR (%)", "FRR (%)", "EER-approx (%)",
#             "TP", "TN", "FP", "FN",
#             "Avg Detection (ms)", "Avg Inference (ms)", "Avg Total (ms)",
#             "Min Total (ms)", "Max Total (ms)", "FPS",
#         ],
#         "Value": [
#             round(metrics["accuracy"]*100, 4),
#             round(metrics["precision_macro"]*100, 4),
#             round(metrics["recall_macro"]*100, 4),
#             round(metrics["f1_macro"]*100, 4),
#             round(metrics["FAR"]*100, 4),
#             round(metrics["FRR"]*100, 4),
#             round(metrics["EER_approx"]*100, 4),
#             metrics["TP"], metrics["TN"], metrics["FP"], metrics["FN"],
#             round(metrics["avg_detection_ms"], 3),
#             round(metrics["avg_inference_ms"], 3),
#             round(metrics["avg_total_ms"], 3),
#             round(metrics["min_total_ms"], 3),
#             round(metrics["max_total_ms"], 3),
#             round(metrics["fps"], 2),
#         ],
#     }
#     pd.DataFrame(summary).to_csv(out_dir / "eval_summary.csv", index=False)
#     log.info("CSV reports saved to %s", out_dir)


# # ─────────────────────────────────────────────────────────────────────────────
# #  SECTION 6 — Tolerance sweep (optional: find optimal threshold)
# # ─────────────────────────────────────────────────────────────────────────────

# def tolerance_sweep(test_df: pd.DataFrame,
#                     known_encodings: list,
#                     known_ids: list,
#                     out_dir: Path,
#                     steps: int = 15):
#     """
#     Evaluate the model at multiple tolerance thresholds and plot how
#     Accuracy, FAR, and FRR change.  Helps choose the optimal tolerance.
#     """
#     log.info("Running tolerance sweep (%d steps)…", steps)
#     thresholds  = np.linspace(0.30, 0.75, steps)
#     accs, fars, frrs, f1s = [], [], [], []

#     known_student_ids = list(set(known_ids))

#     for tol in thresholds:
#         res = predict(test_df, known_encodings, known_ids, tol)
#         m   = compute_metrics(res, known_student_ids)
#         accs.append(m["accuracy"] * 100)
#         fars.append(m["FAR"] * 100)
#         frrs.append(m["FRR"] * 100)
#         f1s.append(m["f1_macro"] * 100)

#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.plot(thresholds, accs, "g-o",  linewidth=2, markersize=5, label="Accuracy")
#     ax.plot(thresholds, f1s,  "b-s",  linewidth=2, markersize=5, label="F1-Score")
#     ax.plot(thresholds, fars, "r--^", linewidth=2, markersize=5, label="FAR")
#     ax.plot(thresholds, frrs, "m--v", linewidth=2, markersize=5, label="FRR")

#     # Mark approx EER crossing point
#     eer_idx = np.argmin(np.abs(np.array(fars) - np.array(frrs)))
#     ax.axvline(thresholds[eer_idx], color="gray", linestyle=":",
#                label=f"EER ≈ tol={thresholds[eer_idx]:.2f}")

#     ax.set_xlabel("Tolerance Threshold", fontsize=12)
#     ax.set_ylabel("Score (%)", fontsize=12)
#     ax.set_title("Tolerance Sweep: Accuracy / F1 / FAR / FRR vs Threshold",
#                  fontsize=13, fontweight="bold")
#     ax.legend(fontsize=11)
#     ax.set_ylim(-5, 105)
#     ax.grid(alpha=0.4, linestyle="--")
#     ax.set_facecolor("#f8f9fa")
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)

#     plt.tight_layout()
#     path = out_dir / "tolerance_sweep.png"
#     plt.savefig(path, dpi=150, bbox_inches="tight")
#     plt.close()
#     log.info("Tolerance sweep chart saved → %s", path)
#     log.info("Suggested optimal tolerance (near EER): %.2f", thresholds[eer_idx])


# # ─────────────────────────────────────────────────────────────────────────────
# #  MAIN
# # ─────────────────────────────────────────────────────────────────────────────

# def main():
#     parser = argparse.ArgumentParser(
#         description="Evaluate the face recognition attendance model",
#         formatter_class=argparse.RawTextHelpFormatter,
#     )
#     parser.add_argument(
#         "--test-dir", default=None,
#         help="Labelled test folder (default: reuse students/ with leave-one-out style).\n"
#              "Layout: test_images/<student_id>/*.jpg",
#     )
#     parser.add_argument("--tolerance", type=float, default=0.5,
#                         help="Match distance threshold (default: 0.5)")
#     parser.add_argument("--model",     default="hog", choices=["hog", "cnn"],
#                         help="Detection backend: hog (CPU) or cnn (GPU)")
#     parser.add_argument("--out-dir",   default="eval_results",
#                         help="Output folder for plots and CSVs")
#     parser.add_argument("--sweep",     action="store_true",
#                         help="Run tolerance sweep to find optimal threshold")
#     args = parser.parse_args()

#     out_dir = Path(args.out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # ── 1. Load known encodings ───────────────────────────────────────────────
#     known_encodings, known_ids = load_known_encodings()
#     known_student_ids = list(set(known_ids))

#     # ── 2. Build test dataset ─────────────────────────────────────────────────
#     test_dir = Path(args.test_dir) if args.test_dir else STUDENTS_DIR
#     if not test_dir.exists():
#         log.error("Test directory not found: %s", test_dir)
#         sys.exit(1)

#     test_df = build_test_dataset(test_dir, model=args.model)
#     if test_df.empty:
#         log.error("No valid test images found. Check folder layout.")
#         sys.exit(1)

#     # ── 3. Run predictions ────────────────────────────────────────────────────
#     log.info("Running predictions (tolerance=%.2f, model=%s)…", args.tolerance, args.model)
#     result_df = predict(test_df, known_encodings, known_ids, args.tolerance)

#     # ── 4. Compute metrics ────────────────────────────────────────────────────
#     metrics = compute_metrics(result_df, known_student_ids)

#     # ── 5. Print to terminal ──────────────────────────────────────────────────
#     print_summary(metrics, args.tolerance, args.model, len(result_df))

#     # ── 6. Save plots and CSVs ────────────────────────────────────────────────
#     save_confusion_matrix(metrics, out_dir)
#     save_metrics_bar_chart(metrics, out_dir)
#     save_timing_histogram(result_df, out_dir)
#     save_per_class_chart(metrics, out_dir)
#     save_csv_report(result_df, metrics, out_dir)

#     # ── 7. Optional tolerance sweep ───────────────────────────────────────────
#     if args.sweep:
#         tolerance_sweep(test_df, known_encodings, known_ids, out_dir)

#     print(f"\n  All evaluation outputs saved to:  {out_dir.resolve()}")
#     print(f"  Files generated:")
#     for f in sorted(out_dir.iterdir()):
#         print(f"    • {f.name}")
#     print()


# if __name__ == "__main__":
#     main()

"""
================================================================================
  evaluation.py — Face Recognition Model Evaluation Suite
  --------------------------------------------------------
  Metrics computed:
    • Accuracy          • Precision (macro & per-class)
    • Recall            • F1-Score  (macro & per-class)
    • FAR  (False Acceptance Rate)
    • FRR  (False Rejection Rate)
    • Confusion Matrix  (saved as PNG + printed as table)
    • Recognition Time  (per-image ms) and FPS

  Usage
  ─────
  # Basic evaluation (uses students/ folder as test set)
  python evaluation.py

  # Use a separate labelled test folder
  python evaluation.py --test-dir test_images/

  # Change tolerance or model
  python evaluation.py --tolerance 0.45 --model cnn

  # Save all plots/reports to a custom folder
  python evaluation.py --out-dir eval_results/

  Test-folder layout (same as students/ enrollment layout):
  ──────────────────────────────────────────────────────────
  test_images/
      S001/          ← folder name = ground-truth student ID
          img1.jpg
          img2.jpg
      S002/
          img1.jpg
      unknown/       ← images that should NOT match anyone
          stranger1.jpg
================================================================================
"""

import cv2
import face_recognition
import numpy as np
import pandas as pd
import sqlite3
import json
import time
import argparse
import logging
import warnings
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from collections import defaultdict

# Matplotlib for plots (non-interactive backend so it works on headless servers)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Suppress harmless sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, classification_report,
        ConfusionMatrixDisplay, roc_curve, precision_recall_curve, auc,
    )
except ImportError:
    print("ERROR: scikit-learn not found.  Run:  pip install scikit-learn matplotlib seaborn")
    sys.exit(1)

# ── Logging ──────────────────────────────────────────────────────────────────
_file_handler   = logging.FileHandler("evaluation.log", encoding="utf-8")
_stream_handler = logging.StreamHandler()
_stream_handler.stream = open(
    _stream_handler.stream.fileno(), mode="w",
    encoding="utf-8", errors="replace", closefd=False, buffering=1,
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[_stream_handler, _file_handler],
)
log = logging.getLogger(__name__)

# ── Paths (must match face_attendance.py) ────────────────────────────────────
BASE_DIR        = Path(__file__).parent
STUDENTS_DIR    = BASE_DIR / "students"
ENCODINGS_FILE  = BASE_DIR / "encodings.json"
IMAGE_EXTS      = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — Load known encodings (reuse cache built during enrollment)
# ─────────────────────────────────────────────────────────────────────────────

def load_known_encodings(cache_path: Path = ENCODINGS_FILE) -> tuple[list, list]:
    """Load the pre-built face encodings cache produced by face_attendance.py."""
    if not cache_path.exists():
        log.error("encodings.json not found. Run face_attendance.py --rebuild first.")
        sys.exit(1)
    data      = json.loads(cache_path.read_text())
    encodings = [np.array(e) for e in data["encodings"]]
    ids       = data["ids"]
    log.info("Loaded %d known encoding(s) for %d student(s).",
             len(ids), len(set(ids)))
    return encodings, ids


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — Build test dataset from a labelled folder tree
# ─────────────────────────────────────────────────────────────────────────────

def build_test_dataset(test_dir: Path, model: str = "hog") -> pd.DataFrame:
    """
    Walk test_dir.  For every image:
      - Detect face(s)
      - Encode the first detected face
      - Record ground-truth label (folder name) + raw encoding

    Returns a DataFrame with columns:
      image_path | ground_truth | encoding | detection_time_ms
    """
    records = []
    folders = sorted([d for d in test_dir.iterdir() if d.is_dir()])

    if not folders:
        log.error("No sub-folders found in %s", test_dir)
        sys.exit(1)

    log.info("Scanning test dataset in %s  (%d class folders)…", test_dir, len(folders))

    for folder in folders:
        gt_label = folder.name   # Ground truth = folder name (student ID or "unknown")

        for img_path in sorted(folder.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue

            # --- time the detection step ---
            t0  = time.perf_counter()
            img = face_recognition.load_image_file(str(img_path))
            locs = face_recognition.face_locations(img, model=model)
            encs = face_recognition.face_encodings(img, locs)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if encs:
                records.append({
                    "image_path"        : str(img_path),
                    "ground_truth"      : gt_label,
                    "encoding"          : encs[0],        # first face only
                    "detection_time_ms" : elapsed_ms,
                })
            else:
                log.warning("  No face in %s — skipped.", img_path.name)

    df = pd.DataFrame(records)
    log.info("Test dataset: %d images with detected faces.", len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — Predict labels for every test image
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    test_df: pd.DataFrame,
    known_encodings: list,
    known_ids: list,
    tolerance: float,
) -> pd.DataFrame:
    """
    For each test encoding, compute distances to all known encodings and pick
    the nearest neighbour.  Label "unknown" if distance > tolerance.

    Adds columns to test_df:
      predicted | confidence_pct | match_distance | inference_time_ms
    """
    predictions, confidences, distances, inf_times = [], [], [], []

    for enc in test_df["encoding"]:
        t0 = time.perf_counter()

        dists   = face_recognition.face_distance(known_encodings, enc)
        best_i  = int(np.argmin(dists))
        best_d  = float(dists[best_i])

        elapsed = (time.perf_counter() - t0) * 1000

        if best_d <= tolerance:
            pred = known_ids[best_i]
            conf = round((1.0 - best_d) * 100, 2)
        else:
            pred = "unknown"
            conf = 0.0

        predictions.append(pred)
        confidences.append(conf)
        distances.append(round(best_d, 4))
        inf_times.append(elapsed)

    result = test_df.copy()
    result["predicted"]        = predictions
    result["confidence_pct"]   = confidences
    result["match_distance"]   = distances
    result["inference_time_ms"] = inf_times
    result["total_time_ms"]    = result["detection_time_ms"] + result["inference_time_ms"]
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — Compute all evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame, known_student_ids: list) -> dict:
    """
    Compute the full evaluation suite.

    ── Standard Classification Metrics ──────────────────────────────────────
    Accuracy  = (TP + TN) / Total
                Fraction of images correctly identified (or correctly rejected).

    Precision = TP / (TP + FP)    [macro-averaged across classes]
                Of all faces the model labelled as student X, how many were
                actually student X?  High precision → few imposters accepted.

    Recall    = TP / (TP + FN)    [macro-averaged across classes]
                Of all faces that actually belong to student X, how many did
                the model correctly identify?  High recall → few real students
                rejected.

    F1-Score  = 2 × (Precision × Recall) / (Precision + Recall)
                Harmonic mean — balances precision and recall.

    ── Biometric-Specific Metrics ───────────────────────────────────────────
    FAR  (False Acceptance Rate)
         = FP / (FP + TN)
         Probability that an *unknown* person is incorrectly accepted as a
         known student.  Critical for security — a high FAR means imposters
         can be marked present.

    FRR  (False Rejection Rate)
         = FN / (FN + TP)
         Probability that a *genuine* student is incorrectly rejected (marked
         as Unknown).  High FRR means real students are not marked.

    EER  (Equal Error Rate) — conceptual point where FAR == FRR.
         Lower EER → better overall biometric system.

    ── Timing Metrics ───────────────────────────────────────────────────────
    Recognition Time = detection time + inference (matching) time per image.
    FPS = 1000 / avg_total_time_ms  (equivalent frames per second).
    """

    y_true = list(df["ground_truth"])
    y_pred = list(df["predicted"])

    # ── All unique labels (union of ground-truth and predicted) ───────────────
    all_labels = sorted(set(y_true) | set(y_pred))

    # ── Standard metrics ──────────────────────────────────────────────────────
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro",
                                zero_division=0, labels=all_labels)
    recall    = recall_score(y_true, y_pred, average="macro",
                             zero_division=0, labels=all_labels)
    f1        = f1_score(y_true, y_pred, average="macro",
                         zero_division=0, labels=all_labels)

    per_class = classification_report(
        y_true, y_pred, labels=all_labels,
        zero_division=0, output_dict=True,
    )

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    # ── Biometric metrics: FAR & FRR ─────────────────────────────────────────
    #
    # Strategy: treat each known student individually in a 1-vs-rest scheme.
    # For each genuine attempt (ground_truth == student_id):
    #   • If model ALSO predicts that student  → True Positive  (correct accept)
    #   • If model predicts "unknown" or wrong → False Negative  → counts in FRR
    # For each impostor attempt (ground_truth != student_id, i.e. "unknown" or
    # a different student in the test set):
    #   • If model predicts that student        → False Positive → counts in FAR
    #   • If model rejects (predicts "unknown") → True Negative  (correct reject)

    total_FP = total_FN = total_TP = total_TN = 0

    for sid in known_student_ids:
        for gt, pred in zip(y_true, y_pred):
            genuine  = (gt == sid)      # this image IS that student
            accepted = (pred == sid)    # model said it IS that student

            if genuine and accepted:
                total_TP += 1           # correct accept
            elif genuine and not accepted:
                total_FN += 1           # genuine user rejected → FRR numerator
            elif not genuine and accepted:
                total_FP += 1           # impostor accepted    → FAR numerator
            else:
                total_TN += 1           # impostor correctly rejected

    FAR = total_FP / (total_FP + total_TN) if (total_FP + total_TN) > 0 else 0.0
    FRR = total_FN / (total_FN + total_TP) if (total_FN + total_TP) > 0 else 0.0

    # ── Timing stats ──────────────────────────────────────────────────────────
    avg_detection_ms  = df["detection_time_ms"].mean()
    avg_inference_ms  = df["inference_time_ms"].mean()
    avg_total_ms      = df["total_time_ms"].mean()
    fps               = 1000.0 / avg_total_ms if avg_total_ms > 0 else 0.0
    min_total_ms      = df["total_time_ms"].min()
    max_total_ms      = df["total_time_ms"].max()

    return {
        # Classification
        "accuracy"           : accuracy,
        "precision_macro"    : precision,
        "recall_macro"       : recall,
        "f1_macro"           : f1,
        # Biometric
        "FAR"                : FAR,
        "FRR"                : FRR,
        "EER_approx"         : (FAR + FRR) / 2,
        # Raw confusion counts
        "TP"                 : total_TP,
        "FP"                 : total_FP,
        "TN"                 : total_TN,
        "FN"                 : total_FN,
        # Timing
        "avg_detection_ms"   : avg_detection_ms,
        "avg_inference_ms"   : avg_inference_ms,
        "avg_total_ms"       : avg_total_ms,
        "min_total_ms"       : min_total_ms,
        "max_total_ms"       : max_total_ms,
        "fps"                : fps,
        # Per-class detail
        "per_class_report"   : per_class,
        "confusion_matrix"   : cm,
        "labels"             : all_labels,
    }


def _compute_open_set_roc_pr(df: pd.DataFrame) -> dict:
    """Compute ROC, PR, AUC and exact EER for the open-set (known vs unknown).

    Uses `confidence_pct` as the score (higher means more likely known).
    Returns dict with arrays + scalar AUC/EER values (or None when not computable).
    """
    if "confidence_pct" not in df.columns:
        log.warning("No confidence_pct column available for ROC/PR computation.")
        return {"roc_auc": None, "pr_auc": None, "eer": None}

    # Binary label: 1 = known student, 0 = unknown
    y_true = np.array([0 if gt == "unknown" else 1 for gt in df["ground_truth"]])
    y_score = (df["confidence_pct"].astype(float).fillna(0.0) / 100.0).to_numpy()

    if len(np.unique(y_true)) < 2:
        log.warning("Not enough positive/negative samples for ROC/PR computation.")
        return {"roc_auc": None, "pr_auc": None, "eer": None}

    fpr, tpr, roc_th = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    precision, recall, pr_th = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)

    # Exact EER: find operating point where FPR ~= FNR (FNR = 1 - TPR)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)

    return {
        "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc,
        "precision_curve": precision, "recall_curve": recall, "pr_auc": pr_auc,
        "eer": eer,
    }


def save_roc_pr(result_df: pd.DataFrame, out_dir: Path):
    """Save ROC and Precision-Recall plots + write scalar scores to disk."""
    scores = _compute_open_set_roc_pr(result_df)
    if scores.get("roc_auc") is None:
        log.warning("Skipping ROC/PR plots (insufficient data).")
        return

    # ROC plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(scores["fpr"], scores["tpr"], label=f"ROC (AUC = {scores['roc_auc']:.3f})", lw=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Open-set ROC (Known vs Unknown)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    path_roc = out_dir / "roc_curve.png"
    plt.tight_layout()
    plt.savefig(path_roc, dpi=150)
    plt.close()
    log.info("ROC curve saved -> %s", path_roc)

    # Precision-Recall plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(scores["recall_curve"], scores["precision_curve"],
            label=f"PR (AUC = {scores['pr_auc']:.3f})", lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall (Known vs Unknown)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    path_pr = out_dir / "pr_curve.png"
    plt.tight_layout()
    plt.savefig(path_pr, dpi=150)
    plt.close()
    log.info("PR curve saved -> %s", path_pr)

    # Save scalar summary
    summary = {
        "metric": ["roc_auc", "pr_auc", "eer"],
        "value": [scores["roc_auc"], scores["pr_auc"], scores["eer"]],
    }
    pd.DataFrame(summary).to_csv(out_dir / "roc_pr_summary.csv", index=False)
    log.info("ROC/PR summary saved -> %s", out_dir / "roc_pr_summary.csv")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5 — Print & save results
# ─────────────────────────────────────────────────────────────────────────────

DIVIDER = "=" * 72

def print_summary(metrics: dict, tolerance: float, model: str, n_images: int):
    """Print a formatted summary of all metrics to the terminal."""

    print(f"\n{DIVIDER}")
    print("  FACE RECOGNITION ATTENDANCE SYSTEM — EVALUATION REPORT")
    print(f"  Generated : {datetime.now():%Y-%m-%d  %H:%M:%S}")
    print(f"  Model     : {model.upper()}   Tolerance: {tolerance}   Images tested: {n_images}")
    print(DIVIDER)

    # ── Classification metrics ────────────────────────────────────────────────
    print("\n  ┌─────────────────────────────────────────┐")
    print("  │     CLASSIFICATION METRICS               │")
    print("  ├─────────────────────────────────────────┤")
    print(f"  │  Accuracy        : {metrics['accuracy']*100:6.2f} %             │")
    print(f"  │  Precision (macro): {metrics['precision_macro']*100:6.2f} %            │")
    print(f"  │  Recall    (macro): {metrics['recall_macro']*100:6.2f} %            │")
    print(f"  │  F1-Score  (macro): {metrics['f1_macro']*100:6.2f} %            │")
    print("  └─────────────────────────────────────────┘")

    # ── Biometric metrics ─────────────────────────────────────────────────────
    print("\n  ┌─────────────────────────────────────────┐")
    print("  │     BIOMETRIC SECURITY METRICS           │")
    print("  ├─────────────────────────────────────────┤")
    print(f"  │  FAR  (False Acceptance Rate): {metrics['FAR']*100:6.2f} %  │")
    print(f"  │  FRR  (False Rejection Rate) : {metrics['FRR']*100:6.2f} %  │")
    print(f"  │  EER  (approx)               : {metrics['EER_approx']*100:6.2f} %  │")
    print("  ├─────────────────────────────────────────┤")
    print(f"  │  True  Positives (TP) : {metrics['TP']:5d}             │")
    print(f"  │  True  Negatives (TN) : {metrics['TN']:5d}             │")
    print(f"  │  False Positives (FP) : {metrics['FP']:5d}             │")
    print(f"  │  False Negatives (FN) : {metrics['FN']:5d}             │")
    print("  └─────────────────────────────────────────┘")

    # ── Timing ────────────────────────────────────────────────────────────────
    print("\n  ┌─────────────────────────────────────────┐")
    print("  │     RECOGNITION TIME / SPEED             │")
    print("  ├─────────────────────────────────────────┤")
    print(f"  │  Avg Detection time  : {metrics['avg_detection_ms']:7.2f} ms        │")
    print(f"  │  Avg Inference time  : {metrics['avg_inference_ms']:7.2f} ms        │")
    print(f"  │  Avg Total  time     : {metrics['avg_total_ms']:7.2f} ms        │")
    print(f"  │  Min Total  time     : {metrics['min_total_ms']:7.2f} ms        │")
    print(f"  │  Max Total  time     : {metrics['max_total_ms']:7.2f} ms        │")
    print(f"  │  Equivalent FPS      : {metrics['fps']:7.2f} fps       │")
    print("  └─────────────────────────────────────────┘")

    # ── Per-class breakdown ───────────────────────────────────────────────────
    print("\n  PER-CLASS METRICS:")
    print(f"  {'Student ID':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("  " + "-" * 60)
    report = metrics["per_class_report"]
    for label in metrics["labels"]:
        if label not in report:
            continue
        r = report[label]
        print(f"  {label:<15} {r['precision']*100:>9.2f}% {r['recall']*100:>9.2f}% "
              f"{r['f1-score']*100:>9.2f}% {int(r['support']):>10}")
    print(DIVIDER + "\n")


def save_confusion_matrix(metrics: dict, out_dir: Path):
    """Render and save the confusion matrix as a colour-coded PNG."""
    cm     = metrics["confusion_matrix"]
    labels = metrics["labels"]

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(labels) * 1.4 + 2),
                                             max(8, len(labels) * 1.2)))

    # ── Raw counts ────────────────────────────────────────────────────────────
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix — Raw Counts", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Predicted Label", fontsize=11)
    axes[0].set_ylabel("True Label", fontsize=11)
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].tick_params(axis="y", rotation=0)

    # ── Normalised (row-wise recall) ──────────────────────────────────────────
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1          # avoid divide by zero
    cm_norm /= row_sums

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Greens",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, vmin=0, vmax=1, ax=axes[1],
    )
    axes[1].set_title("Confusion Matrix — Normalised (Recall %)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Predicted Label", fontsize=11)
    axes[1].set_ylabel("True Label", fontsize=11)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="y", rotation=0)

    plt.tight_layout()
    path = out_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Confusion matrix saved -> %s", path)


def save_metrics_bar_chart(metrics: dict, out_dir: Path):
    """Bar chart comparing Accuracy / Precision / Recall / F1 / FAR / FRR."""
    labels = ["Accuracy", "Precision", "Recall", "F1-Score", "FAR", "FRR"]
    values = [
        metrics["accuracy"],
        metrics["precision_macro"],
        metrics["recall_macro"],
        metrics["f1_macro"],
        metrics["FAR"],
        metrics["FRR"],
    ]
    colours = ["#2ecc71", "#3498db", "#9b59b6", "#1abc9c", "#e74c3c", "#e67e22"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, [v * 100 for v in values], color=colours,
                  edgecolor="white", linewidth=1.2, width=0.55)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.0,
                f"{val*100:.2f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylim(0, 115)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Face Recognition Model — Evaluation Metrics", fontsize=14, fontweight="bold")
    ax.axhline(y=100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#ffffff")
    ax.grid(axis="y", alpha=0.4, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = out_dir / "metrics_bar_chart.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Metrics bar chart saved -> %s", path)


def save_timing_histogram(df: pd.DataFrame, out_dir: Path):
    """Histogram of per-image total recognition times."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    axes[0].hist(df["total_time_ms"], bins=20, color="#3498db",
                 edgecolor="white", linewidth=0.8)
    axes[0].axvline(df["total_time_ms"].mean(), color="red",
                    linestyle="--", linewidth=1.5, label=f"Mean = {df['total_time_ms'].mean():.1f} ms")
    axes[0].set_xlabel("Total Recognition Time (ms)", fontsize=11)
    axes[0].set_ylabel("Image Count", fontsize=11)
    axes[0].set_title("Recognition Time Distribution", fontsize=13, fontweight="bold")
    axes[0].legend()
    axes[0].set_facecolor("#f8f9fa")

    # Stacked bar: detection vs inference
    mean_det = df["detection_time_ms"].mean()
    mean_inf = df["inference_time_ms"].mean()
    axes[1].bar(["Avg Time"], [mean_det], label=f"Detection  ({mean_det:.1f} ms)",
                color="#e67e22", width=0.4)
    axes[1].bar(["Avg Time"], [mean_inf], bottom=[mean_det],
                label=f"Inference  ({mean_inf:.1f} ms)", color="#2ecc71", width=0.4)
    axes[1].set_ylabel("Time (ms)", fontsize=11)
    axes[1].set_title(f"Time Breakdown\nFPS ≈ {1000/(mean_det+mean_inf):.1f}",
                      fontsize=13, fontweight="bold")
    axes[1].legend()
    axes[1].set_facecolor("#f8f9fa")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.4, linestyle="--")

    plt.tight_layout()
    path = out_dir / "timing_histogram.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Timing histogram saved -> %s", path)


def save_per_class_chart(metrics: dict, out_dir: Path):
    """Per-student Precision / Recall / F1 grouped bar chart."""
    report = metrics["per_class_report"]
    labels = [l for l in metrics["labels"] if l in report]

    prec = [report[l]["precision"] * 100 for l in labels]
    rec  = [report[l]["recall"]    * 100 for l in labels]
    f1   = [report[l]["f1-score"]  * 100 for l in labels]

    x   = np.arange(len(labels))
    w   = 0.26

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.2 + 2), 6))
    ax.bar(x - w, prec, w, label="Precision", color="#3498db", edgecolor="white")
    ax.bar(x,     rec,  w, label="Recall",    color="#2ecc71", edgecolor="white")
    ax.bar(x + w, f1,   w, label="F1-Score",  color="#9b59b6", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=10)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_title("Per-Student  Precision / Recall / F1-Score", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_facecolor("#f8f9fa")
    ax.grid(axis="y", alpha=0.4, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = out_dir / "per_class_metrics.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Per-class chart saved -> %s", path)


def save_csv_report(df: pd.DataFrame, metrics: dict, out_dir: Path):
    """Save the per-image prediction table + a summary sheet to CSV."""
    # Per-image results (drop raw encoding — not human-readable)
    results_df = df.drop(columns=["encoding"], errors="ignore")
    results_df.to_csv(out_dir / "eval_per_image.csv", index=False)

    # Summary metrics
    summary = {
        "Metric": [
            "Accuracy (%)", "Precision-Macro (%)", "Recall-Macro (%)", "F1-Macro (%)",
            "FAR (%)", "FRR (%)", "EER-approx (%)",
            "TP", "TN", "FP", "FN",
            "Avg Detection (ms)", "Avg Inference (ms)", "Avg Total (ms)",
            "Min Total (ms)", "Max Total (ms)", "FPS",
        ],
        "Value": [
            round(metrics["accuracy"]*100, 4),
            round(metrics["precision_macro"]*100, 4),
            round(metrics["recall_macro"]*100, 4),
            round(metrics["f1_macro"]*100, 4),
            round(metrics["FAR"]*100, 4),
            round(metrics["FRR"]*100, 4),
            round(metrics["EER_approx"]*100, 4),
            metrics["TP"], metrics["TN"], metrics["FP"], metrics["FN"],
            round(metrics["avg_detection_ms"], 3),
            round(metrics["avg_inference_ms"], 3),
            round(metrics["avg_total_ms"], 3),
            round(metrics["min_total_ms"], 3),
            round(metrics["max_total_ms"], 3),
            round(metrics["fps"], 2),
        ],
    }
    pd.DataFrame(summary).to_csv(out_dir / "eval_summary.csv", index=False)
    log.info("CSV reports saved to %s", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6 — Tolerance sweep (optional: find optimal threshold)
# ─────────────────────────────────────────────────────────────────────────────

def tolerance_sweep(test_df: pd.DataFrame,
                    known_encodings: list,
                    known_ids: list,
                    out_dir: Path,
                    steps: int = 15):
    """
    Evaluate the model at multiple tolerance thresholds and plot how
    Accuracy, FAR, and FRR change.  Helps choose the optimal tolerance.
    """
    log.info("Running tolerance sweep (%d steps)…", steps)
    thresholds  = np.linspace(0.30, 0.75, steps)
    accs, fars, frrs, f1s = [], [], [], []

    known_student_ids = list(set(known_ids))

    for tol in thresholds:
        res = predict(test_df, known_encodings, known_ids, tol)
        m   = compute_metrics(res, known_student_ids)
        accs.append(m["accuracy"] * 100)
        fars.append(m["FAR"] * 100)
        frrs.append(m["FRR"] * 100)
        f1s.append(m["f1_macro"] * 100)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, accs, "g-o",  linewidth=2, markersize=5, label="Accuracy")
    ax.plot(thresholds, f1s,  "b-s",  linewidth=2, markersize=5, label="F1-Score")
    ax.plot(thresholds, fars, "r--^", linewidth=2, markersize=5, label="FAR")
    ax.plot(thresholds, frrs, "m--v", linewidth=2, markersize=5, label="FRR")

    # Mark approx EER crossing point
    eer_idx = np.argmin(np.abs(np.array(fars) - np.array(frrs)))
    ax.axvline(thresholds[eer_idx], color="gray", linestyle=":",
               label=f"EER ≈ tol={thresholds[eer_idx]:.2f}")

    ax.set_xlabel("Tolerance Threshold", fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Tolerance Sweep: Accuracy / F1 / FAR / FRR vs Threshold",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.4, linestyle="--")
    ax.set_facecolor("#f8f9fa")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = out_dir / "tolerance_sweep.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Tolerance sweep chart saved -> %s", path)
    log.info("Suggested optimal tolerance (near EER): %.2f", thresholds[eer_idx])



# ─────────────────────────────────────────────────────────────────────────────
#  GROUP MODE — evaluate directly on classroom / group photos
# ─────────────────────────────────────────────────────────────────────────────

def run_group_mode(
    image_dir: Path,
    known_encodings: list,
    known_ids: list,
    tolerance: float,
    model: str,
    upsample: int,
    out_dir: Path,
):
    """
    Scan a flat folder of group photos (no per-student sub-folders needed).
    Detects EVERY face in each image, identifies them, and reports:
      - Which students were recognised in each photo
      - Per-student confidence scores and timing
      - Recognition summary table + CSV
      - Metrics vs enrolled student list (who was found / who was missing)

    Ground-truth here = "a student should be recognisable if they are enrolled".
    FAR and FRR are computed relative to the full enrolled roster.

    Parameters
    ----------
    upsample : int
        How many times to upsample the image before HOG detection.
        0 = no upsampling (fast, misses small/distant faces)
        1 = 1x upsampling (finds smaller faces, recommended for classrooms)
        2 = 2x upsampling (finds very small faces, slower)
    """
    image_files = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
        and "_result" not in p.stem          # skip annotated output images
    ])

    if not image_files:
        # Also try sub-folders one level deep (user may have one folder inside)
        for sub in image_dir.iterdir():
            if sub.is_dir():
                image_files += sorted([
                    p for p in sub.iterdir()
                    if p.suffix.lower() in IMAGE_EXTS
                    and "_result" not in p.stem
                ])

    if not image_files:
        log.error(
            "No usable images found in %s.\n"
            "  - Make sure images are .jpg / .png\n"
            "  - Do not use _result annotated images as test input\n"
            "  - Original classroom photos work best", image_dir
        )
        sys.exit(1)

    log.info("Group mode: found %d image(s) in %s (upsample=%d)",
             len(image_files), image_dir, upsample)

    enrolled_ids  = set(known_ids)
    all_rows      = []
    per_image_log = []

    for img_path in image_files:
        t0  = time.perf_counter()
        img = face_recognition.load_image_file(str(img_path))

        # Optionally scale up so HOG finds small/distant faces
        if upsample > 0:
            h, w = img.shape[:2]
            scale = 2 ** upsample
            img_up = cv2.resize(
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                (w * scale, h * scale),
                interpolation=cv2.INTER_LINEAR,
            )
            img_up = cv2.cvtColor(img_up, cv2.COLOR_BGR2RGB)
        else:
            img_up = img

        locs = face_recognition.face_locations(img_up, model=model)
        encs = face_recognition.face_encodings(img_up, locs)
        det_ms = (time.perf_counter() - t0) * 1000

        if not encs:
            log.warning("  No faces detected in %s — try --upsample 1 or --upsample 2",
                        img_path.name)
            continue

        found_ids   = []
        face_detail = []

        for enc in encs:
            t1   = time.perf_counter()
            dists   = face_recognition.face_distance(known_encodings, enc)
            best_i  = int(np.argmin(dists))
            best_d  = float(dists[best_i])
            inf_ms  = (time.perf_counter() - t1) * 1000

            if best_d <= tolerance:
                sid  = known_ids[best_i]
                conf = round((1.0 - best_d) * 100, 2)
            else:
                sid  = "unknown"
                conf = 0.0

            found_ids.append(sid)
            face_detail.append({
                "image"            : img_path.name,
                "predicted_id"     : sid,
                "confidence_pct"   : conf,
                "match_distance"   : round(best_d, 4),
                "det_time_ms"      : round(det_ms / max(len(encs), 1), 2),
                "inf_time_ms"      : round(inf_ms, 2),
                "total_time_ms"    : round(det_ms / max(len(encs), 1) + inf_ms, 2),
            })
            all_rows.append(face_detail[-1])

        recognised   = [s for s in found_ids if s != "unknown"]
        unrecognised = found_ids.count("unknown")
        per_image_log.append({
            "image"        : img_path.name,
            "faces_found"  : len(encs),
            "recognised"   : len(recognised),
            "unknown"      : unrecognised,
            "student_ids"  : ", ".join(sorted(set(recognised))),
        })
        log.info("  %s -> %d face(s): %s + %d unknown",
                 img_path.name, len(encs),
                 sorted(set(recognised)) or "none", unrecognised)

    if not all_rows:
        log.error("No faces were detected in any image. Try --upsample 1")
        sys.exit(1)

    results_df   = pd.DataFrame(all_rows)
    per_image_df = pd.DataFrame(per_image_log)

    # ── Roster-level metrics ──────────────────────────────────────────────────
    # Across all images, which enrolled students were ever recognised?
    recognised_ids = set(r["predicted_id"] for r in all_rows if r["predicted_id"] != "unknown")
    missing_ids    = enrolled_ids - recognised_ids
    unknown_count  = sum(1 for r in all_rows if r["predicted_id"] == "unknown")
    total_faces    = len(all_rows)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  GROUP MODE EVALUATION REPORT")
    print(f"  Images processed : {len(per_image_log)}")
    print(f"  Total faces found: {total_faces}")
    print(f"  Recognised       : {total_faces - unknown_count}  "
          f"({(total_faces - unknown_count)/max(total_faces,1)*100:.1f}%)")
    print(f"  Unknown          : {unknown_count}  "
          f"({unknown_count/max(total_faces,1)*100:.1f}%)")
    print(f"  Enrolled students seen : {len(recognised_ids)}/{len(enrolled_ids)}")
    print(f"  Avg confidence   : {results_df['confidence_pct'].mean():.1f}%")
    print(f"  Avg total time   : {results_df['total_time_ms'].mean():.2f} ms/face")
    print(f"  Equivalent FPS   : {1000/max(results_df['total_time_ms'].mean(),1):.1f}")

    if missing_ids:
        print(f"\n  Students NOT recognised in any image:")
        for sid in sorted(missing_ids):
            print(f"    - {sid}")
    print(f"{'='*65}\n")

    print("  Per-image breakdown:")
    print(per_image_df.to_string(index=False))
    print()

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / "group_mode_faces.csv",      index=False)
    per_image_df.to_csv(out_dir / "group_mode_per_image.csv", index=False)

    # Confidence distribution chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    conf_known = results_df[results_df["predicted_id"] != "unknown"]["confidence_pct"]
    if not conf_known.empty:
        axes[0].hist(conf_known, bins=15, color="#3498db", edgecolor="white", linewidth=0.8)
        axes[0].axvline(conf_known.mean(), color="red", linestyle="--",
                        linewidth=1.5, label=f"Mean = {conf_known.mean():.1f}%")
        axes[0].set_xlabel("Confidence (%)", fontsize=11)
        axes[0].set_ylabel("Face count", fontsize=11)
        axes[0].set_title("Confidence Distribution (recognised faces)", fontsize=12, fontweight="bold")
        axes[0].legend()
        axes[0].set_facecolor("#f8f9fa")

    # Student recognition bar
    sid_counts = results_df[results_df["predicted_id"] != "unknown"]["predicted_id"].value_counts()
    if not sid_counts.empty:
        axes[1].barh(sid_counts.index, sid_counts.values, color="#2ecc71", edgecolor="white")
        axes[1].set_xlabel("Times recognised", fontsize=11)
        axes[1].set_title("Recognitions per student", fontsize=12, fontweight="bold")
        axes[1].set_facecolor("#f8f9fa")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x" if ax == axes[1] else "y", alpha=0.4, linestyle="--")

    plt.tight_layout()
    chart_path = out_dir / "group_mode_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()

    log.info("Group mode outputs saved to %s", out_dir)
    log.info("  group_mode_faces.csv    - per-face detail")
    log.info("  group_mode_per_image.csv - per-image summary")
    log.info("  group_mode_chart.png     - confidence + recognition chart")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the face recognition attendance model",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--test-dir", default=None,
        help="Labelled test folder (default: reuse students/ with leave-one-out style).\n"
             "Layout: test_images/<student_id>/*.jpg",
    )
    parser.add_argument("--tolerance", type=float, default=0.5,
                        help="Match distance threshold (default: 0.5)")
    parser.add_argument("--model",     default="hog", choices=["hog", "cnn"],
                        help="Detection backend: hog (CPU) or cnn (GPU)")
    parser.add_argument("--out-dir",   default="eval_results",
                        help="Output folder for plots and CSVs")
    parser.add_argument("--sweep",     action="store_true",
                        help="Run tolerance sweep to find optimal threshold")
    parser.add_argument("--group-mode", action="store_true",
                        help="Group photo mode: flat folder of classroom images, no sub-folders needed")
    parser.add_argument("--upsample",  type=int, default=1,
                        help="Upsample count for HOG in group mode (0=off, 1=recommended, 2=max)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load known encodings ───────────────────────────────────────────────
    known_encodings, known_ids = load_known_encodings()
    known_student_ids = list(set(known_ids))

    test_dir = Path(args.test_dir) if args.test_dir else STUDENTS_DIR
    if not test_dir.exists():
        log.error("Test directory not found: %s", test_dir)
        sys.exit(1)

    # ── GROUP MODE: flat folder of classroom photos ───────────────────────────
    if args.group_mode:
        run_group_mode(
            image_dir      = test_dir,
            known_encodings= known_encodings,
            known_ids      = known_ids,
            tolerance      = args.tolerance,
            model          = args.model,
            upsample       = args.upsample,
            out_dir        = out_dir,
        )
        return

    # ── STANDARD MODE: labelled per-student sub-folders ───────────────────────
    # Auto-detect if user accidentally ran without --group-mode
    subfolders = [d for d in test_dir.iterdir() if d.is_dir()]
    flat_images = [p for p in test_dir.iterdir()
                   if p.suffix.lower() in IMAGE_EXTS and "_result" not in p.stem]

    if flat_images and not subfolders:
        log.warning("Detected flat image folder (no sub-folders).")
        log.warning("Automatically switching to --group-mode.")
        log.warning("To use standard mode, organise images as: test_images/<student_id>/*.jpg")
        run_group_mode(
            image_dir      = test_dir,
            known_encodings= known_encodings,
            known_ids      = known_ids,
            tolerance      = args.tolerance,
            model          = args.model,
            upsample       = args.upsample,
            out_dir        = out_dir,
        )
        return

    # ── 2. Build test dataset ─────────────────────────────────────────────────
    test_df = build_test_dataset(test_dir, model=args.model)
    if test_df.empty:
        log.error("No valid test images found.")
        log.error("If your folder contains group/classroom photos without sub-folders, use:")
        log.error("  python evaluation.py --test-dir test_images/ --group-mode")
        sys.exit(1)

    # ── 3. Run predictions ────────────────────────────────────────────────────
    log.info("Running predictions (tolerance=%.2f, model=%s)…", args.tolerance, args.model)
    result_df = predict(test_df, known_encodings, known_ids, args.tolerance)

    # ── 4. Compute metrics ────────────────────────────────────────────────────
    metrics = compute_metrics(result_df, known_student_ids)

    # ── 5. Print to terminal ──────────────────────────────────────────────────
    print_summary(metrics, args.tolerance, args.model, len(result_df))

    # ── 6. Save plots and CSVs ────────────────────────────────────────────────
    save_confusion_matrix(metrics, out_dir)
    save_metrics_bar_chart(metrics, out_dir)
    save_timing_histogram(result_df, out_dir)
    save_per_class_chart(metrics, out_dir)
    save_csv_report(result_df, metrics, out_dir)
    # Open-set ROC / PR + exact EER
    save_roc_pr(result_df, out_dir)

    # ── 7. Optional tolerance sweep ───────────────────────────────────────────
    if args.sweep:
        tolerance_sweep(test_df, known_encodings, known_ids, out_dir)

    print(f"\n  All evaluation outputs saved to:  {out_dir.resolve()}")
    print(f"  Files generated:")
    for f in sorted(out_dir.iterdir()):
        print(f"    • {f.name}")
    print()


if __name__ == "__main__":
    main()