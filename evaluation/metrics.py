"""
Evaluation metrics with 1,000-replicate bootstrap confidence intervals.

Metrics reported in the paper (Section 2.5.2):
    - AUROC          — threshold-free ranking
    - AP             — Average Precision
    - ACC@0.5        — accuracy at threshold t=0.5
    - F1@0.5         — F1 at t=0.5
    - Precision@0.5
    - Recall@0.5
    - Prec@Rec=0.90  — clinically motivated operating point

Bootstrap (Section 2.5.4):
    - n=1,000 example-level replicates
    - 95% percentile interval (2.5th–97.5th)
    - Single-class replicates excluded from threshold-free metrics

Usage:
    python evaluation/metrics.py \
        --predictions_csv results/fedavg_resnet50/predictions.csv \
        --output_dir      results/fedavg_resnet50/

Reference:
    Ali et al., Front. Digit. Health 8:1715858 (2026)
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, f1_score, precision_score,
    recall_score, precision_recall_curve,
)


# --------------------------------------------------------------------------- #
#  Core metric computation                                                     #
# --------------------------------------------------------------------------- #

def precision_at_recall(y_true, y_prob, target_recall=0.90):
    """
    Precision@Recall=0.90 via PR curve interpolation (Section 2.5.2).
    Returns None if recall=0.90 cannot be attained.
    """
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    # prec/rec are in decreasing recall order — find first where rec >= target
    valid = rec >= target_recall
    if not valid.any():
        return None
    return float(prec[valid][-1])


def compute_all_metrics(y_true, y_prob, threshold=0.5):
    """
    Compute all paper metrics for one set of predictions.

    Args:
        y_true:    array-like binary labels
        y_prob:    array-like predicted malignant probabilities
        threshold: decision threshold (default 0.5)

    Returns:
        dict of metric values (None where undefined)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    n_classes = len(np.unique(y_true))

    auroc = roc_auc_score(y_true, y_prob) if n_classes > 1 else None
    ap    = average_precision_score(y_true, y_prob) if n_classes > 1 else None

    return {
        "auroc":       round(auroc, 4) if auroc is not None else None,
        "ap":          round(ap, 4)    if ap    is not None else None,
        "acc":         round(accuracy_score(y_true, y_pred), 4),
        "f1":          round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision":   round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":      round(recall_score(y_true, y_pred, zero_division=0), 4),
        "prec_at_rec90": precision_at_recall(y_true, y_prob, 0.90),
        "n":           len(y_true),
    }


# --------------------------------------------------------------------------- #
#  Bootstrap confidence intervals                                              #
# --------------------------------------------------------------------------- #

def bootstrap_metrics(y_true, y_prob, n_replicates=1000,
                      threshold=0.5, random_state=42):
    """
    1,000 example-level bootstrap replicates → mean ± SD and 95% CI.

    Single-class replicates are excluded from threshold-free metrics
    (AUROC, AP) per Section 2.5.4.

    Returns:
        summary: dict {metric: {"mean": ..., "sd": ..., "ci_low": ..., "ci_high": ...}}
    """
    rng    = np.random.default_rng(random_state)
    n      = len(y_true)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    boot_results = {m: [] for m in [
        "auroc", "ap", "acc", "f1", "precision", "recall", "prec_at_rec90"
    ]}

    for _ in range(n_replicates):
        idx       = rng.integers(0, n, size=n)
        bt_true   = y_true[idx]
        bt_prob   = y_prob[idx]
        bt_metrics = compute_all_metrics(bt_true, bt_prob, threshold)

        for m, v in bt_metrics.items():
            if m == "n":
                continue
            # Exclude single-class replicates from threshold-free metrics
            if m in ("auroc", "ap") and len(np.unique(bt_true)) < 2:
                continue
            if v is not None:
                boot_results[m].append(v)

    summary = {}
    for m, vals in boot_results.items():
        if not vals:
            summary[m] = {"mean": None, "sd": None, "ci_low": None, "ci_high": None}
            continue
        arr            = np.array(vals)
        summary[m]     = {
            "mean":    round(float(arr.mean()), 4),
            "sd":      round(float(arr.std()),  4),
            "ci_low":  round(float(np.percentile(arr, 2.5)),  4),
            "ci_high": round(float(np.percentile(arr, 97.5)), 4),
        }
    return summary


# --------------------------------------------------------------------------- #
#  Per-domain evaluation (Tables 3, 5, 7)                                     #
# --------------------------------------------------------------------------- #

def evaluate_per_domain(predictions_csv, output_dir,
                        n_bootstrap=1000, threshold=0.5):
    """
    Evaluate on combined test set and each domain separately.
    Reproduces the reporting format of Tables 3, 5, 6, 7.
    """
    df         = pd.read_csv(predictions_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for domain in ["all", "cbis", "vindr"]:
        if domain == "all":
            sub = df
            label = "Combined"
        else:
            sub   = df[df["dataset"] == domain]
            label = "CBIS-DDSM" if domain == "cbis" else "VinDr-Mammo"

        if len(sub) == 0:
            continue

        y_true = sub["label"].values
        y_prob = sub["proba"].values

        # Point estimates
        point = compute_all_metrics(y_true, y_prob, threshold)

        # Bootstrap CIs
        boot  = bootstrap_metrics(y_true, y_prob, n_bootstrap, threshold)

        print(f"\n  [{label}]  n={len(sub)}")
        for m in ["auroc", "ap", "acc", "f1", "precision", "recall", "prec_at_rec90"]:
            b = boot[m]
            if b["mean"] is None:
                continue
            print(f"    {m:<20}: {b['mean']:.4f} ± {b['sd']:.4f}  "
                  f"95% CI [{b['ci_low']:.4f}, {b['ci_high']:.4f}]")

        row = {"domain": label}
        for m, b in boot.items():
            if b["mean"] is not None:
                row[f"{m}_mean"] = b["mean"]
                row[f"{m}_sd"]   = b["sd"]
                row[f"{m}_ci95"] = f"[{b['ci_low']:.4f}, {b['ci_high']:.4f}]"
        rows.append(row)

    results_df = pd.DataFrame(rows)
    out_path   = output_dir / "bootstrap_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved: {out_path}")
    return results_df


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="Bootstrap evaluation (Tables 3-7)")
    parser.add_argument("--predictions_csv", required=True)
    parser.add_argument("--output_dir",      default="results/eval")
    parser.add_argument("--n_bootstrap",     type=int, default=1000)
    parser.add_argument("--threshold",       type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_per_domain(
        args.predictions_csv, args.output_dir,
        args.n_bootstrap, args.threshold
    )
