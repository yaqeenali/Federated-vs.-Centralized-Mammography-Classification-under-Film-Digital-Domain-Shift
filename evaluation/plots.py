"""
ROC and Precision-Recall curve plots — reproduces Figures 3 and 4.

Figure 3: Swin V2-T on combined VinDr+CBIS test set (CL, FedAvg, FedProx)
Figure 4: ResNet-50 on combined VinDr+CBIS test set (CL, FedAvg, FedProx)

Usage:
    python evaluation/plots.py \
        --predictions_dir results/ \
        --output_dir      figures/
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score


# Method colours matching paper style
METHOD_STYLES = {
    "CL":       {"color": "#333333", "linestyle": "--", "linewidth": 2.0},
    "FedAvg":   {"color": "#E07B39", "linestyle": "-",  "linewidth": 1.8},
    "FedProx":  {"color": "#3A7DC9", "linestyle": "-",  "linewidth": 1.8},
    "SCAFFOLD": {"color": "#5BA85A", "linestyle": "-",  "linewidth": 1.5},
    "FedBN":    {"color": "#A855B5", "linestyle": "-",  "linewidth": 1.5},
    "Local":    {"color": "#999999", "linestyle": ":",  "linewidth": 1.5},
}


def plot_roc_pr(predictions_dict, backbone_name, output_dir, domain="combined"):
    """
    Plot ROC and PR curves for multiple methods on one backbone.
    Reproduces Figures 3 (Swin) and 4 (ResNet-50).

    Args:
        predictions_dict: {method_name: DataFrame with columns proba, label}
        backbone_name:    'ResNet-50' | 'Swin V2-T'
        output_dir:       where to save PNG
        domain:           test domain for subtitle
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_roc, ax_pr = axes

    for method, df in predictions_dict.items():
        style  = METHOD_STYLES.get(method, {"color": "black", "linestyle": "-",
                                            "linewidth": 1.5})
        y_true = df["label"].values
        y_prob = df["proba"].values

        if len(np.unique(y_true)) < 2:
            continue

        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc         = roc_auc_score(y_true, y_prob)
        ax_roc.plot(fpr, tpr, label=f"{method} (AUC={auc:.3f})", **style)

        # PR curve
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap            = average_precision_score(y_true, y_prob)
        ax_pr.plot(rec, prec, label=f"{method} (AP={ap:.3f})", **style)

    # ROC formatting
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4)
    ax_roc.set_xlabel("False Positive Rate", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate", fontsize=11)
    ax_roc.set_title(f"ROC — {domain.replace('_',' ').title()}", fontsize=11)
    ax_roc.legend(fontsize=8, loc="lower right")
    ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1.02)

    # PR formatting
    # Baseline precision = malignant prevalence
    if len(predictions_dict) > 0:
        first_df   = next(iter(predictions_dict.values()))
        prevalence = first_df["label"].mean()
        ax_pr.axhline(prevalence, color="gray", linestyle=":",
                      linewidth=0.8, label=f"Chance (prev={prevalence:.2f})")

    ax_pr.set_xlabel("Recall", fontsize=11)
    ax_pr.set_ylabel("Precision", fontsize=11)
    ax_pr.set_title(f"Precision–Recall — {domain.replace('_',' ').title()}", fontsize=11)
    ax_pr.legend(fontsize=8, loc="upper right")
    ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1.02)

    fig.suptitle(f"{backbone_name} — {domain}", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()

    backbone_tag = backbone_name.lower().replace("-", "").replace(" ", "_")
    out_path     = output_dir / f"roc_pr_{backbone_tag}_{domain}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_training_curves(log_csvs, output_dir):
    """
    Plot validation AUROC over FL rounds.
    Helps visualise convergence of each algorithm.

    Args:
        log_csvs: dict {method_name: path_to_training_log.csv}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    for method, csv_path in log_csvs.items():
        df     = pd.read_csv(csv_path)
        style  = METHOD_STYLES.get(method, {})
        rounds = df["round"].values
        aucs   = df["val_auroc"].values
        ax.plot(rounds, aucs, label=method, **style)

    ax.set_xlabel("FL Round", fontsize=11)
    ax.set_ylabel("Validation AUROC", fontsize=11)
    ax.set_title("Federated Learning Convergence", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0.4, 1.0)
    fig.tight_layout()

    out_path = output_dir / "training_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce Figures 3 and 4")
    parser.add_argument("--predictions_dir", required=True,
                        help="Directory containing {method}_predictions.csv files")
    parser.add_argument("--backbone",        default="ResNet-50",
                        help="ResNet-50 | Swin V2-T")
    parser.add_argument("--output_dir",      default="figures")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pred_dir = Path(args.predictions_dir)

    # Auto-discover prediction CSVs: {method}_predictions.csv
    preds = {}
    for csv_path in sorted(pred_dir.glob("*_predictions.csv")):
        method = csv_path.stem.replace("_predictions", "").replace("_", " ").title()
        df     = pd.read_csv(csv_path)
        preds[method] = df

    if not preds:
        print(f"No *_predictions.csv files found in {pred_dir}")
    else:
        print(f"Found predictions for: {list(preds.keys())}")
        plot_roc_pr(preds, args.backbone, args.output_dir, domain="VinDr+CBIS")
