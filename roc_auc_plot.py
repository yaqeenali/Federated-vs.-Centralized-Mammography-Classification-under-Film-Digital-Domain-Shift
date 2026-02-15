#!/usr/bin/env python3
import argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from tqdm import tqdm

# ---------- Pretty labels & style ----------
ARCH_PRINT = {'resnet':'ResNet-50','swin':'Swin'}
STRAT_PRINT = {
    'cl':'CL','fedavg':'FedAvg','fedprox':'FedProx','fedscaffold':'SCAFFOLD','fedbn':'FedBN',
    'cbis':'CBIS(Local)','vindr':'VinDr(Local)'
}
TEST_PRINT = {'vindr_test':'VinDr','cbis_test':'CBIS','vindr_cbis':'VinDr+CBIS'}

# Colors per strategy (color-blind friendly)
STRAT_COLOR = {
    'CL': '#000000',           # black
    'FedAvg': '#0072B2',       # blue
    'FedProx': '#D55E00',      # vermillion (this was FedBN before; keeping your newer mapping)
    'SCAFFOLD': '#009E73',     # bluish green
    'FedBN': '#CC79A7',        # reddish purple (swapped to keep distinct from FedProx)
    'CBIS(Local)': '#CC79A7',  # reddish purple
    'VinDr(Local)': '#56B4E9'  # sky blue
}

# Linestyles per backbone
ARCH_LS = {'ResNet-50':'-', 'Swin':'--'}

def _safe_series(x):
    x = np.asarray(x).astype(float)
    return x

def _load_predictions(xlsx_path: Path):
    # Expect columns: dataset, patient_id (optional), path, y_true, y_pred, y_prob
    try:
        df = pd.read_excel(xlsx_path)
    except Exception:
        # fallback if saved as CSV by user
        csv_path = xlsx_path.with_suffix('.csv')
        df = pd.read_csv(csv_path) if csv_path.exists() else None
    return df

def _per_seed_curves(df):
    """
    Return dict with curves: {'roc':[(fpr,tpr,auc,seed_tag)...], 'pr':[(rec,prec,ap,seed_tag)...]}
    Seeds are not labeled inside the excel; we pass a seed_tag externally when calling.
    Here, we just compute curves from df (y_true, y_prob).
    """
    y_true = _safe_series(df['y_true'])
    y_prob = _safe_series(df['y_prob'])
    if len(np.unique(y_true)) < 2:
        return None  # cannot compute ROC/AP
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    rec, prec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    return (fpr, tpr, roc_auc), (rec, prec, ap)
def _compute_prevalence_from_rows(rows):
    """Pool y_true across given rows (each row points to a predictions file) and compute prevalence."""
    y_all = []
    for _, r in rows.iterrows():
        df_pred = _load_predictions(Path(r['predictions_xlsx']))
        if df_pred is None or 'y_true' not in df_pred.columns:
            continue
        y_all.append(_safe_series(df_pred['y_true']))
    if not y_all:
        return None
    y_cat = np.concatenate(y_all)
    # Prevalence = positive fraction
    return float(np.mean(y_cat))

def _interp_mean_curve(xs, ys_list, x_grid):
    """
    Given multiple curves (xs_i, ys_i), interpolate each onto x_grid and return mean and std.
    Requires xs to be monotonically increasing (we assume roc fpr or pr recall).
    """
    Y = []
    for xi, yi in zip(xs, ys_list):
        # ensure sorting (just in case)
        order = np.argsort(xi)
        xi = np.asarray(xi)[order]
        yi = np.asarray(yi)[order]
        # remove potential duplicates to avoid issues in interpolation
        uniq, idx = np.unique(xi, return_index=True)
        yi = yi[idx]
        xi = uniq
        try:
            y_interp = np.interp(x_grid, xi, yi)
            Y.append(y_interp)
        except Exception:
            continue
    if not Y:
        return None, None
    Y = np.vstack(Y)
    return Y.mean(axis=0), Y.std(axis=0)
def _collect_runs(master_csv: Path, tests_filter=None, archs_filter=None, strats_filter=None):
    m = pd.read_csv(master_csv)
    # normalize labels used in plotting
    m['arch_pr'] = m['arch'].map(ARCH_PRINT).fillna(m['arch'])
    m['strat_pr'] = m['strategy'].map(STRAT_PRINT).fillna(m['strategy'])
    m['test_pr']  = m['test_name'].map(TEST_PRINT).fillna(m['test_name'])

    if tests_filter:
        m = m[m['test_name'].isin(tests_filter)]
    if archs_filter:
        m = m[m['arch'].isin(archs_filter)]
    if strats_filter:
        m = m[m['strategy'].isin(strats_filter)]

    # Ensure predictions_xlsx exists
    m = m[pd.notna(m['predictions_xlsx'])]
    m = m[m['predictions_xlsx'].apply(lambda p: Path(p).exists())]
    return m

def _plot_one_dataset_old(df_sub, dataset_label, out_dir: Path, dpi=250, legend_loc='lower right'):
    """
    df_sub: rows for a single dataset (VinDr / CBIS / VinDr+CBIS)
    Creates a single figure with 2 subplots (ROC, PR), overlaying (arch × strategy × seeds).
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=dpi)
    ax_roc, ax_pr = axes

    # For legend: track per (arch,strat) mean AUC/AP across seeds
    legend_stats = {}

    # --- Compute dataset-level prevalence for PR baseline ---
    prevalence = _compute_prevalence_from_rows(df_sub)
    # Fallback if unavailable:
    if prevalence is None or not np.isfinite(prevalence):
        prevalence = 0.5  # fallback, but usually we should have valid prevalence

    # Grids for mean curves
    roc_x_grid = np.linspace(0, 1, 400)   # FPR grid
    pr_x_grid  = np.linspace(0, 1, 400)   # Recall grid

    groups = df_sub.groupby(['arch_pr','strat_pr'])
    for (arch_pr, strat_pr), g in groups:
        color = STRAT_COLOR.get(strat_pr, None)
        ls = ARCH_LS.get(arch_pr, '-.')

        auc_vals, ap_vals = [], []
        # For mean curves
        roc_xs, roc_ys = [], []
        pr_xs, pr_ys   = [], []

        # plot per-seed thin curves
        for _, r in g.iterrows():
            df_pred = _load_predictions(Path(r['predictions_xlsx']))
            if df_pred is None or 'y_true' not in df_pred.columns or 'y_prob' not in df_pred.columns:
                continue
            curves = _per_seed_curves(df_pred)
            if curves is None:
                continue
            (fpr, tpr, roc_auc), (rec, prec, ap) = curves
            auc_vals.append(roc_auc); ap_vals.append(ap)

            # Thicker, slightly less transparent per-seed lines
            ax_roc.plot(fpr, tpr, color=color, ls=ls, alpha=0.25, lw=1.2)
            ax_pr.plot(rec, prec, color=color, ls=ls, alpha=0.25, lw=1.2)

            # collect for mean curve
            roc_xs.append(fpr); roc_ys.append(tpr)
            pr_xs.append(rec);  pr_ys.append(prec)

        if len(auc_vals) == 0:
            continue

        # Store average stats for legend
        legend_stats[(arch_pr, strat_pr)] = (float(np.mean(auc_vals)), float(np.mean(ap_vals)))

        # --- Mean curves ---
        roc_mean, roc_std = _interp_mean_curve(roc_xs, roc_ys, roc_x_grid)
        pr_mean,  pr_std  = _interp_mean_curve(pr_xs,  pr_ys,  pr_x_grid)


        if roc_mean is not None:
            ax_roc.plot(
                roc_x_grid, roc_mean,
                color=color, ls=ls, lw=2.8, alpha=0.95,
                markevery=80, marker='o', markersize=3
            )
            if roc_std is not None:
                ax_roc.fill_between(
                    roc_x_grid,
                    np.maximum(0, roc_mean - roc_std),
                    np.minimum(1, roc_mean + roc_std),
                    color=color, alpha=0.08, linewidth=0
        )


        if pr_mean is not None:
            ax_pr.plot(pr_x_grid, pr_mean, color=color, ls=ls, lw=2.8, alpha=0.95)
            if pr_std is not None:
                ax_pr.fill_between(pr_x_grid,
                                   np.maximum(0, pr_mean - pr_std),
                                   np.minimum(1, pr_mean + pr_std),
                                   color=color, alpha=0.08, linewidth=0)

    # Reference diagonals / baselines
    ax_roc.plot([0,1],[0,1],'k--',lw=0.9,alpha=0.6)
    # Correct PR baseline at prevalence
    ax_pr.hlines(y=prevalence, xmin=0, xmax=1, colors='k', linestyles=':', lw=0.9, alpha=0.6)

    # Build legend items as bold lines (no data) showing styles, with mean AUC/AP
    legend_lines = []
    legend_labels = []
    for (arch_pr, strat_pr), (auc_m, ap_m) in sorted(legend_stats.items(), key=lambda x: (x[0][0], x[0][1])):
        color = STRAT_COLOR.get(strat_pr, None)
        ls = ARCH_LS.get(arch_pr, '-.')
        ln, = ax_roc.plot([], [], color=color, ls=ls, lw=2.8)
        legend_lines.append(ln)
        legend_labels.append(f"{arch_pr} / {strat_pr}  (AUC={auc_m:.3f}, AP={ap_m:.3f})")

    # Titles, labels
    ax_roc.set_title(f"ROC — {dataset_label}")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_pr.set_title(f"Precision–Recall — {dataset_label}")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")

    # Consistent limits
    ax_roc.set_xlim(0,1); ax_roc.set_ylim(0,1)
    ax_pr.set_xlim(0,1);  ax_pr.set_ylim(0,1)

    # Light grids for readability
    for ax in (ax_roc, ax_pr):
        ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.35)

    if legend_lines:
        fig.legend(legend_lines, legend_labels, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.subplots_adjust(bottom=0.22)
    fig.tight_layout(rect=[0,0.05,1,1])
    out_png = out_dir / f"panel_rocpr_{dataset_label.replace('+','_')}.png"
    out_pdf = out_dir / f"panel_rocpr_{dataset_label.replace('+','_')}.pdf"
    fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
    fig.savefig(out_pdf, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"[WRITE] {out_png} / {out_pdf}")
def _plot_one_dataset(df_sub, dataset_label, out_dir: Path, dpi=250, legend_loc='lower right'):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=dpi)
    ax_roc, ax_pr = axes

    legend_stats = {}

    # Compute prevalence baseline for PR
    prevalence = _compute_prevalence_from_rows(df_sub)
    if prevalence is None or not np.isfinite(prevalence):
        prevalence = 0.5

    # Grids for mean curves
    roc_x_grid = np.linspace(0, 1, 400)
    pr_x_grid  = np.linspace(0, 1, 400)

    groups = df_sub.groupby(['arch_pr', 'strat_pr'])
    for (arch_pr, strat_pr), g in groups:
        color = STRAT_COLOR.get(strat_pr, None)
        ls = ARCH_LS.get(arch_pr, '-.')

        auc_vals, ap_vals = [], []
        roc_xs, roc_ys = [], []
        pr_xs, pr_ys = [], []

        # -------- collect all seed curves (no plotting here) --------
        for _, r in g.iterrows():
            df_pred = _load_predictions(Path(r['predictions_xlsx']))
            if df_pred is None or 'y_true' not in df_pred.columns or 'y_prob' not in df_pred.columns:
                continue

            curves = _per_seed_curves(df_pred)
            if curves is None:
                continue

            (fpr, tpr, roc_auc), (rec, prec, ap) = curves

            auc_vals.append(roc_auc)
            ap_vals.append(ap)

            roc_xs.append(fpr)
            roc_ys.append(tpr)
            pr_xs.append(rec)
            pr_ys.append(prec)

        if len(auc_vals) == 0:
            continue

        # store stats for legend
        legend_stats[(arch_pr, strat_pr)] = (float(np.mean(auc_vals)),
                                             float(np.mean(ap_vals)))

        # -------- compute mean curves --------
        roc_mean, roc_std = _interp_mean_curve(roc_xs, roc_ys, roc_x_grid)
        pr_mean,  pr_std  = _interp_mean_curve(pr_xs,  pr_ys,  pr_x_grid)

        # -------- plot ONLY mean curves (+ optional std) --------
        if roc_mean is not None:
            ax_roc.plot(
                roc_x_grid, roc_mean,
                color=color, ls=ls, lw=2.8, alpha=0.95,
                markevery=80, marker='o', markersize=3
            )
            if roc_std is not None:
                ax_roc.fill_between(
                    roc_x_grid,
                    np.maximum(0, roc_mean - roc_std),
                    np.minimum(1, roc_mean + roc_std),
                    color=color, alpha=0.08, linewidth=0
                )

        if pr_mean is not None:
            ax_pr.plot(
                pr_x_grid, pr_mean,
                color=color, ls=ls, lw=2.8, alpha=0.95
            )
            if pr_std is not None:
                ax_pr.fill_between(
                    pr_x_grid,
                    np.maximum(0, pr_mean - pr_std),
                    np.minimum(1, pr_mean + pr_std),
                    color=color, alpha=0.08, linewidth=0
                )

    # ===== reference lines =====
    ax_roc.plot([0,1],[0,1],'k--',lw=0.9,alpha=0.6)
    ax_pr.hlines(prevalence, 0, 1, colors='k', linestyles=':', lw=0.9, alpha=0.6)

    # ===== legend =====
    legend_lines, legend_labels = [], []
    for (arch_pr, strat_pr), (auc_m, ap_m) in sorted(legend_stats.items()):
        color = STRAT_COLOR.get(strat_pr)
        ls = ARCH_LS.get(arch_pr)
        ln, = ax_roc.plot([], [], color=color, ls=ls, lw=2.8)
        legend_lines.append(ln)
        legend_labels.append(f"{arch_pr} / {strat_pr}  (AUC={auc_m:.3f}, AP={ap_m:.3f})")

    # labels, titles
    ax_roc.set_title(f"ROC — {dataset_label}")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")

    ax_pr.set_title(f"Precision–Recall — {dataset_label}")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")

    # axis limits
    ax_roc.set_xlim(0,1); ax_roc.set_ylim(0,1)
    ax_pr.set_xlim(0,1);  ax_pr.set_ylim(0,1)

    # nice grid
    for ax in (ax_roc, ax_pr):
        ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.35)

    # legend at bottom
    if legend_lines:
        fig.legend(legend_lines, legend_labels, loc='lower center',
                   ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.12))

    fig.subplots_adjust(bottom=0.22)
    fig.tight_layout(rect=[0,0.05,1,1])

    out_png = out_dir / f"panel_rocpr_{dataset_label.replace('+','_')}.png"
    out_pdf = out_dir / f"panel_rocpr_{dataset_label.replace('+','_')}.pdf"
    fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
    fig.savefig(out_pdf, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"[WRITE] {out_png} / {out_pdf}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, default='results_test', help='Folder containing metrics_master.csv')
    ap.add_argument('--tests', nargs='*', default=['vindr_test','cbis_test','vindr_cbis'],
                    help='Which test_name panels to render')
    ap.add_argument('--archs', nargs='*', default=['resnet','swin'],
                    help='Architectures to include (folder names: resnet, swin)')
    ap.add_argument('--strategies', nargs='*', default=None,
                    help='If set, limit to these strategies (e.g., cl fedavg fedprox fedscaffold fedbn)')
    ap.add_argument('--outdir', type=str, default='results_test/fig_panels')
    ap.add_argument('--dpi', type=int, default=300)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.outdir); out_dir.mkdir(parents=True, exist_ok=True)

    master_csv = root / 'metrics_master_1.csv'
    if not master_csv.exists():
        raise FileNotFoundError(f"Not found: {master_csv}. Run your evaluation/auto-scan first.")

    df = _collect_runs(master_csv,
                       tests_filter=args.tests,
                       archs_filter=args.archs,
                       strats_filter=args.strategies)

    if df.empty:
        print("[INFO] No runs match filters; nothing to plot.")
        return

    for t in tqdm(sorted(df['test_name'].unique()), desc='Panels'):
        ds_label = TEST_PRINT.get(t, t)
        _plot_one_dataset(df[df['test_name']==t], ds_label, out_dir, dpi=args.dpi)

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    main()