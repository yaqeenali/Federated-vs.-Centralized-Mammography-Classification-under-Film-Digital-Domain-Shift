#!/usr/bin/env python3
"""
test_runner.py
Evaluate a saved model checkpoint on a provided test CSV (or multiple named test CSVs).

Expected CSV schema (configurable): dataset, img, label, roi_png_path
Default path column: roi_png_path
Default label column: label

Outputs:
  - <out>/<prefix>.json            : summary metrics (acc, f1, prec, rec, auc, ap)
  - <out>/roc_<prefix>.png        : ROC curve
  - <out>/pr_<prefix>.png         : PR curve
  - <out>/<prefix>_ci.txt         : bootstrap CIs for AUC/AP/ACC/PREC/REC/F1 and Precision@Recall=0.90
"""

import os, math, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pydicom
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tvm
import torchvision.transforms as T

from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    accuracy_score, f1_score, precision_score, recall_score
)
import matplotlib.pyplot as plt

# ------------------------ Dataset ------------------------
class MammogramDataset(Dataset):
    def __init__(self, csv_path, path_col='roi_png_path', label_col='label',
                 img_size=224, augment=False):
        self.df = pd.read_csv(csv_path)
        assert {path_col, label_col}.issubset(set(self.df.columns)), \
            f"CSV must contain columns: {path_col}, {label_col}"
        self.path_col = path_col
        self.label_col = label_col
        self.img_size = img_size
        self.augment = augment
        mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
        base = [T.Resize((img_size, img_size)), T.ToTensor(), T.Normalize(mean, std)]
        self.eval_tf  = T.Compose(base)

    def __len__(self): return len(self.df)

    def _read_image(self, path):
        ext = str(path).lower()
        try:
            if ext.endswith(".dcm"):
                ds = pydicom.dcmread(path)
                arr = ds.pixel_array.astype(np.float32)
                p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
                arr = (arr - p1) / (p99 - p1 + 1e-6)
                arr = np.clip(arr, 0, 1)
                img = np.stack([arr, arr, arr], axis=-1)
            else:
                with Image.open(path) as im:
                    img = np.array(im.convert("RGB"), dtype=np.float32) / 255.0
            return img
        except Exception as e_pil:
            # Fallback: OpenCV (tolerant + Unicode paths)
            try:
                data = np.fromfile(path, dtype=np.uint8)
                bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise RuntimeError("cv2.imdecode returned None")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                return rgb
            except Exception as e_cv:
                raise RuntimeError(f"Failed to read image '{path}'. PIL: {e_pil}; OpenCV: {e_cv}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row[self.path_col]
        img = self._read_image(path)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        x = Image.fromarray((img*255).astype(np.uint8))
        x = self.eval_tf(x)
        y = int(row[self.label_col])
        meta = {
            'dataset': row.get('dataset', 'NA'),
            'path': path
        }
        return x, y, meta

# ------------------------ Models ------------------------
def build_model(name="resnet50", num_classes=2, pretrained=False):
    """
    Build the same model variants as in training. Set pretrained=False when loading a fine-tuned checkpoint.
    """
    name = name.lower()
    if name=="resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    elif name=="swin_t":
        m = tvm.swin_v2_t(weights=tvm.Swin_V2_T_Weights.IMAGENET1K_V1 if pretrained else None)
        m.head = nn.Linear(m.head.in_features, num_classes)
        return m
    elif name=="convnext_b":
        m = tvm.convnext_base(weights=tvm.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unknown model {name}")

# ------------------------ Evaluation utils ------------------------
@torch.no_grad()
def evaluate(model, loader, device, return_probs=False):
    model.eval()
    y_true, y_prob, y_pred = [], [], []
    for xb, yb, _ in loader:
        xb = xb.to(device)
        logits = model(xb)
        prob1 = torch.softmax(logits, dim=1)[:,1].detach().cpu().numpy()
        y_prob.append(prob1)
        y_true.append(yb.numpy())
        y_pred.append(np.argmax(logits.detach().cpu().numpy(), axis=1))
    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred); y_prob = np.concatenate(y_prob)
    metrics = {
        'acc': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'prec': precision_score(y_true, y_pred, zero_division=0),
        'rec': recall_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true))>1 else float('nan'),
        'ap': average_precision_score(y_true, y_prob) if len(np.unique(y_true))>1 else float('nan')
    }
    if return_probs: return metrics, y_true, y_prob
    return metrics

def bootstrap_ci(
    y_true, y_prob, n_boot=1000, metric="auc", seed=123,
    threshold=0.5, strategy="fixed", target_recall=None
):
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    chosen_thr = None
    if metric in {"acc","prec","rec","f1"}:
        if strategy=="fixed":
            chosen_thr = float(threshold)
        else:
            # simple fallback: fixed 0.5 to keep parity with the training code’s write_all_cis
            chosen_thr = float(threshold)

    vals = []
    for _ in range(n_boot):
        ii = rng.choice(len(y_true), size=len(y_true), replace=True)
        yt = y_true[ii]; yp = y_prob[ii]
        if metric=="auc":
            if len(np.unique(yt))<2: continue
            vals.append(roc_auc_score(yt, yp))
        elif metric=="ap":
            if len(np.unique(yt))<2: continue
            vals.append(average_precision_score(yt, yp))
        elif metric in {"acc","prec","rec","f1"}:
            yhat = (yp >= chosen_thr).astype(int)
            if metric=="acc":  vals.append(accuracy_score(yt, yhat))
            if metric=="prec": vals.append(precision_score(yt, yhat, zero_division=0))
            if metric=="rec":  vals.append(recall_score(yt, yhat))
            if metric=="f1":   vals.append(f1_score(yt, yhat))
        elif metric=="prec_at_recall":
            assert target_recall is not None, "target_recall required for prec_at_recall"
            prec_b, rec_b, _ = precision_recall_curve(yt, yp)
            idx = np.where(rec_b>=target_recall)[0]
            if len(idx)==0: continue
            vals.append(float(prec_b[idx[-1]]))
        else:
            raise ValueError(f"Unknown metric {metric}")

    if not vals: return (float("nan"), float("nan"), float("nan"))
    vals = np.array(vals)
    return (float(np.mean(vals)), float(np.percentile(vals,2.5)), float(np.percentile(vals,97.5)))

def plot_curves(y_true, y_prob, out_dir, prefix):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'k--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC {prefix}")
    plt.tight_layout(); plt.savefig(Path(out_dir)/f"roc_{prefix}.png", dpi=200); plt.close()
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR {prefix}")
    plt.tight_layout(); plt.savefig(Path(out_dir)/f"pr_{prefix}.png", dpi=200); plt.close()

def write_all_cis(y_true, y_prob, out_file):
    with open(out_file, "w") as f:
        for met in ["auc","ap"]:
            m, lo, hi = bootstrap_ci(y_true, y_prob, metric=met)
            f.write(f"{met.upper()}: mean={m:.4f}, 95%CI=({lo:.4f},{hi:.4f})\n")
        for met in ["acc","prec","rec","f1"]:
            m, lo, hi = bootstrap_ci(y_true, y_prob, metric=met, strategy="fixed", threshold=0.5)
            f.write(f"{met.upper()}@0.5: mean={m:.4f}, 95%CI=({lo:.4f},{hi:.4f})\n")
        m, lo, hi = bootstrap_ci(y_true, y_prob, metric="prec_at_recall", target_recall=0.90)
        f.write(f"Precision@Recall=0.90: mean={m:.4f}, 95%CI=({lo:.4f},{hi:.4f})\n")

def parse_tests_arg(args_tests):
    """
    Parse multiple --tests name:path pairs into list of (name, path)
    Example: --tests cbis:data/cbis_test.csv --tests vindr:data/vindr_test.csv
    """
    pairs = []
    if args_tests:
        for s in args_tests:
            if ":" in s:
                name, path = s.split(":", 1)
                pairs.append((name.strip(), path.strip()))
            else:
                p = Path(s); pairs.append((p.stem, s))
    return pairs

def eval_and_write(model, device, csv_path, outdir, prefix, img_size, path_col, label_col, batch, workers):
    ds = MammogramDataset(csv_path, path_col=path_col, label_col=label_col, img_size=img_size, augment=False)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)
    metrics, y_true, y_prob = evaluate(model, loader, device, return_probs=True)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    json.dump(metrics, open(Path(outdir)/f"{prefix}.json","w"))
    plot_curves(y_true, y_prob, outdir, prefix)
    write_all_cis(y_true, y_prob, Path(outdir)/f"{prefix}_ci.txt")
    print(f"[Done] Wrote metrics/curves/CIs for {prefix} into {outdir}")
    print(json.dumps(metrics, indent=2))

# ------------------------ CLI ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['resnet50','swin_t','convnext_b'], required=True,
                    help="Model architecture used in training.")
    ap.add_argument('--ckpt', type=str, required=True,
                    help="Path to the trained checkpoint (.pt) to load.")
    ap.add_argument('--test_csv', type=str, default=None,
                    help="Optional single combined test CSV.")
    ap.add_argument('--tests', action='append',
                    help="Optional multiple test sets as name:path (can be used multiple times).")
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--path_col', type=str, default='roi_png_path')
    ap.add_argument('--label_col', type=str, default='label')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--device', type=str, default='auto', choices=['auto','cpu','cuda'])
    ap.add_argument('--out', type=str, default='results_test',
                    help="Output directory to store metrics/plots/CI.")
    ap.add_argument('--num_classes', type=int, default=2,
                    help="Number of classes (keep 2 if you trained for binary).")
    ap.add_argument('--pretrained_backbone', action='store_true',
                    help="Load imagenet weights before checkpoint (usually not needed).")
    args = ap.parse_args()

    # Select device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Build model and load checkpoint
    model = build_model(args.model, num_classes=args.num_classes, pretrained=args.pretrained_backbone).to(device)
    ckpt_path = Path(args.ckpt)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
    state = torch.load(str(ckpt_path), map_location=device)
    # Accept both full state_dict and raw state_dict
    if isinstance(state, dict) and 'state_dict' in state:
        model.load_state_dict(state['state_dict'], strict=False)
    else:
        model.load_state_dict(state, strict=False)

    # Evaluate on provided test CSV (if any)
    if args.test_csv:
        prefix = f"{args.model}_test"
        eval_and_write(model, device, args.test_csv, args.out, prefix,
                       args.img_size, args.path_col, args.label_col, args.batch, args.workers)

    # Evaluate on named tests (if any)
    for name, path in parse_tests_arg(args.tests):
        prefix = f"{args.model}_test_{name}"
        eval_and_write(model, device, path, args.out, prefix,
                       args.img_size, args.path_col, args.label_col, args.batch, args.workers)

    if not args.test_csv and not args.tests:
        print("No test CSV provided. Use --test_csv or --tests name:path.")

if __name__ == "__main__":
    main()