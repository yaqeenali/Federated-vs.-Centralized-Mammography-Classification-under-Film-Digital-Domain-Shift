# runner.py
# CL, Local-only, FL (FedAvg/FedProx/FedBN/SCAFFOLD)
# Seeds + bootstrap CIs (no K-fold)
# Validation is created from train.csv via StratifiedShuffleSplit when --val_from_train is given.
# CSV schema expected: dataset, img, label, roi_png_path

import os, math, json, random, argparse, copy
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pydicom
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tvm
import torchvision.transforms as T
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Optional tqdm progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False

# ------------------------ Reproducibility ------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

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
        aug = [T.RandomHorizontalFlip(0.5),
               T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
               T.RandomRotation(15)]
        base = [T.Resize((img_size, img_size)), T.ToTensor(), T.Normalize(mean, std)]
        self.train_tf = T.Compose(aug + base)
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
                # First attempt: PIL
                with Image.open(path) as im:
                    img = np.array(im.convert("RGB"), dtype=np.float32) / 255.0
            return img
        except Exception as e_pil:
            # Second attempt: OpenCV (more tolerant + Windows-safe)
            try:
                data = np.fromfile(path, dtype=np.uint8)  # handles Unicode/Windows paths
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
        x = self.train_tf(x) if self.augment else self.eval_tf(x)
        y = int(row[self.label_col])
        meta = {
            'dataset': row.get('dataset', 'NA'),
            'path': path
        }
        return x, y, meta

# ------------------------ Models ------------------------
def build_model(name="resnet50", num_classes=2, pretrained=True):
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

def make_optim(model, lr=1e-2, wd=1e-4, momentum=0.9):
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

# ------------------------ Eval & Bootstrap ------------------------
@torch.no_grad()
def evaluate(model, loader, device, return_probs=False):
    model.eval()
    y_true, y_prob, y_pred = [], [], []
    it = tqdm(loader, desc="Eval", leave=False) if TQDM_AVAILABLE else loader

    for xb, yb, _ in it:
        xb = xb.to(device)
        logits = model(xb)

        # Sanitize logits to avoid NaN/Inf → NaN after softmax
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

        prob1 = torch.softmax(logits, dim=1)[:, 1]
        # Sanitize probabilities as well (belt and suspenders)
        prob1 = torch.nan_to_num(prob1, nan=0.5, posinf=1.0, neginf=0.0).detach().cpu().numpy()

        y_prob.append(prob1)
        y_true.append(yb.numpy())
        y_pred.append(np.argmax(logits.detach().cpu().numpy(), axis=1))

    if len(y_true) == 0:  # empty loader guard
        empty = {'acc': float('nan'),'f1': float('nan'),'prec': float('nan'),
                 'rec': float('nan'),'auc': float('nan'),'ap': float('nan')}
        return (empty, np.array([]), np.array([])) if return_probs else empty

    y_true = np.concatenate(y_true).astype(int)
    y_pred = np.concatenate(y_pred).astype(int)
    y_prob = np.concatenate(y_prob).astype(float)

    # Drop any non-finite probabilities
    finite_mask = np.isfinite(y_prob)
    if not finite_mask.all():
        # optional: print how many were dropped
        print(f"[WARN] Dropping {np.sum(~finite_mask)} non-finite probabilities during evaluation.")
        y_true = y_true[finite_mask]
        y_pred = y_pred[finite_mask]
        y_prob = y_prob[finite_mask]

    # If everything got dropped, bail gracefully
    if y_true.size == 0:
        empty = {'acc': float('nan'),'f1': float('nan'),'prec': float('nan'),
                 'rec': float('nan'),'auc': float('nan'),'ap': float('nan')}
        return (empty, np.array([]), np.array([])) if return_probs else empty

    metrics = {
        'acc': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'prec': precision_score(y_true, y_pred, zero_division=0),
        'rec': recall_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float('nan'),
        'ap': average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float('nan')
    }
    if return_probs:
        return metrics, y_true, y_prob
    return metrics

def eval_fedbn_per_client(
    global_model,
    bn_cache,
    bn_keys,
    device,
    tests_by_client,   # dict: cname -> csv_path
    outdir,
    prefix,
    img_size,
    path_col,
    label_col,
    batch,
    workers
):
    """
    Proper FedBN evaluation:
    - For each client, inject its BN stats
    - Evaluate ONLY on that client's test data
    """
    for cname, test_csv in tests_by_client.items():
        assert cname in bn_cache and bn_cache[cname] is not None, \
            f"Missing BN cache for client {cname}"

        # Clone global model
        model = copy.deepcopy(global_model).to(device)

        # Inject client-specific BN
        sd = model.state_dict()
        for k in bn_keys:
            sd[k] = bn_cache[cname][k].to(device)
        model.load_state_dict(sd, strict=True)

        loader = DataLoader(
            MammogramDataset(
                test_csv,
                path_col=path_col,
                label_col=label_col,
                img_size=img_size,
                augment=False
            ),
            batch_size=batch,
            shuffle=False,
            num_workers=workers
        )

        metrics, y_true, y_prob = evaluate(model, loader, device, return_probs=True)

        out_prefix = f"{prefix}_client_{cname}"
        json.dump(metrics, open(Path(outdir) / f"{out_prefix}.json", "w"))

        plot_curves(y_true, y_prob, outdir, out_prefix)
        write_all_cis(y_true, y_prob, Path(outdir) / f"{out_prefix}_ci.txt")

def precision_at_recall(y_true, y_prob, target_recall=0.90):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    idx = np.where(rec>=target_recall)[0]
    if len(idx)==0: return None
    return float(prec[idx[-1]])

def bootstrap_ci(
    y_true, y_prob, n_boot=1000, metric="auc", seed=123,
    threshold=0.5, strategy="fixed", target_recall=None
):
    """
    metric: "auc","ap","acc","prec","rec","f1","prec_at_recall"
    strategy for thresholded metrics: "fixed" (thr=0.5), or "optimal_f1"/"youden"/"at_recall"
    For "prec_at_recall", use target_recall.
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    chosen_thr = None
    if metric in {"acc","prec","rec","f1"}:
        if strategy=="fixed":
            chosen_thr = float(threshold)
        elif strategy=="optimal_f1":
            prec, rec, thr = precision_recall_curve(y_true, y_prob)
            f1_vals = 2*prec*rec / (prec + rec + 1e-12)
            if len(thr)>0:
                best_i = int(np.nanargmax(f1_vals[:-1]))
                chosen_thr = float(thr[best_i])
            else: chosen_thr = float(threshold)
        elif strategy=="youden":
            fpr, tpr, thr = roc_curve(y_true, y_prob)
            if len(thr)>0:
                youden = tpr - fpr
                best_i = int(np.nanargmax(youden))
                chosen_thr = float(thr[best_i])
            else: chosen_thr = float(threshold)
        elif strategy=="at_recall":
            assert target_recall is not None, "target_recall required for strategy='at_recall'"
            prec, rec, thr = precision_recall_curve(y_true, y_prob)
            idx = np.where(rec>=target_recall)[0]
            if len(idx)==0: chosen_thr = 0.0
            else:
                j = idx[-1]
                chosen_thr = float(thr[j-1]) if (j-1)>=0 and (j-1)<len(thr) else float(threshold)
        else:
            raise ValueError(f"Unknown strategy {strategy}")

    vals = []
    it = tqdm(range(n_boot), desc=f"Bootstrap {metric}", leave=False) if TQDM_AVAILABLE else range(n_boot)
    for _ in it:
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
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if len(np.unique(y_true)) > 1 and np.isfinite(y_prob).all():
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            plt.figure(); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'k--')
            plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC {prefix}")
            plt.tight_layout(); plt.savefig(Path(out_dir)/f"roc_{prefix}.png", dpi=200); plt.close()
        except Exception as e:
            print(f"[WARN] ROC plot skipped: {e}")
        try:
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            plt.figure(); plt.plot(rec, prec); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR {prefix}")
            plt.tight_layout(); plt.savefig(Path(out_dir)/f"pr_{prefix}.png", dpi=200); plt.close()
        except Exception as e:
            print(f"[WARN] PR plot skipped: {e}")

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

# ------------------------ Training ------------------------
def train_one_epoch(
    model, loader, device, optimizer, scaler=None,
    prox_mu=0.0, global_state=None, desc="Train",
    scaffold=False, c_local=None, c_global=None
):
    """
    Supports FedProx via (prox_mu, global_state) and SCAFFOLD via (scaffold, c_local, c_global).
    """
    model.train()
    ce = nn.CrossEntropyLoss()
    losses = []
    it = tqdm(loader, desc=desc, leave=False) if TQDM_AVAILABLE else loader

    for xb, yb, _ in it:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is None:
            logits = model(xb)
            loss = ce(logits, yb)

            # FedProx proximal term
            if prox_mu > 0.0 and global_state is not None:
                prox = 0.0
                for (name, p) in model.named_parameters():
                    if p.requires_grad and name in global_state:
                        prox = prox + torch.sum((p - global_state[name].to(device))**2)
                loss = loss + (prox_mu/2.0)*prox

            loss.backward()

            # SCAFFOLD gradient correction (after backward, before step)
            if scaffold and (c_local is not None) and (c_global is not None):
                with torch.no_grad():
                    for (name, p) in model.named_parameters():
                        if p.grad is not None and p.requires_grad and (name in c_local) and (name in c_global):
                            p.grad += (c_local[name].to(p.device) - c_global[name].to(p.device))
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        else:
            with torch.amp.autocast('cuda', enabled=True):
                logits = model(xb)
                loss = ce(logits, yb)
                if prox_mu > 0.0 and global_state is not None:
                    prox = 0.0
                    for (name, p) in model.named_parameters():
                        if p.requires_grad and name in global_state:
                            prox = prox + torch.sum((p - global_state[name].to(device))**2)
                    loss = loss + (prox_mu/2.0)*prox

            scaler.scale(loss).backward()

            # AMP-safe: unscale before modifying gradients
            scaler.unscale_(optimizer)
            if scaffold and (c_local is not None) and (c_global is not None):
                with torch.no_grad():
                    for (name, p) in model.named_parameters():
                        if p.grad is not None and p.requires_grad and (name in c_local) and (name in c_global):
                            p.grad += (c_local[name].to(p.device) - c_global[name].to(p.device))
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

        losses.append(loss.item())

    return float(np.mean(losses)) if losses else 0.0

# ------------------------ Federated helpers ------------------------
def get_param_vector(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def set_param_vector(model, state):
    model.load_state_dict(state, strict=True)

def average_states(states, weights=None, skip_bn=False):
    """
    Robust averaging:
    - Never average 'num_batches_tracked' (int counter).
    - If skip_bn=True: keep BN params/buffers from the first client (FedBN handles BN separately).
    - Do not average non-floating tensors (copy from first).
    """
    keys = states[0].keys()
    if weights is None:
        weights = [1.0/len(states)]*len(states)
    out = {}
    for k in keys:
        # Avoid averaging the int64 counter
        if "num_batches_tracked" in k:
            out[k] = states[0][k]
            continue

        is_bn = (("bn" in k.lower()) or ("running_mean" in k) or ("running_var" in k))
        if skip_bn and is_bn:
            out[k] = states[0][k]
            continue

        t0 = states[0][k]
        if not t0.is_floating_point():
            out[k] = t0
            continue

        tensor = sum(weights[i] * states[i][k] for i in range(len(states)))
        out[k] = tensor
    return out

def split_clients(csv_path, by_col='dataset'):
    df = pd.read_csv(csv_path)
    assert by_col in df.columns, f"{by_col} must be in CSV"
    return {k: sub for k, sub in df.groupby(by_col)}

def split_bn_and_nonbn_keys(state_dict):
    bn_keys, nonbn_keys = [], []
    for k in state_dict.keys():
        if ("bn" in k.lower()) or ("running_mean" in k) or ("running_var" in k) or ("num_batches_tracked" in k):
            bn_keys.append(k)
        else:
            nonbn_keys.append(k)
    return bn_keys, nonbn_keys

def named_param_dict(model):
    # map of parameter names -> Parameter tensors
    return {name: p for name, p in model.named_parameters()}

# ------------------------ Test sets parser ------------------------
def parse_tests_arg(args_tests):
    """
    Parse multiple --tests name:path pairs into list of (name, path)
    Example: --tests cbis:combined\\cbis_test.csv --tests vindr:combined\\vindr_test.csv
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

# ------------------------ Evaluate & write ------------------------
def eval_and_write(model, device, csv_path, outdir, prefix, img_size, path_col, label_col, batch, workers):
    loader = DataLoader(MammogramDataset(csv_path, path_col=path_col, label_col=label_col, img_size=img_size),
                        batch_size=batch, shuffle=False, num_workers=workers)
    metrics, y_true, y_prob = evaluate(model, loader, device, return_probs=True)
    json.dump(metrics, open(Path(outdir)/f"{prefix}.json","w"))
    plot_curves(y_true, y_prob, outdir, prefix)
    write_all_cis(y_true, y_prob, Path(outdir)/f"{prefix}_ci.txt")

# ------------------------ Train/Val split from train.csv ------------------------
def make_val_from_train(train_csv, val_ratio, seed, label_col, outdir):
    """
    Creates (train_split.csv, val_split.csv) in outdir/tmp_valsplit_seed{seed}/
    stratified on label_col.
    """
    df = pd.read_csv(train_csv).reset_index(drop=True)
    y = df[label_col].values.astype(int)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    tr_idx, va_idx = next(splitter.split(df, y))
    df_tr, df_va = df.iloc[tr_idx].copy(), df.iloc[va_idx].copy()
    tmpdir = Path(outdir)/f"tmp_valsplit_seed{seed}"
    tmpdir.mkdir(parents=True, exist_ok=True)
    tr_csv = tmpdir/"train_split.csv"; va_csv = tmpdir/"val_split.csv"
    df_tr.to_csv(tr_csv, index=False); df_va.to_csv(va_csv, index=False)
    return str(tr_csv), str(va_csv)

def make_val_from_train_by_dataset_and_patient(train_csv, val_ratio, seed,
                                               label_col, dataset_col, patient_col, outdir):
    """
    Leak-safe: splits at patient level. For each (dataset, label), we split groups (patients)
    to form validation, preserving label balance within each dataset.
    """
    rng = np.random.RandomState(seed)
    df = pd.read_csv(train_csv).reset_index(drop=True)

    val_patient_ids = set()

    for (ds_name, label), sub in df.groupby([dataset_col, label_col]):
        # patient groups within this (dataset,label)
        groups = sub[patient_col].dropna().astype(str)
        if groups.empty:
            print(f"[WARN] ({ds_name},{label}) has missing {patient_col}; falling back to row-level split.")
            # fallback: row-level sampling
            n_val = int(round(len(sub) * val_ratio))
            val_idx = rng.choice(sub.index.values, size=n_val, replace=False)
            val_patient_ids.update(df.loc[val_idx, patient_col].dropna().astype(str).tolist())
            continue

        unique_patients = groups.unique()
        n_val_groups = max(1, int(round(len(unique_patients) * val_ratio)))

        # sample patients, not rows
        val_patients = rng.choice(unique_patients, size=n_val_groups, replace=False)
        val_patient_ids.update(val_patients.tolist())

    # Build splits by patient membership
    is_val = df[patient_col].astype(str).isin(val_patient_ids)
    df_va = df[is_val].copy()
    df_tr = df[~is_val].copy()

    tmpdir = Path(outdir)/f"tmp_valsplit_seed{seed}"
    tmpdir.mkdir(parents=True, exist_ok=True)
    tr_csv = tmpdir/"train_split.csv"; va_csv = tmpdir/"val_split.csv"
    df_tr.to_csv(tr_csv, index=False); df_va.to_csv(va_csv, index=False)
    return str(tr_csv), str(va_csv)

#____________________helper function for ablation study_____________


# --- Add these helpers somewhere above `main()` (e.g., after other helpers) ---

def stratify_single_dataset_into_n_clients(df, dataset_col, label_col, n_clients, seed):
    """
    If df has only one unique dataset in `dataset_col`, split it into n_clients pseudo-clients
    via stratified (by label_col) partitioning. Returns a modified copy where `dataset_col`
    is replaced with client names: "<orig>_c1", "<orig>_c2", ...
    """
    assert n_clients >= 2, "n_clients must be >= 2"
    uniq = df[dataset_col].dropna().astype(str).unique()
    assert len(uniq) == 1, "stratify_single_dataset_into_n_clients expects exactly one dataset"
    base = uniq[0]
    rng = np.random.RandomState(seed)
    df = df.copy().reset_index(drop=True)
    client_names = [f"{base}_c{i+1}" for i in range(n_clients)]
    assign = np.empty(len(df), dtype=object)

    # stratify by label
    for label, sub in df.groupby(label_col):
        idx = sub.index.values.copy()
        rng.shuffle(idx)
        splits = np.array_split(idx, n_clients)
        for i, sp in enumerate(splits):
            if len(sp) > 0:
                assign[sp] = client_names[i]

    # Fill any unassigned (edge cases) in a round-robin
    rr = 0
    for i in range(len(df)):
        if assign[i] is None or assign[i] == "":
            assign[i] = client_names[rr % n_clients]
            rr += 1

    df[dataset_col] = assign
    return df


def balance_vindr_by_dropping_negatives_to_match(df, dataset_col, label_col, seed):
    """
    Downsample rows where (dataset=='vindr' and label==0) so that total size of 'vindr'
    equals total size of 'cbis'. If 'vindr' is not larger than 'cbis', or datasets missing,
    returns df unchanged. Uses `seed` for reproducibility.
    """
    rng = np.random.RandomState(seed)
    dsl = df[dataset_col].astype(str).str.lower()
    if not (("vindr" in set(dsl)) and ("cbis" in set(dsl))):
        print("[balance] Skip: need both ViNDR and CBIS in dataset_col.")
        return df

    # Get canonical keys as they appear
    ds_vindr = df.loc[dsl.eq("vindr"), dataset_col].iloc[0]
    ds_cbis  = df.loc[dsl.eq("cbis"), dataset_col].iloc[0]

    sizes = df.groupby(dataset_col, dropna=False).size().to_dict()
    n_vindr = int(sizes.get(ds_vindr, 0))
    n_cbis  = int(sizes.get(ds_cbis, 0))

    if n_vindr <= n_cbis:
        print(f"[balance] Skip: ViNDR (n={n_vindr}) is not larger than CBIS (n={n_cbis}).")
        return df

    excess = n_vindr - n_cbis
    vindr_neg_idx = df.index[(df[dataset_col] == ds_vindr) & (df[label_col].astype(int) == 0)].to_numpy()

    if len(vindr_neg_idx) == 0:
        print("[balance] WARNING: No ViNDR negatives to drop; cannot equalize totals.")
        return df

    drop_n = min(excess, len(vindr_neg_idx))
    to_drop = rng.choice(vindr_neg_idx, size=drop_n, replace=False)
    print(f"[balance] Dropping {drop_n} ViNDR negatives (label=0) to match CBIS size (target total={n_cbis}).")

    df_bal = df.drop(index=to_drop).reset_index(drop=True)

    # Optional info if not fully equalized
    new_sizes = df_bal.groupby(dataset_col, dropna=False).size().to_dict()
    if int(new_sizes.get(ds_vindr, 0)) != n_cbis:
        print(f"[balance] NOTE: After dropping all negatives, ViNDR still != CBIS "
              f"(ViNDR={new_sizes.get(ds_vindr, 0)}, CBIS={n_cbis}).")

    return df_bal
# ------------------------ CL / Local-only / FL ------------------------
def run_centralized(args, seed, device, outdir, tr_csv, va_csv):
    set_seed(seed); Path(outdir).mkdir(parents=True, exist_ok=True)

    train_ds = MammogramDataset(tr_csv, path_col=args.path_col, label_col=args.label_col,
                                img_size=args.img_size, augment=True)
    val_ds   = MammogramDataset(va_csv, path_col=args.path_col, label_col=args.label_col,
                                img_size=args.img_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=args.workers)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    model = build_model(args.model, num_classes=2, pretrained=True).to(device)
    opt   = make_optim(model, lr=args.lr, wd=args.weight_decay, momentum=args.momentum)
    scaler = torch.amp.GradScaler('cuda') if (args.amp and device.type=='cuda') else None

    log_path = Path(outdir)/f"cl_{args.model}_train_log_seed{seed}.csv"; log_df = []

    best_score, best_state = -1, None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs+1):
        train_loss = train_one_epoch(model, train_loader, device, opt, scaler=scaler, desc=f"Train E{epoch}")
        m_val = evaluate(model, val_loader, device)
        score = m_val['auc'] if not math.isnan(m_val['auc']) else m_val['acc']

        log_df.append({'epoch':epoch, 'train_loss':train_loss, **m_val})
        pd.DataFrame(log_df).to_csv(log_path, index=False)

        # Improvement check with min_delta
        if (score - best_score) > args.min_delta:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            json.dump({'epoch':epoch, 'val':m_val},
                      open(Path(outdir)/f"cl_{args.model}_best_val_seed{seed}.json","w"))
            torch.save(best_state, Path(outdir)/f"cl_{args.model}_seed{seed}.pt")
        else:
            epochs_no_improve += 1

        # Early stopping condition
        if args.early_stop and epochs_no_improve >= args.patience:
            print(f"[Early stop] No improvement for {epochs_no_improve} epochs (patience={args.patience}).")
            break

    # Load best model before evaluation
    if best_state is not None:
        model.load_state_dict(best_state)

    # Combined test (if provided)
    if args.test_csv:
        eval_and_write(model, device, args.test_csv, outdir,
                       f"cl_{args.model}_seed{seed}_test",
                       args.img_size, args.path_col, args.label_col, args.batch, args.workers)

    # Additional named test sets
    for name, path in parse_tests_arg(args.tests):
        eval_and_write(model, device, path, outdir,
                       f"cl_{args.model}_seed{seed}_test_{name}",
                       args.img_size, args.path_col, args.label_col, args.batch, args.workers)

def run_local_only(args, seed, device, outdir, tr_csv, va_csv):
    set_seed(seed); Path(outdir).mkdir(parents=True, exist_ok=True)
    clients = split_clients(tr_csv, by_col=args.dataset_col)

    for cname, ctrain in clients.items():
        c_csv = Path(outdir)/f"tmp_{cname}_train_seed{seed}.csv"
        ctrain.to_csv(c_csv, index=False)

        train_ds = MammogramDataset(c_csv, path_col=args.path_col, label_col=args.label_col,
                                    img_size=args.img_size, augment=True)
        val_ds   = MammogramDataset(va_csv, path_col=args.path_col, label_col=args.label_col,
                                    img_size=args.img_size, augment=False)
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers)
        val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)

        model = build_model(args.model, num_classes=2, pretrained=True).to(device)
        opt   = make_optim(model, lr=args.lr, wd=args.weight_decay, momentum=args.momentum)
        scaler = torch.amp.GradScaler('cuda') if (args.amp and device.type=='cuda') else None

        log_path = Path(outdir)/f"local_{cname}_{args.model}_train_log_seed{seed}.csv"; log_df = []
        best_score, best_state = -1, None
        epochs_no_improve = 0

        for epoch in range(1, args.epochs+1):
            train_loss = train_one_epoch(model, train_loader, device, opt, scaler=scaler, desc=f"[{cname}] E{epoch}")
            m_val = evaluate(model, val_loader, device)
            score = m_val['auc'] if not math.isnan(m_val['auc']) else m_val['acc']

            log_df.append({'epoch':epoch, 'train_loss':train_loss, **m_val})
            pd.DataFrame(log_df).to_csv(log_path, index=False)

            if (score - best_score) > args.min_delta:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                json.dump({'epoch':epoch, 'val':m_val},
                          open(Path(outdir)/f"local_{cname}_{args.model}_best_val_seed{seed}.json","w"))
                torch.save(best_state, Path(outdir)/f"local_{cname}_{args.model}_seed{seed}.pt")
            else:
                epochs_no_improve += 1

            if args.early_stop and epochs_no_improve >= args.patience:
                print(f"[{cname}] Early stop: no improvement for {epochs_no_improve} epochs (patience={args.patience}).")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        if args.test_csv:
            eval_and_write(model, device, args.test_csv, outdir,
                           f"local_{cname}_{args.model}_seed{seed}_test",
                           args.img_size, args.path_col, args.label_col, args.batch, args.workers)
        for name, path in parse_tests_arg(args.tests):
            eval_and_write(model, device, path, outdir,
                           f"local_{cname}_{args.model}_seed{seed}_test_{name}",
                           args.img_size, args.path_col, args.label_col, args.batch, args.workers)

def run_federated(args, seed, device, outdir, tr_csv, va_csv):
    set_seed(seed); Path(outdir).mkdir(parents=True, exist_ok=True)

    clients = split_clients(tr_csv, by_col=args.dataset_col)
    train_loaders = {}
    for cname, cdf in clients.items():
        c_csv = Path(outdir)/f"tmp_{cname}_train_seed{seed}.csv"; cdf.to_csv(c_csv, index=False)
        ds = MammogramDataset(c_csv, path_col=args.path_col, label_col=args.label_col,
                              img_size=args.img_size, augment=True)
        train_loaders[cname] = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers)

    val_loader = DataLoader(MammogramDataset(va_csv, path_col=args.path_col, label_col=args.label_col,
                                             img_size=args.img_size), batch_size=args.batch, shuffle=False, num_workers=args.workers)

    global_model = build_model(args.model, num_classes=2, pretrained=True).to(device)
    algo = args.algo.lower()

    # ----- SCAFFOLD control variates over PARAMETERS only -----
    if algo == "scaffold":
        with torch.no_grad():
            pmap = named_param_dict(global_model)
            c_global = {name: torch.zeros_like(p, device='cpu') for name, p in pmap.items()}
            c_locals = {cname: {name: torch.zeros_like(p, device='cpu') for name, p in pmap.items()}
                        for cname in clients.keys()}
    else:
        c_global, c_locals = None, None

    # ----- FedBN per-client BN cache -----
    bn_cache = None
    bn_keys, nonbn_keys = split_bn_and_nonbn_keys(global_model.state_dict())
    if algo == "fedbn":
        bn_cache = {cname: None for cname in clients.keys()}
        bn_cache_best = None  # snapshot BN cache when validation improves

    scaler_enable = (args.amp and device.type=='cuda')
    best_val, best_state = -1, None
    rounds_no_improve = 0

    log_path = Path(outdir)/f"fl_{args.algo}_{args.model}_fl_log_seed{seed}.csv"; log_df = []

    for rnd in range(1, args.rounds+1):
        client_states, client_weights, rnd_losses = [], [], []

        # For SCAFFOLD: accumulate delta c_i across clients this round
        if algo == "scaffold":
            scaffold_c_deltas = {name: torch.zeros_like(val) for name, val in c_global.items()}

        itc = tqdm(list(clients.items()), desc=f"Round {rnd}", leave=False) if TQDM_AVAILABLE else clients.items()
        for cname, _ in itc:
            loader = train_loaders[cname]
            local = copy.deepcopy(global_model).to(device)

            # ---- FedBN: inject client's BN before training ----
            if algo == "fedbn" and bn_cache[cname] is not None:
                sd = local.state_dict()
                for k in bn_keys:
                    sd[k] = bn_cache[cname][k]
                local.load_state_dict(sd, strict=True)
            
            if algo == "scaffold":
                # SCAFFOLD paper uses plain SGD (no momentum)
                opt = optim.SGD(local.parameters(), lr=args.lr, momentum=0.0, weight_decay=args.weight_decay)
            else:
                opt = make_optim(local, lr=args.lr, wd=args.weight_decay, momentum=args.momentum)

            scaler = torch.amp.GradScaler('cuda') if scaler_enable else None
            
            gstate = None
            if algo == "fedprox" and args.mu > 0:
                # Anchor to the global round-start parameters (trainable only)
                with torch.no_grad():
                    gstate = {name: p.detach().clone() for name, p in global_model.named_parameters()}

            # SCAFFOLD: keep old c_i
            if algo == "scaffold":
                c_old = {k: v.clone() for k, v in c_locals[cname].items()}

            # ---- Local update ----
            for _ in range(args.local_epochs):
                loss = train_one_epoch(
                    local, loader, device, opt, scaler=scaler,
                    prox_mu=(args.mu if algo=="fedprox" else 0.0),
                    global_state=gstate, desc=f"[{cname}] R{rnd}",
                    scaffold=(algo=="scaffold"),
                    c_local=(c_locals[cname] if algo=="scaffold" else None),
                    c_global=(c_global if algo=="scaffold" else None)
                )
                rnd_losses.append(loss)

            l_state = local.state_dict()

            # ---- FedBN: cache client's BN for next round/eval ----
            if algo == "fedbn":
                bn_cache[cname] = {k: l_state[k].detach().cpu().clone() for k in bn_keys}

            # ---- SCAFFOLD: update client control variate c_i and accumulate for global c ----
            if algo == "scaffold":
                with torch.no_grad():
                    # K = steps per client this round = (#batches * local_epochs)
                    K = max(len(loader) * args.local_epochs, 1)
                    lr = args.lr
                    g_state = global_model.state_dict()

                    # Only for parameter keys (match c_locals keys)
                    for name, p in local.named_parameters():
                        delta = (g_state[name].detach().cpu() - l_state[name].detach().cpu()) / float(lr * K)
                        c_locals[cname][name] = c_locals[cname][name] - c_global[name] + delta

                    # Accumulate (c_i^{new} - c_i^{old})
                    for name in c_global.keys():
                        scaffold_c_deltas[name] += (c_locals[cname][name] - c_old[name])

            # Collect model for server aggregation
            client_states.append(get_param_vector(local))
            client_weights.append(len(loader.dataset))

        # --- CBIS Weight Boosting ---
        #boost_factor = 3.0  # 2–5x works well. Start with 3.
        #for i, cname in enumerate(clients.keys()):
         #   if "cbis" in cname.lower():
          #      print(f"[CBIS weighting] Boosting weight for client {cname} by x{boost_factor}")
           #     client_weights[i] *= boost_factor

        weights = np.array(client_weights, dtype=np.float32)
        weights = (weights / weights.sum()).tolist()

        if algo == "fedbn":
            # Aggregate only non-BN params
            agg_nonbn = {}
            for k in nonbn_keys:
                agg_nonbn[k] = sum(weights[i] * client_states[i][k] for i in range(len(client_states)))
            gsd = global_model.state_dict()
            for k in nonbn_keys:
                gsd[k] = agg_nonbn[k]
            # keep global BN as-is (clients will inject their own BN next round)
            global_model.load_state_dict(gsd, strict=True)
        else:
            # FedAvg / FedProx / SCAFFOLD: robust averaging
            new_state = average_states(client_states, weights=weights, skip_bn=False)
            set_param_vector(global_model, new_state)

        # ---- SCAFFOLD: update global control variate ----
        if algo == "scaffold":
            with torch.no_grad():
                m = float(len(clients))
                for name in c_global.keys():
                    c_global[name] = c_global[name] + (scaffold_c_deltas[name] / m)

        # Validation
        m_val = evaluate(global_model, val_loader, device)
        score = m_val['auc'] if not math.isnan(m_val['auc']) else m_val['acc']
        log_df.append({'round':rnd, 'mean_train_loss': float(np.mean(rnd_losses)) if rnd_losses else 0.0, **m_val})
        pd.DataFrame(log_df).to_csv(log_path, index=False)

        if (score - (best_val if best_val >= 0 else -1)) > args.min_delta:
            best_val = score
            best_state = copy.deepcopy(global_model.state_dict())
            rounds_no_improve = 0
            json.dump({'round':rnd, 'val':m_val},
                      open(Path(outdir)/f"fl_{args.algo}_{args.model}_best_val_seed{seed}.json","w"))
            torch.save(best_state, Path(outdir)/f"fl_{args.algo}_{args.model}_seed{seed}.pt")

            # Snapshot BN cache at the same time (FedBN)
            if algo == "fedbn":
                bn_cache_best = {
                    cname: ({k: v.detach().cpu().clone() for k, v in bn_cache[cname].items()}
                            if bn_cache[cname] is not None else None)
                    for cname in bn_cache.keys()
                }
        else:
            rounds_no_improve += 1

        if args.early_stop and rounds_no_improve >= args.patience:
            print(f"[Early stop] No improvement for {rounds_no_improve} rounds (patience={args.patience}).")
            break

    if best_state is not None:
        global_model.load_state_dict(best_state)

    # NOTE (FedBN): For per-dataset evaluation, you can load the corresponding BN from bn_cache[cname]
    # before evaluating on that dataset. Combined-domain evaluation has no single "right" BN.

    # ---------- Persist BN + optional personalized export ----------
    if args.algo == "fedbn":
        print("[FedBN] Persisting per-client BN caches and (optionally) exporting personalized models")

        # Prefer BN cache aligned with best validation round
        cache_to_use = bn_cache_best if (bn_cache is not None) else bn_cache

        # (A) Save each client's BN cache as a lightweight file
        bn_index = {}
        for cname, bn_state in cache_to_use.items():
            if bn_state is None:
                continue
            bn_file = Path(outdir) / f"fl_fedbn_{args.model}_seed{seed}_BN_{cname}.pt"
            torch.save(bn_state, bn_file)
            bn_index[cname] = bn_file.name

        # Save an index for convenience
        json.dump(
            bn_index,
            open(Path(outdir)/f"fl_fedbn_{args.model}_seed{seed}_BN_index.json", "w"),
            indent=2
        )

        # (B) Optional: export full personalized checkpoints (global + client BN)
        if args.export_personalized:
            for cname, bn_state in cache_to_use.items():
                if bn_state is None:
                    continue
                personalized = copy.deepcopy(global_model).to('cpu')
                sd = personalized.state_dict()
                # inject this client's BN keys
                for k in bn_keys:
                    if k in bn_state:
                        sd[k] = bn_state[k]
                personalized.load_state_dict(sd, strict=True)
                out_file = Path(outdir) / f"fl_fedbn_{args.model}_seed{seed}_client_{cname}.pt"
                torch.save(personalized.state_dict(), out_file)

        # ---- Evaluate per client using the same (best) BN cache ----
        print("[FedBN] Evaluating per-client using client-specific BN")
        # Build mapping: client -> test CSV
        tests_by_client = {}

        if args.test_csv:
            # Assume test_csv contains column dataset_col
            df_test = pd.read_csv(args.test_csv)
            for cname, sub in df_test.groupby(args.dataset_col):
                tmp_csv = Path(outdir) / f"tmp_test_{cname}_seed{seed}.csv"
                sub.to_csv(tmp_csv, index=False)
                tests_by_client[cname] = str(tmp_csv)

        # Additional named tests (optional)
        for name, path in parse_tests_arg(args.tests):
            df_test = pd.read_csv(path)
            for cname, sub in df_test.groupby(args.dataset_col):
                tmp_csv = Path(outdir) / f"tmp_test_{name}_{cname}_seed{seed}.csv"
                sub.to_csv(tmp_csv, index=False)
                tests_by_client[cname] = str(tmp_csv)

        eval_fedbn_per_client(
            global_model=global_model,
            bn_cache=cache_to_use,  # use best BN snapshot if available
            bn_keys=bn_keys,
            device=device,
            tests_by_client=tests_by_client,
            outdir=outdir,
            prefix=f"fl_fedbn_{args.model}_seed{seed}_test",
            img_size=args.img_size,
            path_col=args.path_col,
            label_col=args.label_col,
            batch=args.batch,
            workers=args.workers
        )

    else:
        # FedAvg / FedProx / SCAFFOLD (unchanged)
        if args.test_csv:
            eval_and_write(global_model, device, args.test_csv, outdir,
                        f"fl_{args.algo}_{args.model}_seed{seed}_test",
                        args.img_size, args.path_col, args.label_col, args.batch, args.workers)

        for name, path in parse_tests_arg(args.tests):
            eval_and_write(global_model, device, path, outdir,
                        f"fl_{args.algo}_{args.model}_seed{seed}_test_{name}",
                        args.img_size, args.path_col, args.label_col, args.batch, args.workers)

# ------------------------ CLI ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['cl','local','fl'], required=True)
    ap.add_argument('--algo', choices=['fedavg','fedprox','fedbn','scaffold'], default='scaffold')
    ap.add_argument('--model', choices=['resnet50','swin_t','convnext_b'], default='resnet50')
    ap.add_argument('--img_size', type=int, default=224)

    ap.add_argument('--train_csv', type=str, required=True)
    ap.add_argument('--val_csv', type=str, default=None)
    # may be ignored when --val_from_train
    ap.add_argument('--val_from_train', action='store_true', help="Create validation split from train.csv")
    ap.add_argument('--val_ratio', type=float, default=0.15, help="Validation ratio from train.csv when --val_from_train")
    ap.add_argument('--test_csv', type=str, default=None)    # optional combined test
    ap.add_argument('--tests', action='append', help="Optional multiple test sets as name:path, e.g., --tests cbis:combined\\cbis_test.csv")

    ap.add_argument('--dataset_col', type=str, default='dataset')
    ap.add_argument('--path_col', type=str, default='roi_png_path')
    ap.add_argument('--label_col', type=str, default='label')

    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=100)       # CL/local epochs
    ap.add_argument('--rounds', type=int, default=100)       # FL rounds
    ap.add_argument('--local_epochs', type=int, default=1)   # FL local steps
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--momentum', type=float, default=0.9)
    ap.add_argument('--mu', type=float, default=0.01)        # FedProx μ
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', type=str, default='results')

    ap.add_argument('--early_stop', action='store_true', help="Enable early stopping based on validation score")
    ap.add_argument('--patience', type=int, default=10, help="Epochs/rounds without improvement before stopping")
    ap.add_argument('--min_delta', type=float, default=0.0, help="Required minimal improvement to reset patience")

    # NEW: export per-client personalized checkpoints (global + client BN)
    ap.add_argument('--export_personalized', action='store_true',
                    help="(FedBN) Export full per-client personalized checkpoints (global + client BN)")
    
    ap.add_argument('--n_clients_if_single', type=int, default=2,
                        help="If train split contains a single dataset, stratify by label into this many pseudo-clients (e.g., 2).")
    ap.add_argument('--balance_vindr_negatives', action='store_true',
                        help="On mixed ViNDR+CBIS train data, drop ViNDR label=0 rows to match total size to CBIS (per seed).")

    args = ap.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for seed in range(args.seeds):
        # Create per-seed outdir and (if requested) per-seed train/val split from train.csv
        outdir = Path(args.out)/f"{args.mode}_{args.algo}_{args.model}_sz{args.img_size}_seed{seed}"
        outdir.mkdir(parents=True, exist_ok=True)

        if args.val_from_train:
            
            tr_csv, va_csv = make_val_from_train_by_dataset_and_patient(
                train_csv=args.train_csv,
                val_ratio=args.val_ratio,
                seed=seed,
                label_col=args.label_col,
                dataset_col=args.dataset_col,
                patient_col = 'img',
                outdir=outdir
            )

        else:
            assert args.val_csv is not None, "Provide --val_csv or use --val_from_train"
            tr_csv, va_csv = args.train_csv, args.val_csv

        # ---------- Optional re-shaping of TRAIN split ONLY ----------
        # (A) Balance ViNDR vs CBIS by downsampling ViNDR negatives to match total counts
        if args.balance_vindr_negatives:
            _df_tr = pd.read_csv(tr_csv)
            _df_tr_bal = balance_vindr_by_dropping_negatives_to_match(
                _df_tr, dataset_col=args.dataset_col, label_col=args.label_col, seed=seed
            )
            if len(_df_tr_bal) != len(_df_tr):
                tr_bal_csv = Path(outdir) / f"tmp_train_balanced_seed{seed}.csv"
                _df_tr_bal.to_csv(tr_bal_csv, index=False)
                tr_csv = str(tr_bal_csv)

        # (B) If FL over a single dataset (e.g., only CBIS), split into K pseudo-clients stratified by label
        if args.mode == 'fl' and args.n_clients_if_single and args.n_clients_if_single >= 2:
            _df_tr = pd.read_csv(tr_csv)
            if _df_tr[args.dataset_col].dropna().astype(str).nunique() == 1:
                _df_tr_clients = stratify_single_dataset_into_n_clients(
                    _df_tr, dataset_col=args.dataset_col, label_col=args.label_col,
                    n_clients=args.n_clients_if_single, seed=seed
                )
                # Save modified train CSV with pseudo-client dataset names
                tr_clients_csv = Path(outdir) / f"tmp_train_{args.n_clients_if_single}clients_seed{seed}.csv"
                _df_tr_clients.to_csv(tr_clients_csv, index=False)
                tr_csv = str(tr_clients_csv)
        if args.mode=='cl':
            run_centralized(args, seed, device, outdir, tr_csv, va_csv)
        elif args.mode=='local':
            run_local_only(args, seed, device, outdir, tr_csv, va_csv)
        else:
            run_federated(args, seed, device, outdir, tr_csv, va_csv)

if __name__ == "__main__":
    main()


"""
python runner.py --mode fl --algo fedbn --model resnet50 \
  --train_csv combined/cbis_train.csv \
  --val_from_train \
  --rounds 100 --local_epochs 1 --seeds 3 \
  --n_clients_if_single 2 \
  --export_personalized \
  --out results_fl_fedbnv6

 python runner.py --mode fl --algo fedavg --model resnet50 --train_csv combined/cbis_train.csv --val_from_train --rounds 100 --local_epochs 1 --seeds 3 --n_clients_if_single 2 --out results_fl_fedavg_resnet_homm_cbis


python runner.py --mode fl --algo fedbn --model resnet50 \
  --train_csv combined/vindr_cbis_train.csv \
  --val_from_train \
  --test_csv combined/vindr_cbis_test.csv \
  --rounds 100 --local_epochs 1 --seeds 3 \
  --balance_vindr_negatives \
  --export_personalized \
  --out results_fl_fedbnv6

python runner.py --mode fl --algo fedavg --model resnet50 --train_csv combined/vindr_cbis_train.csv --val_from_train --test_csv combined/vindr_cbis_test.csv --rounds 100 --local_epochs 1 --seeds 3 --balance_vindr_negatives  --out results_fl_fedavg_resnet_blc
python runner.py --mode cl --algo fedavg --model resnet50 --train_csv combined/vindr_cbis_train.csv --val_from_train --test_csv combined/vindr_cbis_test.csv --rounds 100 --local_epochs 1 --seeds 3 --balance_vindr_negatives  --out results_cl_resnet_blc



"""