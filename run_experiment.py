"""
Single entry-point to run any experiment from the paper.

Usage examples:

    # Centralized learning baseline (Table 3)
    python run_experiment.py --mode centralized --backbone resnet50

    # Heterogeneous FL — FedAvg (Table 5)
    python run_experiment.py --mode federated --algorithm fedavg --backbone resnet50

    # Heterogeneous FL — all four algorithms
    python run_experiment.py --mode federated --algorithm fedprox  --backbone resnet50
    python run_experiment.py --mode federated --algorithm scaffold --backbone resnet50
    python run_experiment.py --mode federated --algorithm fedbn   --backbone resnet50

    # Swin V2-T backbone
    python run_experiment.py --mode federated --algorithm fedavg --backbone swin_v2_t

    # Homogeneous FL control — VinDr only (Table 4)
    python run_experiment.py --mode federated --algorithm fedavg \
        --backbone resnet50 --homogeneous vindr

    # Homogeneous FL control — CBIS only (Table 4)
    python run_experiment.py --mode federated --algorithm fedavg \
        --backbone resnet50 --homogeneous cbis

    # Size-balancing ablation (Table 7)
    python run_experiment.py --mode federated --algorithm fedavg \
        --backbone resnet50 --size_balance

    # Resolution ablation 324px (Table 6)
    python run_experiment.py --mode federated --algorithm fedavg \
        --backbone resnet50 --input_size 324

    # Local-only baselines (Table 3)
    python run_experiment.py --mode local --domain cbis  --backbone resnet50
    python run_experiment.py --mode local --domain vindr --backbone resnet50

Reference:
    Ali et al., Front. Digit. Health 8:1715858 (2026)
    doi: 10.3389/fdgth.2026.1715858
"""

import argparse
import os
import torch
import random
import numpy as np
import yaml
from pathlib import Path


# --------------------------------------------------------------------------- #
#  Reproducibility (Section 2.6.7)                                            #
# --------------------------------------------------------------------------- #

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN determinism (Section 2.6.7)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False    # benchmarking disabled


# --------------------------------------------------------------------------- #
#  Config loader                                                               #
# --------------------------------------------------------------------------- #

def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
#  Centralized learning                                                        #
# --------------------------------------------------------------------------- #

def run_centralized(cfg, args, device):
    from models.backbones import build_backbone
    from data.dataset import build_centralized_loaders
    import torch.nn as nn
    from torch.optim import SGD
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from evaluation.metrics import compute_all_metrics
    import pandas as pd

    manifest_csvs = cfg["data"]["manifest_csvs"]
    output_dir    = Path(cfg["output"]["results_dir"]) / f"centralized_{args.backbone}"
    output_dir.mkdir(parents=True, exist_ok=True)

    loaders = build_centralized_loaders(
        manifest_csvs, args.input_size,
        cfg["training"]["batch_size"],
    )
    model     = build_backbone(args.backbone, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=cfg["training"]["lr"],
                    momentum=cfg["training"]["momentum"],
                    weight_decay=cfg["training"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])

    best_val_auc = 0.0
    print(f"\nCentralized — {args.backbone} | "
          f"input={args.input_size}px | device={device}")

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        # Training
        model.train()
        for batch in loaders["train"]:
            images  = batch["image"].to(device)
            labels  = batch["label"].to(device)
            loss    = criterion(model(images), labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        logits_all, labels_all = [], []
        with torch.no_grad():
            for batch in loaders["val"]:
                logits_all.append(model(batch["image"].to(device)).cpu())
                labels_all.append(batch["label"])
        probs  = torch.cat(logits_all).softmax(1)[:, 1].numpy()
        labels = torch.cat(labels_all).numpy()
        val_metrics = compute_all_metrics(labels, probs)
        val_auc     = val_metrics["auroc"] or 0.0

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | Val AUC={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_auc": val_auc}, output_dir / "best_model.pth")

    print(f"Best val AUC: {best_val_auc:.4f}")


# --------------------------------------------------------------------------- #
#  Local-only baseline                                                         #
# --------------------------------------------------------------------------- #

def run_local(cfg, args, device):
    from models.backbones import build_backbone
    from data.dataset import build_federated_client_loaders
    import torch.nn as nn
    from torch.optim import SGD
    from evaluation.metrics import compute_all_metrics

    manifest_csvs = cfg["data"]["manifest_csvs"]
    domain        = args.domain
    output_dir    = Path(cfg["output"]["results_dir"]) / f"local_{domain}_{args.backbone}"
    output_dir.mkdir(parents=True, exist_ok=True)

    loaders   = build_federated_client_loaders(
        manifest_csvs, domain, args.input_size, cfg["training"]["batch_size"]
    )
    model     = build_backbone(args.backbone, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=cfg["training"]["lr"],
                    momentum=cfg["training"]["momentum"],
                    weight_decay=cfg["training"]["weight_decay"])

    best_val_auc = 0.0
    print(f"\nLocal-only — domain={domain} | backbone={args.backbone}")

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        for batch in loaders["train"]:
            loss = criterion(model(batch["image"].to(device)), batch["label"].to(device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        logits_all, labels_all = [], []
        with torch.no_grad():
            for batch in loaders["val"]:
                logits_all.append(model(batch["image"].to(device)).cpu())
                labels_all.append(batch["label"])
        probs   = torch.cat(logits_all).softmax(1)[:, 1].numpy()
        labels  = torch.cat(labels_all).numpy()
        val_auc = (compute_all_metrics(labels, probs)["auroc"] or 0.0)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({"model_state_dict": model.state_dict(),
                        "val_auc": val_auc}, output_dir / "best_model.pth")

    print(f"Best val AUC ({domain}): {best_val_auc:.4f}")


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="Run any paper experiment")
    parser.add_argument("--mode",        choices=["centralized", "federated", "local"],
                        required=True)
    parser.add_argument("--backbone",    choices=["resnet50", "swin_v2_t"],
                        default="resnet50")
    parser.add_argument("--algorithm",   choices=["fedavg", "fedprox", "scaffold", "fedbn"],
                        default="fedavg")
    parser.add_argument("--homogeneous", choices=["cbis", "vindr"], default=None,
                        help="Homogeneous FL control (Table 4)")
    parser.add_argument("--domain",      choices=["cbis", "vindr"], default="cbis",
                        help="Domain for local-only baseline")
    parser.add_argument("--size_balance", action="store_true",
                        help="Size-balancing ablation (Table 7)")
    parser.add_argument("--input_size",  type=int, default=224,
                        help="224 (default) or 324 (resolution ablation, Table 6)")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--config",      default="configs/config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Three fixed random seeds used in the paper (Section 2.6.7)
    seeds = [42, 123, 456] if args.seed == 42 else [args.seed]

    for seed in seeds:
        set_seed(seed)
        print(f"\n{'='*65}")
        print(f"  Seed: {seed}")
        print(f"{'='*65}")

        if args.mode == "centralized":
            run_centralized(cfg, args, device)

        elif args.mode == "local":
            run_local(cfg, args, device)

        elif args.mode == "federated":
            from federated.server import run_federated
            tag = f"{args.algorithm}_{args.backbone}_seed{seed}"
            if args.size_balance:  tag += "_balanced"
            if args.homogeneous:   tag += f"_hom{args.homogeneous}"
            if args.input_size != 224: tag += f"_{args.input_size}px"

            run_federated(
                manifest_csvs  = cfg["data"]["manifest_csvs"],
                backbone_name  = args.backbone,
                algorithm_name = args.algorithm,
                num_rounds     = cfg["federated"]["num_rounds"],
                local_epochs   = cfg["federated"]["local_epochs"],
                lr             = cfg["training"]["lr"],
                batch_size     = cfg["training"]["batch_size"],
                input_size     = args.input_size,
                size_balance   = args.size_balance,
                homogeneous    = args.homogeneous,
                output_dir     = Path(cfg["output"]["results_dir"]) / tag,
                device         = device,
                seed           = seed,
            )
