"""
Federated learning server — synchronous parameter-server orchestration.

Protocol (Section 2.3.2 & 2.6):
    For each of 100 rounds:
        1. Server distributes current global model to all clients
        2. Each client trains E=3 local epochs on private data
        3. Client updates returned to server
        4. Server aggregates updates → new global model
        5. Server evaluates on combined validation set; saves best checkpoint

Clients in this study (two cross-silo clients):
    Client 1: CBIS-DDSM  (scanned film)
    Client 2: VinDr-Mammo (full-field digital)

Reference:
    Ali et al., Front. Digit. Health 8:1715858 (2026)
"""

import copy
import json
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from models.backbones import build_backbone
from data.dataset import build_federated_client_loaders, MammographyDataset
from federated.algorithms.fl_algorithms import FedAvg, FedProx, SCAFFOLD, FedBN
from evaluation.metrics import compute_all_metrics
from torch.utils.data import DataLoader


CLIENTS  = ["cbis", "vindr"]
DOMAINS  = {"cbis": "CBIS-DDSM (film)", "vindr": "VinDr-Mammo (digital)"}


def build_algorithm(name, mu=0.01):
    name = name.lower()
    if name == "fedavg":    return FedAvg()
    if name == "fedprox":   return FedProx(mu=mu)
    if name == "scaffold":  return SCAFFOLD()
    if name == "fedbn":     return FedBN()
    raise ValueError(f"Unknown FL algorithm: {name}")


def run_federated(manifest_csvs, backbone_name="resnet50",
                  algorithm_name="fedavg", num_rounds=100,
                  local_epochs=3, lr=0.01, batch_size=32,
                  input_size=224, size_balance=False,
                  homogeneous=None, output_dir="results/federated",
                  device=None, seed=42):
    """
    Run full federated learning experiment.

    Args:
        manifest_csvs:   list of manifest CSV paths
        backbone_name:   'resnet50' | 'swin_v2_t'
        algorithm_name:  'fedavg' | 'fedprox' | 'scaffold' | 'fedbn'
        num_rounds:      FL communication rounds (paper: 100)
        local_epochs:    local epochs per round (paper: 3)
        lr:              learning rate (paper: 0.01)
        batch_size:      (paper: 32)
        input_size:      224 (default) or 324 (resolution ablation)
        size_balance:    apply size-balancing ablation (Table 7)
        homogeneous:     None (heterogeneous) | 'cbis' | 'vindr'
                         (homogeneous controls, Table 4)
        output_dir:      where to save checkpoints and logs
        seed:            random seed
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  Federated Learning — {algorithm_name.upper()} | {backbone_name}")
    print(f"  Rounds={num_rounds} | E={local_epochs} | lr={lr} | "
          f"input={input_size}px")
    if homogeneous:
        print(f"  Mode: HOMOGENEOUS control on [{homogeneous}]")
    if size_balance:
        print(f"  Size-balancing ablation: ON")
    print(f"{'='*65}\n")

    # Build global model
    global_model = build_backbone(backbone_name, pretrained=True).to(device)
    algorithm    = build_algorithm(algorithm_name)

    # Determine active clients
    if homogeneous:
        # IID split within one domain (Table 4 controls)
        active_clients = [homogeneous, homogeneous + "_2"]
        client_domains = {homogeneous: homogeneous, homogeneous + "_2": homogeneous}
    else:
        active_clients = CLIENTS
        client_domains = {c: c for c in CLIENTS}

    # Build per-client data loaders
    client_loaders = {}
    for cid in [c for c in active_clients if c in CLIENTS]:
        client_loaders[cid] = build_federated_client_loaders(
            manifest_csvs, cid, input_size, batch_size,
            size_balance=size_balance, seed=seed
        )

    # Combined validation loader for model selection (by val AUROC)
    val_dataset = MammographyDataset(manifest_csvs, "val", input_size)
    val_loader  = DataLoader(val_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=4)

    best_val_auc = 0.0
    log_rows     = []

    for rnd in range(1, num_rounds + 1):
        client_states = []
        client_sizes  = []
        cv_deltas     = []   # SCAFFOLD only

        # Each client trains locally
        for cid in active_clients:
            local_model = copy.deepcopy(global_model).to(device)
            loader      = client_loaders.get(cid, {}).get("train")
            if loader is None:
                continue

            if algorithm_name == "fedprox":
                state, n = algorithm.local_update(
                    local_model, loader, device, global_model,
                    local_epochs, lr
                )
            elif algorithm_name == "scaffold":
                state, n, cvd = algorithm.local_update(
                    local_model, loader, device, cid, global_model,
                    local_epochs, lr
                )
                cv_deltas.append(cvd)
            else:
                state, n = algorithm.local_update(
                    local_model, loader, device, local_epochs, lr
                )

            client_states.append(state)
            client_sizes.append(n)

        # Server aggregation
        if algorithm_name == "scaffold":
            global_model = algorithm.aggregate(
                global_model, client_states, client_sizes, cv_deltas
            )
        elif algorithm_name == "fedbn":
            global_model = algorithm.aggregate(
                global_model, client_states, client_sizes, active_clients
            )
        else:
            global_model = algorithm.aggregate(
                global_model, client_states, client_sizes
            )

        # Validate on combined val set (model selection by AUROC, Section 2.5.3)
        val_metrics = _evaluate(global_model, val_loader, device)
        val_auc     = val_metrics["auroc"]

        log_rows.append({"round": rnd, **{f"val_{k}": v
                                          for k, v in val_metrics.items()}})

        if rnd % 10 == 0:
            print(f"  Round {rnd:3d}/{num_rounds} | "
                  f"Val AUC={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                "round": rnd, "val_auc": val_auc,
                "model_state_dict": global_model.state_dict(),
                "algorithm": algorithm_name,
                "backbone": backbone_name,
            }, output_dir / "best_model.pth")

    pd.DataFrame(log_rows).to_csv(output_dir / "training_log.csv", index=False)
    print(f"\nBest val AUC: {best_val_auc:.4f} | Logs: {output_dir}")
    return global_model


def _evaluate(model, loader, device):
    """Quick evaluation for model selection during training."""
    from evaluation.metrics import compute_all_metrics
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device))
            all_logits.append(logits.cpu())
            all_labels.append(batch["label"])
    probs  = torch.cat(all_logits).softmax(dim=1)[:, 1].numpy()
    labels = torch.cat(all_labels).numpy()
    return compute_all_metrics(labels, probs)
