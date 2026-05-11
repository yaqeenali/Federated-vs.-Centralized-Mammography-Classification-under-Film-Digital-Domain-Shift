# Federated vs. Centralized Mammography Classification under Film–Digital Domain Shift

[![Paper](https://img.shields.io/badge/Frontiers_Digital_Health-Published_2026-blue)](https://doi.org/10.3389/fdgth.2026.1715858)
[![Open Access](https://img.shields.io/badge/Open_Access-CC_BY-green)](https://creativecommons.org/licenses/by/4.0/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **Performance of federated versus centralized learning for mammography classification across film–digital domain shift**
> Yaqeen Ali, Julia Müller, Andreas Weinmann, Johannes Gregori
> *Frontiers in Digital Health*, 8:1715858, 2026
> DOI: [10.3389/fdgth.2026.1715858](https://doi.org/10.3389/fdgth.2026.1715858)

---

## Overview

A rigorous comparative evaluation of **federated learning (FL)** vs **centralized learning (CL)** for benign–malignant mammography lesion classification under **film–digital domain shift**. This study answers four clinical research questions:

| RQ | Question | Key Finding |
|---|---|---|
| **RQ1** | Can FL match CL in homogeneous settings? | ✅ Yes — FL ≈ CL within-domain |
| **RQ2** | Can FL algorithms mitigate film–digital degradation? | ⚠️ No — FedAvg most stable; FedProx/SCAFFOLD/FedBN offer no consistent gain |
| **RQ3** | Is the gap from dataset size or feature shift? | Feature/quality shift dominates — not dataset size |
| **RQ4** | Does resolution help? | ✅ 224→324 px improves calcification F1 (0.49→0.54) |

---

## Key Results

### Heterogeneous FL (ResNet-50, Table 5)

| Method | CBIS AUC | VinDr AUC | Combined AUC |
|---|---|---|---|
| Centralized (CL) | 0.70 ± 0.02 | **0.97 ± 0.00** | **0.96 ± 0.00** |
| FedAvg | 0.62 ± 0.01 | 0.95 ± 0.00 | 0.93 ± 0.00 |
| FedProx | 0.61 ± 0.01 | 0.94 ± 0.01 | 0.92 ± 0.01 |
| SCAFFOLD | 0.48 ± 0.02 | 0.93 ± 0.01 | 0.90 ± 0.00 |
| FedBN | 0.53 ± 0.03 | 0.75 ± 0.21 | 0.64 ± 0.16 |

### Homogeneous FL — FL matches CL (Table 4)

| Domain | FL (FedAvg) AUC | CL AUC |
|---|---|---|
| VinDr (digital) | **0.97 ± 0.00** | 0.96 ± 0.00 |
| CBIS (film) | **0.75 ± 0.03** | 0.73 ± 0.01 |

### Size-Balancing Ablation (Table 7)

Size balancing: CBIS AUC **0.62 → 0.68** but VinDr AUC **0.95 → 0.83** — confirming feature/quality shift, not quantity, as the dominant bottleneck.

---

## Repository Structure

```
.
├── data/
│   ├── dataset.py               # PyTorch Dataset for CBIS-DDSM and VinDr-Mammo
│   ├── preprocessing.py         # ROI extraction, windowing, PNG export, BI-RADS mapping
│   ├── augmentation.py          # Random flip, rotation ±15°, Gaussian blur (p=0.2)
│   └── manifests/               # CSV manifests: dataset, roi_png_path, label
│       ├── cbis_train.csv
│       ├── cbis_test.csv
│       ├── vindr_train.csv
│       └── vindr_test.csv
│
├── models/
│   ├── backbones.py             # ResNet-50 and Swin V2-T with binary classification head
│   └── centralized.py          # Centralized learning training loop
│
├── federated/
│   ├── server.py                # Parameter-server: aggregation + round orchestration
│   ├── client.py                # Client: local training (E=3 epochs/round)
│   └── algorithms/
│       ├── fedavg.py            # FedAvg — sample-size-weighted averaging
│       ├── fedprox.py           # FedProx — proximal regularizer (μ=0.01)
│       ├── scaffold.py          # SCAFFOLD — control-variate drift correction
│       └── fedbn.py             # FedBN — client-specific BN statistics
│
├── evaluation/
│   ├── metrics.py               # AUROC, AP, ACC, F1, Precision, Recall, Prec@Rec=0.90
│   ├── bootstrap.py             # 1000-replicate bootstrap confidence intervals
│   └── plots.py                 # ROC curves, PR curves (Figures 3 & 4 from paper)
│
├── configs/
│   └── config.yaml              # All hyperparameters (SGD, lr=0.01, E=3, 100 rounds)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Dataset statistics (Table 1)
│   ├── 02_homogeneous_fl.ipynb        # RQ1: homogeneous controls (Table 4)
│   ├── 03_heterogeneous_fl.ipynb      # RQ2: film–digital FL (Table 5)
│   ├── 04_ablations.ipynb             # RQ3: size-balancing (Table 7)
│   └── 05_resolution_plots.ipynb      # RQ4: resolution ablation (Table 6) + Figs 3,4
│
├── figures/
│   ├── workflow.png             # Figure 1 — data pipeline
│   ├── roc_swin.png             # Figure 3 — Swin V2-T ROC/PR curves
│   └── roc_resnet.png           # Figure 4 — ResNet-50 ROC/PR curves
│
├── paper/
│   └── Ali_et_al_FrontiersDigHealth2026_FL_Mammography.pdf
│
├── run_experiment.py            # Single entry-point: run any experiment from CLI
├── requirements.txt
└── README.md
```

---

## Datasets

| Dataset | Type | Train+Val | Test | Malignant% |
|---|---|---|---|---|
| **CBIS-DDSM** | Scanned film (FL Client 1) | 2,864 | 1,403 | 41.2% / 39.0% |
| **VinDr-Mammo** | Full-field digital (FL Client 2) | 16,370 | 8,184 | 11.6% / 11.7% |
| **Combined** | — | 19,234 | 9,587 | 16.1% / 15.7% |

- **CBIS-DDSM**: [The Cancer Imaging Archive](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM)
- **VinDr-Mammo**: [PhysioNet](https://physionet.org/content/vindr-mammo/)

> BI-RADS mapping: BI-RADS 1–3 → benign (0), BI-RADS 4–6 → malignant (1). BI-RADS 0 and missing excluded.

---

## Installation

```bash
git clone https://github.com/yaqeenali/Federated-vs.-Centralized-Mammography-Classification-under-Film-Digital-Domain-Shift.git
cd Federated-vs.-Centralized-Mammography-Classification-under-Film-Digital-Domain-Shift
pip install -r requirements.txt
```

---

## Usage

### Prepare data manifests
```bash
python data/preprocessing.py \
    --cbis_dir  /data/cbis-ddsm \
    --vindr_dir /data/vindr-mammo \
    --output_dir data/manifests
```

### Centralized learning baseline
```bash
python run_experiment.py --mode centralized \
    --backbone resnet50 \
    --config configs/config.yaml
```

### Federated learning — all algorithms
```bash
# FedAvg (most stable per paper)
python run_experiment.py --mode federated --algorithm fedavg \
    --backbone resnet50 --config configs/config.yaml

# FedProx (μ=0.01)
python run_experiment.py --mode federated --algorithm fedprox \
    --backbone resnet50 --config configs/config.yaml

# SCAFFOLD
python run_experiment.py --mode federated --algorithm scaffold \
    --backbone resnet50 --config configs/config.yaml

# FedBN
python run_experiment.py --mode federated --algorithm fedbn \
    --backbone resnet50 --config configs/config.yaml
```

### Ablations
```bash
# Size-balancing ablation (Table 7)
python run_experiment.py --mode federated --algorithm fedavg \
    --backbone resnet50 --size_balance --config configs/config.yaml

# Resolution ablation: 224→324 px (Table 6)
python run_experiment.py --mode federated --algorithm fedavg \
    --backbone resnet50 --input_size 324 --config configs/config.yaml

# Homogeneous FL controls (Table 4)
python run_experiment.py --mode federated --algorithm fedavg \
    --backbone resnet50 --homogeneous cbis --config configs/config.yaml
python run_experiment.py --mode federated --algorithm fedavg \
    --backbone resnet50 --homogeneous vindr --config configs/config.yaml
```

### Evaluation with bootstrap CIs
```bash
python evaluation/metrics.py \
    --predictions_csv results/fedavg_resnet50/predictions.csv \
    --n_bootstrap 1000 \
    --output_dir results/fedavg_resnet50/
```

---

## Implementation Details

| Setting | Value |
|---|---|
| Optimizer | SGD, momentum=0.9, weight decay=1e-4 |
| Learning rate | 0.01 |
| Batch size | 32 |
| Local epochs per round (E) | 3 |
| FL rounds | 100 |
| Backbone initialization | ImageNet pretrained |
| Classification head | 2-logit FC layer |
| Input resolution (default) | 224 × 224 px |
| Resolution ablation | 324 × 324 px |
| Normalization | ImageNet stats: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] |
| Augmentation | Random H-flip (p=0.5), rotation ±15°, Gaussian blur (kernel=5, σ∈[0.1,1.5], p=0.2) |
| Mixed precision | torch.amp with gradient scaling |
| Model selection | Best validation AUROC checkpoint |
| Bootstrap replicates | 1,000 (95% percentile interval) |
| GPU | NVIDIA Tesla T4 (16 GB) |
| Seeds | 3 fixed seeds for Python, NumPy, PyTorch |

---

## Citation

```bibtex
@article{ali2026mammography,
  title     = {Performance of federated versus centralized learning for
               mammography classification across film--digital domain shift},
  author    = {Ali, Yaqeen and M{\"u}ller, Julia and Weinmann, Andreas
               and Gregori, Johannes},
  journal   = {Frontiers in Digital Health},
  volume    = {8},
  pages     = {1715858},
  year      = {2026},
  publisher = {Frontiers Media SA},
  doi       = {10.3389/fdgth.2026.1715858}
}
```

---

## Related Repositories

| Repository | Venue | Method |
|---|---|---|
| [tnbc-mri-radiomics](https://github.com/yaqeenali/tnbc-mri-radiomics) | IEEE CBMS 2025 | Radiomics + EasyEnsemble (TNBC) |
| [pcr-3dresnet-transformer](https://github.com/yaqeenali/pcr-3dresnet-transformer) | SPIE 2026 | 3D ResNet + Transformer (pCR) |
| [FL-XGBoost-pCR](https://github.com/yaqeenali/Explainableand-Fair-Federated-Learning-with-XGBoost-for-Predicting-Pathological-Complete-Response) | SPIE 2026 | Federated XGBoost + SHAP + Fairness |
| **This repo** | Frontiers Digital Health 2026 | FL vs CL mammography, 4 FL algorithms |

---

## Funding

Supported by the **Marie Skłodowska-Curie Doctoral Network** (HORIZON-MSCA-2021-DN-01-01) under Grant Agreement No. 101073222 (BosomShield) and **BMFTR** project number 01KD25015 (MICRATE).

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
The paper is open access under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
