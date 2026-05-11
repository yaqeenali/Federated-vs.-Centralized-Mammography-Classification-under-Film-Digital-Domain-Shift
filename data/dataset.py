"""
PyTorch Dataset for CBIS-DDSM and VinDr-Mammo mammography ROI classification.

Preprocessing (Section 2.2.3 & 2.2.4):
    - Resize to 224×224 px (or 324×324 for resolution ablation)
    - Normalize with ImageNet stats: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
    - Augmentation (training only):
        * Random horizontal flip (p=0.5)
        * Random rotation ±15°
        * Random Gaussian blur (kernel=5, σ∈[0.1,1.5], p=0.2)

Reference:
    Ali et al., Front. Digit. Health 8:1715858 (2026)
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# --------------------------------------------------------------------------- #
#  Transforms                                                                  #
# --------------------------------------------------------------------------- #

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(split="train", input_size=224):
    """
    Build torchvision transform pipeline matching Section 2.2.3-4.

    Args:
        split:      'train' (with augmentation) or 'val'/'test' (no augmentation)
        input_size: 224 (default) or 324 (resolution ablation, Table 6)
    """
    base = [
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    if split == "train":
        aug = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.2),
        ]
        # Augmentation before resize/normalize
        return T.Compose(aug + base)
    return T.Compose(base)


# --------------------------------------------------------------------------- #
#  Dataset                                                                     #
# --------------------------------------------------------------------------- #

class MammographyDataset(Dataset):
    """
    Unified dataset for CBIS-DDSM and VinDr-Mammo.

    Manifest CSV columns: roi_png_path, label, dataset (cbis|vindr), split

    Args:
        manifest_csv:  path to manifest CSV (or list of paths to combine)
        split:         'train' | 'val' | 'test'
        input_size:    224 or 324 px
        domains:       list of domains to include ['cbis', 'vindr'] or None (all)
        size_balance:  if True, downsample VinDr benign to match CBIS size
                       (size-balancing ablation, Table 7)
    """

    def __init__(self, manifest_csv, split="train", input_size=224,
                 domains=None, size_balance=False, seed=42):
        super().__init__()
        self.split      = split
        self.transform  = get_transforms(split, input_size)

        # Load manifest(s)
        if isinstance(manifest_csv, (list, tuple)):
            df = pd.concat([pd.read_csv(p) for p in manifest_csv], ignore_index=True)
        else:
            df = pd.read_csv(manifest_csv)

        # Filter by split
        if "split" in df.columns:
            df = df[df["split"] == split].reset_index(drop=True)

        # Filter by domain
        if domains is not None:
            df = df[df["dataset"].isin(domains)].reset_index(drop=True)

        # Size-balancing ablation (Section 2.7, Table 7):
        # Downsample VinDr benign cases to match CBIS size
        # All VinDr malignant cases retained; only benign subsampled
        if size_balance and "vindr" in (domains or df["dataset"].unique()):
            df = self._apply_size_balance(df, seed)

        # Drop missing files
        df = df[df["roi_png_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
        self.df = df

        pos = (df["label"] == 1).sum()
        neg = (df["label"] == 0).sum()
        print(f"[{split}] {len(df)} samples | "
              f"benign={neg} ({100*neg/max(len(df),1):.1f}%) "
              f"malignant={pos} ({100*pos/max(len(df),1):.1f}%)"
              + (f" | domains={domains}" if domains else ""))

    def _apply_size_balance(self, df, seed):
        """
        Size-balancing ablation: match VinDr size to CBIS.
        All VinDr malignant retained; VinDr benign subsampled.
        Paper Table 2: CBIS=2,301 | VinDr balanced=2,434
        """
        cbis_size = len(df[df["dataset"] == "cbis"])
        rng       = np.random.default_rng(seed)

        vindr_mal = df[(df["dataset"] == "vindr") & (df["label"] == 1)]
        vindr_ben = df[(df["dataset"] == "vindr") & (df["label"] == 0)]
        cbis_df   = df[df["dataset"] == "cbis"]

        # Target size for VinDr = cbis_size (all malignant kept)
        n_vindr_ben = max(0, cbis_size - len(vindr_mal))
        n_vindr_ben = min(n_vindr_ben, len(vindr_ben))
        vindr_ben_sub = vindr_ben.sample(n=n_vindr_ben, random_state=seed)

        balanced = pd.concat([cbis_df, vindr_mal, vindr_ben_sub], ignore_index=True)
        print(f"[size-balance] CBIS={len(cbis_df)} VinDr={len(vindr_mal)+len(vindr_ben_sub)} "
              f"(mal={len(vindr_mal)}, ben={len(vindr_ben_sub)})")
        return balanced.sample(frac=1, random_state=seed).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = Image.open(row["roi_png_path"]).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return {
            "image":    image,
            "label":    label,
            "dataset":  row.get("dataset", "unknown"),
            "filepath": row["roi_png_path"],
        }


# --------------------------------------------------------------------------- #
#  DataLoader factories                                                        #
# --------------------------------------------------------------------------- #

def build_centralized_loaders(manifest_csvs, input_size=224,
                               batch_size=32, num_workers=4, seed=42):
    """Pooled train/val/test loaders for centralized learning."""
    train = MammographyDataset(manifest_csvs, "train", input_size)
    val   = MammographyDataset(manifest_csvs, "val",   input_size)
    test  = MammographyDataset(manifest_csvs, "test",  input_size)

    return {
        "train": DataLoader(train, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True),
        "val":   DataLoader(val,   batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
        "test":  DataLoader(test,  batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
    }


def build_federated_client_loaders(manifest_csvs, domain,
                                   input_size=224, batch_size=32,
                                   num_workers=4, size_balance=False, seed=42):
    """Per-client loaders for one FL site (cbis or vindr)."""
    train = MammographyDataset(manifest_csvs, "train", input_size,
                               domains=[domain], size_balance=size_balance, seed=seed)
    val   = MammographyDataset(manifest_csvs, "val",   input_size,
                               domains=[domain], size_balance=False, seed=seed)
    test  = MammographyDataset(manifest_csvs, "test",  input_size,
                               domains=[domain], size_balance=False, seed=seed)

    return {
        "train": DataLoader(train, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True),
        "val":   DataLoader(val,   batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
        "test":  DataLoader(test,  batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
    }
