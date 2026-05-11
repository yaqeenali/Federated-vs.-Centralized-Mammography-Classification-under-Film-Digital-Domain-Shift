"""
Preprocessing pipeline for CBIS-DDSM and VinDr-Mammo.

CBIS-DDSM (scanned film):
    - Lesion-centered ROI crops used as released
    - Already PNG; resize only at training time

VinDr-Mammo (full-field digital):
    - Apply provided bounding boxes to original DICOMs
    - Export ROIs to 8-bit PNG after windowing full dynamic range
    - Apply DICOM photometric VOI LUT when present

Label harmonization (Section 2.2.2):
    VinDr BI-RADS 1-3  →  benign (0)
    VinDr BI-RADS 4-6  →  malignant (1)
    BI-RADS 0 / missing → excluded

Usage:
    python data/preprocessing.py \
        --cbis_dir  /data/cbis-ddsm \
        --vindr_dir /data/vindr-mammo \
        --output_dir data/manifests

Reference:
    Ali et al., Front. Digit. Health 8:1715858 (2026)
    doi: 10.3389/fdgth.2026.1715858
"""

import os
import argparse
import numpy as np
import pandas as pd
import pydicom
import cv2
from pathlib import Path
from tqdm import tqdm


# --------------------------------------------------------------------------- #
#  Constants                                                                   #
# --------------------------------------------------------------------------- #

# BI-RADS → binary label mapping (Section 2.2.2)
BIRADS_MAP = {
    1: 0, 2: 0, 3: 0,   # benign
    4: 1, 5: 1, 6: 1,   # malignant
    0: None,             # excluded
}

# ImageNet normalisation stats (used at training time, not here)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- #
#  VinDr-Mammo: DICOM → PNG with VOI LUT windowing                           #
# --------------------------------------------------------------------------- #

def apply_voi_lut(dicom_ds):
    """
    Apply photometric VOI LUT when present (Section 2.2.1).
    Falls back to full-range window if no VOI LUT metadata.

    Returns float32 array normalised to [0, 255].
    """
    pixel_array = dicom_ds.pixel_array.astype(np.float32)

    # Try VOI LUT (window center / width)
    if hasattr(dicom_ds, "WindowCenter") and hasattr(dicom_ds, "WindowWidth"):
        wc = float(dicom_ds.WindowCenter) if not isinstance(
            dicom_ds.WindowCenter, pydicom.multival.MultiValue
        ) else float(dicom_ds.WindowCenter[0])
        ww = float(dicom_ds.WindowWidth) if not isinstance(
            dicom_ds.WindowWidth, pydicom.multival.MultiValue
        ) else float(dicom_ds.WindowWidth[0])

        lo = wc - ww / 2
        hi = wc + ww / 2
    else:
        # Full dynamic range fallback
        lo = pixel_array.min()
        hi = pixel_array.max()

    # Clip and rescale to 8-bit
    clipped = np.clip(pixel_array, lo, hi)
    if hi > lo:
        normalised = (clipped - lo) / (hi - lo) * 255.0
    else:
        normalised = np.zeros_like(clipped)

    # Handle photometric interpretation (MONOCHROME1 = inverted)
    if hasattr(dicom_ds, "PhotometricInterpretation"):
        if dicom_ds.PhotometricInterpretation == "MONOCHROME1":
            normalised = 255.0 - normalised

    return normalised.astype(np.uint8)


def extract_vindr_roi(dicom_path, bbox, output_png_path):
    """
    Extract a lesion ROI from a VinDr DICOM using a bounding box.

    Args:
        dicom_path:     path to .dicom file
        bbox:           dict with keys x_min, y_min, x_max, y_max
        output_png_path: where to save the 8-bit PNG
    """
    ds       = pydicom.dcmread(str(dicom_path))
    img_8bit = apply_voi_lut(ds)

    # Convert to 3-channel (required for ImageNet-pretrained models)
    img_rgb = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2RGB)

    # Crop ROI
    x1, y1 = int(bbox["x_min"]), int(bbox["y_min"])
    x2, y2 = int(bbox["x_max"]), int(bbox["y_max"])
    roi     = img_rgb[y1:y2, x1:x2]

    if roi.size == 0:
        raise ValueError(f"Empty ROI for {dicom_path}: bbox={bbox}")

    output_png_path = Path(output_png_path)
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_png_path), cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))


# --------------------------------------------------------------------------- #
#  Build manifest CSVs                                                         #
# --------------------------------------------------------------------------- #

def build_cbis_manifest(cbis_dir, output_dir):
    """
    Build CBIS-DDSM manifest CSV.
    Expects the standard CBIS-DDSM directory structure with
    calc_case_description_train_set.csv and mass_case_description_train_set.csv.

    Returns DataFrame with columns: patient_id, roi_png_path, label, lesion_type, split
    """
    cbis_dir   = Path(cbis_dir)
    output_dir = Path(output_dir)

    rows = []
    for split in ["train", "test"]:
        for ltype in ["calc", "mass"]:
            csv_path = cbis_dir / f"{ltype}_case_description_{split}_set.csv"
            if not csv_path.exists():
                print(f"[SKIP] {csv_path} not found")
                continue

            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                # CBIS uses pathology column: MALIGNANT / BENIGN / BENIGN_WITHOUT_CALLBACK
                pathology = str(row.get("pathology", "")).upper()
                if "MALIGNANT" in pathology:
                    label = 1
                elif "BENIGN" in pathology:
                    label = 0
                else:
                    continue

                # Locate the pre-cropped ROI PNG
                img_path = row.get("cropped image file path", "")
                if pd.isna(img_path) or not img_path:
                    continue

                rows.append({
                    "patient_id":   row.get("patient_id", ""),
                    "roi_png_path": str(cbis_dir / img_path.strip()),
                    "label":        label,
                    "lesion_type":  ltype,
                    "dataset":      "cbis",
                    "split":        split,
                })

    df_out = pd.DataFrame(rows)
    out_path = output_dir / "cbis_manifest.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"CBIS manifest: {len(df_out)} entries → {out_path}")
    return df_out


def build_vindr_manifest(vindr_dir, output_dir, roi_output_dir=None):
    """
    Build VinDr-Mammo manifest CSV.
    Expects finding_annotations.csv (with StudyInstanceUID, SeriesInstanceUID,
    xmin, ymin, xmax, ymax, finding_birads).

    Extracts ROI PNGs from DICOMs and maps BI-RADS to binary labels.
    """
    vindr_dir      = Path(vindr_dir)
    output_dir     = Path(output_dir)
    roi_output_dir = Path(roi_output_dir) if roi_output_dir else output_dir / "vindr_rois"

    annot_path = vindr_dir / "finding_annotations.csv"
    if not annot_path.exists():
        print(f"[ERROR] {annot_path} not found")
        return pd.DataFrame()

    df_annot = pd.read_csv(annot_path)
    rows     = []

    for idx, row in tqdm(df_annot.iterrows(), total=len(df_annot),
                         desc="Extracting VinDr ROIs"):
        birads = row.get("finding_birads", 0)
        try:
            birads_int = int(str(birads).replace("BI-RADS ", ""))
        except (ValueError, TypeError):
            continue

        label = BIRADS_MAP.get(birads_int)
        if label is None:
            continue    # BI-RADS 0 or missing — exclude

        study_uid  = row["study_id"]
        series_uid = row.get("series_id", "")
        image_id   = row.get("image_id", f"{study_uid}_{idx}")

        # Find the DICOM
        dicom_path = vindr_dir / "images" / study_uid / f"{image_id}.dicom"
        if not dicom_path.exists():
            # Try common alternative paths
            dicom_path = vindr_dir / "images" / study_uid / f"{image_id}.dcm"
        if not dicom_path.exists():
            continue

        bbox = {
            "x_min": row["xmin"], "y_min": row["ymin"],
            "x_max": row["xmax"], "y_max": row["ymax"],
        }

        png_name   = f"{image_id}_{idx}.png"
        png_path   = roi_output_dir / study_uid / png_name

        try:
            extract_vindr_roi(dicom_path, bbox, png_path)
        except Exception as e:
            print(f"[ERROR] {image_id}: {e}")
            continue

        split = "train" if row.get("split", "training") == "training" else "test"
        rows.append({
            "patient_id":   study_uid,
            "roi_png_path": str(png_path),
            "label":        label,
            "birads":       birads_int,
            "dataset":      "vindr",
            "split":        split,
        })

    df_out   = pd.DataFrame(rows)
    out_path = output_dir / "vindr_manifest.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"VinDr manifest: {len(df_out)} entries → {out_path}")
    return df_out


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess CBIS-DDSM + VinDr-Mammo")
    parser.add_argument("--cbis_dir",   default=None)
    parser.add_argument("--vindr_dir",  default=None)
    parser.add_argument("--output_dir", default="data/manifests")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.cbis_dir:
        build_cbis_manifest(args.cbis_dir, args.output_dir)
    if args.vindr_dir:
        build_vindr_manifest(args.vindr_dir, args.output_dir)
