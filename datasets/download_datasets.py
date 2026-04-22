"""
Download and preprocess healthcare benchmark datasets for HPO study.

Datasets:
  1. Pima Indians Diabetes            (768 x 8,  binary)   -- endocrinology
  2. Indian Liver Patient Dataset     (583 x 10, binary)   -- hepatology

Usage (from the datasets/ folder):
    pip install pandas numpy requests
    python download_datasets.py

Outputs (in the same folder):
    pima_diabetes_raw.csv
    pima_diabetes_clean.csv      <- use this for modeling
    ilpd_raw.csv
    ilpd_clean.csv               <- use this for modeling
    dataset_manifest.json        <- provenance + SHA-256 hashes for reproducibility

Author: Saad (shroub@nvidia.com)
License of downloader: MIT. Datasets retain their original licenses (see dataset_card.md).
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:
    sys.exit("Please install requests:  pip install requests")

# --------------------------------------------------------------------------- #
# Source URLs (tried in order; first success wins)                             #
# --------------------------------------------------------------------------- #

PIMA_SOURCES = [
    # Jason Brownlee's dataset mirror; headerless, integer target
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
    # npradaschnor mirror; has a header row
    "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv",
]

PIMA_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]

ILPD_SOURCES = [
    # UCI canonical filename (case-sensitive on some mirrors)
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv",
    # Common mirror
    "https://raw.githubusercontent.com/aniruddhachoudhury/Indian-Liver-Patient-Dataset/master/indian_liver_patient.csv",
]

ILPD_COLUMNS = [
    "Age", "Gender", "TotalBilirubin", "DirectBilirubin",
    "AlkalinePhosphatase", "Alamine_Aminotransferase", "Aspartate_Aminotransferase",
    "TotalProtein", "Albumin", "AG_Ratio", "Selector",
]

# --------------------------------------------------------------------------- #
# Download helpers                                                             #
# --------------------------------------------------------------------------- #


def _try_fetch(url: str, timeout: int = 30) -> Optional[bytes]:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "hpo-study/1.0"})
        r.raise_for_status()
        return r.content
    except Exception as e:  # pragma: no cover -- network vagaries
        print(f"  [WARN] {url} -> {e.__class__.__name__}: {e}")
        return None


def fetch_first_success(urls: list[str], label: str) -> bytes:
    for url in urls:
        print(f"[{label}] fetching {url}")
        data = _try_fetch(url)
        if data is not None:
            print(f"[{label}] OK ({len(data)} bytes)")
            return data
    raise RuntimeError(f"All sources failed for {label}. Download manually and place the CSV here.")


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


# --------------------------------------------------------------------------- #
# Dataset-specific preprocessing                                               #
# --------------------------------------------------------------------------- #


def preprocess_pima(raw_bytes: bytes) -> pd.DataFrame:
    """
    Pima preprocessing:
      - Attach column headers.
      - Replace biologically-impossible zeros in
        Glucose, BloodPressure, SkinThickness, Insulin, BMI with NaN.
      - Impute NaN with per-column median computed within the training split
        NOTE: This function performs *full-data* median imputation. For
        leak-free evaluation do median imputation inside your CV fold.
      - Target column is integer 0/1.
    """
    from io import BytesIO

    # Detect whether first line is a header by sniffing digits
    first_line = raw_bytes.splitlines()[0].decode("utf-8", errors="replace")
    has_header = not first_line.split(",")[0].strip().lstrip("-").isdigit()

    df = pd.read_csv(
        BytesIO(raw_bytes),
        header=0 if has_header else None,
        names=PIMA_COLUMNS if not has_header else None,
    )
    # If the mirror uses a different header, rename to our canonical set
    if has_header:
        # Canonicalize common variants
        rename_map = {c: PIMA_COLUMNS[i] for i, c in enumerate(df.columns)}
        df = df.rename(columns=rename_map)

    assert df.shape == (768, 9), f"Pima expected (768,9), got {df.shape}"

    # Columns where zero is biologically implausible
    zero_as_nan = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df_clean = df.copy()
    for col in zero_as_nan:
        df_clean[col] = df_clean[col].replace(0, np.nan)

    print(f"[Pima] missing value counts after zero->NaN:")
    print(df_clean[zero_as_nan].isna().sum().to_string())

    # Median imputation (NOTE: for formal evaluation, do this inside CV folds)
    medians = df_clean[zero_as_nan].median()
    df_clean[zero_as_nan] = df_clean[zero_as_nan].fillna(medians)

    # Confirm target coding
    df_clean["Outcome"] = df_clean["Outcome"].astype(int)

    return df_clean


def preprocess_ilpd(raw_bytes: bytes) -> pd.DataFrame:
    """
    ILPD preprocessing:
      - Attach column headers (UCI file is headerless).
      - Encode Gender: Male -> 1, Female -> 0.
      - Recode Selector: original is 1 = liver patient, 2 = non-patient.
        We map to {1: 1, 2: 0} so target is 1 = disease present.
      - 4 rows have NaN in AG_Ratio -> median imputation.
    """
    from io import BytesIO

    # ILPD has no header; detect if first cell is numeric
    first_line = raw_bytes.splitlines()[0].decode("utf-8", errors="replace")
    has_header = not first_line.split(",")[0].strip().lstrip("-").isdigit()

    df = pd.read_csv(
        BytesIO(raw_bytes),
        header=0 if has_header else None,
        names=ILPD_COLUMNS if not has_header else None,
    )
    if has_header:
        rename_map = {c: ILPD_COLUMNS[i] for i, c in enumerate(df.columns)}
        df = df.rename(columns=rename_map)

    # Expect 583 rows
    assert df.shape == (583, 11), f"ILPD expected (583,11), got {df.shape}"

    # Encode Gender
    df["Gender"] = df["Gender"].astype(str).str.strip().map({"Male": 1, "Female": 0})
    assert df["Gender"].notna().all(), "Unrecognized Gender value(s) encountered."

    # AG_Ratio has a handful of NaN -- impute with median
    n_missing_ag = df["AG_Ratio"].isna().sum()
    print(f"[ILPD] AG_Ratio missing: {n_missing_ag} -> imputing with median")
    df["AG_Ratio"] = df["AG_Ratio"].fillna(df["AG_Ratio"].median())

    # Recode target: 1 = liver patient -> 1 (disease), 2 = non-patient -> 0
    df["Selector"] = df["Selector"].map({1: 1, 2: 0}).astype(int)
    df = df.rename(columns={"Selector": "Outcome"})

    return df


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def main() -> None:
    here = Path(__file__).resolve().parent
    print(f"Output folder: {here}\n")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "datasets": {},
    }

    # ---- Pima ------------------------------------------------------------- #
    pima_bytes = fetch_first_success(PIMA_SOURCES, "Pima")
    (here / "pima_diabetes_raw.csv").write_bytes(pima_bytes)
    pima_clean = preprocess_pima(pima_bytes)
    pima_clean.to_csv(here / "pima_diabetes_clean.csv", index=False)
    pima_hash = sha256(pima_bytes)
    print(f"[Pima] raw SHA-256: {pima_hash}")
    print(f"[Pima] clean shape: {pima_clean.shape}")
    print(f"[Pima] class balance: {pima_clean['Outcome'].value_counts().to_dict()}\n")

    manifest["datasets"]["pima_diabetes"] = {
        "rows": int(pima_clean.shape[0]),
        "features": int(pima_clean.shape[1] - 1),
        "target": "Outcome",
        "class_balance": {str(k): int(v) for k, v in pima_clean["Outcome"].value_counts().items()},
        "raw_sha256": pima_hash,
        "raw_file": "pima_diabetes_raw.csv",
        "clean_file": "pima_diabetes_clean.csv",
    }

    # ---- ILPD ------------------------------------------------------------- #
    ilpd_bytes = fetch_first_success(ILPD_SOURCES, "ILPD")
    (here / "ilpd_raw.csv").write_bytes(ilpd_bytes)
    ilpd_clean = preprocess_ilpd(ilpd_bytes)
    ilpd_clean.to_csv(here / "ilpd_clean.csv", index=False)
    ilpd_hash = sha256(ilpd_bytes)
    print(f"[ILPD] raw SHA-256: {ilpd_hash}")
    print(f"[ILPD] clean shape: {ilpd_clean.shape}")
    print(f"[ILPD] class balance: {ilpd_clean['Outcome'].value_counts().to_dict()}\n")

    manifest["datasets"]["ilpd"] = {
        "rows": int(ilpd_clean.shape[0]),
        "features": int(ilpd_clean.shape[1] - 1),
        "target": "Outcome",
        "class_balance": {str(k): int(v) for k, v in ilpd_clean["Outcome"].value_counts().items()},
        "raw_sha256": ilpd_hash,
        "raw_file": "ilpd_raw.csv",
        "clean_file": "ilpd_clean.csv",
    }

    # ---- Manifest --------------------------------------------------------- #
    with (here / "dataset_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print("Wrote dataset_manifest.json")
    print("\nDONE. You now have pima_diabetes_clean.csv and ilpd_clean.csv ready to feed into your HPO pipeline.")


if __name__ == "__main__":
    main()
