#!/usr/bin/env python3
"""
PHASE 1 CORRIGÉE - Conforme PV P2M
- Split aléatoire stratifié
- Aucune dépendance aux clips pour le split
- Stats détaillées
- Vérifications automatiques
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


AUGMENTED_TYPES = ["GaussianBlur", "AveragedBlur"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


# =========================================================
# LOAD METADATA
# =========================================================
def load_metadata(meta_path: Path) -> pd.DataFrame:
    df = pd.read_csv(meta_path)

    if "clip_id" in df.columns:
        df["clip_prefix"] = df["clip_id"].astype(str).str[:8]

    return df


# =========================================================
# FIND IMAGES
# =========================================================
def find_images(root: Path, is_augmented: bool, aug_type: Optional[str] = None) -> List[Dict]:
    records = []

    for gx_dir in root.iterdir():
        if not gx_dir.is_dir():
            continue

        images_dir = gx_dir / "images"
        if not images_dir.exists():
            images_dir = gx_dir

        for img_path in images_dir.iterdir():
            if img_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            records.append({
                "image_path": img_path,
                "clip_prefix": gx_dir.name,  # gardé mais NON utilisé pour split
                "is_augmented": is_augmented,
                "aug_type": aug_type or "",
            })

    return records


# =========================================================
# BUILD DATAFRAME
# =========================================================
def build_dataframe(dataset_root: Path, meta_df: pd.DataFrame) -> pd.DataFrame:
    records = []

    records += find_images(dataset_root / "Frames/Original", False)

    for aug in AUGMENTED_TYPES:
        records += find_images(dataset_root / f"Frames/Augmented/{aug}", True, aug)

    rows = []
    for r in tqdm(records):
        rows.append({
            "image_path": r["image_path"],
            "clip_prefix": r["clip_prefix"],
            "is_augmented": r["is_augmented"],
            "aug_type": r["aug_type"],
        })

    df = pd.DataFrame(rows)

    df = df.merge(meta_df, on="clip_prefix", how="left")

    return df


# =========================================================
# SPLIT RANDOM STRATIFIED
# =========================================================
def stratified_random_split(df: pd.DataFrame, seed=42):

    df_orig = df[df["is_augmented"] == False]

    y = df_orig["turbidity_NTU"]

    train_idx, temp_idx = train_test_split(
        df_orig.index,
        test_size=0.4,
        stratify=y,
        random_state=seed
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=df_orig.loc[temp_idx, "turbidity_NTU"],
        random_state=seed
    )

    split = pd.Series(index=df.index, dtype=str)

    split.loc[train_idx] = "TRAIN"
    split.loc[val_idx] = "VAL"
    split.loc[test_idx] = "TEST"

    # Augmentées → TRAIN uniquement
    split.loc[df["is_augmented"]] = "TRAIN"

    return split


# =========================================================
# STATS
# =========================================================
def _compute_distribution_table(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    subset = df[df["split"].isin(["TRAIN", "VAL", "TEST"])][["split", value_col]].dropna()

    grouped = (
        subset.groupby("split")[value_col]
        .value_counts(normalize=True)
        .mul(100)
        .rename("pct")
        .reset_index()
    )

    table = grouped.pivot(index=value_col, columns="split", values="pct").fillna(0)
    table = table.reindex(columns=["TRAIN", "VAL", "TEST"], fill_value=0)

    return table.sort_index()


def _print_distribution_table(df: pd.DataFrame, value_col: str, title: str):
    if value_col not in df.columns:
        print(f"\n⚠️ Colonne '{value_col}' introuvable, stats ignorées.")
        return

    table = _compute_distribution_table(df, value_col)

    if table.empty:
        print(f"\n⚠️ Aucune donnée disponible pour {title}.")
        return

    print(f"\n📊 Distribution de {title} (%)")
    print(table.to_string(float_format=lambda x: f"{x:.2f}"))


def compute_turbidity_stats(df):
    _print_distribution_table(df, "turbidity_NTU", "la turbidité")


def _find_do_column(df: pd.DataFrame) -> Optional[str]:
    do_candidates = [
        "DO",
        "do",
        "DO_mgL",
        "do_mgL",
        "DO_mg_L",
        "DO_mg_l",
        "do_mg_l",
        "dissolved_oxygen",
        "dissolved_oxygen_mg_l",
    ]

    for col in do_candidates:
        if col in df.columns:
            return col

    for col in df.columns:
        col_low = col.lower()
        if col_low == "do" or ("dissolved" in col_low and "oxygen" in col_low):
            return col

    # Matching tolérant: ignore underscore/tiret/espace et casse
    normalized_map = {
        "".join(ch for ch in col.lower() if ch.isalnum()): col for col in df.columns
    }
    for key in ["domgl", "dissolvedoxygen", "dissolvedoxygenmgl"]:
        if key in normalized_map:
            return normalized_map[key]

    return None


def compute_do_stats(df: pd.DataFrame):
    do_col = _find_do_column(df)
    if do_col is None:
        print("\n⚠️ Colonne DO introuvable, distribution DO ignorée.")
        return

    _print_distribution_table(df, do_col, "la DO")


def check_all_turbidity_present(df):
    all_vals = set(df["turbidity_NTU"].dropna().unique())

    for split in ["TRAIN", "VAL", "TEST"]:
        vals = set(df[df["split"] == split]["turbidity_NTU"].dropna().unique())

        if vals == all_vals:
            print(f"✅ {split} OK")
        else:
            print(f"❌ {split} incomplet")


# =========================================================
# SAVE
# =========================================================
def finalize(df: pd.DataFrame, output_dir: Path):

    df["split"] = stratified_random_split(df)

    compute_turbidity_stats(df)
    compute_do_stats(df)
    check_all_turbidity_present(df)

    # Filtrer augmentées hors TRAIN
    df = df[(df["is_augmented"] == False) | (df["split"] == "TRAIN")]

    df["image_path"] = df["image_path"].astype(str)

    output_dir.mkdir(exist_ok=True)

    df.to_csv(output_dir / "images_labels.csv", index=False)

    print("\n✅ Dataset prêt")


# =========================================================
# MAIN
# =========================================================
def main():
    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    root = script_dir.parent.parent / "Tilapia RAS Dataset"
    meta = root / "Documentation/meta_tilapia_set.csv"

    meta_df = load_metadata(meta)
    df = build_dataframe(root, meta_df)

    finalize(df, script_dir / "processed")


if __name__ == "__main__":
    main()