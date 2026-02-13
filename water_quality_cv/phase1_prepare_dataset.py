#!/usr/bin/env python3
"""
PHASE 1 COMPLÈTE - Script unique optimisé
Prépare le dataset avec split et filtrage automatique
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import pandas as pd
from tqdm import tqdm


AUGMENTED_TYPES = ["GaussianBlur", "AveragedBlur"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def load_metadata(meta_path: Path) -> pd.DataFrame:
    """Charge et prépare le CSV de métadonnées"""
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df = pd.read_csv(meta_path)
    print(f"✅ Métadonnées chargées : {len(df)} clips")

    if "clip_id" in df.columns:
        id_col = "clip_id"
    else:
        for col in df.columns:
            if "id" in col.lower() or "video" in col.lower():
                id_col = col
                print(f"⚠️  Utilisation de '{id_col}' comme identifiant de clip")
                df["clip_id"] = df[id_col]
                break
        else:
            raise ValueError("Aucune colonne d'identifiant trouvée dans le CSV")

    df["clip_prefix"] = df["clip_id"].astype(str).str.strip().str[:8]

    return df


def find_images(root: Path, is_augmented: bool, aug_type: Optional[str] = None) -> List[Dict]:
    """Trouve toutes les images dans une arborescence"""
    records: List[Dict] = []

    if not root.exists():
        print(f"⚠️  Répertoire manquant : {root}")
        return records

    for gx_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        clip_prefix = gx_dir.name

        images_dir = gx_dir / "images"
        if not images_dir.exists():
            images_dir = gx_dir

        ann_dir = None
        for ann_name in ["annotations", "annotation"]:
            potential_ann = gx_dir / ann_name
            if potential_ann.exists():
                ann_dir = potential_ann
                break

        if ann_dir is None:
            ann_dir = images_dir

        for img_path in images_dir.iterdir():
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            ann_path = ann_dir / f"{img_path.stem}.json"

            records.append(
                {
                    "image_path": img_path,
                    "annotation_path": ann_path if ann_path.exists() else None,
                    "clip_prefix": clip_prefix,
                    "is_augmented": is_augmented,
                    "aug_type": aug_type or "",
                }
            )

    return records


def load_annotation_info(annotation_path: Optional[Path]) -> Tuple[bool, int, Optional[int], Optional[int]]:
    """Extrait les infos d'une annotation JSON"""
    if annotation_path is None or not annotation_path.exists():
        return False, 0, None, None

    try:
        with annotation_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        shapes = data.get("shapes", []) if isinstance(data, dict) else []
        num_instances = len(shapes) if isinstance(shapes, list) else 0
        width = data.get("imageWidth") if isinstance(data, dict) else None
        height = data.get("imageHeight") if isinstance(data, dict) else None

        return True, num_instances, width, height

    except Exception as exc:
        print(f"⚠️  Erreur lecture annotation {annotation_path.name}: {exc}")
        return False, 0, None, None


def get_image_size(image_path: Path) -> Tuple[Optional[int], Optional[int]]:
    """Lit les dimensions d'une image"""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None, None
        height, width = img.shape[:2]
        return width, height
    except Exception:
        return None, None


def split_by_clip_prefix(clip_prefixes: List[str], seed: int = 42) -> Dict[str, str]:
    """Crée le split train/val/test au niveau des clips"""
    unique_prefixes = sorted(set(clip_prefixes))
    if not unique_prefixes:
        return {}

    rnd = pd.Series(unique_prefixes).sample(frac=1.0, random_state=seed).tolist()
    n_total = len(rnd)

    # cas petit n (comme ici: 4 clips)
    if n_total >= 3:
        n_test = 1
        n_val = 1
        n_train = n_total - n_val - n_test
    else:
        # fallback: au moins 1 train, le reste val
        n_test = 0
        n_val = max(1, n_total - 1)
        n_train = 1

    train_prefixes = set(rnd[:n_train])
    val_prefixes = set(rnd[n_train : n_train + n_val])
    test_prefixes = set(rnd[n_train + n_val :])

    mapping = {}
    for cp in unique_prefixes:
        if cp in train_prefixes:
            mapping[cp] = "TRAIN"
        elif cp in val_prefixes:
            mapping[cp] = "VAL"
        else:
            mapping[cp] = "TEST"
    return mapping


def build_dataframe(dataset_root: Path, meta_df: pd.DataFrame) -> pd.DataFrame:
    """Construit le DataFrame complet avec toutes les images"""
    frames_root = dataset_root / "Frames"
    original_root = frames_root / "Original"
    augmented_root = frames_root / "Augmented"

    print(f"\n{'='*80}")
    print("COLLECTE DES IMAGES")
    print(f"{'='*80}")

    records: List[Dict] = []

    print("\n📸 Recherche des images originales...")
    orig_records = find_images(original_root, is_augmented=False, aug_type="")
    records.extend(orig_records)
    print(f"   Trouvé : {len(orig_records)} images originales")

    for aug_type in AUGMENTED_TYPES:
        print(f"\n🔄 Recherche des images {aug_type}...")
        aug_records = find_images(augmented_root / aug_type, is_augmented=True, aug_type=aug_type)
        records.extend(aug_records)
        print(f"   Trouvé : {len(aug_records)} images {aug_type}")

    print(f"\n✅ Total images trouvées : {len(records)}")

    print(f"\n{'='*80}")
    print("EXTRACTION DES MÉTADONNÉES")
    print(f"{'='*80}\n")

    rows: List[Dict] = []
    for rec in tqdm(records, desc="Traitement"):
        has_ann, num_instances, width, height = load_annotation_info(rec["annotation_path"])

        if width is None or height is None:
            w, h = get_image_size(rec["image_path"])
            width = width if width is not None else w
            height = height if height is not None else h

        rows.append(
            {
                "image_path": rec["image_path"],
                "annotation_path": rec["annotation_path"],
                "clip_prefix": rec["clip_prefix"],
                "is_augmented": rec["is_augmented"],
                "aug_type": rec["aug_type"],
                "has_annotation": has_ann,
                "num_instances": num_instances,
                "image_width": width,
                "image_height": height,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        print("❌ Aucune image trouvée !")
        return df

    print(f"\n{'='*80}")
    print("FUSION AVEC MÉTADONNÉES PHYSICOCHIMIQUES")
    print(f"{'='*80}")

    merged = df.merge(meta_df, on="clip_prefix", how="left", suffixes=("", "_meta"))

    matched = merged["temperature_C"].notna().sum() if "temperature_C" in merged.columns else 0
    print("\n✅ Fusion terminée :")
    print(f"   Total images : {len(merged)}")
    if len(merged):
        print(f"   Images enrichies : {matched} ({matched/len(merged)*100:.1f}%)")

    return merged


def finalize_and_save(df: pd.DataFrame, dataset_root: Path, output_dir: Path) -> None:
    """Finalise le dataset et sauvegarde"""
    output_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        print("❌ Aucune donnée à sauvegarder")
        return

    print(f"\n{'='*80}")
    print("CRÉATION DU SPLIT TRAIN/VAL/TEST")
    print(f"{'='*80}")

    clip_map = split_by_clip_prefix(df["clip_prefix"].tolist(), seed=42)
    df["split"] = df["clip_prefix"].map(clip_map)

    split_counts = df.groupby(["split", "is_augmented"]).size().unstack(fill_value=0)
    print("\n📊 Répartition avant filtrage :")
    print(split_counts)

    before_filter = len(df)
    mask = (df["is_augmented"] == False) | (df["split"] == "TRAIN")
    df = df.loc[mask].copy()
    filtered_out = before_filter - len(df)

    if filtered_out > 0:
        print(f"\n🗑️  {filtered_out} images augmentées filtrées (VAL/TEST)")

    df.loc[:, "image_path"] = df["image_path"].apply(
        lambda p: str(Path(p).relative_to(dataset_root)) if pd.notna(p) else ""
    )
    df.loc[:, "annotation_path"] = df["annotation_path"].apply(
        lambda p: str(Path(p).relative_to(dataset_root)) if pd.notna(p) and p else ""
    )

    columns = [
        "image_path",
        "annotation_path",
        "clip_prefix",
        "split",
        "is_augmented",
        "aug_type",
        "temperature_C",
        "pH",
        "DO_mgL",
        "turbidity_NTU",
        "lighting_mode",
        "resolution_px",
        "tank_id",
        "has_annotation",
        "num_instances",
        "image_width",
        "image_height",
    ]

    for col in columns:
        if col not in df.columns:
            df[col] = "" if col in {"annotation_path", "aug_type"} else pd.NA

    df = df[columns]

    csv_path = output_dir / "images_labels.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ CSV sauvegardé : {csv_path}")

    total_images = len(df)
    per_split = df["split"].value_counts(dropna=False).to_dict()
    per_split = {k: int(v) for k, v in per_split.items()}

    for key in ["TRAIN", "VAL", "TEST"]:
        per_split.setdefault(key, 0)

    num_aug = int(df["is_augmented"].sum())
    num_clips = df["clip_prefix"].nunique(dropna=True)
    merged_ok = int(df["temperature_C"].notna().sum()) if "temperature_C" in df.columns else 0
    merge_rate = float(merged_ok / total_images) if total_images else 0.0

    stats = {
        "total_images": total_images,
        "per_split": per_split,
        "num_augmented": num_aug,
        "num_clips": num_clips,
        "merge_rate": merge_rate,
    }

    stats_path = output_dir / "stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"✅ Statistiques sauvegardées : {stats_path}")

    print(f"\n{'='*80}")
    print("📊 RÉSUMÉ FINAL")
    print(f"{'='*80}")
    print(f"\n  Total images      : {total_images}")
    print(f"  Images augmentées : {num_aug}")
    print(f"  Clips uniques     : {num_clips}")
    print(f"  Taux de fusion    : {merge_rate*100:.1f}%")
    print(f"\n  TRAIN : {per_split['TRAIN']} images")
    print(f"  VAL   : {per_split['VAL']} images")
    print(f"  TEST  : {per_split['TEST']} images")


def main() -> None:
    """Point d'entrée principal"""
    print("=" * 80)
    print("PHASE 1 : PRÉPARATION COMPLÈTE DU DATASET")
    print("=" * 80)

    project_root = Path(__file__).resolve().parent
    dataset_root = None
    for parent in [project_root, *project_root.parents]:
        candidate = parent / "Tilapia RAS Dataset"
        if candidate.exists():
            dataset_root = candidate
            break
    if dataset_root is None:
        dataset_root = project_root / "Tilapia RAS Dataset"
    meta_path = dataset_root / "Documentation" / "meta_tilapia_set.csv"
    output_dir = project_root / "processed"

    print("\n📁 Configuration :")
    print(f"   Projet      : {project_root}")
    print(f"   Dataset     : {dataset_root}")
    print(f"   Métadonnées : {meta_path}")
    print(f"   Sortie      : {output_dir}")

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset introuvable : {dataset_root}")

    if not meta_path.exists():
        raise FileNotFoundError(f"CSV métadonnées introuvable : {meta_path}")

    meta_df = load_metadata(meta_path)
    df = build_dataframe(dataset_root, meta_df)
    finalize_and_save(df, dataset_root, output_dir)

    print(f"\n{'='*80}")
    print("✅ PHASE 1 TERMINÉE - DATASET PRÊT POUR L'ENTRAÎNEMENT")
    print(f"{'='*80}")
    print("\n💡 Fichier généré : processed/images_labels.csv")
    print("   Utilisez ce fichier pour créer le PyTorch Dataset\n")


if __name__ == "__main__":
    main()
