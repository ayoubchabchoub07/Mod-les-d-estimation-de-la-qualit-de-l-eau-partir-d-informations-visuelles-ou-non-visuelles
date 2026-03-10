"""
PyTorch Dataset pour l'estimation de qualité d'eau
Charge les images et labels depuis le CSV généré en Phase 1
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from preprocess import UnderwaterPreprocessor, create_preprocessor


class WaterQualityDataset(Dataset):
    """
    Dataset PyTorch pour prédiction multi-tâches de qualité d'eau

    Charge :
    - Images prétraitées
    - 4 labels de régression : turbidité, pH, température, DO
    """

    def __init__(
        self,
        csv_path: Path,
        dataset_root: Path,
        split: str = "TRAIN",
        preprocessor: Optional[UnderwaterPreprocessor] = None,
        transform: Optional[Callable] = None,
        target_params: Optional[list] = None,
    ):
        """
        Args:
            csv_path: Chemin vers images_labels.csv
            dataset_root: Racine du dataset (pour résoudre les chemins relatifs)
            split: "TRAIN", "VAL" ou "TEST"
            preprocessor: Instance de UnderwaterPreprocessor
            transform: Transformations Albumentations additionnelles (optionnel)
            target_params: Liste des paramètres à prédire (défaut: tous)
        """
        self.dataset_root = Path(dataset_root)
        self.split = split.upper()
        self.preprocessor = preprocessor
        self.transform = transform

        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV introuvable : {csv_path}")

        df = pd.read_csv(csv_path)

        if self.split not in df["split"].unique():
            raise ValueError(
                f"Split '{self.split}' introuvable dans le CSV. "
                f"Splits disponibles : {df['split'].unique().tolist()}"
            )

        self.df = df[df["split"] == self.split].reset_index(drop=True)

        print(f"✅ {self.split} Dataset chargé : {len(self.df)} images")
        n_aug = int(self.df["is_augmented"].sum())
        if len(self.df) > 0:
            print(f"   Images augmentées : {n_aug} ({n_aug/len(self.df)*100:.1f}%)")

        if target_params is None:
            self.target_params = ["turbidity_NTU", "pH", "temperature_C", "DO_mgL"]
        else:
            self.target_params = target_params

        for param in self.target_params:
            if param not in self.df.columns:
                raise ValueError(f"Paramètre '{param}' introuvable dans le CSV")

        for param in self.target_params:
            n_missing = int(self.df[param].isna().sum())
            if n_missing > 0 and len(self.df) > 0:
                print(
                    f"   ⚠️  {param}: {n_missing} valeurs manquantes "
                    f"({n_missing/len(self.df)*100:.1f}%)"
                )

        if self.preprocessor is None:
            self.preprocessor = create_preprocessor()
            print("   ℹ️  Preprocessor par défaut créé")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        row = self.df.iloc[idx]

        image_rel_path = row["image_path"]
        image_path = self.dataset_root / image_rel_path

        if not image_path.exists():
            raise FileNotFoundError(f"Image introuvable : {image_path}")

        try:
            image = self.preprocessor(image_path)
        except Exception as exc:
            raise RuntimeError(f"Erreur prétraitement {image_path}: {exc}")

        if self.transform:
            if isinstance(image, torch.Tensor):
                image_np = image.permute(1, 2, 0).numpy()
            else:
                image_np = image

            augmented = self.transform(image=image_np)
            image = torch.from_numpy(augmented["image"]).permute(2, 0, 1)

        targets: Dict[str, torch.Tensor] = {}
        for param in self.target_params:
            value = row[param]
            if pd.isna(value):
                targets[param] = torch.tensor(float("nan"), dtype=torch.float32)
            else:
                targets[param] = torch.tensor(float(value), dtype=torch.float32)

        return image, targets

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        for param in self.target_params:
            valid_values = self.df[param].dropna()
            stats[param] = {
                "mean": float(valid_values.mean()),
                "std": float(valid_values.std()),
                "min": float(valid_values.min()),
                "max": float(valid_values.max()),
                "count": int(len(valid_values)),
            }
        return stats


def create_dataloaders(
    csv_path: Path,
    dataset_root: Path,
    preprocessor: UnderwaterPreprocessor,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """
    Crée les DataLoaders pour train/val/test
    """
    dataloaders: Dict[str, DataLoader] = {}

    df = pd.read_csv(csv_path)
    available_splits = df["split"].unique().tolist()

    print(f"\n{'='*80}")
    print("CRÉATION DES DATALOADERS")
    print(f"{'='*80}")
    print(f"\nSplits disponibles : {available_splits}")

    for split in available_splits:
        dataset = WaterQualityDataset(
            csv_path=csv_path,
            dataset_root=dataset_root,
            split=split,
            preprocessor=preprocessor,
            transform=None,
        )

        shuffle = split == "TRAIN"

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=split == "TRAIN",
        )

        dataloaders[split] = dataloader

        print(f"\n✅ {split} DataLoader créé :")
        print(f"   Batches : {len(dataloader)}")
        print(f"   Shuffle : {shuffle}")

    return dataloaders


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    csv_path = project_root / "processed" / "images_labels.csv"

    dataset_root = None
    for parent in [project_root, *project_root.parents]:
        candidate = parent / "Tilapia RAS Dataset"
        if candidate.exists():
            dataset_root = candidate
            break
    if dataset_root is None:
        dataset_root = project_root.parent / "Tilapia RAS Dataset"

    print("=" * 80)
    print("TEST DU DATASET")
    print("=" * 80)

    preprocessor = create_preprocessor()

    dataloaders = create_dataloaders(
        csv_path=csv_path,
        dataset_root=dataset_root,
        preprocessor=preprocessor,
        batch_size=16,
        num_workers=2,
    )

    if "TRAIN" in dataloaders:
        print(f"\n{'='*80}")
        print("TEST D'UN BATCH TRAIN")
        print(f"{'='*80}")

        train_loader = dataloaders["TRAIN"]
        images, targets = next(iter(train_loader))

        print("\n✅ Batch chargé avec succès :")
        print(f"   Images shape   : {images.shape}")
        print(f"   Images dtype   : {images.dtype}")
        print(f"   Images range   : [{images.min():.2f}, {images.max():.2f}]")

        print("\n   Targets :")
        for param, values in targets.items():
            print(
                f"     {param:15s} : shape={values.shape}, "
                f"mean={values.mean():.2f}, std={values.std():.2f}"
            )

        dataset = train_loader.dataset
        stats = dataset.get_statistics()

        print(f"\n{'='*80}")
        print("STATISTIQUES DU DATASET TRAIN")
        print(f"{'='*80}")

        for param, stat in stats.items():
            print(f"\n{param}:")
            print(f"  Moyenne  : {stat['mean']:.3f}")
            print(f"  Écart-type : {stat['std']:.3f}")
            print(f"  Min      : {stat['min']:.3f}")
            print(f"  Max      : {stat['max']:.3f}")
            print(f"  N samples: {stat['count']}")
