#!/usr/bin/env python3
"""
Test du module de prétraitement sur une image du dataset
"""

from preprocess import UnderwaterPreprocessor, create_preprocessor
from pathlib import Path
import cv2
import numpy as np

# Chemins
project_root = Path(__file__).resolve().parent
dataset_root = None
for parent in [project_root, *project_root.parents]:
    candidate = parent / "Tilapia RAS Dataset"
    if candidate.exists():
        dataset_root = candidate
        break
if dataset_root is None:
    dataset_root = project_root.parent / "Tilapia RAS Dataset"

images_dir = dataset_root / "Frames" / "Original" / "GX010206" / "images"
print(f"📁 Dataset trouvé : {dataset_root}")

# Trouver une première image
images = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))

if not images:
    print(f"❌ Aucune image trouvée dans {images_dir}")
    exit(1)

test_image = images[0]
print(f"📸 Test sur : {test_image.name}")

# ============================================
# Test 1 : Preprocessor de base
# ============================================
print("\n" + "="*60)
print("TEST 1 : Preprocessor complet (WB + CLAHE + ImageNet)")
print("="*60)

preprocessor = UnderwaterPreprocessor(
    target_size=(260, 260),
    use_clahe=True,
    use_white_balance=True,
    return_numpy=False  # Retourne tensor PyTorch
)

try:
    tensor = preprocessor(test_image)
    print(f"✅ Succès !")
    print(f"   Type  : {type(tensor)}")
    print(f"   Shape : {tensor.shape} (C, H, W)")
    print(f"   Dtype : {tensor.dtype}")
    print(f"   Range : [{tensor.min():.3f}, {tensor.max():.3f}]")
except Exception as e:
    print(f"❌ Erreur : {e}")

# ============================================
# Test 2 : Ablation (sans CLAHE)
# ============================================
print("\n" + "="*60)
print("TEST 2 : Sans CLAHE (WB uniquement)")
print("="*60)

preprocessor_no_clahe = UnderwaterPreprocessor(
    use_clahe=False,
    use_white_balance=True,
    return_numpy=True
)

try:
    array = preprocessor_no_clahe(test_image)
    print(f"✅ Succès !")
    print(f"   Shape : {array.shape} (H, W, C)")
    print(f"   Dtype : {array.dtype}")
    print(f"   Range : [{array.min():.3f}, {array.max():.3f}]")
except Exception as e:
    print(f"❌ Erreur : {e}")

# ============================================
# Test 3 : Ablation (sans White Balance)
# ============================================
print("\n" + "="*60)
print("TEST 3 : Sans White Balance (CLAHE uniquement)")
print("="*60)

preprocessor_no_wb = UnderwaterPreprocessor(
    use_clahe=True,
    use_white_balance=False,
    return_numpy=True
)

try:
    array = preprocessor_no_wb(test_image)
    print(f"✅ Succès !")
    print(f"   Shape : {array.shape} (H, W, C)")
    print(f"   Dtype : {array.dtype}")
    print(f"   Range : [{array.min():.3f}, {array.max():.3f}]")
except Exception as e:
    print(f"❌ Erreur : {e}")

# ============================================
# Test 4 : Factory avec config personnalisée
# ============================================
print("\n" + "="*60)
print("TEST 4 : Factory avec config personnalisée")
print("="*60)

config = {
    'target_size': (224, 224),
    'clahe_clip_limit': 3.0,
    'wb_percentile': (2.0, 2.0),
    'return_numpy': True
}

try:
    preprocessor_custom = create_preprocessor(config)
    array = preprocessor_custom(test_image)
    print(f"✅ Succès !")
    print(f"   Shape : {array.shape} (H, W, C)")
    print(f"   Config appliquée : {config}")
except Exception as e:
    print(f"❌ Erreur : {e}")

print("\n" + "="*60)
print("✅ TOUS LES TESTS SONT PASSÉS !")
print("="*60)
