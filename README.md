# Estimation de la Qualité de l'Eau - Images Sous-Marines Tilapia RAS

Projet de computer vision pour l'estimation de paramètres physico-chimiques de la qualité de l'eau (température, pH, DO, turbidité) à partir d'images sous-marines de poissons Tilapia en système RAS (Recirculating Aquaculture System).

## 📁 Structure du Projet

```
p2M/
├── Tilapia RAS Dataset/          # Dataset original (non versionné)
│   ├── Frames/
│   │   ├── Original/             # Images originales par clip (GX010206, GX020019, etc.)
│   │   └── Augmented/            # Images augmentées (GaussianBlur, AveragedBlur)
│   ├── Documentation/
│   │   └── meta_tilapia_set.csv  # Métadonnées physico-chimiques
│   └── Videos/
│
├── Mod-les-d-estimation-de-la-qualit-de-l-eau-partir-d-informations-visuelles-ou-non-visuelles/
│   ├── water_quality_cv/
│   │   ├── phase1_prepare_dataset.py   # Phase 1: Préparation et split du dataset
│   │   ├── preprocess.py               # Pipeline de prétraitement (CLAHE + White Balance)
│   │   ├── train.py                    # Script d'entraînement (à implémenter)
│   │   ├── predict.py                  # Script d'inférence (à implémenter)
│   │   ├── main.py                     # CLI principal
│   │   ├── test_preprocess.py          # Tests du preprocessing
│   │   └── processed/
│   │       ├── images_labels.csv       # Dataset indexé avec labels
│   │       └── stats.json              # Statistiques du dataset
│   │
│   ├── requirements.txt          # Dépendances Python
│   └── .gitignore
│
├── .venv/                        # Environnement virtuel (non versionné)
└── README.md                     # Ce fichier
```

## 🚀 Installation

### 1. Créer un environnement virtuel

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Installer les dépendances

```powershell
cd Mod-les-d-estimation-de-la-qualit-de-l-eau-partir-d-informations-visuelles-ou-non-visuelles
pip install -r requirements.txt
```

**Dépendances principales :**
- `opencv-python` : Traitement d'images
- `torch` / `torchvision` : Deep learning (EfficientNet-B2)
- `pandas` : Manipulation de données
- `tqdm` : Barres de progression
- `matplotlib` : Visualisation

## 📊 Pipeline

### Phase 1 : Préparation du Dataset

Indexe les images (originales + augmentées), fusionne avec les métadonnées physico-chimiques, et crée un split TRAIN/VAL/TEST au niveau des clips.

```powershell
cd water_quality_cv
python phase1_prepare_dataset.py
```

**Sorties :**
- `processed/images_labels.csv` : Dataset complet avec chemins relatifs et labels
- `processed/stats.json` : Statistiques (split, augmentations, fusion)

**Règles de split :**
- Split au niveau des **clips** (pas des frames) pour éviter le data leakage
- VAL et TEST contiennent **uniquement des images originales**
- TRAIN contient images originales + augmentées
- Proportions : ~70% TRAIN / 15% VAL / 15% TEST (minimum 1 clip en VAL et TEST)

### Phase 2 : Prétraitement d'Images

Pipeline de prétraitement optimisé pour images sous-marines :

```python
from preprocess import UnderwaterPreprocessor

preprocessor = UnderwaterPreprocessor(
    target_size=(260, 260),          # Taille pour EfficientNet-B2
    use_white_balance=True,          # Correction colorimétrique
    use_clahe=True,                  # Amélioration du contraste
    clahe_clip_limit=2.0,
    normalize_imagenet=True          # Normalisation ImageNet
)

# Retourne un tensor PyTorch (3, 260, 260)
tensor = preprocessor("path/to/image.jpg")
```

**Étapes du pipeline :**
1. Chargement et redimensionnement (INTER_AREA)
2. **White Balance** (Simplest Color Balance)
3. **CLAHE** sur canal L en espace LAB
4. Normalisation [0, 1]
5. Normalisation ImageNet (mean/std)
6. Conversion HWC → CHW (PyTorch)

**Tester le preprocessing :**

```powershell
python test_preprocess.py
```

### Phase 3 : Entraînement (à venir)

Entraînement d'un modèle EfficientNet-B2 pour régression multi-tâches.

```powershell
python train.py
```

### Phase 4 : Inférence (à venir)

Prédiction des paramètres physico-chimiques sur nouvelles images.

```powershell
python predict.py --image path/to/image.jpg --model checkpoints/best_model.pth
```

## 📈 Dataset Tilapia RAS

- **4 clips vidéo** (GX010206, GX020019, GX020206, GX030013)
- **~10560 images** total (originales + augmentées)
- **Annotations** : Bounding boxes LabelMe (JSON) pour détection de poissons
- **Métadonnées :** température (°C), pH, DO (mg/L), turbidité (NTU), lighting_mode, profondeur

**Paramètres cibles :**
- `temperature_C` : Température de l'eau
- `pH` : Acidité
- `DO_mgL` : Oxygène dissous
- `turbidity_NTU` : Turbidité

## 🧪 Tests

```powershell
# Tester le preprocessing
python test_preprocess.py

# Vérifier les stats du dataset
python -c "import json; print(json.load(open('processed/stats.json', 'r')))"
```

## 📝 Notes Techniques

- **Split par clip** : Évite le data leakage temporel (frames consécutives)
- **Ablation** : Flags `use_clahe`, `use_white_balance` pour études d'ablation
- **Augmentations** : GaussianBlur et AveragedBlur uniquement en TRAIN
- **Normalisation** : ImageNet (requis pour transfer learning EfficientNet)

## 🛠️ Développement

```powershell
# Activer l'environnement
.\.venv\Scripts\Activate.ps1

# Lancer la phase 1
python water_quality_cv/phase1_prepare_dataset.py

# Tests
python water_quality_cv/test_preprocess.py
```

## 📚 Références

- **Dataset** : Tilapia RAS (RAS = Recirculating Aquaculture System)
- **Architecture** : EfficientNet-B2 (à implémenter)
- **Prétraitement** : CLAHE + White Balance inspiré de méthodes d'amélioration d'images sous-marines

---

**Auteur** : Ayoub Chabchoub et Mohamed Chtourou
**Dernière mise à jour** : Février 2026
