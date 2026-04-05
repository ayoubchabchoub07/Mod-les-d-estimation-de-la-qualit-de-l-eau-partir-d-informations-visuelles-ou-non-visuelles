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

## 📊 Pipeline Global

```
FLUX COMPLET DU PIPELINE :

1. DONNÉES BRUTES
   ↓ (phase1_prepare_dataset.py)
2. DATASET INDEXÉ
   ↓ (train.py / predict.py)
3. MODÈLE ENTRAÎNÉ
   ↓ (predict.py)
4. PRÉDICTIONS
```

---

### Phase 1 : Préparation du Dataset

**📍 Localisation du code :** [`water_quality_cv/phase1_prepare_dataset.py`](water_quality_cv/phase1_prepare_dataset.py)

**Entrées :**
- Images originales : `Tilapia RAS Dataset/Frames/Original/{GX010206,GX020019,GX020206,GX030013}/images/`
- Images augmentées : `Tilapia RAS Dataset/Frames/Augmented/{GaussianBlur,AveragedBlur}/{GX010206,...}/images/`
- Métadonnées : `Tilapia RAS Dataset/Documentation/meta_tilapia_set.csv`

**Exécution :**

```powershell
cd water_quality_cv
python phase1_prepare_dataset.py
```

**Sorties générées :**
- `water_quality_cv/processed/images_labels.csv` : Dataset complet avec chemins relatifs et labels
- `water_quality_cv/processed/stats.json` : Statistiques (split, augmentations, fusion)

**Règles de split :**
- Split au niveau des **clips** (pas des frames) pour éviter le data leakage
- VAL et TEST contiennent **uniquement des images originales**
- TRAIN contient images originales + augmentées
- Proportions : ~70% TRAIN / 15% VAL / 15% TEST (minimum 1 clip en VAL et TEST)

---

### Phase 2 : Prétraitement d'Images

**📍 Localisation du code :** [`water_quality_cv/preprocess.py`](water_quality_cv/preprocess.py)

**Classe à utiliser :** `UnderwaterPreprocessor`

Pipeline de prétraitement optimisé pour images sous-marines :

```python
from water_quality_cv.preprocess import UnderwaterPreprocessor

# Initialisation du pipeline
preprocessor = UnderwaterPreprocessor(
    target_size=(260, 260),          # Taille pour EfficientNet-B2
    use_white_balance=True,          # Correction colorimétrique
    use_clahe=True,                  # Amélioration du contraste
    clahe_clip_limit=2.0,
    normalize_imagenet=True          # Normalisation ImageNet
)

# Traitement d'une image
# Retourne un tensor PyTorch de shape (3, 260, 260)
tensor = preprocessor("path/to/image.jpg")
```

**Étapes du pipeline (séquentielles) :**
1. Chargement et redimensionnement (INTER_AREA)
2. **White Balance** (Simplest Color Balance) → correction colorimétrique
3. **CLAHE** sur canal L en espace LAB → amélioration du contraste
4. Normalisation [0, 1] → conversion uint8 → float32
5. Normalisation ImageNet (mean/std) → standardisation
6. Conversion HWC → CHW (PyTorch) → format réseau

**Fichier de test :**

```powershell
cd water_quality_cv
python test_preprocess.py  # Valide le pipeline sur images du dataset
```

---

### Phase 3 : Entraînement (à venir)

**📍 Localisation du code :** [`water_quality_cv/train.py`](water_quality_cv/train.py)

**Entrées :**
- Dataset indexé : `water_quality_cv/processed/images_labels.csv`

**Architecture :** EfficientNet-B2 pour régression multi-tâches

Entraînement d'un modèle pour prédire les paramètres physico-chimiques :

```powershell
cd water_quality_cv
python train.py \
  --batch_size 32 \
  --epochs 100 \
  --learning_rate 0.001 \
  --model_name efficientnet_b2
```

**Sorties :**
- Meilleur modèle : `water_quality_cv/checkpoints/best_model.pth`
- Modèle dernier epoch : `water_quality_cv/checkpoints/last_model.pth`
- Logs d'entraînement : `ablation_results/{timestamp}/`

---

### Phase 4 : Inférence (Prédiction)

**📍 Localisation du code :** [`water_quality_cv/predict.py`](water_quality_cv/predict.py)

**Entrées :**
- Modèle entraîné : `water_quality_cv/checkpoints/best_model.pth`
- Image à prédire : n'importe quel format (JPG, PNG)

Prédiction des paramètres physico-chimiques sur nouvelles images :

```powershell
cd water_quality_cv
python predict.py \
  --image path/to/image.jpg \
  --model checkpoints/best_model.pth \
  --output predictions.json
```

**Paramètres prédits :**
- `temperature_C` : Température (°C)
- `pH` : Acidité
- `DO_mgL` : Oxyde dissous (mg/L)
- `turbidity_NTU` : Turbidité (NTU)

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

## 🛠️ Utilisation Complète du Pipeline

### Étape par étape

```powershell
# 1. Activer l'environnement virtuel
.\.venv\Scripts\Activate.ps1

# 2. Phase 1 : Préparer le dataset
cd Mod-les-d-estimation-de-la-qualit-de-l-eau-partir-d-informations-visuelles-ou-non-visuelles
cd water_quality_cv
python phase1_prepare_dataset.py
# Génère : processed/images_labels.csv et processed/stats.json

# 3. Phase 2 : Tester le prétraitement
python test_preprocess.py
# Valide le pipeline sur 5 images du TRAIN set

# 4. Phase 3 : Entraîner le modèle (une fois implémenté)
python train.py --epochs 100 --batch_size 32
# Génère : checkpoints/best_model.pth

# 5. Phase 4 : Faire des prédictions (une fois implémenté)
python predict.py --image "path/to/image.jpg" --model checkpoints/best_model.pth
# Affiche les prédictions
```

### Vérifier les résultats de chaque phase

```powershell
# Vérifier stats du dataset (Phase 1)
python -c "import json; stats=json.load(open('processed/stats.json')); print(f'Train: {stats[\"train_count\"]}, Val: {stats[\"val_count\"]}, Test: {stats[\"test_count\"]}')"

# Vérifier les images indexées (Phase 1)
cd water_quality_cv
python -c "import pandas as pd; df=pd.read_csv('processed/images_labels.csv'); print(f'Total: {len(df)}\n{df.head()}')"
```

## 📚 Références

- **Dataset** : Tilapia RAS (RAS = Recirculating Aquaculture System)
- **Architecture** : EfficientNet-B2 (à implémenter)
- **Prétraitement** : CLAHE + White Balance inspiré de méthodes d'amélioration d'images sous-marines

---

**Auteur** : Ayoub Chabchoub et Mohamed Chtourou
**Dernière mise à jour** : Février 2026
