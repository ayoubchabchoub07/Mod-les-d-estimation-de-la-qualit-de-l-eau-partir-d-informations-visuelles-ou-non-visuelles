# Water Quality CV

Petit projet de détection/estimation de paramètres de qualité de l'eau à partir d'images.

## Structure

- dataset/
  - images/      -> placez ici vos images (ex: `img1.jpg`)
  - labels.csv   -> fichier CSV contenant les labels (voir format)
- preprocess.py   -> chargement et prétraitement des images
- train.py        -> script d'entraînement (sauvegarde `model.h5`)
- predict.py      -> script de prédiction à partir d'un modèle existant
- main.py         -> CLI pour exécuter `train` ou `predict`

## Format de `labels.csv`

Le CSV doit contenir l'en-tête suivant :

```
filename,turbidity,pH,DO,temperature
images/img1.jpg,0.5,7.0,8.0,20.0
```

- `filename` : chemin relatif depuis `dataset/` vers l'image
- `turbidity`, `pH`, `DO`, `temperature` : valeurs numériques cibles

## Dépendances

Recommandé :

```
pip install opencv-python numpy pandas matplotlib scikit-learn tensorflow
```

Vous pouvez aussi mettre ces lignes dans un `requirements.txt`.

## Exemples d'utilisation

Pour entraîner :

```bash
python main.py --step train --dataset ./water_quality_cv/dataset --model model.h5
```

Pour prédire :

```bash
python main.py --step predict --dataset ./water_quality_cv/dataset --model model.h5
```

Les scripts utilisent `model.h5` par défaut comme nom de modèle.

## Notes

- `dataset/images/` est actuellement vide : ajoutez vos images et mettez à jour `labels.csv`.
- Les scripts sont des squelettes minimalistes ; adaptez l'architecture, la normalisation et la gestion des données selon vos besoins.

---

Si vous voulez, j'ajoute un `requirements.txt` dans le projet ou j'ajoute quelques images d'exemple.
