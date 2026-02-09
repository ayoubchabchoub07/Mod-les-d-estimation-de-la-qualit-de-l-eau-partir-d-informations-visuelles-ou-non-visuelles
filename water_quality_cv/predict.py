import os
import numpy as np
from tensorflow.keras.models import load_model
from preprocess import load_and_preprocess


def predict(dataset_dir, model_path='model.h5'):
    X, _ = load_and_preprocess(dataset_dir)
    if len(X) == 0:
        print('Aucune image trouvée pour la prédiction.')
        return
    model = load_model(model_path)
    preds = model.predict(X)
    return preds


if __name__ == '__main__':
    preds = predict(os.path.join(os.path.dirname(__file__), 'dataset'))
    if preds is not None:
        print(preds)
