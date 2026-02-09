import os
import cv2
import pandas as pd
import numpy as np


def load_and_preprocess(dataset_dir, img_size=(224, 224)):
    labels_path = os.path.join(dataset_dir, "labels.csv")
    labels = pd.read_csv(labels_path)
    images = []
    targets = []
    for _, row in labels.iterrows():
        img_path = os.path.join(dataset_dir, row['filename'])
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img.astype(np.float32) / 255.0)
        targets.append([row['turbidity'], row['pH'], row['DO'], row['temperature']])
    return np.array(images), np.array(targets)


if __name__ == '__main__':
    print('Utilisez ce module depuis `train.py` ou `predict.py`.')
