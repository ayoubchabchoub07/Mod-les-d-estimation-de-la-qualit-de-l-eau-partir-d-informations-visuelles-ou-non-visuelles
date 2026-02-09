import os
import numpy as np
from tensorflow.keras import models, layers
from preprocess import load_and_preprocess


def build_model(input_shape, output_dim=4):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(output_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def train(dataset_dir, epochs=5, model_out='model.h5'):
    X, y = load_and_preprocess(dataset_dir)
    if len(X) == 0:
        print('Aucune image trouvée. Placez des images dans dataset/images et mettez à jour labels.csv')
        return
    model = build_model(X.shape[1:])
    model.fit(X, y, epochs=epochs, batch_size=8)
    model.save(model_out)
    print(f'Model saved to {model_out}')


if __name__ == '__main__':
    train(os.path.join(os.path.dirname(__file__), 'dataset'))
