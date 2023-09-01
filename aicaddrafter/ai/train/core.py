import sys
import warnings
import os
from pathlib import Path
import numpy as np
import pandas as pd

from .model import get_gru

warnings.filterwarnings("ignore")


def train_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    name: str,
    config: dict[str, any]
):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """
    model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])

    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05
    )

    model.save(os.path.join('model', f'{name}.h5'))
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv(os.path.join('model', f'{name} loss.csv'), encoding='utf-8', index=False)
