import logging
import typing as T
import numpy as np
import pandas as pd
from pathlib import Path
from shapely import LineString
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split


logger = logging.getLogger()


def prepare_training_data(
    df: pd.DataFrame,
    input_size: int,
    output_size: int
):
    xy_values = df[['x', 'y']].values

    # normalise the x and y values
    xy_scalar = MinMaxScaler(feature_range=(0, 1)).fit(xy_values)
    df[['x', 'y']] = xy_scalar.transform(xy_values)

    # build train and test data
    X, y = [], []
    for _ in df['file'].unique():
        X_temp = []
        for xy in df[df['label'] == 'wall'][['x', 'y']].values:
            X_temp.extend(xy)

        y_temp = []
        for xy in df[df['label'] == 'lintel'][['x', 'y']].values:
            y_temp.extend(xy)

        X_temp = np.pad(X_temp, ((0, input_size - len(X_temp))), mode='constant')
        y_temp = np.pad(y_temp, ((0, output_size - len(y_temp))), mode='constant')

        X.append(X_temp)
        y.append(y_temp)

        # TODO remove this, I don't have enough testing data yet lol
        X.append(X_temp)
        y.append(y_temp)

    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    return train_test_split(X, y, test_size=0.2)
