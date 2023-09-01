import logging
import numpy as np
import pandas as pd
import typing as T
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


logger = logging.getLogger()


def prepare_training_data(
    df: pd.DataFrame,
    input_size: int,
    output_size: int,
    test_size: float = 0.2
) -> T.Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    # build train and test data
    X, y = [], []
    for file in df['file'].unique():
        X_file, y_file, _ = prepare_file_data(
            df=pd.DataFrame(df[df['file'] == file]),
            input_size=input_size,
            output_size=output_size
        )
        X.append(X_file)
        y.append(y_file)

    # convert to numpy arrays
    X = np.array(X).reshape(-1, input_size)
    y = np.array(y).reshape(-1, output_size)

    return train_test_split(X, y, test_size=test_size)


def prepare_file_data(
    df: pd.DataFrame,
    input_size: int,
    output_size: int,
    scaler: MinMaxScaler = None
) -> T.Tuple[
    np.ndarray,
    np.ndarray,
    MinMaxScaler
]:
    scaler = scale_xy(df, scaler)
    return (
        prepare_X(df, input_size),
        prepare_y(df, output_size),
        scaler
    )


def scale_xy(
    df: pd.DataFrame,
    scaler: MinMaxScaler = None
) -> MinMaxScaler:
    xy_values = df[['x', 'y']].values
    scaler = scaler or MinMaxScaler(feature_range=(0, 1)).fit(xy_values)
    df[['x', 'y']] = scaler.transform(xy_values)
    return scaler


def inverse_scale_xy(
    df: pd.DataFrame,
    scaler: MinMaxScaler
) -> MinMaxScaler:
    xy_values = df[['x', 'y']].values
    df[['x', 'y']] = scaler.inverse_transform(xy_values)
    return scaler


def prepare_X(df: pd.DataFrame, input_size: int) -> np.ndarray:
    X = df[df['label'] == 'wall'][['x', 'y']].values.reshape(1, -1)[0]
    X = np.pad(
        X,
        pad_width=(0, input_size - len(X)),
        mode='constant'
    )
    return X.reshape(-1, input_size)


def prepare_y(df: pd.DataFrame, output_size: int) -> np.ndarray:
    y = df[df['label'] == 'lintel'][['x', 'y']].values.reshape(1, -1)[0]
    y = np.pad(
        y,
        (0, output_size - len(y)),
        mode='constant'
    )
    return y.reshape(-1, output_size)
