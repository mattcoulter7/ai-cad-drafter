import logging
import numpy as np
import pandas as pd
import typing as T
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


logger = logging.getLogger()



def prepare_training_data(
    df: pd.DataFrame,
    config: dict
) -> T.Tuple[
    np.ndarray,
    np.ndarray,
]:
    # build train and test data
    X, y = [], []
    for file in df['file'].unique():
        df_file = df[df['file'] == file]
        X_file, y_file, _ = prepare_file_data(
            df=pd.DataFrame(df_file),
            config=config
        )
        X.append(X_file[0])
        y.append(y_file[0])

    return (np.array(X), np.array(y))


def prepare_file_data(
    df: pd.DataFrame,
    config: dict
) -> T.Tuple[
    np.ndarray,
    np.ndarray,
    MinMaxScaler
]:
    scaler = scale_xy(df)
    X = prepare_X(
        df,
        config
    )
    y = prepare_y(
        df,
        config
    )

    return (X, y, scaler)


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


def prepare_X(df: pd.DataFrame, config: dict) -> np.ndarray:
    X = df[df['label'] == 'wall'][['x', 'y']].values.reshape(1, -1)[0]
    X = np.pad(
        X,
        pad_width=(0, config["input_size"] - len(X)),
        mode='constant'
    )
    return X.reshape(1, X.shape[0] // config["n_input_features"], config["n_input_features"])


def prepare_y(df: pd.DataFrame, config: dict) -> np.ndarray:
    y = df[df['label'] == 'lintel'][['x', 'y']].values.reshape(1, -1)[0]
    y = np.pad(
        y,
        (0, config["output_size"] - len(y)),
        mode='constant'
    )
    return y.reshape(1, y.shape[0] // config["n_output_features"], config["n_output_features"])
