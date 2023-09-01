import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from shapely import LineString

logger = logging.getLogger()


def interpret_df(
    df: pd.DataFrame
):
    wall_points = df[df['label'] == 'wall'][['x', 'y']].values
    wall_lines = build_lines(wall_points)

    lintel_points = df[df['label'] == 'lintel'][['x', 'y']].values
    lintel_lines = build_lines(lintel_points)

    return (
        wall_lines,
        lintel_lines
    )


def interpret(
    X: np.ndarray,
    y: np.ndarray,
    scaler: MinMaxScaler = None
):
    wall_lines = build_lines(
        points=build_points(
            arr=X,
            scaler=scaler
        )
    )

    lintel_lines = build_lines(
        points=build_points(
            arr=y,
            scaler=scaler
        )
    )

    return (
        wall_lines,
        lintel_lines
    )


def build_lines(
    points: np.ndarray
) -> list[LineString]:
    lines = []

    for i in range(0, len(points), 2):
        start_point = points[i]
        end_point = points[i + 1]

        lines.append(
            LineString((start_point, end_point))
        )

    return lines


def build_points(
    arr: np.ndarray,
    scaler: MinMaxScaler = None
) -> np.ndarray:
    x_vals = []
    y_vals = []

    for entry in arr:
        for i in range(0, len(entry), 2):
            x_val = entry[i]
            y_val = entry[i+1]
            x_vals.append(x_val)
            y_vals.append(y_val)

    xy_values = np.array(list(zip(x_vals, y_vals)))
    if scaler is not None:
        xy_values = scaler.inverse_transform(xy_values)
    return xy_values
