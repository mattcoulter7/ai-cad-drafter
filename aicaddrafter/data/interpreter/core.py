import logging
import typing as T
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from shapely import LineString

logger = logging.getLogger()


def interpret_df(
    df: pd.DataFrame
) -> T.Tuple[T.List[LineString], T.List[LineString]]:
    wall_points = df[df['label'] == 'wall'][['x', 'y']].values
    wall_lines = build_lines(wall_points)

    lintel_points = df[df['label'] == 'lintel'][['x', 'y']].values
    lintel_lines = build_lines(lintel_points)

    return (
        wall_lines,
        lintel_lines
    )


def interpret_xy(
    X: np.ndarray,
    y: np.ndarray,
    scaler: MinMaxScaler,
    file: str
) -> T.Tuple[T.List[LineString], T.List[LineString]]:
    df = build_df(
        X=X,
        y=y,
        scaler=scaler,
        file=file
    )

    return interpret_df(df)


def build_df(
    X: np.ndarray,
    y: np.ndarray,
    scaler: MinMaxScaler,
    file: str
) -> pd.DataFrame:
    points = [
        *build_points(
            arr=X,
            label="wall",
            file=file
        ),
        *build_points(
            arr=y,
            label="lintel",
            file=file
        )
    ]

    df = pd.DataFrame(points)
    df[['x', 'y']] = scaler.inverse_transform(df[['x', 'y']].values)

    return df


def build_lines(
    points: np.ndarray
) -> T.List[LineString]:
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
    label: str,
    file: str
) -> np.ndarray:
    points = []

    for entry in arr:
        for i in range(0, len(entry), 2):
            points.append({
                'x': entry[i],
                'y': entry[i+1],
                'label': label,
                'file': file
            })
    return points
