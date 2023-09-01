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
    config: dict,
    scaler: MinMaxScaler,
    file: str
) -> T.Tuple[T.List[LineString], T.List[LineString]]:
    df = build_df(
        X=X,
        y=y,
        config=config,
        scaler=scaler,
        file=file
    )

    return interpret_df(df)


def build_df(
    X: np.ndarray,
    y: np.ndarray,
    config: dict,
    scaler: MinMaxScaler,
    file: str
) -> pd.DataFrame:
    points = [
        *build_points(
            arr=X,
            config=config,
            label="wall",
            file=file
        ),
        *build_points(
            arr=y,
            config=config,
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
    config: dict,
    label: str,
    file: str
) -> np.ndarray:
    points = []

    for entry in arr:
        point_buff = []
        for generation in entry:
            for token in generation:
                point_buff.append(token)
                if len(point_buff) == 2:
                    points.append({
                        "x": point_buff[0],
                        "y": point_buff[1],
                        "label": label,
                        "file": file
                    })
                    point_buff = []

    return points
