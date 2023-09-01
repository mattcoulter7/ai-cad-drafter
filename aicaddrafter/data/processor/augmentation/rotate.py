"""
Generates 3 variants of the oriignl, each rotated by an additional 90 degrees
"""
import pandas as pd
import typing as T


def rotate(df: pd.DataFrame) -> T.List[pd.DataFrame]:
    df_90 = df.copy()
    df_90['x'] = df['y']
    df_90['y'] = -df['x']
    df_90['file'] = df_90['file'] + f"_rotate90"

    df_180 = df_90.copy()
    df_180['x'] = df_90['y']
    df_180['y'] = -df_90['x']
    df_180['file'] = df_180['file'] + f"_rotate180"

    df_270 = df_180.copy()
    df_270['x'] = df_180['y']
    df_270['y'] = -df_180['x']
    df_270['file'] = df_270['file'] + f"_rotate270"

    return [
        df_90,
        df_180,
        df_270
    ]
