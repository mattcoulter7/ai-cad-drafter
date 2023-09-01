import logging
import typing as T
import numpy as np
import pandas as pd
from pathlib import Path
from shapely import LineString
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ThreadPoolExecutor

from .entity import (
    BaseEntityExtractor,
    LineExtractor,
    LWPolyLineExtractor,
)

from .loader import (
    load_drawing
)

from .augmentation import augment, rotate, transpose


logger = logging.getLogger()


def process_files(
    file_names: T.List[str],
    wall_layers: T.List[str],
    lintels_layers: T.List[str],
    perform_augmentation: bool = False
) -> pd.DataFrame:
    points = []
    with ThreadPoolExecutor(128) as executor:
        for dfs in executor.map(
            lambda file_name: process_file(
                file_name=file_name,
                wall_layers=wall_layers,
                lintels_layers=lintels_layers,
                perform_augmentation=perform_augmentation
            ),
            file_names
        ):
            for df in dfs:
                points.extend(df.values)

    return pd.DataFrame(points)


def process_file(
    file_name: str,
    wall_layers: T.List[str],
    lintels_layers: T.List[str],
    perform_augmentation: bool = False
) -> T.List[pd.DataFrame]:
    logger.info(f"Processing file `{file_name}`")
    drawing = load_drawing(file_name)

    points = []
    for line in extract_lines(drawing, wall_layers):
        for x_val, y_val in line.coords:
            points.append({'x': x_val, 'y': y_val, 'label': 'wall', 'file': file_name})

    for line in extract_lines(drawing, lintels_layers):
        for x_val, y_val in line.coords:
            points.append({'x': x_val, 'y': y_val, 'label': 'lintel', 'file': file_name})

    if not points:
        return []

    df = pd.DataFrame(points)
    dfs = [df]
    if perform_augmentation is True:
        dfs = augment(
            dfs=dfs,
            augmentors=[
                rotate,
                transpose
            ]
        )

    return dfs


def extract_lines(
    drawing,
    layers
) -> T.List[LineString]:
    line_extractors: T.List[BaseEntityExtractor] = [
        LineExtractor(),
        LWPolyLineExtractor()
    ]
    entities = [
        ent for ent in drawing.modelspace() if (
            ent.layer in layers
        )
    ]
    lines = []
    for extractor in line_extractors:
        lines.extend(extractor.extract_lines(entities))
    return lines
