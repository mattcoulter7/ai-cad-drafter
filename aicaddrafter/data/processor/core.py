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


logger = logging.getLogger()


def process_files(
    file_names: T.List[str],
    wall_layers: T.List[str],
    lintels_layers: T.List[str],
) -> T.Tuple[np.ndarray, np.ndarray]:
    points = []
    with ThreadPoolExecutor(16) as executor:
        for df in executor.map(
            lambda file_name: process_file(
                file_name=file_name,
                wall_layers=wall_layers,
                lintels_layers=lintels_layers
            ),
            file_names
        ):
            points.extend(df.values)

    return pd.DataFrame(points)


def process_file(
    file_name: str,
    wall_layers: T.List[str],
    lintels_layers: T.List[str],
) -> pd.DataFrame:
    drawing = load_drawing(file_name)

    points = []
    for line in _extract_lines(drawing, wall_layers):
        for x_val, y_val in line.xy:
            points.append({'x': x_val, 'y': y_val, 'label': 'wall', 'file': file_name})

    for line in _extract_lines(drawing, lintels_layers):
        for x_val, y_val in line.xy:
            points.append({'x': x_val, 'y': y_val, 'label': 'lintel', 'file': file_name})

    return pd.DataFrame(points)


def _extract_lines(
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
