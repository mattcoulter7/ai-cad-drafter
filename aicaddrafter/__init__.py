from dxfgrabber.drawing import Drawing
import ezdxf
import pyautocad
import logging
import typing as T

from .entity import (
    LineExtractor,
    LWPolyLineExtractor,
)

from .objects import WindowExtractor

from .loader import (
    load_drawing
)

from .renderer import render


logger = logging.getLogger()


def test(
    file_name: str,
    wall_layers: T.List[str],
    window_layers: T.List[str]
):
    line_extractors = [
        LineExtractor(),
        LWPolyLineExtractor()
    ]

    polygon_extractors = [
        WindowExtractor(
            layers=window_layers
        )
    ]

    drawing = load_drawing(
        file_name=file_name
    )

    wall_entities = [
        ent for ent in drawing.paperspace() if (
            ent.layer in wall_layers
        )
    ]
    lines = []
    for extractor in line_extractors:
        lines.extend(extractor.extract_lines(wall_entities))

    polygons = []
    for extractor in polygon_extractors:
        polygons.extend(extractor.extract(drawing))

    render(
        lines=lines,
        polygons=polygons
    )
