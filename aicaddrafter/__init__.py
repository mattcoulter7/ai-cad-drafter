import dxfgrabber
from dxfgrabber.drawing import Drawing
import logging
import typing as T
import pickle
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .entity import (
    LineExtractor,
    LWPolyLineExtractor,
    Line
)


logger = logging.getLogger()


def graph_lines(line_objects):
    plt.figure()

    for line in line_objects:
        plt.plot([line.x1, line.x2], [line.y1, line.y2])

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.show()


def get_drawing(file_name) -> Drawing:
    dir = "aicaddrafter/data/"

    file_path = os.path.join(dir, file_name + ".dxf")
    pickle_path = os.path.join(dir, file_name + ".pkl")

    drawing = None
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as fp:
            drawing = pickle.load(fp)
    else:
        drawing = dxfgrabber.readfile(
            filename=file_path
        )
        with open(pickle_path, "wb") as fp:
            pickle.dump(drawing, fp, pickle.HIGHEST_PROTOCOL)
    return drawing


def test(
    file_name: str,
    layers: T.List[str]
):
    line_extractors = [
        LineExtractor(),
        LWPolyLineExtractor()
    ]

    drawing = get_drawing(
        file_name=file_name
    )

    layers_entities = [
        ent for ent in drawing.entities._entities if (
            ent.layer in layers
        )
    ]

    lines = []
    for extractor in line_extractors:
        lines.extend(extractor.extract_lines(layers_entities))

    graph_lines(
        lines
    )
