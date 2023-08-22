import dxfgrabber
from dxfgrabber.drawing import Drawing
import pickle
import os

from pathlib import Path


def load_drawing(
    file_name: str,
    cache: bool = True
) -> Drawing:
    file_path = Path(file_name)
    pickle_path = file_path.with_suffix(".pkl")

    drawing = None
    if (
        cache and
        os.path.exists(pickle_path)
    ):
        with open(pickle_path, "rb") as fp:
            drawing = pickle.load(fp)
    else:
        drawing = dxfgrabber.readfile(
            filename=file_path
        )
        with open(pickle_path, "wb") as fp:
            pickle.dump(drawing, fp, pickle.HIGHEST_PROTOCOL)
    return drawing
