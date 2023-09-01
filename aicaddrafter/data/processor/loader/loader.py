import dxfgrabber
from dxfgrabber.drawing import Drawing
import pickle
import os
import gdown
from pathlib import Path


def download_file(url, destination):
    gdown.download(url, destination, quiet=False)


def load_drawing(
    file_name: str,
    cache: bool = True
) -> Drawing:
    file_path = Path(file_name)
    if file_name.startswith("https://drive.google.com"):
        file_id = file_name.split("/")[-2]
        file_path = Path(f'data/{file_id}.dxf')
    pickle_path = file_path.with_suffix(".pkl")

    drawing = None
    if cache and os.path.exists(pickle_path):
        with open(pickle_path, "rb") as fp:
            drawing = pickle.load(fp)
    else:
        if file_name.startswith("https://drive.google.com"):
            file_id = file_name.split("/")[-2]
            download_url = f"https://drive.google.com/uc?id={file_id}"
            download_file(download_url, str(file_path.resolve()))
        drawing = dxfgrabber.readfile(filename=file_path)

        with open(pickle_path, "wb") as fp:
            pickle.dump(drawing, fp, pickle.HIGHEST_PROTOCOL)
    return drawing
