import aicaddrafter
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def main():
    file_name = os.getenv("FILE_NAMES").split(", ")[0]
    wall_layers = os.getenv("WALL_LAYERS").split(",")
    lintels_layers = os.getenv("LINTELS_LAYERS").split(",")

    drawing = aicaddrafter.data.processor.loader.load_drawing(file_name)
    wall_lines_raw = aicaddrafter.data.processor.extract_lines(drawing, wall_layers)
    lintel_lines_raw = aicaddrafter.data.processor.extract_lines(drawing, lintels_layers)

    aicaddrafter.renderer.render(
        lines=[
            *wall_lines_raw,
            *lintel_lines_raw
        ],
        polygons=[]
    )

    df = aicaddrafter.data.processor.process_file(
        file_name=file_name,
        wall_layers=wall_layers,
        lintels_layers=lintels_layers,
    )

    (
        wall_lines,
        lintel_lines,
    ) = aicaddrafter.data.interpreter.interpret_df(df)

    aicaddrafter.renderer.render(
        lines=[
            *wall_lines,
            *lintel_lines
        ],
        polygons=[]
    )

    pass


if __name__ == "__main__":
    main()
