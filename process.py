import aicaddrafter
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def main():
    df = aicaddrafter.data.processor.process_files(
        file_names=os.getenv("FILE_NAMES").split(", "),
        wall_layers=os.getenv("WALL_LAYERS").split(","),
        lintels_layers=os.getenv("LINTELS_LAYERS").split(","),
        perform_augmentation=True
    )

    path = Path(f"data/{version}.processed.csv")
    df.to_csv(
        path,
        index=False,
        header=['x', 'y', 'label', 'file']
    )

    df = pd.read_csv(path)
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


if __name__ == "__main__":
    main()
