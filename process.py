import aicaddrafter
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def main():
    df = aicaddrafter.data.processor.process_files(
        file_names=os.getenv("FILE_NAMES").split(","),
        wall_layers=os.getenv("WALL_LAYERS").split(","),
        lintels_layers=os.getenv("LINTELS_LAYERS").split(","),
    )

    df.to_csv(
        Path("data/processed.csv"),
        index=False,
        header=['x', 'y', 'label', 'file']
    )


if __name__ == "__main__":
    main()