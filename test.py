import aicaddrafter
import os
from dotenv import load_dotenv

load_dotenv()


def main():
    aicaddrafter.test(
        file_name=os.getenv("FILE_NAME"),
        wall_layers=os.getenv("WALL_LAYERS").split(","),
        lintels_layers=os.getenv("LINTELS_LAYERS").split(","),
        window_layers=os.getenv("WINDOW_LAYERS").split(",")
    )


if __name__ == "__main__":
    main()
