import aicaddrafter
import os
from dotenv import load_dotenv

load_dotenv()


def main():
    aicaddrafter.test(
        file_name=os.getenv("FILE_NAME"),
        layers=os.getenv("LAYERS").split(",")
    )


if __name__ == "__main__":
    main()
