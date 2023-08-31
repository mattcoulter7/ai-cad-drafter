import aicaddrafter
import pandas as pd
from pathlib import Path


def main():
    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = aicaddrafter.data.preparer.prepare_training_data(
        pd.read_csv(Path("data/processed.csv")),
        input_size=1024,
        output_size=128
    )
    pass


if __name__ == "__main__":
    main()
