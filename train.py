import aicaddrafter
import pandas as pd
import json
import numpy as np

version = "0.1.5"


def main():
    config = {
        "input_size": 3072,
        "output_size": 128,
        "n_input_features": 4,  # x and y, start and end,
        "n_output_features": 4,  # x and y, start and end
        "max_output_seq_len": 32,
        "n_units": 64,
        "batch": 64,
        "epochs": 100
    }
    json.dump(config, open(f"model/{version} spec.json", "w"))

    (
        X_train,
        y_train,
    ) = aicaddrafter.data.preparer.prepare_training_data(
        pd.read_csv(f"data/{version}.processed.csv"),
        config=config
    )

    aicaddrafter.ai.train.train_model(
        model=aicaddrafter.ai.train.model.get_seq2seq(
            n_input_features=config["n_input_features"],
            n_units=config["n_units"],
            n_output_features=config["n_output_features"]
        ),
        X_train=[X_train, y_train],
        y_train=y_train,
        name=version,
        config=config
    )

    pass


if __name__ == "__main__":
    main()
