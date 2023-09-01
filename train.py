import aicaddrafter
import pandas as pd
import json

version = "0.1.6"


def main():
    config = {
        "input_size": 3072,
        "output_size": 128,
        "n_input_features": 2,  # x and y, start and end,
        "n_output_features": 2,  # x and y, start and end,
        "max_output_seq_len": 128,
        "n_units": 64,
        "batch": 1,
        "epochs": 8,
        "schedule_ratio": 0.5,
        "learning_rate": 0.0001,
        "print_every": 1,
        "k_max": 1,
        "schedule_type": "linear"
    }
    json.dump(config, open(f"model/{version} spec.json", "w"))

    (
        X_train,
        y_train,
    ) = aicaddrafter.data.preparer.prepare_training_data(
        pd.read_csv(f"data/processed.csv"),
        config=config
    )

    aicaddrafter.ai.train.train_model(
        model=aicaddrafter.ai.train.model.get_seq2seq(config),
        X_train=[X_train, y_train],
        y_train=y_train,
        name=version,
        config=config
    )

    pass


if __name__ == "__main__":
    main()
