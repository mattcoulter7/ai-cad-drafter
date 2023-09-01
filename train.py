import aicaddrafter
import pandas as pd
import json


def main():
    version = "0.1.1"

    config = {
        "input_size": 2048,
        "output_size": 128,
        "batch": 32,
        "epochs": 30
    }
    json.dump(open(f"model/{version} spec.json", "w"))

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = aicaddrafter.data.preparer.prepare_training_data(
        pd.read_csv(f"data/{version}.processed.csv"),
        input_size=config["input_size"],
        output_size=config["output_size"]
    )

    aicaddrafter.ai.train.train_model(
        model=aicaddrafter.ai.train.model.get_gru(
            [config["input_size"], 1024, 512, 256, config["output_size"]]
        ),
        X_train=X_train,
        y_train=y_train,
        name=version,
        config=config
    )

    pass


if __name__ == "__main__":
    main()
