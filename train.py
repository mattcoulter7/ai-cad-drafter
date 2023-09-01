import aicaddrafter
import pandas as pd
import json


def main():
    version = "0.1.2"

    config = {
        "input_size": 3072,
        "output_size": 256,
        "batch": 256,
        "epochs": 50
    }
    json.dump(config, open(f"model/{version} spec.json", "w"))

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
        model=aicaddrafter.ai.train.model.get_average(
            [config["input_size"], 1024, 512, config["output_size"]]
        ),
        X_train=X_train,
        y_train=y_train,
        name=version,
        config=config
    )

    pass


if __name__ == "__main__":
    main()
