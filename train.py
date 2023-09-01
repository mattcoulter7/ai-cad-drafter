import aicaddrafter
import pandas as pd


def main():
    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = aicaddrafter.data.preparer.prepare_training_data(
        pd.read_csv("data/processed.csv"),
        input_size=1024,
        output_size=128
    )

    aicaddrafter.ai.train.train_model(
        model=aicaddrafter.ai.train.model.get_gru(
            [1024, 512, 256, 128]
        ),
        X_train=X_train,
        y_train=y_train,
        name="0.1.0",
        config={
            "batch": 128,
            "epochs": 200
        }
    )

    pass


if __name__ == "__main__":
    main()
