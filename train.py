import aicaddrafter
import pandas as pd
import json

version = "0.2.0"


def main():
    config = {
        "input_size": 3072,
        "output_size": 128,
        "n_input_features": 2,  # x and y, start and end,
        "n_output_features": 2,  # x and y, start and end,
        "max_output_seq_len": 128,
        "n_units": 32,
        "batch": 256,
        "epochs": 8,
        "learning_rate": 0.0001,
        # "min_context_length": 8,
        # "max_context_lag": 4,
        "max_context_length": 64,
        "lookahead": 1
    }
    json.dump(config, open(f"model/{version} spec.json", "w"))

    # prepare the training data in terms of inputs and outputs
    (
        X_train,
        y_train,
    ) = aicaddrafter.data.preparer.prepare_training_data(
        pd.read_csv(f"data/processed.csv"),
        config=config
    )

    # transform the training data for seq2seq model
    # this is for optimising generation of a single token
    (
        encoder_inputs,
        decoder_inputs
    ),  decoder_outputs = aicaddrafter.data.preparer.prepare_sequence_training_data(
        X=X_train,
        y=y_train,
        config=config
    )

    # run the training
    aicaddrafter.ai.train.train_model(
        model=aicaddrafter.ai.train.model.get_seq2seq(config),
        X_train=(
            encoder_inputs,
            decoder_inputs
        ),
        y_train=decoder_outputs,
        name=version,
        config=config
    )


if __name__ == "__main__":
    main()
