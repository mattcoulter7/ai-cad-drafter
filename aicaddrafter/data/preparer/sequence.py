import logging
import numpy as np
import pandas as pd
import typing as T
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger()


def prepare_sequence_training_data(
    X: np.ndarray,
    y: np.ndarray,
    config: dict
) -> T.Tuple[T.Tuple[np.ndarray, np.ndarray], np.ndarray]:
    (
        decoder_inputs,
        decoder_outputs,
        encoder_inputs
    ) = [], [], []

    max_context_length = config["max_context_length"]
    min_context_length = config.get("min_context_length")
    lookahead = config.get("lookahead", 1)
    max_context_lag = config.get("max_context_lag")

    for X_entry, y_entry in zip(X, y):
        y_entry = remove_padding(y_entry)
        for i in range(1, min(max_context_length, len(y_entry) - lookahead)):
            entry_input = y_entry[:i]
            if (
                min_context_length is not None and
                len(entry_input) < min_context_length
            ):
                continue
            if (
                max_context_lag is not None and
                len(y_entry) - len(entry_input) > max_context_lag
            ):
                continue
            entry_input = np.pad(
                entry_input,
                pad_width=(
                    (max_context_length - len(entry_input), 0),
                    (0, 0)                    
                ),
                mode='constant'
            )
            entry_output = y_entry[i+1:i+1+lookahead]

            encoder_inputs.append(X_entry)
            decoder_inputs.append(entry_input)
            decoder_outputs.append(entry_output)

    return (
        (
            np.array(encoder_inputs),
            np.array(decoder_inputs),
        ),
        np.array(decoder_outputs)
    )


def remove_padding(arr: np.ndarray, pad: object = 0) -> np.ndarray:
    return arr[[all(_) for _ in arr != pad]]
