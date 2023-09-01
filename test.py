import aicaddrafter
import pandas as pd
import json
import random
import numpy as np
from dotenv import load_dotenv
load_dotenv()

version = "0.1.5"


def predict_sequence(
    input_sequence,
    encoder_model,
    decoder_model,
    config: dict
):
    states = encoder_model.predict(input_sequence)
    target_sequence = np.zeros((1, 1, config["n_output_features"]))
    predicted_sequence = []

    for _ in range(config["max_output_seq_len"]):
        output_tokens, h, c = decoder_model.predict([target_sequence] + states)
        predicted_sequence.append(output_tokens[0,0,:])
        states = [h, c]
        target_sequence = output_tokens

    return np.array([predicted_sequence])

def main():
    config = json.load(open(f"model/{version} spec.json", "r"))
    config["max_output_seq_len"] = config.get("max_output_seq_len", 64)
    df = pd.read_csv(f"data/{version}.processed.csv")
    file_name = random.choice(df['file'].unique())
    df = df[df["file"] == file_name]

    X_test, _, scaler = aicaddrafter.data.preparer.prepare_file_data(
        df,
        config=config
    )
    encoder_model, decoder_model = aicaddrafter.ai.train.model.load_models(
        f"model/{version}.h5",
        config=config
    )
    y_pred = predict_sequence(
        input_sequence=X_test,
        encoder_model=encoder_model,
        decoder_model=decoder_model,
        config=config
    )

    pass

    # Render 1 - pred
    (
        wall_lines,
        lintel_lines_pred,
    ) = aicaddrafter.data.interpreter.interpret_xy(
        X=X_test,
        y=y_pred,
        config=config,
        scaler=scaler,
        file=file_name
    )
    aicaddrafter.renderer.render(
        lines=[
            *wall_lines,
            *lintel_lines_pred
        ],
        polygons=[]
    )


if __name__ == "__main__":
    main()
