import aicaddrafter
import pandas as pd
import json
import random
from dotenv import load_dotenv

load_dotenv()


def main():
    version = "0.1.1"

    config = json.load(open(f"model/{version} spec.json", "r"))

    df = pd.read_csv(f"data/{version}.processed.csv")
    file_name = random.choice(df['file'].unique())
    df = df[df["file"] == file_name]

    # Render 1 - file untouched
    (
        wall_lines_raw,
        lintel_lines_raw,
    ) = aicaddrafter.data.interpreter.interpret_df(df)
    aicaddrafter.renderer.render(
        lines=[
            *wall_lines_raw,
            *lintel_lines_raw
        ],
        polygons=[]
    )

    # Render 2 - file with scaling HAS ISSUE
    X_test, y_true, scaler = aicaddrafter.data.preparer.prepare_file_data(
        df,
        input_size=config["input_size"],
        output_size=config["output_size"]
    )

    (
        wall_lines,
        lintel_lines,
    ) = aicaddrafter.data.interpreter.interpret_xy(
        X=X_test,
        y=y_true,
        scaler=scaler,
        file=file_name
    )
    aicaddrafter.renderer.render(
        lines=[
            *wall_lines,
            *lintel_lines
        ],
        polygons=[]
    )

    # Render 3 - prediction
    model = aicaddrafter.ai.train.model.load_model(
        f"model/{version}.h5"
    )
    y_pred = model.predict(X_test)
    (
        wall_lines,
        lintel_lines_pred,
    ) = aicaddrafter.data.interpreter.interpret_xy(
        X=X_test,
        y=y_pred,
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
