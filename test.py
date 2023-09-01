import aicaddrafter
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def main():
    file_name = os.getenv("FILE_NAMES").split(", ")[0]
    df = pd.read_csv("data/processed.csv")
    X_test, y_true, scaler = aicaddrafter.data.preparer.prepare_file_data(
        df[df["file"] == file_name],
        input_size=1024,
        output_size=128
    )
    model = aicaddrafter.ai.train.model.load_model(
        "model/0.1.0.h5"
    )
    y_pred = model.predict(X_test)

    (
        wall_lines,
        lintel_lines,
    ) = aicaddrafter.data.interpreter.interpret(
        X=X_test,
        y=y_true,
        scaler=scaler
    )
    aicaddrafter.renderer.render(
        lines=[
            *wall_lines,
            *lintel_lines
        ],
        polygons=[]
    )

    (
        wall_lines,
        lintel_lines_pred,
    ) = aicaddrafter.data.interpreter.interpret(
        X=X_test,
        y=y_pred,
        scaler=scaler
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
