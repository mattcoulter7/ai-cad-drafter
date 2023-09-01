"""
Generates 3 permutations: transposed horizontally, veritcally and both
"""
import pandas as pd
import typing as T


def transpose(df: pd.DataFrame) -> T.List[pd.DataFrame]:
    transposed_horizontally = df.copy()
    transposed_horizontally['y'] = -transposed_horizontally['y']
    transposed_horizontally['file'] = transposed_horizontally['file'] + f"_transposeH"

    transposed_vertically = df.copy()
    transposed_vertically['x'] = -transposed_vertically['x']
    transposed_vertically['file'] = transposed_vertically['file'] + f"_transposeV"

    transposed_both = df.copy()
    transposed_both['x'] = -transposed_both['x']
    transposed_both['y'] = -transposed_both['y']
    transposed_both['file'] = transposed_both['file'] + f"_transposeHV"

    return [
        transposed_horizontally,
        transposed_vertically,
        transposed_both,
    ]
