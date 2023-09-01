import pandas as pd
import typing as T

from .rotate import rotate
from .transpose import transpose


def augment(
    dfs: T.List[pd.DataFrame],
    augmentors: T.List[T.Callable] = None,
    depth: int = 0
) -> T.List[pd.DataFrame]:
    augmentors = augmentors or []
    if not augmentors:
        return []

    augmentations = []
    for df in dfs:
        if depth == 0:
            augmentations.append(df)
        augmentations.extend(augmentors[0](df))
    
    # augmentation recursion = exponential growth
    augmentations.extend(
        augment(
            dfs=augmentations,
            augmentors=augmentors[1:],
            depth=depth + 1
        )
    )

    return augmentations
