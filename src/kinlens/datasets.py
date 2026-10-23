from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class Dataset:
    x: np.ndarray
    y: np.ndarray

    features: list[str]
    classTitles: list[str]

    name: str
    images: np.ndarray | None


