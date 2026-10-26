from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Dataset:

    x: np.ndarray
    y: np.ndarray

    features: list[str]
    classes: list[str]

    name: str

    targetEncodingMap: dict[str, int]

    extras: dict[str, np.ndarray] = field(default_factory=dict)


    def __post_init__(self):
        if not (self.x.ndim == 2 or self.y.ndim == 2 or len(self.x) == len(self.y) or
              len(self.features) == self.x.shape[0]):
            raise ValueError("The input dataset is not valid.")

        # assert self.x.ndim == 2
        # assert self.y.ndim == 1
        # assert len(self.x) == len(self.y)

        # assert len(self.features) == self.x.shape[1]

