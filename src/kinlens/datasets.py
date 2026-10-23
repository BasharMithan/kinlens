from dataclasses import dataclass
import unmpy as np


@dataclass(frozen=True)
class Dataset:
    x: np.array
    y: np.array

    features: list[str]
    classTitles: list[str]

    name: str
    images: np.array | None



