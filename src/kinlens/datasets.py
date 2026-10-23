from dataclasses import dataclass



@dataclass(frozen=True)
class Dataset:
    x: list[float]
    y: list[float]

    features: list[str]
    classTitles: list[str]

    name: str
    images: list[float] | None


