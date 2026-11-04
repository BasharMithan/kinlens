from dataclasses import dataclass
from numpy import ndarray


@dataclass
class Explanation:
    """The ccomplete evidence behind one KNN prediction.
    
    Attributes:
        prediction: Integer code of the prediction class.
        indices: Training-row indices (nearest first).
        labels: Integer class labels of the neighbors.
        weights: Vote weights aligned with the neighbors.
        classScore: Total vote weights for every class code.
        query: The query vector in it's original feature units.
    """


    prediction: int
    indices: ndarray
    labels: ndarray
    distances: ndarray
    weights: ndarray
    classScore: dict[int, float]
    confidence: float

    query: ndarray