from dataclasses import dataclass

import numpy as np


@dataclass
class ClassScore:
    classes: np.ndarray
    scores: np.ndarray



@dataclass
class Prediction:
    query: np.ndarray
    Class: int
    score: float
    confidence: float