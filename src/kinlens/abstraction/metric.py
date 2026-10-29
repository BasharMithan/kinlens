
from abc import ABC, abstractmethod

import numpy as np

from kinlens.exceptions.metricsExceptions import (
    EmptyVectorError,
    VectorsLengthMissMatch,
)


class DistanceMetric(ABC):
    "An abstract class for the distance class, `Euclidean`, `Manhattan`, ..."

    @abstractmethod
    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return the distance between two equal-length numeric vectors."""
        ...

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """Allow instances to be used like functions: ``metric(x, y)``."""
        self.validate(x=x, y=y)

        return self.distance(x=x, y=y)


    def validate(self, x: np.ndarray, y: np.ndarray) -> None:

        if x.shape != y.shape:
            raise VectorsLengthMissMatch()

        if x.size == 0:
            raise EmptyVectorError()


