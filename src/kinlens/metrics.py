from abc import ABC, abstractmethod

import math
import numpy as np

from kinlens.exceptions.metricesExceptions import (
    EmptyVectorError, VectorsLengthMissMatch,
    CosineDistanceUndefined)


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



class Euclidean(DistanceMetric):
    """L2 distance: sqrt(sum((x_i, - y_i)^2))"""

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:

        return float(np.linalg.norm(x - y))


class Manhattan(DistanceMetric):
    """L1 distance: sum(|x_1 - y_i|)"""

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:

        return float(np.sum(np.abs(x - y)))


class Cosine(DistanceMetric):
    """Measures the orientation or angle between two non-zero vectors."""

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:

        xNorm = np.linalg.norm(x)
        yNorm = np.linalg.norm(y)

        if xNorm == 0 or yNorm == 0:
            raise CosineDistanceUndefined()

        similarity = np.dot(x, y) / (xNorm * yNorm)

        return float(1.0 - similarity)





if __name__ == "__main__":
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 6.0, 8.0])

    euclidean = Euclidean()
    manhattan = Manhattan()
    cosine = Cosine()

    print(f"Euclidean: {euclidean(x=x, y=y)}")
    print(f"Manhattan: {manhattan(x=x, y=y)}")
    print(f"Cosine: {cosine(x=x, y=y)}")

