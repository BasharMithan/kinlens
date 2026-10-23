from abc import ABC, abstractmethod

import numpy as np
import math



class DistanceMetric(ABC):
    "An abstract class for the distance class, `Euclidean`, `Manhattan`, ..."

    @abstractmethod
    def distance(self, x: list[float], y: list[float]) -> float:
        """Return the distance between two equal-length numeric vectors."""
        ...

    def __call__(self, x: list[float], y: list[float]) -> float:
        """Allow instances to be used like functions: ``metric(x, y)``."""
        return self.distance(x=x, y=y)


    def validate(self, x: list[float], y: list[float]) -> None:

        if len(x) != len(y):
            raise ValueError("Vectors must have the same length.")

        if len(x) == 0:
            raise ValueError("Vector must not be empty.")



class Euclidean(DistanceMetric):
    """L2 distance: sqrt(sum((x_i, - y_i)^2))"""

    def distance(self, x: list[float], y: list[float]) -> float:
        self.validate(x=x, y=y)

        return math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))


class Manhattan(DistanceMetric):
    """L1 distance: sum(|x_1 - y_i|)"""

    def distance(self, x: list[float], y: list[float]) -> float:
        self.validate(x=x, y=y)

        return sum(abs(a - b) for a, b in zip(x, y))




if __name__ == "__main__":
    x = [1.0, 2.0, 3.0]
    y = [4.0, 6.0, 8.0]

    euclidean = Euclidean()
    manhattan = Manhattan()

    print(f"Euclidean: {euclidean(x=x, y=y)}")
    print(f"Manhattan: {manhattan(x=x, y=y)}")

