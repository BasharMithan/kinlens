
from typing import Any

from numpy import ndarray, ones, float64

from kinlens.abstraction.voting import Voting


class Uniform(Voting):
    "Gives every neighbor exactly one vote."

    @property
    def name(self) -> str:
        return "Uniform"


    def weight(self, distances: ndarray) -> ndarray:
        """Return equal weights for every neighbor.

        Args:
            distances: Neighbor distances with shape ``(k,)``.

        Returns:
            An array of ones with shape ``(k,)``.
        """

        return ones(distances.shape[0], dtype=float64)



