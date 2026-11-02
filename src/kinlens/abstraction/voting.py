"""
Class: Voting

Description: Voting is the step that turns the k neighbors found by the searcher into a final class prediction.
The searcher says, “these are the closest training samples”; voting says, “given their labels and distances, which class wins?”

Date: 2026/11/02

Developer: Bashar Mithan
"""

import numpy as np
from abc import ABC, abstractmethod


class Voting(ABC):
    """Convert neighbor distances into vote weights.

    A voting strategy receives distances sorted from nearest to
    farthest and returns one non-negative weight per neighbor.
    """ 

    @property
    @abstractmethod
    def name(self) -> str:
        """A human-readable title for the voting method."""
        ...



    @abstractmethod
    def weight(self, distances: np.ndarray) -> np.ndarray:
        """Compute a vote weight for each neighbor.

        Args:
            distances: Neighbor distances with shape ``(k,)``, sorted
                in ascending order.

        Returns:
            Non-negative finite weights with shape ``(k,)``.
        """
        ...




