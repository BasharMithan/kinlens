"""
Class: Voting

Description: Voting is the step that turns the k neighbors found by the searcher into a final class prediction.
The searcher says, “these are the closest training samples”; voting says, “given their labels and distances,
which class wins?”

Date: 2026/11/02

Developer: Bashar Mithan
"""

from abc import ABC, abstractmethod

from kinlens.models.search import SearchResult
from kinlens.models.voting import VotingResult


class Voting(ABC):
    """Convert search results into vote weights.

    A voting strategy receives a search result and returns one
    non-negative weight per neighbor.
    """ 

    @property
    @abstractmethod
    def name(self) -> str:
        """A human-readable title for the voting method."""
        ...



    @abstractmethod
    def weight(self, result: SearchResult) -> VotingResult:
        """Compute a vote weight for each neighbor.

        Args:
            result: Neighbor indices and sorted distances from a search.

        Returns:
            Non-negative finite weights with the same shape as
            ``result.distances``.
        """
        ...



