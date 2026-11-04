from dataclasses import dataclass

from numpy import ndarray


@dataclass
class VotingResult:
    indices: ndarray
    distances: ndarray
    weights: ndarray