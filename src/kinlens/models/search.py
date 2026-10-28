

from dataclasses import dataclass

from numpy import ndarray


@dataclass(frozen=True)
class SearchResult:
    """Neighbor indices and distances for one or more queries.

    Attributes:
        indices: Training-row indices, ordered nearest to farthest.
            Shape: ``(k,)`` for one query or ``(m, k)`` for a batch.

        distances: Distances to the corresponding neighbors.
            Shape matches ``indices`` and is sorted ascending.
    """

    indices: ndarray
    distances: ndarray

    def __post_init__(self) -> None:

        if self.indices.shape != self.distances.shape:

            raise ValueError(
                "indices and distances must have the same shape; "
                f"got {self.indices.shape} and {self.distances.shape}."
            )
