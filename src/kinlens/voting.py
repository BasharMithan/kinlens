
from numpy import ndarray, ones, float64, power

from kinlens.abstraction.voting import Voting
from kinlens.models.search import SearchResult


class Uniform(Voting):
    "Gives every neighbor exactly one vote."

    @property
    def name(self) -> str:
        return "Uniform"


    def weight(self, result: SearchResult) -> ndarray:
        """Return equal weights for every neighbor.

        Args:
            result: Neighbor indices and distances from a search.

        Returns:
            An array of ones matching ``result.distances``.
        """

        return ones(result.distances.shape, dtype=float64)




class InverseDistance(Voting):
    r"""Give closer neighbors more voting power.

    The weight is:

        $$w_i = \frac{1}{(d_i + \epsilon)^p}$$
    

    Attributes:
        power: Exponent applied to the adjusted distance.
        epsilon: Small value preventing division by zero.
    """

    def __init__(self, power: int = 1, epsilon: float = 1e-12) -> None:
        """Initialize inverse-distance voting.

        Args:
            power: Distance exponent. `1` gives standard inverse-distance
                weighting; `2` penalizes distant neighbors more strongly.

            epsilon: Small positive value added to every distance.

        Raises:
            ValueError: If ``power`` is negative or ``epsilon`` is not
                positive.
        """
        if power < 0:
            raise ValueError("power must be non-negative.")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")

        self.power = power
        self.epsilon = epsilon

    @property
    def name(self) -> str:
        """Return the strategy name."""
        return f"inverse-distance-p{self.power}"

    def weight(self, result: SearchResult) -> ndarray:
        """Compute inverse-distance weights.

        Args:
            result: Neighbor indices and distances from a search.

        Returns:
            Non-negative finite weights matching ``result.distances``.
        """

        distances = result.distances
        adjusted = distances + self.epsilon
        return 1.0 / power(adjusted, self.power)



if __name__ == "__main__":
    import numpy as np
    from kinlens.utils.dataset import DatasetBuilder
    from kinlens.search import BruteForseSearch
    from pathlib import Path
    from kinlens.metrics import Euclidean
    from kinlens.voting import InverseDistance


    path = Path(__file__).parent.parent / "kinlens" / "samples" / "sample-larger.csv"
    builder = DatasetBuilder(path, features=["feature_1", "feature_2"], label="target")
    dataset = builder.build()
    searcher = BruteForseSearch(metric=Euclidean())
    searcher.fit(dataset)
    searchResults = searcher.findNeighbors(query=np.array([0.2, 4.1]), k=3)
    print(searchResults)

    voting = InverseDistance()

    print(voting.weight(searchResults))

