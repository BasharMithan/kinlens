
from typing import Any

from numpy import ndarray, ones, float64, power

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

    def weights(self, distances: ndarray) -> ndarray:
        """Compute inverse-distance weights.

        Args:
            distances: Neighbor distances with shape ``(k,)``.

        Returns:
            Non-negative finite weights with shape ``(k,)``.
        """
        adjusted = distances + self.epsilon
        return 1.0 / power(adjusted, self.power)