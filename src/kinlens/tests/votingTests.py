import numpy as np
import pytest

from kinlens.models.search import SearchResult
from kinlens.models.voting import VotingResult
from kinlens.voting import InverseDistance, Uniform


def testUniformHasName():
    assert Uniform().name == "Uniform"


def testUniformReceivesSearchResult():
    result = SearchResult(
        indices=np.array([0, 1]),
        distances=np.array([0.5, 1.0]),
    )

    weights = Uniform().weight(result).weights

    np.testing.assert_array_equal(weights, np.array([1.0, 1.0]))


def testUniformReturnsOnesForBatchedSearchResult():
    result = SearchResult(
        indices=np.array([[0, 1], [1, 0]]),
        distances=np.array([[0.5, 1.0], [1.5, 2.0]]),
    )

    weights = Uniform().weight(result).weights

    np.testing.assert_array_equal(weights, np.ones((2, 2)))


def testInverseDistanceReceivesSearchResult():
    result = SearchResult(
        indices=np.array([0, 1]),
        distances=np.array([0.5, 1.0]),
    )

    weights = InverseDistance(epsilon=0.1).weight(result).weights

    np.testing.assert_allclose(weights, np.array([1 / 0.6, 1 / 1.1]))


def testInverseDistanceUsesConfiguredPower():
    result = SearchResult(
        indices=np.array([0, 1]),
        distances=np.array([0.5, 1.0]),
    )

    weights = InverseDistance(power=2, epsilon=0.1).weight(result).weights

    np.testing.assert_allclose(weights, np.array([1 / 0.6**2, 1 / 1.1**2]))


def testInverseDistanceWithZeroPowerReturnsUnitWeights():
    result = SearchResult(
        indices=np.array([0, 1]),
        distances=np.array([0.5, 1.0]),
    )

    weights = InverseDistance(power=0).weight(result).weights

    np.testing.assert_array_equal(weights, np.ones(2))


def testInverseDistanceReturnsFiniteWeightForZeroDistance():
    result = SearchResult(
        indices=np.array([0]),
        distances=np.array([0.0]),
    )
    voting = InverseDistance(epsilon=0.1)

    weights = voting.weight(result).weights

    np.testing.assert_allclose(weights, np.array([10.0]))
    assert np.all(np.isfinite(weights))


@pytest.mark.parametrize("power", [-1, -2])
def testInverseDistanceRejectsNegativePower(power):
    with pytest.raises(ValueError):
        InverseDistance(power=power)


@pytest.mark.parametrize("epsilon", [0.0, -0.1])
def testInverseDistanceRejectsNonPositiveEpsilon(epsilon):
    with pytest.raises(ValueError):
        InverseDistance(epsilon=epsilon)


def testInverseDistanceNameIncludesPower():
    assert InverseDistance(power=2).name == "inverse-distance-p2"


def testVotingPreservesBatchedSearchResultShape():
    result = SearchResult(
        indices=np.array([[0, 1], [1, 0]]),
        distances=np.array([[0.5, 1.0], [1.5, 2.0]]),
    )

    uniformWeights = Uniform().weight(result).weights
    inverseDistanceWeights = InverseDistance(epsilon=0.1).weight(result).weights

    assert uniformWeights.shape == result.distances.shape
    assert inverseDistanceWeights.shape == result.distances.shape
    np.testing.assert_array_equal(uniformWeights, np.ones((2, 2)))
    np.testing.assert_allclose(
        inverseDistanceWeights,
        np.array([[1 / 0.6, 1 / 1.1], [1 / 1.6, 1 / 2.1]]),
    )


def testVotingResultPreservesSearchDataAndWeights():
    indices = np.array([[0, 1], [1, 0]])
    distances = np.array([[0.5, 1.0], [1.5, 2.0]])
    weights = np.array([[2.0, 1.0], [2 / 3, 0.5]])

    result = VotingResult(
        indices=indices,
        distances=distances,
        weights=weights,
    )

    np.testing.assert_array_equal(result.indices, indices)
    np.testing.assert_array_equal(result.distances, distances)
    np.testing.assert_array_equal(result.weights, weights)
