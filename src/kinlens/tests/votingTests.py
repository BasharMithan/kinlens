import numpy as np

from kinlens.models.search import SearchResult
from kinlens.voting import InverseDistance, Uniform


def testUniformReceivesSearchResult():
    result = SearchResult(
        indices=np.array([0, 1]),
        distances=np.array([0.5, 1.0]),
    )

    weights = Uniform().weight(result)

    np.testing.assert_array_equal(weights, np.array([1.0, 1.0]))


def testInverseDistanceReceivesSearchResult():
    result = SearchResult(
        indices=np.array([0, 1]),
        distances=np.array([0.5, 1.0]),
    )

    weights = InverseDistance(epsilon=0.1).weight(result)

    np.testing.assert_allclose(weights, np.array([1 / 0.6, 1 / 1.1]))


def testVotingPreservesBatchedSearchResultShape():
    result = SearchResult(
        indices=np.array([[0, 1], [1, 0]]),
        distances=np.array([[0.5, 1.0], [1.5, 2.0]]),
    )

    assert Uniform().weight(result).shape == result.distances.shape
    assert InverseDistance().weight(result).shape == result.distances.shape
