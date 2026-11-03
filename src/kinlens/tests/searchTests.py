import pytest

import numpy as np

from kinlens.metrics import Euclidean
from kinlens.models.search import SearchResult
from kinlens.search import BruteForseSearch



def testSearchResultOnValidDataset():
    search = BruteForseSearch(metric=Euclidean()).fit(
        np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]])
    )

    result = search.query(np.array([0.0, 0.0]), k=2)

    assert isinstance(result, SearchResult)
    np.testing.assert_array_equal(result.indices, np.array([0, 1]))
    np.testing.assert_allclose(result.distances, np.array([0.0, np.sqrt(2.0)]))


def testSearchBatchReturnsSearchResult():
    search = BruteForseSearch(metric=Euclidean()).fit(
        np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]])
    )

    result = search.queryBatch(np.array([[0.0, 0.0], [3.0, 3.0]]), k=1)

    assert isinstance(result, SearchResult)
    assert result.indices.shape == (2, 1)
    assert result.distances.shape == (2, 1)
    np.testing.assert_array_equal(result.indices, np.array([[0], [2]]))


def testSearchRejectsInvalidFeaturesShape():
    ...


def testSearchRejectsInfiniteValues():
    ...

