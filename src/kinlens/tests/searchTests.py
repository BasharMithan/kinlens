import numpy as np
import pytest

from kinlens.exceptions.searchExceptions import NotFittedError
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


@pytest.mark.parametrize(
    "features",
    [np.array([0.0, 1.0]), np.zeros((1, 1, 2))],
)
def testSearchRejectsInvalidFeaturesShape(features):
    search = BruteForseSearch(metric=Euclidean())

    with pytest.raises(ValueError):
        search.fit(features)


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def testSearchRejectsInfiniteValues(value):
    search = BruteForseSearch(metric=Euclidean())

    with pytest.raises(ValueError):
        search.fit(np.array([[0.0, value]]))


def testSearchRejectsNanValues():
    search = BruteForseSearch(metric=Euclidean())

    with pytest.raises(ValueError):
        search.fit(np.array([[0.0, np.nan]]))


def testSearchRejectsEmptyTrainingFeatures():
    search = BruteForseSearch(metric=Euclidean())

    with pytest.raises(ValueError):
        search.fit(np.empty((0, 2)))


def testSearchRejectsNonNumericTrainingFeatures():
    search = BruteForseSearch(metric=Euclidean())

    with pytest.raises(ValueError):
        search.fit(np.array([["a", "b"]]))


def testSearchRejectsQueriesBeforeFit():
    search = BruteForseSearch(metric=Euclidean())

    with pytest.raises(NotFittedError):
        search.query(np.array([0.0, 0.0]), k=1)

    with pytest.raises(NotFittedError):
        search.queryBatch(np.array([[0.0, 0.0]]), k=1)


@pytest.mark.parametrize(
    "query",
    [
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.array([0.0]),
        np.array([0.0, np.inf]),
    ],
)
def testSearchRejectsInvalidQueries(query):
    search = BruteForseSearch(metric=Euclidean()).fit(
        np.array([[0.0, 0.0], [1.0, 1.0]])
    )

    with pytest.raises(ValueError):
        search.query(query, k=1)


@pytest.mark.parametrize("k", [0, 3])
def testSearchRejectsKOutsideTrainingSet(k):
    search = BruteForseSearch(metric=Euclidean()).fit(
        np.array([[0.0, 0.0], [1.0, 1.0]])
    )

    with pytest.raises(ValueError):
        search.query(np.array([0.0, 0.0]), k=k)


@pytest.mark.parametrize("k", [1.5, "1", True])
def testSearchRejectsNonIntegerK(k):
    search = BruteForseSearch(metric=Euclidean()).fit(
        np.array([[0.0, 0.0], [1.0, 1.0]])
    )

    with pytest.raises(TypeError):
        search.query(np.array([0.0, 0.0]), k=k)


@pytest.mark.parametrize(
    "queries",
    [
        np.array([0.0, 0.0]),
        np.array([[0.0]]),
        np.array([[0.0, np.nan]]),
    ],
)
def testSearchRejectsInvalidQueryBatches(queries):
    search = BruteForseSearch(metric=Euclidean()).fit(
        np.array([[0.0, 0.0], [1.0, 1.0]])
    )

    with pytest.raises(ValueError):
        search.queryBatch(queries, k=1)


def testSearchResultRejectsMismatchedShapes():
    with pytest.raises(ValueError):
        SearchResult(
            indices=np.array([0, 1]),
            distances=np.array([0.5]),
        )
