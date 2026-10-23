import pytest
import math
import numpy as np


from kinlens.metrics import Manhattan, Euclidean, Cosine
from kinlens.exceptions.metricesExceptions import (
    VectorsLengthMissMatch, EmptyVectorError,
    CosineDistanceUndefined)


@pytest.fixture
def x() -> np.ndarray:
    return np.array([1.1, 1.2, 1.3, 1.4, 1.5]) 

@pytest.fixture
def y() -> np.ndarray:
    return np.array([2.1, 2.2, 2.3, 2.4, 2.5])



def testEmptyVectorsException():
    x: np.ndarray = np.array([])
    y: np.ndarray = np.array([])

    # The metric should not accept empty vectors
    with pytest.raises(EmptyVectorError):
        euclidean = Euclidean()
        euclidean(x=x, y=y)


def testNotSameLengthsVectorsException(x: np.ndarray, y: np.ndarray):

    # The vectors must be equal-length.
    with pytest.raises(VectorsLengthMissMatch):
        manhattan = Manhattan()
        manhattan(x=x[:-1], y=y)


def testmetricEuclideanResult(x: np.ndarray, y: np.ndarray):

    euclidean = Euclidean()

    calculatedRecult: float = euclidean(x=x, y=y)
    originalResult: float = math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))

    assert calculatedRecult == originalResult



def testManhattanResult(x: np.ndarray, y: np.ndarray):

    manhattan = Manhattan()

    calculatedResult: float = manhattan(x=x, y=y)
    originalResult: float = sum(abs(a - b) for a, b in zip(x, y))

    assert calculatedResult == originalResult


def testCosineResult(x: np.ndarray, y: np.ndarray):
    cosine = Cosine()

    dotProduct = sum(a * b for a, b in zip(x, y))

    xNorm = math.sqrt(sum(xValue ** 2 for xValue in x))
    yNorm = math.sqrt(sum(yValue ** 2 for yValue in y))

    similarity = dotProduct / (xNorm * yNorm)

    originalResult = 1.0 - similarity

    calculatedResult = cosine(x=x, y=y)

    assert round(originalResult, 6) == round(calculatedResult, 6)


    
def testCosineRejectsZeroVerctors(y: np.ndarray):

    x = np.zeros_like(y)
    cosine = Cosine()

    with pytest.raises(CosineDistanceUndefined):
        cosine(x=x, y=y)