import pytest
import math
import numpy as np


from kinlens.metrics import Manhattan, Euclidean, Cosine
from kinlens.exceptions.metricesExceptions import (
    VectorsLengthMissMatch, EmptyVectorError,
    CosineDistanceUndefined)


def testEmptyVectorsException():
    x: np.ndarray = np.array([])
    y: np.ndarray = np.array([])

    # The metric should not accept empty vectors
    with pytest.raises(EmptyVectorError):
        euclidean = Euclidean()
        euclidean(x=x, y=y)


def testNotSameLengthsVectorsException():
    x =  np.array([1.1, 1.2, 1.3, 1.4])
    y =  np.array([2.1, 2.2, 2.3, 2.4, 2.5])

    # The vectors must be equal-length.
    with pytest.raises(VectorsLengthMissMatch):
        manhattan = Manhattan()
        manhattan(x=x, y=y)


def testmetricEuclideanResult():

    euclidean = Euclidean()

    x: np.ndarray = np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1])
    y: np.ndarray = np.array([2.1, 4.1, 6.1, 8.1, 10.1, 12.1])

    calculatedRecult: float = euclidean(x=x, y=y)
    originalResult: float = math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))

    assert calculatedRecult == originalResult



def testManhattanResult():

    manhattan = Manhattan()

    x: np.ndarray = np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1])
    y: np.ndarray = np.array([2.1, 4.1, 6.1, 8.1, 10.1, 12.1])

    calculatedResult: float = manhattan(x=x, y=y)
    originalResult: float = sum(abs(a - b) for a, b in zip(x, y))

    assert calculatedResult == originalResult


def testCosineResult():
    cosine = Cosine()

    x: np.ndarray = np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1])
    y: np.ndarray = np.array([2.1, 4.1, 6.1, 8.1, 10.1, 12.1])

    dotProduct = sum(a * b for a, b in zip(x, y))

    xNorm = math.sqrt(sum(xValue ** 2 for xValue in x))
    yNorm = math.sqrt(sum(yValue ** 2 for yValue in y))

    similarity = dotProduct / (xNorm * yNorm)

    originalResult = 1.0 - similarity

    calculatedResult = cosine(x=x, y=y)

    assert round(originalResult, 6) == round(calculatedResult, 6)


    
def testCosineRejectsZeroVerctors():

    x: np.ndarray = np.array([0, 0, 0, 0, 0, 0])
    y: np.ndarray = np.array([0, 4.1, 6.1, 8.1, 10.1, 12.1])

    cosine = Cosine()

    with pytest.raises(CosineDistanceUndefined):
        cosine(x=x, y=y)