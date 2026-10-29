"""
Abstract class: Search: A parent class for neighbor searching
stratigies.
Developer: Bashar Mithan.
Date: 2026, 10, 26
"""


from abc import ABC, abstractmethod

import numpy as np

from kinlens.abstraction.metric import DistanceMetric
from kinlens.datasets import Dataset
from kinlens.models.search import SearchResult
from kinlens.validation import SearchValidation


class Search(ABC):
    """Base class for neighbor-search strategies.

    A searcher is fitted on numeric training features and queried with
    unlabeled points. It returns the closest training rows and their
    distances; it does not classify, vote, scale, or explain.
    """

    def __init__(self, metric: DistanceMetric) -> None:
        self.metric = metric
        self.x: np.ndarray | None = None

        SearchValidation.validateMetric(metric=self.metric)

        

    def fit(self, data: Dataset | np.ndarray) -> Search:
        """Store training features and prepare the searcher.

        Args:
            data: A ``Dataset`` or a feature matrix with shape
                ``(n_samples, n_features)``.

        Returns:
            The fitted searcher.

        Raises:
            TypeError: If ``data`` has an unsupported type.
            ValueError: If the feature matrix is invalid.
        """

        self.x = SearchValidation.extractFeatures(data=data)
        self.x = SearchValidation.validateTrainingMatrix(self.x)

        self.buildIndex()
                
        return self


    @property
    def nSamples(self) -> np.integer:
        return self.x.shape[0] #type: ignore


    @property
    def nFeatures(self) -> np.integer:
        return self.x.shape[1] #type: ignore
        


    def query(self, q: np.ndarray, k: int | np.integer) -> SearchResult:
        """Find the ``k`` nearest training rows to one query.

        Args:
            q: Query vector with shape ``(n_features,)``. A
                ``(1, n_features)`` array is also accepted.

            k: Number of neighbors to return.

        Returns:
            Neighbor indices and distances, sorted nearest first.

        Raises:
            NotFittedError: If the searcher has not been fitted.
            TypeError: If the query or ``k`` is invalid.
        """

        x = SearchValidation.ensureFitted(self.x)

        q = SearchValidation.validateQuery(q=q, features=x)
        k = SearchValidation.validateK(k, x[0]) 

        indices, distances = self.findNeighbors(q, k)

        return SearchResult(indices=indices, distances=distances)




    def queryBatch(self, queries: np.ndarray, k: int | np.integer) -> SearchResult:
        """Find neighbors for multiple queries.

        Args:
            queries: Query matrix with shape ``(m_queries, n_features)``.

            k: Number of neighbors returned for every query.

        Returns:
            Neighbor indices and distances, each with shape
            ``(m_queries, k)``.

        Raises:
            ValueError: If the query batch or ``k`` is invalid.
        """

        x = SearchValidation.ensureFitted(self.x)

        k = SearchValidation.validateK(k=k, features=x) 

        queries = SearchValidation.validateQueryBatch(queries=queries, features=x)

        indices = np.empty((queries.shape[0], k), dtype=np.int64)
        distances = np.empty((queries.shape[0], k), dtype=np.float64)


        for row, query in enumerate(queries):
            rowIndex, rowDistance = self.findNeighbors(query, k)

            indices[row] = rowIndex
            distances[row] = rowDistance

        return SearchResult(indices=indices, distances=distances)


    @abstractmethod
    def buildIndex(self) -> None:
        ...


    @abstractmethod
    def findNeighbors(self, query: np.ndarray, k: int | np.integer) -> tuple[np.ndarray, np.ndarray]:
        """Return neighbor indices and distances for one query.

        Args:
            q: Validated query vector.

            k: Validated neighbor count.

        Returns:
            Indices and distances sorted from nearest to farthest.
        """
        ...



    
    


