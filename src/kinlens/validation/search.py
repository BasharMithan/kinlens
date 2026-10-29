

import numpy as np

from kinlens.datasets import Dataset
from kinlens.exceptions.searchExceptions import NotFittedError
from kinlens.metrics import DistanceMetric


class SearchValidation:
    """Validates inputs and state for neighbor-search components.

    This class is stateless: every method receives the values it needs.
    It contains no search algorithm and no fitted state.
    """

    def __init__(self, features: np.ndarray) -> None:
        self.features = features


    @staticmethod
    def validateMetric(metric: object) -> None:

        if not isinstance(metric, DistanceMetric):
            raise TypeError(
                f"""
                The distance metric must be an instance of DistanceMetric class,
                got: {type(metric).__name__}.""")


    @staticmethod
    def ensureFitted(features: np.ndarray | None) -> np.ndarray:

        if features is None:
            raise NotFittedError()

        else: return features



    @staticmethod
    def validateTrainingMatrix(features: np.ndarray) -> np.ndarray:

        if features.ndim != 2:
            raise ValueError(f"Training features must be a 2-D array with shape (n_samples, n_features); got shape {features.shape}.")
            


        if features.shape[0] == 0:
            raise ValueError("Training features must contain at least one sample.")
            


        if not np.issubdtype(features.dtype, np.number):
            raise ValueError(f"Training metrix must be numeric; got: {features.dtype}")


        if not np.all(np.isfinite(features)):
            raise ValueError("Training features contain NaN or infinite values")



        return np.array(features, dtype=np.float64, copy=True)


    @staticmethod       
    def validateK(k: int | np.integer, features: np.ndarray) -> int | np.integer:

        if not isinstance(k, np.integer | int) or isinstance(k, bool):
            raise TypeError(f"K must be an integer; got: {type(k).__name__}")


        if not 1 <= k <= features.shape[0]:
            raise ValueError(f"K must satify 1 <= k <= {features.shape[0]}.")

        return k


    @staticmethod
    def validateQuery(q: np.ndarray, features: np.ndarray) -> np.ndarray:

        if q.ndim == 2 and q.shape[0] == 1:
            q = q.reshape(-1)

        if q.ndim != 1:
            raise ValueError(f"A query must be 1-D array with shape: ({features.shape[1]},); "
                             f"got: {q.shape}")


        if q.shape[0] != features.shape[1]:
            raise ValueError(
                f"A query must contain {features.shape[1]} features; "
                f"got: {q.shape[0]}.")


        if not all(np.isfinite(q)):
            raise ValueError("Query contains NaN or infinite values.")


        return q


    @staticmethod
    def extractFeatures(data: Dataset | np.ndarray) -> np.ndarray:
        """Extract the feature matrix from a Dataset or raw array.

        Args:
            data: A Kinlens ``Dataset`` or a numeric feature matrix.

        Returns:
            The feature matrix used for fitting.

        Raises:
            TypeError: If ``data`` is neither a ``Dataset`` nor an
                array-like object.
        """

        if isinstance(data, Dataset):
            return data.x

        if isinstance(data, np.ndarray):
            return data

        raise TypeError(
            "data must be a Dataset or a NumPy array; "
            f"got {type(data).__name__}."
        )


    @staticmethod
    def validateQueryBatch(queries: np.ndarray, features: np.ndarray) -> np.ndarray:

        queries = np.array(queries, dtype=np.float64)

        if queries.ndim != 2:
            raise ValueError(
                "queries must be a 2-D array with shape "
                f"(m_queries, {features.shape[1]}); got shape {queries.shape}."
            )

        if queries.shape[1] != features.shape[1]:
            raise ValueError(
                f"queries must contain {features.shape[1]} features; "
                f"got {queries.shape[1]}."
            )

        if not np.all(np.isfinite(queries)):
            raise ValueError("queries contains NaN or infinite values.")

        return queries



    
