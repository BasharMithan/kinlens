
import numpy as np

from kinlens.abstraction.search import Search
from kinlens.metrics import DistanceMetric, Manhattan
from kinlens.validation import SearchValidation


class BruteForseSearch(Search):
    def __init__(self, metric: DistanceMetric) -> None:
        super().__init__(metric)

    def findNeighbors(self, query: np.ndarray, k: int | np.integer) -> tuple[np.ndarray, np.ndarray]:

        x = SearchValidation.ensureFitted(self.x)

        distances = np.array([self.metric.distance(x=sample, y=query) for sample in x], dtype=np.float64)
        candidatesIndices = np.argpartition(distances, k - 1)[:k]
        order = np.argsort(distances[candidatesIndices], kind="stable")

        neighborDistances = candidatesIndices[order]

        return neighborDistances, distances[neighborDistances]


    def buildIndex(self) -> None:
        return None



if __name__ == "__main__":
    import sys
    from pathlib import Path

    from kinlens.utils.loader import DatasetBuilder

    

    sys.path.append("..")

    path = Path(__file__).resolve().parents[0] / "samples" / "sample.csv"

    builder = DatasetBuilder(path=path, features=["feature_1", "feature_2"], label="target")
    dataset = builder.build()

    
    st = BruteForseSearch(metric=Manhattan())
    searcher = st.fit(data=dataset)

    q = np.array([6.0, 0.3])  
    result = searcher.query(q, k=np.int64(2))