"""
Model: Classifier: Explainable K-Nearst Neighbors classifier,
it combines all three
"""

import pprint

import numpy as np

from kinlens.abstraction.metric import DistanceMetric
from kinlens.abstraction.search import Search
from kinlens.abstraction.voting import Voting
from kinlens.datasets import Dataset
from kinlens.metrics import Euclidean
from kinlens.models.classifier import ClassScore
from kinlens.models.explaination import Explanation
from kinlens.search import BruteForseSearch
from kinlens.voting import InverseDistance


class Classifier:
    def __init__(
            self, k: int,
            metric: DistanceMetric = Euclidean(),
            voting: Voting = InverseDistance(),
            search: Search | None = None,
            
                 ) -> None:
        
        self.k = k
        self.metric = metric
        self.voting = voting
        self.search = search if search is not None else BruteForseSearch(self.metric)



    def fit(self, dataset: Dataset) -> Classifier:
        self.dataset = dataset
        self.search = self.search.fit(dataset)
        return self



    def explain(self, query: np.ndarray):

        searchResult = self.search.findNeighbors(query=query, k=self.k)
        neighborClasses = self.dataset.y[searchResult.indices]
        votingResult = self.voting.weight(result=searchResult)
        scores = self.calculateClassScores(classes=neighborClasses, weights=votingResult.weights)

        prediction = self.chooseClass(scores=scores)

        totalScores = float(np.sum(scores.scores))
        if totalScores <= 0.0:
            confidence = 0.0
        else:
            predictionIndex = np.where(scores.classes == prediction)[0]
            confidence = (
                float(scores.scores[predictionIndex[0]] / totalScores)
                if predictionIndex.size > 0 else 0.0
            )

        return Explanation(
            prediction=int(prediction),
            distances=searchResult.distances,
            indices=searchResult.indices,
            labels=neighborClasses,
            weights=votingResult.weights,
            classScore={
                int(classCode): float(score)
                for classCode, score in zip(scores.classes, scores.scores)
            },
            confidence=confidence,
            query=query
            )


    def calculateClassScores(
        self, classes: np.ndarray | list[int], weights: np.ndarray
    ) -> ClassScore:
        scores: dict[int, float] = {
            classCode: 0.0 for classCode in range(len(self.dataset.classes))
        }

        for label, weight in zip(classes, weights):
            scores[int(label)] += float(weight)

        return ClassScore(
            classes=np.array(list(scores.keys()), dtype=np.int64),
            scores=np.array(list(scores.values()), dtype=np.float64)
            )



    def predict(self, query: np.ndarray) -> str:

        chosenClassEncoded: int = int(self.explain(query=query).prediction)

        encodingTable = {
            int(encoded): original
            for original, encoded in self.dataset.targetEncodingMap.items()
        }

        return encodingTable.get(chosenClassEncoded, str(chosenClassEncoded))


    def chooseClass(self, scores: ClassScore) -> int:

        if scores.scores.size == 0:
            return 0

        highestScore = float(np.max(scores.scores))
        tiedIndices = np.flatnonzero(scores.scores == highestScore)
        return int(scores.classes[tiedIndices[0]])





if __name__ == "__main__":
    from pathlib import Path

    import numpy as np

    from kinlens.metrics import Euclidean
    from kinlens.search import BruteForseSearch
    from kinlens.utils.dataset import DatasetBuilder
    from kinlens.voting import InverseDistance


    path = Path(__file__).parent.parent / "kinlens" / "samples" / "sample-larger.csv"

    dataset = DatasetBuilder(path, features=["feature_1", "feature_2"], label="target").build()

    searcher = BruteForseSearch(metric=Euclidean())

    classifier = Classifier(k=5, search=searcher)

    classifier.fit(dataset=dataset)

    explanation = classifier.explain(query=np.array([5.0, -1.0]))

    pprint.pprint(explanation)
