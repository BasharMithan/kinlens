"""
Model: Classifier: Explainable K-Nearst Neighbors classifier,
it combines all three
"""

import pprint

import numpy as np

from kinlens.validation import DatasetValidation
from kinlens.abstraction.metric import DistanceMetric
from kinlens.abstraction.search import Search
from kinlens.abstraction.voting import Voting

from kinlens.metrics import Euclidean, Cosine, Manhattan
from kinlens.search import BruteForseSearch
from kinlens.voting import Uniform, InverseDistance

from kinlens.datasets import Dataset
from kinlens.models.explaination import Explanation
from kinlens.models.classifier import ClassScore, Prediction

from kinlens.utils.encode import ClassEncoder

search = BruteForseSearch(metric=Euclidean())

class Classifier:
    def __init__(
            self, k: int,
            search: Search = search,
            metric: DistanceMetric = Euclidean(),
            voting: Voting = InverseDistance()
                 ) -> None:
        
        self.k = k
        self.metric = metric
        self.search = search
        self.voting = voting



    def fit(self, dataset: Dataset) -> "Classifier":
        self.dataset = dataset
        self.search = self.search.fit(dataset)
        return self



    def explain(self, query: np.ndarray):

        searchResult = self.search.findNeighbors(query=query, k=self.k)
        neighborClasses = self.dataset.y[searchResult.indices]
        votingResult = self.voting.weight(result=searchResult)
        scores = self.calculateClassScores(classes=neighborClasses, weights=votingResult.weights)
        topClasses = self.chooseClass(scores=scores)

        prediction = topClasses[0][0]

        totalScores = sum(scores.scores)

        confidence: float = (scores.scores[np.where(scores.classes == prediction)[0][0]] / totalScores if totalScores > 0 else 0.0) #type: ignore

        return Explanation(
            prediction=prediction,
            distances=searchResult.distances,
            indices=searchResult.indices,
            labels=neighborClasses,
            weights=votingResult.weights,
            classScore=topClasses[1][0],
            confidence=confidence,
            query=query
            )


    def calculateClassScores(self, classes: np.ndarray | list[str | int], weights: np.ndarray) -> ClassScore:
        scores: dict[int, float] = {int(label): 0.0 for label in classes}

        for label, weight in zip(classes, weights):
            scores[int(label)] += float(weight)

        return ClassScore(classes=np.array(list(scores.keys()), dtype=np.int64), scores=np.array(list(scores.values()), dtype=np.float64))



    def predict(self, query: np.ndarray) -> str:

        choosenClassEncoded: int = self.explain(query=query).prediction

        encodingTable = self.dataset.targetEncodingMap.items()

        decodedClass: str = [original for original, encoded in encodingTable if encoded == choosenClassEncoded][0]

        return decodedClass


    def chooseClass(self, scores: ClassScore, top: int = 3) -> np.ndarray:

        highestScoreIndeces: np.ndarray =  np.argsort(scores.scores)[-top:][::-1]


        return np.array([
        [scores.classes[index] for index in highestScoreIndeces],
        [scores.scores[index] for index in highestScoreIndeces]
        ])





if __name__ == "__main__":
    from pathlib import Path

    import numpy as np

    from kinlens.metrics import Euclidean
    from kinlens.models.voting import VotingResult
    from kinlens.search import BruteForseSearch
    from kinlens.utils.dataset import DatasetBuilder
    from kinlens.voting import InverseDistance


    path = Path(__file__).parent.parent / "kinlens" / "samples" / "sample-larger.csv"
    builder = DatasetBuilder(path, features=["feature_1", "feature_2"], label="target")
    dataset = builder.build()
    searcher = BruteForseSearch(metric=Euclidean())

    classifier = Classifier(k=5, search=searcher)

    classifier.fit(dataset=dataset)

    prediction = classifier.predict(query=np.array([5.0, -1.0]))

    pprint.pprint(prediction)

