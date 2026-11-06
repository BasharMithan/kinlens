import numpy as np

from kinlens.classifier import Classifier
from kinlens.datasets import Dataset
from kinlens.voting import Uniform


def testExplanationIncludesScoreForEveryClass():
    dataset = Dataset(
        x=np.array([[10.0], [0.0], [1.0], [2.0], [3.0], [4.0]]),
        y=np.array([2, 0, 0, 1, 0, 1]),
        features=["feature"],
        classes=["A", "B", "C"],
        name="three-classes",
        targetEncodingMap={"A": 0, "B": 1, "C": 2},
    )
    classifier = Classifier(k=5, voting=Uniform()).fit(dataset)

    explanation = classifier.explain(query=np.array([0.1]))

    assert explanation.classScore == {0: 3.0, 1: 2.0, 2: 0.0}
    assert sum(explanation.classScore.values()) == sum(explanation.weights)
