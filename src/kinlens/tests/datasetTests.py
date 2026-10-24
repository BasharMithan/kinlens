from pathlib import Path

import pytest

from kinlens.exceptions.datasetExceptions import *
from kinlens.utils.loader import DatasetBuilder


@pytest.fixture
def csvFile() -> Path:
    return Path("src/kinlens/samples/sample.csv")

@pytest.fixture
def noHeaderCSVFile() -> Path:
    return Path("src/kinlens/samples/no-header.csv")

@pytest.fixture
def duplicatedHeader() -> Path:
    return Path("src/kinlens/samples/duplicated.csv")



def testDatasetRejectsNoCSVHeader(noHeaderCSVFile: Path):

    builder = DatasetBuilder(
        path=noHeaderCSVFile, features=["feature_1"],
        label="target")

    with pytest.raises(NoCSVHeaders):
        builder.build()



def testDatasetRejectsUnknownTargetFeatures(csvFile):
    builder = DatasetBuilder(
        path=csvFile,
        features=["feature_3"],
        label="target")

    with pytest.raises(FeatureVectorDoesNotExitst):
        builder.build()


def testDatasetRejectsUnknownTarget(csvFile):
    builder = DatasetBuilder(
        path=csvFile,
        features=["feature_2"],
        label="unkown")

    with pytest.raises(TargetVectorDoesNotExist):
        builder.build()



def testDatasetDetectsDataLeakage(csvFile):
    builder = DatasetBuilder(
        path=csvFile,
        features=["feature_2", "target"],
        label="target")

    with pytest.raises(DataLeakageError):
        builder.build()



def testDatasetRejectsDuplicatedFeatures(csvFile):
    builder = DatasetBuilder(
        path=csvFile,
        features=["feature_2", "feature_2"],
        label="target")

    with pytest.raises(DuplicatedFeatureColumn):
        builder.build()



def testDatasetRejectsDuplicatedCSVHeaders(duplicatedHeader):
    builder = DatasetBuilder(
        path=duplicatedHeader,
        features=["feature_1", "feature_2"],
        label="target")

    with pytest.raises(DuplicatedCSVHeaders):
        builder.build()





def testBuildingDataset(csvFile):

    builder = DatasetBuilder(
        path=csvFile, features=["feature_2", "feature_1"], label="target"
    ).build()

    assert builder.x.shape == (3, 2)
    assert builder.x[0].tolist() == [5.0, 0.2]
    assert builder.y.tolist() == [0, 1, 0]
    assert builder.classes == ["A", "B"]
    assert builder.name == "sample"