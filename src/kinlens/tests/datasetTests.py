from pathlib import Path

import pytest

from kinlens.exceptions.datasetExceptions import *
from kinlens.utils.loader import DatasetBuilder


@pytest.fixture
def samplesPath() -> Path:
    return Path(__file__).parent.parent / "samples"

@pytest.fixture
def csvFile(samplesPath) -> Path:
    return samplesPath / "sample.csv"

@pytest.fixture
def noHeaderCSVFile(samplesPath) -> Path:
    return samplesPath / "no-header.csv" 

@pytest.fixture
def duplicatedHeader(samplesPath) -> Path:
    return samplesPath / "duplicated.csv"

@pytest.fixture
def nanCSV(samplesPath) -> Path:
    return samplesPath / "nan.csv"



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


def testDatasetRejectsNanValues(nanCSV):
    builder = DatasetBuilder(
        path=nanCSV,
        features=["feature_1", "feature_2"],
        label="target"
    )

    with pytest.raises(MissingValuesError):
        builder.build()


def testDatasetRejectsMissingValuesInUnusedCSVColumn(tmp_path: Path):
    csvFile = tmp_path / "missing-unused-column.csv"
    csvFile.write_text(
        "feature_1,target,unused\n0.2,A,ok\n0.4,B,NA\n",
        encoding="utf-8",
    )
    builder = DatasetBuilder(
        path=csvFile,
        features=["feature_1"],
        label="target",
    )

    with pytest.raises(MissingValuesError, match="unused"):
        builder.build()


def testDatasetRejectsMissingCellsInShortCSVRows(tmp_path: Path):
    csvFile = tmp_path / "short-row.csv"
    csvFile.write_text(
        "feature_1,target,unused\n0.2,A,ok\n0.4,B\n",
        encoding="utf-8",
    )
    builder = DatasetBuilder(
        path=csvFile,
        features=["feature_1"],
        label="target",
    )

    with pytest.raises(MissingValuesError, match="unused"):
        builder.build()