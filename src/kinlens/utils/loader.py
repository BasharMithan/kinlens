
from collections import defaultdict
from collections.abc import Sequence
from csv import DictReader
from pathlib import Path
from typing import Any

import numpy as np

from kinlens.datasets import Dataset
from kinlens.exceptions.datasetExceptions import (
    DataLeakageError,
    DuplicatedCSVHeaders,
    DuplicatedFeatureColumn,
    FeatureVectorDoesNotExitst,
    NoCSVHeaders,
    TargetVectorDoesNotExist,
)


class DatasetBuilder:
    def __init__(self, path: Path, features: list[str], label: str, delimiter: str = ",") -> None:
        self.path = path
        self.delimiter = delimiter
        self.features = features
        self.label = label


    def build(self) -> Dataset:
        zippedPairs = self.readCSV()

        dictDataset = self.buildDictDataset(zippedPairs=zippedPairs)

        targetFeatures: list[dict[str, list[Any]]] = self.filterFeatures(dictDataset=dictDataset)

        label: list[Any] = self.findTarget(target=self.label, dictDataset=dictDataset)

        return self.buildDataset(target=label, features=targetFeatures)




    def readCSV(self) ->  list[zip[tuple[str, Any]]]:
        """Reads the csv file, extracts the column names & rows,
        parses them into a list that contains a zipped tuple of keys and values pairs.
        """

        with open(self.path, newline="", encoding="utf-8") as csvFile:

            # Produces the DictReader object to read the csv file
            reader = DictReader(f=csvFile, delimiter=self.delimiter) 

            self.columns: Sequence = reader.fieldnames or []

            self.validate(reader=reader) # Checking the validity of the csv file before parsing.

            # A pair of key, values.
            pairs: list[zip[tuple[str, Any]]] = self.buildPairs(reader=reader)

            return pairs


    def findTarget(self, target: str, dictDataset: dict[str, list[Any]]) -> list[Any]:
        """Finds the target/label in the dictDataset and returns it's values.

        Args:
            target (str): Target/label name.
            dictDataset (dict[str, list[Any]]): The dictionary that holds all the columns and their values read from the CSV file.

        Returns:
           list[Any]: The values of the target/label.
        """

        return dictDataset.get(target, [])


    def filterFeatures(self, dictDataset: dict[str, list[Any]]) -> list[dict[str, list[Any]]]:
        """Filters the target features in the `dictDataset`, adds them to a list.

        Args:
            dictDataset (dict[str, list[Any]]): The dictionary that holds the dataset.

        Returns:
            list[dict[str, list[Any]]]: A list of the target fratures and their values.
        """

        vectors: list[Any] = [dictDataset[name] for name in self.features]

        return vectors



    def validate(self, reader: DictReader) -> None:

        if reader.fieldnames is None:
            raise NoCSVHeaders()

        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise DuplicatedCSVHeaders()

        self.checkFeatureColumns()        

        if len(set(self.features)) != len(self.features):
            raise DuplicatedFeatureColumn()

        if self.label not in self.columns:
            raise TargetVectorDoesNotExist(targetVector=self.label)


        if self.label in self.features:
            raise DataLeakageError(self.label, self.features)




    def checkFeatureColumns(self) -> None:
        """Checks if the passed features exist in the columns of the csv file."""

        featuresSet: set[str] = set(self.features)
        columnsSet: set[str] = set(self.columns)

        wrongFeatures: set[str] = featuresSet - columnsSet

        if len(wrongFeatures) > 0:
            raise FeatureVectorDoesNotExitst(wrongFeatures)




    def buildPairs(self, reader: DictReader) -> list[zip[tuple[str, Any]]]:
        """Builds the pair of (key, value) and stores them into a list.

        Args:
            reader (DictReader): The IO csv reader.

        Returns:
            list[zip[tuple[str, Any]]]: The list containing the (key, value) pairs.
        """

        zippedPairs: list[zip[tuple[str, Any]]] = []
        # Each row = {key: value}. Reader = [{key1: value1}, {key2: value2}, ... , {keyN: valueN}]
        for row in reader:
            zippedPairs.append(zip(row.keys(), row.values()))

        # [{key1: value1}, {key2: value2}, ... , {keyN: valueN}] -> [(k1, v1), (k2, v2), ..., (kN, vN)]
        return zippedPairs 



    def buildDictDataset(self, zippedPairs: list[zip[tuple[str, Any]]]) -> dict[str, Any]:
        """Assigns each unique key in the zipped tuple to key in a default dictionary (list),
        and appends each value in the pair to it's key in the default dictionany.

        Args:
            zippedPairs (list[zip[tuple[str, Any]]]): The list of (key, value) pairs.

        Returns:
            dict[str, Any]: The default dictionary that holder each unique key, and it's values.
        """

        dictDataset: dict[str, Any] = defaultdict(list)

        # zippedPairs: [(k1, v1), (k2, v2), ..., (kN, vN)] -> {unique(key1): list(values), unique(key2): (values), ..., unique(keyN): list[values]}
        for zippedPair in zippedPairs:

            # Pair: (k, v) - Keys are not unique, they're duplicated accross pairs.
            for pair in zippedPair:

                # {unique(key): list(values)} - Appends each values is keyed by a unique to a one default dictionary. 
                dictDataset.setdefault(pair[0], []).append(pair[1])

        return dict(dictDataset)



    def buildDataset(self, target: list[Any], features: list[dict[str, list[Any]]]) -> Dataset:
        """Build the `Dataset` object out of the values of the target and the values
        of the features.

        Args:
            target (list[Any]): The list of the target's values.
            features (list[list[Any]]): The features' values.

        Returns:
            Dataset: The output dataset.
        """


        x: np.ndarray = np.ascontiguousarray(np.array(features).astype(float).T)
        y, encodingMap, _ = self.encodeLabels(labels=np.array(target))
        classes: list[str] = [key for key in encodingMap]

        dataset: Dataset = Dataset(
            x=x,
            y=y,
            features=self.features,
            name=self.path.stem,
            targetEncodingMap=encodingMap,
            extras={},
            classes=classes)

        return dataset



    def encodeLabels(self, labels: np.ndarray):
        """
        Convert categorical labels to integer codes.

        Returns:
            encoded: np.ndarray of integers
            labelToCode: dict mapping original label -> integer
            codeToLabel: dict mapping integer -> original label
        """

        uniqueLabels = np.unique(labels)

        labelToCode = {str(label): i for i, label in enumerate(uniqueLabels)}
        codeToLabel = {i: str(label) for label, i in labelToCode.items()}

        encoded = np.array([labelToCode[label] for label in labels], dtype=int)
        return encoded, labelToCode, codeToLabel


        
if __name__ == "__main__":
    file = Path("src/kinlens/samples/sample.csv")
    builder = DatasetBuilder(path=file, features=["feature_2", "feature_1"], label="target")
    dataset = builder.build()
    print(dataset)



