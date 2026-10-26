"""
Module: DatasetValidation: Responsible for checking the validity of the input
CSV file, choosen lable, and the selected features like:
- Checking the features names and the target column do exist in the file.
- Checking if the CSV file is not empty (headers).
- Checking if the CSV file doesn't contain duplicated columns.
- Checking if the selected features are not duplicated.
- Checking if the target/label is not included in the features (data leak).

Developer: Bashar Mithan.
Date: 2026, 10, 26
"""

from collections.abc import Sequence
from csv import DictReader
from typing import Any

from kinlens.exceptions.datasetExceptions import (
    DataLeakageError,
    DuplicatedCSVHeaders,
    DuplicatedFeatureColumn,
    FeatureVectorDoesNotExitst,
    MissingValuesError,
    NoCSVHeaders,
    TargetVectorDoesNotExist,
)


class DatasetValidation:

    def __init__(self, features: list[str], label: str) -> None:
        self.features = features
        self.label = label


    def validateCSV(self, reader: DictReader) -> None:
        """Raises the proper exception for the CSV issue.

        Args:
            reader (DictReader): The CSV reader instance.

        Raises:
            NoCSVHeaders: When the CSV file is empty
            DuplicatedCSVHeaders: When the CSV file has duplicated column names.
            FeatureVectorDoesNotExitst: When there is a feature (x) that is not in the CSV file.
            DuplicatedFeatureColumn: When there are two features with the same name.
            TargetVectorDoesNotExist: When the label (y) is not in the CSV file.
            DataLeakageError: When the label (y) is in the features (x).
        """

        columns: Sequence[str] | None = reader.fieldnames

        if columns is None:
            raise NoCSVHeaders()

        if len(set(columns)) != len(columns):
            raise DuplicatedCSVHeaders()

        featuresSet: set[str] = set(self.features)
        columnsSet: set[str] = set(columns)

        wrongFeatures: set[str] = featuresSet - columnsSet

        if len(wrongFeatures) > 0:
            raise FeatureVectorDoesNotExitst(wrongFeatures)

        if len(set(self.features)) != len(self.features):
            raise DuplicatedFeatureColumn()

        if self.label not in columns:
            raise TargetVectorDoesNotExist(self.label)

        if self.label in self.features:
            raise DataLeakageError(label=self.label, features=self.features)

    def validateMissingValues(self, dictDataset: dict[str, list[Any]]) -> None:
        """Raise an error when any CSV column contains a missing value."""
        missingTokens: set[str] = {"", "nan", "na", "n/a", "null", "none", "?"}

        for column, values in dictDataset.items():
            for cell in values:
                if cell is None or (
                    isinstance(cell, str) and cell.strip().casefold() in missingTokens):
                    raise MissingValuesError(column)
