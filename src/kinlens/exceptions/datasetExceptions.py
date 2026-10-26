

class NoCSVHeaders(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__("The dataset file is empty.")


class TargetVectorDoesNotExist(Exception):
    def __init__(self, targetVector: str) -> None:
        super().__init__(f"The target column: '{targetVector}' does not exit in the dataset.")




class FeatureVectorDoesNotExitst(Exception):
    def __init__(self, *args: object) -> None:
        features = ', '.join(list(*args))

        super().__init__(f"The features columns: ({features}) do not exits in the dataset.")



class DuplicatedFeatureColumn(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__("You inputed a column twice. Please remove the duplicated column name.")



class DataLeakageError(Exception):
    def __init__(self, label: str, features: list[str]) -> None:
        super().__init__(f"The target/label: {label} is in the features: {features}. This leads to data leakage.")



class DuplicatedCSVHeaders(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__("The provided csv file has duplicated headers.")
        


class MissingValuesError(Exception):
    def __init__(self, vector: str) -> None:
        super().__init__(f"The vector: {vector} has missing values.")
