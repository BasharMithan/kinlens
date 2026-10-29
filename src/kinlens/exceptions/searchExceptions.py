

class NotFittedError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__("The training data is not fitted. Please run fit() and try again.")