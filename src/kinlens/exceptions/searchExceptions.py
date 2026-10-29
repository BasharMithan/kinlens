

class NotFittedError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__("The training data is fitted. Please run fit() and try again.")