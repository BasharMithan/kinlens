

class EmptyVectorError(Exception):
    "Raised when the length of a vector = 0 (empty)."
    def __init__(self, *args: object) -> None:
        super().__init__("The vector must not be empty.")


class VectorsLengthMissMatch(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__("The vectors must be length-equal.")



class CosineDistanceUndefined(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__("Cosine distance is undefined for zero vectors.")

