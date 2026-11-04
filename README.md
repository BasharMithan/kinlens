# Kinlens

<div align="center">

[![Created Date](https://img.shields.io/github/created-at/BasharMithan/kinlens)](https://github.com/BasharMithan/Kinlens)
[![Last commit](https://img.shields.io/github/last-commit/BasharMithan/kinlens)](https://github.com/BasharMithan/Kinlens/commits/main)
[![Tests](https://github.com/BasharMithan/kinlens/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/BasharMithan/Kinlens/actions/workflows/tests.yml)

<img src="docs/kinlens-logo.svg" alt="Kinlens logo">


Kinlens is an educational Python/NumPy project exploring the building blocks
of nearest-neighbor learning. It currently provides CSV dataset loading,
Euclidean, Manhattan, and cosine distances, brute-force nearest-neighbor
search, and uniform and inverse-distance voting weights.

</div>

**Kinlens is not yet a complete K-nearest-neighbor classifier.** It does not
currently aggregate neighbor votes into class predictions, provide prediction
probabilities, or implement the planned scaling, explanations,
counterfactuals, or alternative search backends. See the [mathematical
paper](docs/paper.md) for the implemented formulas and a clearly marked
discussion of planned features.

## Quick start

Kinlens requires Python 3.14 or newer. To install the project and its
development dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --group dev
```

Run the test suite:

```bash
uv run pytest
```

## Example

The current API retrieves neighbors and computes their voting weights. It
does not yet use those weights to choose a class prediction.

```python
import numpy as np

from kinlens.metrics import Euclidean
from kinlens.search import BruteForseSearch
from kinlens.voting import InverseDistance

training_features = np.array([
    [0.0, 0.0],
    [1.0, 1.0],
    [3.0, 3.0],
])
query = np.array([0.2, 0.1])

search = BruteForseSearch(metric=Euclidean()).fit(training_features)
result = search.query(query, k=2)
votes = InverseDistance().weight(result)

print(votes.indices)    # indices into training_features
print(votes.distances)  # corresponding distances, nearest first
print(votes.weights)    # one weight per returned neighbor
```

For CSV data, use `DatasetBuilder` and pass the resulting `Dataset` to
`fit`:

```python
from pathlib import Path

from kinlens.utils.dataset import DatasetBuilder

dataset = DatasetBuilder(
    path=Path("src/kinlens/samples/sample.csv"),
    features=["feature_1", "feature_2"],
    label="target",
).build()

search.fit(dataset)
```

## Current scope

The search implementation uses exhaustive brute-force distance calculation;
it is simple and useful for learning, but evaluates every training row for
each query. Feature scaling and train/test evaluation utilities are not
currently provided, so users should choose their feature representation and
scaling deliberately.

Class-score aggregation and prediction, calibrated probabilities, a KD-tree
or approximate search backend, Gaussian voting, counterfactual explanations,
and anomaly detection are future work rather than supported features today.

## Development

The test suite uses pytest. CI runs `uv sync --group dev` followed by
`uv run pytest` on Python 3.14. The example notebook is in
[`src/kinlens/notebooks/`](src/kinlens/notebooks/).
