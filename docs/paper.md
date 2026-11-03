# Kinlens: A Mathematical Tour of Nearest-Neighbor Learning

## Abstract

Kinlens is a Python and NumPy project for exploring the mathematics behind
nearest-neighbor methods. Its implemented core loads labeled tabular data,
computes Euclidean, Manhattan, and cosine distances, retrieves the nearest
training rows with brute-force search, and assigns uniform or inverse-distance
weights to those neighbors. The search layer returns a `SearchResult` containing
both training-row indices and distances.

The repository is a developing learning project, not yet a complete
classification product. In particular, it does not currently aggregate votes
into a predicted class, expose class probabilities, scale features, generate
explanations or counterfactuals, or provide the planned alternative search
backends. This paper separates the implemented mathematics from proposed
extensions so that the design roadmap is not mistaken for shipped behavior.

## 1. Problem setup

Represent a training set by a feature matrix

$$
X =
\begin{bmatrix}
x_1^\top \\
x_2^\top \\
\vdots \\
x_n^\top
\end{bmatrix}
\in \mathbb{R}^{n \times d},
\qquad
y = (y_1,\ldots,y_n),
$$

where each row $x_i \in \mathbb{R}^d$ is one training sample and $y_i$ is its
class label. Given a query $q \in \mathbb{R}^d$, a distance function
$\delta(x_i,q)$ measures how far each training row is from the query. For a
chosen neighbor count $k$, the search stage selects the $k$ rows with the
smallest distances.

This decomposition is useful because the distance rule, the search algorithm,
and the way neighbors contribute votes are distinct choices. Kinlens currently
implements these pieces as interchangeable distance, search, and voting
strategies. It does not yet implement the final class-score aggregation step.

## 2. Implemented distance functions

For equal-length vectors $x,q \in \mathbb{R}^d$, Kinlens implements three
distance functions. The metric interface checks that its two arguments have
equal shapes and are non-empty when called directly.

### 2.1 Euclidean distance

The Euclidean, or $L_2$, distance is

$$
\delta_2(x,q)
= \lVert x-q \rVert_2
= \sqrt{\sum_{j=1}^{d}(x_j-q_j)^2}.
$$

The implementation evaluates this with NumPy's vector norm.

### 2.2 Manhattan distance

The Manhattan, or $L_1$, distance is

$$
\delta_1(x,q)
= \lVert x-q \rVert_1
= \sum_{j=1}^{d} |x_j-q_j|.
$$

The implementation sums the absolute coordinate-wise differences.

### 2.3 Cosine distance

For non-zero vectors, cosine distance is one minus cosine similarity:

$$
\delta_{\cos}(x,q)
= 1 - \frac{x^\top q}{\lVert x\rVert_2\lVert q\rVert_2}.
$$

It emphasizes vector orientation rather than magnitude. The implementation
raises a specific exception if either vector has zero norm, because cosine
similarity is undefined in that case. In exact arithmetic, this distance lies
between $0$ and $2$. Unlike Euclidean and Manhattan distance, cosine distance
does not generally satisfy the triangle inequality; the current exhaustive
search does not require that property.

The project does not currently implement Minkowski distance, a learned
feature-weighted distance, or feature scaling. Consequently, the units and
scales of the input columns directly affect Euclidean and Manhattan
comparisons. Cosine distance has different scale behavior, but it does not
remove the need to choose a representation appropriate to the data.

## 3. Implemented neighbor search

For each query, `BruteForseSearch` computes a distance from the query to every
training row:

$$
d_i = \delta(x_i,q), \qquad i \in \{1,\ldots,n\}.
$$

It uses NumPy's partial partitioning operation to identify the $k$ smallest
distances, then sorts those selected neighbors by distance. The result contains
the selected training-row indices and the corresponding sorted distances:

$$
\texttt{SearchResult.indices} = (i_{(1)},\ldots,i_{(k)}),
\qquad
\texttt{SearchResult.distances} =
(d_{i_{(1)}},\ldots,d_{i_{(k)}}),
$$

where $d_{i_{(1)}} \leq \cdots \leq d_{i_{(k)}}$. Indices refer to rows of the
matrix passed to `fit`; they are not class labels. The implementation accepts
$1 \leq k \leq n$ and validates that fitted features and queries have
appropriate shapes and finite numeric values.

The current query algorithm evaluates all $n$ distances, so for a
$d$-dimensional query its distance work is $O(nd)$. Partial selection avoids
fully sorting all $n$ distances, though the selected neighbors are sorted
afterward. The batch-query method currently loops over queries and applies
single-query search; it does not yet use a vectorized distance matrix or a
separate index structure.

The `SearchResult` data class can represent either one query, with arrays of
shape $(k,)$, or a batch, with arrays of shape $(m,k)$. It checks that indices
and distances have matching shapes.

## 4. Implemented voting weights

Voting strategies receive a `SearchResult` and return a weight for each
neighbor, with the same shape as its distance array. They do not currently
combine weights by class or select a prediction.

### 4.1 Uniform voting

Uniform voting gives each neighbor weight

$$
w_i = 1.
$$

For a query with $k$ neighbors, all neighbors therefore contribute equally if
a later component sums their weights by class.

### 4.2 Inverse-distance voting

Inverse-distance voting computes

$$
w_i = \frac{1}{(d_i+\varepsilon)^p},
$$

where $d_i$ is the search distance, $\varepsilon > 0$ is a small offset, and
$p \geq 0$ is the configured exponent. The defaults are
$\varepsilon=10^{-12}$ and $p=1$. Larger $p$ makes the weights decay more
quickly with distance. When $p=0$, all weights are $1$. When a distance is
exactly zero and $p>0$, the weight is large but finite due to $\varepsilon$;
the implementation does not give exact matches a separate special-case rule.

These are raw weights, not normalized probabilities. No implemented component
currently turns them into a probability distribution over classes.

## 5. Data preparation implemented today

`DatasetBuilder` reads a delimited text file, checks its headers and selected
columns, rejects duplicate feature names and target leakage, and checks for
missing-value tokens. The selected feature columns are converted to a
floating-point matrix. Target labels are encoded as integer class IDs using
the sorted unique labels, and the dataset retains the mapping from original
labels to IDs.

This loader does not perform train/test splitting, fit a scaler, impute missing
data, or infer a suitable metric. The search validation layer accepts either a
`Dataset` or a NumPy feature matrix; the search itself uses only the feature
matrix and does not use target labels.

## 6. What is not implemented

The following items appear in the project's design notes or roadmap, but are
not currently implemented as working Kinlens features.

### 6.1 Class prediction and probabilities

A conventional weighted $k$-nearest-neighbor classifier would aggregate the
weights for each class $c$:

$$
S_c(q) = \sum_{i \in N_k(q)} w_i\,\mathbf{1}[y_i=c],
$$

where $N_k(q)$ is the set of selected neighbor rows and $\mathbf{1}[\cdot]$ is
the indicator function. A prediction could then be chosen by
$\arg\max_c S_c(q)$, subject to a defined tie-breaking rule. Normalizing the
scores,

$$
P(c\mid q) =
\frac{S_c(q)}{\sum_{c'}S_{c'}(q)},
$$

would produce vote proportions, but should not automatically be interpreted
as calibrated statistical probabilities.

Kinlens currently has no classifier class that performs this aggregation. The
equations above describe the natural next layer; they are not a claim that
`predict`, `predict_proba`, class-score aggregation, or tie-breaking are
available APIs.

### 6.2 Feature preprocessing and evaluation

Standardization, min-max scaling, train/test splitting, stratified
cross-validation, classification reports, confusion matrices, and $k$-sweep
evaluation are planned in the architecture notes but are not implemented.
These additions matter mathematically: for coordinate-based distances,
rescaling a feature changes its contribution to neighbor ordering. Any learned
preprocessor must be fit using training data only to avoid leaking information
from evaluation data.

### 6.3 Additional metrics and voting rules

Minkowski distance,

$$
\delta_p(x,q) =
\left(\sum_{j=1}^{d}|x_j-q_j|^p\right)^{1/p},
\qquad p\geq 1,
$$

is a planned extension; $p=1$ gives Manhattan distance and $p=2$ gives
Euclidean distance. It is not currently an implemented metric.

A Gaussian voting rule of the form

$$
w_i = \exp\left(-\frac{d_i^2}{2\sigma^2}\right),
\qquad \sigma>0,
$$

is also described in the design notes but is not implemented. The implemented
voting choices are only uniform and inverse-distance weighting.

### 6.4 Alternative search, explanations, and other tasks

The design notes propose a KD-tree search backend, benchmarks, per-prediction
explanations, counterfactual explanations, anomaly detection, approximate
search, and KNN regression. These are future directions, not current
capabilities:

- A KD-tree would partition feature space and prune regions that cannot
  contain a closer neighbor. No KD-tree backend is present.
- A counterfactual feature would seek a modified query that changes the
  predicted class, ideally while minimizing a specified change cost. Neither
  the classifier nor a counterfactual optimizer is present, and no change
  cost has been established.
- A proposed anomaly score uses a neighbor distance, such as the distance to
  the $k$-th neighbor or the mean of the $k$ neighbor distances. No anomaly
  detector or thresholding rule is present.
- Approximate nearest-neighbor indexing and KNN regression are also not
  implemented.

## 7. Scope and limitations

The present code provides building blocks for an educational nearest-neighbor
workflow, rather than a complete end-to-end KNN classifier. Brute-force search
is straightforward and can use each implemented distance function, but it
computes distances to all training rows for every query. There is no
preprocessing pipeline to compensate for differences in feature scale, and
there is no classifier layer to turn returned neighbors and weights into a
prediction.

The design notes express broader goals for explainability and extensibility.
Those goals should be read as planned work unless a feature is explicitly
listed above as implemented. The mathematical formulas for unimplemented
features in this paper are specifications or common formulations, not
evidence of tested or shipped behavior.
