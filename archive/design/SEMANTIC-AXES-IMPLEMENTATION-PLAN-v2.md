# Plan: Semantic Axes Interpretation Layer (v2, revised)

## Context

Two projects need to connect:
- **semantic-pole** (`~/Code/semantic-pole`): Generates pole corpora as JSONL files (e.g., `professionalism.jsonl` + `casualness.jsonl`). Each record: `{"text": "...", "metadata": {"concept": "pole_name", ...}}`.
- **model-equality-testing**: Has quantum-inspired metrics (density matrices, trace distance, entropy) on SBERT embeddings for detecting LLM distributional differences.

**Goal**: Build an interpretation layer so that after quantum metrics detect a difference ("trace distance = 0.23"), semantic axes explain *what* differs ("professionalism +0.54, g=0.82, p<0.001").

**Key constraint**: Semantic axes are an **interpretation layer only** — they don't replace PCA or density matrices. They can also be used independently to profile a single model or compare models regardless of quantum detection results.

**Axis design note**: Choosing which dimensions to measure and generating pole corpora is the user's responsibility, done via the `semantic-pole` CLI. A good axis pair has clear conceptual contrast, well-generated corpora (diverse, on-topic), and ideally convergence-tested stability. See the `semantic-pole` project for corpus generation.

---

## New File

**`model_equality_testing/src/semantic_axes.py`** — single new module, no changes to existing code.

---

## Data Structures

### SemanticAxis
```python
@dataclass
class SemanticAxis:
    name: str                    # "professionalism-casualness"
    negative_pole: str           # "professionalism" (negative direction)
    positive_pole: str           # "casualness" (positive direction)
    axis_vector: np.ndarray      # (d,) unit vector, negative → positive
    negative_centroid: np.ndarray # (d,) mean embedding of negative pole corpus
    positive_centroid: np.ndarray # (d,) mean embedding of positive pole corpus
    embedding_model: str         # "all-mpnet-base-v2"
    embedding_dim: int           # 768
    pole_distance: float         # ||positive_centroid - negative_centroid|| before normalization
    metadata: dict               # creation info — see Metadata Schema below
```

Validation in `__post_init__`:
- `axis_vector` is unit norm
- Dimensions match across all vectors
- `pole_distance > 1e-6` (error if poles don't separate)

#### Metadata Schema

The `metadata` dict contains provenance and quality information computed during axis construction. `pole_distance` is NOT duplicated here — it is a first-class field on `SemanticAxis`.

```python
metadata = {
    "negative_pole_path": str,       # file path to negative pole JSONL
    "positive_pole_path": str,       # file path to positive pole JSONL
    "negative_pole_samples": int,    # number of texts in negative pole corpus
    "positive_pole_samples": int,    # number of texts in positive pole corpus
    "negative_pole_intra_std": float, # mean L2 distance of embeddings from centroid (negative pole)
    "positive_pole_intra_std": float, # mean L2 distance of embeddings from centroid (positive pole)
    "separation_ratio": float,       # pole_distance / mean(intra_std_neg, intra_std_pos)
    "created_at": str,               # ISO 8601 timestamp
}
```

The `separation_ratio` is analogous to Fisher's discriminant ratio — higher values indicate better-separated poles. Values below 1.0 suggest the poles overlap substantially and the axis may have poor discriminating power.

### AxisProjectionResult
Per-axis comparison between two samples:
```python
@dataclass
class AxisProjectionResult:
    axis_name: str
    negative_pole: str       # for display: "professionalism ← → casualness"
    positive_pole: str
    mean_a: float            # mean projection score, sample A
    mean_b: float            # mean projection score, sample B
    std_a: float
    std_b: float
    sem_a: float             # standard error of mean
    sem_b: float
    delta: float             # mean_a - mean_b
    hedges_g: float          # Hedges' g (bias-corrected effect size)
    t_pvalue: float          # Welch's t-test p-value (does not assume equal variance)
    corrected_pvalue: float  # after Benjamini-Hochberg FDR correction
    wasserstein: float       # 1D Wasserstein distance
    n_a: int
    n_b: int
```

**Effect size formula**: Hedges' g uses the average-SD denominator (does not assume equal variance), then applies the bias correction factor J:
```
s_avg = sqrt((s_a² + s_b²) / 2)
d = (mean_a - mean_b) / s_avg
J = 1 - 3 / (4 * (n_a + n_b) - 9)
hedges_g = d * J
```

### SemanticInterpretation
Collection of per-axis results with summary methods:
```python
@dataclass
class SemanticInterpretation:
    results: List[AxisProjectionResult]
    embedding_model: str
    label_a: str                # "fp32", "Model A", etc.
    label_b: str                # "int8", "Model B", etc.

    def sorted_by_effect(self) -> List[AxisProjectionResult]   # |hedges_g| descending
    def summary(self, top_k=5) -> str                          # human-readable report
```

---

## Functions

### Loading Axes

**`load_axis_from_jsonl(negative_pole_path, positive_pole_path, axis_name, embedding_model="all-mpnet-base-v2", batch_size=32) → SemanticAxis`**
- Load both JSONL files, extract `"text"` field from each record
- Filter out empty texts
- Extract pole names from `metadata.concept` in the first record of each file
- Embed all texts using `EmbeddingModel` from `embeddings.py` (reuse existing class)
- L2-normalize each embedding before computing centroid (ensures cosine-similarity semantics regardless of SBERT model)
- Compute centroids: `centroid = mean(normalized_embeddings, axis=0)`
- Compute `pole_distance = ||centroid_pos - centroid_neg||`
- Error if `pole_distance < 1e-6` (poles don't separate)
- Compute axis vector: `axis = (centroid_pos - centroid_neg) / pole_distance`
- Store metadata: paths, corpus sizes, timestamp, pole_distance
- Warn if either corpus has fewer than 50 samples

### Serialization

**`save_axes(axes, path)`** — Save list of computed axes to `.npz` with version field. Avoids re-embedding corpora on subsequent runs.

**`load_axes(path) → List[SemanticAxis]`** — Load previously saved axes. Check version for forward compatibility.

Format: numpy `.npz` with arrays + JSON-encoded metadata string + integer version field.

### Projection

**`project_onto_axis(embeddings, axis) → np.ndarray`**
- `scores = embeddings @ axis.axis_vector` → (N,) array
- Positive = toward positive_pole, negative = toward negative_pole
- Validates embedding dimension matches axis dimension
- Validates embedding model name matches (warn on mismatch)

**`project_onto_axes(embeddings, axes) → np.ndarray`**
- Stack axis vectors into matrix, single matmul: `scores = embeddings @ axis_matrix.T` → (N, K)
- Returns (N, K) array where K = number of axes

### Interpretation

**`interpret_difference(embeddings_a, embeddings_b, axes) → SemanticInterpretation`**
- Project both samples onto all axes
- For each axis compute:
  - Means, standard deviations, SEM
  - Delta = mean_a - mean_b
  - Hedges' g (bias-corrected Cohen's d) using Welch's formula (does not assume equal variance)
  - Welch's t-test via `scipy.stats.ttest_ind(equal_var=False)` (does not assume equal variance)
  - 1D Wasserstein distance via `scipy.stats.wasserstein_distance`
- Apply Benjamini-Hochberg FDR correction across all axes
- Return SemanticInterpretation with all results

**`interpret_samples(sample_a, sample_b, axes, _precomputed_embeddings=None) → SemanticInterpretation`**
- Convenience wrapper accepting CompletionSample objects
- Embeds if needed, otherwise uses precomputed embeddings
- Follows the `_precomputed_embeddings` pattern from `tests.py`

### Axis Quality

**`axis_quality(axis) → dict`** — returns separation metrics stored in axis metadata:
- `pole_distance`: raw inter-pole centroid distance
- `pole_a_samples`, `pole_b_samples`: corpus sizes

**`axis_correlations(axes) → np.ndarray`** — cosine similarity matrix between axis vectors. Warns if any pair correlates above 0.7.

### Summary Output

**`SemanticInterpretation.summary(top_k=5)`** generates:
```
Semantic Interpretation (3 axes, sorted by effect size):

  professionalism ← → casualness:  Δ = +0.54  d = 0.82  p < 0.001 *
    Sample A is more toward casualness

  formality ← → informality:       Δ = +0.38  d = 0.61  p = 0.003 *
    Sample A is more toward informality

  technical ← → layperson:         Δ = -0.21  d = 0.31  p = 0.15
    No significant difference (p > 0.05 after FDR correction)

* = significant after Benjamini-Hochberg correction
```

---

## Integration Pattern

Reuses the same SBERT embeddings that quantum metrics already compute:

```python
from model_equality_testing.src.embeddings import embed_sample
from model_equality_testing.src.quantum_metrics import pca_density_matrix, fit_pca, trace_distance
from model_equality_testing.src.semantic_axes import load_axis_from_jsonl, interpret_difference

# 1. Embed once
emb_a = embed_sample(sample_a)
emb_b = embed_sample(sample_b)

# 2. Detect with quantum metrics
pca = fit_pca(np.vstack([emb_a, emb_b]), k=50)
rho_a = pca_density_matrix(emb_a, pca)
rho_b = pca_density_matrix(emb_b, pca)
td = trace_distance(rho_a, rho_b)

# 3. Interpret with semantic axes
prof_axis = load_axis_from_jsonl(
    "professionalism.jsonl", "casualness.jsonl", "prof-casual"
)
interp = interpret_difference(emb_a, emb_b, [prof_axis])
print(interp.summary())
```

No changes to quantum_metrics.py, tests.py, algorithm.py, or pvalue.py.

---

## Implementation Phases

### Phase 1: Core data structures and loading
1. `SemanticAxis` dataclass with validation (dimension, unit norm, pole distance)
2. `_load_jsonl_texts()` helper to read JSONL and extract text
3. `load_axis_from_jsonl()` — load corpora, embed, L2-normalize, compute centroids and axis vector, validate separation

### Phase 2: Projection and interpretation
4. `project_onto_axis()` / `project_onto_axes()` — dot product projection with dimension/model validation
5. `AxisProjectionResult` dataclass with SEM, Hedges' g, corrected p-value
6. `interpret_difference()` — per-axis statistics with Welch's t-test and BH-FDR correction
7. `SemanticInterpretation` with `sorted_by_effect()` and `summary()`

### Phase 3: Serialization and utilities
8. `save_axes()` / `load_axes()` — versioned npz serialization
9. `interpret_samples()` — CompletionSample convenience wrapper
10. `axis_correlations()` — cosine similarity matrix with warnings

---

## Dependencies

- `scipy.stats` — `ttest_ind`, `wasserstein_distance` (already used by the project)
- `sentence-transformers` — already a dependency (used by `embeddings.py`)
- No new dependencies required
- BH-FDR correction: implement directly (sort p-values, multiply by n_tests/rank) — no need for statsmodels

---

## Verification

1. **Unit test**: Create synthetic axis (known direction), project known embeddings, verify scores match expected dot products
2. **Round-trip test**: `save_axes()` → `load_axes()` → verify vectors match within floating-point tolerance
3. **Edge case tests**: Near-zero pole distance (should error), dimension mismatch (should error), empty corpus (should error)
4. **Integration test with real data**: Load professionalism/casualness corpora from `~/Code/semantic-pole/output/`, build axis, project LLM completions from the dataset, verify professionalism axis separates formal vs informal content
5. **Statistical tests**: Verify Hedges' g matches scipy/manual computation, verify BH correction is monotonic
6. **End-to-end**: Run quantum detection (trace distance) + semantic interpretation on fp32 vs int8 Llama-3-8B, verify interpretable output
