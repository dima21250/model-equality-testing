# Plan: Semantic Axes Interpretation Layer (v3)

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

**`load_axis_from_jsonl(negative_pole_path: str, positive_pole_path: str, axis_name: str, embedding_model: str = "all-mpnet-base-v2", batch_size: int = 32) → SemanticAxis`**
- Load both JSONL files, extract `"text"` field from each record
- Filter out empty texts; error if all texts are empty
- Extract pole names from `metadata.concept` in the first record of each file
- Embed all texts using `EmbeddingModel` from `embeddings.py` (reuse existing class)
- L2-normalize each embedding before computing centroid (ensures cosine-similarity semantics regardless of SBERT model)
- Compute centroids: `centroid = mean(normalized_embeddings, axis=0)`
- Compute intra-pole standard deviation: mean L2 distance of each embedding from its centroid
- Compute `pole_distance = ||centroid_pos - centroid_neg||`
- Compute `separation_ratio = pole_distance / mean(intra_std_neg, intra_std_pos)`
- Error if `pole_distance < 1e-6` (poles don't separate)
- Compute axis vector: `axis = (centroid_pos - centroid_neg) / pole_distance`
- Store metadata: paths, corpus sizes, intra-pole std, separation ratio, timestamp
- Warn if either corpus has fewer than 50 samples
- Warn if `separation_ratio < 1.0` (poles overlap substantially)

### Serialization

**`save_axis(axis: SemanticAxis, path: str)`** — Save a single axis to `.npz`. One file per axis for selective loading.

**`load_axis(path: str) → SemanticAxis`** — Load a single saved axis.

**`save_axes(axes: List[SemanticAxis], directory: str)`** — Save each axis as `{directory}/{axis.name}.npz`.

**`load_axes(directory: str) → List[SemanticAxis]`** — Load all `.npz` files from the directory.

Format: numpy `.npz` with:
- `version`: integer (start at 1)
- `axis_vector`, `negative_centroid`, `positive_centroid`: arrays
- `metadata_json`: JSON string with name, pole names, embedding_model, embedding_dim, pole_distance, and freeform metadata dict

On load, check `version` and handle future migrations.

### Projection

**`project_onto_axis(embeddings: np.ndarray, axis: SemanticAxis) → np.ndarray`**
- L2-normalize incoming embeddings (ensures consistency with axis construction)
- `scores = normalized_embeddings @ axis.axis_vector` → (N,) array
- Positive = toward positive_pole, negative = toward negative_pole
- Validates embedding dimension matches axis dimension
- No model name validation here (raw numpy arrays don't carry model provenance)

**`project_onto_axes(embeddings: np.ndarray, axes: List[SemanticAxis]) → np.ndarray`**
- L2-normalize incoming embeddings
- Stack axis vectors into matrix, single matmul: `scores = normalized @ axis_matrix.T` → (N, K)
- Returns (N, K) array where K = number of axes

**Normalization consistency**: Both `load_axis_from_jsonl()` and `project_onto_axis()` L2-normalize embeddings before any operations. This ensures the axis vector and projection scores live in the same cosine-similarity space, regardless of whether the caller passes raw or pre-normalized embeddings.

### Interpretation

**`interpret_difference(embeddings_a: np.ndarray, embeddings_b: np.ndarray, axes: List[SemanticAxis], label_a: str = "Sample A", label_b: str = "Sample B") → SemanticInterpretation`**
- Project both samples onto all axes
- For each axis compute:
  - Means, standard deviations, SEM
  - Delta = mean_a - mean_b
  - Hedges' g using average-SD denominator (see formula in AxisProjectionResult)
  - Welch's t-test via `scipy.stats.ttest_ind(equal_var=False)`
  - 1D Wasserstein distance via `scipy.stats.wasserstein_distance`
- Apply Benjamini-Hochberg FDR correction across all axes:
  ```python
  sorted_indices = np.argsort(pvalues)
  sorted_pvalues = pvalues[sorted_indices]
  corrected = sorted_pvalues * n_axes / np.arange(1, n_axes + 1)
  corrected = np.minimum.accumulate(corrected[::-1])[::-1]  # enforce monotonicity
  corrected = np.minimum(corrected, 1.0)  # cap at 1.0
  result = np.empty_like(corrected)
  result[sorted_indices] = corrected
  ```
- For K=1 axis, corrected p-value equals raw p-value (no correction needed)
- Return SemanticInterpretation with all results

**`interpret_samples(sample_a: CompletionSample, sample_b: CompletionSample, axes: List[SemanticAxis], label_a: str = "Sample A", label_b: str = "Sample B", embedding_model: str = None, _precomputed_embeddings: tuple = None) → SemanticInterpretation`**
- Convenience wrapper accepting CompletionSample objects
- Embeds if needed, otherwise uses precomputed embeddings
- Validates embedding model name matches axis model (warn on mismatch) — this is the correct scope for model validation since this function knows the model name
- Follows the `_precomputed_embeddings` pattern from `tests.py`

### Axis Correlations

**`axis_correlations(axes: List[SemanticAxis]) → np.ndarray`** — cosine similarity matrix between axis vectors. Warns if any pair correlates above 0.7 (indicates overlapping dimensions — interpretations may overcount differences).

### Summary Output

**`SemanticInterpretation.summary(top_k=5)`** generates:

The summary always names both poles with `←→` notation, uses the caller-provided sample labels, and states the direction from the perspective of the larger absolute delta:

```
Semantic Interpretation (3 axes, sorted by effect size):

  professionalism ← → casualness:  Δ = +0.54  g = 0.82  p < 0.001 *
    fp32 is more toward casualness than int8

  formality ← → informality:       Δ = +0.38  g = 0.61  p = 0.003 *
    fp32 is more toward informality than int8

  technical ← → layperson:         Δ = -0.21  g = 0.31  p = 0.15
    No significant difference (p > 0.05 after FDR correction)

  * significant after Benjamini-Hochberg correction (α = 0.05)
```

---

## Integration Pattern

Reuses the same SBERT embeddings that quantum metrics already compute:

```python
import numpy as np
from model_equality_testing.dataset import load_distribution
from model_equality_testing.src.embeddings import embed_sample
from model_equality_testing.src.quantum_metrics import pca_density_matrix, fit_pca, trace_distance
from model_equality_testing.src.semantic_axes import load_axis_from_jsonl, interpret_difference

# 0. Load distributions and draw samples
dist_fp32 = load_distribution(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    prompt_ids={"wikipedia_en": [0, 1, 2]},
    L=500, source="fp32", load_in_unicode=True,
)
dist_int8 = load_distribution(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    prompt_ids={"wikipedia_en": [0, 1, 2]},
    L=500, source="int8", load_in_unicode=True,
)
sample_fp32 = dist_fp32.draw_completion_sample(n=100)
sample_int8 = dist_int8.draw_completion_sample(n=100)

# 1. Embed once
emb_fp32 = embed_sample(sample_fp32)
emb_int8 = embed_sample(sample_int8)

# 2. Detect with quantum metrics
pca = fit_pca(np.vstack([emb_fp32, emb_int8]), k=50)
rho_fp32 = pca_density_matrix(emb_fp32, pca)
rho_int8 = pca_density_matrix(emb_int8, pca)
td = trace_distance(rho_fp32, rho_int8)
print(f"Trace distance: {td:.4f}")

# 3. Interpret with semantic axes
prof_axis = load_axis_from_jsonl(
    "/path/to/professionalism.jsonl",
    "/path/to/casualness.jsonl",
    "professionalism-casualness"
)
interp = interpret_difference(
    emb_fp32, emb_int8, [prof_axis],
    label_a="fp32", label_b="int8"
)
print(interp.summary())
```

No changes to quantum_metrics.py, tests.py, algorithm.py, or pvalue.py.

---

## Implementation Phases

### Phase 1: Core data structures and loading
1. `SemanticAxis` dataclass with validation (dimension, unit norm, pole distance)
2. `_load_jsonl_texts()` helper to read JSONL and extract text
3. `load_axis_from_jsonl()` — load corpora, embed, L2-normalize, compute centroids and axis vector, validate separation, compute intra-pole std and separation ratio

### Phase 2: Projection and interpretation
4. `project_onto_axis()` / `project_onto_axes()` — L2-normalize inputs, dot product projection, dimension validation
5. `AxisProjectionResult` dataclass with SEM, Hedges' g (average-SD denominator + J correction), corrected p-value
6. `_benjamini_hochberg()` helper with monotonicity enforcement
7. `interpret_difference()` — per-axis statistics with Welch's t-test and BH-FDR correction, sample labels
8. `SemanticInterpretation` with `sorted_by_effect()` and `summary()` using labels and both-pole notation

### Phase 3: Serialization and utilities
9. `save_axis()` / `load_axis()` — single axis, versioned npz
10. `save_axes()` / `load_axes()` — directory of per-axis npz files
11. `interpret_samples()` — CompletionSample convenience wrapper with model name validation
12. `axis_correlations()` — cosine similarity matrix with warnings

---

## Dependencies

- `scipy.stats` — `ttest_ind`, `wasserstein_distance` (already used by the project)
- `sentence-transformers` — already a dependency (used by `embeddings.py`)
- No new dependencies required
- BH-FDR correction: implement directly with monotonicity enforcement — no need for statsmodels

---

## Verification

1. **Unit test**: Create synthetic axis (known direction), project known embeddings, verify scores match expected dot products. Use synthetic JSONL fixtures in `tests/fixtures/` (e.g., 20 records per pole) — no dependency on `~/Code/semantic-pole/`
2. **Round-trip test**: `save_axis()` → `load_axis()` → verify vectors match within floating-point tolerance
3. **Edge case tests**: Near-zero pole distance (should error), dimension mismatch (should error), empty corpus (should error), single axis BH correction (corrected = raw)
4. **Statistical tests**: Verify Hedges' g formula against manual computation, verify BH correction produces monotonically non-decreasing corrected p-values, verify K=1 passthrough
5. **Normalization test**: Verify that passing raw vs pre-normalized embeddings to `project_onto_axis()` produces identical scores
6. **Integration test** (optional, skip if semantic-pole not present): Load professionalism/casualness corpora from `~/Code/semantic-pole/output/`, build axis, verify separation ratio > 1.0
7. **End-to-end**: Run quantum detection (trace distance) + semantic interpretation on fp32 vs int8 Llama-3-8B, verify interpretable output with correct labels and FDR markers
