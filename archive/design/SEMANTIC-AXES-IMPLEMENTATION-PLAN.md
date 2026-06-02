# Plan: Semantic Axes Interpretation Layer

## Context

The user has two projects that need to connect:
- **semantic-pole** (`~/Code/semantic-pole`): Generates pole corpora as JSONL files (e.g., `professionalism.jsonl` + `casualness.jsonl`). Each record: `{"text": "...", "metadata": {"concept": "pole_name", ...}}`.
- **model-equality-testing**: Has quantum-inspired metrics (density matrices, trace distance, entropy) on SBERT embeddings for detecting LLM distributional differences.

**Goal**: Build an interpretation layer so that after quantum metrics detect a difference ("trace distance = 0.23"), semantic axes explain *what* differs ("professionalism +0.54, formality +0.38").

**Key constraint**: Semantic axes are an **interpretation layer only** — they don't replace PCA or density matrices.

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
    pole_a_name: str             # "professionalism" (negative direction)
    pole_b_name: str             # "casualness" (positive direction)
    axis_vector: np.ndarray      # (d,) unit vector, pole_a → pole_b
    pole_a_centroid: np.ndarray  # (d,) mean embedding of pole A corpus
    pole_b_centroid: np.ndarray  # (d,) mean embedding of pole B corpus
    embedding_model: str         # "all-mpnet-base-v2"
    embedding_dim: int           # 768
    metadata: dict               # creation info (paths, corpus sizes, timestamp)
```

### AxisProjectionResult
Per-axis comparison between two samples:
```python
@dataclass
class AxisProjectionResult:
    axis_name: str
    pole_a_name: str
    pole_b_name: str
    mean_a: float           # mean projection score, sample A
    mean_b: float           # mean projection score, sample B
    delta: float            # mean_a - mean_b
    std_a: float
    std_b: float
    cohens_d: float         # effect size
    t_pvalue: float         # two-sample t-test p-value
    wasserstein: float      # 1D Wasserstein distance between projection distributions
    n_a: int
    n_b: int
```

### SemanticInterpretation
Collection of per-axis results with summary methods:
```python
@dataclass
class SemanticInterpretation:
    results: List[AxisProjectionResult]
    embedding_model: str

    def sorted_by_effect(self) -> List[AxisProjectionResult]   # |Cohen's d| descending
    def summary(self, top_k=5) -> str                          # human-readable report
```

---

## Functions

### Loading Axes

**`load_axis_from_jsonl(pole_a_path, pole_b_path, axis_name, embedding_model="all-mpnet-base-v2", batch_size=32) → SemanticAxis`**
- Load both JSONL files, extract `"text"` field from each record
- Extract pole names from `metadata.concept` in the first record of each file
- Embed all texts using `EmbeddingModel` from `embeddings.py` (reuse existing class)
- Compute centroids: `centroid = mean(embeddings, axis=0)`
- Compute axis vector: `axis = (centroid_b - centroid_a) / ||centroid_b - centroid_a||`
- Store metadata: paths, corpus sizes, timestamp

**`load_axes_from_config(config_path, embedding_model="all-mpnet-base-v2") → List[SemanticAxis]`**
- Load semantic-pole axis config JSON (has `poles` array with `name` and `output_file`)
- Resolve `output_file` paths relative to config file location
- Pair poles (first = pole_a, second = pole_b) and call `load_axis_from_jsonl` for each pair
- Return list of axes

### Serialization

**`save_axes(axes, path)`** — Save list of computed axes to `.npz` for reuse (avoids re-embedding corpora).

**`load_axes(path) → List[SemanticAxis]`** — Load previously saved axes.

Format: numpy `.npz` with arrays + JSON-encoded metadata string.

### Projection

**`project_onto_axis(embeddings, axis) → np.ndarray`**
- `scores = embeddings @ axis.axis_vector` → (N,) array
- Positive = toward pole_b, negative = toward pole_a
- Validates dimension match

**`project_onto_axes(embeddings, axes) → np.ndarray`**
- Stack axis vectors into matrix, single matmul: `scores = embeddings @ axis_matrix.T` → (N, K)
- Returns (N, K) array where K = number of axes

### Interpretation

**`interpret_difference(embeddings_a, embeddings_b, axes) → SemanticInterpretation`**
- Project both samples onto all axes
- For each axis compute: means, delta, std, Cohen's d, t-test (scipy.stats.ttest_ind), Wasserstein (scipy.stats.wasserstein_distance)
- Return SemanticInterpretation with all results

**`interpret_samples(sample_a, sample_b, axes, _precomputed_embeddings=None) → SemanticInterpretation`**
- Convenience wrapper accepting CompletionSample objects
- Embeds if needed, otherwise uses precomputed embeddings
- Follows the `_precomputed_embeddings` pattern from `tests.py`

### Summary Output

**`SemanticInterpretation.summary(top_k=5)`** generates:
```
Semantic Interpretation (3 axes, sorted by effect size):

  professionalism-casualness:  Δ = +0.54  d = 0.82  p < 0.001
    Sample A is more toward casualness (+0.54)

  formality-informality:       Δ = +0.38  d = 0.61  p = 0.003
    Sample A is more toward informality (+0.38)

  technical-layperson:         Δ = -0.21  d = 0.31  p = 0.12
    No significant difference (p > 0.05)
```

---

## Integration Pattern

The interpretation reuses the same SBERT embeddings that quantum metrics already compute. Typical workflow:

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
prof_axis = load_axis_from_jsonl("professionalism.jsonl", "casualness.jsonl", "prof-casual")
interp = interpret_difference(emb_a, emb_b, [prof_axis])
print(interp.summary())
```

No changes to quantum_metrics.py, tests.py, algorithm.py, or pvalue.py.

---

## Implementation Phases

### Phase 1: Core data structures and loading
1. `SemanticAxis` dataclass with validation
2. `_load_jsonl_texts()` helper to read JSONL and extract text
3. `load_axis_from_jsonl()` — load corpora, embed, compute axis vector
4. `load_axes_from_config()` — load from semantic-pole project config

### Phase 2: Projection and interpretation
5. `project_onto_axis()` / `project_onto_axes()` — dot product projection
6. `AxisProjectionResult` dataclass
7. `interpret_difference()` — per-axis statistics (means, Cohen's d, t-test, Wasserstein)
8. `SemanticInterpretation` with `sorted_by_effect()` and `summary()`

### Phase 3: Serialization and convenience
9. `save_axes()` / `load_axes()` — npz serialization
10. `interpret_samples()` — CompletionSample convenience wrapper

---

## Dependencies

- `scipy.stats` — `ttest_ind`, `wasserstein_distance` (scipy is already used by the project)
- `sentence-transformers` — already a dependency (used by `embeddings.py`)
- No new dependencies required

---

## Verification

1. **Unit test**: Create synthetic axis (known direction), project known embeddings, verify scores are correct
2. **Round-trip test**: `save_axes()` → `load_axes()` → verify vectors match
3. **Integration test with real data**: Load professionalism/casualness corpora from `~/Code/semantic-pole/output/`, build axis, project LLM completions from the dataset, verify professionalism axis separates formal vs informal prompts
4. **End-to-end**: Run quantum detection (trace distance) + semantic interpretation on fp32 vs int8 Llama-3-8B, verify interpretable output
