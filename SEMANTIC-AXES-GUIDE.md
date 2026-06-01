# Semantic Axes Interpretation Guide

This guide provides comprehensive documentation for using semantic axes to interpret distributional differences between LLM outputs.

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Quick Start](#quick-start)
4. [Axis Construction](#axis-construction)
5. [Quality Metrics](#quality-metrics)
6. [Statistical Methodology](#statistical-methodology)
7. [Interpretation Workflow](#interpretation-workflow)
8. [Best Practices](#best-practices)
9. [API Reference](#api-reference)
10. [Troubleshooting](#troubleshooting)

---

## Overview

### What Are Semantic Axes?

A **semantic axis** is a unit vector in SBERT embedding space that represents a dimension of semantic variation. It points from one concept (the **negative pole**, e.g., "professionalism") to another (the **positive pole**, e.g., "casualness").

By projecting LLM outputs onto these axes, you can:
- **Quantify** how two models differ along interpretable dimensions
- **Compare** models on dimensions like formality, technicality, sentiment, politeness, etc.
- **Explain** distributional differences detected by quantum metrics or other tests

### When to Use Semantic Axes

**Use semantic axes when:**
- You want to understand *what* differs between models, not just *if* they differ
- You need interpretable, stakeholder-friendly explanations
- You want to profile a single model along semantic dimensions
- You're comparing fine-tuned models and want to understand the effect of fine-tuning

**Don't use semantic axes when:**
- You only need to detect *if* two models differ (use quantum metrics or MMD tests directly)
- The dimensions of interest aren't easily captured by pole corpora (e.g., factual accuracy)
- You don't have or can't generate appropriate pole corpora

### Integration with Quantum Metrics

Semantic axes are designed as an **interpretation layer** that complements quantum-inspired detection:

1. **Quantum metrics** (trace distance, von Neumann entropy) detect *if* models differ
2. **Semantic axes** explain *what* differs

Both use the same SBERT embeddings, so there's no additional embedding cost when using them together.

---

## Core Concepts

### Pole Corpora

A **pole corpus** is a collection of texts that exemplify one pole of a semantic dimension. Each corpus is stored as a JSONL file with records like:

```json
{"text": "Pursuant to our discussion, I hereby submit...", "metadata": {"concept": "professionalism"}}
{"text": "With regard to the aforementioned matter...", "metadata": {"concept": "professionalism"}}
```

**Key properties:**
- Each corpus should contain 50+ diverse, on-topic examples
- Texts should strongly exemplify the concept without being extreme outliers
- The two pole corpora should represent a clear conceptual contrast

**How to generate pole corpora:** Use the [`semantic-pole`](https://github.com/yourusername/semantic-pole) CLI tool, which uses LLMs to generate diverse examples via iterative refinement and convergence testing.

### Axis Vector

The axis vector is computed by:
1. **Embedding** all texts in both pole corpora using SBERT (default: `all-mpnet-base-v2`)
2. **L2-normalizing** each embedding (ensures cosine-similarity semantics)
3. **Computing centroids**: `centroid_neg = mean(normalized_embeddings_neg)`, `centroid_pos = mean(normalized_embeddings_pos)`
4. **Computing the axis**: `axis = (centroid_pos - centroid_neg) / ||centroid_pos - centroid_neg||`

The result is a unit vector pointing from the negative pole toward the positive pole.

### Projection Scores

To project an LLM output onto an axis:
1. Embed the text using the same SBERT model
2. L2-normalize the embedding
3. Compute the dot product with the axis vector: `score = normalized_embedding · axis_vector`

**Interpretation:**
- **Positive scores** → text is more toward the positive pole
- **Negative scores** → text is more toward the negative pole
- **Magnitude** → strength of the association

---

## Quick Start

### Installation

Ensure you have the development version of the package:
```bash
cd model_equality_testing
pip install -e .
```

Dependencies (already included): `numpy`, `scipy`, `sentence-transformers`

### Minimal Example

```python
import numpy as np
from model_equality_testing.dataset import load_distribution
from model_equality_testing.src.embeddings import embed_sample
from model_equality_testing.src.semantic_axes import load_axis_from_jsonl, interpret_difference

# 1. Load samples
dist_a = load_distribution(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    prompt_ids={"wikipedia_en": [0, 1, 2]},
    L=500, source="fp32", load_in_unicode=True,
)
dist_b = load_distribution(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    prompt_ids={"wikipedia_en": [0, 1, 2]},
    L=500, source="int8", load_in_unicode=True,
)
sample_a = dist_a.draw_completion_sample(n=100)
sample_b = dist_b.draw_completion_sample(n=100)

# 2. Embed samples
emb_a = embed_sample(sample_a)
emb_b = embed_sample(sample_b)

# 3. Load semantic axis
prof_axis = load_axis_from_jsonl(
    "/path/to/professionalism.jsonl",
    "/path/to/casualness.jsonl",
    "professionalism-casualness"
)
print(prof_axis)  # Shows compact summary with separation_ratio

# 4. Interpret difference
interp = interpret_difference(emb_a, emb_b, [prof_axis],
                               label_a="fp32", label_b="int8")
print(interp.summary())
```

**Output:**
```
SemanticAxis(name='professionalism-casualness', negative='professionalism', 
             positive='casualness', dim=768, pole_distance=0.834, separation_ratio=2.14)

Semantic Interpretation (1 axes, sorted by effect size):

  professionalism ← → casualness:  Δ = +0.54  g = 0.82  p < 0.001 *
    fp32 is more toward casualness than int8

  * significant after Benjamini-Hochberg correction (α = 0.05)
```

---

## Axis Construction

### Loading Axes from JSONL

```python
from model_equality_testing.src.semantic_axes import load_axis_from_jsonl

axis = load_axis_from_jsonl(
    negative_pole_path="/path/to/professionalism.jsonl",
    positive_pole_path="/path/to/casualness.jsonl",
    axis_name="professionalism-casualness",
    embedding_model="all-mpnet-base-v2",  # default
    batch_size=32  # embedding batch size
)
```

**What happens during loading:**
1. Reads JSONL files and extracts `"text"` fields
2. Filters out empty texts (errors if all texts are empty)
3. Extracts pole names from `metadata.concept` in first record (falls back to filename if missing)
4. Embeds all texts using `EmbeddingModel`
5. L2-normalizes embeddings
6. Computes centroids, pole distance, intra-pole RMS std, separation ratio
7. Constructs unit axis vector
8. Validates separation (errors if pole_distance < 1e-6)
9. Warns if either corpus has < 50 samples or separation_ratio < 1.0

**Metadata stored in axis:**
```python
axis.metadata = {
    "negative_pole_path": str,
    "positive_pole_path": str,
    "negative_pole_samples": int,
    "positive_pole_samples": int,
    "negative_pole_intra_std": float,  # RMS of deviations from centroid
    "positive_pole_intra_std": float,
    "separation_ratio": float,         # pole_distance / mean(intra_std_neg, intra_std_pos)
    "created_at": str,                 # ISO 8601 timestamp
}
```

### Saving and Loading Axes

**Save a single axis:**
```python
from model_equality_testing.src.semantic_axes import save_axis, load_axis

save_axis(prof_axis, "/path/to/professionalism-casualness.npz")
loaded_axis = load_axis("/path/to/professionalism-casualness.npz")
```

**Save multiple axes to a directory:**
```python
from model_equality_testing.src.semantic_axes import save_axes, load_axes

axes = [prof_axis, form_axis, tech_axis]
save_axes(axes, "/path/to/axes_directory")  # Creates prof-casual.npz, formality-informality.npz, etc.

# Load all axes from directory (sorted alphabetically by filename)
loaded_axes = load_axes("/path/to/axes_directory")
```

**Notes:**
- Axis names are sanitized for filenames (non-alphanumeric chars → `_`)
- `save_axes()` errors if axis names are not unique
- Loading is deterministic (alphabetical order by filename)
- Format: versioned `.npz` with arrays + JSON metadata (forward-compatible)

---

## Quality Metrics

### Separation Ratio

**Formula:** `separation_ratio = pole_distance / mean(intra_std_neg, intra_std_pos)`

Where:
- `pole_distance = ||centroid_pos - centroid_neg||` (after L2 normalization)
- `intra_std = sqrt(mean(||embedding_i - centroid||²))` (RMS of deviations)

**Interpretation:**
- **> 2.0**: Excellent separation, poles are well-distinguished
- **1.0 - 2.0**: Good separation, axis should work well
- **< 1.0**: Poor separation, poles overlap substantially (**warning issued**)
- **< 0.5**: Very poor separation, axis may not be meaningful

**Analogy:** Similar to Fisher's discriminant ratio — higher values mean the between-pole distance is large relative to within-pole variance.

**When separation is low:**
- The pole corpora may not represent distinct concepts
- The concepts may not be distinguishable in SBERT space
- Consider regenerating pole corpora with stronger exemplars

### Intra-Pole Standard Deviation

**Formula:** `intra_std = sqrt(mean(||embedding_i - centroid||²))`

This is the RMS (root-mean-square) of L2 distances from each embedding to its pole's centroid, computed *after* L2 normalization.

**Interpretation:**
- **Low intra_std** (< 0.2): Pole corpus is tightly clustered (homogeneous concept)
- **Moderate intra_std** (0.2 - 0.4): Pole corpus has reasonable diversity
- **High intra_std** (> 0.4): Pole corpus is very spread out (may be too diverse or off-topic)

**Note:** Since embeddings are L2-normalized to unit length, they lie on the unit sphere. The centroid (mean of points on the sphere) lies *inside* the sphere. Intra-std measures dispersion on this sphere.

### Pole Distance

**Formula:** `pole_distance = ||centroid_pos - centroid_neg||`

The Euclidean distance between the two pole centroids (after L2 normalization, before normalizing the axis vector).

**Range:** [0, 2] (maximum is 2 when centroids are antipodal on the unit sphere)

**Interpretation:**
- **> 0.8**: Strong separation
- **0.3 - 0.8**: Moderate separation
- **< 0.3**: Weak separation (**warning issued if < 1e-6**)

**Use:** Primarily a component of separation_ratio; by itself, it doesn't account for within-pole variance.

### Axis Correlations

Check if axes overlap using `axis_correlations()`:

```python
from model_equality_testing.src.semantic_axes import axis_correlations

corr_matrix = axis_correlations([prof_axis, form_axis, tech_axis])
# Returns (3, 3) cosine similarity matrix
# Warns if any pair has |correlation| > 0.7
```

**Interpretation:**
- **|correlation| < 0.3**: Axes are nearly orthogonal (good)
- **0.3 - 0.7**: Axes have some overlap (acceptable)
- **> 0.7**: Axes are highly correlated (**warning issued**)

**When axes correlate highly:**
- They measure similar dimensions
- Interpretation may overcount differences (e.g., professionalism and formality both contribute to the same underlying shift)
- Consider dropping one or using PCA to decorrelate

---

## Statistical Methodology

### Welch's t-Test

**Why Welch's t-test?** Does not assume equal variances between the two samples. This is appropriate because:
- Different models may have different variances in their projection distributions
- Quantization, fine-tuning, etc. can change variance as well as mean

**Implementation:** `scipy.stats.ttest_ind(scores_a, scores_b, equal_var=False)`

**Null hypothesis:** The two samples have the same mean projection score.

**p-value:** Probability of observing a delta this large (or larger) if the null is true.

### Hedges' g Effect Size

**Formula:**
```
s_avg = sqrt((s_a² + s_b²) / 2)      # average standard deviation
d = (mean_a - mean_b) / s_avg        # Cohen's d
J = 1 - 3 / (4 * (n_a + n_b) - 9)   # bias correction
hedges_g = d * J
```

**Why Hedges' g?** 
- **Bias-corrected** version of Cohen's d (important for small samples)
- **Average-SD denominator** does not assume equal variances (consistent with Welch's t-test)

**Guards:**
- If `s_avg < 1e-10`: `d = 0.0` (avoid division by zero)
- If `n_a + n_b < 4`: `J = 1.0` (bias correction unreliable for tiny samples)

**Interpretation (conventional thresholds):**
- **|g| < 0.2**: Negligible effect
- **0.2 ≤ |g| < 0.5**: Small effect
- **0.5 ≤ |g| < 0.8**: Medium effect
- **|g| ≥ 0.8**: Large effect

**Note:** These are rough guidelines. Practical significance depends on your use case.

### Benjamini-Hochberg FDR Correction

When testing multiple axes simultaneously, we correct for multiple comparisons using the **Benjamini-Hochberg** procedure to control the **False Discovery Rate (FDR)**.

**Why FDR correction?**
- Without correction, testing 10 axes at α=0.05 gives ~40% chance of at least one false positive
- BH controls the *expected proportion* of false discoveries among all rejections

**Algorithm:**
1. Sort p-values: `p₁ ≤ p₂ ≤ ... ≤ pₖ`
2. Compute `corrected_i = p_i × k / i`
3. Enforce monotonicity: `corrected_i = min(corrected_i, corrected_{i+1})`
4. Cap at 1.0

**Implementation:**
```python
def _benjamini_hochberg(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    if n <= 1:
        return pvalues.copy()  # No correction needed
    sorted_indices = np.argsort(pvalues)
    sorted_pvalues = pvalues[sorted_indices]
    corrected = sorted_pvalues * n / np.arange(1, n + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]  # monotonicity
    corrected = np.minimum(corrected, 1.0)
    result = np.empty(n)
    result[sorted_indices] = corrected
    return result
```

**Interpretation:**
- A corrected p-value < 0.05 means: "This axis is significant even after accounting for testing multiple axes."

### Wasserstein Distance

In addition to mean differences, `AxisProjectionResult` includes the **1D Wasserstein distance** (also called Earth Mover's Distance) between the two projection distributions.

**Why Wasserstein?**
- Captures full distributional shape, not just mean and variance
- Sensitive to differences in skewness, multimodality, outliers
- Complements effect size (Hedges' g)

**When Wasserstein adds value:**
- When distributions are non-Gaussian
- When you care about tail behavior or outliers
- When one model is more variable than the other

**Units:** Same as projection scores (dot products in cosine-similarity space after L2 normalization). No standardized thresholds; interpret relative to pole_distance or other axes.

---

## Interpretation Workflow

### Single Comparison

```python
from model_equality_testing.src.semantic_axes import interpret_difference

# Assume emb_a, emb_b are already computed (see Quick Start)
interp = interpret_difference(
    emb_a, emb_b,
    axes=[prof_axis, form_axis, tech_axis],
    label_a="fp32",
    label_b="int8"
)

# Print summary
print(interp.summary(top_k=5))

# Programmatic access
for result in interp.sorted_by_effect():
    print(f"{result.axis_name}: Δ={result.delta:.2f}, g={result.hedges_g:.2f}, p={result.corrected_pvalue:.4f}")
```

### Wrapper for CompletionSamples

```python
from model_equality_testing.src.semantic_axes import interpret_samples

# Convenience wrapper that embeds CompletionSamples automatically
interp = interpret_samples(
    sample_a, sample_b,
    axes=[prof_axis, form_axis, tech_axis],
    label_a="fp32",
    label_b="int8",
    embedding_model="all-mpnet-base-v2"  # optional, defaults to axis model
)
print(interp.summary())
```

### Reusing Embeddings (Efficient Pattern)

```python
from model_equality_testing.src.embeddings import embed_sample
from model_equality_testing.src.quantum_metrics import pca_density_matrix, fit_pca, trace_distance
from model_equality_testing.src.semantic_axes import interpret_difference

# 1. Embed once
emb_a = embed_sample(sample_a)
emb_b = embed_sample(sample_b)

# 2. Use for quantum metrics
pca = fit_pca(np.vstack([emb_a, emb_b]), k=50)
rho_a = pca_density_matrix(emb_a, pca)
rho_b = pca_density_matrix(emb_b, pca)
td = trace_distance(rho_a, rho_b)
print(f"Trace distance: {td:.4f}")

# 3. Use for semantic axes (no re-embedding!)
interp = interpret_difference(emb_a, emb_b, axes, label_a="fp32", label_b="int8")
print(interp.summary())
```

### Understanding the Summary Output

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

**Reading the output:**
- **Axes** are sorted by |Hedges' g| (largest effects first)
- **Δ** (delta) = mean_a - mean_b (positive → sample A is more toward positive pole)
- **g** (Hedges' g) = bias-corrected effect size
- **p** = FDR-corrected p-value
- **\*** = significant after multiple-comparisons correction (α = 0.05)
- **Direction statement**: Interprets the sign of Δ in plain language using the provided labels

**Common questions:**
- *"Large g but high p-value?"* → Large effect, but uncertain (small sample). Increase sample size.
- *"Small g but low p-value?"* → Small but reliable effect (large sample). Statistically significant but may not be practically important.
- *"Positive Δ but text says 'toward negative pole'?"* → This shouldn't happen; check that you passed the correct labels and that negative/positive poles are correctly assigned.

---

## Best Practices

### Axis Design

**1. Choose clear conceptual contrasts**
   - ✅ professionalism ↔ casualness
   - ✅ formality ↔ informality
   - ✅ technical ↔ layperson
   - ❌ professionalism ↔ technical (not opposites)
   - ❌ positive ↔ not-positive (vague)

**2. Generate diverse, on-topic pole corpora**
   - Aim for 50-100 examples per pole
   - Vary sentence structure, length, vocabulary
   - Stay on-concept; don't drift into related but distinct concepts
   - Use the `semantic-pole` tool's convergence testing to validate stability

**3. Validate axes before use**
   - Check `separation_ratio` (should be > 1.0, ideally > 2.0)
   - Check `axis_correlations()` (axes shouldn't correlate > 0.7)
   - Spot-check: Project a few known examples and verify scores make sense

**4. Match the embedding model**
   - Use the same SBERT model for axis construction and analysis
   - Default (`all-mpnet-base-v2`) is a good balance of quality and speed
   - If using a different model, regenerate axes with that model

### Sample Size Guidelines

- **Minimum:** 30-50 completions per group for reliable t-tests
- **Recommended:** 100+ per group for stable effect size estimates
- **Large samples:** 500+ per group for detecting small effects

**Note:** These are per-group sample sizes (n_a and n_b), not total.

### Multiple Comparisons

- Always use FDR correction when testing multiple axes
- Interpret corrected p-values, not raw p-values
- If you have a priori hypotheses (e.g., "we only care about professionalism"), test fewer axes to reduce correction penalty

### Reporting Results

**For stakeholders:**
- Lead with the summary output (it's stakeholder-friendly)
- Explain axes in domain terms (e.g., "professionalism measures formal business language")
- Report effect sizes (Hedges' g) alongside p-values

**For technical audiences:**
- Report separation_ratio for each axis
- Include sample sizes (n_a, n_b)
- Report both raw and FDR-corrected p-values
- Provide axis metadata (pole corpus sizes, intra_std)

**Example table:**

| Axis | Δ | Hedges' g | p (raw) | p (FDR) | Separation Ratio |
|------|---|-----------|---------|---------|------------------|
| professionalism ↔ casualness | +0.54 | 0.82 | <0.001 | <0.001 | 2.14 |
| formality ↔ informality | +0.38 | 0.61 | 0.001 | 0.003 | 1.87 |
| technical ↔ layperson | -0.21 | 0.31 | 0.082 | 0.15 | 1.52 |

---

## API Reference

### Core Functions

#### `load_axis_from_jsonl()`
```python
def load_axis_from_jsonl(
    negative_pole_path: str,
    positive_pole_path: str,
    axis_name: str,
    embedding_model: str = "all-mpnet-base-v2",
    batch_size: int = 32
) -> SemanticAxis
```

**Load a semantic axis from pole corpus JSONL files.**

**Args:**
- `negative_pole_path`: Path to negative pole JSONL (e.g., professionalism.jsonl)
- `positive_pole_path`: Path to positive pole JSONL (e.g., casualness.jsonl)
- `axis_name`: Name for this axis (e.g., "professionalism-casualness")
- `embedding_model`: SBERT model name (default: "all-mpnet-base-v2")
- `batch_size`: Batch size for embedding

**Returns:** `SemanticAxis` object

**Raises:**
- `FileNotFoundError`: If either JSONL file doesn't exist
- `ValueError`: If poles don't separate (pole_distance < 1e-6) or all texts are empty

**Warnings logged:**
- If either corpus has < 50 samples
- If separation_ratio < 1.0

---

#### `save_axis() / load_axis()`
```python
def save_axis(axis: SemanticAxis, path: str)
def load_axis(path: str) -> SemanticAxis
```

**Save/load a single axis to/from an .npz file.**

---

#### `save_axes() / load_axes()`
```python
def save_axes(axes: List[SemanticAxis], directory: str)
def load_axes(directory: str) -> List[SemanticAxis]
```

**Save/load multiple axes to/from a directory (one .npz per axis).**

**Notes:**
- `save_axes()` creates the directory if it doesn't exist
- `save_axes()` errors if axis names are not unique
- `load_axes()` returns axes sorted alphabetically by filename

---

#### `project_onto_axis()`
```python
def project_onto_axis(
    embeddings: np.ndarray,
    axis: SemanticAxis
) -> np.ndarray
```

**Project embeddings onto a semantic axis.**

**Args:**
- `embeddings`: Embeddings array (N, d)
- `axis`: SemanticAxis

**Returns:** Projection scores (N,). Positive = toward positive_pole.

**Raises:** `ValueError` if embedding dimension doesn't match axis dimension

**Notes:** L2-normalizes embeddings before projection for consistency with axis construction.

---

#### `project_onto_axes()`
```python
def project_onto_axes(
    embeddings: np.ndarray,
    axes: List[SemanticAxis]
) -> np.ndarray
```

**Project embeddings onto multiple axes (single matrix multiply).**

**Returns:** (N, K) array where K = len(axes)

---

#### `interpret_difference()`
```python
def interpret_difference(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    axes: List[SemanticAxis],
    label_a: str = "Sample A",
    label_b: str = "Sample B"
) -> SemanticInterpretation
```

**Interpret semantic differences between two samples.**

**Args:**
- `embeddings_a`: Embeddings for sample A (N_a, d)
- `embeddings_b`: Embeddings for sample B (N_b, d)
- `axes`: List of semantic axes
- `label_a`: Label for sample A (e.g., "fp32", "Model A")
- `label_b`: Label for sample B (e.g., "int8", "Model B")

**Returns:** `SemanticInterpretation` with per-axis results

**Statistics computed per axis:**
- Means, standard deviations, SEM
- Delta = mean_a - mean_b
- Hedges' g (bias-corrected effect size)
- Welch's t-test p-value
- Wasserstein distance
- FDR-corrected p-values (across all axes)

---

#### `interpret_samples()`
```python
def interpret_samples(
    sample_a: CompletionSample,
    sample_b: CompletionSample,
    axes: List[SemanticAxis],
    label_a: str = "Sample A",
    label_b: str = "Sample B",
    embedding_model: str = None,
    _precomputed_embeddings: tuple = None
) -> SemanticInterpretation
```

**Convenience wrapper for CompletionSample objects.**

Embeds samples and calls `interpret_difference()`. Supports `_precomputed_embeddings` pattern for efficiency.

---

#### `axis_correlations()`
```python
def axis_correlations(
    axes: List[SemanticAxis]
) -> np.ndarray
```

**Compute cosine similarity matrix between axis vectors.**

**Returns:** (K, K) correlation matrix

**Warns:** If any pair correlates above 0.7 (indicates overlapping dimensions)

---

### Data Classes

#### `SemanticAxis`
```python
@dataclass
class SemanticAxis:
    name: str
    negative_pole: str
    positive_pole: str
    axis_vector: np.ndarray
    negative_centroid: np.ndarray
    positive_centroid: np.ndarray
    embedding_model: str
    embedding_dim: int
    pole_distance: float
    metadata: dict
```

**Custom `__repr__`:** Compact summary without array dumps

**Validation in `__post_init__`:**
- `axis_vector` is unit norm
- Dimensions match
- `pole_distance > 1e-6`

---

#### `AxisProjectionResult`
```python
@dataclass
class AxisProjectionResult:
    axis_name: str
    negative_pole: str
    positive_pole: str
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    sem_a: float
    sem_b: float
    delta: float
    hedges_g: float
    t_pvalue: float
    corrected_pvalue: float
    wasserstein: float
    n_a: int
    n_b: int
```

---

#### `SemanticInterpretation`
```python
@dataclass
class SemanticInterpretation:
    results: List[AxisProjectionResult]
    embedding_model: str
    label_a: str
    label_b: str

    def sorted_by_effect(self) -> List[AxisProjectionResult]
    def summary(self, top_k: int = 5) -> str
```

---

## Troubleshooting

### "Poles don't separate" Error

**Error:** `ValueError: Poles don't separate (distance=1.2e-07 < 1e-6)`

**Causes:**
- The two pole corpora represent the same concept
- The pole corpora texts are nearly identical
- The concepts are not distinguishable in SBERT space

**Solutions:**
1. Regenerate pole corpora with stronger, more distinctive examples
2. Check that you loaded the correct JSONL files (not two copies of the same file)
3. Try a different embedding model
4. The concepts may genuinely not be separable — choose a different axis

---

### Low Separation Ratio Warning

**Warning:** `Separation ratio is 0.73 < 1.0. Poles overlap substantially; axis may have poor discriminating power.`

**Interpretation:** The pole centroids are close relative to within-pole variance. The axis can still be used, but results may be noisy.

**Solutions:**
1. Increase pole corpus size (more samples → tighter centroids)
2. Filter pole corpora to remove off-topic examples
3. Regenerate pole corpora with more focused prompts
4. Accept the low quality if the axis is exploratory

---

### High Axis Correlation Warning

**Warning:** `High correlation (0.84) between axes 'professionalism-casualness' and 'formality-informality'. These dimensions may overlap; interpretation may overcount differences.`

**Interpretation:** The two axes measure similar (but not identical) dimensions.

**Solutions:**
1. Drop one axis (keep the one with higher separation_ratio)
2. Acknowledge overlap in reporting ("These axes are correlated, so the effects are not independent")
3. Use PCA to decorrelate axes (advanced)

---

### Model Name Mismatch Warning

**Warning:** `Embedding model 'all-MiniLM-L6-v2' != axis model 'all-mpnet-base-v2'. Results may be unreliable.`

**Cause:** You're using a different embedding model than the axes were built with.

**Solution:** Regenerate axes using the same model, or re-embed samples with the axis model.

---

### Empty Embeddings

**Warning:** `Empty text at index 42, will use zero vector for embedding`

**Cause:** A completion in your sample is empty (after filtering padding).

**Impact:** That completion will have zero projection score on all axes.

**Solution:** Usually harmless if rare. If many completions are empty, check your data pipeline.

---

### Tiny Sample Sizes

**Behavior:** When `n_a + n_b < 4`, Hedges' g bias correction is set to J=1.0 (no correction).

**Recommendation:** Use sample sizes ≥ 30 per group for reliable statistics.

---

### No Significant Differences Found

**Scenario:** All corrected p-values > 0.05.

**Possible causes:**
1. The two models genuinely don't differ on these dimensions
2. Sample size is too small to detect the effect
3. The axes don't capture the relevant dimensions of variation

**Solutions:**
1. Increase sample size
2. Test different axes
3. Use exploratory analysis to identify which dimensions *do* differ (e.g., quantum metrics)

---

## Further Reading

- **Paper:** ["Model Equality Testing: Which model is this API serving?"](https://arxiv.org/abs/2410.20247)
- **Semantic-pole project:** [github.com/yourusername/semantic-pole](https://github.com/yourusername/semantic-pole) (for generating pole corpora)
- **SBERT documentation:** [www.sbert.net](https://www.sbert.net)
- **Welch's t-test:** [Wikipedia](https://en.wikipedia.org/wiki/Welch%27s_t-test)
- **Hedges' g:** [Wikipedia](https://en.wikipedia.org/wiki/Effect_size#Hedges'_g)
- **Benjamini-Hochberg:** [Wikipedia](https://en.wikipedia.org/wiki/False_discovery_rate#Benjamini%E2%80%93Hochberg_procedure)

---

## Questions or Issues?

- **GitHub Issues:** [github.com/yourusername/model-equality-testing/issues](https://github.com/yourusername/model-equality-testing/issues)
- **Paper Authors:** See contact information in [the paper](https://arxiv.org/abs/2410.20247)
