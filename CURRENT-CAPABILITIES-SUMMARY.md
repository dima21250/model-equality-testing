# Current Capabilities Summary: What You Already Have

## Overview

After tonight's exploration of integration approaches, the conclusion is: **v1 (semantic axes) + density matrices are sufficient**. You don't need a covariance framework or forced integration. Here's what you already have and where it's documented.

## Three Complementary Tools

### 1. Token-Level Testing (Original Paper)
**Location**: `model_equality_testing/src/tests.py`, `algorithm.py`

**What it does**: Statistical tests on token distributions
- MMD with Hamming, k-spectrum, all-subsequences kernels
- Chi-squared tests (standard, truncated)
- L1/L2 distances
- Two-sample Kolmogorov-Smirnov test

**Use when**: Detecting if distributions differ at token level

**Documentation**: `README.md` (main example), paper

### 2. Quantum-Inspired Metrics (Semantic Level)
**Location**: `model_equality_testing/src/quantum_metrics.py`

**What it does**: Distributional metrics on SBERT embeddings
- Density matrices: ρ = G/trace(G) where G is Gram matrix
- Trace distance: D(ρ_A, ρ_B) - distinguishability
- Von Neumann entropy: S(ρ) - diversity measure
- Quantum relative entropy: S(ρ_A || ρ_B) - divergence
- PCA compression (768-D → 50-D) for efficiency

**Use when**: 
- Comparing semantic-level distributions
- Equal sample sizes (n_a = n_b) for valid trace distance
- Want quantum-inspired interpretation

**Known limitations** (from docstring):
> "Cross-matrix metrics (trace distance, QRE) are not well-defined in the quantum-mechanical sense, and trace distance exhibits pathological sample-size dependence."

**Documentation**: 
- `QUANTUM-METRICS-ADVANTAGES.md` - honest assessment of advantages/limitations
- `QUANTUM-VS-SEMANTIC-INTERPRETATION.md` - what quantum metrics tell you vs don't tell you
- Inline docstrings in `quantum_metrics.py`

**Key functions**:
```python
from model_equality_testing.src.quantum_metrics import (
    fit_pca,
    pca_density_matrix,
    trace_distance,
    von_neumann_entropy,
    quantum_relative_entropy
)

# Fit PCA on combined embeddings
pca = fit_pca(np.vstack([emb_a, emb_b]), k=50)

# Compute density matrices
rho_a = pca_density_matrix(emb_a, pca)
rho_b = pca_density_matrix(emb_b, pca)

# Metrics
td = trace_distance(rho_a, rho_b)  # Distinguishability
s_a = von_neumann_entropy(rho_a)   # Diversity
s_b = von_neumann_entropy(rho_b)
kl = quantum_relative_entropy(rho_a, rho_b)  # Divergence
```

### 3. Semantic Axes Interpretation (V1)
**Location**: `model_equality_testing/src/semantic_axes.py`

**What it does**: Interpretable semantic analysis along named dimensions
- Load axes from pole corpora (JSONL files)
- Project embeddings onto axes
- Statistical comparison: means, effect sizes, p-values
- Multiple testing correction (Benjamini-Hochberg FDR)
- Quality metrics: separation ratio, pole statistics

**Use when**: 
- Want to understand *what* differs (not just *that* they differ)
- Need interpretable dimensions (professionalism, technicality, etc.)
- Want statistical rigor (effect sizes, p-values)

**Key features**:
- ✓ Works for any sample sizes (n_a, n_b can differ)
- ✓ Statistical tests: Welch's t-test (no equal variance assumption)
- ✓ Effect sizes: Hedges' g with bias correction
- ✓ Multiple testing: Benjamini-Hochberg FDR correction
- ✓ L2 normalization throughout (consistent cosine-similarity semantics)

**Documentation**:
- `README.md` section "Semantic Axes Interpretation" - quick start
- `SEMANTIC-AXES-GUIDE.md` - comprehensive guide (600+ lines)
  - Core concepts, quick start, axis construction
  - Quality metrics, statistical methodology
  - Interpretation workflow, best practices
  - API reference, troubleshooting
- Inline docstrings in `semantic_axes.py`

**Key functions**:
```python
from model_equality_testing.src.semantic_axes import (
    load_axis_from_jsonl,
    interpret_difference,
    save_axes,
    load_axes
)

# Load semantic axis from pole corpora
axis = load_axis_from_jsonl(
    "professionalism.jsonl",  # Negative pole corpus
    "casualness.jsonl",       # Positive pole corpus
    "prof-casual"             # Axis name
)

# Interpret difference between two distributions
interp = interpret_difference(
    emb_a, emb_b, [axis1, axis2, axis3],
    label_a="fp32", label_b="int8"
)

# Results include:
# - mean_a, mean_b, delta (shift in mean projection)
# - std_a, std_b (spread)
# - cohens_d / hedges_g (effect size)
# - t_pvalue, corrected_pvalue (statistical significance)
# - wasserstein (1D Wasserstein distance)

print(interp.summary())  # Human-readable output
```

**What v1 provides**:

| Feature | Capability |
|---------|-----------|
| Mean projection | μ @ v - location along semantic axis |
| Variance | Sample variance of projections |
| Standard deviation | Spread along axis |
| Effect size | Hedges' g with bias correction |
| Statistical test | Welch's t-test (unequal variances) |
| P-value | With Benjamini-Hochberg FDR correction |
| Distance | 1D Wasserstein distance |
| Different sample sizes | ✓ Works for n_a ≠ n_b |
| Quality metrics | Separation ratio, pole statistics |

## Embeddings (Shared Infrastructure)

**Location**: `model_equality_testing/src/embeddings.py`

**What it does**: SBERT embedding generation
- Uses `sentence-transformers` library
- Default model: `all-mpnet-base-v2` (768-D)
- Batched processing for efficiency
- L2 normalization for consistent semantics

**Key function**:
```python
from model_equality_testing.src.embeddings import embed_sample

# Embed a CompletionSample
embeddings = embed_sample(
    sample,
    model_name="all-mpnet-base-v2",
    batch_size=32
)
# Returns: (n, 768) array
```

**Shared usage**: Both quantum metrics and semantic axes use these embeddings, so you compute them once and reuse.

## Typical Workflow

### Workflow 1: Detect + Interpret
```python
# 1. Load data
dist_a = load_distribution(..., source="fp32")
dist_b = load_distribution(..., source="int8")
sample_a = dist_a.draw_completion_sample(100)
sample_b = dist_b.draw_completion_sample(100)

# 2. Embed once (reuse for both analyses)
emb_a = embed_sample(sample_a)
emb_b = embed_sample(sample_b)

# 3. Detect with quantum metrics
pca = fit_pca(np.vstack([emb_a, emb_b]), k=50)
rho_a = pca_density_matrix(emb_a, pca)
rho_b = pca_density_matrix(emb_b, pca)
td = trace_distance(rho_a, rho_b)
print(f"Trace distance: {td:.4f}")  # e.g., 0.23

# 4. Interpret with semantic axes
axes = [prof_axis, tech_axis, form_axis]
interp = interpret_difference(emb_a, emb_b, axes)
print(interp.summary())
# Shows: professionalism +0.54 (g=0.82, p<0.001), etc.
```

### Workflow 2: Semantic-Only Analysis
```python
# If you just want semantic interpretation (no quantum metrics)
emb_a = embed_sample(sample_a)
emb_b = embed_sample(sample_b)

axes = [load_axis_from_jsonl(...) for ... in axis_pairs]
interp = interpret_difference(emb_a, emb_b, axes)
print(interp.summary())

# This already gives you:
# - Location shifts (mean projection differences)
# - Spread changes (variance differences)
# - Effect sizes (Hedges' g)
# - Statistical significance (p-values with FDR correction)
```

## What's NOT Needed

After tonight's exploration, we determined you **don't need**:

### ✗ Covariance Framework
**Reason**: 
- v1 already computes mean and variance along semantic axes
- Principal components don't correspond across distributions (can't compare PC1 in A to PC1 in B)
- Pooled covariance for Mahalanobis has issues when covariances differ
- Sample size requirements (n ≥ 5d) are impractical for typical use

**If you want correlation structure**: Add one simple function to v1:
```python
def semantic_axis_correlation(embeddings, axis1, axis2):
    """Correlation between two semantic axes."""
    proj1 = embeddings @ axis1.axis_vector
    proj2 = embeddings @ axis2.axis_vector
    return np.corrcoef(proj1, proj2)[0, 1]
```

### ✗ Forced Integration
**Reason**:
- Density matrices (n×n) and semantic axes (d-dimensional) live in different spaces
- All integration attempts had mathematical issues:
  - Subspace decomposition: separate normalizations break valid decomposition
  - Diagonal observables: sample-dependent, ignore similarity structure
  - Stratification: different bin sizes → can't compute trace distance
- The tools are complementary, not unifiable

### ✗ Spectral-Semantic Analysis
**Reason**:
- Eigenmodes of Gram matrices are sample-dependent
- Can't compare "Mode 1 in A" to "Mode 1 in B" (different eigenvectors)
- If you want spectral analysis, use PCA on covariance (but this has its own issues)
- For semantic questions, v1 projections are more direct and interpretable

## What You Might Want to Add (Optional)

### 1. Correlation Function (30 minutes)
Add to `semantic_axes.py`:
```python
def semantic_axis_correlations(
    embeddings: np.ndarray,
    axes: List[SemanticAxis]
) -> pd.DataFrame:
    """
    Correlation matrix between semantic axes in a distribution.
    
    Use case: "Are professionalism and technicality correlated?"
    """
    k = len(axes)
    corr = np.zeros((k, k))
    for i, axis_i in enumerate(axes):
        for j, axis_j in enumerate(axes):
            proj_i = embeddings @ axis_i.axis_vector
            proj_j = embeddings @ axis_j.axis_vector
            corr[i, j] = np.corrcoef(proj_i, proj_j)[0, 1]
    
    return pd.DataFrame(corr,
                       index=[a.name for a in axes],
                       columns=[a.name for a in axes])
```

**Value**: Understand if semantic dimensions covary (e.g., "high professionalism → low technicality?")

**Effort**: ~30 minutes to implement and test

### 2. Better Visualization Helpers
Add summary visualization functions to `semantic_axes.py`:
- Scatter plot of two semantic dimensions
- Distribution comparison plots (before/after)
- Effect size visualizations

**Value**: Easier exploration and presentation

**Effort**: ~2 hours

## File Organization

**Implementation**:
```
model_equality_testing/src/
├── embeddings.py          # SBERT embedding generation
├── quantum_metrics.py     # Density matrices, trace distance, entropy
├── semantic_axes.py       # Semantic interpretation (v1)
├── tests.py              # Token-level test statistics
├── algorithm.py          # Two-sample and goodness-of-fit test framework
└── pvalue.py             # P-value calculation methods
```

**Documentation**:
```
README.md                           # Main documentation with examples
SEMANTIC-AXES-GUIDE.md             # Comprehensive semantic axes guide
QUANTUM-METRICS-ADVANTAGES.md      # Honest assessment of quantum metrics
QUANTUM-VS-SEMANTIC-INTERPRETATION.md  # What each layer provides
CLAUDE.md                          # Developer guide (for Claude Code)
```

**Generated Tonight** (critiques and proposals):
```
SEMANTIC-SUBSPACE-CRITIQUE.md      # Why subspace decomposition doesn't work
SEMANTIC-OBSERVABLES-CRITIQUE.md   # Why observables approach doesn't work
SEMANTIC-STRATIFICATION-CRITIQUE.md # Why stratification doesn't work
SPECTRAL-SEMANTIC-CRITIQUE.md      # Why Gram eigendecomposition doesn't work
COVARIANCE-IMPLEMENTATION-CRITIQUE.md  # Issues with covariance framework
FINAL-IMPLEMENTATION-V3-CRITIQUE.md    # Final critique: don't build it

HONEST-ASSESSMENT.md               # Summary of real options
COVARIANCE-FRAMEWORK.md            # What covariance matrices are
```

## Key Insights From Tonight

1. **v1 already does what you need** for semantic analysis (mean, variance, effect sizes, tests)

2. **Density matrices have known limitations** (sample-size dependent, not quantum in strict sense) but are useful for equal-sample comparisons

3. **Integration isn't achievable** because n×n (sample space) and d-dimensional (embedding space) can't be unified without forcing invalid mathematics

4. **All integration attempts failed** for fundamental mathematical reasons, not implementation issues

5. **The aesthetic goal (unified, parsimonious)** isn't achievable, but the **pragmatic tools work well**

## Recommendations

### Use What You Have
- ✓ Quantum metrics for detection and quantification (when n_a = n_b)
- ✓ Semantic axes (v1) for interpretation
- ✓ Token-level tests for baseline comparisons

### Optional Additions
- Consider: correlation function (30 min) if you need covariance structure
- Consider: visualization helpers (2 hours) for easier exploration

### Don't Build
- ✗ Covariance framework (mathematical issues, sample size impractical)
- ✗ Forced integration (not possible without invalid math)
- ✗ Spectral-semantic (correspondence issues)

### Move Forward
- Use existing tools for actual LLM comparison research
- Stop seeking impossible unification
- Focus on empirical questions, not framework development

## Summary

**You have everything you need**:
- Token-level testing (original paper contribution)
- Quantum metrics (semantic-level detection)
- Semantic axes (interpretable analysis)
- Comprehensive documentation

**Total implementation**: Already complete, well-tested, documented

**Time to add correlation**: 30 minutes if desired

**Time spent tonight**: Exploring integration approaches that don't work

**Recommendation**: Use what you have. Start analyzing data.
