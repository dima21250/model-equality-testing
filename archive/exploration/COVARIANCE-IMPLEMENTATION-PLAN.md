# Covariance Matrix Implementation Plan

## Goal

Implement unified covariance-based analysis with natural semantic integration, achieving the parsimony and theoretical coherence that density matrices couldn't provide.

## Architecture

**Core principle**: Covariance matrix C is the central representation. Everything flows through C.

**Three layers**:
1. **State representation**: Covariance matrices from embeddings
2. **Global metrics**: Distance, entropy, divergence between states
3. **Semantic projections**: Variance along interpretable axes

All in one unified framework.

## New Module: `covariance_metrics.py`

Create `model_equality_testing/src/covariance_metrics.py` alongside `quantum_metrics.py`.

### Core Functions

```python
def covariance_matrix(
    embeddings: np.ndarray,
    center: bool = True,
    regularization: float = 1e-6
) -> np.ndarray:
    """
    Compute covariance matrix from embeddings.
    
    Args:
        embeddings: (n, d) array of n samples in d dimensions
        center: If True, center data before computing covariance
        regularization: Add ε·I for numerical stability
    
    Returns:
        C: (d, d) covariance matrix
        
    Note: Always d×d regardless of sample count n.
    """
    n, d = embeddings.shape
    
    if center:
        embeddings = embeddings - embeddings.mean(axis=0, keepdims=True)
    
    # C = (1/n) X^T X
    C = (embeddings.T @ embeddings) / n
    
    # Add regularization for numerical stability
    C = C + regularization * np.eye(d)
    
    return C


def log_euclidean_distance(C_a: np.ndarray, C_b: np.ndarray) -> float:
    """
    Log-Euclidean distance between covariance matrices.
    
    This is the theoretically grounded metric on the manifold of
    symmetric positive definite matrices.
    
    Args:
        C_a: (d, d) covariance matrix for distribution A
        C_b: (d, d) covariance matrix for distribution B
    
    Returns:
        Distance in [0, ∞)
        
    References:
        Arsigny et al., "Geometric Means in a Novel Vector Space 
        Structure on Symmetric Positive-Definite Matrices" (2007)
    """
    import scipy.linalg
    
    log_C_a = scipy.linalg.logm(C_a)
    log_C_b = scipy.linalg.logm(C_b)
    
    # Frobenius norm of difference of log matrices
    diff = log_C_a - log_C_b
    distance = np.linalg.norm(diff, 'fro')
    
    return distance


def frobenius_distance(C_a: np.ndarray, C_b: np.ndarray) -> float:
    """
    Frobenius norm distance (simple Euclidean metric).
    
    Less theoretically grounded than log-Euclidean, but fast and simple.
    """
    return np.linalg.norm(C_a - C_b, 'fro')


def differential_entropy(C: np.ndarray) -> float:
    """
    Differential entropy of Gaussian N(0, C).
    
    H = (1/2) log|C| + (d/2)(1 + log(2π))
    
    For comparison purposes, we return just (1/2) log|C| (the data-dependent part).
    
    Args:
        C: (d, d) covariance matrix
    
    Returns:
        Entropy (nat units)
    """
    sign, logdet = np.linalg.slogdet(C)
    
    if sign <= 0:
        raise ValueError("Covariance matrix must be positive definite")
    
    return 0.5 * logdet


def gaussian_kl_divergence(
    mu_a: np.ndarray, C_a: np.ndarray,
    mu_b: np.ndarray, C_b: np.ndarray
) -> float:
    """
    KL divergence between two Gaussian distributions.
    
    KL(N(μ_A, C_A) || N(μ_B, C_B))
    
    Args:
        mu_a: (d,) mean of distribution A
        C_a: (d, d) covariance of distribution A
        mu_b: (d,) mean of distribution B
        C_b: (d, d) covariance of distribution B
    
    Returns:
        KL divergence in [0, ∞)
    """
    d = len(mu_a)
    
    # Invert C_b
    C_b_inv = np.linalg.inv(C_b)
    
    # Determinants
    _, logdet_a = np.linalg.slogdet(C_a)
    _, logdet_b = np.linalg.slogdet(C_b)
    
    # Mean difference
    delta_mu = mu_b - mu_a
    
    # KL formula
    kl = 0.5 * (
        np.trace(C_b_inv @ C_a) +
        delta_mu.T @ C_b_inv @ delta_mu -
        d +
        logdet_b - logdet_a
    )
    
    return kl


def variance_along_axis(C: np.ndarray, axis: np.ndarray) -> float:
    """
    Variance along a semantic axis.
    
    σ² = v^T C v
    
    Args:
        C: (d, d) covariance matrix
        axis: (d,) unit vector defining semantic axis
    
    Returns:
        Variance along axis
    """
    return axis.T @ C @ axis


def covariance_between_axes(
    C: np.ndarray,
    axis_1: np.ndarray,
    axis_2: np.ndarray
) -> float:
    """
    Covariance between two semantic axes.
    
    Cov(v₁, v₂) = v₁^T C v₂
    
    Args:
        C: (d, d) covariance matrix
        axis_1: (d,) first semantic axis
        axis_2: (d,) second semantic axis
    
    Returns:
        Covariance between axes
    """
    return axis_1.T @ C @ axis_2
```

## Integration with Semantic Axes

Extend `semantic_axes.py` with covariance-based analysis:

```python
@dataclass
class CovarianceSemanticMeasurement:
    """
    Semantic measurement via covariance quadratic form.
    
    Natural integration: v^T C v is variance along axis v.
    """
    axis_name: str
    variance_a: float          # v^T C_A v
    variance_b: float          # v^T C_B v
    delta_variance: float      # variance_b - variance_a
    std_a: float               # sqrt(variance_a)
    std_b: float               # sqrt(variance_b)
    negative_pole: str
    positive_pole: str
    
    def interpretation(self) -> str:
        """Human-readable interpretation."""
        if self.delta_variance > 0:
            direction = "increased"
        else:
            direction = "decreased"
        
        return (
            f"Variance along {self.axis_name}: "
            f"A = {self.variance_a:.3f}, B = {self.variance_b:.3f} "
            f"({direction} by {abs(self.delta_variance):.3f})"
        )


@dataclass
class CovarianceSemanticComparison:
    """
    Unified covariance-based comparison with semantic integration.
    
    Single central representation (covariance matrix) with:
    - Global metrics (distance, entropy)
    - Semantic measurements (variance along axes)
    """
    # Global metrics
    log_euclidean_distance: float
    frobenius_distance: float
    entropy_a: float
    entropy_b: float
    kl_divergence_a_b: float  # KL(A || B)
    kl_divergence_b_a: float  # KL(B || A)
    
    # Covariance matrices (for reference)
    C_a: np.ndarray
    C_b: np.ndarray
    
    # Semantic measurements
    semantic_measurements: List[CovarianceSemanticMeasurement]
    
    # Eigenvalue information (spectral structure)
    eigenvalues_a: np.ndarray
    eigenvalues_b: np.ndarray
    
    def summary(self, top_k_axes: int = 5) -> str:
        """Unified summary: global + semantic."""
        pass


def compare_with_covariance_semantics(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    axes: List[SemanticAxis],
    pca_dim: Optional[int] = 50
) -> CovarianceSemanticComparison:
    """
    Unified covariance-based comparison with semantic integration.
    
    This is the main entry point. Everything flows through covariance matrices.
    
    Workflow:
        1. Optionally apply PCA (compress embeddings to pca_dim)
        2. Compute covariance matrices C_A and C_B (d×d, always same size)
        3. Compute global metrics (log-Euclidean distance, entropy, KL)
        4. For each semantic axis: compute v^T C v (variance along axis)
        5. Return unified CovarianceSemanticComparison
    
    Args:
        embeddings_a: (n_a, d) embeddings for distribution A
        embeddings_b: (n_b, d) embeddings for distribution B
        axes: Semantic axes for interpretation
        pca_dim: If provided, compress to this dimension first
    
    Returns:
        CovarianceSemanticComparison with all metrics
    """
    from .covariance_metrics import (
        covariance_matrix, log_euclidean_distance, frobenius_distance,
        differential_entropy, gaussian_kl_divergence, variance_along_axis
    )
    
    # PCA compression (optional)
    if pca_dim is not None:
        from .quantum_metrics import fit_pca
        pca = fit_pca(np.vstack([embeddings_a, embeddings_b]), k=pca_dim)
        embeddings_a = pca.transform(embeddings_a)
        embeddings_b = pca.transform(embeddings_b)
        
        # Project axes into PCA space
        axes = [project_axis_to_pca(axis, pca) for axis in axes]
    
    # Compute means (for KL divergence)
    mu_a = embeddings_a.mean(axis=0)
    mu_b = embeddings_b.mean(axis=0)
    
    # Compute covariance matrices (d×d, same size!)
    C_a = covariance_matrix(embeddings_a)
    C_b = covariance_matrix(embeddings_b)
    
    # Global metrics
    led = log_euclidean_distance(C_a, C_b)
    fd = frobenius_distance(C_a, C_b)
    ent_a = differential_entropy(C_a)
    ent_b = differential_entropy(C_b)
    kl_ab = gaussian_kl_divergence(mu_a, C_a, mu_b, C_b)
    kl_ba = gaussian_kl_divergence(mu_b, C_b, mu_a, C_a)
    
    # Eigenvalues (spectral structure)
    evals_a = np.linalg.eigvalsh(C_a)[::-1]  # descending
    evals_b = np.linalg.eigvalsh(C_b)[::-1]
    
    # Semantic measurements
    measurements = []
    for axis in axes:
        v = axis.axis_vector
        
        var_a = variance_along_axis(C_a, v)
        var_b = variance_along_axis(C_b, v)
        delta_var = var_b - var_a
        
        meas = CovarianceSemanticMeasurement(
            axis_name=axis.name,
            variance_a=var_a,
            variance_b=var_b,
            delta_variance=delta_var,
            std_a=np.sqrt(var_a),
            std_b=np.sqrt(var_b),
            negative_pole=axis.negative_pole,
            positive_pole=axis.positive_pole
        )
        measurements.append(meas)
    
    return CovarianceSemanticComparison(
        log_euclidean_distance=led,
        frobenius_distance=fd,
        entropy_a=ent_a,
        entropy_b=ent_b,
        kl_divergence_a_b=kl_ab,
        kl_divergence_b_a=kl_ba,
        C_a=C_a,
        C_b=C_b,
        semantic_measurements=measurements,
        eigenvalues_a=evals_a,
        eigenvalues_b=evals_b
    )
```

## Summary Output

```
Covariance-Based Unified Analysis
==================================

Global Metrics:
  Log-Euclidean distance:  0.234
  Frobenius distance:      0.412
  Differential entropy:    A = 12.3, B = 13.1 (Δ = +0.8)
  KL divergence:           KL(A||B) = 0.156, KL(B||A) = 0.143

Spectral Structure:
  Top eigenvalues A:  [2.1, 1.8, 1.3, 0.9, 0.7, ...]
  Top eigenvalues B:  [2.5, 1.9, 1.2, 0.8, 0.6, ...]
  → B has higher variance along principal mode

Semantic Variance Analysis:

  professionalism ← → casualness
    Variance: A = 0.42, B = 0.68 (Δ = +0.26)
    Std dev:  A = 0.65, B = 0.82
    → B is more variable along professionalism axis

  technical ← → layperson
    Variance: A = 0.31, B = 0.25 (Δ = -0.06)
    Std dev:  A = 0.56, B = 0.50
    → B is less variable along technicality axis

  formal ← → informal
    Variance: A = 0.38, B = 0.41 (Δ = +0.03)
    Std dev:  A = 0.62, B = 0.64
    → Similar variability on formality

Interpretation:
  Distributions differ significantly (d_LE = 0.234). Distribution B exhibits
  higher overall entropy (+0.8) and substantially increased variance along the
  professionalism axis (+0.26), indicating more diverse semantic character in
  that dimension.
```

## Advantages of This Design

**1. Unified representation**: C is the central object

**2. Natural semantic integration**: 
- v^T C v is the standard way to query covariance along direction v
- Not forced or artificial
- Same v for both C_A and C_B (truly measuring the same thing)

**3. No sample-count dependency**:
- C is always d×d (50×50 after PCA)
- Works for any sample sizes n_a, n_b

**4. Theoretically grounded**:
- Log-Euclidean distance: established metric on SPD manifold
- Differential entropy: standard information-theoretic measure
- Variance along axis: textbook quadratic form

**5. Rich interpretation**:
- Global: distance, entropy, divergence
- Spectral: eigenvalues show variance structure
- Semantic: variance along interpretable axes
- All from one representation

**6. Clean API**:
```python
result = compare_with_covariance_semantics(emb_a, emb_b, axes)
print(result.summary())
```

One function call, complete analysis.

## Implementation Phases

### Phase 1: Core Covariance Metrics (2-3 hours)
1. Create `covariance_metrics.py`
2. Implement: covariance_matrix, log_euclidean_distance, frobenius_distance
3. Implement: differential_entropy, gaussian_kl_divergence
4. Implement: variance_along_axis, covariance_between_axes
5. Unit tests for each function

### Phase 2: Semantic Integration (2-3 hours)
1. Add `CovarianceSemanticMeasurement` dataclass to `semantic_axes.py`
2. Add `CovarianceSemanticComparison` dataclass
3. Implement `compare_with_covariance_semantics()`
4. Implement `CovarianceSemanticComparison.summary()`
5. Integration tests with real data

### Phase 3: Documentation (1-2 hours)
1. Update `README.md` with covariance approach
2. Update `CLAUDE.md` with new module and functions
3. Add examples to docstrings
4. Create usage notebook/script

### Phase 4: Validation (1-2 hours)
1. Compare to existing v1 semantic projections (should align)
2. Verify metrics are sensible on known test cases
3. Check numerical stability (regularization, eigenvalues)
4. Document when covariance approach is appropriate

**Total**: ~8 hours for complete implementation

## Relationship to Existing Code

**No changes needed to**:
- `quantum_metrics.py` (keep for existing density matrix analysis)
- `embeddings.py` (just use `embed_sample()` as before)
- `dataset.py`, `distribution.py` (infrastructure)

**Extensions to**:
- `semantic_axes.py` (add covariance-based functions)

**New module**:
- `covariance_metrics.py` (parallel to `quantum_metrics.py`)

**User choice**:
- Use density matrices for sample-level analysis (when n_a = n_b)
- Use covariance matrices for variance-based analysis with semantic integration
- Both are available, serve different purposes

## When to Use Covariance vs Density Matrices

**Use covariance matrices when**:
- ✓ You want semantic integration (variance along axes)
- ✓ Sample sizes differ (n_a ≠ n_b)
- ✓ You care about variance structure more than sample-level detail
- ✓ You want spectral analysis in embedding space

**Use density matrices when**:
- ✓ You want sample-level granularity (n×n captures all pairwise similarities)
- ✓ Sample sizes are equal (n_a = n_b)
- ✓ You want the quantum aesthetic/interpretation
- ✓ You're doing permutation tests (need sample-level swapping)

**Use both when**:
- ✓ You want comprehensive analysis from multiple perspectives

## Summary

This plan achieves:
- ✓ **Unified aesthetic**: C is central, everything flows through it
- ✓ **Natural semantic integration**: v^T C v (standard quadratic form)
- ✓ **Theoretically coherent**: Established manifold metrics and statistical theory
- ✓ **No Frankenstein**: Clean, principled design
- ✓ **Solves sample-size problem**: Always d×d regardless of n

Implementation is straightforward (existing patterns from `quantum_metrics.py`, just in different space).

Ready to implement?
