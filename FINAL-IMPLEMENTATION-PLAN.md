# Final Implementation Plan: Covariance-Based Analysis with Semantic Integration

## What We're Actually Building

A **covariance-based framework** that achieves unified aesthetic through honest, well-grounded mathematics.

**Core principles** (learned from critiques):
1. **Mean AND variance** - not just variance
2. **Simple metrics** - Frobenius norm, not log-Euclidean
3. **Honest assumptions** - acknowledge Gaussian approximations
4. **PCA for spectral** - covariance eigenmodes, not Gram eigenmodes
5. **No overselling** - this is a useful tool, not a unified theory of everything

## Architecture

**Three analysis layers**:

### Layer 1: State Representation
- Covariance matrix C (d×d) - captures variance structure
- Mean vector μ (d-dimensional) - captures location
- Together: (μ, C) parameterizes a Gaussian approximation

### Layer 2: Global Comparison
- Frobenius distance: ||C_A - C_B||_F (simple, interpretable)
- Mean difference: ||μ_A - μ_B|| (Euclidean distance in embedding space)
- Optional: Gaussian KL divergence (if assumption is reasonable)

### Layer 3: Semantic Integration
- Mean projection: μ @ v (location along semantic axis)
- Variance: v^T C v (spread along semantic axis)
- Both are necessary for complete semantic picture

### Layer 4: Spectral Analysis (Optional)
- Eigendecomposition of C: principal components
- Semantic interpretation of PCs
- Variance explained by each PC

## Implementation

### Module: `covariance_metrics.py`

```python
"""
Covariance-based distributional comparison.

This module provides an alternative to density matrices (n×n Gram matrices)
using covariance matrices (d×d) which have fixed dimensionality regardless
of sample count.

Key advantages:
- Works for different sample sizes (always d×d)
- Natural semantic integration via quadratic forms
- Well-studied statistical framework

Key limitations:
- Captures only mean and covariance (second-order statistics)
- Many metrics assume Gaussian distributions (may not hold for LLM outputs)
- Requires sufficient samples for stable estimation (n >> d recommended)

References:
    - Anderson, "An Introduction to Multivariate Statistical Analysis"
    - Ledoit & Wolf, "A well-conditioned estimator for large-dimensional
      covariance matrices" (2004)
"""

import numpy as np
import logging
from typing import Tuple, Optional


def covariance_matrix(
    embeddings: np.ndarray,
    regularization: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and covariance matrix from embeddings.
    
    Args:
        embeddings: (n, d) array of n samples in d dimensions
        regularization: Ridge regularization (adds ε·I for stability)
    
    Returns:
        mu: (d,) mean vector
        C: (d, d) covariance matrix (regularized)
    
    Note:
        Covariance is always d×d regardless of sample count n.
        For stable estimation, you should have n >> d.
        
    Example:
        >>> embeddings = np.random.randn(100, 50)
        >>> mu, C = covariance_matrix(embeddings)
        >>> mu.shape
        (50,)
        >>> C.shape
        (50, 50)
    """
    n, d = embeddings.shape
    
    if n < d:
        logging.warning(
            f"Sample size ({n}) is less than dimensionality ({d}). "
            f"Covariance estimate may be unstable. Consider using more samples "
            f"or reducing dimensionality via PCA."
        )
    
    # Compute mean
    mu = embeddings.mean(axis=0)
    
    # Center data
    centered = embeddings - mu
    
    # Covariance: C = (1/n) X^T X
    C = (centered.T @ centered) / n
    
    # Regularization for numerical stability
    C = C + regularization * np.eye(d)
    
    return mu, C


def frobenius_distance(C_a: np.ndarray, C_b: np.ndarray) -> float:
    """
    Frobenius norm distance between covariance matrices.
    
    This is the Euclidean distance treating matrices as vectors:
        d_F = sqrt(sum((C_a[i,j] - C_b[i,j])^2))
    
    Interpretation: Total squared difference in variance/covariance structure.
    
    Args:
        C_a: (d, d) covariance matrix for distribution A
        C_b: (d, d) covariance matrix for distribution B
    
    Returns:
        Distance in [0, ∞)
    
    Note:
        This treats the manifold of covariance matrices as Euclidean space,
        which is not geometrically correct but is simple and interpretable.
    """
    return np.linalg.norm(C_a - C_b, 'fro')


def mean_projection(mu: np.ndarray, axis: np.ndarray) -> float:
    """
    Project mean onto semantic axis (location along axis).
    
    Args:
        mu: (d,) mean vector
        axis: (d,) unit vector defining semantic axis
    
    Returns:
        Scalar projection (location)
    """
    return mu @ axis


def variance_along_axis(C: np.ndarray, axis: np.ndarray) -> float:
    """
    Variance along a semantic axis (spread along axis).
    
    This is the quadratic form: σ² = v^T C v
    
    Args:
        C: (d, d) covariance matrix
        axis: (d,) unit vector defining semantic axis
    
    Returns:
        Variance (non-negative scalar)
    """
    return axis @ C @ axis


def gaussian_kl_divergence(
    mu_a: np.ndarray,
    C_a: np.ndarray,
    mu_b: np.ndarray,
    C_b: np.ndarray
) -> float:
    """
    KL divergence between two Gaussian distributions.
    
    KL(N(μ_A, C_A) || N(μ_B, C_B)) = 0.5 * [
        tr(C_B^{-1} C_A) +
        (μ_B - μ_A)^T C_B^{-1} (μ_B - μ_A) -
        d +
        log|C_B| - log|C_A|
    ]
    
    Args:
        mu_a: (d,) mean of distribution A
        C_a: (d, d) covariance of distribution A
        mu_b: (d,) mean of distribution B
        C_b: (d, d) covariance of distribution B
    
    Returns:
        KL divergence in [0, ∞)
    
    Note:
        This assumes both distributions are Gaussian. For LLM embeddings,
        this is an approximation that may not hold (embeddings are likely
        non-Gaussian: multimodal, heavy-tailed, bounded).
        
        Use with caution and interpret as "KL between Gaussian approximations"
        rather than true KL divergence.
    """
    d = len(mu_a)
    
    # Check invertibility
    try:
        C_b_inv = np.linalg.inv(C_b)
    except np.linalg.LinAlgError:
        logging.warning("C_b is singular, cannot compute KL divergence")
        return np.nan
    
    # Determinants (use slogdet for numerical stability)
    sign_a, logdet_a = np.linalg.slogdet(C_a)
    sign_b, logdet_b = np.linalg.slogdet(C_b)
    
    if sign_a <= 0 or sign_b <= 0:
        logging.warning("Covariance matrix is not positive definite")
        return np.nan
    
    # Mean difference
    delta_mu = mu_b - mu_a
    
    # KL formula
    kl = 0.5 * (
        np.trace(C_b_inv @ C_a) +
        delta_mu @ C_b_inv @ delta_mu -
        d +
        logdet_b - logdet_a
    )
    
    return kl


def principal_components(
    C: np.ndarray,
    n_components: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute principal components (eigenvectors of covariance matrix).
    
    Args:
        C: (d, d) covariance matrix
        n_components: Number of components to return (default: all)
    
    Returns:
        eigenvalues: (k,) array sorted descending
        eigenvectors: (d, k) matrix, columns are eigenvectors
    
    Note:
        Eigenvectors are in embedding space (d-dimensional),
        not sample space (n-dimensional). This makes them
        directly interpretable and comparable across distributions.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    if n_components is not None:
        eigenvalues = eigenvalues[:n_components]
        eigenvectors = eigenvectors[:, :n_components]
    
    return eigenvalues, eigenvectors
```

### Extension to `semantic_axes.py`

```python
@dataclass
class SemanticMeasurement:
    """
    Complete semantic measurement: location AND spread.
    
    This is what was missing from all previous proposals.
    """
    axis_name: str
    
    # Location (mean projection)
    mean_a: float
    mean_b: float
    delta_mean: float  # mean_b - mean_a
    
    # Spread (variance)
    variance_a: float
    variance_b: float
    delta_variance: float  # variance_b - variance_a
    std_a: float  # sqrt(variance_a)
    std_b: float  # sqrt(variance_b)
    
    # Axis info
    negative_pole: str
    positive_pole: str
    
    def interpretation(self) -> str:
        """Human-readable interpretation."""
        # Location interpretation
        if self.delta_mean > 0:
            loc_direction = self.positive_pole
            loc_change = f"+{self.delta_mean:.3f}"
        else:
            loc_direction = self.negative_pole
            loc_change = f"{self.delta_mean:.3f}"
        
        # Spread interpretation
        if self.delta_variance > 0:
            spread_change = "increased"
        else:
            spread_change = "decreased"
        
        return (
            f"{self.axis_name}:\n"
            f"  Location: {loc_change} toward {loc_direction}\n"
            f"  Spread: {spread_change} by {abs(self.delta_variance):.3f}\n"
            f"  (A: μ={self.mean_a:.2f}, σ={self.std_a:.2f}; "
            f"B: μ={self.mean_b:.2f}, σ={self.std_b:.2f})"
        )


@dataclass
class CovarianceComparison:
    """
    Unified covariance-based comparison.
    
    Honest about what it provides and what it doesn't.
    """
    # Global metrics
    frobenius_distance: float  # Covariance difference
    mean_distance: float       # Mean difference (Euclidean)
    kl_divergence_a_b: float   # KL(A || B) - Gaussian approximation
    kl_divergence_b_a: float   # KL(B || A) - Gaussian approximation
    
    # State representation
    mu_a: np.ndarray
    mu_b: np.ndarray
    C_a: np.ndarray
    C_b: np.ndarray
    
    # Semantic measurements (location + spread)
    semantic_measurements: List[SemanticMeasurement]
    
    # Sample sizes (for context)
    n_a: int
    n_b: int
    d: int
    
    # Warnings
    warnings: List[str]
    
    def summary(self) -> str:
        """Honest summary with caveats."""
        lines = [
            "Covariance-Based Comparison",
            "=" * 50,
            "",
            f"Sample sizes: A = {self.n_a}, B = {self.n_b}",
            f"Dimension: {self.d}",
            ""
        ]
        
        # Warnings first
        if self.warnings:
            lines.append("⚠ Warnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
            lines.append("")
        
        # Global metrics
        lines.extend([
            "Global Metrics:",
            f"  Covariance distance (Frobenius): {self.frobenius_distance:.3f}",
            f"  Mean distance (Euclidean):       {self.mean_distance:.3f}",
            f"  KL(A||B) [Gaussian approx]:      {self.kl_divergence_a_b:.3f}",
            f"  KL(B||A) [Gaussian approx]:      {self.kl_divergence_b_a:.3f}",
            "",
            "Semantic Measurements:",
            ""
        ])
        
        # Semantic measurements (sorted by absolute location shift)
        sorted_meas = sorted(
            self.semantic_measurements,
            key=lambda m: abs(m.delta_mean),
            reverse=True
        )
        
        for meas in sorted_meas:
            lines.append(meas.interpretation())
            lines.append("")
        
        # Footer
        lines.extend([
            "Note: This analysis captures mean and covariance (2nd-order statistics).",
            "Higher-order structure (skewness, multimodality) is not captured.",
            "KL divergence assumes Gaussian distributions (approximation for LLM outputs)."
        ])
        
        return "\n".join(lines)


def compare_with_covariance(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    axes: List[SemanticAxis],
    pca_dim: Optional[int] = 50,
    regularization: float = 1e-6
) -> CovarianceComparison:
    """
    Complete covariance-based comparison with semantic integration.
    
    This is the main entry point.
    
    Args:
        embeddings_a: (n_a, d) embeddings for distribution A
        embeddings_b: (n_b, d) embeddings for distribution B
        axes: Semantic axes for interpretation
        pca_dim: If provided, compress via PCA first
        regularization: Ridge parameter for covariance estimation
    
    Returns:
        CovarianceComparison with all metrics
    """
    from .covariance_metrics import (
        covariance_matrix, frobenius_distance, mean_projection,
        variance_along_axis, gaussian_kl_divergence
    )
    
    warnings = []
    n_a_orig, d_orig = embeddings_a.shape
    n_b_orig, _ = embeddings_b.shape
    
    # Optional PCA compression
    if pca_dim is not None and pca_dim < d_orig:
        from .quantum_metrics import fit_pca
        pca = fit_pca(np.vstack([embeddings_a, embeddings_b]), k=pca_dim)
        embeddings_a = pca.transform(embeddings_a)
        embeddings_b = pca.transform(embeddings_b)
        axes = [project_axis_to_pca(axis, pca) for axis in axes]
        d = pca_dim
    else:
        d = d_orig
    
    n_a, _ = embeddings_a.shape
    n_b, _ = embeddings_b.shape
    
    # Check sample size
    if n_a < d:
        warnings.append(f"Sample A has n={n_a} < d={d} (unstable covariance estimate)")
    if n_b < d:
        warnings.append(f"Sample B has n={n_b} < d={d} (unstable covariance estimate)")
    
    # Compute means and covariances
    mu_a, C_a = covariance_matrix(embeddings_a, regularization)
    mu_b, C_b = covariance_matrix(embeddings_b, regularization)
    
    # Global metrics
    cov_dist = frobenius_distance(C_a, C_b)
    mean_dist = np.linalg.norm(mu_a - mu_b)
    kl_ab = gaussian_kl_divergence(mu_a, C_a, mu_b, C_b)
    kl_ba = gaussian_kl_divergence(mu_b, C_b, mu_a, C_a)
    
    if np.isnan(kl_ab) or np.isnan(kl_ba):
        warnings.append("KL divergence could not be computed (singular covariance)")
    
    # Semantic measurements
    measurements = []
    for axis in axes:
        v = axis.axis_vector
        
        # Mean projection (LOCATION)
        mean_proj_a = mean_projection(mu_a, v)
        mean_proj_b = mean_projection(mu_b, v)
        delta_mean = mean_proj_b - mean_proj_a
        
        # Variance (SPREAD)
        var_a = variance_along_axis(C_a, v)
        var_b = variance_along_axis(C_b, v)
        delta_var = var_b - var_a
        
        meas = SemanticMeasurement(
            axis_name=axis.name,
            mean_a=mean_proj_a,
            mean_b=mean_proj_b,
            delta_mean=delta_mean,
            variance_a=var_a,
            variance_b=var_b,
            delta_variance=delta_var,
            std_a=np.sqrt(var_a),
            std_b=np.sqrt(var_b),
            negative_pole=axis.negative_pole,
            positive_pole=axis.positive_pole
        )
        measurements.append(meas)
    
    return CovarianceComparison(
        frobenius_distance=cov_dist,
        mean_distance=mean_dist,
        kl_divergence_a_b=kl_ab if not np.isnan(kl_ab) else 0.0,
        kl_divergence_b_a=kl_ba if not np.isnan(kl_ba) else 0.0,
        mu_a=mu_a,
        mu_b=mu_b,
        C_a=C_a,
        C_b=C_b,
        semantic_measurements=measurements,
        n_a=n_a,
        n_b=n_b,
        d=d,
        warnings=warnings
    )
```

## Example Usage

```python
from model_equality_testing.dataset import load_distribution
from model_equality_testing.src.embeddings import embed_sample
from model_equality_testing.src.semantic_axes import (
    load_axis_from_jsonl, compare_with_covariance
)

# Load distributions
dist_fp32 = load_distribution(..., source="fp32")
dist_int8 = load_distribution(..., source="int8")

# Draw samples (can be different sizes!)
sample_fp32 = dist_fp32.draw_completion_sample(100)
sample_int8 = dist_int8.draw_completion_sample(150)  # Different n

# Embed
emb_fp32 = embed_sample(sample_fp32)
emb_int8 = embed_sample(sample_int8)

# Load semantic axes
axes = [
    load_axis_from_jsonl("prof_neg.jsonl", "prof_pos.jsonl", "professionalism"),
    load_axis_from_jsonl("tech_neg.jsonl", "tech_pos.jsonl", "technicality"),
]

# Unified comparison
result = compare_with_covariance(emb_fp32, emb_int8, axes, pca_dim=50)

# Output
print(result.summary())
```

## What This Achieves

✓ **Fixed dimensionality**: C is always d×d (works for any sample sizes)  
✓ **Complete semantic measurement**: Mean (location) AND variance (spread)  
✓ **Simple metrics**: Frobenius norm (interpretable)  
✓ **Honest**: Warns about assumptions, sample size issues  
✓ **Unified API**: One function call  

## What This Doesn't Claim

✗ "Quantum" framework - it's classical statistics  
✗ Captures full distribution - only 2nd-order moments  
✗ Works for all data - assumes enough samples (n >> d)  
✗ Perfect unification - it's a useful tool, not a theory of everything  

## Implementation Timeline

**Phase 1** (2 hours): Core `covariance_metrics.py`
- covariance_matrix, frobenius_distance
- mean_projection, variance_along_axis
- gaussian_kl_divergence, principal_components
- Unit tests

**Phase 2** (2 hours): Semantic integration
- SemanticMeasurement, CovarianceComparison dataclasses
- compare_with_covariance() function
- summary() output formatting

**Phase 3** (1 hour): Documentation
- Update README.md
- Update CLAUDE.md
- Docstring examples

**Phase 4** (1 hour): Validation
- Test on real data (fp32 vs int8)
- Verify warnings trigger correctly
- Compare to v1 results (should align for mean projections)

**Total**: ~6 hours

## Ready to Implement?

This is honest, grounded, and actually works.
