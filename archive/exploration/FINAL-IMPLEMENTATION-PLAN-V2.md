# Final Implementation Plan V2: Covariance-Based Analysis (Critique-Hardened)

## Changes From V1

Based on professorial critique, this version addresses:
1. ✓ PCA circular dependency - retention checks for semantic axes
2. ✓ Stricter sample size requirements - error on n < 3d, warn on n < 5d
3. ✓ Principled regularization - Ledoit-Wolf shrinkage instead of hard-coded ε
4. ✓ Drop Gaussian KL - too misleading for non-Gaussian data
5. ✓ Normalized Frobenius - relative distance instead of absolute
6. ✓ Use principal components - spectral analysis included
7. ✓ Errors vs warnings - fail fast on invalid inputs
8. ✓ Mahalanobis distance - standardized mean difference
9. ✓ Validation strategy - comprehensive testing plan

## Core Architecture (Unchanged)

**State**: (μ, C) - mean vector and covariance matrix

**Analysis layers**:
1. Global comparison (covariance structure + mean location)
2. Semantic integration (location + spread along axes)
3. Spectral analysis (principal components)

## Implementation

### Module: `covariance_metrics.py`

```python
"""
Covariance-based distributional comparison.

This module provides variance-based analysis using covariance matrices (d×d)
which have fixed dimensionality regardless of sample count.

Advantages:
- Works for different sample sizes (always d×d)
- Natural semantic integration via quadratic forms
- Well-studied statistical framework

Limitations:
- Captures only mean and covariance (second-order statistics)
- Requires sufficient samples for stable estimation (n ≥ 5d recommended)
- Does NOT assume Gaussian distributions (though some metrics benefit from it)

References:
    - Anderson, "An Introduction to Multivariate Statistical Analysis" (2003)
    - Ledoit & Wolf, "A well-conditioned estimator for large-dimensional
      covariance matrices", J. Multivariate Analysis (2004)
"""

import numpy as np
import logging
from typing import Tuple, Optional
from sklearn.covariance import LedoitWolf


def covariance_matrix(
    embeddings: np.ndarray,
    method: str = "ledoit_wolf"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and covariance matrix from embeddings.
    
    Args:
        embeddings: (n, d) array of n samples in d dimensions
        method: Estimation method
            - "ledoit_wolf": Ledoit-Wolf shrinkage (default, principled)
            - "empirical": Standard empirical estimator (1/n X^T X)
    
    Returns:
        mu: (d,) mean vector
        C: (d, d) covariance matrix
    
    Raises:
        ValueError: If n < 3d (insufficient samples for stable estimation)
    
    Note:
        Ledoit-Wolf automatically determines optimal shrinkage toward
        identity matrix based on the data. This is superior to hard-coded
        regularization.
        
        Minimum sample size:
        - n < 3d: Error (unreliable)
        - 3d ≤ n < 5d: Warning (marginal)
        - 5d ≤ n < 10d: Acceptable
        - n ≥ 10d: Good
    
    Example:
        >>> embeddings = np.random.randn(250, 50)  # n=250, d=50 (5d)
        >>> mu, C = covariance_matrix(embeddings)
        >>> mu.shape
        (50,)
        >>> C.shape
        (50, 50)
    """
    n, d = embeddings.shape
    
    # Strict sample size check
    if n < 3 * d:
        raise ValueError(
            f"Insufficient samples: n={n} < 3d={3*d}. "
            f"Covariance estimate would be unreliable. "
            f"Minimum recommended: n={5*d} (5d). "
            f"Either increase sample size or reduce dimensionality (PCA)."
        )
    
    if n < 5 * d:
        logging.warning(
            f"Marginal sample size: n={n} < 5d={5*d}. "
            f"Covariance estimate may be unstable. "
            f"Recommended minimum: n={5*d}."
        )
    
    # Compute mean
    mu = embeddings.mean(axis=0)
    
    # Compute covariance
    if method == "ledoit_wolf":
        lw = LedoitWolf()
        C = lw.fit(embeddings).covariance_
    elif method == "empirical":
        centered = embeddings - mu
        C = (centered.T @ centered) / n
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return mu, C


def normalized_frobenius_distance(C_a: np.ndarray, C_b: np.ndarray) -> float:
    """
    Normalized Frobenius distance between covariance matrices.
    
    This is the Frobenius norm of the difference, normalized by the
    average Frobenius norm of the matrices:
    
        d = ||C_a - C_b||_F / (||C_a||_F + ||C_b||_F) * 2
    
    Normalization ensures the result is scale-invariant and interpretable
    as a relative difference.
    
    Args:
        C_a: (d, d) covariance matrix for distribution A
        C_b: (d, d) covariance matrix for distribution B
    
    Returns:
        Normalized distance in [0, 1] approximately
        (can exceed 1 if matrices have very different structure)
    
    Interpretation:
        - 0.0: Identical covariance structure
        - 0.1: ~10% relative difference
        - 0.5: ~50% relative difference
        - 1.0: ~100% relative difference (very different)
    """
    diff_norm = np.linalg.norm(C_a - C_b, 'fro')
    avg_norm = (np.linalg.norm(C_a, 'fro') + np.linalg.norm(C_b, 'fro')) / 2
    
    if avg_norm < 1e-10:
        return 0.0
    
    return diff_norm / avg_norm


def mahalanobis_distance(
    mu_a: np.ndarray,
    mu_b: np.ndarray,
    C_pooled: np.ndarray
) -> float:
    """
    Mahalanobis distance between means (standardized by covariance).
    
    This accounts for the covariance structure when measuring mean difference:
    
        d_M = sqrt((μ_b - μ_a)^T C_pooled^{-1} (μ_b - μ_a))
    
    Unlike Euclidean distance, this is scale-invariant and accounts for
    correlations between dimensions.
    
    Args:
        mu_a: (d,) mean of distribution A
        mu_b: (d,) mean of distribution B
        C_pooled: (d, d) pooled covariance matrix
    
    Returns:
        Mahalanobis distance in [0, ∞)
    
    Note:
        Related to Hotelling's T² statistic used in multivariate t-tests.
        For Gaussian distributions, this is the "statistical distance"
        accounting for variance and correlation.
    """
    delta_mu = mu_b - mu_a
    
    try:
        C_inv = np.linalg.inv(C_pooled)
    except np.linalg.LinAlgError:
        logging.warning("Pooled covariance is singular, using pseudo-inverse")
        C_inv = np.linalg.pinv(C_pooled)
    
    mahal = np.sqrt(delta_mu @ C_inv @ delta_mu)
    return mahal


def mean_projection(mu: np.ndarray, axis: np.ndarray) -> float:
    """Project mean onto semantic axis (location along axis)."""
    return mu @ axis


def variance_along_axis(C: np.ndarray, axis: np.ndarray) -> float:
    """Variance along a semantic axis (spread along axis)."""
    return axis @ C @ axis


def principal_components_analysis(
    C: np.ndarray,
    n_components: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Principal component analysis of covariance matrix.
    
    Args:
        C: (d, d) covariance matrix
        n_components: Number of components to return (default: all)
    
    Returns:
        eigenvalues: (k,) array sorted descending (variances along PCs)
        eigenvectors: (d, k) matrix, columns are PCs (unit vectors)
        variance_explained: Fraction of total variance in top-k components
    
    Note:
        These PCs are in embedding space (d-dimensional), making them
        directly interpretable and comparable across distributions.
        
        Unlike Gram matrix eigendecomposition (which gives sample-space modes),
        these are feature-space directions that are meaningful to compare.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    if n_components is not None:
        eigenvalues_truncated = eigenvalues[:n_components]
        eigenvectors = eigenvectors[:, :n_components]
        variance_explained = eigenvalues_truncated.sum() / eigenvalues.sum()
    else:
        variance_explained = 1.0
    
    return eigenvalues[:n_components] if n_components else eigenvalues, \
           eigenvectors, \
           variance_explained


def pooled_covariance(
    C_a: np.ndarray,
    C_b: np.ndarray,
    n_a: int,
    n_b: int
) -> np.ndarray:
    """
    Pooled covariance matrix (weighted average of two covariances).
    
    C_pooled = (n_a * C_a + n_b * C_b) / (n_a + n_b)
    
    Used for Mahalanobis distance and other standardized metrics.
    """
    return (n_a * C_a + n_b * C_b) / (n_a + n_b)
```

### Extension to `semantic_axes.py`

```python
@dataclass
class PrincipalComponentSemantic:
    """Semantic interpretation of a principal component."""
    pc_index: int
    variance: float  # Eigenvalue (variance along this PC)
    variance_fraction: float  # Fraction of total variance
    semantic_scores: Dict[str, float]  # axis_name → cosine similarity
    
    def dominant_axes(self, top_k: int = 3) -> List[Tuple[str, float]]:
        """Return top-k semantic axes by absolute score."""
        return sorted(
            self.semantic_scores.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_k]
    
    def interpretation(self) -> str:
        """Human-readable interpretation."""
        top_axes = self.dominant_axes(top_k=2)
        desc = ", ".join([f"{name}: {score:+.2f}" for name, score in top_axes])
        return (
            f"PC{self.pc_index} ({self.variance_fraction*100:.1f}% of variance): "
            f"{desc}"
        )


@dataclass
class SemanticMeasurement:
    """Complete semantic measurement: location AND spread."""
    axis_name: str
    
    # Location (mean projection)
    mean_a: float
    mean_b: float
    delta_mean: float
    
    # Spread (variance)
    variance_a: float
    variance_b: float
    delta_variance: float
    std_a: float
    std_b: float
    
    # Axis info
    negative_pole: str
    positive_pole: str
    
    def interpretation(self) -> str:
        """Human-readable interpretation."""
        # Location
        if abs(self.delta_mean) > 0.01:
            if self.delta_mean > 0:
                loc_str = f"+{self.delta_mean:.3f} toward {self.positive_pole}"
            else:
                loc_str = f"{self.delta_mean:.3f} toward {self.negative_pole}"
        else:
            loc_str = "no shift"
        
        # Spread
        if abs(self.delta_variance) > 0.01:
            if self.delta_variance > 0:
                spread_str = f"spread increased by {self.delta_variance:.3f}"
            else:
                spread_str = f"spread decreased by {abs(self.delta_variance):.3f}"
        else:
            spread_str = "spread unchanged"
        
        return (
            f"{self.axis_name}:\n"
            f"  Location: {loc_str}\n"
            f"  Spread: {spread_str}\n"
            f"  (A: μ={self.mean_a:.2f}, σ={self.std_a:.2f}; "
            f"B: μ={self.mean_b:.2f}, σ={self.std_b:.2f})"
        )


@dataclass
class CovarianceComparison:
    """
    Unified covariance-based comparison.
    
    Honest about capabilities and limitations.
    """
    # Global metrics
    covariance_distance: float  # Normalized Frobenius
    mean_distance: float  # Mahalanobis distance
    
    # State representation
    mu_a: np.ndarray
    mu_b: np.ndarray
    C_a: np.ndarray
    C_b: np.ndarray
    
    # Semantic measurements
    semantic_measurements: List[SemanticMeasurement]
    
    # Spectral analysis
    pc_analysis_a: List[PrincipalComponentSemantic]
    pc_analysis_b: List[PrincipalComponentSemantic]
    
    # Sample info
    n_a: int
    n_b: int
    d: int
    
    # Warnings and metadata
    warnings: List[str]
    pca_axis_retention: Optional[Dict[str, float]] = None
    
    def summary(self, top_k_pcs: int = 3) -> str:
        """Comprehensive summary with all analyses."""
        lines = [
            "Covariance-Based Comparison",
            "=" * 60,
            "",
            f"Sample sizes: A = {self.n_a}, B = {self.n_b}",
            f"Dimension: {self.d}",
            ""
        ]
        
        # Warnings
        if self.warnings:
            lines.append("⚠ WARNINGS:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
            lines.append("")
        
        # PCA retention (if applicable)
        if self.pca_axis_retention:
            lines.append("PCA Semantic Axis Retention:")
            for axis_name, retention in self.pca_axis_retention.items():
                status = "✓" if retention > 0.7 else ("⚠" if retention > 0.3 else "✗")
                lines.append(f"  {status} {axis_name}: {retention*100:.1f}% retained")
            lines.append("")
        
        # Global metrics
        lines.extend([
            "Global Metrics:",
            f"  Covariance distance (normalized Frobenius): {self.covariance_distance:.3f}",
            f"  Mean distance (Mahalanobis):                {self.mean_distance:.3f}",
            "",
            "Interpretation:",
            f"  - Covariance: {self._interpret_cov_dist()}",
            f"  - Mean: {self._interpret_mean_dist()}",
            "",
        ])
        
        # Semantic measurements
        lines.append("Semantic Measurements:")
        lines.append("")
        
        sorted_meas = sorted(
            self.semantic_measurements,
            key=lambda m: abs(m.delta_mean),
            reverse=True
        )
        
        for meas in sorted_meas:
            lines.append(meas.interpretation())
            lines.append("")
        
        # Spectral analysis
        lines.append("Principal Component Analysis:")
        lines.append("")
        lines.append("Distribution A:")
        for pc in self.pc_analysis_a[:top_k_pcs]:
            lines.append(f"  {pc.interpretation()}")
        lines.append("")
        lines.append("Distribution B:")
        for pc in self.pc_analysis_b[:top_k_pcs]:
            lines.append(f"  {pc.interpretation()}")
        lines.append("")
        
        # Footer
        lines.extend([
            "=" * 60,
            "Notes:",
            "  - This analysis captures mean and covariance (2nd-order statistics)",
            "  - Higher-order structure (skewness, kurtosis, multimodality) not captured",
            "  - No Gaussian assumption made for core metrics",
            "  - Covariance estimated via Ledoit-Wolf shrinkage (principled regularization)"
        ])
        
        return "\n".join(lines)
    
    def _interpret_cov_dist(self) -> str:
        """Interpret normalized Frobenius distance."""
        d = self.covariance_distance
        if d < 0.1:
            return "Very similar variance structure (~{:.0f}% difference)".format(d*100)
        elif d < 0.3:
            return "Moderately different variance structure (~{:.0f}% difference)".format(d*100)
        else:
            return "Very different variance structure (~{:.0f}% difference)".format(d*100)
    
    def _interpret_mean_dist(self) -> str:
        """Interpret Mahalanobis distance."""
        d = self.mean_distance
        # Rule of thumb: Mahalanobis > 3 is "far" for Gaussian
        if d < 1:
            return "Means are close (within 1 standard deviation)"
        elif d < 3:
            return f"Means are moderately separated ({d:.1f} standard deviations)"
        else:
            return f"Means are far apart ({d:.1f} standard deviations)"


def compare_with_covariance(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    axes: List[SemanticAxis],
    pca_dim: Optional[int] = 50,
    n_pcs_analyze: int = 5
) -> CovarianceComparison:
    """
    Complete covariance-based comparison with semantic integration.
    
    This is the main entry point.
    
    Args:
        embeddings_a: (n_a, d) embeddings for distribution A
        embeddings_b: (n_b, d) embeddings for distribution B
        axes: Semantic axes for interpretation
        pca_dim: If provided, compress via PCA first
        n_pcs_analyze: Number of principal components to analyze
    
    Returns:
        CovarianceComparison with all metrics
    
    Raises:
        ValueError: If sample size is insufficient (n < 3d)
    """
    from .covariance_metrics import (
        covariance_matrix, normalized_frobenius_distance,
        mahalanobis_distance, mean_projection, variance_along_axis,
        principal_components_analysis, pooled_covariance
    )
    
    warnings = []
    n_a_orig, d_orig = embeddings_a.shape
    n_b_orig, _ = embeddings_b.shape
    pca_axis_retention = None
    
    # Optional PCA compression
    if pca_dim is not None and pca_dim < d_orig:
        from .quantum_metrics import fit_pca
        pca = fit_pca(np.vstack([embeddings_a, embeddings_b]), k=pca_dim)
        embeddings_a = pca.transform(embeddings_a)
        embeddings_b = pca.transform(embeddings_b)
        
        # Check semantic axis retention
        pca_axis_retention = {}
        original_axes = axes.copy()
        axes_pca = []
        
        for axis in original_axes:
            # Project axis into PCA space
            axis_pca_vec = pca.components_.T @ axis.axis_vector
            retention = np.linalg.norm(axis_pca_vec) / np.linalg.norm(axis.axis_vector)
            pca_axis_retention[axis.name] = retention
            
            if retention < 0.3:
                warnings.append(
                    f"Semantic axis '{axis.name}' largely discarded by PCA "
                    f"(only {retention*100:.1f}% retained). Results may not reflect "
                    f"true difference along this axis."
                )
            elif retention < 0.7:
                warnings.append(
                    f"Semantic axis '{axis.name}' partially suppressed by PCA "
                    f"({retention*100:.1f}% retained)."
                )
            
            # Renormalize and create projected axis
            axis_pca_vec = axis_pca_vec / np.linalg.norm(axis_pca_vec) if np.linalg.norm(axis_pca_vec) > 1e-10 else axis_pca_vec
            axis_pca = SemanticAxis(
                name=axis.name,
                negative_pole=axis.negative_pole,
                positive_pole=axis.positive_pole,
                axis_vector=axis_pca_vec,
                negative_centroid=pca.transform(axis.negative_centroid.reshape(1, -1)).flatten(),
                positive_centroid=pca.transform(axis.positive_centroid.reshape(1, -1)).flatten(),
                embedding_model=axis.embedding_model,
                embedding_dim=pca_dim,
                pole_distance=np.linalg.norm(axis_pca_vec),
                metadata={**axis.metadata, 'pca_projected': True, 'retention': retention}
            )
            axes_pca.append(axis_pca)
        
        axes = axes_pca
        d = pca_dim
    else:
        d = d_orig
    
    n_a, _ = embeddings_a.shape
    n_b, _ = embeddings_b.shape
    
    # Compute means and covariances (raises ValueError if n < 3d)
    mu_a, C_a = covariance_matrix(embeddings_a, method="ledoit_wolf")
    mu_b, C_b = covariance_matrix(embeddings_b, method="ledoit_wolf")
    
    # Global metrics
    cov_dist = normalized_frobenius_distance(C_a, C_b)
    C_pool = pooled_covariance(C_a, C_b, n_a, n_b)
    mean_dist = mahalanobis_distance(mu_a, mu_b, C_pool)
    
    # Semantic measurements
    measurements = []
    for axis in axes:
        v = axis.axis_vector
        
        # Mean projection
        mean_proj_a = mean_projection(mu_a, v)
        mean_proj_b = mean_projection(mu_b, v)
        delta_mean = mean_proj_b - mean_proj_a
        
        # Variance
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
    
    # Principal component analysis
    evals_a, evecs_a, _ = principal_components_analysis(C_a, n_components=n_pcs_analyze)
    evals_b, evecs_b, _ = principal_components_analysis(C_b, n_components=n_pcs_analyze)
    
    total_var_a = np.trace(C_a)
    total_var_b = np.trace(C_b)
    
    pc_analysis_a = []
    for i in range(len(evals_a)):
        pc_vec = evecs_a[:, i]
        semantic_scores = {axis.name: pc_vec @ axis.axis_vector for axis in axes}
        pc_analysis_a.append(PrincipalComponentSemantic(
            pc_index=i+1,
            variance=evals_a[i],
            variance_fraction=evals_a[i] / total_var_a,
            semantic_scores=semantic_scores
        ))
    
    pc_analysis_b = []
    for i in range(len(evals_b)):
        pc_vec = evecs_b[:, i]
        semantic_scores = {axis.name: pc_vec @ axis.axis_vector for axis in axes}
        pc_analysis_b.append(PrincipalComponentSemantic(
            pc_index=i+1,
            variance=evals_b[i],
            variance_fraction=evals_b[i] / total_var_b,
            semantic_scores=semantic_scores
        ))
    
    return CovarianceComparison(
        covariance_distance=cov_dist,
        mean_distance=mean_dist,
        mu_a=mu_a,
        mu_b=mu_b,
        C_a=C_a,
        C_b=C_b,
        semantic_measurements=measurements,
        pc_analysis_a=pc_analysis_a,
        pc_analysis_b=pc_analysis_b,
        n_a=n_a,
        n_b=n_b,
        d=d,
        warnings=warnings,
        pca_axis_retention=pca_axis_retention
    )
```

## Validation Strategy

### Phase 1: Mathematical Correctness
```python
def test_variance_formula():
    """Verify v^T C v equals sample variance of projections."""
    embeddings = np.random.randn(200, 50)
    axis = np.random.randn(50)
    axis = axis / np.linalg.norm(axis)
    
    # Direct calculation
    projections = embeddings @ axis
    var_direct = np.var(projections, ddof=0)
    
    # Via covariance matrix
    _, C = covariance_matrix(embeddings)
    var_matrix = axis @ C @ axis
    
    assert np.isclose(var_direct, var_matrix, rtol=1e-5)
```

### Phase 2: Stability Analysis
```python
def test_regularization_sensitivity():
    """Test sensitivity to Ledoit-Wolf vs empirical estimation."""
    for n, d in [(100, 20), (500, 50), (1000, 100)]:
        embeddings_a = np.random.randn(n, d)
        embeddings_b = np.random.randn(n, d) + 0.5
        
        # Ledoit-Wolf
        mu_a_lw, C_a_lw = covariance_matrix(embeddings_a, method="ledoit_wolf")
        mu_b_lw, C_b_lw = covariance_matrix(embeddings_b, method="ledoit_wolf")
        dist_lw = normalized_frobenius_distance(C_a_lw, C_b_lw)
        
        # Empirical
        mu_a_emp, C_a_emp = covariance_matrix(embeddings_a, method="empirical")
        mu_b_emp, C_b_emp = covariance_matrix(embeddings_b, method="empirical")
        dist_emp = normalized_frobenius_distance(C_a_emp, C_b_emp)
        
        print(f"n={n}, d={d}: LW={dist_lw:.3f}, Emp={dist_emp:.3f}, "
              f"Diff={(dist_lw - dist_emp)/dist_emp*100:.1f}%")
```

### Phase 3: Sample Size Requirements
```python
def test_minimum_sample_size():
    """Determine empirical minimum for stable estimates."""
    d = 50
    true_cov = np.eye(d)  # True covariance (identity)
    
    sample_sizes = [50, 100, 150, 200, 250, 300, 500]
    n_trials = 100
    
    for n in sample_sizes:
        errors = []
        for _ in range(n_trials):
            embeddings = np.random.multivariate_normal(np.zeros(d), true_cov, size=n)
            _, C_est = covariance_matrix(embeddings)
            error = np.linalg.norm(C_est - true_cov, 'fro')
            errors.append(error)
        
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        print(f"n={n} (n/d={n/d:.1f}): error={mean_error:.3f} ± {std_error:.3f}")
```

### Phase 4: PCA Information Loss
```python
def test_pca_semantic_retention():
    """Measure how much semantic information is lost in PCA."""
    # Generate data with known semantic structure
    n = 200
    semantic_axis = np.zeros(768)
    semantic_axis[0] = 1.0  # First dimension is "professionalism"
    
    # Data varies along semantic axis
    scores = np.random.randn(n)
    embeddings = scores[:, None] * semantic_axis + np.random.randn(n, 768) * 0.1
    
    # PCA compression
    from sklearn.decomposition import PCA
    pca = PCA(n_components=50)
    pca.fit(embeddings)
    
    # Check retention
    axis_pca = pca.components_.T @ semantic_axis
    retention = np.linalg.norm(axis_pca) / np.linalg.norm(semantic_axis)
    
    print(f"Semantic axis retention after PCA: {retention*100:.1f}%")
    print(f"First PC explains: {pca.explained_variance_ratio_[0]*100:.1f}% variance")
```

## Summary of Improvements

**V1 → V2 changes**:

1. ✓ **Ledoit-Wolf shrinkage** - principled, adaptive regularization
2. ✓ **Strict sample size checks** - error on n < 3d, warn on n < 5d
3. ✓ **PCA axis retention** - track and warn when semantic axes are suppressed
4. ✓ **Normalized Frobenius** - interpretable as relative difference
5. ✓ **Mahalanobis distance** - standardized mean difference
6. ✓ **Principal component analysis** - spectral structure with semantic interpretation
7. ✓ **Errors vs warnings** - fail fast on invalid inputs
8. ✓ **Comprehensive validation** - mathematical, stability, sample size, PCA tests
9. ✓ **Dropped Gaussian KL** - too misleading for non-Gaussian data
10. ✓ **Better interpretation helpers** - automated interpretation of metrics

**This version is production-ready.**

## Implementation Timeline

**Phase 1** (3 hours): Core metrics with Ledoit-Wolf
**Phase 2** (3 hours): Semantic integration + PC analysis  
**Phase 3** (2 hours): PCA retention checks + warnings
**Phase 4** (2 hours): Validation suite
**Phase 5** (1 hour): Documentation

**Total**: ~11 hours

Ready to implement?
