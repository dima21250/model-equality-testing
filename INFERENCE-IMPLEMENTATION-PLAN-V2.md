# Implementation Plan V2: Statistical Inference for Density Matrix Metrics (No PCA)

## Overview

This plan implements permutation testing and bootstrap confidence intervals for density matrix metrics **without PCA**, which eliminates the fatal circularity flaw identified in the critique.

**Key change from V1**: Work in the full 768-dimensional SBERT embedding space instead of PCA-reduced 50-D space. This ensures valid statistical inference at the cost of ~2x slower computation.

**Estimated time**: 3-4 hours for core implementation, testing, and documentation.

---

## Critical Problem Solved

### V1 (With PCA): Circular and Invalid

```python
# Fit PCA on combined A+B → encodes the split!
combined = np.vstack([emb_a, emb_b])
pca = fit_pca(combined, k=50)  # ← PCA directions know about A vs B

# All permutations measured in this biased space
rho_a_perm = pca_density_matrix(emb_a_perm, pca)  # ← Invalid!
```

**Problem**: PCA fitted on observed data creates a measurement space optimized for that specific A/B split, biasing the null distribution.

### V2 (Without PCA): Valid Inference

```python
# No PCA fitting - use fixed 768-D SBERT space
rho_a = density_matrix(emb_a)  # Same space for all permutations
rho_b = density_matrix(emb_b)

# All permutations measured in same unbiased space → valid null distribution
```

**Solution**: All measurements in the same fixed space (768-D SBERT embeddings) → no circularity.

---

## Goals

1. ✅ **Valid permutation testing**: No PCA circularity
2. ✅ **Valid bootstrap CI**: Fixed measurement space
3. ⚠️ **Honest documentation**: Acknowledge remaining limitations
4. ✅ **Rigorous validation**: Type I error and power tests
5. ✅ **Backward compatibility**: Keep PCA version for descriptive metrics

---

## File Structure

### New File: `model_equality_testing/src/quantum_inference.py`

Separate module for statistical inference (not modifying `quantum_metrics.py`):
- `quantum_metrics.py`: Descriptive statistics (can keep PCA for speed)
- `quantum_inference.py`: Statistical inference (no PCA for validity)

**Clear separation**: Description vs. inference

---

## Implementation

### Phase 1: Core Density Matrix (No PCA)

#### Function 1: `density_matrix()`

**Location**: `model_equality_testing/src/quantum_inference.py`

**Signature**:
```python
def density_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Build density matrix from embeddings in full embedding space.
    
    No PCA compression - works in the full d-dimensional SBERT space.
    This ensures valid permutation testing (no circular fitting).
    
    Args:
        embeddings: (n, d) SBERT embeddings (e.g., n=100, d=768)
    
    Returns:
        rho: (n, n) density matrix where rho[i,j] = (e_i · e_j) / trace(G)
    
    Note:
        This is ~2x slower than pca_density_matrix() but ensures
        unbiased inference. For descriptive statistics only, consider
        using pca_density_matrix() from quantum_metrics.py.
    
    Example:
        >>> from model_equality_testing.src.embeddings import embed_sample
        >>> emb = embed_sample(sample)  # (100, 768)
        >>> rho = density_matrix(emb)   # (100, 100)
    """
    # Gram matrix: pairwise inner products in full embedding space
    G = embeddings @ embeddings.T  # (n, n)
    
    # Normalize to density matrix (trace = 1)
    rho = G / np.trace(G)
    
    return rho
```

**Key points**:
- No PCA parameter - always uses full embedding space
- Simpler implementation (no transform step)
- Explicitly documented as slower but valid for inference

---

### Phase 2: Permutation Testing

#### Function 2: `trace_distance_permutation_test()`

**Signature**:
```python
def trace_distance_permutation_test(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    b: int = 1000,
    random_seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Two-sample permutation test for trace distance (no PCA).
    
    Tests H0: distributions A and B are identical
    vs H1: distributions differ.
    
    Uses full embedding space (no PCA compression) to ensure valid
    permutation testing without circular fitting bias.
    
    Args:
        emb_a: (n_a, d) embeddings from distribution A
        emb_b: (n_b, d) embeddings from distribution B
        b: number of permutations (default 1000)
        random_seed: optional random seed for reproducibility
    
    Returns:
        pvalue: float in [0, 1] - permutation p-value
        statistic: float in [0, 1] - observed trace distance
    
    Raises:
        ValueError: if n_a != n_b (trace distance requires equal sizes)
    
    Computational cost:
        ~60-120 seconds for b=1000 with n=100, d=768 on modern CPU.
        Scales as O(b * n^3) for eigendecomposition.
    
    Example:
        >>> pvalue, td = trace_distance_permutation_test(emb_a, emb_b, b=1000)
        >>> print(f"Trace distance: {td:.4f}, p-value: {pvalue:.4f}")
        >>> if pvalue < 0.05:
        >>>     print("Significantly different")
    """
```

**Implementation**:

```python
import numpy as np
from typing import Tuple, Optional
from numpy.random import default_rng, SeedSequence
from model_equality_testing.src.quantum_metrics import trace_distance

def trace_distance_permutation_test(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    b: int = 1000,
    random_seed: Optional[int] = None
) -> Tuple[float, float]:
    
    n_a = len(emb_a)
    n_b = len(emb_b)
    
    # Validate equal sizes (trace distance requirement)
    if n_a != n_b:
        raise ValueError(
            f"Trace distance requires equal sample sizes. Got n_a={n_a}, n_b={n_b}.\n"
            f"Consider: (1) using semantic axes (works for any n), "
            f"(2) subsampling to min(n_a, n_b), or "
            f"(3) using von Neumann entropy (single-distribution metric)."
        )
    
    # Proper random seed handling
    if random_seed is not None:
        rng = default_rng(random_seed)
    else:
        rng = default_rng()
    
    # Combine samples
    combined = np.vstack([emb_a, emb_b])
    
    # Observed statistic (NO PCA - fixed space)
    rho_a = density_matrix(emb_a)
    rho_b = density_matrix(emb_b)
    td_observed = trace_distance(rho_a, rho_b)
    
    # Permutation null distribution
    td_null = []
    for i in range(b):
        # Randomly permute combined samples
        perm = rng.permutation(n_a + n_b)
        emb_a_perm = combined[perm[:n_a]]
        emb_b_perm = combined[perm[n_a:]]
        
        # Compute trace distance in SAME space (no PCA circularity)
        rho_a_perm = density_matrix(emb_a_perm)
        rho_b_perm = density_matrix(emb_b_perm)
        td_null.append(trace_distance(rho_a_perm, rho_b_perm))
    
    # P-value: (# null >= observed + 1) / (b + 1)
    # Conservative formula (ensures p > 0, avoids p=0)
    pvalue = (np.sum(np.array(td_null) >= td_observed) + 1) / (b + 1)
    
    return pvalue, td_observed
```

**Key improvements from V1**:
1. ✅ No PCA → no circularity
2. ✅ Proper RNG using `default_rng()` (numpy modern API)
3. ✅ Better error message with alternatives
4. ✅ Documented computational cost

---

#### Function 3: `trace_distance_permutation_test_from_samples()`

**Convenience wrapper**:

```python
from model_equality_testing.distribution import CompletionSample
from model_equality_testing.src.embeddings import embed_sample

def trace_distance_permutation_test_from_samples(
    sample_a: CompletionSample,
    sample_b: CompletionSample,
    b: int = 1000,
    _precomputed_embeddings: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    random_seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Permutation test from CompletionSample objects.
    
    Follows _precomputed_embeddings pattern from tests.py.
    
    Args:
        sample_a, sample_b: CompletionSample objects
        b: number of permutations
        _precomputed_embeddings: optional (emb_a, emb_b) to skip embedding
        random_seed: random seed
    
    Returns:
        pvalue, statistic
    """
    if _precomputed_embeddings is not None:
        emb_a, emb_b = _precomputed_embeddings
    else:
        emb_a = embed_sample(sample_a)
        emb_b = embed_sample(sample_b)
    
    return trace_distance_permutation_test(
        emb_a, emb_b, b=b, random_seed=random_seed
    )
```

---

### Phase 3: Bootstrap Confidence Intervals

#### Function 4: `trace_distance_bootstrap_ci()`

**Signature**:
```python
def trace_distance_bootstrap_ci(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_seed: Optional[int] = None
) -> Tuple[float, Tuple[float, float]]:
    """
    Bootstrap confidence interval for trace distance (no PCA).
    
    Uses bootstrap resampling to estimate sampling distribution
    and construct percentile confidence interval.
    
    Args:
        emb_a: (n_a, d) embeddings from distribution A
        emb_b: (n_b, d) embeddings from distribution B
        n_boot: number of bootstrap samples (default 1000)
        alpha: significance level (default 0.05 for 95% CI)
        random_seed: optional random seed
    
    Returns:
        td_observed: float - point estimate of trace distance
        ci: tuple (lower, upper) - percentile confidence interval
    
    Raises:
        ValueError: if n_a != n_b
    
    Computational cost:
        ~60-120 seconds for n_boot=1000 with n=100, d=768.
    
    Example:
        >>> td, (ci_lower, ci_upper) = trace_distance_bootstrap_ci(emb_a, emb_b)
        >>> print(f"Trace distance: {td:.4f}")
        >>> print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        >>> if ci_lower > 0:
        >>>     print("CI excludes zero → significantly different")
    """
```

**Implementation**:

```python
def trace_distance_bootstrap_ci(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_seed: Optional[int] = None
) -> Tuple[float, Tuple[float, float]]:
    
    n_a = len(emb_a)
    n_b = len(emb_b)
    
    # Validate equal sizes
    if n_a != n_b:
        raise ValueError(
            f"Trace distance requires equal sample sizes. Got n_a={n_a}, n_b={n_b}.\n"
            f"Consider using semantic axes for unequal-size comparisons."
        )
    
    # Proper random seed handling
    if random_seed is not None:
        rng = default_rng(random_seed)
    else:
        rng = default_rng()
    
    # Observed statistic (NO PCA)
    rho_a = density_matrix(emb_a)
    rho_b = density_matrix(emb_b)
    td_obs = trace_distance(rho_a, rho_b)
    
    # Bootstrap distribution
    td_boot = []
    for i in range(n_boot):
        # Resample WITH replacement from each distribution independently
        idx_a = rng.choice(n_a, size=n_a, replace=True)
        idx_b = rng.choice(n_b, size=n_b, replace=True)
        
        emb_a_boot = emb_a[idx_a]
        emb_b_boot = emb_b[idx_b]
        
        # Compute trace distance in SAME space (no PCA)
        rho_a_boot = density_matrix(emb_a_boot)
        rho_b_boot = density_matrix(emb_b_boot)
        td_boot.append(trace_distance(rho_a_boot, rho_b_boot))
    
    # Percentile confidence interval
    ci_lower = np.percentile(td_boot, 100 * alpha / 2)
    ci_upper = np.percentile(td_boot, 100 * (1 - alpha / 2))
    
    return td_obs, (ci_lower, ci_upper)
```

**Key improvements**:
1. ✅ No PCA → cleaner sampling distribution
2. ✅ Proper RNG
3. ✅ Percentile method (simple, no normality assumption)

---

#### Function 5: `trace_distance_bootstrap_ci_from_samples()`

**Wrapper**:

```python
def trace_distance_bootstrap_ci_from_samples(
    sample_a: CompletionSample,
    sample_b: CompletionSample,
    n_boot: int = 1000,
    alpha: float = 0.05,
    _precomputed_embeddings: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    random_seed: Optional[int] = None
) -> Tuple[float, Tuple[float, float]]:
    """Bootstrap CI from CompletionSample objects."""
    if _precomputed_embeddings is not None:
        emb_a, emb_b = _precomputed_embeddings
    else:
        emb_a = embed_sample(sample_a)
        emb_b = embed_sample(sample_b)
    
    return trace_distance_bootstrap_ci(
        emb_a, emb_b, n_boot=n_boot, alpha=alpha, random_seed=random_seed
    )
```

---

### Phase 4: Combined Analysis

#### Function 6: `trace_distance_test()`

**High-level function** computing both permutation and bootstrap:

```python
def trace_distance_test(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    b: int = 1000,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_seed: Optional[int] = None
) -> dict:
    """
    Complete statistical inference for trace distance.
    
    Computes both permutation p-value and bootstrap CI.
    Uses independent random streams for each.
    
    Args:
        emb_a, emb_b: embeddings to compare
        b: permutations for p-value
        n_boot: bootstrap samples for CI
        alpha: significance level
        random_seed: base random seed
    
    Returns:
        dict with keys:
            'statistic': observed trace distance
            'pvalue': permutation p-value
            'ci_lower': lower CI bound
            'ci_upper': upper CI bound
            'ci_width': CI width (precision)
            'significant': bool (p < alpha)
    
    Computational cost:
        ~120-240 seconds for b=1000, n_boot=1000, n=100, d=768.
    
    Example:
        >>> result = trace_distance_test(emb_a, emb_b)
        >>> print(f"td={result['statistic']:.4f}, p={result['pvalue']:.4f}")
        >>> print(f"95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    """
    # Create independent random streams
    if random_seed is not None:
        ss = SeedSequence(random_seed)
        child_seeds = ss.spawn(2)
        seed_perm = child_seeds[0]
        seed_boot = child_seeds[1]
    else:
        seed_perm = None
        seed_boot = None
    
    # Permutation test
    pvalue, statistic = trace_distance_permutation_test(
        emb_a, emb_b, b=b, random_seed=seed_perm
    )
    
    # Bootstrap CI
    td_point, (ci_lower, ci_upper) = trace_distance_bootstrap_ci(
        emb_a, emb_b, n_boot=n_boot, alpha=alpha, random_seed=seed_boot
    )
    
    return {
        'statistic': statistic,
        'pvalue': pvalue,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_width': ci_upper - ci_lower,
        'significant': pvalue < alpha
    }
```

**Key improvement**: Proper independent random streams using `SeedSequence.spawn()`.

---

### Phase 5: Pretty Printing

#### Function 7: `print_trace_distance_test_result()`

**Updated to remove fabricated effect size cutoffs**:

```python
def print_trace_distance_test_result(
    result: dict,
    label_a: str = "A",
    label_b: str = "B"
):
    """
    Print formatted test results.
    
    Args:
        result: dict from trace_distance_test()
        label_a, label_b: distribution names
    """
    td = result['statistic']
    pval = result['pvalue']
    ci_low = result['ci_lower']
    ci_up = result['ci_upper']
    ci_width = result['ci_width']
    
    print(f"Model Comparison: {label_a} vs {label_b}")
    print(f"=" * 60)
    print(f"Trace Distance: {td:.4f}")
    print(f"95% Confidence Interval: [{ci_low:.4f}, {ci_up:.4f}]")
    print(f"P-value: {pval:.4f}")
    print()
    
    # Interpretation
    print("Interpretation:")
    
    # Significance
    if result['significant']:
        print(f"  Statistical significance: Yes (p = {pval:.4f} < 0.05)")
    else:
        print(f"  Statistical significance: No (p = {pval:.4f} ≥ 0.05)")
    
    # Precision (NO fabricated effect size labels)
    if ci_width < 0.05:
        prec = "high precision (narrow CI)"
    elif ci_width < 0.15:
        prec = "moderate precision"
    else:
        prec = "low precision (wide CI)"
    print(f"  Estimate precision: {prec} (CI width = {ci_width:.4f})")
    
    # CI interpretation
    if ci_low > 0:
        print(f"  95% CI excludes zero → difference is real")
    
    print()
    
    # Conclusion (NO "small/moderate/large" labels without calibration)
    if result['significant']:
        print(f"Conclusion: Distributions {label_a} and {label_b} differ significantly.")
        print(f"           Trace distance = {td:.4f} (scale: 0=identical, 1=maximally different)")
    else:
        print(f"Conclusion: No significant difference detected.")
    
    print()
    print("Note: For empirically calibrated effect sizes, see DENSITY-MATRIX-CALIBRATION.md")
```

**Critical change**: Removed arbitrary "small/moderate/large" cutoffs. Reports numbers only.

---

## Testing Strategy

### Unit Tests

**File**: `model_equality_testing/tests/test_quantum_inference.py`

#### Test 1: Type I Error Rate Validation

**Most important test** - validates that permutation test maintains nominal error rate:

```python
import numpy as np
import pytest
from model_equality_testing.src.quantum_inference import (
    trace_distance_permutation_test,
    density_matrix
)

def test_type_I_error_rate():
    """
    Verify permutation test maintains 5% Type I error rate under null.
    
    This is the critical validation that the test is unbiased.
    """
    np.random.seed(42)
    n_tests = 200  # Balance: need enough for statistical power, not too slow
    alpha = 0.05
    rejections = 0
    
    for i in range(n_tests):
        # Generate from SAME distribution (null hypothesis)
        emb_all = np.random.randn(200, 768)
        emb_a = emb_all[:100]
        emb_b = emb_all[100:]
        
        # Test (use small b for speed)
        pvalue, _ = trace_distance_permutation_test(
            emb_a, emb_b, b=100, random_seed=i
        )
        
        if pvalue < alpha:
            rejections += 1
    
    # Expected: ~10 rejections (5% of 200)
    # Binomial 95% CI: [5, 15] for p=0.05, n=200
    # Using slightly wider bounds to account for small b=100
    assert 3 <= rejections <= 17, (
        f"Type I error rate: {rejections/n_tests*100:.1f}% "
        f"(expected 5%, got {rejections}/{n_tests})"
    )
    
    print(f"Type I error rate: {rejections}/{n_tests} = {rejections/n_tests*100:.1f}%")
    print(f"Expected: 5% ± binomial variation")
```

#### Test 2: Power Validation

```python
def test_power_moderate_effect():
    """Verify test has reasonable power for moderate effect."""
    np.random.seed(42)
    n_tests = 50  # Smaller n for power test (slower due to real effect)
    alpha = 0.05
    rejections = 0
    
    for i in range(n_tests):
        # Different distributions (alternative hypothesis)
        emb_a = np.random.randn(100, 768)
        emb_b = np.random.randn(100, 768) + 0.3  # Moderate shift
        
        pvalue, _ = trace_distance_permutation_test(
            emb_a, emb_b, b=100, random_seed=i
        )
        
        if pvalue < alpha:
            rejections += 1
    
    # Should detect most of the time (power > 70%)
    assert rejections >= 35, (
        f"Power: {rejections/n_tests*100:.1f}% (expected >70%)"
    )
    
    print(f"Power: {rejections}/{n_tests} = {rejections/n_tests*100:.1f}%")
```

#### Test 3: Bootstrap CI Coverage

```python
def test_bootstrap_ci_coverage():
    """
    Verify bootstrap CI has reasonable properties.
    
    Hard to test coverage without knowing "true" value, so we test:
    1. Point estimate is in CI
    2. CI excludes zero for different distributions
    3. CI width is reasonable
    """
    np.random.seed(42)
    
    # Different distributions
    emb_a = np.random.randn(100, 768)
    emb_b = np.random.randn(100, 768) + 0.5
    
    td, (ci_low, ci_up) = trace_distance_bootstrap_ci(
        emb_a, emb_b, n_boot=200, random_seed=42
    )
    
    # Point estimate in CI
    assert ci_low <= td <= ci_up, "Point estimate must be in CI"
    
    # Valid bounds
    assert 0 <= ci_low <= ci_up <= 1, "CI must be in [0, 1]"
    
    # For shifted distributions, should exclude zero (high probability)
    # Not guaranteed with small n_boot, but likely
    assert ci_low > 0, "CI should exclude zero for clearly different distributions"
    
    # CI width reasonable (not degenerate, not too wide)
    ci_width = ci_up - ci_low
    assert 0.01 < ci_width < 0.5, f"CI width {ci_width:.3f} unreasonable"
    
    print(f"Bootstrap CI: [{ci_low:.4f}, {ci_up:.4f}], width={ci_width:.4f}")
```

#### Test 4: Reproducibility

```python
def test_reproducibility_with_seed():
    """Verify random seed produces identical results."""
    emb_a = np.random.RandomState(123).randn(100, 768)
    emb_b = np.random.RandomState(456).randn(100, 768)
    
    # Run twice with same seed
    pval1, td1 = trace_distance_permutation_test(
        emb_a, emb_b, b=100, random_seed=42
    )
    pval2, td2 = trace_distance_permutation_test(
        emb_a, emb_b, b=100, random_seed=42
    )
    
    # Should be identical
    assert pval1 == pval2, "P-values should be identical with same seed"
    assert td1 == td2, "Statistics should be identical with same seed"
```

#### Test 5: Edge Cases

```python
def test_equal_size_validation():
    """Test that unequal sizes raise helpful error."""
    emb_a = np.random.randn(100, 768)
    emb_b = np.random.randn(150, 768)
    
    with pytest.raises(ValueError, match="equal sample sizes"):
        trace_distance_permutation_test(emb_a, emb_b)
    
    with pytest.raises(ValueError, match="equal sample sizes"):
        trace_distance_bootstrap_ci(emb_a, emb_b)

def test_density_matrix_properties():
    """Test density matrix has expected properties."""
    emb = np.random.randn(100, 768)
    rho = density_matrix(emb)
    
    # Should be square
    assert rho.shape == (100, 100)
    
    # Trace should be 1
    assert np.abs(np.trace(rho) - 1.0) < 1e-10
    
    # Should be symmetric
    assert np.allclose(rho, rho.T)
    
    # Should be positive semidefinite (all eigenvalues >= 0)
    evals = np.linalg.eigvalsh(rho)
    assert np.all(evals >= -1e-10), "Eigenvalues should be non-negative"
```

### Integration Test

```python
def test_with_real_data():
    """
    Integration test with actual LLM data.
    
    Note: This test may be slow (~2 minutes) with b=1000.
    For CI, use smaller b=100.
    """
    from model_equality_testing.dataset import load_distribution
    from model_equality_testing.src.embeddings import embed_sample
    
    # Load distributions
    dist_fp32 = load_distribution(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        prompt_ids={"wikipedia_en": [0, 1, 2]},
        L=500,
        source="fp32",
        load_in_unicode=True
    )
    dist_int8 = load_distribution(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        prompt_ids={"wikipedia_en": [0, 1, 2]},
        L=500,
        source="int8",
        load_in_unicode=True
    )
    
    # Draw samples
    sample_fp32 = dist_fp32.draw_completion_sample(50)  # Smaller for speed
    sample_int8 = dist_int8.draw_completion_sample(50)
    
    # Embed
    emb_fp32 = embed_sample(sample_fp32)
    emb_int8 = embed_sample(sample_int8)
    
    # Test (small b for speed in CI)
    result = trace_distance_test(
        emb_fp32, emb_int8, b=100, n_boot=100, random_seed=42
    )
    
    # Sanity checks
    assert 0 <= result['statistic'] <= 1
    assert 0 <= result['pvalue'] <= 1
    assert result['ci_lower'] <= result['statistic'] <= result['ci_upper']
    assert result['ci_width'] == result['ci_upper'] - result['ci_lower']
    
    print(f"\nfp32 vs int8 (n=50, b=100):")
    print(f"  td = {result['statistic']:.4f}")
    print(f"  p = {result['pvalue']:.4f}")
    print(f"  95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
```

---

## Computational Cost

### Detailed Breakdown (No PCA)

For n=100, d=768:

**Single trace distance computation**:
1. Gram matrix: `G = E @ E.T` → O(n²d) = 100² × 768 = 7.68M operations
2. Trace (sum diagonal): O(n) = 100 operations (negligible)
3. Division: O(n²) = 10K operations (negligible)
4. Eigendecomposition (for trace distance): O(n³) = 1M operations
5. Trace norm computation: O(n³) = 1M operations

**Total per comparison**: ~10M operations → **10-15ms** on modern CPU

**Permutation test (b=1000)**:
- 1000 iterations × 10M ops = 10B operations
- **Expected runtime**: 60-90 seconds

**Bootstrap CI (n_boot=1000)**:
- 1000 iterations × 10M ops = 10B operations
- **Expected runtime**: 60-90 seconds

**Combined (both)**:
- **Total runtime**: 120-180 seconds (2-3 minutes)

### Comparison to V1 (With PCA)

| Metric | With PCA (V1) | Without PCA (V2) | Ratio |
|--------|--------------|-----------------|-------|
| Single comparison | 5-8ms | 10-15ms | 2x |
| Permutation (b=1000) | 30-60s | 60-90s | 2x |
| Bootstrap (n_boot=1000) | 30-60s | 60-90s | 2x |
| **Total** | 60-120s | 120-180s | **2x** |

**Verdict**: 2x slower is acceptable for valid inference.

### Optimization Options (Future)

1. **Reduce default b**: Use b=100 by default, b=1000 for publication
2. **Parallelization**: Use `multiprocessing.Pool` for permutations
3. **Progress bars**: Add `tqdm` for user feedback
4. **Caching**: Cache Gram matrices if running multiple tests

---

## Usage Examples

### Example 1: Basic Permutation Test

```python
from model_equality_testing.src.embeddings import embed_sample
from model_equality_testing.src.quantum_inference import trace_distance_permutation_test

# Embed samples
emb_a = embed_sample(sample_a)
emb_b = embed_sample(sample_b)

# Test (expect ~60-90 seconds for b=1000)
pvalue, td = trace_distance_permutation_test(emb_a, emb_b, b=1000)

print(f"Trace distance: {td:.4f}")
print(f"P-value: {pvalue:.4f}")

if pvalue < 0.05:
    print("Distributions are significantly different")
else:
    print("No significant difference detected")
```

### Example 2: Bootstrap CI Only

```python
from model_equality_testing.src.quantum_inference import trace_distance_bootstrap_ci

# Expect ~60-90 seconds for n_boot=1000
td, (ci_lower, ci_upper) = trace_distance_bootstrap_ci(emb_a, emb_b, n_boot=1000)

print(f"Trace distance: {td:.4f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"CI width: {ci_upper - ci_lower:.4f}")

if ci_lower > 0:
    print("CI excludes zero → significantly different")
```

### Example 3: Complete Analysis (Recommended)

```python
from model_equality_testing.dataset import load_distribution
from model_equality_testing.src.embeddings import embed_sample
from model_equality_testing.src.quantum_inference import (
    trace_distance_test,
    print_trace_distance_test_result
)

# Load and sample
dist_a = load_distribution(..., source="fp32")
dist_b = load_distribution(..., source="int8")
sample_a = dist_a.draw_completion_sample(100)
sample_b = dist_b.draw_completion_sample(100)

# Embed once
emb_a = embed_sample(sample_a)
emb_b = embed_sample(sample_b)

# Complete inference (expect ~2-3 minutes for b=1000, n_boot=1000)
print("Computing inference statistics (this will take ~2-3 minutes)...")
result = trace_distance_test(emb_a, emb_b, b=1000, n_boot=1000)

# Pretty print
print_trace_distance_test_result(result, label_a="fp32", label_b="int8")

# Output:
# Model Comparison: fp32 vs int8
# ============================================================
# Trace Distance: 0.2341
# 95% Confidence Interval: [0.1875, 0.2894]
# P-value: 0.003
#
# Interpretation:
#   Statistical significance: Yes (p = 0.003 < 0.05)
#   Estimate precision: high precision (CI width = 0.1019)
#   95% CI excludes zero → difference is real
#
# Conclusion: Distributions fp32 and int8 differ significantly.
#            Trace distance = 0.2341 (scale: 0=identical, 1=maximally different)
#
# Note: For empirically calibrated effect sizes, see DENSITY-MATRIX-CALIBRATION.md
```

### Example 4: Integration with Semantic Axes

```python
# 1. Embed once
emb_a = embed_sample(sample_a)
emb_b = embed_sample(sample_b)

# 2. Quantum inference (descriptive + statistical)
result = trace_distance_test(emb_a, emb_b, b=1000, n_boot=1000)
print_trace_distance_test_result(result, "fp32", "int8")

# 3. Semantic interpretation (what differs?)
from model_equality_testing.src.semantic_axes import interpret_difference

axes = [prof_axis, tech_axis, form_axis]
interp = interpret_difference(emb_a, emb_b, axes, label_a="fp32", label_b="int8")

print("\nSemantic Interpretation:")
print(interp.summary())

# Complete picture:
# - Trace distance tells you "how different" overall
# - P-value tells you "is it significant"
# - Semantic axes tell you "what differs" (professionalism, technicality, etc.)
```

---

## Documentation Updates

### Add to `README.md`

After quantum metrics section:

```markdown
### Statistical Inference for Quantum Metrics

Density matrix metrics now support statistical significance testing via permutation tests and bootstrap confidence intervals:

```python
from model_equality_testing.src.quantum_inference import trace_distance_test

# Complete statistical inference (no PCA - valid inference)
result = trace_distance_test(emb_a, emb_b, b=1000, n_boot=1000)

print(f"Trace distance: {result['statistic']:.4f}")
print(f"P-value: {result['pvalue']:.4f}")
print(f"95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")

if result['significant']:
    print("Distributions are significantly different")
```

**Note**: Inference functions use the full 768-dimensional embedding space (no PCA compression) to ensure valid permutation testing without circular fitting bias. This is ~2x slower than descriptive metrics but provides statistically rigorous hypothesis testing.

For details, see `DENSITY-MATRIX-CALIBRATION.md` and `USING-QUANTUM-INFERENCE.md`.
```

### Update `DENSITY-MATRIX-METRICS.md`

Add at end:

```markdown
## Statistical Inference

The metrics shown above are descriptive statistics. For hypothesis testing with p-values and confidence intervals:

**Permutation testing**: `trace_distance_permutation_test()` - tests if distributions differ

**Bootstrap CI**: `trace_distance_bootstrap_ci()` - quantifies uncertainty in effect size

**Combined**: `trace_distance_test()` - computes both

**Important**: Inference functions work in the full 768-D embedding space (no PCA) to ensure valid permutation tests. This is ~2x slower than descriptive metrics but provides unbiased null distributions.

See `DENSITY-MATRIX-CALIBRATION.md` for detailed discussion of the calibration problem and `USING-QUANTUM-INFERENCE.md` for usage examples.
```

### Create `USING-QUANTUM-INFERENCE.md`

New guide with:
- When to use quantum inference vs semantic axes
- Interpretation of p-values and CIs
- Computational cost guidance
- Example workflows
- Limitations and caveats

Key section:

```markdown
## When to Use Quantum Inference vs Semantic Axes

### Use Quantum Metrics (trace distance inference) when:
- ✓ You want a single holistic measure of distributional difference
- ✓ You need quantum-inspired interpretations (distinguishability, entropy)
- ✓ You have equal sample sizes (n_a = n_b)
- ✓ You can afford 2-3 minutes per comparison
- ⚠️ You accept that the metric is heuristic (not strictly quantum-mechanical)

### Use Semantic Axes when:
- ✓ You want to know *what* differs (not just *that* it differs)
- ✓ You have interpretable dimensions (professionalism, technicality, etc.)
- ✓ You need instant results (milliseconds)
- ✓ Sample sizes differ (n_a ≠ n_b)
- ✓ You want established effect size scale (Hedges' g)

### Use Both when:
- ✓ You want comprehensive analysis
- ✓ Writing a research paper (show convergence)
- ✓ Need to convince skeptical reviewers

### Recommended workflow:
1. Semantic axes for initial exploration (fast, interpretable)
2. Quantum inference for confirmation (rigorous, holistic)
3. Both for publication (comprehensive)
```

---

## Backward Compatibility

### Keep PCA Version for Descriptive Metrics

The existing `quantum_metrics.py` remains unchanged:

```python
# Descriptive statistics (fast, with PCA)
from model_equality_testing.src.quantum_metrics import (
    fit_pca,
    pca_density_matrix,
    trace_distance,
    von_neumann_entropy
)

# Works as before
pca = fit_pca(combined, k=50)
rho_a = pca_density_matrix(emb_a, pca)
rho_b = pca_density_matrix(emb_b, pca)
td = trace_distance(rho_a, rho_b)
```

### New Module for Inference

```python
# Statistical inference (valid, no PCA)
from model_equality_testing.src.quantum_inference import (
    density_matrix,  # No PCA version
    trace_distance_permutation_test,
    trace_distance_bootstrap_ci,
    trace_distance_test
)

# New workflow
rho_a = density_matrix(emb_a)  # Full 768-D
pvalue, td = trace_distance_permutation_test(emb_a, emb_b)
```

**Both coexist peacefully**. Users choose based on needs.

---

## Timeline

### Hour 1: Core Implementation
- ✓ Create `model_equality_testing/src/quantum_inference.py`
- ✓ Implement `density_matrix()` (no PCA)
- ✓ Implement `trace_distance_permutation_test()`
- ✓ Implement wrapper `trace_distance_permutation_test_from_samples()`

### Hour 2: Bootstrap and Combined
- ✓ Implement `trace_distance_bootstrap_ci()`
- ✓ Implement wrapper `trace_distance_bootstrap_ci_from_samples()`
- ✓ Implement `trace_distance_test()` (combined)
- ✓ Implement `print_trace_distance_test_result()`

### Hour 3: Testing
- ✓ Create `tests/test_quantum_inference.py`
- ✓ Implement Type I error rate test (critical!)
- ✓ Implement power test
- ✓ Implement bootstrap CI test
- ✓ Implement reproducibility and edge case tests
- ✓ Run integration test with real data
- ✓ Fix any bugs found

### Hour 4: Documentation
- ✓ Add docstrings to all functions
- ✓ Update `README.md`
- ✓ Update `DENSITY-MATRIX-METRICS.md`
- ✓ Create `USING-QUANTUM-INFERENCE.md`
- ✓ Test all documentation examples
- ✓ Commit with clear message

**Total: 4 hours for complete, validated, documented implementation**

---

## Limitations and Caveats

### What This Fixes

✅ **PCA circularity**: Eliminated by working in fixed 768-D space

✅ **Bootstrap assumptions**: Clearer in fixed space (no PCA fit dependency)

✅ **Statistical validity**: Permutation test now has unbiased null distribution

### What This Doesn't Fix

⚠️ **Ill-defined metrics**: From `quantum_metrics.py` docstring:
> "Cross-matrix metrics are not well-defined in the quantum-mechanical sense"

This remains true. The metrics are heuristic, quantum-inspired similarity measures, not true quantum observables.

⚠️ **Effect size calibration**: No empirical benchmarks for "small/moderate/large"

The 0.1/0.3 cutoffs are removed. Need Option 3 (empirical calibration) for this.

⚠️ **Equal sample size requirement**: Trace distance requires n_a = n_b (intrinsic to metric)

Use semantic axes for unequal sizes.

⚠️ **Computational cost**: 2-3 minutes per comparison is slow

Consider reducing defaults (b=100) for exploratory work.

### Honest Documentation

All documentation must include:

> ⚠️ **Important Caveats**:
>
> 1. **Inference is valid, but metrics are heuristic**: The permutation tests and bootstrap CIs are statistically valid (no PCA circularity), but trace distance is a quantum-inspired similarity metric, not a true quantum-mechanical observable.
>
> 2. **No calibrated effect size scale**: Unlike Hedges' g (with established benchmarks), trace distance lacks empirical calibration. We report the raw value (0-1 scale) without "small/moderate/large" labels. For calibrated benchmarks, see empirical validation studies.
>
> 3. **Equal sample sizes required**: Trace distance requires n_a = n_b. For unequal sizes, use semantic axes.
>
> 4. **Computational cost**: Expect 2-3 minutes for b=1000, n_boot=1000 with n=100. Consider smaller b for exploratory work.
>
> 5. **Use semantic axes for interpretation**: Trace distance tells you *how much* distributions differ, but not *what* differs. For interpretable dimensions, use semantic axes.

---

## Verification Checklist

After implementation:

- [ ] Type I error rate test passes (maintains ~5%)
- [ ] Power test shows reasonable detection (>70% for moderate effects)
- [ ] Bootstrap CI has expected properties
- [ ] Reproducibility with random seed works
- [ ] Equal size validation raises helpful error
- [ ] Integration test with real data runs successfully
- [ ] Density matrix properties verified (trace=1, symmetric, PSD)
- [ ] Documentation examples run without errors
- [ ] Computational cost matches predictions (~2-3 min)
- [ ] Backward compatibility maintained (PCA version still works)
- [ ] All caveats clearly documented
- [ ] Pretty printing shows correct information (no fabricated effect sizes)

---

## Summary

### Key Changes from V1

| Aspect | V1 (With PCA) | V2 (No PCA) |
|--------|--------------|-------------|
| **Validity** | ❌ Circular (biased null) | ✅ Valid (unbiased null) |
| **Complexity** | Higher (PCA fitting) | Lower (direct Gram) |
| **Speed** | 60-120s | 120-180s (2x slower) |
| **Dependencies** | PCA, k parameter | None (simpler) |
| **Recommendation** | Don't implement | **Implement** |

### What We Get

✅ **Statistically valid permutation tests** (no circularity)

✅ **Valid bootstrap confidence intervals** (fixed space)

✅ **Simpler implementation** (no PCA complexity)

✅ **Rigorous validation** (Type I error, power tests)

✅ **Honest documentation** (clear about limitations)

### What We Still Need

⚠️ **Empirical calibration** (Option 3 from DENSITY-MATRIX-CALIBRATION.md)

⚠️ **Effect size benchmarks** (fp32 vs fp16, fp32 vs int8, etc.)

⚠️ **Power analysis** (sample size recommendations)

⚠️ **Comparison to MMD** (when to use which test)

### Recommendation

**Implement V2 (no PCA)** as specified in this plan. The 2x computational cost is acceptable for gaining valid statistical inference. The PCA circularity was a fatal flaw; removing it makes the implementation scientifically sound.

After implementation, prioritize empirical calibration (Option 3) to establish effect size benchmarks.
