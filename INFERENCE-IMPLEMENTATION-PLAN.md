# Implementation Plan: Statistical Inference for Density Matrix Metrics

## Overview

This plan details implementation of permutation testing (Option 1) and bootstrap confidence intervals (Option 2) for density matrix metrics. Both will be implemented to provide comprehensive statistical inference.

**Estimated time**: 2-3 hours for core implementation, 1 hour for testing, 30 minutes for documentation.

---

## Goals

1. **Permutation testing**: Provide p-values for trace distance comparisons
2. **Bootstrap CI**: Quantify uncertainty in trace distance estimates
3. **Integration**: Make both work seamlessly with existing workflow
4. **Documentation**: Clear usage examples and interpretation guidance

---

## File Structure

### New File: `model_equality_testing/src/quantum_inference.py`

Create a new module (rather than modifying `quantum_metrics.py`) to keep separation of concerns:
- `quantum_metrics.py`: Core metric computation (descriptive statistics)
- `quantum_inference.py`: Statistical inference (hypothesis tests, confidence intervals)

**Why separate?**
- Clear conceptual distinction (description vs. inference)
- Keeps `quantum_metrics.py` dependency-free (no scipy.stats needed there)
- Easier to maintain and test independently
- Follows pattern: `tests.py` (statistics) + `pvalue.py` (inference)

---

## Implementation

### Phase 1: Permutation Testing

#### Function 1: `trace_distance_permutation_test()`

**Location**: `model_equality_testing/src/quantum_inference.py`

**Signature**:
```python
def trace_distance_permutation_test(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    b: int = 1000,
    k: int = 50,
    pca: Optional[PCA] = None,
    random_seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Two-sample permutation test for trace distance.
    
    Tests H0: distributions A and B are identical
    vs H1: distributions differ.
    
    Args:
        emb_a: (n_a, d) embeddings from distribution A
        emb_b: (n_b, d) embeddings from distribution B
        b: number of permutations (default 1000)
        k: number of PCA components (default 50)
        pca: optional pre-fitted PCA model. If None, fits on combined data.
        random_seed: optional random seed for reproducibility
    
    Returns:
        pvalue: float - permutation p-value
        statistic: float - observed trace distance
    
    Raises:
        ValueError: if n_a != n_b (trace distance requires equal sizes)
    
    Example:
        >>> pvalue, td = trace_distance_permutation_test(emb_a, emb_b, b=1000)
        >>> print(f"Trace distance: {td:.4f}, p-value: {pvalue:.4f}")
        >>> if pvalue < 0.05:
        >>>     print("Significantly different")
    """
```

**Implementation details**:

```python
import numpy as np
from typing import Tuple, Optional
from sklearn.decomposition import PCA
from model_equality_testing.src.quantum_metrics import (
    fit_pca,
    pca_density_matrix,
    trace_distance
)

def trace_distance_permutation_test(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    b: int = 1000,
    k: int = 50,
    pca: Optional[PCA] = None,
    random_seed: Optional[int] = None
) -> Tuple[float, float]:
    
    n_a = len(emb_a)
    n_b = len(emb_b)
    
    # Validate equal sizes
    if n_a != n_b:
        raise ValueError(
            f"Trace distance requires equal sample sizes. Got n_a={n_a}, n_b={n_b}. "
            "Consider using semantic axes for unequal-size comparisons."
        )
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Combine samples
    combined = np.vstack([emb_a, emb_b])
    
    # Fit PCA if not provided
    if pca is None:
        pca = fit_pca(combined, k=k)
    
    # Observed statistic
    rho_a = pca_density_matrix(emb_a, pca)
    rho_b = pca_density_matrix(emb_b, pca)
    td_observed = trace_distance(rho_a, rho_b)
    
    # Permutation null distribution
    td_null = []
    for i in range(b):
        # Randomly permute combined samples
        perm = np.random.permutation(n_a + n_b)
        emb_a_perm = combined[perm[:n_a]]
        emb_b_perm = combined[perm[n_a:]]
        
        # Compute trace distance under permutation
        rho_a_perm = pca_density_matrix(emb_a_perm, pca)
        rho_b_perm = pca_density_matrix(emb_b_perm, pca)
        td_null.append(trace_distance(rho_a_perm, rho_b_perm))
    
    # P-value: (# null >= observed + 1) / (b + 1)
    # +1 in numerator and denominator ensures proper probability
    # and avoids p=0 (conservative)
    pvalue = (np.sum(np.array(td_null) >= td_observed) + 1) / (b + 1)
    
    return pvalue, td_observed
```

**Key design choices**:

1. **Equal size requirement**: Explicit error message with suggestion to use semantic axes
2. **PCA parameter**: Allow pre-fitted PCA for consistency with other analyses
3. **Random seed**: Support reproducibility for testing and debugging
4. **Conservative p-value**: Use (count + 1) / (b + 1) formula (standard practice)
5. **Same PCA for all permutations**: Reuse fitted PCA (don't refit for each permutation)

---

#### Function 2: `trace_distance_permutation_test_from_samples()`

**Convenience wrapper** for CompletionSample objects:

```python
from model_equality_testing.distribution import CompletionSample
from model_equality_testing.src.embeddings import embed_sample

def trace_distance_permutation_test_from_samples(
    sample_a: CompletionSample,
    sample_b: CompletionSample,
    b: int = 1000,
    k: int = 50,
    _precomputed_embeddings: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    random_seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Permutation test from CompletionSample objects.
    
    Args:
        sample_a: CompletionSample from distribution A
        sample_b: CompletionSample from distribution B
        b: number of permutations
        k: PCA components
        _precomputed_embeddings: optional (emb_a, emb_b) tuple to skip embedding
        random_seed: optional random seed
    
    Returns:
        pvalue, statistic
    """
    # Use precomputed embeddings if provided, otherwise embed
    if _precomputed_embeddings is not None:
        emb_a, emb_b = _precomputed_embeddings
    else:
        emb_a = embed_sample(sample_a)
        emb_b = embed_sample(sample_b)
    
    return trace_distance_permutation_test(
        emb_a, emb_b, b=b, k=k, random_seed=random_seed
    )
```

**Why this wrapper?**
- Follows the `_precomputed_embeddings` pattern from `tests.py`
- Allows reusing embeddings computed for quantum metrics or semantic axes
- Consistent API with rest of package

---

### Phase 2: Bootstrap Confidence Intervals

#### Function 3: `trace_distance_bootstrap_ci()`

**Location**: `model_equality_testing/src/quantum_inference.py`

**Signature**:
```python
def trace_distance_bootstrap_ci(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    k: int = 50,
    pca: Optional[PCA] = None,
    random_seed: Optional[int] = None
) -> Tuple[float, Tuple[float, float]]:
    """
    Bootstrap confidence interval for trace distance.
    
    Uses bootstrap resampling to estimate sampling distribution
    of trace distance and construct confidence interval.
    
    Args:
        emb_a: (n_a, d) embeddings from distribution A
        emb_b: (n_b, d) embeddings from distribution B
        n_boot: number of bootstrap samples (default 1000)
        alpha: significance level for CI (default 0.05 → 95% CI)
        k: PCA components (default 50)
        pca: optional pre-fitted PCA model
        random_seed: optional random seed
    
    Returns:
        td_observed: float - point estimate of trace distance
        ci: tuple (lower, upper) - percentile confidence interval
    
    Example:
        >>> td, (ci_lower, ci_upper) = trace_distance_bootstrap_ci(emb_a, emb_b)
        >>> print(f"Trace distance: {td:.4f}")
        >>> print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        >>> if ci_lower > 0:
        >>>     print("Significantly different from zero")
    """
```

**Implementation**:

```python
def trace_distance_bootstrap_ci(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    k: int = 50,
    pca: Optional[PCA] = None,
    random_seed: Optional[int] = None
) -> Tuple[float, Tuple[float, float]]:
    
    n_a = len(emb_a)
    n_b = len(emb_b)
    
    # Validate equal sizes (for trace distance)
    if n_a != n_b:
        raise ValueError(
            f"Trace distance requires equal sample sizes. Got n_a={n_a}, n_b={n_b}. "
            "Consider using semantic axes for unequal-size comparisons."
        )
    
    # Set random seed
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Fit PCA on original combined data
    combined = np.vstack([emb_a, emb_b])
    if pca is None:
        pca = fit_pca(combined, k=k)
    
    # Observed statistic
    rho_a = pca_density_matrix(emb_a, pca)
    rho_b = pca_density_matrix(emb_b, pca)
    td_obs = trace_distance(rho_a, rho_b)
    
    # Bootstrap distribution
    td_boot = []
    for i in range(n_boot):
        # Resample WITH replacement from each distribution
        idx_a = np.random.choice(n_a, size=n_a, replace=True)
        idx_b = np.random.choice(n_b, size=n_b, replace=True)
        
        emb_a_boot = emb_a[idx_a]
        emb_b_boot = emb_b[idx_b]
        
        # Compute trace distance on bootstrap sample
        rho_a_boot = pca_density_matrix(emb_a_boot, pca)
        rho_b_boot = pca_density_matrix(emb_b_boot, pca)
        td_boot.append(trace_distance(rho_a_boot, rho_b_boot))
    
    # Percentile confidence interval
    ci_lower = np.percentile(td_boot, 100 * alpha / 2)
    ci_upper = np.percentile(td_boot, 100 * (1 - alpha / 2))
    
    return td_obs, (ci_lower, ci_upper)
```

**Key design choices**:

1. **Percentile method**: Simple, widely understood, no normality assumption
2. **Resampling strategy**: With replacement from each distribution separately
3. **Fixed PCA**: Use same PCA fit for all bootstrap samples (consistency)
4. **Alpha parameter**: Flexible significance level (default 0.05 for 95% CI)

---

#### Function 4: `trace_distance_bootstrap_ci_from_samples()`

**Convenience wrapper**:

```python
def trace_distance_bootstrap_ci_from_samples(
    sample_a: CompletionSample,
    sample_b: CompletionSample,
    n_boot: int = 1000,
    alpha: float = 0.05,
    k: int = 50,
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
        emb_a, emb_b, n_boot=n_boot, alpha=alpha, k=k, random_seed=random_seed
    )
```

---

### Phase 3: Combined Function (Best of Both)

#### Function 5: `trace_distance_test()`

**High-level function** that computes both permutation p-value and bootstrap CI:

```python
def trace_distance_test(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    b: int = 1000,
    n_boot: int = 1000,
    alpha: float = 0.05,
    k: int = 50,
    random_seed: Optional[int] = None
) -> dict:
    """
    Complete statistical inference for trace distance.
    
    Computes both permutation test and bootstrap CI.
    
    Args:
        emb_a, emb_b: embeddings to compare
        b: permutations for p-value
        n_boot: bootstrap samples for CI
        alpha: significance level
        k: PCA components
        random_seed: random seed
    
    Returns:
        dict with keys:
            'statistic': observed trace distance
            'pvalue': permutation p-value
            'ci_lower': lower CI bound
            'ci_upper': upper CI bound
            'ci_width': CI width (precision measure)
            'significant': bool (p < alpha)
    
    Example:
        >>> result = trace_distance_test(emb_a, emb_b)
        >>> print(f"Trace distance: {result['statistic']:.4f}")
        >>> print(f"P-value: {result['pvalue']:.4f}")
        >>> print(f"95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
        >>> if result['significant']:
        >>>     print("Significantly different")
    """
    # Fit PCA once, reuse for both
    combined = np.vstack([emb_a, emb_b])
    pca = fit_pca(combined, k=k)
    
    # Permutation test
    pvalue, statistic = trace_distance_permutation_test(
        emb_a, emb_b, b=b, k=k, pca=pca, random_seed=random_seed
    )
    
    # Bootstrap CI (use different seed to avoid correlation)
    boot_seed = None if random_seed is None else random_seed + 1
    td_point, (ci_lower, ci_upper) = trace_distance_bootstrap_ci(
        emb_a, emb_b, n_boot=n_boot, alpha=alpha, k=k, pca=pca, random_seed=boot_seed
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

---

### Phase 4: Pretty Printing / Summary

#### Function 6: `print_trace_distance_test_result()`

**Human-readable output**:

```python
def print_trace_distance_test_result(result: dict, label_a: str = "A", label_b: str = "B"):
    """
    Print formatted test results.
    
    Args:
        result: dict from trace_distance_test()
        label_a, label_b: names for distributions
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
        print(f"  ✓ Statistically significant (p = {pval:.4f} < 0.05)")
    else:
        print(f"  ✗ Not significant (p = {pval:.4f} ≥ 0.05)")
    
    # Effect magnitude
    if td < 0.1:
        mag = "small"
    elif td < 0.3:
        mag = "moderate"
    else:
        mag = "large"
    print(f"  Effect magnitude: {mag} (trace distance = {td:.4f})")
    
    # Precision
    if ci_width < 0.05:
        prec = "precisely estimated (narrow CI)"
    elif ci_width < 0.15:
        prec = "moderate precision"
    else:
        prec = "substantial uncertainty (wide CI)"
    print(f"  Precision: {prec} (CI width = {ci_width:.4f})")
    
    # CI interpretation
    if ci_low > 0:
        print(f"  95% CI excludes zero → difference is significant")
    
    print()
    
    # Conclusion
    if result['significant']:
        print(f"Conclusion: Distributions {label_a} and {label_b} are significantly different")
        print(f"           with a {mag} effect size.")
    else:
        print(f"Conclusion: No significant difference detected between {label_a} and {label_b}.")
```

---

## Testing Strategy

### Unit Tests

**File**: `model_equality_testing/tests/test_quantum_inference.py`

**Test cases**:

1. **Test permutation test with synthetic data**
   - Same distribution → high p-value
   - Different distributions → low p-value
   - Reproducibility with random seed

2. **Test bootstrap CI with synthetic data**
   - CI includes true value
   - CI excludes zero for different distributions
   - Reproducibility with random seed

3. **Test equal size validation**
   - Should raise ValueError for n_a ≠ n_b

4. **Test edge cases**
   - Very small sample sizes (n=10)
   - Identical samples (td = 0)
   - Maximum difference samples

5. **Test integration**
   - Combined test returns consistent results
   - PCA reuse works correctly

**Example test**:

```python
import numpy as np
import pytest
from model_equality_testing.src.quantum_inference import (
    trace_distance_permutation_test,
    trace_distance_bootstrap_ci,
    trace_distance_test
)

def test_permutation_test_null_hypothesis():
    """Test permutation test under null (same distribution)."""
    np.random.seed(42)
    
    # Generate samples from same distribution
    emb = np.random.randn(200, 768)
    emb_a = emb[:100]
    emb_b = emb[100:]
    
    # Test
    pvalue, td = trace_distance_permutation_test(
        emb_a, emb_b, b=100, random_seed=42
    )
    
    # Under null, should have high p-value (not always, but on average)
    # Check td is small
    assert td < 0.2, "Trace distance should be small for same distribution"
    assert 0 <= pvalue <= 1, "P-value must be in [0, 1]"

def test_permutation_test_alternative():
    """Test permutation test under alternative (different distributions)."""
    np.random.seed(42)
    
    # Generate samples from different distributions
    emb_a = np.random.randn(100, 768)
    emb_b = np.random.randn(100, 768) + 1.0  # Shifted mean
    
    pvalue, td = trace_distance_permutation_test(
        emb_a, emb_b, b=100, random_seed=42
    )
    
    # Should detect difference
    assert td > 0.1, "Should detect shift"
    # p-value might still be high with only b=100, but td should be nonzero

def test_bootstrap_ci_coverage():
    """Test bootstrap CI excludes zero for different distributions."""
    np.random.seed(42)
    
    # Different distributions
    emb_a = np.random.randn(100, 768)
    emb_b = np.random.randn(100, 768) + 1.0
    
    td, (ci_low, ci_up) = trace_distance_bootstrap_ci(
        emb_a, emb_b, n_boot=100, random_seed=42
    )
    
    assert ci_low <= td <= ci_up, "Point estimate should be in CI"
    assert ci_low >= 0, "CI lower bound should be non-negative"
    assert ci_up <= 1, "CI upper bound should be ≤ 1"
    # For shifted distributions, CI should exclude zero
    # (not guaranteed with small n_boot, but likely)

def test_unequal_sizes_raises_error():
    """Test that unequal sample sizes raise error."""
    emb_a = np.random.randn(100, 768)
    emb_b = np.random.randn(150, 768)  # Different size
    
    with pytest.raises(ValueError, match="equal sample sizes"):
        trace_distance_permutation_test(emb_a, emb_b)
    
    with pytest.raises(ValueError, match="equal sample sizes"):
        trace_distance_bootstrap_ci(emb_a, emb_b)

def test_combined_function():
    """Test combined inference function."""
    np.random.seed(42)
    
    emb_a = np.random.randn(100, 768)
    emb_b = np.random.randn(100, 768) + 0.5
    
    result = trace_distance_test(emb_a, emb_b, b=50, n_boot=50, random_seed=42)
    
    # Check all keys present
    required_keys = ['statistic', 'pvalue', 'ci_lower', 'ci_upper', 'ci_width', 'significant']
    for key in required_keys:
        assert key in result
    
    # Check consistency
    assert result['ci_width'] == result['ci_upper'] - result['ci_lower']
    assert result['significant'] == (result['pvalue'] < 0.05)
```

---

### Integration Tests

**Test with real data** from the dataset:

```python
def test_with_real_data():
    """Integration test with fp32 vs int8 Llama-3-8B."""
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
    sample_fp32 = dist_fp32.draw_completion_sample(100)
    sample_int8 = dist_int8.draw_completion_sample(100)
    
    # Embed
    emb_fp32 = embed_sample(sample_fp32)
    emb_int8 = embed_sample(sample_int8)
    
    # Test (use small b for speed)
    result = trace_distance_test(emb_fp32, emb_int8, b=100, n_boot=100)
    
    # fp32 vs int8 should be different (this is a known difference)
    # Can't assert significance (depends on random sample), but can check structure
    assert 0 <= result['statistic'] <= 1
    assert 0 <= result['pvalue'] <= 1
    assert result['ci_lower'] <= result['statistic'] <= result['ci_upper']
    
    print(f"fp32 vs int8: td={result['statistic']:.4f}, p={result['pvalue']:.4f}")
```

---

## Usage Examples

### Example 1: Basic Permutation Test

```python
from model_equality_testing.src.embeddings import embed_sample
from model_equality_testing.src.quantum_inference import trace_distance_permutation_test

# Embed samples
emb_a = embed_sample(sample_a)
emb_b = embed_sample(sample_b)

# Test
pvalue, td = trace_distance_permutation_test(emb_a, emb_b, b=1000)

print(f"Trace distance: {td:.4f}")
print(f"P-value: {pvalue:.4f}")

if pvalue < 0.05:
    print("Distributions are significantly different")
else:
    print("No significant difference detected")
```

---

### Example 2: Bootstrap Confidence Interval

```python
from model_equality_testing.src.quantum_inference import trace_distance_bootstrap_ci

td, (ci_lower, ci_upper) = trace_distance_bootstrap_ci(emb_a, emb_b, n_boot=1000)

print(f"Trace distance: {td:.4f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

if ci_lower > 0:
    print("Significantly different from zero (CI excludes 0)")
```

---

### Example 3: Complete Analysis (Recommended)

```python
from model_equality_testing.dataset import load_distribution
from model_equality_testing.src.embeddings import embed_sample
from model_equality_testing.src.quantum_metrics import (
    fit_pca, pca_density_matrix, von_neumann_entropy, quantum_relative_entropy
)
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

# 1. Complete inference (permutation + bootstrap)
result = trace_distance_test(emb_a, emb_b, b=1000, n_boot=1000)
print_trace_distance_test_result(result, label_a="fp32", label_b="int8")

# 2. Additional quantum metrics (for context)
pca = fit_pca(np.vstack([emb_a, emb_b]), k=50)
rho_a = pca_density_matrix(emb_a, pca)
rho_b = pca_density_matrix(emb_b, pca)

s_a = von_neumann_entropy(rho_a)
s_b = von_neumann_entropy(rho_b)
kl = quantum_relative_entropy(rho_a, rho_b)

print(f"\nAdditional metrics:")
print(f"  Von Neumann entropy: A={s_a:.3f}, B={s_b:.3f}, Δ={s_b-s_a:.3f}")
print(f"  Quantum relative entropy: KL(A||B)={kl:.4f}")

# 3. Semantic interpretation (optional)
from model_equality_testing.src.semantic_axes import interpret_difference
axes = [prof_axis, tech_axis, form_axis]  # Pre-loaded axes
interp = interpret_difference(emb_a, emb_b, axes, label_a="fp32", label_b="int8")
print(f"\nSemantic interpretation:")
print(interp.summary())
```

---

## Documentation Updates

### Update `README.md`

Add new section after quantum metrics example:

```markdown
### Statistical Inference for Quantum Metrics

The quantum metrics now support statistical significance testing:

```python
from model_equality_testing.src.quantum_inference import trace_distance_test

# Complete statistical inference
result = trace_distance_test(emb_a, emb_b, b=1000, n_boot=1000)

print(f"Trace distance: {result['statistic']:.4f}")
print(f"P-value: {result['pvalue']:.4f}")
print(f"95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")

if result['significant']:
    print("Distributions are significantly different")
```

For details, see `DENSITY-MATRIX-CALIBRATION.md`.
```

---

### Update `DENSITY-MATRIX-METRICS.md`

Add section at end:

```markdown
## Statistical Inference

The metrics shown above are descriptive statistics. For hypothesis testing and confidence intervals, see:

- **Permutation testing**: `trace_distance_permutation_test()` - tests if distributions differ
- **Bootstrap CI**: `trace_distance_bootstrap_ci()` - quantifies uncertainty in effect size
- **Combined**: `trace_distance_test()` - computes both

See `DENSITY-MATRIX-CALIBRATION.md` for implementation details and usage examples.
```

---

### Create Usage Guide

**File**: `USING-QUANTUM-INFERENCE.md`

Brief guide with:
- When to use permutation vs bootstrap vs both
- Interpretation of results
- Example workflows
- Computational considerations
- Integration with semantic axes

---

## Timeline

### Hour 1: Core Implementation
- ✓ Create `quantum_inference.py`
- ✓ Implement `trace_distance_permutation_test()`
- ✓ Implement `trace_distance_bootstrap_ci()`
- ✓ Implement wrapper functions

### Hour 2: Combined Functions and Pretty Printing
- ✓ Implement `trace_distance_test()`
- ✓ Implement `print_trace_distance_test_result()`
- ✓ Test manually with synthetic data

### Hour 3: Testing
- ✓ Write unit tests
- ✓ Write integration test with real data
- ✓ Fix bugs found during testing

### Hour 4: Documentation
- ✓ Add docstrings to all functions
- ✓ Update README.md
- ✓ Update DENSITY-MATRIX-METRICS.md
- ✓ Create USING-QUANTUM-INFERENCE.md

**Total: ~4 hours for complete, tested, documented implementation**

---

## Dependencies

**New imports needed**:
```python
# Already in use by the package:
import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple, Optional

# No new external dependencies required
```

All dependencies already present in the project.

---

## Checklist

- [ ] Create `model_equality_testing/src/quantum_inference.py`
- [ ] Implement `trace_distance_permutation_test()`
- [ ] Implement `trace_distance_permutation_test_from_samples()`
- [ ] Implement `trace_distance_bootstrap_ci()`
- [ ] Implement `trace_distance_bootstrap_ci_from_samples()`
- [ ] Implement `trace_distance_test()` (combined)
- [ ] Implement `print_trace_distance_test_result()`
- [ ] Create `tests/test_quantum_inference.py`
- [ ] Write unit tests (null hypothesis, alternative, edge cases)
- [ ] Write integration test with real data
- [ ] Update `README.md`
- [ ] Update `DENSITY-MATRIX-METRICS.md`
- [ ] Create `USING-QUANTUM-INFERENCE.md`
- [ ] Test all examples in documentation
- [ ] Commit with clear message

---

## Verification

After implementation, verify:

1. **Permutation test works**:
   - High p-value for same distribution
   - Low p-value for different distributions
   - Reproducible with random seed

2. **Bootstrap CI works**:
   - CI includes point estimate
   - CI excludes zero for different distributions
   - Width reflects sample size

3. **Combined function consistent**:
   - td from permutation = td from bootstrap = td from combined
   - p-value and significance flag consistent
   - CI properties maintained

4. **Integration with existing code**:
   - Can reuse embeddings from `embed_sample()`
   - PCA compatibility with `quantum_metrics.py`
   - Works with `CompletionSample` objects

5. **Documentation complete**:
   - All functions have docstrings
   - Examples run without errors
   - Clear interpretation guidance

---

## Future Enhancements (Not in This Plan)

**Could add later** (not required for initial implementation):

1. **Other metrics**: Permutation/bootstrap for von Neumann entropy, quantum relative entropy
2. **Other CI methods**: BCa (bias-corrected accelerated) bootstrap
3. **Parallel computation**: Use multiprocessing for permutations/bootstrap
4. **Integration with `algorithm.py`**: Make trace distance work with `run_two_sample_test()`
5. **Empirical calibration**: Build reference databases (Option 3 from calibration doc)

These can be tackled incrementally after the core implementation is working and tested.
