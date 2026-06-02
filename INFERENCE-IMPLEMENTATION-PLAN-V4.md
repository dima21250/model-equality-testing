# Implementation Plan V4: Spectral Metrics with Equal Sample Sizes

## Overview

This plan implements **eigenvalue-based spectral metrics** for quantifying distributional differences. Based on the V3 critique, this version:

1. **Requires equal sample sizes** (n_a = n_b) - eliminates sample-size confounds
2. **Fixes computational cost estimates** (realistic: 60-90 seconds)
3. **Honest about limitations** (what "modes" really means)
4. **Includes baseline comparisons** (simple spread metrics)
5. **Rigorous validation** (Type I error tests)

**Goal**: Provide holistic spectral quantification as a complement to per-dimension semantic axes.

**Estimated time**: 4 hours (including proper validation and baselines)

---

## Key Changes from V3

### Addressed Critical Flaws

| Issue (from critique) | V3 | V4 |
|----------------------|----|----|
| Sample-size dependence | ⚠️ Claimed to work for n_a ≠ n_b | ✅ Require n_a = n_b |
| Computational cost | ❌ "40-50s" (wrong) | ✅ "60-90s" (realistic) |
| "Ill-defined" claim | ❌ "Fixed" | ✅ "Sidestepped" (honest) |
| Effective rank interpretation | ⚠️ Oversold | ✅ Conservative framing |
| Type I error validation | ⚠️ Only equal-n case | ✅ Explicit equal-n validation |
| Baseline comparison | ❌ Missing | ✅ Include spread metrics |
| Entropy normalization | ❌ Not addressed | ✅ Normalize by log(n) |

### Design Philosophy

**V3 tried to do too much** (work for any n, claim "no issues", overstate speed).

**V4 is honest and focused**:
- Accept equal-n constraint (same as trace distance, not a new limitation)
- Realistic about what we're measuring (eigenvalue decay patterns, not semantic modes)
- Compare to simple baselines (prove eigenvalues add value)
- Rigorous validation (Type I error, power)

---

## Why Equal Sample Sizes Simplifies Everything

### With n_a = n_b = 100:

**Entropy comparison**:
- Both have max entropy = log(100) ≈ 4.6
- Ranges identical
- Differences directly comparable

**Eigenvalue distributions**:
- Both have at most 100 eigenvalues
- Similar lengths (both ~60-80 positive eigenvalues typically)
- Wasserstein comparison is apple-to-apple

**Permutation test**:
- Under null, expected entropy difference = 0 (no finite-sample bias difference)
- Clean null distribution
- Valid Type I error control

**Effective rank ratio**:
- Both measured with same n
- Ratio meaningful: "B has 2.5x more effective modes than A (both n=100)"

---

## Motivation: What Problem Are We Solving?

### You already have:

| Tool | Provides |
|------|----------|
| **Semantic axes** | Per-dimension quantification ("prof: +0.54, g=0.82") |
| **MMD** | Holistic test ("MMD = 0.23, p < 0.001") but unclear scale |

### Gap:

**Holistic spectral characterization with interpretable magnitudes**:
- Not just "significant difference" (T/F)
- Not just per-dimension (semantic axes)
- But: "Overall, how different is the spectral structure?"

### Solution:

Two complementary eigenvalue-based metrics:

1. **Von Neumann entropy difference**: Quantifies change in diversity/spread
2. **Wasserstein on eigenvalues**: Quantifies spectral distance

Plus baseline comparison to simple spread metrics.

---

## File Structure

**New file**: `model_equality_testing/src/spectral_metrics.py`

**Dependencies**: numpy, scipy.stats (already in project)

**No changes to**: quantum_metrics.py, semantic_axes.py, embeddings.py

---

## Implementation

### Phase 1: Core Functions (1 hour)

#### Function 1: `eigenvalue_spectrum()`

```python
import numpy as np
from typing import Optional

def eigenvalue_spectrum(embeddings: np.ndarray) -> np.ndarray:
    """
    Extract normalized eigenvalue distribution from embeddings.
    
    Computes eigenvalues of the Gram matrix G = E @ E^T and normalizes
    to a probability distribution (sum to 1).
    
    Args:
        embeddings: (n, d) SBERT embeddings
    
    Returns:
        evals: (k,) positive eigenvalues, sorted descending, normalized
               where k ≤ min(n, d) is the number of positive eigenvalues
    
    Note:
        The eigenvalues represent the spectral decomposition of the
        embedding similarity structure. NOT semantic clusters or topics.
        
        Typical result: ~60-80% of eigenvalues are positive for n=100, d=768.
    
    Example:
        >>> evals = eigenvalue_spectrum(emb)
        >>> print(f"Positive eigenvalues: {len(evals)}/{len(emb)}")
        >>> print(f"Top 3: {evals[:3]}")  # e.g., [0.15, 0.09, 0.06]
    """
    # Gram matrix
    G = embeddings @ embeddings.T  # (n, n)
    
    # Eigenvalues (sorted ascending by eigvalsh)
    evals = np.linalg.eigvalsh(G)
    
    # Keep positive only (remove numerical zeros)
    evals = evals[evals > 1e-10]
    
    # Sort descending (largest first)
    evals = evals[::-1]
    
    # Normalize to probability distribution
    evals = evals / evals.sum()
    
    return evals
```

---

#### Function 2: `von_neumann_entropy_from_embeddings()`

```python
def von_neumann_entropy_from_embeddings(embeddings: np.ndarray) -> float:
    """
    Compute Von Neumann entropy from embeddings.
    
    S(ρ) = -Σ λᵢ log λᵢ
    
    Measures the concentration vs spread of the eigenvalue distribution.
    
    Args:
        embeddings: (n, d) SBERT embeddings
    
    Returns:
        entropy: float in [0, log(n)]
                 0 = one dominant eigenvalue (concentrated)
                 log(n) = uniform eigenvalue distribution (maximally spread)
    
    Interpretation:
        Low entropy: Embeddings concentrated in few dimensions
        High entropy: Embeddings spread across many dimensions
        
        NOT the same as semantic diversity (number of topics/clusters).
    
    Example:
        >>> s = von_neumann_entropy_from_embeddings(emb)
        >>> max_s = np.log(len(emb))
        >>> print(f"Entropy: {s:.2f} / {max_s:.2f} ({s/max_s*100:.0f}%)")
    """
    evals = eigenvalue_spectrum(embeddings)
    
    # S = -Σ λᵢ log λᵢ
    # Use mask to avoid log(0); eigenvalues already positive from spectrum
    entropy = -np.sum(evals * np.log(evals))
    
    return entropy
```

---

#### Function 3: `normalized_entropy()`

```python
def normalized_entropy(embeddings: np.ndarray) -> float:
    """
    Von Neumann entropy normalized by maximum possible entropy.
    
    S_normalized = S / log(n)
    
    Returns:
        norm_entropy: float in [0, 1]
                      0 = completely concentrated
                      1 = maximally spread
    
    This normalization makes entropies comparable across different sample sizes
    (though we require equal n in this module).
    """
    s = von_neumann_entropy_from_embeddings(embeddings)
    n = len(embeddings)
    return s / np.log(n)
```

---

#### Function 4: `effective_rank()`

```python
def effective_rank(embeddings: np.ndarray) -> float:
    """
    Effective dimensionality: exp(S).
    
    Interpretation: Perplexity of the eigenvalue distribution.
    If eigenvalues were uniform over k dimensions, eff_rank ≈ k.
    
    Args:
        embeddings: (n, d) SBERT embeddings
    
    Returns:
        eff_rank: float in [1, n]
                  1 = one dominant eigenvalue
                  n = uniform over all n eigenvalues
    
    WARNING:
        This is NOT the number of semantic clusters or topics!
        It's a measure of eigenvalue concentration.
        
        For typical random data (n=100, d=768), eff_rank ≈ 60-80.
    
    Example:
        >>> r = effective_rank(emb)
        >>> print(f"Effective rank: {r:.1f} / {len(emb)}")
    """
    s = von_neumann_entropy_from_embeddings(embeddings)
    return np.exp(s)
```

---

#### Function 5: `spectral_wasserstein()`

```python
from scipy.stats import wasserstein_distance

def spectral_wasserstein(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """
    Wasserstein distance on eigenvalue spectra.
    
    Measures the Earth Mover's Distance between two eigenvalue distributions.
    
    Args:
        emb_a: (n, d) embeddings from distribution A
        emb_b: (n, d) embeddings from distribution B
    
    Returns:
        distance: float ≥ 0
                  0 = identical eigenvalue distributions
                  larger = more different spectral structure
    
    Note:
        For equal sample sizes (n), the eigenvalue distributions have
        similar lengths, making the comparison well-defined.
    
    Example:
        >>> w = spectral_wasserstein(emb_a, emb_b)
        >>> print(f"Spectral distance: {w:.4f}")
    """
    evals_a = eigenvalue_spectrum(emb_a)
    evals_b = eigenvalue_spectrum(emb_b)
    
    return wasserstein_distance(evals_a, evals_b)
```

---

### Phase 1b: Baseline Metrics (30 minutes)

#### Function 6: `embedding_spread()`

**Simple baseline for comparison**:

```python
def embedding_spread(embeddings: np.ndarray) -> float:
    """
    Average distance from centroid (normalized).
    
    Simple baseline metric for diversity/spread.
    
    Args:
        embeddings: (n, d) embeddings
    
    Returns:
        spread: float - average L2 distance from mean, normalized by sqrt(d)
    
    Interpretation:
        Higher spread = more dispersed embeddings
        
    This is simpler than Von Neumann entropy (no eigendecomposition)
    but captures similar information (concentration vs spread).
    """
    mu = embeddings.mean(axis=0)
    distances = np.linalg.norm(embeddings - mu, axis=1)
    spread = distances.mean() / np.sqrt(embeddings.shape[1])
    return spread


def spread_ratio(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """
    Ratio of embedding spreads.
    
    Returns:
        ratio: float - spread_b / spread_a
               > 1: B is more spread out
               < 1: A is more spread out
    """
    return embedding_spread(emb_b) / embedding_spread(emb_a)
```

---

### Phase 2: Statistical Inference (1.5 hours)

#### Function 7: `entropy_difference_test()`

```python
from typing import Tuple
from numpy.random import default_rng

def entropy_difference_test(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    b: int = 1000,
    random_seed: Optional[int] = None
) -> Tuple[float, float, float, float]:
    """
    Permutation test for Von Neumann entropy difference.
    
    Tests H0: S(ρ_A) = S(ρ_B)
    vs H1: S(ρ_A) ≠ S(ρ_B)
    
    Args:
        emb_a: (n, d) embeddings from distribution A
        emb_b: (n, d) embeddings from distribution B (MUST have n_b = n_a)
        b: number of permutations (default 1000)
        random_seed: optional random seed
    
    Returns:
        pvalue: permutation p-value
        s_a: entropy of A
        s_b: entropy of B
        s_a_norm: normalized entropy of A (in [0,1])
        s_b_norm: normalized entropy of B (in [0,1])
    
    Raises:
        ValueError: if len(emb_a) != len(emb_b)
    
    Computational cost:
        ~60-90 seconds for b=1000, n=100, d=768
    
    Example:
        >>> pval, s_a, s_b, _, _ = entropy_difference_test(emb_a, emb_b)
        >>> if pval < 0.05:
        >>>     print(f"Entropy differs: {s_a:.2f} vs {s_b:.2f}")
    """
    n_a = len(emb_a)
    n_b = len(emb_b)
    
    # Require equal sample sizes
    if n_a != n_b:
        raise ValueError(
            f"Equal sample sizes required for valid comparison. "
            f"Got n_a={n_a}, n_b={n_b}.\n"
            f"Consider: (1) subsampling to min(n_a, n_b), or "
            f"(2) using semantic axes (no equal-n requirement)."
        )
    
    rng = default_rng(random_seed)
    
    # Observed entropies
    s_a = von_neumann_entropy_from_embeddings(emb_a)
    s_b = von_neumann_entropy_from_embeddings(emb_b)
    delta_obs = abs(s_b - s_a)
    
    # Normalized versions (for interpretation)
    s_a_norm = normalized_entropy(emb_a)
    s_b_norm = normalized_entropy(emb_b)
    
    # Permutation null distribution
    combined = np.vstack([emb_a, emb_b])
    
    deltas_null = []
    for _ in range(b):
        # Randomly permute (equal sizes maintained)
        perm = rng.permutation(len(combined))
        a_perm = combined[perm[:n_a]]
        b_perm = combined[perm[n_a:]]
        
        s_a_perm = von_neumann_entropy_from_embeddings(a_perm)
        s_b_perm = von_neumann_entropy_from_embeddings(b_perm)
        deltas_null.append(abs(s_b_perm - s_a_perm))
    
    # P-value
    pvalue = (np.sum(np.array(deltas_null) >= delta_obs) + 1) / (b + 1)
    
    return pvalue, s_a, s_b, s_a_norm, s_b_norm
```

---

#### Function 8: `spectral_wasserstein_test()`

```python
def spectral_wasserstein_test(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    b: int = 1000,
    random_seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Permutation test for Wasserstein distance on eigenvalue spectra.
    
    Tests H0: eigenvalue distributions identical
    vs H1: eigenvalue distributions differ
    
    Args:
        emb_a, emb_b: (n, d) embeddings (MUST have equal n)
        b: number of permutations
        random_seed: optional seed
    
    Returns:
        pvalue: permutation p-value
        w_obs: observed Wasserstein distance
    
    Raises:
        ValueError: if len(emb_a) != len(emb_b)
    """
    n_a = len(emb_a)
    n_b = len(emb_b)
    
    if n_a != n_b:
        raise ValueError(
            f"Equal sample sizes required. Got n_a={n_a}, n_b={n_b}."
        )
    
    rng = default_rng(random_seed)
    
    # Observed
    w_obs = spectral_wasserstein(emb_a, emb_b)
    
    # Permutation null
    combined = np.vstack([emb_a, emb_b])
    
    w_null = []
    for _ in range(b):
        perm = rng.permutation(len(combined))
        a_perm = combined[perm[:n_a]]
        b_perm = combined[perm[n_a:]]
        w_null.append(spectral_wasserstein(a_perm, b_perm))
    
    pvalue = (np.sum(np.array(w_null) >= w_obs) + 1) / (b + 1)
    
    return pvalue, w_obs
```

---

#### Function 9: `spread_difference_test()`

**Baseline comparison**:

```python
def spread_difference_test(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    b: int = 1000,
    random_seed: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Permutation test for embedding spread difference (baseline metric).
    
    Simpler alternative to entropy test - no eigendecomposition.
    
    Returns:
        pvalue: permutation p-value
        spread_a: spread of A
        spread_b: spread of B
    """
    n_a = len(emb_a)
    n_b = len(emb_b)
    
    if n_a != n_b:
        raise ValueError(f"Equal sizes required. Got n_a={n_a}, n_b={n_b}.")
    
    rng = default_rng(random_seed)
    
    # Observed
    spread_a = embedding_spread(emb_a)
    spread_b = embedding_spread(emb_b)
    delta_obs = abs(spread_b - spread_a)
    
    # Permutation null
    combined = np.vstack([emb_a, emb_b])
    
    deltas_null = []
    for _ in range(b):
        perm = rng.permutation(len(combined))
        a_perm = combined[perm[:n_a]]
        b_perm = combined[perm[n_a:]]
        
        sp_a = embedding_spread(a_perm)
        sp_b = embedding_spread(b_perm)
        deltas_null.append(abs(sp_b - sp_a))
    
    pvalue = (np.sum(np.array(deltas_null) >= delta_obs) + 1) / (b + 1)
    
    return pvalue, spread_a, spread_b
```

---

### Phase 3: Unified Interface (1 hour)

#### Dataclass: `SpectralCharacterization`

```python
from dataclasses import dataclass

@dataclass
class SpectralCharacterization:
    """
    Complete spectral analysis comparing two distributions (equal n).
    
    Attributes:
        # Metadata
        label_a, label_b: Distribution names
        n: Sample size (same for both)
        
        # Entropy analysis
        entropy_a: Von Neumann entropy of A
        entropy_b: Von Neumann entropy of B
        entropy_a_norm: Normalized entropy (S / log(n)) in [0, 1]
        entropy_b_norm: Normalized entropy in [0, 1]
        entropy_delta: s_b - s_a
        entropy_pvalue: Permutation p-value
        effective_rank_a: exp(entropy_a)
        effective_rank_b: exp(entropy_b)
        
        # Wasserstein analysis
        wasserstein: Wasserstein distance on eigenvalue spectra
        wasserstein_pvalue: Permutation p-value
        
        # Baseline (spread) analysis
        spread_a: Average distance from centroid (normalized)
        spread_b: Average distance from centroid (normalized)
        spread_ratio: spread_b / spread_a
        spread_pvalue: Permutation p-value for spread difference
    """
    # Metadata
    label_a: str
    label_b: str
    n: int
    
    # Entropy
    entropy_a: float
    entropy_b: float
    entropy_a_norm: float
    entropy_b_norm: float
    entropy_delta: float
    entropy_pvalue: float
    effective_rank_a: float
    effective_rank_b: float
    
    # Wasserstein
    wasserstein: float
    wasserstein_pvalue: float
    
    # Baseline spread
    spread_a: float
    spread_b: float
    spread_ratio: float
    spread_pvalue: float
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = []
        lines.append(f"Spectral Characterization: {self.label_a} vs {self.label_b}")
        lines.append("=" * 60)
        lines.append(f"Sample size: n = {self.n} (both distributions)")
        lines.append("")
        
        # Entropy section
        lines.append("Von Neumann Entropy (eigenvalue concentration):")
        lines.append(f"  {self.label_a}: S = {self.entropy_a:.2f} "
                    f"(normalized: {self.entropy_a_norm:.2f}, "
                    f"eff_rank ≈ {self.effective_rank_a:.0f})")
        lines.append(f"  {self.label_b}: S = {self.entropy_b:.2f} "
                    f"(normalized: {self.entropy_b_norm:.2f}, "
                    f"eff_rank ≈ {self.effective_rank_b:.0f})")
        
        if self.entropy_delta > 0:
            more_spread = self.label_b
            ratio = self.effective_rank_b / self.effective_rank_a
        else:
            more_spread = self.label_a
            ratio = self.effective_rank_a / self.effective_rank_b
        
        lines.append(f"  Δ = {self.entropy_delta:+.2f} "
                    f"({more_spread} effective rank is {ratio:.2f}x)")
        
        sig = "*" if self.entropy_pvalue < 0.05 else ""
        lines.append(f"  P-value: {self.entropy_pvalue:.4f} {sig}")
        lines.append("")
        
        # Wasserstein section
        lines.append("Wasserstein Distance (eigenvalue distribution difference):")
        lines.append(f"  W = {self.wasserstein:.4f}")
        sig = "*" if self.wasserstein_pvalue < 0.05 else ""
        lines.append(f"  P-value: {self.wasserstein_pvalue:.4f} {sig}")
        lines.append("")
        
        # Baseline spread section
        lines.append("Baseline Metric (average distance from centroid):")
        lines.append(f"  {self.label_a}: spread = {self.spread_a:.4f}")
        lines.append(f"  {self.label_b}: spread = {self.spread_b:.4f}")
        lines.append(f"  Ratio: {self.spread_ratio:.2f}x")
        sig = "*" if self.spread_pvalue < 0.05 else ""
        lines.append(f"  P-value: {self.spread_pvalue:.4f} {sig}")
        lines.append("")
        
        # Interpretation
        lines.append("Interpretation:")
        
        if self.entropy_pvalue < 0.05 and self.spread_pvalue < 0.05:
            lines.append(f"  Both entropy and spread differ significantly")
            lines.append(f"  → Eigenvalue analysis confirms simple spread metric")
        elif self.entropy_pvalue < 0.05 and self.spread_pvalue >= 0.05:
            lines.append(f"  Entropy differs but spread doesn't")
            lines.append(f"  → Eigenvalue distribution changed beyond simple spread")
        elif self.spread_pvalue < 0.05 and self.entropy_pvalue >= 0.05:
            lines.append(f"  Spread differs but entropy doesn't")
            lines.append(f"  → Unusual case (investigate further)")
        else:
            lines.append(f"  No significant spectral differences detected")
        
        if self.wasserstein_pvalue < 0.05:
            lines.append(f"  Spectral structure differs significantly (Wasserstein test)")
        
        return "\n".join(lines)
```

---

#### Function 10: `spectral_characterization()`

**Main user-facing function**:

```python
def spectral_characterization(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    label_a: str = "A",
    label_b: str = "B",
    b: int = 1000,
    random_seed: Optional[int] = None
) -> SpectralCharacterization:
    """
    Complete spectral analysis comparing two distributions (equal n required).
    
    Computes:
    - Von Neumann entropy difference (eigenvalue concentration)
    - Wasserstein distance on eigenvalues (spectral distance)
    - Baseline spread metrics (for comparison)
    - Statistical significance for all metrics
    
    Args:
        emb_a: (n, d) embeddings from distribution A
        emb_b: (n, d) embeddings from distribution B (must have same n)
        label_a, label_b: Names for output
        b: Permutations for significance tests (default 1000)
        random_seed: Reproducibility
    
    Returns:
        SpectralCharacterization with complete results
    
    Raises:
        ValueError: if len(emb_a) != len(emb_b)
    
    Computational cost:
        ~60-90 seconds for b=1000, n=100, d=768
        Dominated by eigendecomposition (3 tests × b iterations)
    
    Example:
        >>> result = spectral_characterization(emb_fp32, emb_int8, "fp32", "int8")
        >>> print(result.summary())
        >>> 
        >>> # Check if eigenvalue analysis adds value over simple spread
        >>> if result.entropy_pvalue < 0.05 and result.spread_pvalue >= 0.05:
        >>>     print("Eigenvalue analysis detects difference that spread misses!")
    """
    n_a = len(emb_a)
    n_b = len(emb_b)
    
    if n_a != n_b:
        raise ValueError(
            f"Equal sample sizes required. Got n_a={n_a}, n_b={n_b}.\n"
            f"Subsample to min(n_a, n_b) or use semantic axes (no equal-n requirement)."
        )
    
    # Independent random seeds for three tests
    if random_seed is not None:
        from numpy.random import SeedSequence
        ss = SeedSequence(random_seed)
        seeds = ss.spawn(3)
        seed_entropy = seeds[0]
        seed_wass = seeds[1]
        seed_spread = seeds[2]
    else:
        seed_entropy = None
        seed_wass = None
        seed_spread = None
    
    # Entropy test
    pval_entropy, s_a, s_b, s_a_norm, s_b_norm = entropy_difference_test(
        emb_a, emb_b, b=b, random_seed=seed_entropy
    )
    
    # Effective ranks
    eff_a = np.exp(s_a)
    eff_b = np.exp(s_b)
    
    # Wasserstein test
    pval_wass, w = spectral_wasserstein_test(
        emb_a, emb_b, b=b, random_seed=seed_wass
    )
    
    # Baseline spread test
    pval_spread, spread_a, spread_b = spread_difference_test(
        emb_a, emb_b, b=b, random_seed=seed_spread
    )
    
    spread_r = spread_b / spread_a
    
    return SpectralCharacterization(
        label_a=label_a,
        label_b=label_b,
        n=n_a,
        entropy_a=s_a,
        entropy_b=s_b,
        entropy_a_norm=s_a_norm,
        entropy_b_norm=s_b_norm,
        entropy_delta=s_b - s_a,
        entropy_pvalue=pval_entropy,
        effective_rank_a=eff_a,
        effective_rank_b=eff_b,
        wasserstein=w,
        wasserstein_pvalue=pval_wass,
        spread_a=spread_a,
        spread_b=spread_b,
        spread_ratio=spread_r,
        spread_pvalue=pval_spread
    )
```

---

## Testing Strategy

### Unit Tests (Critical)

**File**: `model_equality_testing/tests/test_spectral_metrics.py`

#### Test 1: Type I Error Rate (CRITICAL)

```python
import numpy as np
import pytest
from model_equality_testing.src.spectral_metrics import (
    entropy_difference_test,
    spectral_wasserstein_test,
    spread_difference_test
)

def test_type_I_error_rate_entropy():
    """
    CRITICAL: Verify entropy test maintains 5% Type I error rate.
    
    This validates that the permutation test is unbiased.
    """
    np.random.seed(42)
    n_tests = 200  # Balance power vs runtime
    alpha = 0.05
    rejections = 0
    
    for i in range(n_tests):
        # Same distribution, equal sizes
        emb_all = np.random.randn(200, 768)
        emb_a = emb_all[:100]
        emb_b = emb_all[100:]  # Different samples, same distribution
        
        pval, _, _, _, _ = entropy_difference_test(
            emb_a, emb_b, b=100, random_seed=i  # Small b for speed
        )
        
        if pval < alpha:
            rejections += 1
    
    # Expected: ~10 rejections (5% of 200)
    # Binomial 95% CI: [3, 17] for p=0.05, n=200
    error_rate = rejections / n_tests
    assert 0.015 <= error_rate <= 0.085, (
        f"Type I error rate: {error_rate*100:.1f}% (expected 5%)"
    )
    
    print(f"Type I error (entropy): {rejections}/{n_tests} = {error_rate*100:.1f}%")


def test_type_I_error_rate_wasserstein():
    """Type I error validation for Wasserstein test."""
    np.random.seed(43)
    n_tests = 200
    rejections = 0
    
    for i in range(n_tests):
        emb_all = np.random.randn(200, 768)
        emb_a = emb_all[:100]
        emb_b = emb_all[100:]
        
        pval, _ = spectral_wasserstein_test(emb_a, emb_b, b=100, random_seed=i)
        
        if pval < 0.05:
            rejections += 1
    
    error_rate = rejections / n_tests
    assert 0.015 <= error_rate <= 0.085
    print(f"Type I error (Wasserstein): {rejections}/{n_tests} = {error_rate*100:.1f}%")


def test_type_I_error_rate_spread():
    """Type I error validation for spread test (baseline)."""
    np.random.seed(44)
    n_tests = 200
    rejections = 0
    
    for i in range(n_tests):
        emb_all = np.random.randn(200, 768)
        emb_a = emb_all[:100]
        emb_b = emb_all[100:]
        
        pval, _, _ = spread_difference_test(emb_a, emb_b, b=100, random_seed=i)
        
        if pval < 0.05:
            rejections += 1
    
    error_rate = rejections / n_tests
    assert 0.015 <= error_rate <= 0.085
    print(f"Type I error (spread): {rejections}/{n_tests} = {error_rate*100:.1f}%")
```

#### Test 2: Power Analysis

```python
def test_power_entropy():
    """Test that entropy test has reasonable power."""
    np.random.seed(42)
    n_tests = 100  # Smaller for power test
    rejections = 0
    
    for i in range(n_tests):
        # Create distributions with different spread
        # A: concentrated
        base = np.random.randn(20, 768)
        emb_a = base[np.random.choice(20, 100)]  # Repeated samples
        
        # B: spread out
        emb_b = np.random.randn(100, 768)
        
        pval, _, _, _, _ = entropy_difference_test(emb_a, emb_b, b=100, random_seed=i)
        
        if pval < 0.05:
            rejections += 1
    
    power = rejections / n_tests
    # Should detect most of the time
    assert power >= 0.70, f"Power: {power*100:.0f}% (expected ≥70%)"
    print(f"Power (entropy): {rejections}/{n_tests} = {power*100:.0f}%")
```

#### Test 3: Eigenvalue Spectrum Properties

```python
from model_equality_testing.src.spectral_metrics import (
    eigenvalue_spectrum,
    von_neumann_entropy_from_embeddings,
    normalized_entropy,
    effective_rank
)

def test_eigenvalue_spectrum_properties():
    """Test eigenvalue spectrum mathematical properties."""
    np.random.seed(42)
    emb = np.random.randn(100, 768)
    
    evals = eigenvalue_spectrum(emb)
    
    # Normalized to 1
    assert np.abs(evals.sum() - 1.0) < 1e-10
    
    # All positive
    assert np.all(evals > 0)
    
    # Sorted descending
    assert np.all(evals[:-1] >= evals[1:])
    
    # Length at most min(n, d)
    assert len(evals) <= min(100, 768)


def test_entropy_bounds():
    """Test entropy respects theoretical bounds."""
    np.random.seed(42)
    emb = np.random.randn(100, 768)
    
    s = von_neumann_entropy_from_embeddings(emb)
    max_s = np.log(len(emb))
    
    # Non-negative
    assert s >= 0
    
    # At most log(n)
    assert s <= max_s + 1e-6
    
    # Normalized version in [0, 1]
    s_norm = normalized_entropy(emb)
    assert 0 <= s_norm <= 1


def test_effective_rank_bounds():
    """Test effective rank in expected range."""
    np.random.seed(42)
    emb = np.random.randn(100, 768)
    
    r = effective_rank(emb)
    
    # In [1, n]
    assert 1 <= r <= len(emb)
    
    # For random data, typically 60-80% of n
    # (this is empirical, not strict)
    print(f"Effective rank for random data: {r:.1f}/{len(emb)} = {r/len(emb)*100:.0f}%")
```

#### Test 4: Equal Size Validation

```python
def test_equal_size_requirement():
    """Test that unequal sizes raise informative error."""
    emb_a = np.random.randn(100, 768)
    emb_b = np.random.randn(150, 768)
    
    with pytest.raises(ValueError, match="Equal sample sizes required"):
        entropy_difference_test(emb_a, emb_b)
    
    with pytest.raises(ValueError, match="Equal sample sizes required"):
        spectral_wasserstein_test(emb_a, emb_b)
    
    with pytest.raises(ValueError, match="Equal sizes required"):
        spread_difference_test(emb_a, emb_b)
```

#### Test 5: Baseline Comparison

```python
from model_equality_testing.src.spectral_metrics import (
    embedding_spread,
    spread_ratio
)

def test_spread_metric_properties():
    """Test spread baseline metric."""
    np.random.seed(42)
    
    # More spread should give higher value
    emb_concentrated = np.random.randn(100, 768) * 0.5
    emb_spread = np.random.randn(100, 768) * 2.0
    
    sp_conc = embedding_spread(emb_concentrated)
    sp_spread = embedding_spread(emb_spread)
    
    assert sp_spread > sp_conc, "More spread should have higher spread metric"
    
    # Ratio
    ratio = spread_ratio(emb_concentrated, emb_spread)
    assert ratio > 1, "Ratio should be > 1 when B more spread"
```

#### Test 6: Integration with Real Data

```python
def test_with_real_data():
    """Integration test with actual LLM data."""
    from model_equality_testing.dataset import load_distribution
    from model_equality_testing.src.embeddings import embed_sample
    
    # Load
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
    
    # Draw equal-size samples
    sample_fp32 = dist_fp32.draw_completion_sample(50)  # Small for speed
    sample_int8 = dist_int8.draw_completion_sample(50)
    
    # Embed
    emb_fp32 = embed_sample(sample_fp32)
    emb_int8 = embed_sample(sample_int8)
    
    # Characterization (small b for speed)
    from model_equality_testing.src.spectral_metrics import spectral_characterization
    
    result = spectral_characterization(
        emb_fp32, emb_int8, "fp32", "int8", b=50, random_seed=42
    )
    
    # Sanity checks
    assert result.n == 50
    assert 0 <= result.entropy_a <= np.log(50)
    assert 0 <= result.entropy_b <= np.log(50)
    assert 0 <= result.entropy_pvalue <= 1
    assert 0 <= result.wasserstein_pvalue <= 1
    assert 0 <= result.spread_pvalue <= 1
    
    print("\nfp32 vs int8 (n=50, b=50):")
    print(result.summary())
```

---

## Computational Cost (Realistic)

### Detailed Analysis

**Per permutation** (n=100, d=768):
- Gram matrix: O(n²d) = 100² × 768 = 7.68M ops
- Eigendecomposition: O(n³) = 1M ops
- Eigenvalue normalization: O(n) = negligible
- **Total**: ~8.7M ops → **~10-12ms**

**Three tests** (entropy, Wasserstein, spread):
- Each needs eigendecomposition or distance computation
- All similar cost
- **Per permutation**: ~30ms total

**With b=1000**:
- 1000 × 30ms = **30 seconds per test**
- **Total (3 tests)**: ~90 seconds

**Realistic estimate**: **60-90 seconds** for complete characterization with b=1000.

### Comparison

| Method | Runtime (n=100, b=1000) |
|--------|------------------------|
| **Spectral metrics V4** | 60-90s |
| Trace distance V2 | 60-120s |
| Semantic axes | <1s |
| MMD | 30-60s |

**Spectral metrics are comparable to trace distance, NOT faster.**

---

## Usage Examples

### Example 1: Basic Usage

```python
from model_equality_testing.src.embeddings import embed_sample
from model_equality_testing.src.spectral_metrics import spectral_characterization

# Embed equal-size samples
emb_a = embed_sample(sample_a)  # (100, 768)
emb_b = embed_sample(sample_b)  # (100, 768)

# Complete characterization
result = spectral_characterization(emb_a, emb_b, "Model A", "Model B", b=1000)

# Print summary
print(result.summary())
```

**Output**:
```
Spectral Characterization: Model A vs Model B
============================================================
Sample size: n = 100 (both distributions)

Von Neumann Entropy (eigenvalue concentration):
  Model A: S = 3.24 (normalized: 0.70, eff_rank ≈ 25)
  Model B: S = 4.15 (normalized: 0.90, eff_rank ≈ 63)
  Δ = +0.91 (Model B effective rank is 2.52x)
  P-value: 0.002 *

Wasserstein Distance (eigenvalue distribution difference):
  W = 0.1243
  P-value: 0.001 *

Baseline Metric (average distance from centroid):
  Model A: spread = 0.0847
  Model B: spread = 0.1023
  Ratio: 1.21x
  P-value: 0.003 *

Interpretation:
  Both entropy and spread differ significantly
  → Eigenvalue analysis confirms simple spread metric
  Spectral structure differs significantly (Wasserstein test)
```

---

### Example 2: Check if Eigenvalues Add Value

```python
result = spectral_characterization(emb_a, emb_b, "A", "B", b=1000)

# Does entropy detect something spread doesn't?
if result.entropy_pvalue < 0.05 and result.spread_pvalue >= 0.05:
    print("Eigenvalue analysis adds value!")
    print("Entropy detects difference that simple spread misses.")
elif result.spread_pvalue < 0.05 and result.entropy_pvalue >= 0.05:
    print("Simple spread is sufficient.")
    print("Eigenvalue analysis doesn't add information.")
else:
    print("Both metrics agree (both significant or both not).")

# Quantify agreement
print(f"\nCorrelation between metrics:")
print(f"  Entropy ratio: {result.effective_rank_b / result.effective_rank_a:.2f}x")
print(f"  Spread ratio: {result.spread_ratio:.2f}x")
# If ratios similar → metrics capturing same thing
```

---

### Example 3: Complete Workflow with Semantic Axes

```python
# 1. Embed once
emb_a = embed_sample(sample_a)
emb_b = embed_sample(sample_b)

# 2. Spectral characterization (holistic)
spectral = spectral_characterization(emb_a, emb_b, "fp32", "int8", b=1000)
print(spectral.summary())

# 3. Semantic interpretation (detailed)
from model_equality_testing.src.semantic_axes import interpret_difference

axes = [prof_axis, tech_axis, form_axis]
semantic = interpret_difference(emb_a, emb_b, axes, label_a="fp32", label_b="int8")

print("\n" + "="*60)
print("Semantic Interpretation:")
print(semantic.summary())

# Complete picture:
# Spectral: "int8 has 2.5x effective rank, spread 1.2x, Wasserstein = 0.12"
# Semantic: "professionalism -0.54 (g=0.82), technicality +0.31 (g=0.45)"
```

---

## Documentation Updates

### Add to README.md

After semantic axes section:

```markdown
### Spectral Metrics for Holistic Quantification

For holistic distributional quantification (requires equal sample sizes):

```python
from model_equality_testing.src.spectral_metrics import spectral_characterization

# Complete spectral analysis (n_a must equal n_b)
result = spectral_characterization(emb_a, emb_b, "fp32", "int8", b=1000)

print(result.summary())
# Shows:
# - Von Neumann entropy (eigenvalue concentration)
# - Wasserstein distance (spectral structure difference)
# - Baseline spread metric (for comparison)
# - Statistical significance for all metrics
```

**Requirements**:
- Equal sample sizes (n_a = n_b)
- Computational cost: ~60-90 seconds for b=1000, n=100

**Provides**:
- Holistic quantification of distributional difference
- Comparison to simple baseline (validates eigenvalue approach)
- Statistical significance tests

See `SPECTRAL-METRICS-GUIDE.md` for details.
```

---

### Create SPECTRAL-METRICS-GUIDE.md

Key sections:

#### What Are Spectral Metrics?

Eigenvalue-based measures of distributional structure. **Not** about semantic clusters or topics - about the concentration vs spread of the embedding similarity matrix eigenvalues.

#### Honest Interpretation

**Von Neumann entropy**:
- Measures: Concentration of eigenvalue distribution
- NOT: Number of semantic topics/clusters
- Range: [0, log(n)]
- High entropy: Eigenvalues spread out (many effective dimensions)
- Low entropy: Few dominant eigenvalues (concentrated)

**Effective rank**:
- Transformation: exp(entropy)
- Interpretation: "Perplexity" of eigenvalue distribution
- If uniform over k eigenvalues: eff_rank ≈ k
- For random data (n=100, d=768): typically 60-80
- **NOT** the number of semantic modes in your data

#### When Eigenvalues Add Value

Compare eigenvalue metrics to simple spread baseline:
- Both significant → Eigenvalues confirm spread (redundant but rigorous)
- Entropy significant, spread not → Eigenvalues detect structural change beyond spread (value added!)
- Spread significant, entropy not → Unusual (investigate)

#### Limitations

1. **Equal sample sizes required** (n_a = n_b)
2. **Computational cost**: ~60-90 seconds (not instant)
3. **Abstract**: Eigenvalues less interpretable than semantic axes
4. **Sample-dependent**: Eigenvalue distributions depend on finite sample
5. **Sidesteps, doesn't solve**: Cross-matrix comparison issues avoided by looking at eigenvalues separately, but still sample-specific Gram matrices

---

## Timeline

### Hour 1: Core + Baseline
- ✓ Create spectral_metrics.py
- ✓ Implement eigenvalue_spectrum()
- ✓ Implement von_neumann_entropy_from_embeddings()
- ✓ Implement normalized_entropy()
- ✓ Implement effective_rank()
- ✓ Implement spectral_wasserstein()
- ✓ Implement embedding_spread() (baseline)
- ✓ Implement spread_ratio()

### Hour 2: Inference
- ✓ Implement entropy_difference_test()
- ✓ Implement spectral_wasserstein_test()
- ✓ Implement spread_difference_test() (baseline)

### Hour 3: Interface + Critical Tests
- ✓ Implement SpectralCharacterization dataclass
- ✓ Implement spectral_characterization()
- ✓ **Type I error tests** (critical validation)
- ✓ Power test
- ✓ Equal-size validation test

### Hour 4: Additional Tests + Documentation
- ✓ Eigenvalue properties tests
- ✓ Baseline comparison tests
- ✓ Integration test with real data
- ✓ Update README.md
- ✓ Create SPECTRAL-METRICS-GUIDE.md
- ✓ Verify all examples run

**Total: 4 hours**

---

## Summary

### Changes from V3

| Aspect | V3 | V4 |
|--------|----|----|
| **Sample sizes** | Claimed n_a ≠ n_b works | **Require n_a = n_b** |
| **Computational cost** | "40-50s" (wrong) | **"60-90s" (realistic)** |
| **"Ill-defined" issues** | "Fixed" | **"Sidestepped" (honest)** |
| **Interpretation** | "Modes" oversold | **Conservative framing** |
| **Baseline comparison** | Missing | **Included** |
| **Type I error tests** | Basic | **Rigorous** |
| **Honesty** | Oversells benefits | **Realistic about limitations** |

### What V4 Provides

**Two complementary metrics**:
1. Von Neumann entropy (eigenvalue concentration)
2. Wasserstein distance (spectral structure)

**Plus baseline**:
3. Simple spread metric (proves eigenvalues add value)

**Statistical rigor**:
- Permutation tests for all three
- Type I error validation
- Power analysis

### What V4 Acknowledges

**Limitations**:
- ✓ Equal sample sizes required
- ✓ Computational cost ~60-90s (not instant)
- ✓ "Modes" are eigenvalue patterns, not semantic clusters
- ✓ Sidesteps cross-matrix issues, doesn't eliminate them
- ✓ May be redundant with simple spread metric

**Honest value proposition**:
- Holistic quantification (single-number summaries)
- Statistically rigorous (permutation tests)
- Comparable to other metrics (spread baseline validates)
- Complements semantic axes (holistic vs per-dimension)

### Recommendation

✅ **Implement V4**

**With caveats**:
- Equal n required (same as trace distance - not new)
- Compare to baselines (prove eigenvalues add value)
- Be honest about interpretation (eigenvalue patterns, not topics)
- Don't oversell speed (60-90s is realistic)

**Skip if**:
- Semantic axes alone are sufficient
- Can't afford 60-90s computational cost
- Need different sample sizes (use semantic axes)

**Time investment**: 4 hours for validated, production-ready implementation with honest documentation.
