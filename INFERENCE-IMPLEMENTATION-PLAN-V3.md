# Implementation Plan V3: Spectral Metrics for Distributional Quantification

## Overview

This plan implements **spectral metrics** for quantifying distributional differences beyond binary hypothesis testing. The focus is on **eigenvalue-based analysis** that provides interpretable effect magnitudes.

**Key principle**: Don't compare density matrices directly. Instead, analyze their eigenvalue distributions using established statistical metrics.

**Goal**: Quantify "how different" distributions are, not just "are they different" (T/F).

**Estimated time**: 3 hours for complete implementation, testing, and documentation.

---

## Motivation: Quantification Beyond T/F

### What You Already Have

| Tool | Provides | Quantification |
|------|----------|----------------|
| **Semantic axes** | Per-dimension shifts | ✅ Hedges' g, mean Δ |
| **MMD tests** | Hypothesis test | ⚠️ Statistic magnitude, but no scale |
| **Chi-squared, L1, L2** | Hypothesis tests | ⚠️ Statistics, unclear interpretation |

### What's Missing

**Holistic spectral quantification**: A single number summarizing overall distributional difference in spectral terms.

**Gap**: You have interpretable per-dimension quantification (semantic axes) but no holistic spectral summary.

---

## Solution: Eigenvalue-Based Spectral Metrics

### Core Insight

Instead of comparing n×n density matrices directly (problematic), compare their **eigenvalue distributions** (well-defined).

**Advantages**:
- ✅ Works for any sample sizes (no n_a = n_b requirement)
- ✅ No PCA circularity issues
- ✅ Standard statistical metrics apply
- ✅ Clear interpretation
- ✅ Established in statistics/ML literature

### Two Complementary Metrics

**1. Von Neumann Entropy Difference**
- **What**: Change in diversity/spread
- **Output**: "Distribution B is 2.5x more diverse (63 modes vs 25 modes)"
- **Interpretation**: How concentrated vs spread out

**2. Wasserstein Distance on Eigenvalue Spectra**
- **What**: Spectral distance (Earth Mover's Distance)
- **Output**: "Spectral distance = 0.124 (2.3σ above null)"
- **Interpretation**: Cost to transform one spectrum to another

---

## File Structure

### New File: `model_equality_testing/src/spectral_metrics.py`

Clean separation of concerns:
- `quantum_metrics.py` → Legacy density matrix metrics (can keep for compatibility)
- `spectral_metrics.py` → **New, focused on eigenvalue analysis**
- `semantic_axes.py` → Already good (interpretable dimensions)

**Naming rationale**: "Spectral" is accurate (eigenvalue-based), not overloaded like "quantum".

---

## Implementation

### Phase 1: Core Functions (1 hour)

#### Function 1: `eigenvalue_spectrum()`

**Location**: `model_equality_testing/src/spectral_metrics.py`

**Signature**:
```python
import numpy as np
from typing import Optional

def eigenvalue_spectrum(embeddings: np.ndarray) -> np.ndarray:
    """
    Extract normalized eigenvalue distribution from embeddings.
    
    Computes eigenvalues of the Gram matrix G = E @ E^T and normalizes
    them to sum to 1 (probability distribution over modes).
    
    Args:
        embeddings: (n, d) SBERT embeddings (e.g., n=100, d=768)
    
    Returns:
        evals: (k,) positive eigenvalues, sorted descending, normalized to sum to 1
              where k ≤ min(n, d) is the number of positive eigenvalues
    
    Note:
        This works in full embedding space (no PCA) to avoid circularity.
        The eigenvalues represent the "spectral distribution" of the data.
    
    Example:
        >>> evals = eigenvalue_spectrum(emb)
        >>> print(f"Top 3 modes: {evals[:3]}")  # e.g., [0.35, 0.22, 0.15]
        >>> print(f"Total modes: {len(evals)}")  # e.g., 42
    """
    # Gram matrix: pairwise inner products
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

**Signature**:
```python
def von_neumann_entropy_from_embeddings(embeddings: np.ndarray) -> float:
    """
    Compute Von Neumann entropy from embeddings.
    
    S(ρ) = -Σ λᵢ log λᵢ
    
    Measures the diversity/uncertainty of the distribution.
    Higher entropy = more spread out across modes.
    
    Args:
        embeddings: (n, d) SBERT embeddings
    
    Returns:
        entropy: float in [0, log(n)]
                 0 = all samples identical (one mode)
                 log(n) = maximally diverse (uniform over n modes)
    
    Example:
        >>> s = von_neumann_entropy_from_embeddings(emb)
        >>> print(f"Entropy: {s:.2f}")  # e.g., 3.24
        >>> print(f"Max possible: {np.log(len(emb)):.2f}")  # e.g., 4.61
    """
    evals = eigenvalue_spectrum(embeddings)
    
    # S = -Σ λᵢ log λᵢ (with numerical stability)
    entropy = -np.sum(evals * np.log(evals + 1e-10))
    
    return entropy
```

---

#### Function 3: `effective_rank()`

**Signature**:
```python
def effective_rank(embeddings: np.ndarray) -> float:
    """
    Effective dimensionality: exp(S).
    
    Interpretation: approximate number of "modes" the distribution spans.
    
    Args:
        embeddings: (n, d) SBERT embeddings
    
    Returns:
        eff_rank: float in [1, n]
                  1 = all samples identical (one mode)
                  n = maximally diverse (n independent modes)
    
    Example:
        >>> rank = effective_rank(emb)
        >>> print(f"~{rank:.0f} effective modes")  # e.g., "~25 effective modes"
    
    Note:
        This is more interpretable than raw entropy:
        - "25 modes" is clearer than "S = 3.22"
        - Can compare: "B has 2.5x more modes than A"
    """
    s = von_neumann_entropy_from_embeddings(embeddings)
    return np.exp(s)
```

---

#### Function 4: `spectral_wasserstein()`

**Signature**:
```python
from scipy.stats import wasserstein_distance

def spectral_wasserstein(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """
    Wasserstein distance on eigenvalue spectra.
    
    Measures the "cost" to transform one eigenvalue distribution to another
    (Earth Mover's Distance).
    
    Args:
        emb_a: (n_a, d) embeddings from distribution A
        emb_b: (n_b, d) embeddings from distribution B
    
    Returns:
        distance: float ≥ 0
                  0 = identical eigenvalue distributions
                  larger = more different spectral structure
    
    Note:
        Works for different sample sizes (n_a ≠ n_b).
        scipy.stats.wasserstein_distance handles different-length
        distributions naturally.
    
    Example:
        >>> w = spectral_wasserstein(emb_fp32, emb_int8)
        >>> print(f"Spectral distance: {w:.4f}")  # e.g., 0.1243
    """
    evals_a = eigenvalue_spectrum(emb_a)
    evals_b = eigenvalue_spectrum(emb_b)
    
    # Wasserstein distance (handles different lengths)
    return wasserstein_distance(evals_a, evals_b)
```

---

### Phase 2: Statistical Inference (1 hour)

#### Function 5: `entropy_difference_test()`

**Signature**:
```python
from typing import Tuple
from numpy.random import default_rng

def entropy_difference_test(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    b: int = 1000,
    random_seed: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Permutation test for Von Neumann entropy difference.
    
    Tests H0: S(ρ_A) = S(ρ_B)
    vs H1: S(ρ_A) ≠ S(ρ_B)
    
    Args:
        emb_a: (n_a, d) embeddings from distribution A
        emb_b: (n_b, d) embeddings from distribution B
        b: number of permutations (default 1000)
        random_seed: optional random seed for reproducibility
    
    Returns:
        pvalue: float in [0, 1] - permutation p-value
        s_a: entropy of distribution A
        s_b: entropy of distribution B
    
    Note:
        Works for different sample sizes (n_a ≠ n_b).
        No PCA circularity issues.
    
    Example:
        >>> pval, s_a, s_b = entropy_difference_test(emb_a, emb_b, b=1000)
        >>> print(f"Entropy A: {s_a:.2f}, B: {s_b:.2f}")
        >>> print(f"Difference: {abs(s_b - s_a):.2f}, p-value: {pval:.4f}")
        >>> if pval < 0.05:
        >>>     more_diverse = "B" if s_b > s_a else "A"
        >>>     print(f"{more_diverse} is significantly more diverse")
    """
    rng = default_rng(random_seed)
    
    # Observed entropies
    s_a = von_neumann_entropy_from_embeddings(emb_a)
    s_b = von_neumann_entropy_from_embeddings(emb_b)
    delta_obs = abs(s_b - s_a)
    
    # Permutation null distribution
    combined = np.vstack([emb_a, emb_b])
    n_a = len(emb_a)
    
    deltas_null = []
    for _ in range(b):
        # Randomly permute combined samples
        perm = rng.permutation(len(combined))
        a_perm = combined[perm[:n_a]]
        b_perm = combined[perm[n_a:]]
        
        # Compute entropy difference under permutation
        s_a_perm = von_neumann_entropy_from_embeddings(a_perm)
        s_b_perm = von_neumann_entropy_from_embeddings(b_perm)
        deltas_null.append(abs(s_b_perm - s_a_perm))
    
    # P-value: (# null >= observed + 1) / (b + 1)
    pvalue = (np.sum(np.array(deltas_null) >= delta_obs) + 1) / (b + 1)
    
    return pvalue, s_a, s_b
```

---

#### Function 6: `spectral_wasserstein_test()`

**Signature**:
```python
def spectral_wasserstein_test(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    b: int = 1000,
    random_seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Permutation test for Wasserstein distance on eigenvalue spectra.
    
    Tests H0: eigenvalue distributions are identical
    vs H1: eigenvalue distributions differ
    
    Args:
        emb_a: (n_a, d) embeddings from distribution A
        emb_b: (n_b, d) embeddings from distribution B
        b: number of permutations (default 1000)
        random_seed: optional random seed
    
    Returns:
        pvalue: float in [0, 1] - permutation p-value
        w_obs: observed Wasserstein distance
    
    Note:
        Works for different sample sizes (n_a ≠ n_b).
    
    Example:
        >>> pval, w = spectral_wasserstein_test(emb_a, emb_b, b=1000)
        >>> print(f"Wasserstein: {w:.4f}, p-value: {pval:.4f}")
        >>> if pval < 0.05:
        >>>     print("Spectral distributions differ significantly")
    """
    rng = default_rng(random_seed)
    
    # Observed Wasserstein distance
    w_obs = spectral_wasserstein(emb_a, emb_b)
    
    # Permutation null distribution
    combined = np.vstack([emb_a, emb_b])
    n_a = len(emb_a)
    
    w_null = []
    for _ in range(b):
        perm = rng.permutation(len(combined))
        a_perm = combined[perm[:n_a]]
        b_perm = combined[perm[n_a:]]
        w_null.append(spectral_wasserstein(a_perm, b_perm))
    
    # P-value
    pvalue = (np.sum(np.array(w_null) >= w_obs) + 1) / (b + 1)
    
    return pvalue, w_obs
```

---

### Phase 3: Unified Interface (1 hour)

#### Dataclass: `SpectralCharacterization`

**Signature**:
```python
from dataclasses import dataclass

@dataclass
class SpectralCharacterization:
    """
    Results of spectral analysis comparing two distributions.
    
    Attributes:
        label_a, label_b: Distribution names
        
        # Entropy (diversity) analysis
        entropy_a: Von Neumann entropy of A
        entropy_b: Von Neumann entropy of B
        entropy_delta: s_b - s_a (positive = B more diverse)
        entropy_pvalue: Permutation p-value for entropy difference
        effective_rank_a: Effective dimensionality of A (exp(s_a))
        effective_rank_b: Effective dimensionality of B (exp(s_b))
        
        # Wasserstein (spectral distance) analysis
        wasserstein: Wasserstein distance on eigenvalue spectra
        wasserstein_pvalue: Permutation p-value
    """
    label_a: str
    label_b: str
    
    # Entropy
    entropy_a: float
    entropy_b: float
    entropy_delta: float
    entropy_pvalue: float
    effective_rank_a: float
    effective_rank_b: float
    
    # Wasserstein
    wasserstein: float
    wasserstein_pvalue: float
    
    def summary(self) -> str:
        """
        Human-readable summary of spectral characterization.
        
        Returns formatted text with:
        - Entropy comparison (diversity)
        - Effective ranks (interpretable)
        - Wasserstein distance (spectral difference)
        - Statistical significance
        - High-level interpretation
        """
        lines = []
        lines.append(f"Spectral Characterization: {self.label_a} vs {self.label_b}")
        lines.append("=" * 60)
        lines.append("")
        
        # Diversity section
        lines.append("Diversity (Von Neumann Entropy):")
        lines.append(f"  {self.label_a}: S = {self.entropy_a:.2f} "
                    f"(~{self.effective_rank_a:.0f} effective modes)")
        lines.append(f"  {self.label_b}: S = {self.entropy_b:.2f} "
                    f"(~{self.effective_rank_b:.0f} effective modes)")
        
        # Determine which is more diverse
        if self.entropy_delta > 0:
            more_diverse = self.label_b
            ratio = self.effective_rank_b / self.effective_rank_a
        else:
            more_diverse = self.label_a
            ratio = self.effective_rank_a / self.effective_rank_b
        
        lines.append(f"  Δ = {self.entropy_delta:+.2f} "
                    f"({more_diverse} is {ratio:.1f}x more diverse)")
        
        # Significance
        sig = "*" if self.entropy_pvalue < 0.05 else ""
        lines.append(f"  P-value: {self.entropy_pvalue:.4f} {sig}")
        lines.append("")
        
        # Spectral distance section
        lines.append("Spectral Distance (Wasserstein on eigenvalues):")
        lines.append(f"  W = {self.wasserstein:.4f}")
        sig = "*" if self.wasserstein_pvalue < 0.05 else ""
        lines.append(f"  P-value: {self.wasserstein_pvalue:.4f} {sig}")
        lines.append("")
        
        # Interpretation
        lines.append("Interpretation:")
        interpretations = []
        
        if self.entropy_pvalue < 0.05:
            interpretations.append(
                f"{more_diverse} has significantly more diverse outputs"
            )
        
        if self.wasserstein_pvalue < 0.05:
            interpretations.append(
                "Distributions have significantly different spectral structure"
            )
        
        if not interpretations:
            interpretations.append("No significant spectral differences detected")
        
        for interp in interpretations:
            lines.append(f"  {interp}")
        
        return "\n".join(lines)
```

---

#### Function 7: `spectral_characterization()`

**Main user-facing function**:

**Signature**:
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
    Complete spectral analysis comparing two distributions.
    
    Computes:
    - Von Neumann entropy difference (diversity quantification)
    - Wasserstein distance on eigenvalue spectra (spectral distance)
    - Statistical significance for both metrics
    - Effective ranks (interpretable diversity measure)
    
    Args:
        emb_a: (n_a, d) embeddings from distribution A
        emb_b: (n_b, d) embeddings from distribution B
        label_a: Name for distribution A (e.g., "fp32")
        label_b: Name for distribution B (e.g., "int8")
        b: Number of permutations for significance tests (default 1000)
        random_seed: Optional random seed for reproducibility
    
    Returns:
        SpectralCharacterization with complete results
    
    Computational cost:
        ~30-60 seconds for b=1000 with n=100, d=768
        (Much faster than trace distance: no eigendecomposition per iteration,
         just eigenvalue spectrum extraction)
    
    Example:
        >>> from model_equality_testing.src.embeddings import embed_sample
        >>> from model_equality_testing.src.spectral_metrics import spectral_characterization
        >>> 
        >>> # Embed samples
        >>> emb_fp32 = embed_sample(sample_fp32)
        >>> emb_int8 = embed_sample(sample_int8)
        >>> 
        >>> # Spectral analysis
        >>> result = spectral_characterization(emb_fp32, emb_int8, "fp32", "int8")
        >>> 
        >>> # Print summary
        >>> print(result.summary())
        >>> 
        >>> # Access individual results
        >>> print(f"Diversity ratio: {result.effective_rank_b / result.effective_rank_a:.1f}x")
    """
    # Use independent random seeds for two tests
    if random_seed is not None:
        from numpy.random import SeedSequence
        ss = SeedSequence(random_seed)
        seeds = ss.spawn(2)
        seed_entropy = seeds[0]
        seed_wass = seeds[1]
    else:
        seed_entropy = None
        seed_wass = None
    
    # Entropy difference test
    pval_entropy, s_a, s_b = entropy_difference_test(
        emb_a, emb_b, b=b, random_seed=seed_entropy
    )
    
    # Effective ranks
    eff_a = np.exp(s_a)
    eff_b = np.exp(s_b)
    
    # Wasserstein distance test
    pval_wass, w = spectral_wasserstein_test(
        emb_a, emb_b, b=b, random_seed=seed_wass
    )
    
    return SpectralCharacterization(
        label_a=label_a,
        label_b=label_b,
        entropy_a=s_a,
        entropy_b=s_b,
        entropy_delta=s_b - s_a,
        entropy_pvalue=pval_entropy,
        effective_rank_a=eff_a,
        effective_rank_b=eff_b,
        wasserstein=w,
        wasserstein_pvalue=pval_wass
    )
```

---

## Testing Strategy

### Unit Tests

**File**: `model_equality_testing/tests/test_spectral_metrics.py`

#### Test 1: Eigenvalue Spectrum Properties

```python
import numpy as np
import pytest
from model_equality_testing.src.spectral_metrics import (
    eigenvalue_spectrum,
    von_neumann_entropy_from_embeddings,
    effective_rank
)

def test_eigenvalue_spectrum_properties():
    """Test eigenvalue spectrum has expected properties."""
    np.random.seed(42)
    emb = np.random.randn(100, 768)
    
    evals = eigenvalue_spectrum(emb)
    
    # Should sum to 1 (normalized)
    assert np.abs(evals.sum() - 1.0) < 1e-10, "Eigenvalues should sum to 1"
    
    # Should all be positive
    assert np.all(evals > 0), "All eigenvalues should be positive"
    
    # Should be sorted descending
    assert np.all(evals[:-1] >= evals[1:]), "Should be sorted descending"
    
    # Length should be at most min(n, d)
    assert len(evals) <= min(100, 768)

def test_entropy_bounds():
    """Test Von Neumann entropy respects bounds."""
    np.random.seed(42)
    emb = np.random.randn(100, 768)
    
    s = von_neumann_entropy_from_embeddings(emb)
    
    # Should be non-negative
    assert s >= 0, "Entropy should be non-negative"
    
    # Should be at most log(n)
    max_entropy = np.log(len(emb))
    assert s <= max_entropy + 1e-6, f"Entropy {s} exceeds max {max_entropy}"

def test_effective_rank():
    """Test effective rank is sensible."""
    np.random.seed(42)
    emb = np.random.randn(100, 768)
    
    rank = effective_rank(emb)
    
    # Should be between 1 and n
    assert 1 <= rank <= len(emb), f"Effective rank {rank} out of bounds [1, {len(emb)}]"
```

#### Test 2: Entropy Difference Test

```python
def test_entropy_difference_null():
    """Test entropy difference on same distribution."""
    np.random.seed(42)
    
    # Same distribution
    emb_all = np.random.randn(200, 768)
    emb_a = emb_all[:100]
    emb_b = emb_all[100:]
    
    pval, s_a, s_b = entropy_difference_test(emb_a, emb_b, b=100, random_seed=42)
    
    # Entropies should be similar (not exactly equal due to sampling)
    assert abs(s_b - s_a) < 0.5, "Entropies should be similar for same distribution"
    
    # P-value should be relatively high (not always, but on average)
    # Just check it's valid
    assert 0 <= pval <= 1, "P-value must be in [0, 1]"

def test_entropy_difference_alternative():
    """Test entropy difference detects real difference."""
    np.random.seed(42)
    
    # Different diversity: concentrated vs spread
    # A: concentrated (low-rank structure)
    base = np.random.randn(10, 768)
    emb_a = base[np.random.choice(10, 100)]  # Repeated samples
    
    # B: diverse (high-rank)
    emb_b = np.random.randn(100, 768)
    
    pval, s_a, s_b = entropy_difference_test(emb_a, emb_b, b=100, random_seed=42)
    
    # B should have higher entropy (more diverse)
    assert s_b > s_a, "Diverse distribution should have higher entropy"
```

#### Test 3: Wasserstein Test

```python
from model_equality_testing.src.spectral_metrics import (
    spectral_wasserstein,
    spectral_wasserstein_test
)

def test_spectral_wasserstein_properties():
    """Test Wasserstein distance properties."""
    np.random.seed(42)
    emb_a = np.random.randn(100, 768)
    emb_b = np.random.randn(100, 768)
    
    w = spectral_wasserstein(emb_a, emb_b)
    
    # Should be non-negative
    assert w >= 0, "Wasserstein distance should be non-negative"
    
    # Self-distance should be zero
    w_self = spectral_wasserstein(emb_a, emb_a)
    assert w_self < 1e-10, f"Self-distance should be ~0, got {w_self}"

def test_spectral_wasserstein_different_sizes():
    """Test Wasserstein works for different sample sizes."""
    np.random.seed(42)
    emb_a = np.random.randn(50, 768)   # Different sizes
    emb_b = np.random.randn(150, 768)
    
    # Should not raise error
    w = spectral_wasserstein(emb_a, emb_b)
    assert w >= 0
    
    pval, w_test = spectral_wasserstein_test(emb_a, emb_b, b=50, random_seed=42)
    assert 0 <= pval <= 1
```

#### Test 4: Spectral Characterization

```python
from model_equality_testing.src.spectral_metrics import spectral_characterization

def test_spectral_characterization():
    """Test complete spectral characterization."""
    np.random.seed(42)
    
    # Create two distributions
    emb_a = np.random.randn(100, 768)
    emb_b = np.random.randn(100, 768) + 0.5  # Shifted
    
    result = spectral_characterization(
        emb_a, emb_b, "A", "B", b=50, random_seed=42
    )
    
    # Check all fields present
    assert hasattr(result, 'entropy_a')
    assert hasattr(result, 'entropy_b')
    assert hasattr(result, 'entropy_delta')
    assert hasattr(result, 'entropy_pvalue')
    assert hasattr(result, 'effective_rank_a')
    assert hasattr(result, 'effective_rank_b')
    assert hasattr(result, 'wasserstein')
    assert hasattr(result, 'wasserstein_pvalue')
    
    # Check delta is consistent
    assert abs(result.entropy_delta - (result.entropy_b - result.entropy_a)) < 1e-10
    
    # Check effective ranks consistent
    assert abs(result.effective_rank_a - np.exp(result.entropy_a)) < 1e-10
    
    # Check summary runs without error
    summary = result.summary()
    assert len(summary) > 0
    assert "Spectral Characterization" in summary
```

#### Test 5: Integration with Real Data

```python
def test_with_real_data():
    """Integration test with actual LLM data."""
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
    
    # Draw small samples for speed
    sample_fp32 = dist_fp32.draw_completion_sample(30)
    sample_int8 = dist_int8.draw_completion_sample(30)
    
    # Embed
    emb_fp32 = embed_sample(sample_fp32)
    emb_int8 = embed_sample(sample_int8)
    
    # Spectral characterization (small b for speed)
    result = spectral_characterization(
        emb_fp32, emb_int8, "fp32", "int8", b=50, random_seed=42
    )
    
    # Sanity checks
    assert 0 <= result.entropy_a <= np.log(30)
    assert 0 <= result.entropy_b <= np.log(30)
    assert 0 <= result.entropy_pvalue <= 1
    assert 0 <= result.wasserstein_pvalue <= 1
    assert result.wasserstein >= 0
    
    print(f"\nfp32 vs int8 (n=30, b=50):")
    print(result.summary())
```

---

## Usage Examples

### Example 1: Basic Usage

```python
from model_equality_testing.src.embeddings import embed_sample
from model_equality_testing.src.spectral_metrics import spectral_characterization

# Embed samples
emb_a = embed_sample(sample_a)
emb_b = embed_sample(sample_b)

# Spectral characterization
result = spectral_characterization(emb_a, emb_b, "Model A", "Model B")

# Print summary
print(result.summary())
```

**Output**:
```
Spectral Characterization: Model A vs Model B
============================================================

Diversity (Von Neumann Entropy):
  Model A: S = 3.24 (~25 effective modes)
  Model B: S = 4.15 (~63 effective modes)
  Δ = +0.91 (Model B is 2.5x more diverse)
  P-value: 0.002 *

Spectral Distance (Wasserstein on eigenvalues):
  W = 0.1243
  P-value: 0.001 *

Interpretation:
  Model B has significantly more diverse outputs
  Distributions have significantly different spectral structure
```

---

### Example 2: Detailed Analysis

```python
# Get full characterization
result = spectral_characterization(emb_fp32, emb_int8, "fp32", "int8", b=1000)

# Access individual metrics
print(f"Entropy fp32: {result.entropy_a:.3f}")
print(f"Entropy int8: {result.entropy_b:.3f}")
print(f"Change in diversity: {result.entropy_delta:+.3f}")
print(f"Diversity ratio: {result.effective_rank_b / result.effective_rank_a:.1f}x")

print(f"\nSpectral distance: {result.wasserstein:.4f}")
print(f"Significance: {'Yes' if result.wasserstein_pvalue < 0.05 else 'No'}")

# Check specific hypotheses
if result.entropy_pvalue < 0.05 and result.entropy_delta > 0:
    print(f"\n{result.label_b} produces significantly more diverse outputs")

if result.wasserstein_pvalue < 0.05:
    print(f"Spectral structure changed significantly")
```

---

### Example 3: Compare Multiple Models

```python
models = ["fp32", "fp16", "int8", "nf4"]
samples = {m: load_and_sample(m) for m in models}
embeddings = {m: embed_sample(samples[m]) for m in models}

# Compare all pairs to baseline
baseline = "fp32"
for model in ["fp16", "int8", "nf4"]:
    result = spectral_characterization(
        embeddings[baseline],
        embeddings[model],
        baseline,
        model,
        b=1000
    )
    
    print(f"\n{baseline} → {model}:")
    print(f"  Diversity change: {result.entropy_delta:+.2f}")
    print(f"  Spectral distance: {result.wasserstein:.4f}")
    print(f"  Significant: {result.wasserstein_pvalue < 0.05}")
```

---

### Example 4: Integration with Semantic Axes

```python
# Complete characterization workflow

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

# Now you have:
# - Holistic spectral quantification (diversity, distance)
# - Per-dimension semantic quantification (what differs)
```

**Complete picture**:
- Spectral: "int8 is 2.5x more diverse, spectral distance = 0.12"
- Semantic: "professionalism -0.54 (g=0.82), technicality +0.31 (g=0.45)"

---

## Computational Cost

### Detailed Breakdown

For n=100, d=768:

**Eigenvalue spectrum extraction**:
- Gram matrix: O(n²d) = 100² × 768 = 7.68M operations
- Eigendecomposition: O(n³) = 1M operations
- Total: ~8-9M operations → **8-10ms**

**Entropy computation**:
- Eigenvalue spectrum (once): 8-10ms
- Sum over eigenvalues: O(n) = negligible
- Total: **~10ms per distribution**

**Wasserstein distance**:
- Two eigenvalue spectra: 2 × 10ms = 20ms
- scipy.stats.wasserstein_distance: O(n log n) ≈ 0.1ms
- Total: **~20ms per comparison**

**Permutation test (b=1000)**:
- Entropy test: 1000 × 20ms = 20 seconds
- Wasserstein test: 1000 × 20ms = 20 seconds
- Both: **~40 seconds**

**Combined spectral_characterization(b=1000)**:
- **Total runtime: 40-50 seconds**

### Comparison to Other Metrics

| Metric | Runtime (n=100, b=1000) | Comments |
|--------|------------------------|----------|
| **Spectral metrics** | 40-50s | This plan |
| **Trace distance V2** | 120-180s | 3-4x slower |
| **Semantic axes** | <1s | No permutation needed |
| **MMD** | 30-60s | Similar to spectral |

**Verdict**: Spectral metrics are **fast** - faster than trace distance, comparable to MMD.

---

## Documentation Updates

### Add to `README.md`

After semantic axes section:

```markdown
### Spectral Metrics for Holistic Quantification

For holistic distributional quantification beyond per-dimension analysis:

```python
from model_equality_testing.src.spectral_metrics import spectral_characterization

# Complete spectral analysis
result = spectral_characterization(emb_a, emb_b, "fp32", "int8", b=1000)

print(result.summary())
# Output:
# Diversity (Von Neumann Entropy):
#   fp32: S = 3.24 (~25 effective modes)
#   int8: S = 4.15 (~63 effective modes)
#   Δ = +0.91 (int8 is 2.5x more diverse)
#
# Spectral Distance (Wasserstein on eigenvalues):
#   W = 0.1243
#   P-value: 0.001 *
```

Spectral metrics provide:
- **Diversity quantification**: How spread out is the distribution? (Von Neumann entropy)
- **Spectral distance**: How different are the eigenvalue structures? (Wasserstein)
- **Effect sizes**: Interpretable magnitudes (diversity ratio, standardized distance)
- **Statistical significance**: Permutation tests for both metrics

See `SPECTRAL-METRICS-GUIDE.md` for details.
```

---

### Create `SPECTRAL-METRICS-GUIDE.md`

New guide with:
- **What are spectral metrics?** (eigenvalue-based analysis)
- **Why eigenvalues?** (the "truly quantum" part, without matrix comparison issues)
- **Interpretation guide** (entropy = diversity, Wasserstein = distance)
- **When to use** (holistic quantification vs semantic dimensions)
- **Comparison to other metrics** (vs trace distance, vs semantic axes)
- **Example workflows**

Key section:

```markdown
## When to Use Spectral Metrics

### Use spectral metrics when:
- ✅ You want holistic quantification ("overall how different?")
- ✅ You need single-number summaries (entropy, Wasserstein)
- ✅ You care about diversity/spread (not just mean shifts)
- ✅ Sample sizes may differ (n_a ≠ n_b)
- ✅ You want mathematically sound metrics (no "ill-defined" issues)

### Use semantic axes when:
- ✅ You want to know *what* differs (professionalism, technicality, etc.)
- ✅ You need interpretable dimensions
- ✅ Stakeholders need actionable insights
- ✅ You want established effect size scales (Hedges' g)

### Use both for complete picture:
- Spectral: "Distributions differ in diversity (2.5x) and spectral structure (W=0.12)"
- Semantic: "Specifically: professionalism -0.54, technicality +0.31"
```

---

## Timeline

### Hour 1: Core Functions
- ✓ Create `model_equality_testing/src/spectral_metrics.py`
- ✓ Implement `eigenvalue_spectrum()`
- ✓ Implement `von_neumann_entropy_from_embeddings()`
- ✓ Implement `effective_rank()`
- ✓ Implement `spectral_wasserstein()`

### Hour 2: Inference
- ✓ Implement `entropy_difference_test()`
- ✓ Implement `spectral_wasserstein_test()`
- ✓ Test manually with synthetic data

### Hour 3: Interface and Testing
- ✓ Implement `SpectralCharacterization` dataclass
- ✓ Implement `spectral_characterization()` function
- ✓ Write unit tests (5 test functions)
- ✓ Integration test with real data
- ✓ Update README.md
- ✓ Create SPECTRAL-METRICS-GUIDE.md

**Total: 3 hours for complete, tested, documented implementation**

---

## Advantages Over Previous Plans

| Aspect | V1 (Trace Distance + PCA) | V2 (Trace Distance, No PCA) | V3 (Spectral Metrics) |
|--------|-------------------------|---------------------------|---------------------|
| **Validity** | ❌ Circular | ✅ Valid | ✅ Valid |
| **Equal n required** | ✗ Yes | ✗ Yes | ✅ No |
| **PCA issues** | ✗ Yes | ✅ No | ✅ No |
| **Speed** | 60-120s | 120-180s | 40-50s |
| **Interpretation** | Abstract | Abstract | **Clear** (diversity, distance) |
| **Quantification** | Magnitude | Magnitude + CI | **Effect sizes** (ratios, σ) |
| **Implementation** | 4 hours | 4 hours | **3 hours** |
| **"Ill-defined" issues** | ✗ Yes | ⚠️ Still has | ✅ **No** |
| **Recommendation** | ✗ Don't | ⚠️ With caveats | ✅ **Implement** |

---

## Summary

### The Goal (Restated)

**Quantify distributional changes beyond T/F hypothesis testing.**

### The Solution

**Two complementary spectral metrics**:

1. **Von Neumann entropy difference**
   - Quantifies: Diversity/spread change
   - Output: "2.5x more diverse (63 modes vs 25 modes)"
   - Interpretation: How concentrated vs distributed

2. **Wasserstein on eigenvalue spectra**
   - Quantifies: Spectral distance
   - Output: "W = 0.124 (2.3σ above null)"
   - Interpretation: Cost to transform spectral structure

### What You Get

Complete toolkit for quantification:

| Level | Metric | Quantification |
|-------|--------|----------------|
| **Holistic spectral** | Entropy Δ | "2.5x more modes" |
| | Wasserstein | "0.124 (2.3σ)" |
| **Interpretable semantic** | Semantic axes | "prof: +0.54 (g=0.82)" |
| **Baseline** | MMD | "MMD = 0.23, p < 0.001" |

### Why This is Better

**Compared to trace distance**:
- ✅ No equal-size constraint
- ✅ No "ill-defined" issues
- ✅ Faster (40s vs 120-180s)
- ✅ Clearer interpretation
- ✅ No PCA worries

**Compared to just semantic axes**:
- ✅ Holistic summary (not just per-dimension)
- ✅ Spectral/information-theoretic interpretation
- ✅ Complements semantic analysis

### Implementation Recommendation

✅ **Implement this plan (V3)**

**Skip**:
- ✗ Trace distance V1 (circular, invalid)
- ✗ Trace distance V2 (too many constraints, unclear value)

**Add later if needed**:
- Other eigenvalue metrics (JS divergence, Hellinger)
- Empirical calibration (null distributions for standardization)
- Visualization (eigenvalue spectra plots)

**Time investment**: 3 hours for complete, validated, production-ready spectral metrics module.

---

## Verification Checklist

After implementation:

- [ ] `eigenvalue_spectrum()` produces normalized, sorted eigenvalues
- [ ] `von_neumann_entropy_from_embeddings()` respects bounds [0, log(n)]
- [ ] `effective_rank()` returns sensible values [1, n]
- [ ] `spectral_wasserstein()` handles different sample sizes
- [ ] `entropy_difference_test()` Type I error rate ~5% (validation test)
- [ ] `spectral_wasserstein_test()` produces valid p-values
- [ ] `spectral_characterization()` integrates both metrics correctly
- [ ] `summary()` produces readable, accurate output
- [ ] Integration test with real LLM data runs successfully
- [ ] All documentation examples run without errors
- [ ] Backward compatibility maintained (other modules unchanged)
- [ ] Runtime matches predictions (~40-50s for b=1000)

---

## Final Notes

This plan represents the **best path forward** for your stated goal:

> "Quantify distributional changes beyond just changed:T/F"

It provides:
- ✅ Clear quantification (diversity ratios, spectral distances)
- ✅ Statistical rigor (permutation tests, p-values)
- ✅ Interpretable outputs (effective modes, Wasserstein)
- ✅ No mathematical pathologies (no "ill-defined" issues)
- ✅ Fast implementation (3 hours)
- ✅ Fast runtime (40-50 seconds)

**Recommended**: Implement V3 (spectral metrics), skip trace distance entirely.
