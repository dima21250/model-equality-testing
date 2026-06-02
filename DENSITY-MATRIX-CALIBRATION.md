# Density Matrix Calibration Problem

## Overview

The density matrix metrics (trace distance, von Neumann entropy, quantum relative entropy) provide **numerical values without statistical calibration**. This document explains the calibration problem and solutions.

## The Problem

The metrics give you numbers but lack the statistical context needed for decision-making:

```python
from model_equality_testing.src.quantum_metrics import (
    trace_distance,
    von_neumann_entropy,
    quantum_relative_entropy
)

td = trace_distance(rho_a, rho_b)
# Returns: 0.2341

# But you CANNOT answer:
# - Is 0.23 statistically significant?
# - Is 0.23 a "large" difference or "small" difference?
# - Should I conclude the model changed?
# - What's the probability this is just sampling variation?
```

### What's Missing

**1. Null distribution**
- What would td be if A and B were actually the same distribution?
- What's the expected value under the null hypothesis?
- What's the variance due to finite sampling?

**2. P-values**
- Is this difference statistically significant?
- Could this occur by chance with random sampling?
- What's the probability of Type I error?

**3. Effect size scale**
- No established conventions like Cohen's d:
  - d = 0.2 → small effect
  - d = 0.5 → medium effect
  - d = 0.8 → large effect
- No interpretation guidelines for trace distance values

**4. Decision threshold**
- No guidance on "td > X → conclude distributions differ"
- No power analysis to determine required sample size
- No published benchmarks from prior work

## Why Semantic Axes Don't Have This Problem

Semantic axes include **full statistical inference**:

```python
from model_equality_testing.src.semantic_axes import interpret_difference

interp = interpret_difference(emb_a, emb_b, axes, label_a="A", label_b="B")

# Returns per axis:
# - mean_a, mean_b, delta         (location)
# - std_a, std_b, sem_a, sem_b   (spread and uncertainty)
# - hedges_g                      (standardized effect size)
# - t_pvalue                      (Welch's t-test, no equal variance assumption)
# - corrected_pvalue              (Benjamini-Hochberg FDR correction)
# - wasserstein                   (distributional distance)

print(interp.summary())
# Output:
#   professionalism ← → casualness:  Δ = +0.54  g = 0.82  p < 0.001 *
#   technicality ← → layperson:      Δ = +0.12  g = 0.18  p = 0.35
#
# * = significant after Benjamini-Hochberg correction
```

**What semantic axes provide**:
- ✓ Statistical significance (p-values)
- ✓ Standardized effect sizes (Hedges' g with established interpretation)
- ✓ Multiple testing correction (controls false discovery rate)
- ✓ Uncertainty quantification (standard errors)
- ✓ Clear decision rule (p < 0.05 after FDR correction)

## The Asymmetry

This is a **fundamental difference** between the two frameworks:

| Framework | Descriptive Statistics | Inferential Statistics | Decision Support |
|-----------|----------------------|----------------------|------------------|
| **Quantum Metrics** | ✓ Effect magnitude | ✗ No p-values | ✗ No thresholds |
| **Semantic Axes** | ✓ Effect magnitude | ✓ P-values, effect sizes | ✓ Established rules |

**Quantum metrics are currently descriptive only** - they quantify magnitude but don't support hypothesis testing.

**Semantic axes provide full inference** - they answer "is this significant?" with established statistical rigor.

## Solutions

### Option 1: Permutation Testing (Recommended)

Add p-value calculation following the same pattern as your paper's MMD tests.

**Implementation**:

```python
import numpy as np
from model_equality_testing.src.quantum_metrics import (
    fit_pca,
    pca_density_matrix,
    trace_distance
)

def trace_distance_test(emb_a, emb_b, b=1000, k=50):
    """
    Two-sample test using trace distance with permutation p-value.
    
    Args:
        emb_a: (n_a, d) embeddings from distribution A
        emb_b: (n_b, d) embeddings from distribution B
        b: number of permutations (default 1000)
        k: PCA components (default 50)
    
    Returns:
        pvalue: float - probability of observing this or more extreme
                difference under the null hypothesis
        statistic: float - observed trace distance
    """
    n_a = len(emb_a)
    n_b = len(emb_b)
    
    # Combine and fit PCA (same as original workflow)
    combined = np.vstack([emb_a, emb_b])
    pca = fit_pca(combined, k=k)
    
    # Observed statistic
    rho_a = pca_density_matrix(emb_a, pca)
    rho_b = pca_density_matrix(emb_b, pca)
    td_observed = trace_distance(rho_a, rho_b)
    
    # Permutation null distribution
    td_null = []
    for _ in range(b):
        # Randomly permute combined samples
        perm = np.random.permutation(n_a + n_b)
        emb_a_perm = combined[perm[:n_a]]
        emb_b_perm = combined[perm[n_a:]]
        
        # Compute trace distance under permutation
        rho_a_perm = pca_density_matrix(emb_a_perm, pca)
        rho_b_perm = pca_density_matrix(emb_b_perm, pca)
        td_null.append(trace_distance(rho_a_perm, rho_b_perm))
    
    # P-value: proportion of null statistics >= observed
    # +1 in numerator and denominator for proper probability
    pvalue = (np.sum(np.array(td_null) >= td_observed) + 1) / (b + 1)
    
    return pvalue, td_observed
```

**Usage**:

```python
from model_equality_testing.src.embeddings import embed_sample

# Embed samples
emb_a = embed_sample(sample_a)
emb_b = embed_sample(sample_b)

# Test with permutation p-value
pvalue, td = trace_distance_test(emb_a, emb_b, b=1000)

print(f"Trace distance: {td:.4f}")
print(f"P-value: {pvalue:.4f}")

if pvalue < 0.05:
    print("Distributions are significantly different (p < 0.05)")
else:
    print("No significant difference detected")
```

**Advantages**:
- Directly tests "are these distributions different?"
- No parametric assumptions (distribution-free)
- Consistent with your paper's methodology (same as MMD tests)
- Establishes null distribution empirically

**Computational cost**: 
- b=1000 permutations × density matrix computation
- ~30-60 seconds for n=100, d=768, k=50 on modern hardware
- Expensive but doable for research use

**Disadvantages**:
- Requires equal sample sizes (n_a = n_b)
- Computationally intensive for large b or large n
- PCA fit on combined data creates circularity (same as original issue)

---

### Option 2: Bootstrap Confidence Intervals

Quantify uncertainty in the metric estimate using bootstrap resampling.

**Implementation**:

```python
def trace_distance_with_ci(emb_a, emb_b, n_boot=1000, alpha=0.05, k=50):
    """
    Bootstrap confidence interval for trace distance.
    
    Args:
        emb_a: (n_a, d) embeddings from distribution A
        emb_b: (n_b, d) embeddings from distribution B
        n_boot: number of bootstrap samples (default 1000)
        alpha: significance level (default 0.05 for 95% CI)
        k: PCA components (default 50)
    
    Returns:
        td_observed: float - point estimate of trace distance
        ci: tuple (lower, upper) - confidence interval
    """
    n_a = len(emb_a)
    n_b = len(emb_b)
    
    # Fit PCA on original combined data
    combined = np.vstack([emb_a, emb_b])
    pca = fit_pca(combined, k=k)
    
    # Bootstrap distribution
    td_boot = []
    for _ in range(n_boot):
        # Resample WITH replacement from each distribution
        idx_a = np.random.choice(n_a, size=n_a, replace=True)
        idx_b = np.random.choice(n_b, size=n_b, replace=True)
        
        emb_a_boot = emb_a[idx_a]
        emb_b_boot = emb_b[idx_b]
        
        rho_a_boot = pca_density_matrix(emb_a_boot, pca)
        rho_b_boot = pca_density_matrix(emb_b_boot, pca)
        
        td_boot.append(trace_distance(rho_a_boot, rho_b_boot))
    
    # Observed statistic
    rho_a = pca_density_matrix(emb_a, pca)
    rho_b = pca_density_matrix(emb_b, pca)
    td_obs = trace_distance(rho_a, rho_b)
    
    # Percentile confidence interval
    ci_lower = np.percentile(td_boot, 100 * alpha / 2)
    ci_upper = np.percentile(td_boot, 100 * (1 - alpha / 2))
    
    return td_obs, (ci_lower, ci_upper)
```

**Usage**:

```python
td, (ci_lower, ci_upper) = trace_distance_with_ci(emb_a, emb_b, n_boot=1000)

print(f"Trace distance: {td:.4f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

if ci_lower > 0:
    print("Significantly different from zero (CI excludes 0)")
```

**Interpretation**:
- If confidence interval excludes 0 → significant difference
- Width of CI indicates uncertainty
- Narrower CI → more precise estimate

**Advantages**:
- Quantifies uncertainty directly
- Works with unequal sample sizes
- Easy to interpret (CI for the effect size)

**Disadvantages**:
- Doesn't directly give p-value
- Assumes the bootstrap distribution approximates sampling distribution
- Still computationally intensive

---

### Option 3: Empirical Calibration

Build a reference database from your 1.6M completion dataset to establish baselines.

**Implementation**:

```python
import pickle
from model_equality_testing.dataset import load_distribution
from model_equality_testing.src.embeddings import embed_sample

def build_null_distribution(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    source="fp32",
    n_pairs=100,
    sample_size=100,
    k=50,
    save_path="trace_distance_null.pkl"
):
    """
    Build empirical null distribution: same model, different random samples.
    
    Returns:
        null_distances: list of trace distances under H0
    """
    dist = load_distribution(
        model=model,
        prompt_ids={"wikipedia_en": list(range(100))},  # Use many prompts
        L=500,
        source=source,
        load_in_unicode=True
    )
    
    null_distances = []
    for i in range(n_pairs):
        # Draw two independent samples from SAME distribution
        sample1 = dist.draw_completion_sample(sample_size)
        sample2 = dist.draw_completion_sample(sample_size)
        
        emb1 = embed_sample(sample1)
        emb2 = embed_sample(sample2)
        
        pca = fit_pca(np.vstack([emb1, emb2]), k=k)
        rho1 = pca_density_matrix(emb1, pca)
        rho2 = pca_density_matrix(emb2, pca)
        
        td = trace_distance(rho1, rho2)
        null_distances.append(td)
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i+1}/{n_pairs} pairs")
    
    # Save for future use
    with open(save_path, 'wb') as f:
        pickle.dump({
            'null_distances': null_distances,
            'model': model,
            'source': source,
            'sample_size': sample_size,
            'k': k
        }, f)
    
    return null_distances

def calibrated_trace_distance_test(emb_a, emb_b, null_distances, k=50):
    """
    Test using pre-computed null distribution.
    
    Args:
        emb_a, emb_b: embeddings to compare
        null_distances: list of trace distances from null distribution
        k: PCA components (must match null distribution)
    
    Returns:
        pvalue: empirical p-value
        td: observed trace distance
        effect_size: standardized effect (z-score)
    """
    # Compute observed
    combined = np.vstack([emb_a, emb_b])
    pca = fit_pca(combined, k=k)
    rho_a = pca_density_matrix(emb_a, pca)
    rho_b = pca_density_matrix(emb_b, pca)
    td = trace_distance(rho_a, rho_b)
    
    # Null statistics
    null_mean = np.mean(null_distances)
    null_std = np.std(null_distances)
    
    # P-value (proportion of null >= observed)
    pvalue = np.mean(np.array(null_distances) >= td)
    
    # Standardized effect size (z-score)
    effect_size = (td - null_mean) / null_std if null_std > 0 else np.inf
    
    return pvalue, td, effect_size
```

**Workflow**:

```python
# One-time: build null distribution (takes ~30 minutes)
null_distances = build_null_distribution(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    source="fp32",
    n_pairs=100,
    sample_size=100
)

# Compute statistics
null_mean = np.mean(null_distances)
null_std = np.std(null_distances)
null_95 = np.percentile(null_distances, 95)

print(f"Null distribution: μ = {null_mean:.4f}, σ = {null_std:.4f}")
print(f"95th percentile: {null_95:.4f}")

# For each new comparison:
pvalue, td, z = calibrated_trace_distance_test(emb_a, emb_b, null_distances)

print(f"Trace distance: {td:.4f}")
print(f"P-value: {pvalue:.4f}")
print(f"Effect size (z): {z:.2f}")

if pvalue < 0.05:
    print("Significantly different (p < 0.05)")
```

**Advantages**:
- Once calibrated, very fast to use (no repeated computation)
- Provides both p-value and effect size scale
- Can build separate null distributions for different settings

**Disadvantages**:
- Requires substantial upfront work
- Specific to:
  - Embedding model (SBERT all-mpnet-base-v2)
  - PCA components (k=50)
  - Sample size (n=100)
  - Model and source (fp32 Llama-3-8B)
- Need separate calibrations for different configurations

**Extension**: Build alternative distributions (known differences):

```python
# Known difference: fp32 vs int8
alt_distances = []
for i in range(100):
    sample_fp32 = dist_fp32.draw_completion_sample(100)
    sample_int8 = dist_int8.draw_completion_sample(100)
    # ... compute trace distance
    alt_distances.append(td)

# Now you have power estimates
print(f"Null mean: {np.mean(null_distances):.4f}")
print(f"Alternative mean: {np.mean(alt_distances):.4f}")
print(f"Separation: {(np.mean(alt_distances) - np.mean(null_distances)) / np.std(null_distances):.2f} σ")
```

---

### Option 4: Use Semantic Axes for Inference

**Accept the asymmetry** and use each tool for its strength:

**Quantum metrics**: Descriptive/exploratory
- Magnitude of difference (trace distance)
- Diversity characterization (von Neumann entropy)
- Divergence direction (quantum relative entropy)

**Semantic axes**: Inferential/confirmatory
- Statistical significance (p-values)
- Effect sizes (Hedges' g)
- Interpretable dimensions

**Workflow**:

```python
# 1. Embed once
emb_a = embed_sample(sample_a)
emb_b = embed_sample(sample_b)

# 2. Explore with quantum metrics (descriptive)
pca = fit_pca(np.vstack([emb_a, emb_b]), k=50)
rho_a = pca_density_matrix(emb_a, pca)
rho_b = pca_density_matrix(emb_b, pca)

td = trace_distance(rho_a, rho_b)
s_a = von_neumann_entropy(rho_a)
s_b = von_neumann_entropy(rho_b)

print(f"Exploratory quantum metrics:")
print(f"  Trace distance: {td:.4f}")
print(f"  Entropy A: {s_a:.3f}, B: {s_b:.3f}")
# Interpret magnitude, but don't claim significance

# 3. Test with semantic axes (inferential)
axes = [prof_axis, tech_axis, form_axis]
interp = interpret_difference(emb_a, emb_b, axes)

print(f"\nStatistical inference:")
print(interp.summary())
# Shows: which dimensions differ significantly, effect sizes, p-values
```

**When to use this approach**:
- You want quantum metrics for their holistic view
- But you need statistical rigor for conclusions
- Accept they serve different purposes
- Don't force quantum metrics into inferential role they weren't designed for

---

## Current State of Code

### Quantum Metrics (`quantum_metrics.py`)
- ✓ Compute trace distance, von Neumann entropy, quantum relative entropy
- ✓ Well-documented mathematical formulas
- ✗ **No p-value calculation**
- ✗ **No integration with `algorithm.py`'s testing framework**
- ✗ **No permutation test support**
- ✗ **No confidence intervals**
- ✗ **No effect size calibration**

### Semantic Axes (`semantic_axes.py`)
- ✓ Compute mean projections, variances, standard errors
- ✓ Welch's t-test (p-values)
- ✓ Hedges' g (bias-corrected effect sizes)
- ✓ Benjamini-Hochberg FDR correction (multiple testing)
- ✓ Wasserstein distance
- ✓ Quality metrics (separation ratio, pole statistics)

**The gap is significant.**

---

## Recommendations

### Immediate: Document the Limitation

**Add to user-facing documentation** (README.md, DENSITY-MATRIX-METRICS.md):

> **Important**: Density matrix metrics are **descriptive**, not **inferential**. They quantify the magnitude of difference but do not include statistical significance tests. Trace distance = 0.23 tells you "how different" but not "is this significant?"
>
> For hypothesis testing with p-values, use semantic axes (`interpret_difference()`) or add permutation testing (see DENSITY-MATRIX-CALIBRATION.md).

### Short-term (2-3 hours): Add Permutation Test

1. Implement `trace_distance_test()` as shown in Option 1
2. Add to `quantum_metrics.py` or create new `quantum_inference.py`
3. Add unit tests with synthetic data (known null, known alternative)
4. Document usage in DENSITY-MATRIX-METRICS.md

**Deliverable**: Function that returns (pvalue, statistic) like your paper's MMD tests.

### Medium-term (1 day): Integrate with Testing Framework

Make density matrix metrics work with existing `run_two_sample_test()`:

```python
from model_equality_testing.algorithm import run_two_sample_test

pvalue, statistic = run_two_sample_test(
    sample_a, sample_b,
    stat_type="trace_distance",  # New option
    pvalue_type="permutation_pvalue",
    b=1000
)
```

**Requirements**:
1. Add `"trace_distance"` to `IMPLEMENTED_TESTS` in `registry.py`
2. Add test function to `tests.py` that accepts `CompletionSample` and `_precomputed_embeddings`
3. Update `algorithm.py` to handle embedding-based tests
4. Add to documentation

### Long-term (1 week): Build Reference Database

**Goal**: Establish empirical baselines using your 1.6M completion dataset.

**Tasks**:
1. Run on multiple model/source pairs:
   - Null: same model, same source, different samples
   - Small difference: fp32 vs fp16 (expected: minimal)
   - Medium difference: fp32 vs int8 (expected: detectable)
   - Large difference: fp32 vs watermark (expected: large)

2. Create lookup tables:
   - Null distributions by (model, source, n, k)
   - Effect size benchmarks
   - Power estimates

3. Publish as supplementary material:
   - Include in paper repository
   - Provide calibration functions
   - Document methodology

**Deliverable**: 
- `trace_distance_calibration.pkl` - empirical null/alternative distributions
- `load_calibration()` function for easy use
- Paper section on empirical validation

---

## Impact on Research Claims

### Current State (Without Calibration)

**You CAN claim**:
- "Trace distance between fp32 and int8 is 0.23"
- "int8 has higher von Neumann entropy (4.15 vs 3.82)"
- "Quantum relative entropy shows asymmetry"

**You CANNOT claim**:
- ✗ "Distributions are significantly different" (no p-value)
- ✗ "This is a large effect" (no effect size scale)
- ✗ "Model has changed" (no decision threshold)

### With Permutation Testing

**You CAN additionally claim**:
- ✓ "Trace distance is statistically significant (p < 0.001)"
- ✓ "Distributions differ beyond sampling variation"
- ✓ "Effect is robust (95% CI excludes null baseline)"

### With Empirical Calibration

**You CAN additionally claim**:
- ✓ "Effect size is 2.3 standard deviations above null baseline"
- ✓ "Comparable to known fp32 vs int8 differences"
- ✓ "Power to detect differences of this magnitude is 0.95"

---

## Comparison Table

| Need | Quantum Metrics (Current) | + Permutation Test | + Empirical Calibration | Semantic Axes |
|------|-------------------------|-------------------|----------------------|---------------|
| **Effect magnitude** | ✓ | ✓ | ✓ | ✓ |
| **Statistical significance** | ✗ | ✓ | ✓ | ✓ |
| **Effect size scale** | ✗ | ✗ | ✓ | ✓ (Hedges' g) |
| **Decision threshold** | ✗ | ✓ (p<0.05) | ✓ (benchmarks) | ✓ (p<0.05) |
| **Computational cost** | Low | Medium | Low (after setup) | Low |
| **Setup required** | None | None | High (build ref DB) | Medium (axes) |
| **Interpretability** | Abstract | Abstract | Abstract | High (semantic) |

---

## Summary

**Yes, density matrix metrics have a calibration problem.**

**What's missing**:
- Statistical significance tests
- Effect size interpretation
- Decision thresholds

**Why semantic axes don't have this problem**:
- Built-in p-values (Welch's t-test)
- Established effect size scale (Hedges' g)
- Multiple testing correction (BH-FDR)

**Solutions**:
1. **Permutation testing** (recommended short-term) - 2-3 hours
2. **Bootstrap CI** - quantifies uncertainty
3. **Empirical calibration** (recommended long-term) - 1 week
4. **Use semantic axes** for inference (accept asymmetry)

**Next steps**:
1. Document the limitation (now)
2. Implement permutation test (2-3 hours)
3. Integrate with testing framework (1 day)
4. Build reference database (1 week)

The calibration problem is **solvable** but requires explicit work. Until then, use quantum metrics as descriptive/exploratory tools and semantic axes for statistical inference.
