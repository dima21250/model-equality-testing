# Quantum-Inspired Metrics: A Critical Assessment

## Overview

This document critically assesses trace distance and other quantum-inspired metrics for comparing LLM text distributions. It provides honest guidance on what works, what's problematic, and what alternatives exist.

---

## The Fundamental Problem

**ALL** cross-matrix quantum metrics suffer from the **same pathology**:

They're defined on n×n density matrices, which in this context are sample-specific Gram matrices. This creates:
1. Sample-size dependence (can't compare different n)
2. "Not well-defined in quantum-mechanical sense" (from `quantum_metrics.py` docstring)
3. Unclear sampling distributions

**This affects**:
- ✗ Trace distance
- ✗ Quantum fidelity
- ✗ Quantum relative entropy (requires same n)
- ✗ Any metric that compares two density matrices directly

---

## Metrics Already Implemented

From `quantum_metrics.py`:

### 1. Von Neumann Entropy (Single-Distribution) ✅

```python
S(ρ) = -Tr(ρ log ρ) = -Σ λᵢ log λᵢ
```

**Formula**:
```python
from model_equality_testing.src.quantum_metrics import von_neumann_entropy

s_a = von_neumann_entropy(rho_a)
s_b = von_neumann_entropy(rho_b)
```

**Advantages**:
- ✅ Well-defined (single matrix, any size)
- ✅ Clear interpretation (diversity, uncertainty)
- ✅ No equal-size requirement
- ✅ Easy to add inference (compare S(ρ_A) vs S(ρ_B))
- ✅ No PCA circularity issues

**How to use for comparison**:
```python
s_a = von_neumann_entropy(rho_a)
s_b = von_neumann_entropy(rho_b)
delta_entropy = s_b - s_a

# Interpret:
# delta > 0: Distribution B is more diverse than A
# delta < 0: Distribution A is more diverse than B
# |delta| large: Substantial difference in diversity
```

**Inference approach**:
```python
def von_neumann_entropy_difference_test(emb_a, emb_b, b=1000):
    """
    Test if entropies differ significantly.
    
    H0: S(ρ_A) = S(ρ_B)
    H1: S(ρ_A) ≠ S(ρ_B)
    
    Advantages:
    - Works for different sample sizes (no n_a = n_b requirement)
    - No PCA circularity
    - Clear interpretation
    """
    from model_equality_testing.src.quantum_inference import density_matrix
    
    # Observed difference
    s_a = von_neumann_entropy(density_matrix(emb_a))
    s_b = von_neumann_entropy(density_matrix(emb_b))
    delta_obs = abs(s_b - s_a)
    
    # Permutation null distribution
    n_a = len(emb_a)
    combined = np.vstack([emb_a, emb_b])
    
    deltas_null = []
    for _ in range(b):
        perm = np.random.permutation(len(combined))
        a_perm = combined[perm[:n_a]]
        b_perm = combined[perm[n_a:]]
        
        s_a_perm = von_neumann_entropy(density_matrix(a_perm))
        s_b_perm = von_neumann_entropy(density_matrix(b_perm))
        deltas_null.append(abs(s_b_perm - s_a_perm))
    
    # P-value
    pvalue = (np.sum(np.array(deltas_null) >= delta_obs) + 1) / (b + 1)
    
    return pvalue, s_a, s_b
```

**Usage example**:
```python
pval, s_a, s_b = von_neumann_entropy_difference_test(emb_a, emb_b)

print(f"Entropy A: {s_a:.3f}")
print(f"Entropy B: {s_b:.3f}")
print(f"Difference: {abs(s_b - s_a):.3f}")
print(f"P-value: {pval:.4f}")

if pval < 0.05:
    if s_b > s_a:
        print("B is significantly more diverse than A")
    else:
        print("A is significantly more diverse than B")
```

**Advantages over trace distance**:
- ✅ Works for different sample sizes (no n_a = n_b requirement)
- ✅ No PCA circularity issue
- ✅ Clear interpretation: "Distribution B is more diverse than A"
- ✅ Fast to compute (single eigendecomposition per distribution)

**Disadvantage**:
- Only measures diversity difference, not "distance"

**Recommended**: ✅ Add inference for this (easy win, 30 minutes)

---

### 2. Quantum Relative Entropy ⚠️

```python
S(ρ_A || ρ_B) = Tr(ρ_A log ρ_A) - Tr(ρ_A log ρ_B)
```

**Problem**: Same as trace distance - requires equal n, same pathologies.

**Recommendation**: ✗ Skip this for inference (same issues as trace distance)

---

### 3. Trace Distance ⚠️

```python
D(ρ_A, ρ_B) = (1/2) ||ρ_A - ρ_B||₁
```

**Issues**:
- Requires equal sample sizes (n_a = n_b)
- PCA circularity in V1
- "Not well-defined in quantum-mechanical sense" (docstring)
- Sample-size dependence

**Recommendation**: 
- ⚠️ Use V2 (no PCA) if you must, with caveats
- ✗ Don't use V1 (PCA circularity breaks inference)
- Consider alternatives below

---

## Alternative: Eigenvalue-Based Metrics

### The Core Idea

Instead of comparing n×n density matrices, compare their **eigenvalue distributions**.

**Why this helps**:
- Eigenvalues are the "quantum probabilities" (the truly quantum part)
- Can compare eigenvalue distributions of different sizes
- Standard distributional metrics apply
- No PCA circularity issues

### Extracting Eigenvalue Spectrum

```python
def eigenvalue_spectrum(embeddings):
    """
    Get eigenvalue distribution from density matrix.
    
    Args:
        embeddings: (n, d) SBERT embeddings
    
    Returns:
        evals: (k,) array of positive eigenvalues, normalized to sum to 1
    """
    from model_equality_testing.src.quantum_inference import density_matrix
    
    rho = density_matrix(embeddings)
    evals = np.linalg.eigvalsh(rho)  # (n,) array, sorted ascending
    
    # Keep only positive eigenvalues (remove numerical zeros)
    evals = evals[evals > 1e-10]
    
    # Normalize (should already sum to 1, but ensure it)
    evals = evals / evals.sum()
    
    return evals
```

---

### Metric 1: Jensen-Shannon Divergence on Eigenvalues ✅

```python
def quantum_js_divergence(emb_a, emb_b):
    """
    Jensen-Shannon divergence on eigenvalue spectra.
    
    JSD(P||Q) = [KL(P||M) + KL(Q||M)] / 2
    where M = (P + Q) / 2
    
    Advantages:
    - Symmetric
    - Bounded [0, 1] (or [0, log(2)] if using log base 2)
    - Works for different sample sizes
    - Information-theoretic interpretation
    """
    evals_a = eigenvalue_spectrum(emb_a)
    evals_b = eigenvalue_spectrum(emb_b)
    
    # Pad to same length with zeros
    n_max = max(len(evals_a), len(evals_b))
    p = np.pad(evals_a, (0, n_max - len(evals_a)))
    q = np.pad(evals_b, (0, n_max - len(evals_b)))
    
    # Mixture distribution
    m = (p + q) / 2
    
    # JS divergence
    js = (kl_divergence(p, m) + kl_divergence(q, m)) / 2
    
    # Convert to [0, 1] scale (divide by log(2))
    return js / np.log(2)

def kl_divergence(p, q):
    """
    KL divergence with numerical stability.
    
    KL(P||Q) = Σ p_i log(p_i / q_i)
    """
    # Only compute where both are positive
    mask = (p > 1e-10) & (q > 1e-10)
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))
```

**Advantages**:
- ✅ Symmetric: JSD(A, B) = JSD(B, A)
- ✅ Works for different n (pad with zeros)
- ✅ Bounded [0, 1]
- ✅ Clear interpretation (information divergence)
- ✅ No PCA needed (works on eigenvalues)
- ✅ Metric properties (square root is a true metric)

**Usage**:
```python
js = quantum_js_divergence(emb_a, emb_b)
print(f"JS divergence: {js:.4f}")

# Interpretation:
# 0 = identical eigenvalue distributions
# 1 = maximally different
# ~0.5 = moderate difference
```

**Recommended**: ✅ Good alternative to trace distance

---

### Metric 2: Wasserstein Distance on Eigenvalues ✅

```python
from scipy.stats import wasserstein_distance

def quantum_wasserstein(emb_a, emb_b):
    """
    Wasserstein distance (Earth Mover's Distance) on eigenvalue spectra.
    
    Measures the minimum "work" to transform one eigenvalue distribution
    into another.
    
    Advantages:
    - Handles different-length distributions naturally (no padding)
    - True metric (triangle inequality)
    - Interpretable as "work to transform one to other"
    - No numerical issues with zeros (unlike KL)
    """
    evals_a = eigenvalue_spectrum(emb_a)
    evals_b = eigenvalue_spectrum(emb_b)
    
    # Wasserstein handles different lengths natively
    return wasserstein_distance(evals_a, evals_b)
```

**Advantages**:
- ✅ Works for any sample sizes (no padding needed)
- ✅ True metric (satisfies triangle inequality)
- ✅ No numerical issues (unlike KL with zeros)
- ✅ Established in ML literature
- ✅ Clear geometric interpretation

**Usage**:
```python
w = quantum_wasserstein(emb_a, emb_b)
print(f"Wasserstein distance: {w:.4f}")

# Interpretation:
# Small values = similar eigenvalue distributions
# Large values = different distributions
# Not bounded above, but typically in [0, 0.5] range
```

**Recommended**: ✅ Best alternative to trace distance (simplest, most robust)

---

### Metric 3: Hellinger Distance on Eigenvalues ✅

```python
def quantum_hellinger(emb_a, emb_b):
    """
    Hellinger distance on eigenvalue spectra.
    
    H(P, Q) = sqrt(Σ (√p_i - √q_i)²) / √2
    
    Advantages:
    - Bounded [0, 1]
    - Symmetric
    - Related to Bhattacharyya distance
    - Numerically stable (square roots, not logs)
    """
    evals_a = eigenvalue_spectrum(emb_a)
    evals_b = eigenvalue_spectrum(emb_b)
    
    # Pad to same length
    n_max = max(len(evals_a), len(evals_b))
    p = np.pad(evals_a, (0, n_max - len(evals_a)))
    q = np.pad(evals_b, (0, n_max - len(evals_b)))
    
    # Hellinger distance
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q))**2)) / np.sqrt(2)
```

**Advantages**:
- ✅ Bounded [0, 1]
- ✅ Symmetric
- ✅ Numerically stable (square roots, not logs)
- ✅ Related to fidelity in quantum information

**Usage**:
```python
h = quantum_hellinger(emb_a, emb_b)
print(f"Hellinger distance: {h:.4f}")

# Interpretation:
# 0 = identical distributions
# 1 = completely different (disjoint support)
```

**Recommended**: ✅ Good if you want bounded [0, 1] metric

---

## Comparison Table

| Metric | Equal n? | PCA Issue? | Interpretation | Bounded? | Symmetric? | Recommended? |
|--------|----------|-----------|----------------|----------|------------|--------------|
| **Trace distance** | ✗ Required | ⚠️ Yes (V1) | Distinguishability | [0,1] | ✅ | ⚠️ Only V2, with caveats |
| **Von Neumann entropy Δ** | ✅ Any | ✅ No | Diversity difference | ℝ | ✗ | ✅ **Yes** (easy) |
| **QRE** | ✗ Required | ⚠️ Yes | Divergence | [0,∞) | ✗ | ✗ No |
| **JS divergence (evals)** | ✅ Any | ✅ No | Info divergence | [0,1] | ✅ | ✅ **Yes** |
| **Wasserstein (evals)** | ✅ Any | ✅ No | Earth mover | [0,∞) | ✅ | ✅ **Yes** (best) |
| **Hellinger (evals)** | ✅ Any | ✅ No | Distance | [0,1] | ✅ | ✅ Yes |

---

## What You Really Need

### Already Have and Should Use:

**1. MMD (from your paper)**
- ✅ Statistically rigorous
- ✅ Well-studied in ML literature
- ✅ Works for any sample sizes
- ✅ Already implemented and tested
- ✅ Published benchmarks exist
- ✅ Permutation tests already working

**2. Semantic Axes**
- ✅ Interpretable dimensions
- ✅ Statistical inference built-in (Welch's t-test, Hedges' g)
- ✅ Effect size calibration (established scale)
- ✅ Fast (milliseconds)
- ✅ Works for any sample sizes
- ✅ Multiple testing correction (BH-FDR)

**These two alone are sufficient for rigorous LLM comparison.**

---

### Could Add (Eigenvalue-Based):

**3. Wasserstein on Eigenvalue Spectra** ⭐ (Best alternative)
- ✅ Quantum-inspired but tractable
- ✅ Works for any sample sizes
- ✅ No PCA circularity
- ✅ Easy to implement (scipy.stats has it)
- ✅ Clear interpretation
- ✅ Robust to numerical issues

**4. Von Neumann Entropy Difference** ⭐ (Easiest to add)
- ✅ Already have the entropy function
- ✅ Just need permutation test wrapper
- ✅ Clear interpretation ("B is more diverse")
- ✅ Works for any sample sizes
- ✅ Complements trace distance

---

## Recommendations by Priority

### Priority 1: Add Von Neumann Entropy Inference (30 minutes)

**Why**:
- Already implemented (just need test wrapper)
- No equal-size requirement
- Clear interpretation
- Complements existing metrics

**Implementation**:
Add `von_neumann_entropy_difference_test()` to `quantum_inference.py` (see code above).

**Value**: High (easy win, fills gap in current metrics)

---

### Priority 2: Add Wasserstein on Eigenvalues (1-2 hours)

**Why**:
- Best eigenvalue-based metric (robust, no constraints)
- Easy to implement (scipy.stats.wasserstein_distance)
- Quantum-inspired but sound
- Works for any n

**Implementation**:
```python
def quantum_wasserstein_test(emb_a, emb_b, b=1000):
    """Permutation test for Wasserstein distance on eigenvalue spectra."""
    # Observed
    w_obs = quantum_wasserstein(emb_a, emb_b)
    
    # Permutation null
    combined = np.vstack([emb_a, emb_b])
    n_a = len(emb_a)
    
    w_null = []
    for _ in range(b):
        perm = np.random.permutation(len(combined))
        a_perm = combined[perm[:n_a]]
        b_perm = combined[perm[n_a:]]
        w_null.append(quantum_wasserstein(a_perm, b_perm))
    
    pvalue = (np.sum(np.array(w_null) >= w_obs) + 1) / (b + 1)
    return pvalue, w_obs
```

**Value**: High (best trace distance alternative)

---

### Priority 3: Trace Distance V2 (No PCA) (4 hours)

**Why**:
- Only if reviewers demand it
- Fixes PCA circularity
- Still has other issues (equal n, ill-defined)

**Implementation**: See `INFERENCE-IMPLEMENTATION-PLAN-V2.md`

**Value**: Medium (fixes validity but still constrained)

**Caveats**: Must include massive documentation about limitations

---

### Don't Implement:

**✗ Quantum relative entropy inference** 
- Same issues as trace distance
- Asymmetric (harder to interpret)
- No advantages over trace distance

**✗ Quantum fidelity**
- Even more complex than trace distance
- Same sample-size issues
- Less intuitive interpretation

**✗ Trace distance V1 (with PCA)**
- Circular (biased null distribution)
- Invalid inference
- Don't resurrect this

---

## The Brutal Honest Assessment

### What Are You Really Doing?

You're comparing **LLM text distributions** using **SBERT embeddings**.

You're **not** doing:
- Quantum computing
- Quantum state tomography
- Physical quantum systems

### The "Quantum" Framing Is:

**Mathematically loose**:
- Gram matrices ≠ quantum density matrices
- No Born rule probabilities
- No Hilbert space structure
- "Not well-defined in quantum-mechanical sense" (your own admission)

**Interpretively unclear**:
- What does "entanglement" mean for text?
- What does "superposition" mean for LLM outputs?
- What's the physical analogue?

**Methodologically problematic**:
- Sample-size dependence
- Unclear sampling distributions
- PCA circularity (V1)

### Why Use "Quantum" Metrics At All?

**Legitimate reasons**:
- Information-geometric interpretation (entropy, divergence)
- Spectral decomposition (eigenvalue structure)
- Paper novelty (distinguishes from prior work)

**Questionable reasons**:
- Reviewer appeal ("quantum" sounds impressive)
- Fashion (quantum ML is trendy)
- Lack of better alternatives (but you have MMD + semantic axes!)

### Honest Alternatives to "Quantum" Framing:

Instead of:
- "Quantum-inspired metrics"
- "Density matrix analysis"
- "Quantum relative entropy"

Consider:
- "Information-geometric metrics"
- "Spectral metrics on embedding geometry"
- "Eigenvalue-based distributional comparison"
- "Gram matrix spectral analysis"

**Be honest**: If it's not quantum mechanics, don't call it quantum.

---

## Practical Decision Framework

### Ask Yourself:

**1. Do I need quantum framing for my paper?**
- Yes, reviewers expect it → Use carefully, with caveats
- No, just need good metrics → Use MMD + semantic axes

**2. Do I need holistic distance metrics?**
- Yes → Consider Wasserstein on eigenvalues
- No → Von Neumann entropy difference is enough

**3. Can I accept equal-size constraint?**
- Yes → Trace distance V2 is viable
- No → Use eigenvalue-based metrics

**4. Do I have time for 4-hour implementation?**
- Yes → Trace distance V2 + full validation
- No → Just add entropy difference test (30 min)

---

## Final Recommendations

### Implement (Priority Order):

**1. Von Neumann Entropy Difference Test** (30 minutes)
- ✅ Easy win
- ✅ No new issues
- ✅ Works for any n
- ✅ Clear interpretation
- ✅ Complements existing metrics

Add to `quantum_inference.py`:
```python
def von_neumann_entropy_difference_test(emb_a, emb_b, b=1000, random_seed=None):
    """Test if entropies differ significantly."""
    # Implementation as shown above
```

**2. Wasserstein on Eigenvalue Spectra** (1-2 hours)
- ✅ Quantum-inspired but sound
- ✅ No constraints
- ✅ Best trace distance alternative
- ✅ Easy to implement (scipy.stats)

Add to `quantum_inference.py`:
```python
def quantum_wasserstein_test(emb_a, emb_b, b=1000, random_seed=None):
    """Permutation test for Wasserstein distance on eigenvalue spectra."""
    # Implementation as shown above
```

**3. Trace Distance V2 (No PCA)** (4 hours) - Optional
- ⚠️ Only if reviewers insist
- ⚠️ Still requires equal n
- ⚠️ Still has "ill-defined" issues
- ⚠️ Needs massive caveats

Follow `INFERENCE-IMPLEMENTATION-PLAN-V2.md` if implementing.

---

### Don't Implement:

**✗ Anything with PCA for inference** (circular, invalid)

**✗ Quantum relative entropy inference** (same issues as trace distance, no advantages)

**✗ Quantum fidelity** (more complex, same constraints)

**✗ Complex eigenvalue metrics** (JS divergence, Hellinger) unless you want variety

---

## Summary

### The Core Issue

**All cross-matrix quantum metrics** (trace distance, QRE) have fundamental problems in this application:
- Sample-size dependence
- Admitted to be "not well-defined"
- Unclear sampling distributions

### Best Path Forward

**Option A (Minimal)**:
- Add von Neumann entropy difference test (30 min)
- Stick with MMD + semantic axes for main results
- Done

**Option B (Moderate)**:
- Add entropy difference test (30 min)
- Add Wasserstein on eigenvalues (1-2 hours)
- Best "quantum-inspired" metrics without constraints
- Skip trace distance entirely

**Option C (Maximal)**:
- Add entropy difference test
- Add Wasserstein on eigenvalues
- Add trace distance V2 (no PCA) with caveats
- Most comprehensive, but 6+ hours of work

### My Honest Recommendation

**Do Option A or B. Skip trace distance entirely.**

You already have:
- ✅ MMD (rigorous, published)
- ✅ Semantic axes (interpretable, calibrated)

Adding:
- ✅ Von Neumann entropy (diversity comparison)
- ✅ Wasserstein on eigenvalues (best "quantum" metric)

Gives you a complete, defensible toolkit without the trace distance headaches.

**If reviewers insist on trace distance**: Point them to the docstring that says it's "not well-defined" and suggest Wasserstein on eigenvalues as a better quantum-inspired alternative.

---

## References

**Information geometry**:
- Amari, S. (2016). Information Geometry and Its Applications. Springer.

**Wasserstein distance**:
- Villani, C. (2008). Optimal Transport: Old and New. Springer.
- Common in ML: Arjovsky et al. (2017) "Wasserstein GAN"

**Quantum information (for context)**:
- Nielsen & Chuang (2010). Quantum Computation and Quantum Information.
- Note: These apply to actual quantum systems, not text distributions!

**Your existing work**:
- `model_equality_testing/src/quantum_metrics.py` - current implementation
- Your paper on model equality testing with MMD
- `INFERENCE-IMPLEMENTATION-CRITIQUE.md` - why trace distance V1 fails
