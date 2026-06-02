# Density Matrix Metrics: How They Work

## Overview

This guide explains how to compute and interpret metrics from two density matrices. After creating `rho_a` and `rho_b` (see `CREATING-DENSITY-MATRICES.md`), use these metrics to quantify differences between LLM distributions.

## The Three Core Metrics

You have: `rho_a` (100×100) and `rho_b` (100×100)

### 1. Trace Distance

**What it computes**:
```python
from model_equality_testing.src.quantum_metrics import trace_distance

td = trace_distance(rho_a, rho_b)
# Range: [0, 1]
```

**The math**:
```
D(ρ_A, ρ_B) = (1/2) * ||ρ_A - ρ_B||₁

where ||M||₁ = trace(√(M† M)) = sum of singular values
```

**Step-by-step**:
1. Compute difference: `Delta = rho_a - rho_b`
2. Compute `M = Delta.T @ Delta` (or `Delta @ Delta.T`)
3. Find eigenvalues of M: `λ₁, λ₂, ..., λₙ`
4. Sum of singular values = `Σ sqrt(λᵢ)`
5. Multiply by 1/2

**What it means**:
- **0**: Identical distributions (ρ_A = ρ_B)
- **1**: Completely distinguishable (orthogonal states)
- **0.23**: Moderate difference (23% distinguishability)

**Interpretation**: Maximum probability advantage of correctly distinguishing the two distributions with an optimal measurement strategy.

---

### 2. Von Neumann Entropy

**What it computes**:
```python
from model_equality_testing.src.quantum_metrics import von_neumann_entropy

s_a = von_neumann_entropy(rho_a)
s_b = von_neumann_entropy(rho_b)
# Range: [0, log(n)] where n=100 → max ≈ 4.6
```

**The math**:
```
S(ρ) = -trace(ρ log ρ) = -Σ λᵢ log(λᵢ)

where λᵢ are eigenvalues of ρ
```

**Step-by-step**:
1. Compute eigendecomposition: `ρ = Q Λ Q^T`
2. Extract eigenvalues: `λ₁, λ₂, ..., λₙ` (all ≥ 0, sum to 1)
3. For each eigenvalue: compute `-λᵢ log(λᵢ)` (with convention `0 log 0 = 0`)
4. Sum them: `S = -Σ λᵢ log λᵢ`

**What it means**:
- **S = 0**: All samples identical (rank-1 density matrix, one eigenvalue = 1)
- **S = log(n) ≈ 4.6**: Maximally diverse (uniform, all eigenvalues = 1/n)
- **S = 3.2**: Moderate diversity (distribution has ~25 effective modes)

**Interpretation**: 
- Uncertainty/diversity in the distribution
- Effective dimensionality: `exp(S)` ≈ number of distinct modes
- Higher entropy = more spread out, lower entropy = more concentrated

**Compare two samples**:
```python
delta_entropy = s_b - s_a
# If delta > 0: Sample B is more diverse than A
# If delta < 0: Sample A is more diverse than B
```

---

### 3. Quantum Relative Entropy (KL Divergence)

**What it computes**:
```python
from model_equality_testing.src.quantum_metrics import quantum_relative_entropy

kl_ab = quantum_relative_entropy(rho_a, rho_b)  # KL(A || B)
kl_ba = quantum_relative_entropy(rho_b, rho_a)  # KL(B || A)
# Range: [0, ∞)
# Usually in [0, 2] for typical cases
```

**The math**:
```
S(ρ_A || ρ_B) = trace(ρ_A (log ρ_A - log ρ_B))
              = trace(ρ_A log ρ_A) - trace(ρ_A log ρ_B)
```

**Step-by-step**:
1. Eigendecompose both: `ρ_A = Q_A Λ_A Q_A^T`, `ρ_B = Q_B Λ_B Q_B^T`
2. Compute matrix logarithms: `log ρ_A = Q_A log(Λ_A) Q_A^T`
3. Compute `ρ_A log ρ_A` and `ρ_A log ρ_B`
4. Take traces: `S(ρ_A || ρ_B) = trace(ρ_A log ρ_A) - trace(ρ_A log ρ_B)`

**What it means**:
- **S(A||B) = 0**: A and B are identical
- **S(A||B) > 0**: Information lost when using B to approximate A
- **Asymmetric**: `S(A||B) ≠ S(B||A)` in general

**Interpretation**:
- How much A diverges from B
- If using distribution B but true distribution is A, you lose `S(A||B)` bits of information
- Larger value = more divergence

**Direction matters**:
```python
# KL(A || B): How well does B approximate A?
kl_ab = quantum_relative_entropy(rho_a, rho_b)

# KL(B || A): How well does A approximate B?
kl_ba = quantum_relative_entropy(rho_b, rho_a)

# They're different!
```

---

## Complete Example

```python
import numpy as np
from model_equality_testing.src.quantum_metrics import (
    trace_distance,
    von_neumann_entropy,
    quantum_relative_entropy
)

# You have two density matrices
# rho_a: (100, 100) from LLM A
# rho_b: (100, 100) from LLM B

# 1. Trace distance (symmetric)
td = trace_distance(rho_a, rho_b)
print(f"Trace distance: {td:.4f}")
# e.g., 0.2341 → 23% distinguishability

# 2. Von Neumann entropy (individual)
s_a = von_neumann_entropy(rho_a)
s_b = von_neumann_entropy(rho_b)
print(f"Entropy A: {s_a:.4f}")  # e.g., 3.82
print(f"Entropy B: {s_b:.4f}")  # e.g., 4.15
print(f"Δ Entropy: {s_b - s_a:.4f}")  # +0.33 → B more diverse

# 3. Quantum relative entropy (asymmetric)
kl_ab = quantum_relative_entropy(rho_a, rho_b)
kl_ba = quantum_relative_entropy(rho_b, rho_a)
print(f"KL(A||B): {kl_ab:.4f}")  # e.g., 0.156
print(f"KL(B||A): {kl_ba:.4f}")  # e.g., 0.142
print(f"Symmetric KL: {(kl_ab + kl_ba) / 2:.4f}")  # Jensen-Shannon-like
```

**Example output interpretation**:
- Trace distance = 0.23: Distributions are moderately distinguishable
- Entropy A = 3.82, B = 4.15: B is more diverse (Δ = +0.33)
- KL asymmetry: Similar in both directions, but B is slightly better at approximating A

---

## What Each Metric Tells You

| Metric | Question Answered | Properties |
|--------|-------------------|------------|
| **Trace Distance** | How distinguishable are they? | Symmetric, bounded [0,1], geometric interpretation |
| **Von Neumann Entropy** | How diverse is each distribution? | Per-distribution, measures spread across modes |
| **Quantum Relative Entropy** | How much does A diverge from B? | Asymmetric, unbounded, information-theoretic |

---

## Relationships Between Metrics

**Pinsker's inequality**:
```
D(ρ_A, ρ_B) ≥ √(S(ρ_A || ρ_B) / 2)
```
Trace distance is lower-bounded by quantum relative entropy.

**Entropy bounds**:
```
S(ρ_A) + S(ρ_B) - 2 S((ρ_A + ρ_B)/2) ≤ S(ρ_A || ρ_B) + S(ρ_B || ρ_A)
```

**Triangle inequality** (trace distance):
```
D(ρ_A, ρ_C) ≤ D(ρ_A, ρ_B) + D(ρ_B, ρ_C)
```

---

## Interpreting Results Together

### Scenario 1: High trace distance, similar entropies
```python
td = 0.85
s_a = 3.2, s_b = 3.3
```
**Interpretation**: Distributions are very different in *where* they concentrate, but have similar diversity levels. They occupy different regions of semantic space but with comparable spread.

### Scenario 2: Low trace distance, different entropies
```python
td = 0.12
s_a = 2.1, s_b = 4.0
```
**Interpretation**: Overall similar distributions, but B is much more diverse (spread across more modes). A is concentrated in fewer dominant modes that overlap with B's spread.

### Scenario 3: Asymmetric KL divergence
```python
kl_ab = 0.05
kl_ba = 0.42
```
**Interpretation**: 
- A approximates B well (low KL(A||B))
- B is a poor approximation of A (high KL(B||A))
- B is more spread out (higher entropy), A is more concentrated
- Using A to predict B loses little information, but using B to predict A loses much

---

## Eigenvalue Interpretation

All three metrics depend on **eigenvalues of the density matrix**:

```python
eigenvalues = np.linalg.eigvalsh(rho_a)  # (100,) array, all ≥ 0, sum = 1
```

**What eigenvalues tell you**:
- Each eigenvalue = probability of a "mode" in the distribution
- Large eigenvalue = dominant mode (many similar samples cluster here)
- Many small eigenvalues = diverse distribution (spread across many modes)

**Effective rank**:
```python
eff_rank = np.exp(von_neumann_entropy(rho_a))
# If eff_rank ≈ 10: distribution has ~10 distinct modes
# If eff_rank ≈ 90: distribution nearly uniform (maximal diversity)
```

**Participation ratio** (alternative diversity measure):
```python
evals = np.linalg.eigvalsh(rho_a)
participation_ratio = 1 / np.sum(evals**2)
# High PR: many modes contribute equally
# Low PR: few modes dominate
```

**Spectrum visualization**:
```python
import matplotlib.pyplot as plt

evals_a = np.sort(np.linalg.eigvalsh(rho_a))[::-1]  # descending
evals_b = np.sort(np.linalg.eigvalsh(rho_b))[::-1]

plt.plot(evals_a, label='A')
plt.plot(evals_b, label='B')
plt.yscale('log')
plt.xlabel('Eigenvalue rank')
plt.ylabel('Eigenvalue (log scale)')
plt.legend()
plt.title('Eigenvalue spectra comparison')
# Steep drop-off = concentrated, slow decay = diverse
```

---

## Practical Workflow

### Step 1: Compute All Metrics
```python
# Distinguishability
td = trace_distance(rho_a, rho_b)

# Diversity
s_a = von_neumann_entropy(rho_a)
s_b = von_neumann_entropy(rho_b)

# Divergence (both directions)
kl_ab = quantum_relative_entropy(rho_a, rho_b)
kl_ba = quantum_relative_entropy(rho_b, rho_a)

# Effective dimensionality
eff_rank_a = np.exp(s_a)
eff_rank_b = np.exp(s_b)
```

### Step 2: Interpret Together
```python
print(f"Trace distance: {td:.4f}")
if td < 0.1:
    print("  → Very similar distributions")
elif td < 0.3:
    print("  → Moderately different")
else:
    print("  → Highly distinguishable")

print(f"\nEntropy: A={s_a:.3f} (eff_rank={eff_rank_a:.1f}), " 
      f"B={s_b:.3f} (eff_rank={eff_rank_b:.1f})")
if abs(s_b - s_a) > 0.5:
    more_diverse = "B" if s_b > s_a else "A"
    print(f"  → Distribution {more_diverse} is notably more diverse")

print(f"\nRelative entropy: KL(A||B)={kl_ab:.4f}, KL(B||A)={kl_ba:.4f}")
if abs(kl_ab - kl_ba) > 0.1:
    print(f"  → Asymmetric: better to use {'A' if kl_ab < kl_ba else 'B'} "
          "to approximate the other")
```

### Step 3: Visualize (Optional)
```python
# Plot eigenvalue spectra
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Spectrum comparison
evals_a = np.sort(np.linalg.eigvalsh(rho_a))[::-1]
evals_b = np.sort(np.linalg.eigvalsh(rho_b))[::-1]
axes[0].plot(evals_a, 'o-', label='A', alpha=0.7)
axes[0].plot(evals_b, 's-', label='B', alpha=0.7)
axes[0].set_yscale('log')
axes[0].set_xlabel('Eigenvalue rank')
axes[0].set_ylabel('Eigenvalue (log scale)')
axes[0].legend()
axes[0].set_title('Eigenvalue spectra')

# Cumulative explained variance
cum_a = np.cumsum(evals_a)
cum_b = np.cumsum(evals_b)
axes[1].plot(cum_a, label='A')
axes[1].plot(cum_b, label='B')
axes[1].axhline(y=0.95, color='k', linestyle='--', alpha=0.3)
axes[1].set_xlabel('Number of modes')
axes[1].set_ylabel('Cumulative eigenvalue sum')
axes[1].legend()
axes[1].set_title('Cumulative variance explained')
axes[1].grid(alpha=0.3)

plt.tight_layout()
```

---

## Common Patterns and What They Mean

### Pattern 1: Steep vs. Flat Spectra
```
Distribution A: λ = [0.6, 0.2, 0.1, 0.05, 0.03, 0.02, ...]  # Steep
Distribution B: λ = [0.15, 0.12, 0.10, 0.09, 0.08, ...]     # Flat
```
- **Steep (A)**: Low entropy, concentrated in few modes, predictable
- **Flat (B)**: High entropy, spread across many modes, diverse

### Pattern 2: Similar Spectra, Different Subspaces
```
td = 0.7  # High
s_a ≈ s_b  # Similar entropy
```
- Distributions have similar diversity but occupy different regions
- Eigenvectors point in different directions (different semantic modes)

### Pattern 3: One Contains the Other
```
kl_ab = 0.05  # A → B is easy
kl_ba = 0.45  # B → A loses information
s_b > s_a     # B more diverse
```
- B is a "spread-out" version of A
- A concentrates where B has mass, but B has additional modes
- Common when comparing base model (diverse) vs. fine-tuned (specialized)

---

## Limitations and Caveats

### Known Issues (from docstring)

> "The NxN density matrices constructed from sample-specific Gram matrices (PIP = E @ E.T) live in different N-dimensional subspaces of R^d for different samples. Cross-matrix metrics (trace distance, QRE) are therefore not well-defined in the quantum-mechanical sense, and trace distance exhibits pathological sample-size dependence."

**What this means**:
- These are **quantum-inspired** metrics, not true quantum mechanics
- Trace distance is not the physical Bures distance
- Requires equal sample sizes (n_a = n_b)
- Should be interpreted as similarity-based distributional metrics

### Sample Size Requirement
```python
# This works
rho_a = pca_density_matrix(emb_a, pca)  # (100, 100)
rho_b = pca_density_matrix(emb_b, pca)  # (100, 100)
td = trace_distance(rho_a, rho_b)  # ✓

# This fails
emb_c = embed_sample(sample_c)  # (150, 768)
rho_c = pca_density_matrix(emb_c, pca)  # (150, 150)
td = trace_distance(rho_a, rho_c)  # ✗ Error: different sizes
```

**For different sample sizes**, use semantic axes instead (they handle n_a ≠ n_b naturally).

### PCA Circular Dependency

PCA is fit on combined data (A + B), creating a circularity:
- PCA finds directions of maximum variance in A+B
- If a semantic dimension has low variance in A+B, PCA discards it
- Later analysis can't detect differences in that dimension

**Mitigation**: Be aware of what PCA might be discarding. For critical dimensions, use semantic axes directly (no PCA).

---

## When to Use Which Metric

**Use trace distance when**:
- You want a single number for "how different are they?"
- You need a symmetric metric (D(A,B) = D(B,A))
- Comparing multiple pairs (A vs B, A vs C, B vs C) and need consistency

**Use von Neumann entropy when**:
- You want to characterize diversity of each distribution independently
- Comparing "concentrated vs. diverse" distributions
- Understanding mode structure (via effective rank)

**Use quantum relative entropy when**:
- You care about directionality (which approximates which better)
- Measuring information loss from using the wrong distribution
- One distribution is a "reference" and the other is being evaluated

**Use all three when**:
- You want a complete characterization
- Preparing results for a paper (show convergence of multiple metrics)
- Different metrics may reveal different aspects of the difference

---

## Relationship to Semantic Axes

**Density matrices** (this guide):
- **Detection**: Quantify *that* distributions differ
- Global, holistic metrics
- Require equal sample sizes
- No interpretable dimensions

**Semantic axes** (see `SEMANTIC-AXES-GUIDE.md`):
- **Interpretation**: Explain *what* differs
- Per-dimension statistics
- Work with any sample sizes
- Interpretable semantic dimensions (professionalism, technicality, etc.)

**Recommended workflow**:
1. **Detect** with density matrices: "Are they different?" (trace distance)
2. **Interpret** with semantic axes: "What differs?" (professionalism +0.54, etc.)

See `CURRENT-CAPABILITIES-SUMMARY.md` for complete overview.

---

## Summary

Given `rho_a` and `rho_b`:

1. **Trace distance** → overall distinguishability (symmetric, 0-1 scale)
2. **Von Neumann entropy** → diversity of each (compare s_a vs s_b)
3. **Quantum relative entropy** → directional divergence (which better approximates which)

All three operate on the **eigenvalue spectra** of the density matrices.

**Quick reference**:
```python
# Distinguishability (symmetric)
td = trace_distance(rho_a, rho_b)  # [0, 1]

# Diversity (per-distribution)
s_a = von_neumann_entropy(rho_a)   # [0, log(n)]
s_b = von_neumann_entropy(rho_b)

# Divergence (asymmetric)
kl = quantum_relative_entropy(rho_a, rho_b)  # [0, ∞)

# Effective modes
eff_rank = np.exp(von_neumann_entropy(rho_a))
```

---

## References

**Implementation**:
- `model_equality_testing/src/quantum_metrics.py` - All density matrix functions

**Documentation**:
- `CREATING-DENSITY-MATRICES.md` - How to build density matrices from samples
- `QUANTUM-METRICS-ADVANTAGES.md` - Honest assessment of advantages/limitations
- `QUANTUM-VS-SEMANTIC-INTERPRETATION.md` - What quantum metrics provide vs. semantic axes
- `CURRENT-CAPABILITIES-SUMMARY.md` - Complete overview of all tools
