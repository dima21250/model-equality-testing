# Creating Density Matrices from LLM Samples

## Overview

This guide explains how to create density matrices from LLM text samples and use them for comparison. It clarifies the relationship between embeddings (state vectors) and density matrices, which was the central question explored tonight.

## The Dimensional Structure

### Key Objects and Their Spaces

**Text samples**: 100 LLM-generated completions

**Embeddings (state vectors)**: E ∈ ℝ^(100×768)
- 100 samples, each embedded in 768-dimensional SBERT space
- Lives in **embedding space** (feature space)
- Each row is a single embedding: e_i ∈ ℝ^768

**Density matrix**: ρ ∈ ℝ^(100×100)
- Lives in **sample space**
- Entry ρ[i,j] = similarity between sample i and sample j
- Normalized so trace(ρ) = 1

**Critical insight**: Embeddings (768-D) and density matrices (100×100) live in **different spaces** and cannot be directly combined.

### How They're Related

Density matrices are **built from** embeddings:

```
Embeddings E (100 × 768)
    ↓ PCA compression
E_pca (100 × 50)
    ↓ Gram matrix: G = E_pca @ E_pca.T
G (100 × 100)
    ↓ Normalize: ρ = G / trace(G)
Density matrix ρ (100 × 100)
```

**The transformation**: 768-dimensional semantic structure → pairwise similarities in 100×100 matrix

## Step-by-Step Process

### Setup

Scenario: Compare 100 samples from LLM A vs 100 samples from LLM B

**Requirement**: Both samples must have the same count (n=100) for valid trace distance comparison.

### Step 1: Load Data and Draw Samples

```python
from model_equality_testing.dataset import load_distribution

# Load distributions
dist_a = load_distribution(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    prompt_ids={"wikipedia_en": [0, 1, 2]},
    L=500,
    source="fp32",  # LLM A
    load_in_unicode=True,
)

dist_b = load_distribution(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    prompt_ids={"wikipedia_en": [0, 1, 2]},
    L=500,
    source="int8",  # LLM B
    load_in_unicode=True,
)

# Draw 100 samples from each
sample_a = dist_a.draw_completion_sample(n=100)
sample_b = dist_b.draw_completion_sample(n=100)
```

### Step 2: Embed Samples with SBERT

```python
from model_equality_testing.src.embeddings import embed_sample

# Embed using SBERT (produces 768-dimensional embeddings)
emb_a = embed_sample(sample_a)  # (100, 768)
emb_b = embed_sample(sample_b)  # (100, 768)

# These are your "state vectors" - one per sample
# emb_a[i] is the 768-D embedding of sample i from LLM A
```

### Step 3: Fit PCA on Combined Data

**Critical**: Fit PCA on the combined data so both A and B are in the same reduced space.

```python
from model_equality_testing.src.quantum_metrics import fit_pca
import numpy as np

# Combine both sample sets
combined = np.vstack([emb_a, emb_b])  # (200, 768)

# Fit PCA: 768 dimensions → 50 dimensions
pca = fit_pca(combined, k=50)

# Why combined?
# - So both A and B project into the SAME 50-D space
# - Critical for valid comparison
# - This is the "circular dependency" noted in critiques
```

### Step 4: Build Density Matrices

```python
from model_equality_testing.src.quantum_metrics import pca_density_matrix

# Build density matrices
rho_a = pca_density_matrix(emb_a, pca)  # (100, 100)
rho_b = pca_density_matrix(emb_b, pca)  # (100, 100)
```

**What happens inside `pca_density_matrix`**:

```python
def pca_density_matrix(embeddings, pca):
    """
    Build density matrix from embeddings via PCA compression.
    
    Args:
        embeddings: (n, 768) SBERT embeddings
        pca: Fitted PCA model (768 → 50)
    
    Returns:
        rho: (n, n) density matrix
    """
    # Project to PCA space
    embeddings_pca = pca.transform(embeddings)  # (n, 50)
    
    # Gram matrix (pairwise inner products in PCA space)
    G = embeddings_pca @ embeddings_pca.T  # (n, n)
    # G[i,j] = e_i · e_j (inner product of PCA-compressed embeddings)
    
    # Normalize to density matrix
    rho = G / np.trace(G)  # trace(ρ) = 1
    
    return rho
```

### Step 5: Compute Quantum Metrics

```python
from model_equality_testing.src.quantum_metrics import (
    trace_distance,
    von_neumann_entropy,
    quantum_relative_entropy
)

# Trace distance (distinguishability)
td = trace_distance(rho_a, rho_b)
print(f"Trace distance: {td:.4f}")
# Range: [0, 1]
# 0 = identical, 1 = completely different

# Von Neumann entropy (diversity)
entropy_a = von_neumann_entropy(rho_a)
entropy_b = von_neumann_entropy(rho_b)
print(f"Entropy A: {entropy_a:.4f}")
print(f"Entropy B: {entropy_b:.4f}")
# Higher entropy = more diverse distribution

# Quantum relative entropy (divergence)
kl = quantum_relative_entropy(rho_a, rho_b)
print(f"Relative entropy: {kl:.4f}")
# Asymmetric: KL(A || B) ≠ KL(B || A)
```

## Complete Example

```python
import numpy as np
from model_equality_testing.dataset import load_distribution
from model_equality_testing.src.embeddings import embed_sample
from model_equality_testing.src.quantum_metrics import (
    fit_pca,
    pca_density_matrix,
    trace_distance,
    von_neumann_entropy,
    quantum_relative_entropy
)

# 1. Load distributions and draw samples
dist_a = load_distribution(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    prompt_ids={"wikipedia_en": [0, 1, 2]},
    L=500, source="fp32", load_in_unicode=True
)
dist_b = load_distribution(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    prompt_ids={"wikipedia_en": [0, 1, 2]},
    L=500, source="int8", load_in_unicode=True
)

sample_a = dist_a.draw_completion_sample(100)
sample_b = dist_b.draw_completion_sample(100)

# 2. Embed with SBERT (100 samples × 768 dimensions)
emb_a = embed_sample(sample_a)  # (100, 768)
emb_b = embed_sample(sample_b)  # (100, 768)

# 3. PCA compression on combined data (768 → 50 dimensions)
pca = fit_pca(np.vstack([emb_a, emb_b]), k=50)

# 4. Build density matrices (100×100 each)
rho_a = pca_density_matrix(emb_a, pca)  # (100, 100)
rho_b = pca_density_matrix(emb_b, pca)  # (100, 100)

# 5. Compute quantum metrics
td = trace_distance(rho_a, rho_b)
s_a = von_neumann_entropy(rho_a)
s_b = von_neumann_entropy(rho_b)
kl = quantum_relative_entropy(rho_a, rho_b)

print(f"Trace distance: {td:.4f}")
print(f"Entropy: A = {s_a:.4f}, B = {s_b:.4f}")
print(f"Relative entropy: {kl:.4f}")

# Example output:
# Trace distance: 0.2341
# Entropy: A = 3.8234, B = 4.1523
# Relative entropy: 0.1562
```

## What Each Metric Tells You

**Trace Distance**: D(ρ_A, ρ_B) ∈ [0, 1]
- Quantum distinguishability
- 0 = distributions identical
- 1 = completely distinguishable
- Related to maximum probability of telling states apart

**Von Neumann Entropy**: S(ρ) ≥ 0
- Quantum uncertainty/diversity
- Higher = more diverse, spread across many modes
- Lower = concentrated, dominated by few modes
- S = 0 means all samples identical
- S = log(n) means maximally diverse (uniform)

**Quantum Relative Entropy**: S(ρ_A || ρ_B) ≥ 0
- Divergence from A to B
- Asymmetric: S(A||B) ≠ S(B||A)
- Measures information loss if using B to approximate A

## Working with State Vectors (Embeddings)

### Question: "I have a density matrix ρ and a 768-D state vector e. How do I use them together?"

**Answer**: You can't directly combine them because they're in different spaces.

**What you can do**:

**Option 1: Similarity to the distribution**

```python
# New sample embedding
e_new = embed_sample(new_sample)  # (768,)

# Original embeddings (used to build ρ)
E_original = emb_a  # (100, 768)

# Compute similarities
similarities = E_original @ e_new  # (100,)
# similarities[i] = how similar is new sample to sample i

# Typicality score (average similarity)
typicality = similarities.mean()
# High = typical, Low = outlier
```

**Option 2: Project onto semantic axis**

```python
# Semantic axis
from model_equality_testing.src.semantic_axes import load_axis_from_jsonl

axis = load_axis_from_jsonl(
    "professionalism.jsonl",
    "casualness.jsonl",
    "prof-casual"
)

# Project new embedding
projection_new = e_new @ axis.axis_vector

# Compare to distribution's mean
mu = E_original.mean(axis=0)
mean_projection = mu @ axis.axis_vector
deviation = projection_new - mean_projection
```

**Option 3: Augment the density matrix**

```python
# Add new sample to the set
E_augmented = np.vstack([E_original, e_new])  # (101, 768)

# Build new density matrix
rho_aug = pca_density_matrix(E_augmented, pca)  # (101, 101)

# Compare to original
# (but they're different sizes, so limited comparison options)
```

## Important Limitations and Caveats

### Known Limitation: Sample-Size Dependence

From the docstring in `quantum_metrics.py`:

> "The NxN density matrices constructed from sample-specific Gram matrices (PIP = E @ E.T) live in different N-dimensional subspaces of R^d for different samples. Cross-matrix metrics (trace distance, QRE) are therefore not well-defined in the quantum-mechanical sense, and trace distance exhibits pathological sample-size dependence."

**Implications**:
- Trace distance is not strictly "quantum" in the physics sense
- Requires equal sample sizes (n_a = n_b)
- Should be interpreted as a similarity-based distributional metric, not pure quantum metric

### Equal Sample Size Requirement

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

**For different sample sizes**, use v1 semantic axes instead - they handle n_a ≠ n_b naturally.

### PCA Circular Dependency

PCA is fit on the combined data (A + B), which creates a circular dependency:
- PCA finds directions of maximum variance in A+B
- If a semantic dimension has low variance in A+B, PCA discards it
- Later analysis in that dimension shows "no difference" (because it was discarded)

This is a known issue documented in tonight's critiques. No perfect solution exists - it's a tradeoff between dimensionality reduction (computational efficiency) and information preservation.

## Relationship to Other Tools

### Density Matrices vs Semantic Axes

**Density matrices** (this guide):
- Operate in sample space (n×n)
- Capture pairwise similarities
- Global metrics (trace distance, entropy)
- Require equal sample sizes
- Good for: detection and quantification of difference

**Semantic axes** (v1):
- Operate in embedding space (768-D or 50-D)
- Project onto interpretable dimensions
- Per-axis statistics (mean, variance, effect size, p-value)
- Work with any sample sizes
- Good for: interpretation of what differs

**Recommended workflow**: Use both!
1. Density matrices for detection: "Do they differ?" (trace distance)
2. Semantic axes for interpretation: "What differs?" (professionalism, technicality, etc.)

See `CURRENT-CAPABILITIES-SUMMARY.md` for complete overview.

## References

**Implementation**:
- `model_equality_testing/src/quantum_metrics.py` - Density matrix functions
- `model_equality_testing/src/embeddings.py` - SBERT embedding generation

**Documentation**:
- `QUANTUM-METRICS-ADVANTAGES.md` - Honest assessment of advantages/limitations
- `QUANTUM-VS-SEMANTIC-INTERPRETATION.md` - What each layer provides
- `CURRENT-CAPABILITIES-SUMMARY.md` - Complete summary of tools

**Related explorations** (archived):
- `archive/exploration/` - Tonight's work on integration approaches
- All attempted to bridge embedding space (768-D) and sample space (n×n)
- All failed due to fundamental dimensional mismatch
