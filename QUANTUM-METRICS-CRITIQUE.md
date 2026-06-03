# Quantum-Inspired Metrics for LLM Comparison: Summary and Critique

## The Thinking

**Rationale**: LLM outputs → SBERT embeddings → high-dimensional probability distributions. Quantum mechanics provides tools (density matrices, trace distance, von Neumann entropy) for comparing such distributions in ways that capture global distributional shape, not just mean differences.

**Approach**: 
- Construct Gram matrix: G = E·E^T (where E = embeddings)
- Normalize to density matrix: ρ = G/Tr(G)
- Apply quantum metrics to compare ρ_A vs ρ_B

---

## The Problems

### 1. Trace Distance is Fundamentally Flawed

- **Sample-size dependent**: bounded by min(rank(ρ_A), rank(ρ_B)) ≤ min(n_A, n_B)
- **Different sample sizes** → incomparable values
- Even with equal n, unclear if differences reflect true distribution or sampling noise
- **Computationally expensive**: SVD of n×n matrices

### 2. The Calibration Problem

- Metrics produce numbers (e.g., "trace distance = 0.23") with **no statistical context**
- No p-values, no effect size scale, no decision thresholds
- Can't answer: "Is 0.23 meaningful or just noise?"
- Unlike t-tests (p-value) or Cohen's d (standardized effect), quantum metrics lack interpretive scaffolding

### 3. Sample Size Dependencies

- **Von Neumann entropy**: bounded by log(n), different for different n
- **Effective rank**: counts eigenvalue "modes" but scale depends on sample size
- Even with equal n, permutation tests can be confounded by sample-size artifacts

### 4. Interpretation Gap

- "Eigenvalue spectra differ" → **what does this mean semantically?**
- "Effective rank = 25" → these are eigenvalue concentration patterns, not semantic topics
- Metrics detect *something* changed, but don't explain *what* changed

### 5. PCA Circularity

If dimensionality reduction is used:
- Fitting PCA on combined samples breaks permutation test validity
- PCA captures structure from both distributions → permutations aren't exchangeable under null

---

## The Solution Path

### What Works

- **Spectral metrics without PCA**: Von Neumann entropy difference, Wasserstein distance on eigenvalue spectra
- **Equal sample sizes**: Eliminates sample-size confounds (n_A = n_B required)
- **Permutation testing**: Provides p-values for significance
- **Bootstrap CIs**: Quantifies uncertainty in effect magnitude

### What's Still Missing

- **Semantic interpretation**: Quantum metrics detect differences but don't explain them
- **Solution**: Use semantic axes as interpretation layer (project embeddings onto human-interpretable dimensions like professionalism ← → casualness)

---

## Bottom Line

Quantum-inspired metrics are **theoretically elegant but practically problematic**: sample-size dependencies, calibration issues, and interpretation gaps make them hard to use for applied work. 

**Spectral metrics** (entropy, Wasserstein on eigenvalues) are more viable than trace distance, but still require:
- Equal sample sizes (n_A = n_B)
- Permutation testing for p-values
- Semantic axes for interpretation

to be scientifically defensible.

---

## Recommendation

For applied LLM comparison work:

1. **Detection**: Use spectral metrics (entropy difference, Wasserstein) with permutation tests
2. **Interpretation**: Use semantic axes to explain *what* differs
3. **Avoid**: Trace distance, PCA-based density matrices, unequal sample sizes
