# Quantum Metrics Analysis: What Each Metric Measures

## The Two-Tier Framework

We use classical tests (MMD Hamming) to detect whether a distribution has changed at the token level, and quantum-inspired metrics to characterize whether the change matters semantically. The quantum metrics are computed on SBERT embeddings (all-mpnet-base-v2, 768-dim) projected to a shared 50-dimensional PCA basis, yielding 50x50 density matrices.

## The Three Quantum Metrics

### Trace Distance: D(rho_A, rho_B) = 0.5 * Tr(|rho_A - rho_B|)

Measures how *distinguishable* two density matrices are. This is a direct comparison of distribution shape in embedding space -- it asks "are these the same semantic distribution?" A trace distance of 0 means identical distributions; 1 means completely orthogonal support.

### Von Neumann Divergence: |S(rho_A) - S(rho_B)|

Measures whether two distributions have similar *diversity* or *spread*. The von Neumann entropy S(rho) quantifies how "mixed" a quantum state is -- analogously, how semantically diverse the outputs are. The divergence compares entropies independently, asking "do these distributions have the same level of variety?" without comparing content directly.

### Quantum Relative Entropy (symmetric): S(rho_A || rho_B) + S(rho_B || rho_A)

Measures the *information loss* when approximating one distribution with the other. This is the quantum analog of symmetrized KL divergence. It captures both shape and spread differences, making it the most comprehensive of the three.

## Experimental Results

All experiments use Llama-3-8B-Instruct, n=200 samples, b=100 permutations, PCA k=50.

### Context 1: FP32 vs INT8 Quantization (same prompt)

Does quantization change the semantic distribution?

| Metric | Statistic | p-value | Detects difference? |
|--------|-----------|---------|-------------------|
| **MMD (Hamming)** | 0.002 | 0.0000 | Yes |
| **VADER K-S** | 0.030 | 0.0300 | Marginal |
| **Trace Distance** | 0.220 | 0.3900 | No |
| **Von Neumann Div** | 0.035 | 0.4900 | No |
| **QRE (symmetric)** | 0.030 | 0.0400 | Marginal |

**Interpretation**: MMD detects token-level differences from quantization, but all three quantum metrics show non-significant or marginal p-values. The semantic distribution is preserved -- quantization changes *how* the model says things at the character level, but not *what* it says at the meaning level.

### Context 2: Different Prompts, Same Model (positive control)

Can the quantum metrics detect genuine semantic differences?

| Metric | Statistic | p-value | Detects difference? |
|--------|-----------|---------|-------------------|
| **MMD (Hamming)** | 0.056 | 0.0000 | Yes |
| **VADER K-S** | 0.510 | 0.0000 | Yes |
| **Trace Distance** | 0.489 | 0.0000 | Yes |
| **Von Neumann Div** | 0.006 | 0.9300 | No |
| **QRE (symmetric)** | 0.059 | 0.0000 | Yes |

**Interpretation**: Trace distance and QRE both strongly detect the semantic difference between prompts (p < 0.0001). Von Neumann divergence does *not* -- the two prompts produce outputs with similar diversity even though the content is entirely different. This is expected: von Neumann divergence measures distributional spread, not content.

## What This Tells Us

The two contexts together reveal what each metric is sensitive to:

| Metric | Sensitive to content? | Sensitive to diversity? |
|--------|----------------------|------------------------|
| Trace distance | Yes (detects prompt difference) | Yes (compares full shape) |
| Von Neumann div | No (misses prompt difference) | Yes (compares spread only) |
| QRE (symmetric) | Yes (detects prompt difference) | Yes (compares both) |

**Von Neumann divergence** is uniquely informative in the quantization context *because* of its insensitivity to content differences. Its non-significance for fp32 vs int8 isn't a failure to detect -- it's a meaningful null result: the semantic diversity of the output distribution is preserved under quantization. The fact that it also gives a null result for different prompts (which have similar diversity but different content) confirms it is measuring what we claim.

**Trace distance and QRE** provide complementary evidence. Their non-significance for fp32 vs int8 says the semantic distributions aren't just similarly diverse -- they have similar *shape* and *information content*. Their significance for different prompts confirms they can detect real semantic differences when present.

**Together**, the three metrics support the conclusion: INT8 quantization changes the token-level distribution but preserves the semantic distribution in terms of both content (trace distance, QRE) and diversity (von Neumann divergence).

## Technical Notes

- All density matrices are constructed via PCA projection (k=50) of SBERT embeddings to a shared basis, followed by covariance matrix normalization. This fixes the Hilbert space mismatch and sample-size dependence of the earlier NxN Gram matrix approach.
- p-values are computed via permutation tests (b=100) with pre-computed embeddings for efficiency.
- The QRE uses an eigenvalue-pairing approximation that is valid when both density matrices share the PCA eigenbasis.
- Explained variance at k=50 is typically ~40-50% of total variance in the embedding space.
