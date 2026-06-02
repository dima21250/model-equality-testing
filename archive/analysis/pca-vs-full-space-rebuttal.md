# Rebuttal: PCA k=50 vs Full 768D Density Matrices

## The Criticism

The reviewer argues that projecting SBERT embeddings to a 50-dimensional PCA basis before constructing density matrices discards too much information. With ~40-50% explained variance at k=50, more than half the embedding structure is lost. The proposed alternative: construct 768x768 density matrices directly from unit-normalized embeddings, preserving all dimensions:

```
ρ = (1/N) Σ_i |ψ_i⟩⟨ψ_i|    where |ψ_i⟩ = e_i / ||e_i||
```

This fixes the Hilbert space mismatch (both density matrices live in R^768) without any dimensionality reduction.

## Point-by-Point Response

### 1. The diagonal covariance claim is wrong

The reviewer claims that per-sample covariance matrices in PCA space are diagonal, and therefore that our density matrices contain no off-diagonal structure. This is factually incorrect.

PCA is fit once on the **combined** pool of both samples. When we then compute per-sample covariance matrices in this shared PCA basis, the resulting matrices are **not** diagonal. PCA diagonalizes the covariance of the data it was fit on — the combined pool — but each sample's individual covariance retains full off-diagonal structure in that basis. This is precisely what makes the comparison informative: the shared basis reveals how each sample's covariance structure differs from the combined-pool structure and from each other.

If we fit PCA separately on each sample, then yes, each sample's covariance would be diagonal in its own basis — but the two bases would differ, reintroducing the Hilbert space mismatch. The shared-basis design is intentional.

### 2. PCA k=50 is regularization, not information loss

The framing of "discarding 50-60% of variance" treats all variance as equally informative. In high-dimensional embedding spaces, this is not the case. SBERT embeddings have 768 dimensions, but the semantic content is concentrated in the top principal components. The tail components carry progressively more noise relative to signal — they encode idiosyncratic features of individual sentences rather than distributional properties of the output population.

Retaining all 768 dimensions means the density matrix is influenced by noise in dimensions that contribute almost nothing to the semantic structure we're trying to compare. This is the standard bias-variance trade-off: PCA introduces a small amount of bias (losing tail variance) in exchange for a large reduction in variance (the density matrix estimate is more stable across samples).

The analogy to classical statistics is direct: we could estimate a 768-parameter model, but with N=200 or N=1000 samples, a 50-parameter model yields more reliable estimates. The effective degrees of freedom in a 768x768 density matrix (768 * 769 / 2 = 295,296 unique entries) far exceeds our sample sizes.

### 3. The experimental record validates PCA k=50

The PCA k=50 approach has produced consistent, interpretable results across four distinct experimental contexts:

- **Null control** (fp32 vs fp32): All metrics non-significant with p >> 0.05. No false positives.
- **Quantization** (fp32 vs int8): Trace distance and QRE detect a small semantic shift at n=1000; von Neumann divergence correctly reports no diversity change. Effect is model-dependent (Llama vs Mistral).
- **Same-language prompt comparison**: All content-sensitive metrics highly significant; von Neumann divergence correctly non-significant (different content, similar diversity).
- **Cross-language comparison**: Von Neumann divergence fires dramatically (0.614) — the only context where it does — correctly detecting a genuine diversity structure change.

This pattern of results would be difficult to produce from an approach that is "throwing away the interesting structure." The metrics are selective in exactly the ways predicted by their mathematical definitions, with von Neumann divergence exhibiting a clean three-way pattern (insensitive to quantization, insensitive to content shifts, sensitive to diversity shifts) that validates both the metric and the density matrix construction.

### 4. Full 768D matrices have their own risks

The full-space approach is not without trade-offs:

**Sampling noise amplification.** A 768x768 density matrix estimated from N=200 samples has rank at most 200, meaning 568 eigenvalues are exactly zero. The non-zero eigenvalues are estimated from fewer effective samples per dimension than in the PCA case. This makes the eigenspectrum — the core input to all three quantum metrics — more sensitive to sampling noise.

**Permutation test calibration.** The permutation null distribution must also be estimated with 768x768 matrices. If the observed statistic is noisy, the null distribution is equally noisy, and the resulting p-values may be less stable across runs. PCA's dimensionality reduction has the effect of stabilizing both the observed statistic and the null distribution.

**QRE support conditions.** Quantum relative entropy requires supp(ρ) ⊆ supp(σ). With 768x768 matrices of rank ~200, support mismatches become more likely, potentially producing infinity values. The PCA approach, with rank-50 matrices, is less susceptible to this.

### 5. Both approaches should be compared empirically

The theoretical arguments favor PCA (regularization, stable estimation, matched effective dimensionality) but the reviewer raises a legitimate empirical question: does the PCA projection mask real effects? The answer should come from data, not theory.

We have implemented the full 768D mode (`--pca_k 0`) alongside the PCA mode (`--pca_k 50`). The decisive experiment is straightforward: run the same comparisons under both modes and check whether the full-space approach detects effects that PCA misses, or whether the additional dimensions introduce noise without improving discrimination. If the results are qualitatively identical, PCA's regularization benefit makes it the better default. If they diverge, the divergence itself is informative about what the tail dimensions contribute.

## Summary

The PCA k=50 approach is a principled regularization choice supported by four contexts of consistent experimental results. The reviewer's criticism overstates the cost of dimensionality reduction and understates the cost of retaining noisy dimensions. The claim about diagonal covariance matrices is factually incorrect. That said, the full-space alternative is now implemented for empirical comparison — the data will settle the question more convincingly than theory alone.
