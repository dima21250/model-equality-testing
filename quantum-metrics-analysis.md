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

All experiments use Llama-3-8B-Instruct, wikipedia_en prompt 0, PCA k=50.

### Context 1: FP32 vs INT8 Quantization (same prompt)

Does quantization change the semantic distribution?

#### n=200, b=100 (exploratory)

| Metric | Statistic | p-value | Detects difference? |
|--------|-----------|---------|-------------------|
| **MMD (Hamming)** | 0.002 | < 0.01 | Yes |
| **VADER K-S** | 0.030 | 0.0300 | Marginal |
| **Trace Distance** | 0.220 | 0.3900 | No |
| **Von Neumann Div** | 0.035 | 0.4900 | No |
| **QRE (symmetric)** | 0.030 | 0.0400 | Marginal |

At this sample size, quantum metrics show no significant semantic difference.

#### n=1000, b=5000 (high-power)

| Metric | Statistic | p-value | Detects difference? |
|--------|-----------|---------|-------------------|
| **MMD (Hamming)** | 0.002 | < 0.0002 | Yes |
| **VADER K-S** | 0.031 | 0.7228 | No |
| **Trace Distance** | 0.143 | < 0.0002 | Yes |
| **Von Neumann Div** | 0.029 | 0.3244 | No |
| **QRE (symmetric)** | 0.020 | < 0.0002 | Yes |

With more data and permutations, a more nuanced picture emerges: trace distance and QRE detect a small but real semantic shift, while von Neumann divergence and VADER K-S remain non-significant.

**Interpretation**: INT8 quantization slightly shifts *where* the semantic distribution sits in embedding space (trace distance, QRE significant), but preserves its *shape* and *spread* (von Neumann divergence non-significant) and its sentiment profile (VADER K-S non-significant). The effect is real but small -- it took n=1000 to detect, and the trace distance magnitude (0.143 on a 0-1 scale) is modest. At n=200 the effect was below the detection threshold.

#### Null baseline: Llama-3-8B fp32 vs fp32

n=1000, b=5000. Both samples drawn from the same fp32 source.

| Metric | fp32 vs fp32 (null) | fp32 vs int8 |
|--------|-------------------|--------------|
| MMD (Hamming) | -0.000, p=0.9300 | 0.002, p < 0.0002 |
| VADER K-S | 0.022, p=0.9690 | 0.031, p=0.7228 |
| Trace distance | 0.086, p=0.9768 | 0.143, p < 0.0002 |
| Von Neumann div | 0.008, p=0.7760 | 0.029, p=0.3244 |
| QRE (symmetric) | 0.003, p=0.5992 | 0.020, p < 0.0002 |

All metrics are non-significant under the null, with p-values well above 0.05. The contrast with fp32 vs int8 is clear: trace distance nearly doubles (0.086 to 0.143) and QRE increases by an order of magnitude (0.003 to 0.020), both becoming highly significant. Von Neumann divergence remains non-significant in both cases, consistent with its role as a diversity-only measure.

### Context 2: Different Prompts, Same Model (positive control)

Can the quantum metrics detect genuine semantic differences?

Llama-3-8B-Instruct, prompt 0 vs prompt 5, fp32.

#### n=200, b=100 (exploratory)

| Metric | Statistic | p-value | Detects difference? |
|--------|-----------|---------|-------------------|
| **MMD (Hamming)** | 0.056 | < 0.01 | Yes |
| **VADER K-S** | 0.510 | < 0.01 | Yes |
| **Trace Distance** | 0.489 | < 0.01 | Yes |
| **Von Neumann Div** | 0.006 | 0.9300 | No |
| **QRE (symmetric)** | 0.059 | < 0.01 | Yes |

#### n=1000, b=5000 (high-power)

| Metric | Statistic | p-value | Detects difference? |
|--------|-----------|---------|-------------------|
| **MMD (Hamming)** | 0.056 | < 0.0002 | Yes |
| **VADER K-S** | 0.469 | < 0.0002 | Yes |
| **Trace Distance** | 0.466 | < 0.0002 | Yes |
| **Von Neumann Div** | 0.032 | 0.2462 | No |
| **QRE (symmetric)** | 0.049 | < 0.0002 | Yes |

**Interpretation**: Trace distance and QRE strongly detect the semantic difference between prompts at both sample sizes. Von Neumann divergence remains non-significant even at n=1000 with b=5000 -- the two prompts produce outputs with similar diversity despite entirely different content. This confirms von Neumann divergence measures distributional spread, not content.

The stability of results across sample sizes is notable: trace distance is consistent (0.489 at n=200 vs 0.466 at n=1000), as is QRE (0.059 vs 0.049). The effect is large and unambiguous, in contrast to the quantization effect which required n=1000 to detect.

### Context 3: Cross-Language Prompt Comparison

Does changing the language of the prompt affect the diversity structure of the output?

Llama-3-8B-Instruct, wikipedia_en prompt 0 vs wikipedia_ru prompt 0, fp32, n=1000, b=5000.

| Metric | Statistic | p-value | Detects difference? |
|--------|-----------|---------|-------------------|
| **MMD (Hamming)** | 0.213 | < 0.0002 | Yes |
| **VADER K-S** | 0.940 | < 0.0002 | Yes |
| **Trace Distance** | 0.405 | < 0.0002 | Yes |
| **Von Neumann Div** | 0.614 | < 0.0002 | Yes |
| **QRE (symmetric)** | 0.217 | < 0.0002 | Yes |

Side-by-side with same-language prompt comparison (en[0] vs en[5], n=1000, b=5000):

| Metric | Same language (en vs en) | Cross language (en vs ru) |
|--------|------------------------|--------------------------|
| MMD (Hamming) | 0.056, p < 0.0002 | 0.213, p < 0.0002 |
| VADER K-S | 0.469, p < 0.0002 | 0.940, p < 0.0002 |
| Trace distance | 0.466, p < 0.0002 | 0.405, p < 0.0002 |
| Von Neumann div | 0.032, p=0.25 | **0.614, p < 0.0002** |
| QRE (symmetric) | 0.049, p < 0.0002 | 0.217, p < 0.0002 |

**Interpretation**: This is the first context in which von Neumann divergence is significant -- and the effect is massive (0.614, dwarfing all other von Neumann values observed). English and Russian completions differ not just in content but in *diversity structure*: the model likely produces a narrower, more constrained distribution in Russian (a non-primary language) compared to English, where it has richer generation capacity.

This result completes the validation of what von Neumann divergence measures. It is insensitive to:
- Quantization (same diversity, different tokens)
- Different prompts in the same language (different content, similar diversity)

But it detects:
- Cross-language differences (genuinely different diversity structure)

The other metrics are significant across both same-language and cross-language comparisons, but note that QRE increases sharply (0.049 to 0.217), reflecting the much greater information-theoretic divergence between languages. Trace distance is comparable (0.466 vs 0.405), suggesting the distributions are similarly distinguishable in shape regardless of whether the difference is topical or linguistic.

### Context 4: Cross-Model Comparison of Quantization Effects

Is the semantic impact of INT8 quantization model-dependent?

Mistral-7B-Instruct-v0.3, wikipedia_en prompt 0, n=1000, b=5000.

| Metric | Statistic | p-value | Detects difference? |
|--------|-----------|---------|-------------------|
| **MMD (Hamming)** | 0.001 | < 0.0002 | Yes |
| **VADER K-S** | 0.056 | 0.0869 | No |
| **Trace Distance** | 0.109 | 0.0020 | Yes |
| **Von Neumann Div** | 0.007 | 0.7794 | No |
| **QRE (symmetric)** | 0.001 | 0.8764 | No |

Side-by-side with Llama-3-8B at the same settings (n=1000, b=5000):

| Metric | Llama-3-8B | Mistral-7B |
|--------|-----------|-----------|
| MMD (Hamming) | 0.002, p < 0.0002 | 0.001, p < 0.0002 |
| VADER K-S | 0.031, p=0.72 | 0.056, p=0.09 |
| Trace distance | 0.143, p < 0.0002 | 0.109, p=0.0020 |
| Von Neumann div | 0.029, p=0.32 | 0.007, p=0.78 |
| QRE (symmetric) | 0.020, p < 0.0002 | 0.001, p=0.88 |

**Interpretation**: Mistral-7B is more robust to INT8 quantization than Llama-3-8B at every level. The token-level effect is half the size (MMD 0.001 vs 0.002). Trace distance is smaller (0.109 vs 0.143) and less significant (p=0.002 vs p < 0.0002). Most strikingly, QRE is completely non-significant for Mistral (p=0.88) while highly significant for Llama (p < 0.0002) -- meaning Mistral's semantic information content is fully preserved under quantization, while Llama's shows a small information-theoretic loss.

The consistent finding across both models: von Neumann divergence is non-significant, confirming that INT8 quantization preserves semantic diversity regardless of model architecture. The degree of semantic perturbation, however, is model-dependent -- Mistral's distribution shifts slightly in location (trace distance significant) but preserves information content (QRE non-significant), while Llama shows both a location shift and a small information-theoretic difference.

#### Null baseline: Mistral-7B fp32 vs fp32

Mistral-7B-Instruct-v0.3, wikipedia_en prompt 0, n=1000, b=5000. Both samples drawn from the same fp32 source.

| Metric | fp32 vs fp32 (null) | fp32 vs int8 |
|--------|-------------------|--------------|
| MMD (Hamming) | 0.000, p=0.2900 | 0.001, p < 0.0002 |
| VADER K-S | 0.050, p=0.1641 | 0.056, p=0.0869 |
| Trace distance | 0.089, p=0.5322 | 0.109, p=0.0020 |
| Von Neumann div | 0.029, p=0.2560 | 0.007, p=0.7794 |
| QRE (symmetric) | 0.002, p=0.5234 | 0.001, p=0.8764 |

All metrics are non-significant under the null, confirming the permutation tests are well-calibrated. Note that trace distance is not zero under the null (0.089) -- there is always sampling noise in density matrix estimates. The permutation test correctly identifies this as non-significant, which is why raw statistics alone are uninterpretable without p-values.

### Context 5: PCA k=50 vs Full 768D Density Matrices

Does dimensionality reduction via PCA affect the conclusions?

An alternative to PCA projection is to construct full d×d density matrices from unit-normalized embeddings: ρ = (1/N) Σ |ψ_i⟩⟨ψ_i|, where each embedding is L2-normalized. This preserves all 768 dimensions of the embedding space.

Llama-3-8B-Instruct, wikipedia_en prompt 0, fp32 vs int8, n=1000, b=5000.

| Metric | PCA k=50 | Full 768D |
|--------|----------|-----------|
| MMD (Hamming) | 0.002, p < 0.0002 | 0.003, p < 0.0002 |
| VADER K-S | 0.031, p=0.7228 | 0.025, p=0.9137 |
| Trace distance | 0.143, p < 0.0002 | 0.101, p < 0.0002 |
| Von Neumann div | 0.029, p=0.3244 | 0.011, p=0.6064 |
| QRE (symmetric) | 0.020, p < 0.0002 | 0.004, p < 0.0002 |

**Interpretation**: The two approaches produce qualitatively identical conclusions. All significance/non-significance calls agree. The raw statistics are uniformly smaller in full-space mode -- trace distance drops from 0.143 to 0.101, QRE from 0.020 to 0.004, von Neumann divergence from 0.029 to 0.011. This is expected: the 768D density matrices are rank ~200 (capped by the n=1000 sample size), so the signal is diluted across 568 zero-eigenvalue dimensions. PCA concentrates the signal into the 50 dimensions that carry the most variance, producing sharper effect sizes.

Despite smaller statistics, the permutation tests still correctly identify trace distance and QRE as significant. The null distribution is diluted by the same factor, so p-values track. Von Neumann divergence is even further from significance in full-space mode (p=0.61 vs p=0.32), consistent with dilution reducing the apparent effect size without changing the conclusion.

The full-space mode carries a substantial computational cost: QRE permutation testing took ~15 minutes at 768D vs ~30 seconds at k=50, roughly a 30x slowdown from eigendecomposing 768×768 matrices instead of 50×50.

For Llama, this validates the PCA k=50 approach: the full embedding space does not reveal effects that PCA misses, while PCA produces larger effect sizes and runs much faster.

#### Mistral-7B: PCA k=50 vs Full 768D

Mistral-7B-Instruct-v0.3, wikipedia_en prompt 0, fp32 vs int8, n=1000, b=5000.

| Metric | PCA k=50 | Full 768D |
|--------|----------|-----------|
| MMD (Hamming) | 0.001, p < 0.0002 | 0.000, p < 0.0002 |
| VADER K-S | 0.056, p=0.0869 | 0.027, p=0.8595 |
| Trace distance | 0.109, p=0.0020 | 0.092, p < 0.0002 |
| Von Neumann div | **0.007, p=0.7794** | **0.097, p < 0.0002** |
| QRE (symmetric) | **0.001, p=0.8764** | **0.003, p=0.0008** |

**Interpretation**: Unlike Llama, the two approaches **disagree qualitatively** for Mistral. Under PCA k=50, Mistral appeared robust to quantization -- only trace distance was marginally significant, and both von Neumann divergence and QRE were solidly non-significant. Under full 768D, von Neumann divergence flips from non-significant (p=0.78) to highly significant (p < 0.0002), with the statistic jumping from 0.007 to 0.097. QRE also becomes significant (p=0.0008).

This means the tail dimensions that PCA discards carry information about Mistral's diversity change under quantization -- information that is absent (or undetectable) in the top 50 principal components. This partially vindicates the concern that PCA projection might mask real effects, though the effect is model-dependent: Llama shows no such discrepancy.

#### Null baseline: Full 768D, Mistral-7B fp32 vs fp32

n=1000, b=5000. Both samples drawn from the same fp32 source.

| Metric | Full 768D (null) | Full 768D (fp32 vs int8) |
|--------|------------------|--------------------------|
| MMD (Hamming) | -0.000, p=0.7500 | 0.000, p < 0.0002 |
| VADER K-S | 0.025, p=0.9137 | 0.027, p=0.8595 |
| Trace distance | 0.057, p=0.5706 | 0.092, p < 0.0002 |
| Von Neumann div | 0.001, p=0.9632 | 0.097, p < 0.0002 |
| QRE (symmetric) | 0.002, p=0.0736 | 0.003, p=0.0008 |

The null is clean -- all metrics non-significant -- confirming that the Mistral full-space results are not false positives. The von Neumann divergence contrast is dramatic: 0.001 under the null vs 0.097 under quantization, an ~80x increase. This confirms that INT8 quantization genuinely alters Mistral's semantic diversity structure in ways that are only visible in the tail dimensions of the embedding space.

The PCA k=50 conclusion that "Mistral is fully robust to quantization" must be revised: Mistral preserves the semantic structure captured by the top 50 principal components, but quantization perturbs the fine-grained diversity structure encoded in the remaining dimensions. Whether this matters in practice depends on what those tail dimensions represent -- but the effect is statistically real.

#### Null baseline: Full 768D, Llama-3-8B fp32 vs fp32

n=1000, b=5000. Both samples drawn from the same fp32 source.

| Metric | PCA k=50 (null) | Full 768D (null) | Full 768D (fp32 vs int8) |
|--------|----------------|------------------|--------------------------|
| MMD (Hamming) | -0.000, p=0.9300 | -0.000, p=0.5600 | 0.003, p < 0.0002 |
| VADER K-S | 0.022, p=0.9690 | 0.023, p=0.9542 | 0.025, p=0.9137 |
| Trace distance | 0.086, p=0.9768 | 0.048, p=0.9786 | 0.101, p < 0.0002 |
| Von Neumann div | 0.008, p=0.7760 | 0.010, p=0.6612 | 0.011, p=0.6064 |
| QRE (symmetric) | 0.003, p=0.5992 | 0.0004, p=0.9320 | 0.004, p < 0.0002 |

The full-space null is well-calibrated -- no false positives, with all p-values well above 0.05. The contrast with fp32 vs int8 is clean: trace distance doubles (0.048 to 0.101) and QRE increases by an order of magnitude (0.0004 to 0.004), both becoming highly significant. Note that trace distance under the null is smaller in full-space mode (0.048) than in PCA mode (0.086), consistent with the general pattern of diluted statistics in higher dimensions. The permutation test correctly identifies both as non-significant regardless of the raw magnitude.

## What This Tells Us

### The Two-Tier Framework: Detection vs Characterization

The central insight from these experiments is that **detection** and **characterization** are distinct tasks requiring different tools.

**MMD Hamming is the most sensitive detector of change.** It operates at the token level and catches any distributional shift, no matter how small or semantically irrelevant. In every context tested -- quantization, different prompts, cross-language, cross-model -- MMD is significant. It answers: "Did something change?" But it cannot tell you whether the change matters.

**The quantum metrics tell you whether to care.** They operate at the semantic level and characterize the *nature* of the change. The Llama quantization comparison illustrates this clearly:

| Metric | Llama fp32 vs fp32 (null) | Llama fp32 vs int8 |
|--------|--------------------------|-------------------|
| MMD Hamming | -0.000, p=0.9300 | 0.002, p < 0.0002 |
| Trace distance | 0.086, p=0.9768 | 0.143, p < 0.0002 |
| QRE (symmetric) | 0.003, p=0.5992 | 0.020, p < 0.0002 |
| Von Neumann div | 0.008, p=0.7760 | 0.029, p=0.3244 |

MMD detects a token-level change. The quantum metrics characterize it: trace distance and QRE show a small semantic shift, but von Neumann divergence remains non-significant -- the diversity structure is preserved. Without the quantum metrics, you would know that quantization changes the output but not whether the change is semantically meaningful.

The picture becomes richer when dimensionality matters. Under PCA k=50, Mistral appeared fully robust to quantization (QRE non-significant, von Neumann non-significant). But full 768D density matrices reveal a diversity change (von Neumann 0.097, p < 0.0002) invisible to PCA -- the quantization effect lives in the tail dimensions that PCA discards. This demonstrates that the characterization depends on what dimensions of the embedding space you examine, and that the choice of density matrix construction is itself an interpretive decision.

This makes the two-tier framework essential, not redundant. MMD alone would detect change without characterizing it. Quantum metrics alone would miss the token-level change at small sample sizes. Together -- and with attention to the density matrix construction -- they provide a complete picture: MMD for detection, quantum metrics for characterization, and the PCA/full-space comparison for understanding at what scale the semantic change occurs.

### What Each Quantum Metric Measures

The five contexts together reveal what each metric is sensitive to:

| Metric | Content shift? | Diversity shift? | Evidence |
|--------|---------------|-----------------|----------|
| Trace distance | Yes | Yes | Detects prompt and language differences |
| Von Neumann div | No | Yes | Only fires for cross-language (diversity structure change) |
| QRE (symmetric) | Yes | Yes | Detects prompt and language differences; scales with effect size |

**Von Neumann divergence** is the most selective metric. It is insensitive to content differences within a language (different prompts), but highly sensitive to diversity structure changes (cross-language, von Neumann 0.614). Its behavior under quantization depends on the density matrix construction: under PCA k=50 it is non-significant for both Llama and Mistral, suggesting diversity is preserved in the top principal components. Under full 768D, it remains non-significant for Llama but becomes highly significant for Mistral (0.097, p < 0.0002), revealing a diversity change in the tail dimensions. Its non-significance for same-language prompt pairs confirms it measures spread, not content.

**Trace distance and QRE** are sensitive to both content and diversity differences. At n=200 they show non-significance for fp32 vs int8, but at n=1000 they detect a small semantic shift. This means quantization does slightly perturb the semantic distribution, but the effect is subtle. Their strong significance for different prompts (trace distance ~0.47) and cross-language comparisons (QRE jumps from 0.049 to 0.217) confirms they can detect real semantic differences at multiple scales.

**Together**, the three metrics paint a precise picture: INT8 quantization introduces a small, model-dependent semantic perturbation. The degree of perturbation depends on both the model architecture and the dimensionality of the analysis. In the top 50 principal components, Llama shows a small semantic shift (trace distance and QRE significant) while Mistral appears robust (only trace distance marginally significant). In the full embedding space, Mistral reveals a diversity change invisible to PCA (von Neumann divergence significant), while Llama's picture remains unchanged. The magnitude of the quantization effect (trace distance 0.09-0.14) is modest compared to a same-language content difference (trace distance ~0.47) or cross-language difference (von Neumann divergence 0.614).

## Technical Notes

- All density matrices are constructed via PCA projection (k=50) of SBERT embeddings to a shared basis, followed by covariance matrix normalization. This fixes the Hilbert space mismatch and sample-size dependence of the earlier NxN Gram matrix approach. A full 768D alternative (Context 5) validates that PCA does not mask real effects.
- p-values are computed via permutation tests with pre-computed embeddings for efficiency.
- The QRE uses an eigenvalue-pairing approximation that is valid when both density matrices share the PCA eigenbasis.
- Explained variance at k=50 is typically ~40-50% of total variance in the embedding space.
- Full-space density matrices (pca_k=0) are constructed from unit-normalized embeddings: ρ = E_norm.T @ E_norm / N, yielding d×d matrices where d is the embedding dimension (768 for all-mpnet-base-v2).
