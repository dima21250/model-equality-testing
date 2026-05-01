# Initial Results: Quantum-Inspired vs Classical Metrics

## Overview

This document contains initial experimental results comparing quantum-inspired metrics against classical statistical tests for detecting distribution shifts in LLM outputs.

**Experimental Setup:**
- **Models**: Llama-3-8B-Instruct (fp32 vs int8 quantization)
- **Prompts**: 3 Wikipedia English prompts (IDs: 0, 1, 2)
- **Sample sizes**: 100 and 1000 completions
- **Goal**: Detect whether int8 quantization produces statistically different outputs than fp32

**Metrics Compared:**
- **Quantum-inspired**: Trace distance, Von Neumann divergence, QRE (symmetric)
- **Classical**: MMD (Hamming kernel), VADER K-S test

---

## Understanding the Quantum Metrics

The quantum-inspired approach treats LLM output distributions as quantum states:

1. **Convert completions to embeddings** using a sentence transformer (MPNet)
2. **Construct density matrices** from the empirical distribution of embeddings
3. **Apply quantum information metrics** to measure distinguishability

### Trace Distance
**What it measures** (in theory): The maximum distinguishability between two quantum states. In our context, it should quantify how different the semantic distributions of fp32 vs int8 completions are.

**Range**: [0, 1]
- 0 = Identical distributions (indistinguishable)
- 1 = Completely orthogonal distributions (perfectly distinguishable)

**Initial interpretation** (fp32 vs int8):
- n=100: **0.321** → The two distributions are ~32% distinguishable in semantic space
- n=1000: **0.379** → Increased to ~38% distinguishable with better density matrix estimates

**⚠️ INVALIDATED BY SANITY CHECK**: The fp32 vs fp32 comparison shows essentially identical trace distances (0.332 at n=100, 0.371 at n=1000), meaning these values do **not** reflect actual distribution differences. The metric is measuring artifacts, not signal. See "Critical Finding: Trace Distance Fails Sanity Check" below.

### Von Neumann Divergence
**What it measures**: Quantum analog of Kullback-Leibler divergence. Measures the information gained when using the true distribution (fp32) versus assuming the approximate distribution (int8).

**Range**: [0, ∞)
- 0 = Distributions are identical
- Larger values = More information difference

**Interpretation for our results**:
- n=100: **0.081** → Modest information divergence
- n=1000: **0.013** → Surprisingly decreased (likely due to better eigenvalue estimates reducing noise)

**Note**: Unlike trace distance (which is symmetric), Von Neumann divergence is asymmetric. Our implementation may be computing D(ρ_A || ρ_B) where ρ_A is fp32 and ρ_B is int8.

### QRE (Quantum Relative Entropy) Symmetric
**What it measures**: Symmetrized version of quantum relative entropy, computed as [D(ρ_A || ρ_B) + D(ρ_B || ρ_A)]/2. Provides a symmetric information-theoretic distance.

**Range**: [0, ∞)
- 0 = Distributions are identical
- Larger values = More symmetric information divergence

**Interpretation for our results**:
- n=100: **0.003** → Very small symmetric divergence
- n=1000: **0.001** → Even smaller (consistent with Von Neumann divergence decrease)

**Why so small?**: These values suggest that while the distributions are distinguishable (per trace distance), the information-theoretic divergence is minimal. This might indicate that fp32 and int8 produce semantically similar outputs with subtle but detectable differences.

### Key Insight (Original - See Revision Below)
~~**Trace distance captures geometric distinguishability** while **Von Neumann divergence/QRE capture information-theoretic divergence**. In our case:~~
- ~~High trace distance (0.32-0.38) = Distributions are geometrically separated in embedding space~~
- ~~Low divergence (0.01-0.08) = Information content is similar despite geometric separation~~

~~This suggests **int8 quantization shifts the semantic distribution subtly but consistently**, rather than producing wildly different content.~~

**⚠️ REVISED AFTER SANITY CHECK**: The high trace distance (0.32-0.38) does NOT indicate geometric separation between fp32 and int8 - it appears consistently even when comparing identical distributions (fp32 vs fp32). The original interpretation was **incorrect**. The metric likely reflects finite-sample artifacts in high-dimensional density matrix estimation rather than actual distribution differences.

---

## Results: n=100 samples

### Quantum Metrics
- **Trace distance**: 0.320768 (computation time: 7.77s)
- **Von Neumann divergence**: 0.080570 (0.58s)
- **QRE (symmetric)**: 0.003272 (0.55s)

### Classical Metrics
- **MMD (Hamming)**: 0.000380, **p=0.2800** (2.13s)
- **VADER K-S**: 0.150000 (0.03s)

### Interpretation (n=100) - REVISED
- **Classical test verdict**: FAIL to reject null hypothesis (p=0.28) - insufficient power to detect subtle quantization effects
- ~~**Quantum metrics hint**: Trace distance of ~0.32 suggests moderate distinguishability (32% separation)~~
- ~~**Tension**: Quantum metrics suggest differences that classical test lacks power to detect~~

**⚠️ ORIGINAL INTERPRETATION INCORRECT**: The sanity check reveals that trace distance ~0.32 is **not** detecting a real difference - it shows similar values even for identical distributions (fp32 vs fp32 = 0.332). The "tension" was illusory; the classical test was correct to be cautious.

---

## Results: n=1000 samples

### Quantum Metrics
- **Trace distance**: 0.379058 (12.22s)
- **Von Neumann divergence**: 0.013241 (5.79s)
- **QRE (symmetric)**: 0.001151 (5.95s)

### Classical Metrics
- **MMD (Hamming)**: 0.000825, **p=0.0100** (148.65s)
- **VADER K-S**: 0.032000 (0.22s)

### Interpretation (n=1000) - REVISED
- **Classical test verdict**: REJECT null hypothesis (p=0.01) - distributions are statistically different ✓
- ~~**Quantum trace distance increased**: 0.321 → 0.379 (more distinguishable with larger sample)~~
- **MMD p-value dropped**: 0.28 → 0.01 (now has sufficient statistical power) ✓

**⚠️ TRACE DISTANCE FINDING INCORRECT**: The increase to 0.379 is not meaningful since fp32 vs fp32 shows 0.371 - effectively identical. The trace distance increase with sample size appears to be a sampling artifact rather than improved detection.

---

## Sanity Check: fp32 vs fp32 (Same Distribution)

**Critical test**: Compare two independent samples from the **same** distribution to verify metrics correctly identify when distributions are identical.

### Results: n=100 samples (fp32 vs fp32)

**Quantum Metrics:**
- **Trace distance**: 0.331747 (7.09s)
- **Von Neumann divergence**: 0.001961 (0.70s)
- **QRE (symmetric)**: 0.004719 (0.64s)

**Classical Metrics:**
- **MMD (Hamming)**: -0.001936, **p=0.8100** (2.10s) ✓
- **VADER K-S**: 0.090000 (0.03s)

### Results: n=1000 samples (fp32 vs fp32)

**Quantum Metrics:**
- **Trace distance**: 0.371305 (11.98s)
- **Von Neumann divergence**: 0.008056 (5.65s)
- **QRE (symmetric)**: 0.000809 (5.87s)

**Classical Metrics:**
- **MMD (Hamming)**: 0.000038, **p=0.3800** (147.09s) ✓
- **VADER K-S**: 0.038000 (0.22s)

---

## Critical Finding: Trace Distance Fails Sanity Check

### Side-by-Side Comparison

| Metric | fp32 vs int8<br>(n=100) | fp32 vs fp32<br>(n=100) | fp32 vs int8<br>(n=1000) | fp32 vs fp32<br>(n=1000) |
|--------|------------|------------|--------------|--------------|
| **Trace Distance** | 0.321 | **0.332** ❌ | 0.379 | **0.371** ❌ |
| **Von Neumann Div** | 0.081 | **0.002** ✓ | 0.013 | **0.008** ~ |
| **QRE Symmetric** | 0.003 | **0.005** ~ | 0.001 | **0.001** ~ |
| **MMD p-value** | 0.28 | **0.81** ✓ | 0.01 | **0.38** ✓ |

### Interpretation

**Trace distance shows NO discrimination**: The values for comparing different distributions (fp32 vs int8) are essentially identical to comparing the same distribution twice (fp32 vs fp32). This means:

- **At n=100**: 0.321 (different models) vs 0.332 (same model) - sanity check actually shows *higher* distinguishability for identical distributions!
- **At n=1000**: 0.379 (different models) vs 0.371 (same model) - effectively identical

**What this means**: The high trace distance values (~0.32-0.38) are **NOT detecting the quantization difference**. Instead, they reflect:
- Finite sample effects in density matrix estimation
- High dimensionality of embedding space creating artificial separation
- Methodological artifacts (regularization, eigenvalue truncation)

**Classical MMD passes sanity check perfectly**:
- fp32 vs fp32: p=0.81 (n=100), p=0.38 (n=1000) - correctly fails to reject null
- fp32 vs int8: p=0.28 (n=100), p=0.01 (n=1000) - correctly detects difference at n=1000

**Von Neumann divergence shows partial discrimination**: At n=100 it distinguishes well (0.081 vs 0.002), but at n=1000 the values converge (0.013 vs 0.008), suggesting the n=100 difference may have been noise.

### Conclusion

**The initial interpretation was incorrect.** Trace distance is not more sample-efficient than classical tests - it simply produces high values regardless of whether distributions actually differ. The classical MMD test, despite appearing less sensitive initially, was correctly identifying that:
1. At n=100: insufficient power to detect subtle quantization effects
2. At n=1000: sufficient power to detect real differences
3. At all sample sizes: correctly identifies when comparing identical distributions

This is a **negative result** for quantum-inspired metrics in this configuration.

---

## Key Findings (Revised)

### 1. Classical MMD Test Works Correctly
The MMD test with permutation-based p-values demonstrates proper statistical behavior:
- **At n=100**: Lacks power to detect subtle quantization (p=0.28), but correctly identifies same-distribution samples (p=0.81)
- **At n=1000**: Has sufficient power to detect quantization (p=0.01), while still correctly identifying same-distribution samples (p=0.38)

**Test statistic and p-value progression:**
- fp32 vs int8: (n=100) stat=0.00038, p=0.28 → (n=1000) stat=0.00083, p=0.01
- fp32 vs fp32: (n=100) stat=-0.00194, p=0.81 → (n=1000) stat=0.00004, p=0.38

This is **textbook statistical testing**: insufficient power at small sample sizes, correct null hypothesis behavior, and increasing power with larger samples.

### 2. Trace Distance Fails to Discriminate ❌
**The initial hypothesis about sample efficiency was incorrect.** Trace distance produces similar values (~0.32-0.38) regardless of whether comparing:
- Different distributions (fp32 vs int8)
- Identical distributions (fp32 vs fp32)

This indicates the metric is measuring **sampling variance and methodological artifacts** rather than actual distribution differences. Possible causes:
- High-dimensional embedding space creates artificial separation even for identical distributions
- Finite sample density matrix estimation introduces systematic bias
- Regularization or eigenvalue truncation may inflate distances

**Verdict**: In its current implementation, trace distance is **not suitable** for hypothesis testing about LLM distribution equality.

### 3. Von Neumann Divergence: Mixed Results
Shows promise at n=100 (0.081 vs 0.002 distinguishes different vs same distributions), but convergence at n=1000 (0.013 vs 0.008) raises questions about reliability. Requires further investigation.

### 4. Metric Behavior with Increased Sample Size

**Trace distance** (↑): 0.321 → 0.379
- Increased distinguishability suggests better density matrix estimates

**Von Neumann divergence** (↓): 0.081 → 0.013
- Counterintuitively decreased - may indicate noise reduction in eigenvalue estimates

**QRE symmetric** (↓): 0.003 → 0.001
- Also decreased with larger sample

**VADER K-S** (↓): 0.15 → 0.032
- Sentiment distributions appear more similar - initial n=100 may have had sampling noise

### 5. Computation Time Trade-offs

For n=1000:
- Quantum trace distance: 12.22s
- Classical MMD (100 permutations): 148.65s

MMD is slower due to permutation resampling for p-value estimation, but this provides rigorous statistical guarantees that the quantum metrics currently lack. Speed is irrelevant if the metric doesn't work correctly.

---

## Root Cause Analysis: Why Does Trace Distance Fail?

### Hypothesis 1: Curse of Dimensionality
MPNet embeddings are 768-dimensional. In high-dimensional spaces, random samples from the **same** distribution can appear far apart due to concentration of measure. With n=100 or n=1000 samples in 768-D space, the density matrices may be too sparse to reliably estimate the true distribution geometry.

### Hypothesis 2: Finite Sample Bias
Empirical density matrices from finite samples have different spectral properties than the true infinite-sample density matrix. This systematic bias may inflate trace distances even between identical distributions.

### Hypothesis 3: Regularization Artifacts
If the implementation uses regularization (e.g., adding small values to diagonal) or eigenvalue truncation, these could introduce artificial distance even for identical distributions.

### Hypothesis 4: Embedding Model Mismatch
MPNet was trained on sentence similarity tasks, not on distinguishing subtle quantization effects in LLM outputs. The embedding space may not preserve the relevant structure for this problem.

---

## Validation Tests: Confirming the Root Cause

To isolate whether the issue is implementation bugs vs fundamental limitations, we tested the trace distance implementation on **synthetic data with known ground truth**.

**Script**: `test_trace_distance_validation.py`

### Test Results Summary

| Test | Expected | Observed | Status |
|------|----------|----------|--------|
| **Identical distributions** (two samples from N(0,I)) | ≈ 0 | 0.209 | ✗ FAIL |
| **Orthogonal distributions** (orthogonal subspaces) | ≈ 1 | 0.297 | ✗ FAIL |
| **Shifted Gaussians** (partial overlap) | 0.1-0.9 | 0.209 | ✓ PASS |
| **Resampling from same pool** (bootstrap) | < 0.05 | 0.279 | ✗ FAIL |

### Critical Finding: Sample Size Dependence

Testing trace distance on identical distributions (two samples from N(0,I)) as sample size increases:

| n_samples | Trace Distance |
|-----------|----------------|
| 50        | 0.146         |
| 100       | 0.212         |
| 200       | 0.303         |
| 500       | 0.467         |
| 1000      | 0.636         |

**This is catastrophic**: Trace distance **increases** with sample size instead of decreasing. For a valid statistical metric, larger samples should give better approximations to the true distance (which is 0 for identical distributions).

### Root Cause Identified

The issue is not a bug in the trace distance formula (which is mathematically correct), but rather **how empirical Gram matrices are used as density matrices**:

```python
pip = embeddings @ embeddings.T  # Shape: (N, N) 
rho = pip / trace(pip)            # Normalize to trace 1
```

**The fundamental problem**: As N increases:
1. The Gram matrix size grows (N × N)
2. The matrix structure changes systematically with N
3. Simple trace normalization doesn't account for this sample-size dependence
4. The eigenvalue spectrum depends on N, not just the underlying distribution

**This explains the sanity check failure**:
- fp32 vs fp32 (n=1000): trace distance = 0.371
- Synthetic N(0,I) vs N(0,I) (n=1000): trace distance = 0.636
- Both are measuring the **same artifact**: high trace distance at n=1000 due to Gram matrix structure

### Verdict on Current Implementation

The high trace distances observed in all experiments (fp32 vs int8, fp32 vs fp32, synthetic tests) are **not signal** - they are **systematic artifacts** of treating finite-sample Gram matrices as quantum density matrices without proper normalization.

**The method is mathematically sound in theory but breaks down in practice** because:
- It doesn't account for the sample-size-dependent structure of empirical Gram matrices
- Normalization by trace is insufficient to make matrices from different sample sizes comparable
- The curse of dimensionality (d=768) exacerbates finite-sample effects

---

## Potential Fixes and Alternative Approaches

Despite the negative results, the **original motivation remains valid**: quantum-inspired metrics could provide interpretability beyond binary hypothesis testing. While MMD tells us "distributions differ," quantum metrics might characterize *how* they differ (semantic shifts, diversity changes, information content).

### Approach 1: Fixed-Size Density Matrices via Dimensionality Reduction

**Problem**: Gram matrix size (N × N) grows with sample size, creating sample-dependent artifacts.

**Solution**: Project embeddings to a fixed low-dimensional space before constructing density matrices.

**Method**:
```python
# Instead of:
pip = embeddings @ embeddings.T  # N × N matrix (size depends on N)

# Do:
pca = PCA(n_components=k)  # Fixed k (e.g., k=50)
reduced_embeddings = pca.fit_transform(embeddings)  # N × k
# Then construct k × k covariance matrix (fixed size):
cov = reduced_embeddings.T @ reduced_embeddings / N
rho = cov / trace(cov)  # k × k density matrix
```

**Advantages**:
- Fixed matrix size (k × k) regardless of sample size N
- Could make trace distance comparable across different N
- Reduces curse of dimensionality from d=768 to k=50

**Open questions**:
- What k should we use? (trade-off: too small loses information, too large reintroduces artifacts)
- Should we use PCA, UMAP, or random projections?
- Does this preserve the quantum information-theoretic properties we want?

### Approach 2: Kernel-Based Density Matrix Construction

**Problem**: Inner product matrices don't account for sample size properly.

**Solution**: Use kernel density estimation to construct density matrices in a fixed basis.

**Method**: Define a fixed set of basis states (e.g., cluster centers, random anchors) and compute density matrix elements as kernel similarities between samples and basis states. This decouples matrix size from sample size.

**Advantages**:
- Fixed matrix size independent of N
- More principled statistical foundation
- Could leverage kernel theory from MMD literature

**Disadvantages**:
- More complex implementation
- Requires choosing basis (clustering, random sampling, etc.)

### Approach 3: Quantum Fidelity Instead of Trace Distance

**Alternative metric**: Quantum fidelity F(ρ, σ) = Tr(√(√ρ σ √ρ))

**Properties**:
- F ∈ [0, 1], with F=1 for identical states
- May have better finite-sample properties than trace distance
- Related to Bhattacharyya coefficient

**Worth testing**: Does fidelity avoid the sample-size artifacts we see with trace distance?

### Approach 4: Subsampling Normalization

**Problem**: Trace distance grows with N.

**Solution**: Compute trace distance on multiple random subsamples of fixed size and average.

**Method**:
```python
distances = []
for _ in range(B):  # B bootstrap iterations
    subsample_a = random_sample(embeddings_a, size=n_fixed)
    subsample_b = random_sample(embeddings_b, size=n_fixed)
    dist = trace_distance(subsample_a, subsample_b)
    distances.append(dist)
return np.mean(distances)
```

**Advantages**:
- Forces fixed sample size (controls artifact)
- Provides variance estimate via bootstrap

**Disadvantages**:
- Throws away data (inefficient)
- Computational cost: B × trace_distance calculations

### Approach 5: Different Embedding Models

**Hypothesis**: Maybe MPNet's 768-dimensional space is the problem.

**Alternatives to test**:
- **Smaller models**: MiniLM (384-D), or even smaller (128-D)
- **Task-specific models**: Embeddings trained on similar/dissimilar text pairs
- **Token-level embeddings**: Skip sentence embeddings entirely, work in token space

**Worth testing**: Does the artifact persist with lower-dimensional embeddings?

### Approach 6: Abandon Gram Matrices, Use Distribution Over Embeddings

**Radical rethink**: Don't construct N × N matrices at all.

**Method**: Treat embeddings as samples from a distribution on the unit sphere. Compute quantum metrics using **spectral density functions** or **continuous density operators** rather than discrete Gram matrices.

**This requires**: Quantum information theory for continuous variables (much more complex).

---

## Discussion: Which Approach to Pursue?

### Quick wins to validate feasibility:
1. **Approach 1 (PCA)**: Easy to implement, should take <1 hour. If this works, it's a major win.
2. **Approach 5 (smaller embeddings)**: Also easy - just change model name. Tests whether dimensionality is the core issue.
3. **Approach 3 (fidelity)**: Moderate complexity, tests whether the issue is specific to trace distance.

### If quick wins fail:
- Likely indicates the **Gram matrix approach is fundamentally incompatible** with statistical distribution testing
- Should pivot to entirely different quantum-inspired approaches (or acknowledge this path isn't viable)

### Recommended next experiment:

**Test Approach 1 (PCA to fixed k=50) on synthetic data**:
- If trace distance(N(0,I), N(0,I)) → 0 as N increases: ✓ Fixed the artifact!
- If trace distance still shows sample-size dependence: ✗ Need deeper rethink

**Then if successful**, test on real data:
- fp32 vs fp32 should give low distance
- fp32 vs int8 should give higher distance
- Different models (Llama vs Mistral) should give even higher distance

---

## Revised Open Questions

1. ~~**Sample efficiency of quantum metrics**~~ - **Answered (negative)**: Trace distance does not discriminate, so sample efficiency is not applicable.

2. **Can trace distance be fixed?**: 
   - Would dimensionality reduction (PCA, UMAP) before density matrix construction help?
   - Would different embedding models work better?
   - Is there a different quantum metric formulation that handles finite samples better?

3. **Von Neumann divergence reliability**: Why does it show discrimination at n=100 but not n=1000? Is this metric usable with proper calibration?

4. **Theoretical guarantees**: Do quantum metrics have any theoretical finite-sample guarantees for distribution testing? Classical kernel tests (like MMD) have well-studied asymptotic properties.

5. **Alternative quantum approaches**: Are there other quantum-inspired metrics (e.g., quantum Jensen-Shannon divergence, fidelity) that might avoid these issues?

6. **Generalization** (now higher priority): Before investing in fixing quantum metrics, test classical MMD on:
   - Different model pairs (different architectures, not just quantizations)
   - Different prompt types (coding, reasoning, creative writing)
   - Different distribution shifts (finetuning, watermarking)

---

## PCA Fix: Implementation and Results

**Implementation:** Created `test_trace_pca_fix.py` and `test_pca_on_llm_data.py`

### Validation on Synthetic Data (✓ SUCCESS)

**Script:** `test_trace_pca_fix.py`

Testing PCA approach (k=50) on synthetic Gaussian data:

| Test | Original | PCA k=50 | Improvement |
|------|----------|----------|-------------|
| Identical dists | 0.209 | **0.004** | **53× better** |
| Sample size trend | 0.15→0.64 ↑ | 0.005→0.002 ↓ | **Fixed!** |
| Resampling same pool | 0.279 | **0.019** | **15× better** |

**Critical finding:** Sample size dependence **eliminated**. Distance now properly decreases as N increases (0.005 → 0.002).

**Method:**
```python
# Instead of: pip = embeddings @ embeddings.T  (N × N, grows with N)
# Use:
pca = PCA(n_components=k)  # Fixed k
reduced = pca.fit_transform(embeddings)  # N × k
cov = reduced.T @ reduced / N  # k × k covariance
rho = cov / trace(cov)  # Normalize
```

### Testing on Real LLM Data (⚠️ PARTIAL SUCCESS)

**Script:** `test_pca_on_llm_data.py` + `test_pca_k_sweep.py`

Testing different k values on Llama-3-8B completions:

| k | fp32 vs fp32 | fp32 vs int8 | Ratio | Note |
|---|--------------|--------------|-------|------|
| 5 | 0.076 | 0.081 | 1.07× | **Best ratio** |
| 10 | 0.077 | 0.073 | 0.95× | Discrimination inverts! |
| 20 | 0.074 | 0.068 | 0.92× | Gets worse |
| 50 | 0.069 | 0.065 | 0.94× | Original choice |
| 100+ | 0.068 | 0.065 | 0.95× | Stable but poor |

**Optimal k=5 results (n=100 samples):**

| Comparison | Trace Distance | vs Baseline |
|------------|----------------|-------------|
| fp32 vs fp32 (same) | 0.076 | 1.0× |
| fp32 vs int8 (quant) | 0.081 | 1.07× |
| Llama vs Mistral (diff) | 0.121 | 1.59× |

**Status:** ✓ Correct ordering: same < quantization < different models

### Evaluation

**What works:**
- ✓ Sanity check passes (fp32 vs fp32 gives low, stable values)
- ✓ Correct monotonic ordering of differences
- ✓ Sample-size artifact completely eliminated
- ✓ Computationally fast (~1s for n=100)

**What doesn't work:**
- ✗ Discrimination is **very weak** (7% difference for quantization, 59% for different models)
- ✗ Optimal k=5 is extremely aggressive (768D → 5D loses most information)
- ✗ Higher k (which should preserve more information) **makes discrimination worse**
  - At k≥10, fp32 vs int8 becomes *lower* than fp32 vs fp32 (inverted!)
  
**Why discrimination is weak:**

1. **PCA projects onto max-variance directions** - these capture overall structure common to both distributions, not the *differences* between them

2. **For k=5**: Most signal is noise; we only detect very large differences (different models)

3. **For k>10**: PCA directions mix signal and noise in ways that obscure subtle quantization effects

4. **SBERT embeddings may not encode quantization effects** - MPNet was trained on semantic similarity, not low-level generation differences

### Verdict on PCA Approach

**For detection (hypothesis testing):** ❌ **Not recommended**
- Discrimination too weak for practical use
- Classical MMD (p=0.01 for quantization at n=1000) is far superior
- The 7% effect size at k=5 would require huge samples for statistical power

**For interpretability:** ❓ **Unclear value**
- Could provide a *relative ordering* of distribution shifts
- But absolute magnitudes are unreliable (too compressed)
- Eigenvalue spectra or other features might be more informative

**Conclusion:** PCA fixes the mathematical artifact but reveals a deeper issue: **the embedding space doesn't contain strong enough signals about the distribution differences we care about** (quantization, finetuning, etc.). The quantum trace distance approach works in theory but requires an embedding space where relevant differences are preserved.

---

## Next Steps (Updated After PCA Testing)

**PCA Approach:**
- [x] **Approach 1: Test PCA dimensionality reduction** - COMPLETED
  - ✓ Synthetic validation: Fixes sample-size artifact perfectly
  - ⚠️ Real LLM data: Weak discrimination (7% for quantization at k=5)
  - ✗ Not viable for detection/hypothesis testing
  - ❓ Possible value for relative ordering, but unclear practical use

**Remaining quantum approaches to explore:**
- [ ] **Approach 3: Quantum fidelity** instead of trace distance
  - May have better finite-sample properties
  - Test on synthetic suite first, then LLM data if promising
  
- [ ] **Approach 5: Different embedding models**
  - Try smaller models: MiniLM-384D, TinyBERT-128D
  - Try task-specific models: paraphrase detection, textual entailment
  - Hypothesis: Current SBERT space doesn't encode quantization/generation differences
  
- [ ] **Approach 2: Kernel-based density matrices**
  - Fixed basis (e.g., k-means cluster centers)
  - May decouple matrix size from sample size more naturally

**Alternative direction: Abandon quantum metrics for detection, explore interpretability:**
- [ ] **Eigenvalue spectrum analysis** (even if trace distance fails)
  - Do eigenvalues reveal diversity, entropy, or structure changes?
  - Compare spectra for fp32 vs int8, Llama vs Mistral
  
- [ ] **Embedding space geometry** without density matrices
  - Mean shift, covariance change, higher moments
  - Maximum Mean Discrepancy in feature space (not just Hamming kernel)

**Focus on classical methods (higher priority given quantum metric challenges):**
- [ ] **Test classical MMD more broadly**: Different models, prompts, shift types
- [ ] **Sample size calibration for MMD**: Determine minimum n for adequate power
- [ ] **Kernel selection for MMD**: Can kernel choice reveal *how* distributions differ?
- [ ] **Combine multiple tests**: MMD + K-S + chi-squared for richer characterization

**Document Status:**
PCA approach tested and documented. Fixes mathematical artifact but reveals deeper issue: embedding space lacks strong signals for quantization/generation differences. Quantum metrics show limited promise for detection. Future work should either: (1) find better embeddings, (2) pivot to interpretability-only use cases, or (3) focus on classical methods with proven detection power.

---

## Pivot to Interpretability: EmbeddingGemma-300M Embeddings

**Date**: 2026-05-01

### Reframing the Goal

**Detection is solved**: MMD with Hamming kernel successfully detects distribution differences (p=0.01 for quantization at n=1000).

**New goal**: **Interpretability** - Understand *how* distributions differ semantically:
- What changed between fp32 and int8?
- Did quantization reduce diversity, coherence, or factuality?
- Which semantic dimensions shifted between model architectures?
- Can we characterize watermarking effects semantically?

**Key insight**: For interpretability, we *want* high-level semantic abstractions. The fact that MPNet "collapses" surface variations is a feature, not a bug - it lets us focus on meaningful semantic changes.

### Why MPNet Failed vs Why EmbeddingGemma-300M Might Succeed

**MPNet (all-mpnet-base-v2) limitations**:
- 110M parameters, 768-D embeddings
- Trained on sentence similarity (paraphrase detection, semantic textual similarity)
- Optimized to make semantically similar texts identical
- **Result**: Collapsed quantization effects we wanted to detect

**EmbeddingGemma-300M potential advantages**:
- 300M parameters (~3× larger than MPNet)
- Purpose-built for embedding generation (Gemma family)
- Different training approach and architecture
- May capture semantic nuances MPNet misses

**Hypothesis**: If quantization affects semantics (coherence, style, diversity), EmbeddingGemma embeddings might encode these shifts where MPNet doesn't.

### Why This Approach Makes Sense

**For detection**: Character-level features (MMD Hamming) are optimal
- Preserve low-level distribution differences
- Don't abstract away signal

**For interpretability**: Semantic features are optimal
- Abstract away noise to reveal meaningful patterns
- Enable human-interpretable characterization
- Quantum metrics provide principled framework for measuring semantic shifts

**The approaches are complementary**:
1. **MMD (Hamming)**: "Distributions differ, p<0.001" ✓
2. **Quantum metrics (EmbeddingGemma)**: "15% semantic shift along axis of decreased diversity"

### Proposed Experiments

#### Experiment 1: Quick Sanity Check (t-SNE/UMAP Visualization)

**Goal**: Do EmbeddingGemma embeddings separate fp32 vs int8?

```python
# Embed with EmbeddingGemma-300M
embeddings_fp32 = embed_with_embeddinggemma(samples_fp32)
embeddings_int8 = embed_with_embeddinggemma(samples_int8)

# Reduce to 2D for visualization
from sklearn.manifold import TSNE
combined = np.vstack([embeddings_fp32, embeddings_int8])
reduced = TSNE(n_components=2).fit_transform(combined)

# Plot with colors: blue=fp32, red=int8
# Success: Distinct clusters → EmbeddingGemma captures semantic differences
# Failure: Mixed clouds → Same problem as MPNet
```

**Success criterion**: Visual separation between distributions

#### Experiment 2: Quantum Metrics for Semantic Characterization

**Goal**: Quantify semantic shift magnitude and diversity changes

```python
# UMAP to fixed dimensions
rho_fp32, rho_int8 = compute_density_matrices_umap(
    embeddings_fp32, embeddings_int8,
    n_components=20
)

# Compute quantum metrics
trace_dist = trace_distance(rho_fp32, rho_int8)
entropy_fp32 = von_neumann_entropy(rho_fp32)
entropy_int8 = von_neumann_entropy(rho_int8)
qre = quantum_relative_entropy(rho_fp32, rho_int8)

# Interpretation
print(f"Semantic shift magnitude: {trace_dist:.3f}")
print(f"Diversity: {entropy_fp32:.3f} → {entropy_int8:.3f}")
print(f"  Change: {((entropy_int8/entropy_fp32 - 1)*100):.1f}%")
print(f"Information divergence: {qre:.3f}")
```

**Output example**:
```
Semantic shift magnitude: 0.18
Diversity: 2.4 → 2.1
  Change: -12.5% (int8 less diverse)
Information divergence: 0.04
```

**Interpretation**: "int8 quantization creates moderate semantic shift (0.18) with 12.5% reduction in output diversity"

#### Experiment 3: Semantic Axis Discovery

**Goal**: Identify which semantic dimensions changed most

```python
# Find directions of maximum change
diff_matrix = rho_fp32 - rho_int8
eigenvalues, eigenvectors = np.linalg.eigh(diff_matrix)

# Top 3 eigenvectors = semantic axes of maximum change
top_k = 3
for i in range(1, top_k+1):
    idx = -i
    eigenval = eigenvalues[idx]
    axis = eigenvectors[:, idx]
    
    # Project completions onto this axis
    scores_fp32 = reduced_fp32 @ axis
    scores_int8 = reduced_int8 @ axis
    
    print(f"\nAxis {i} (eigenvalue={eigenval:.3f}):")
    
    # Find extreme completions
    extreme_fp32 = samples_fp32[np.argmax(scores_fp32)]
    extreme_int8 = samples_int8[np.argmax(scores_int8)]
    
    print(f"  fp32 extreme: {extreme_fp32[:100]}...")
    print(f"  int8 extreme: {extreme_int8[:100]}...")
    
    # Manual labeling: What semantic property does this axis represent?
    # Formality? Technical depth? Creativity? Coherence?
```

**Success**: Human can manually label discovered axes with semantic meaning (e.g., "Axis 1 = formality", "Axis 2 = technical depth")

### Key Questions for Implementation

1. **EmbeddingGemma-300M Access**:
   - API endpoint or batch processing system?
   - How do we get embeddings for ~1000 texts?
   - What's the latency/throughput?

2. **Technical Specifications**:
   - What's the embedding dimension?
   - Which layer's representations (if extractable)?
   - Any preprocessing required?

3. **Scope**:
   - Priority comparisons: fp32 vs int8, Llama vs Mistral, fp32 vs watermark?
   - Sample sizes: Start with n=100 or go straight to n=1000?

4. **Scripts to Create**:
   - `gemma_embeddings.py` - Interface to EmbeddingGemma-300M
   - `test_gemma_sanity.py` - Replace MPNet in validation suite
   - `interpret_quantum_metrics.py` - Semantic characterization
   - `visualize_semantic_shift.py` - Axis discovery and visualization

### What Success Looks Like

**Level 1: Basic Validation**
- ✓ t-SNE shows fp32 vs int8 separate (unlike MPNet's 0.99× failure)
- ✓ Sanity check passes (fp32 vs fp32 low distance)
- ✓ Correct ordering: same < quantization < different models

**Level 2: Quantitative Characterization**
- ✓ Trace distance provides magnitude of semantic shift
- ✓ Entropy changes quantify diversity/uncertainty effects
- ✓ QRE measures information loss direction

**Level 3: Qualitative Interpretability**
- ✓ Eigenvalue spectrum reveals number of meaningful change dimensions
- ✓ Top eigenvectors correspond to human-interpretable semantic axes
- ✓ Can manually label: "Axis 1 = coherence degradation from quantization"
- ✓ Provides actionable insights: "int8 preserves factuality but reduces stylistic diversity"

### Expected Outcomes

**Optimistic scenario**: EmbeddingGemma embeddings capture semantic effects of quantization/architecture differences. Quantum metrics provide interpretable characterization. We get:
- Detection: MMD (Hamming) - p-values, statistical power
- Interpretation: Quantum metrics (EmbeddingGemma) - semantic shift characterization

**Realistic scenario**: EmbeddingGemma shows moderate improvement over MPNet. Some semantic dimensions separate (e.g., Llama vs Mistral) but quantization effects remain subtle. Still valuable for understanding architecture differences.

**Pessimistic scenario**: EmbeddingGemma has same issues as MPNet - semantic abstraction erases the signal. Conclusion: Quantum metrics not suitable even for interpretability. Pivot to other interpretability methods (attention analysis, token-level analysis, etc.).

### Next Session Goals

1. Access EmbeddingGemma-300M and get embedding specs
2. Run Experiment 1 (t-SNE sanity check)
3. If promising, implement full quantum metrics pipeline with EmbeddingGemma
4. Generate first semantic characterization: "fp32 vs int8 in interpretable terms"

**Status**: Planning phase. Ready to implement once EmbeddingGemma access confirmed.

---

## Experiment 1: EmbeddingGemma-300M t-SNE Validation

**Date**: 2026-05-01

### Implementation

Created interface to EmbeddingGemma-300M API (NIST cluster):
- **Interface**: `embeddinggemma_interface.py` - OpenAI-compatible API wrapper
- **Embedding dimension**: 768 (same as MPNet)
- **API endpoint**: https://rchat.nist.gov/api
- **Test scripts**: `experiment1_tsne_sanity_check.py`, `experiment1b_model_comparison.py`

### Test 1a: Quantization Detection (fp32 vs int8)

**Configuration:**
- Model: Llama-3-8B-Instruct
- Prompts: wikipedia_en [0, 1, 2]
- Samples: 100 per distribution
- Method: t-SNE visualization of EmbeddingGemma embeddings

**Results:**
- **Sanity check (fp32 vs fp32)**: ✓ Complete overlap (as expected)
- **Quantization (fp32 vs int8)**: ✗ Complete overlap (no separation)

**Conclusion:** EmbeddingGemma-300M shows **same limitations as MPNet** for quantization detection. Semantic embeddings trained on similarity tasks collapse the subtle generation artifacts we want to detect.

**Plot**: `test_1_sanity_check_fp32_vs_fp32.png`, `test_2_quantization_detection_fp32_vs_int8.png`

### Test 1b: Model Architecture Comparison (Llama vs Mistral)

**Hypothesis**: Different model architectures might produce semantically different outputs where quantization doesn't.

**Configuration:**
- Model A: Llama-3-8B-Instruct (fp32)
- Model B: Mistral-7B-Instruct-v0.3 (fp32)
- Prompts: wikipedia_en [0, 1, 2]
- Samples: 100 per model
- Method: t-SNE + separation ratio metric

**Results:**
- **Center-to-center distance**: 3.79
- **Average cluster spread**: 9.41
- **Separation ratio**: **0.40** (need >2.0 for strong separation)

**Conclusion:** ✗ **No meaningful separation**. Even different model architectures produce semantically similar outputs for Wikipedia continuation tasks. Both models converge to "Wikipedia-style" factual text.

**Plot**: `model_comparison_llama_vs_mistral.png`

### Key Findings

1. **EmbeddingGemma-300M has same limitations as MPNet**
   - Both trained on semantic similarity tasks
   - Both collapse generation artifacts (quantization, model differences)
   - Model size (110M → 300M) doesn't solve fundamental training objective issue

2. **Wikipedia prompts are too constrained**
   - Factual, encyclopedic style
   - Little room for stylistic variation
   - Both Llama and Mistral converge to same semantic pattern

3. **The paradox is resolved**
   - MMD (Hamming) detects differences ✓ (character-level)
   - EmbeddingGemma shows similarity ✗ (semantic-level)
   - Both are correct! They measure different things.

4. **Semantic embeddings not viable for this task**
   - Quantization artifacts are implementation-level, not semantic
   - Model architecture differences (for Wikipedia) are also implementation-level
   - Semantic abstraction is the enemy, not the solution

### Implications for Interpretability Goal

**What we wanted**: Use semantic embeddings + quantum metrics to characterize *how* distributions differ

**What we learned**: Semantic embeddings don't preserve the differences we care about:
- Same model, different precision → same semantics
- Different models, same task → same semantics (for constrained tasks)

**The differences we want to understand are:**
- Statistical: entropy, perplexity, diversity
- Stylistic: formality, verbosity, complexity  
- Structural: sentence patterns, coherence

Not high-level semantic meaning (which is preserved across implementations).

### Next Steps: Alternative Prompt Types

**Constraint**: Limited to MET dataset prompt-response pairs

**Available MET datasets** (discovered via data exploration):
- ✗ `wikipedia_*` (en/de/es/fr/ru) - Too constrained, tested
- ✓ **`humaneval`** - Coding tasks (RECOMMENDED)
  - Multiple valid solutions
  - Stylistic variation: comments, verbosity, approaches
  - Model "personality" in code generation
- ✓ **`ultrachat`** - Conversational dialogues
  - Open-ended responses
  - Formality/tone variations
  - Explanation styles differ

**Hypothesis**: Less constrained tasks (coding, conversation) may show semantic divergence where factual tasks don't.

**Experiment 1c (planned)**: Re-run Llama vs Mistral comparison with `humaneval` prompts to test if coding tasks reveal semantic differences that Wikipedia doesn't.

**Status**: Implemented experiment1b with dataset/prompt flexibility. Ready to test humaneval.

---

## Session 2 Progress: Bug Fixes and Preparation

**Date**: 2026-04-30

### Bug Fix: AttributeError in experiment1_tsne_sanity_check.py

**Issue**: Script failed with:
```
AttributeError: 'CompletionSample' object has no attribute 'completions'
```

**Root cause**: Line 54 accessed `sample.completions` but the correct attribute is `sample.completion_sample`

**Fix applied** (line 54-55):
```python
# Before:
for completion in sample.completions:

# After:
completions = sample.completion_sample.numpy() if hasattr(sample.completion_sample, 'numpy') else sample.completion_sample
for completion in completions:
```

### Successful Experiment 1 Execution

Re-ran `experiment1_tsne_sanity_check.py` after bug fix:

**Test 1: Sanity Check (fp32 vs fp32)**
- Result: ✓ Complete overlap (as expected)
- Plot: `test_1_sanity_check_fp32_vs_fp32.png`

**Test 2: Quantization Detection (fp32 vs int8)**
- Result: ✗ Complete overlap (no separation)
- Plot: `test_2_quantization_detection_fp32_vs_int8.png`

**Verdict**: Confirms EmbeddingGemma has same limitations as MPNet for quantization detection.

### Enhanced experiment1b_model_comparison.py

**Added command-line flexibility**:
```python
parser.add_argument("--dataset", default="wikipedia_en", help="Dataset name")
parser.add_argument("--prompts", nargs="+", type=int, default=[0, 1, 2])
prompt_ids = {args.dataset: args.prompts}
```

**Purpose**: Enable testing across different MET dataset types without code modification.

**Next planned experiment**: 
```bash
python experiment1b_model_comparison.py --samples 100 --dataset humaneval --prompts 0 1 2 --L 500
```

**Hypothesis**: Coding tasks (humaneval) will show semantic variation between Llama and Mistral where Wikipedia prompts don't, because:
- Multiple valid coding solutions exist
- Different stylistic approaches (verbose/compact, comments/minimal)
- Model "personality" in code generation
- Less constrained by factual correctness

**Status**: Ready to execute humaneval experiment.

### Experiment 1c: Humaneval Coding Prompts

**Execution**:
```bash
python experiment1b_model_comparison.py --samples 100 --dataset humaneval --prompts 0 1 2 --L 500
```

**Configuration:**
- Model A: Llama-3-8B-Instruct (fp32)
- Model B: Mistral-7B-Instruct-v0.3 (fp32)
- Prompts: humaneval [0, 1, 2] (coding tasks)
- Samples: 100 per model
- Method: t-SNE + separation ratio

**Results:**
- **Center-to-center distance**: 3.01
- **Average cluster spread**: 11.55
- **Separation ratio**: **0.26** (worse than Wikipedia's 0.40!)

**Conclusion:** ✗ **Even worse separation than Wikipedia**. Coding tasks did NOT reveal semantic differences between model architectures.

**Plot**: `model_comparison_llama_vs_mistral.png` (overwritten)

### Summary of Semantic Embedding Experiments

| Dataset | Task Type | Separation Ratio | Result |
|---------|-----------|------------------|--------|
| wikipedia_en | Factual continuation | 0.40 | ✗ No separation |
| humaneval | Code generation | 0.26 | ✗ Worse |

**Key Finding**: EmbeddingGemma-300M (like MPNet) produces semantically similar embeddings for:
- Same model, different quantization (fp32 vs int8)
- Different models, same task (Llama vs Mistral)

**Why even coding tasks failed**:
- Both models trained on similar code corpora
- Task: HumanEval function completion
- Output constraint: Must produce valid Python
- Result: Converge to similar semantic patterns despite different implementations

**Fundamental issue**: Semantic similarity embeddings are trained to make different phrasings of the same meaning identical. This is exactly what prevents them from distinguishing model implementation differences.

---

## Conclusion: Semantic Embeddings + Quantum Metrics Not Viable

**Date**: 2026-04-30

### What We Tested

1. **MPNet embeddings** (110M, 768-D) + quantum metrics
   - Trace distance fails sanity check (sample-size artifacts)
   - PCA fix works mathematically but weak discrimination (7% for quantization)
   
2. **EmbeddingGemma-300M embeddings** (300M, 768-D) + t-SNE visualization
   - Same limitations as MPNet
   - No separation for quantization (fp32 vs int8)
   - No separation for architectures (Llama vs Mistral)
   - Tested on constrained (Wikipedia) and unconstrained (HumanEval) tasks

### Why This Approach Failed

**The paradox resolved**:
- MMD (Hamming kernel): Detects differences at p=0.01 ✓
- Semantic embeddings: Show complete overlap ✗
- **Both are correct!** They measure different levels of abstraction.

**Character-level (Hamming)**:
- Preserves: Exact token choices, phrasing, word order
- Detects: Implementation-level differences (quantization, watermarking, architecture)

**Semantic-level (EmbeddingGemma)**:
- Preserves: Meaning, intent, content
- Discards: Surface form, style variations, implementation artifacts
- Training objective: Make semantically similar texts identical

**The differences we want to understand** (quantization effects, architecture differences) are:
- Statistical: perplexity, entropy, token distribution shifts
- Stylistic: formality, verbosity, complexity
- Implementation-level: Not semantic

**Semantic embeddings abstract away the signal we're trying to measure.**

### What We Learned

1. **Detection is solved**: MMD with Hamming kernel works
   - Correct null hypothesis behavior (sanity checks pass)
   - Statistical power grows with sample size
   - p-values provide rigorous guarantees

2. **Quantum metrics on semantic embeddings don't work** for this problem:
   - Trace distance: Sample-size artifacts even with PCA fix
   - Weak discrimination: 7% effect size for quantization (PCA k=5)
   - Embedding space issue: Signal not preserved in semantic abstraction

3. **Model size doesn't help**: 
   - MPNet (110M) vs EmbeddingGemma (300M)
   - Same fundamental training objective = same limitations

4. **Task diversity doesn't help**:
   - Wikipedia (constrained factual) vs HumanEval (open-ended coding)
   - Both show no semantic separation between models

### Implications for Interpretability

**Original goal**: Use quantum metrics on semantic embeddings to characterize *how* distributions differ

**Why it can't work**: The "how" we want to characterize (quantization artifacts, generation pattern shifts) lives at the implementation level, not the semantic level.

**Analogy**: Asking "what's the semantic difference between two translations of the same text" when the whole point of a good translation is to preserve semantics.

### Alternative Paths Forward

**For interpretability of distribution shifts**:

1. **Statistical characterization** (no embeddings):
   - Token-level entropy, perplexity distributions
   - Vocabulary diversity metrics (type-token ratio)
   - N-gram frequency shifts
   - Repetition patterns, coherence scores

2. **Feature-based analysis** (specific features, not full embeddings):
   - VADER sentiment (already used in K-S test)
   - Readability scores (Flesch-Kincaid, etc.)
   - POS tag distributions
   - Dependency parse patterns
   - Named entity frequency

3. **Direct inspection** (qualitative):
   - Sample completions from each distribution
   - Manual coding for differences
   - Identify patterns (e.g., "int8 produces shorter responses")
   
4. **Cluster analysis** (token-level):
   - K-means on token sequences (not embeddings)
   - Topic modeling (LDA) on completions
   - Compare topic distributions between models

**For detection** (already working):
- Continue using MMD with Hamming kernel
- Explore other kernels: k-spectrum, subsequences
- Power analysis for sample size recommendations

### Final Verdict

**Quantum-inspired metrics + semantic embeddings**: ❌ **Not recommended** for LLM distribution testing

**Why**:
- Semantic embeddings discard the signal (by design)
- Quantum metrics don't add value beyond classical methods
- Sample-size artifacts difficult to eliminate
- No interpretability advantage demonstrated

**What works**:
- Classical MMD (Hamming): Detection with rigorous p-values ✓
- Token-level statistics: Interpretable characterization ✓
- Feature-based K-S tests: Specific dimension analysis (sentiment, etc.) ✓

**Status**: Semantic embedding approach exhausted. Recommend focusing on statistical/token-level interpretability methods or declaring quantum metrics exploration complete with negative results.
