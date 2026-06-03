# MMD on Embeddings: A Clean Alternative to Quantum Metrics

## The Proposal

Calculate **Maximum Mean Discrepancy (MMD)** between two sets of SBERT embeddings and use permutation testing (or asymptotic MMD distribution) to find the null distribution for calibration.

**Key insight**: This gives you all the benefits of quantum metrics without the problems.

---

## Why This Works

**MMD on embeddings** gives you everything quantum metrics promised but without the headaches:

✅ **Well-calibrated**: Permutation testing gives you valid p-values  
✅ **Effect size**: MMD value itself is interpretable (average kernel distance between distributions)  
✅ **No sample-size artifacts**: Unlike trace distance, MMD doesn't have weird bounds  
✅ **Fits existing framework**: Your codebase already has MMD + permutation infrastructure  
✅ **Asymptotic option**: Yes, MMD has an asymptotic null distribution (Gretton et al., "A Kernel Two-Sample Test")  

---

## How It Would Work

```python
from model_equality_testing.algorithm import run_two_sample_test

# Just like your existing MMD tests, but on embeddings
pvalue, mmd_stat = run_two_sample_test(
    sample_a, sample_b,
    stat_type="mmd_rbf_embeddings",  # New test type
    pvalue_type="permutation_pvalue",
    b=1000  # number of permutations
)

print(f"MMD = {mmd_stat:.4f}, p = {pvalue:.4f}")
# "MMD = 0.0347, p = 0.002" ← calibrated and interpretable!
```

---

## Implementation

### Add to `tests.py`

```python
def mmd_rbf_embeddings(sample1, sample2, gamma=None, _precomputed_embeddings=None):
    """
    MMD with RBF (Gaussian) kernel on SBERT embeddings.
    
    Args:
        sample1, sample2: CompletionSample objects
        gamma: RBF bandwidth parameter (if None, use median heuristic)
        _precomputed_embeddings: Optional (emb1, emb2) tuple to reuse embeddings
    
    Returns:
        MMD statistic (non-negative float)
    """
    if _precomputed_embeddings is None:
        from model_equality_testing.src.embeddings import embed_sample
        emb1 = embed_sample(sample1)
        emb2 = embed_sample(sample2)
    else:
        emb1, emb2 = _precomputed_embeddings
    
    # Median heuristic for gamma if not specified
    if gamma is None:
        from scipy.spatial.distance import pdist
        all_emb = np.vstack([emb1, emb2])
        pairwise_dists = pdist(all_emb, 'euclidean')
        gamma = 1.0 / (2 * np.median(pairwise_dists)**2)
    
    # RBF kernel: k(x,y) = exp(-gamma * ||x-y||^2)
    def rbf_kernel(X, Y):
        from scipy.spatial.distance import cdist
        sq_dists = cdist(X, Y, 'sqeuclidean')
        return np.exp(-gamma * sq_dists)
    
    K_xx = rbf_kernel(emb1, emb1)
    K_yy = rbf_kernel(emb2, emb2)
    K_xy = rbf_kernel(emb1, emb2)
    
    n1, n2 = len(emb1), len(emb2)
    
    # Unbiased MMD^2 estimator
    mmd_sq = (K_xx.sum() - np.trace(K_xx)) / (n1 * (n1 - 1))
    mmd_sq += (K_yy.sum() - np.trace(K_yy)) / (n2 * (n2 - 1))
    mmd_sq -= 2 * K_xy.mean()
    
    return np.sqrt(max(0, mmd_sq))
```

### Register in `registry.py`

```python
IMPLEMENTED_TESTS['mmd_rbf_embeddings'] = 'model_equality_testing.src.tests:mmd_rbf_embeddings'
```

---

## Permutation vs Asymptotic Null

### Permutation Testing (Recommended for n < 10,000)

**Advantages**:
- Already implemented in your `pvalue.py`
- Exact finite-sample validity
- No distributional assumptions
- Works for any sample size

**How it works**:
1. Pool both samples
2. Randomly permute labels (A vs B)
3. Recalculate MMD on permuted data
4. Repeat B times
5. P-value = proportion of permuted MMDs ≥ observed MMD

**Implementation**: No changes needed, just call with `pvalue_type="permutation_pvalue"`

### Asymptotic Distribution (For large n)

**Advantages**:
- Much faster (no resampling)
- MMD² follows weighted sum of χ² under null (Gretton et al., 2012)

**Disadvantages**:
- Requires eigendecomposition of kernel matrix
- More complex to implement
- Asymptotic approximation may be poor for small n

**Asymptotic null** (under H₀: distributions are equal):
```
MMD² → Σ λᵢ(zᵢ² - 1)
```
where λᵢ are eigenvalues of the centered kernel matrix and zᵢ ~ N(0,1).

**Implementation complexity**: ~2-3 hours to add to `pvalue.py`

**Recommendation**: Start with permutation testing. Add asymptotic version later if runtime becomes an issue.

---

## Comparison to Quantum Metrics

| Aspect | Quantum Metrics | MMD on Embeddings |
|--------|----------------|-------------------|
| **Calibration** | ❌ No p-values without hacky permutation | ✅ Direct permutation testing |
| **Effect size** | ❌ Trace distance has weird bounds | ✅ MMD is interpretable distance |
| **Sample size** | ❌ Bounded by min(n_A, n_B) | ✅ No artificial bounds |
| **Framework fit** | ❌ Separate implementation | ✅ Extends existing MMD tests |
| **Interpretation** | ❌ Eigenvalue spectra hard to explain | ⚠️ Still need semantic axes |
| **Computation** | ❌ Expensive (SVD of n×n matrix) | ⚠️ Similar (kernel matrix O(n²)) |
| **Theory** | ⚠️ Novel, less established | ✅ Well-studied (Gretton et al.) |

**Bottom line**: MMD on embeddings is simpler, better calibrated, and fits your existing framework.

---

## Full Workflow: Detection + Interpretation

```python
from model_equality_testing.algorithm import run_two_sample_test
from model_equality_testing.src.embeddings import embed_sample
from model_equality_testing.src.semantic_axes import interpret_difference, load_axis_from_jsonl

# Step 1: Test for difference (MMD + permutation)
pvalue, mmd_stat = run_two_sample_test(
    sample_a, sample_b,
    stat_type="mmd_rbf_embeddings",
    pvalue_type="permutation_pvalue",
    b=1000
)

print(f"Detection: MMD = {mmd_stat:.4f}, p = {pvalue:.4f}")

# Step 2: If significant, interpret with semantic axes
if pvalue < 0.05:
    # Embed once (reuse for both MMD and semantic axes)
    emb_a = embed_sample(sample_a)
    emb_b = embed_sample(sample_b)
    
    # Load semantic axes
    axes = [
        load_axis_from_jsonl("professionalism.jsonl", "casualness.jsonl", "prof-casual"),
        load_axis_from_jsonl("formality.jsonl", "informality.jsonl", "formal-informal"),
        # ... more axes
    ]
    
    # Interpret what differs
    interp = interpret_difference(emb_a, emb_b, axes, "fp32", "int8")
    print("\nInterpretation:")
    print(interp.summary())
```

**Example output**:

```
Detection: MMD = 0.0347, p = 0.002

Interpretation:
Semantic Interpretation (3 axes, sorted by effect size):

  professionalism ← → casualness:  Δ = +0.54  d = 0.82  p < 0.001 *
    Sample A is more toward casualness

  formality ← → informality:       Δ = +0.38  d = 0.61  p = 0.003 *
    Sample A is more toward informality

  technical ← → layperson:         Δ = -0.21  d = 0.31  p = 0.15
    No significant difference (p > 0.05 after FDR correction)

* = significant after Benjamini-Hochberg correction
```

**What you get**:
1. **Detection**: "The distributions differ" (MMD p-value)
2. **Magnitude**: "The difference is this big" (MMD statistic)
3. **Interpretation**: "They differ in these ways" (semantic axes)

---

## Kernel Choice

### RBF (Gaussian) Kernel - Recommended

```python
k(x, y) = exp(-γ ||x - y||²)
```

**Advantages**:
- Universal kernel (can detect any distributional difference given enough data)
- Well-studied for MMD
- Smooth, continuous

**Bandwidth selection**:
- **Median heuristic** (implemented above): γ = 1 / (2 * median²(pairwise distances))
- Works well in practice
- No hyperparameter tuning needed

### Alternative Kernels

**Linear kernel**: `k(x, y) = x · y`
- Fast, but only detects mean differences
- Not recommended (too weak)

**Polynomial kernel**: `k(x, y) = (x · y + c)^d`
- Detects higher-order moments
- Requires tuning d (degree) and c (constant)

**Recommendation**: Stick with RBF + median heuristic.

---

## Implementation Time Estimate

**Phase 1: Basic MMD on embeddings** (2-3 hours)
1. Implement `mmd_rbf_embeddings()` in `tests.py` (1 hour)
2. Register in `registry.py` (5 min)
3. Write unit tests (1 hour)
4. Test on real data (fp32 vs int8) (30 min)

**Phase 2: Integration with semantic axes** (1 hour)
5. Test full workflow: MMD detection → semantic interpretation
6. Verify embeddings are reused (not recomputed)

**Phase 3: Asymptotic null (optional)** (2-3 hours)
7. Implement eigenvalue-based null distribution in `pvalue.py`
8. Add `pvalue_type="mmd_asymptotic"` option
9. Benchmark against permutation testing

**Total**: 3-4 hours for fully functional MMD on embeddings with permutation testing.

---

## Relationship to Original Paper

Your paper (arxiv.org/abs/2410.20247) already uses:
- MMD with Hamming kernel on token sequences
- MMD with k-spectrum kernel on token sequences
- Permutation testing for p-values

**This extends the same framework to embedding space**:
- Same MMD theory
- Same permutation testing infrastructure
- Same interpretability challenges → solved with semantic axes

**Continuity**: You're not adding a new methodology, just applying your existing approach to a richer representation (embeddings vs tokens).

---

## Bottom Line

**Yes, you're absolutely right.** MMD on embeddings with permutation testing gives you exactly what you need:

✅ **Calibrated p-values** (answers "is it significant?")  
✅ **Effect magnitude** (MMD value)  
✅ **Fits your existing framework** (extends current MMD tests)  
✅ **No quantum metric headaches** (no sample-size bounds, no calibration mysteries)  

**Semantic axes still add value** by explaining *what* differs (professionalism, formality, etc.), but now you have a statistically sound detection layer.

**Recommended approach**:
1. **Detection**: MMD on embeddings (this document)
2. **Interpretation**: Semantic axes (already planned in `semantic_axes.py`)
3. **Skip**: Quantum metrics (too many problems, not enough benefit)

---

## Three-Level Hierarchical Testing Framework

### Overview

The MMD framework enables a **hierarchical testing strategy** with consistent methodology across all levels:

**Level 1: Token (Syntactic)**  
→ Test: MMD with Hamming/k-spectrum kernel on token sequences  
→ Detects: Character-level, word-level changes  
→ Question: "Did the token distribution change?"

**Level 2: Embedding (Semantic)**  
→ Test: MMD with RBF kernel on full 768-D embeddings  
→ Detects: Any semantic shift in the high-dimensional space  
→ Question: "Did the semantic distribution change?"

**Level 3: Axis (Interpretable)**  
→ Test: MMD on 1D projections along each semantic axis  
→ Detects: Distributional differences along specific interpretable dimensions  
→ Question: "HOW did it change? Which semantic properties shifted?"

---

### Level 3: MMD on Axis Projections

After detecting semantic changes at Level 2, quantify the "strength" of each semantic axis using MMD on projected values.

#### Implementation

```python
def mmd_on_axis_projection(emb_a, emb_b, axis, gamma=None):
    """
    MMD on 1D projections along a semantic axis.
    
    Args:
        emb_a, emb_b: Embeddings (N×768 arrays)
        axis: SemanticAxis object
        gamma: RBF bandwidth (if None, use median heuristic on 1D data)
    
    Returns:
        MMD statistic on projected values
    """
    # Project onto axis
    proj_a = emb_a @ axis.axis_vector  # (N_a,)
    proj_b = emb_b @ axis.axis_vector  # (N_b,)
    
    # Reshape to (N, 1) for kernel computation
    proj_a = proj_a.reshape(-1, 1)
    proj_b = proj_b.reshape(-1, 1)
    
    # Median heuristic for 1D data
    if gamma is None:
        from scipy.spatial.distance import pdist
        all_proj = np.vstack([proj_a, proj_b])
        pairwise_dists = pdist(all_proj, 'euclidean')
        gamma = 1.0 / (2 * np.median(pairwise_dists)**2)
    
    # RBF kernel on 1D projections
    def rbf_kernel_1d(X, Y):
        sq_dists = (X[:, None, 0] - Y[None, :, 0])**2
        return np.exp(-gamma * sq_dists)
    
    K_xx = rbf_kernel_1d(proj_a, proj_a)
    K_yy = rbf_kernel_1d(proj_b, proj_b)
    K_xy = rbf_kernel_1d(proj_a, proj_b)
    
    n1, n2 = len(proj_a), len(proj_b)
    
    # Unbiased MMD^2
    mmd_sq = (K_xx.sum() - np.trace(K_xx)) / (n1 * (n1 - 1))
    mmd_sq += (K_yy.sum() - np.trace(K_yy)) / (n2 * (n2 - 1))
    mmd_sq -= 2 * K_xy.mean()
    
    return np.sqrt(max(0, mmd_sq))


def interpret_with_mmd(emb_a, emb_b, axes, b=1000):
    """
    Interpret differences using MMD on each axis projection.
    
    Args:
        emb_a, emb_b: Embeddings (N×768 arrays)
        axes: List of SemanticAxis objects
        b: Number of permutations
    
    Returns:
        List of dicts with axis_name, mmd, p_value, p_corrected
    """
    results = []
    
    for axis in axes:
        # Observed MMD on this axis
        obs_mmd = mmd_on_axis_projection(emb_a, emb_b, axis)
        
        # Permutation test
        combined = np.vstack([emb_a, emb_b])
        n_a = len(emb_a)
        
        null_mmds = []
        for _ in range(b):
            perm_idx = np.random.permutation(len(combined))
            perm_a = combined[perm_idx[:n_a]]
            perm_b = combined[perm_idx[n_a:]]
            null_mmd = mmd_on_axis_projection(perm_a, perm_b, axis)
            null_mmds.append(null_mmd)
        
        # P-value
        p_value = (np.array(null_mmds) >= obs_mmd).sum() / b
        
        results.append({
            'axis_name': axis.name,
            'mmd': obs_mmd,
            'p_value': p_value
        })
    
    # FDR correction
    from scipy.stats import false_discovery_control
    p_values = [r['p_value'] for r in results]
    corrected = false_discovery_control(p_values)
    
    for r, p_corr in zip(results, corrected):
        r['p_corrected'] = p_corr
    
    return results
```

---

### Complete Unified Workflow

```python
from model_equality_testing.algorithm import run_two_sample_test
from model_equality_testing.src.embeddings import embed_sample
from model_equality_testing.src.semantic_axes import load_all_axes

# Level 1: Token-level detection
p_token, mmd_token = run_two_sample_test(
    sample_a, sample_b,
    stat_type="mmd_hamming",
    pvalue_type="permutation_pvalue",
    b=1000
)
print(f"Level 1 (Token):     MMD = {mmd_token:.4f}, p = {p_token:.4f}")

# Level 2: Embedding-level detection
p_emb, mmd_emb = run_two_sample_test(
    sample_a, sample_b,
    stat_type="mmd_rbf_embeddings",
    pvalue_type="permutation_pvalue",
    b=1000
)
print(f"Level 2 (Embedding): MMD = {mmd_emb:.4f}, p = {p_emb:.4f}")

# Level 3: Axis-level interpretation (if Level 2 significant)
if p_emb < 0.05:
    # Embed once
    emb_a = embed_sample(sample_a)
    emb_b = embed_sample(sample_b)
    
    # Load semantic axes
    axes = load_all_axes()
    
    # Test each axis
    axis_results = interpret_with_mmd(emb_a, emb_b, axes, b=1000)
    
    print("\nLevel 3 (Axis-level interpretation):")
    for r in sorted(axis_results, key=lambda x: x['mmd'], reverse=True):
        sig = "*" if r['p_corrected'] < 0.05 else ""
        print(f"  {r['axis_name']:30s} MMD={r['mmd']:.4f} p={r['p_corrected']:.4f} {sig}")
```

**Example output**:

```
Level 1 (Token):     MMD = 0.0521, p = 0.001
Level 2 (Embedding): MMD = 0.0347, p = 0.002

Level 3 (Axis-level interpretation):
  professionalism-casualness     MMD=0.0821 p=0.001 *
  formality-informality          MMD=0.0634 p=0.003 *
  technical-layperson            MMD=0.0189 p=0.234
```

**Interpretation**: 
- Token and embedding distributions both differ significantly
- The difference is primarily along professionalism and formality axes
- Technical language shows no significant distributional shift

---

### MMD vs T-test for 1D Projections

For Level 3 axis interpretation, there are two complementary approaches:

#### MMD on 1D Projections

**Advantages**:
- ✅ **Unified framework**: Same test at all three levels
- ✅ **Detects non-location shifts**: Can catch changes in variance, skew, multimodality
- ✅ **Kernel flexibility**: RBF kernel adapts to data scale

**Disadvantages**:
- ⚠️ **More complex**: Overkill for simple mean differences
- ⚠️ **Less familiar**: Reviewers expect t-tests for 1D comparisons
- ⚠️ **Computation**: Slower than t-test (though still fast for 1D)

#### T-test (Welch's)

**Advantages**:
- ✅ **Standard**: Reviewers immediately understand
- ✅ **Fast**: Closed-form computation
- ✅ **Effect size**: Hedges' g is interpretable
- ✅ **Directional**: Shows which pole increased

**Disadvantages**:
- ❌ **Only mean**: Misses changes in shape/variance
- ❌ **Assumes normality**: (Though robust to violations)

---

### Recommendation: Hybrid Approach

**Use both** to get complementary information:

1. **T-test** for mean differences and direction (Hedges' g effect size)
2. **MMD** for distributional differences (shape changes)

This gives you:
- Standard interpretable metrics (Δmean, Hedges' g, direction)
- Sensitivity to non-location shifts (MMD catches variance/skew changes)
- Unified MMD story across all three levels

#### Enhanced AxisProjectionResult

```python
@dataclass
class AxisProjectionResult:
    axis_name: str
    negative_pole: str
    positive_pole: str
    
    # Mean-based metrics (t-test)
    mean_a: float             # mean projection, sample A
    mean_b: float             # mean projection, sample B
    delta: float              # mean_a - mean_b (directional)
    cohens_d: float           # Hedges' g (bias-corrected)
    t_pvalue: float           # Welch's t-test p-value
    
    # Distribution-based metrics (MMD)
    mmd: float                # MMD on 1D projections
    mmd_pvalue: float         # Permutation p-value
    
    # Simple distance (already planned)
    wasserstein: float        # 1D Wasserstein distance
    
    # After FDR correction
    t_pvalue_corrected: float
    mmd_pvalue_corrected: float
```

#### Complete Paper Narrative

**Three numbers per axis**:
1. **MMD statistic**: Calibrated distributional effect magnitude
2. **P-value**: Statistical significance (with FDR correction)
3. **Direction**: Which pole increased (Δmean from t-test, Hedges' g)

**Example statement**: 

> "The professionalism axis showed significant distributional shift (MMD = 0.082, p_FDR = 0.001), with int8 outputs shifted toward the casualness pole (Δ = +0.54, d = 0.82). The formality axis showed a similar pattern (MMD = 0.063, p_FDR = 0.003, Δ = +0.38, d = 0.61). Technical language showed no significant difference (MMD = 0.019, p_FDR = 0.234)."

---

### What You Get: Complete Story

#### Paper Narrative

1. **Syntactic detection**: "We detected syntactic changes (token-level MMD = 0.052, p < 0.001)"
2. **Semantic detection**: "We confirmed semantic changes (embedding-level MMD = 0.035, p = 0.002)"
3. **Interpretable characterization**: "We identified that professionalism decreased (MMD = 0.082, Δ = +0.54 toward casualness, p = 0.001) and formality decreased (MMD = 0.063, Δ = +0.38 toward informality, p = 0.003)"

#### Methodological Consistency

- **All levels use MMD**: Token → Embedding → Axes
- **All levels use permutation testing**: Calibrated p-values throughout
- **Each level answers a different question**: 
  - Level 1: "Did syntax change?"
  - Level 2: "Did semantics change?"
  - Level 3: "HOW did semantics change?"

#### Complementary Metrics at Level 3

- **MMD**: Unified framework, detects distributional shape changes
- **T-test + Hedges' g**: Standard metrics, shows direction and effect size
- **Wasserstein**: Simple 1D distance metric

---

### Implementation Priority

#### Phase 1: Core functionality (3-4 hours)
1. Implement `mmd_rbf_embeddings()` (Level 2)
2. Extend `semantic_axes.py` with hybrid approach (Level 3)
3. Test on fp32 vs int8 data

#### Phase 2: Polish (1-2 hours)
4. Add `mmd_on_axis_projection()` helper
5. Update `AxisProjectionResult` dataclass with MMD fields
6. Test unified workflow across all three levels

#### Phase 3: Documentation (1 hour)
7. Update README with hierarchical testing example
8. Add comparison table: MMD vs t-test vs Wasserstein

**Total**: 5-7 hours for complete hierarchical framework

---

### Summary: Hierarchical Testing Benefits

✅ **Methodological consistency**: MMD at all levels  
✅ **Calibrated throughout**: Permutation p-values at every step  
✅ **Interpretable**: Combines distributional testing (MMD) with standard metrics (Hedges' g)  
✅ **Complete story**: Syntactic → Semantic → Interpretable characterization  
✅ **Extends existing work**: Builds on your paper's MMD framework  

This hierarchical approach gives you the best of all worlds: rigorous statistical testing with intuitive interpretation.
