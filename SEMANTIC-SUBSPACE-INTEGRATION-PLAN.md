# Plan: Semantic Subspace Integration with Density Matrices

## Overview

**Goal**: Integrate semantic axes into the density matrix framework as subspace projections, rather than treating them as a separate parallel analysis. This makes semantic axes a natural decomposition tool within the density matrix workflow.

**Key Insight**: A semantic axis defines a 1D subspace in embedding space. Multiple axes define a k-dimensional semantic subspace. We can project the density matrix onto this subspace and its orthogonal complement, enabling rigorous decomposition of distributional differences.

## Motivation

**Current state (v1 - implemented)**:
- Density matrices: compute trace distance, entropy, divergence on full embedding space
- Semantic axes: compute projections, effect sizes, p-values separately
- These feel "bolted on" - parallel analyses with no mathematical integration

**Desired state (v2 - this plan)**:
- Semantic axes define an interpretable subspace
- Density matrix analysis decomposes into:
  - **Semantic component**: differences captured by interpretable axes
  - **Residual component**: differences orthogonal to semantic axes
- Unified workflow: one analysis, coherent decomposition

**Advantage**: Streamlined conceptual flow with rigorous quantification. Instead of "here's the trace distance, and separately here are some semantic projections," we get "trace distance = 0.23, of which 0.14 (61%) is in the semantic subspace and 0.09 (39%) is residual."

## Mathematical Formulation

### Projection Operators

**Semantic subspace projection**:

Given k semantic axes as unit vectors $\{v_1, v_2, \ldots, v_k\} \in \mathbb{R}^d$, stack into matrix $V \in \mathbb{R}^{d \times k}$.

Projection onto semantic subspace:
$$P_{\text{sem}} = VV^T \in \mathbb{R}^{d \times d}$$

This is a rank-k projection matrix (idempotent: $P_{\text{sem}}^2 = P_{\text{sem}}$).

Projection onto orthogonal complement (residual):
$$P_{\text{res}} = I - VV^T$$

Properties:
- $P_{\text{sem}} + P_{\text{res}} = I$
- $P_{\text{sem}} P_{\text{res}} = 0$ (orthogonal)

### Decomposed Gram Matrices

**Full Gram matrix**: $G = EE^T$ where $E \in \mathbb{R}^{n \times d}$ (n samples, d embedding dimensions)

**Semantic Gram matrix**:
$$G_{\text{sem}} = E P_{\text{sem}} E^T = E (VV^T) E^T \in \mathbb{R}^{n \times n}$$

Efficient computation: Let $S = EV \in \mathbb{R}^{n \times k}$ (project embeddings onto axes), then:
$$G_{\text{sem}} = SS^T$$

**Residual Gram matrix**:
$$G_{\text{res}} = E P_{\text{res}} E^T = E(I - VV^T)E^T$$

Computation: $G_{\text{res}} = G - G_{\text{sem}}$ (no need to explicitly form $P_{\text{res}}$)

**Verification**: $G = G_{\text{sem}} + G_{\text{res}}$ (exact decomposition)

### Decomposed Density Matrices

Normalize each Gram component:

$$\rho_{\text{sem}} = \frac{G_{\text{sem}}}{\text{trace}(G_{\text{sem}})}$$

$$\rho_{\text{res}} = \frac{G_{\text{res}}}{\text{trace}(G_{\text{res}})}$$

**Note**: $\rho \neq \rho_{\text{sem}} + \rho_{\text{res}}$ because of different normalizations. These represent **separate density matrices for the two subspaces**, not additive components of the full density matrix.

### Decomposed Metrics

**Trace distance decomposition**:

Total: $D_{\text{total}} = D(\rho_A, \rho_B)$

Semantic: $D_{\text{sem}} = D(\rho_A^{\text{sem}}, \rho_B^{\text{sem}})$

Residual: $D_{\text{res}} = D(\rho_A^{\text{res}}, \rho_B^{\text{res}})$

**Explained fraction**: 
$$f_{\text{sem}} = \frac{D_{\text{sem}}}{D_{\text{total}}}$$

**Interpretation**: What fraction of the total distributional difference is captured by the semantic subspace?

**Entropy decomposition**:

For each sample, compute:
- $S_{\text{total}} = S(\rho)$ - entropy of full distribution
- $S_{\text{sem}} = S(\rho_{\text{sem}})$ - entropy within semantic subspace
- $S_{\text{res}} = S(\rho_{\text{res}})$ - entropy within residual subspace

**Interpretation**: Is diversity concentrated in interpretable dimensions or spread across all dimensions?

**Relative entropy decomposition**:

- $S(\rho_A || \rho_B)$ - total divergence
- $S(\rho_A^{\text{sem}} || \rho_B^{\text{sem}})$ - semantic divergence
- $S(\rho_A^{\text{res}} || \rho_B^{\text{res}})$ - residual divergence

### PCA Integration

**Question**: Apply semantic decomposition before or after PCA?

**Option 1 - Decompose in full embedding space, then apply PCA**:
```
Embeddings (n×768) 
  → G_sem and G_res (n×n each)
  → PCA on concatenated embeddings for shared basis
  → ρ_sem and ρ_res in compressed space
```

**Option 2 - Apply PCA first, then decompose**:
```
Embeddings (n×768)
  → PCA → (n×50)
  → Project semantic axes into PCA space: V' = PCA.components_.T @ V
  → G_sem and G_res in PCA space
```

**Recommendation**: **Option 2** - PCA first for consistency with existing workflow.

**Rationale**:
- Existing code applies PCA before density matrices
- Semantic axes project cleanly into PCA space
- Computationally identical to current approach
- Decomposition operates in the same k=50 space as full density matrix

**Implementation**: Project semantic axes into PCA space:
```python
# V is d×k (768×k) matrix of semantic axes in full SBERT space
# PCA.components_ is (50, 768)
V_pca = PCA.components_.T @ V  # (50, k) - axes in PCA space
# Now do decomposition in 50-D space
```

## API Design

### New Data Structure

```python
@dataclass
class SubspaceDecomposition:
    """Results of semantic subspace decomposition of density matrices."""
    
    # Full space metrics
    trace_distance_total: float
    entropy_a_total: float
    entropy_b_total: float
    rel_entropy_total: float
    
    # Semantic subspace metrics
    trace_distance_semantic: float
    entropy_a_semantic: float
    entropy_b_semantic: float
    rel_entropy_semantic: float
    
    # Residual subspace metrics
    trace_distance_residual: float
    entropy_a_residual: float
    entropy_b_residual: float
    rel_entropy_residual: float
    
    # Decomposition fractions
    semantic_fraction: float  # D_sem / D_total
    residual_fraction: float  # D_res / D_total
    
    # Semantic axes used
    axes: List[SemanticAxis]
    n_axes: int
    
    # Per-axis interpretation (from existing SemanticInterpretation)
    per_axis_results: SemanticInterpretation
    
    # Density matrices (optional, for debugging/advanced use)
    rho_a_semantic: Optional[np.ndarray] = None
    rho_b_semantic: Optional[np.ndarray] = None
    rho_a_residual: Optional[np.ndarray] = None
    rho_b_residual: Optional[np.ndarray] = None
    
    def summary(self, show_per_axis=True, top_k=5) -> str:
        """Human-readable summary of decomposition."""
        pass
```

### Core Functions

**1. Subspace projection**

```python
def project_to_subspace(
    embeddings: np.ndarray,
    axes: List[SemanticAxis],
    pca: Optional[PCA] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project embeddings onto semantic subspace and residual.
    
    Args:
        embeddings: (n, d) array
        axes: List of semantic axes (unit vectors in d-dimensional space)
        pca: Optional PCA model (if provided, work in PCA space)
    
    Returns:
        embeddings_semantic: (n, k) projections onto k axes
        embeddings_residual: (n, d) embeddings with semantic component removed
                            (or (n, pca_dim) if PCA provided)
    """
    pass
```

**2. Subspace density matrices**

```python
def subspace_density_matrices(
    embeddings: np.ndarray,
    axes: List[SemanticAxis],
    pca: Optional[PCA] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute density matrices in semantic subspace and residual.
    
    Args:
        embeddings: (n, d) array
        axes: List of semantic axes
        pca: Optional PCA model
    
    Returns:
        rho_semantic: (n, n) density matrix in semantic subspace
        rho_residual: (n, n) density matrix in residual subspace
    """
    # Option 1: Efficient via projection
    S = embeddings @ V  # (n, k) where V is stacked axes
    G_sem = S @ S.T
    rho_sem = G_sem / np.trace(G_sem)
    
    G_full = embeddings @ embeddings.T
    G_res = G_full - G_sem
    rho_res = G_res / np.trace(G_res)
    
    return rho_sem, rho_res
```

**3. Unified decomposition analysis**

```python
def analyze_with_semantic_decomposition(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    axes: List[SemanticAxis],
    pca_dim: int = 50,
    compute_full_matrices: bool = False
) -> SubspaceDecomposition:
    """
    Unified density matrix analysis with semantic subspace decomposition.
    
    This is the main entry point for integrated semantic + density matrix analysis.
    
    Args:
        embeddings_a: (n_a, d) embeddings for sample A
        embeddings_b: (n_b, d) embeddings for sample B
        axes: List of semantic axes defining interpretable subspace
        pca_dim: Number of PCA dimensions (default 50)
        compute_full_matrices: Store full density matrices in result (default False)
    
    Returns:
        SubspaceDecomposition with all metrics and per-axis interpretations
    
    Workflow:
        1. Fit PCA on combined embeddings
        2. Project semantic axes into PCA space
        3. Compute full density matrices (ρ_A, ρ_B)
        4. Compute semantic subspace density matrices
        5. Compute residual subspace density matrices
        6. Compute all metrics (trace distance, entropy, relative entropy) for each
        7. Compute per-axis projections and effect sizes
        8. Return unified SubspaceDecomposition
    """
    pass
```

**4. Convenience wrapper for CompletionSample**

```python
def analyze_samples_with_semantic_decomposition(
    sample_a: CompletionSample,
    sample_b: CompletionSample,
    axes: List[SemanticAxis],
    pca_dim: int = 50,
    _precomputed_embeddings: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> SubspaceDecomposition:
    """
    Convenience wrapper for CompletionSample objects.
    
    Follows the _precomputed_embeddings pattern from tests.py and interpret_samples().
    """
    if _precomputed_embeddings is not None:
        emb_a, emb_b = _precomputed_embeddings
    else:
        emb_a = embed_sample(sample_a)
        emb_b = embed_sample(sample_b)
    
    return analyze_with_semantic_decomposition(
        emb_a, emb_b, axes, pca_dim
    )
```

### Summary Output Format

```
Semantic Subspace Decomposition
================================

Total distributional difference:
  Trace distance:     0.234
  von Neumann entropy: Sample A = 3.82, Sample B = 4.15
  Relative entropy:   0.156

Decomposition (3 semantic axes: professionalism, technicality, formality):

  Semantic subspace:
    Trace distance:   0.143 (61.1% of total)
    Entropy:          Sample A = 2.31, Sample B = 2.58
    Relative entropy: 0.094
  
  Residual subspace:
    Trace distance:   0.091 (38.9% of total)
    Entropy:          Sample A = 1.51, Sample B = 1.57
    Relative entropy: 0.062

Interpretation: 61% of the distributional difference is captured by 3 interpretable
semantic dimensions. The remaining 39% represents variation orthogonal to these axes.

Per-Axis Effects (sorted by effect size):
  professionalism ← → casualness:  Δ = +0.54, d = 0.82, p < 0.001 *
  technicality ← → layperson:      Δ = -0.38, d = 0.61, p = 0.003 *
  formality ← → informality:       Δ = +0.21, d = 0.31, p = 0.15

* = significant after Benjamini-Hochberg FDR correction
```

## Implementation Phases

### Phase 1: Subspace Projection Functions

**File**: `model_equality_testing/src/semantic_axes.py` (extend existing)

**Add**:
1. `_stack_axes(axes)` - helper to create V matrix (d×k) from list of axes
2. `project_to_subspace(embeddings, axes, pca=None)` - compute semantic and residual projections
3. Unit tests:
   - Verify $P_{\text{sem}} + P_{\text{res}} = I$
   - Verify orthogonality: $\langle E_{\text{sem}}, E_{\text{res}} \rangle = 0$
   - Test with known axes (e.g., first principal component)

### Phase 2: Subspace Density Matrices

**Add**:
1. `subspace_density_matrices(embeddings, axes, pca=None)` - compute $\rho_{\text{sem}}$ and $\rho_{\text{res}}$
2. Verification tests:
   - $G = G_{\text{sem}} + G_{\text{res}}$ (Gram decomposition is exact)
   - Traces: $\text{trace}(\rho_{\text{sem}}) = 1$ and $\text{trace}(\rho_{\text{res}}) = 1$
   - Both are PSD (positive semi-definite)

### Phase 3: Decomposed Metrics

**Add**:
1. Compute all three trace distances (total, semantic, residual)
2. Compute all entropies for both samples (6 values total)
3. Compute all relative entropies
4. Compute semantic fraction: $D_{\text{sem}} / D_{\text{total}}$
5. Tests:
   - Semantic fraction in [0, 1]
   - Edge case: single axis captures 100% of difference
   - Edge case: axes orthogonal to difference direction (fraction ≈ 0)

### Phase 4: Unified Analysis Function

**Add**:
1. `SubspaceDecomposition` dataclass
2. `analyze_with_semantic_decomposition()` - main entry point
3. Integration with PCA workflow (apply PCA first, project axes into PCA space)
4. `SubspaceDecomposition.summary()` method for human-readable output
5. End-to-end test with real data:
   - Load fp32 vs int8 samples
   - Load 3-5 semantic axes
   - Run unified analysis
   - Verify semantic fraction is reasonable (20-80%)
   - Verify per-axis results match standalone `interpret_difference()`

### Phase 5: Convenience Wrappers and Documentation

**Add**:
1. `analyze_samples_with_semantic_decomposition()` - wrapper for CompletionSample
2. Update `SEMANTIC-AXES-GUIDE.md`:
   - New section: "Subspace Decomposition with Density Matrices"
   - Example workflow
   - Interpretation guidance for semantic fractions
3. Update `README.md` with integrated example
4. Update `CLAUDE.md` with new functions

### Phase 6: Optional Advanced Features

**Consider for future**:
1. **Axis importance ranking**: Which axes contribute most to semantic subspace difference?
   - Compute trace distance for each individual axis
   - Report: "professionalism axis alone captures 35% of difference"
2. **Residual PCA**: Apply PCA to residual subspace to find unnamed axes of variation
3. **Adaptive axis selection**: Given a budget of k axes, which k maximize semantic fraction?
4. **Visualization**: 2D scatter of samples projected onto top-2 semantic axes, colored by density

## Integration with Existing Code

**No changes required to**:
- `quantum_metrics.py` - existing functions work as-is
- `embeddings.py` - just use `embed_sample()` as before
- `tests.py`, `algorithm.py`, `pvalue.py` - orthogonal to two-sample testing

**Extensions to**:
- `semantic_axes.py` - add new functions (phases 1-5 above)

**New top-level workflow**:
```python
from model_equality_testing.dataset import load_distribution
from model_equality_testing.src.embeddings import embed_sample
from model_equality_testing.src.semantic_axes import (
    load_axis_from_jsonl,
    analyze_samples_with_semantic_decomposition
)

# Load data
dist_a = load_distribution("meta-llama/Meta-Llama-3-8B-Instruct", ..., source="fp32")
dist_b = load_distribution("meta-llama/Meta-Llama-3-8B-Instruct", ..., source="int8")
sample_a = dist_a.draw_completion_sample(100)
sample_b = dist_b.draw_completion_sample(100)

# Load semantic axes (from semantic-pole project output)
axes = [
    load_axis_from_jsonl("professionalism.jsonl", "casualness.jsonl", "prof"),
    load_axis_from_jsonl("technical.jsonl", "layperson.jsonl", "tech"),
    load_axis_from_jsonl("formal.jsonl", "informal.jsonl", "formal"),
]

# Unified analysis
result = analyze_samples_with_semantic_decomposition(
    sample_a, sample_b, axes, pca_dim=50
)

print(result.summary())
```

**Output**:
- Total trace distance
- Semantic subspace metrics (with fraction of total)
- Residual subspace metrics
- Per-axis effect sizes and p-values
- Interpretation guidance

One function call, conceptually unified framework.

## Key Design Decisions

### 1. PCA Before Decomposition

**Decision**: Apply PCA first (compress 768→50), then decompose in PCA space.

**Rationale**:
- Consistency with existing workflow
- Semantic axes project cleanly into PCA space
- Same computational complexity as current approach
- All comparisons happen in same compressed space

**Implementation**: $V' = U^T V$ where $U$ is PCA components matrix

### 2. Separate Normalizations

**Decision**: $\rho_{\text{sem}}$ and $\rho_{\text{res}}$ are separately normalized (each has trace 1).

**Rationale**:
- They represent independent density matrices for the two subspaces
- Enables direct application of quantum metrics to each
- Not claiming $\rho = \rho_{\text{sem}} + \rho_{\text{res}}$ (this would be incorrect)

**Implication**: Trace distances don't sum: $D_{\text{total}} \neq D_{\text{sem}} + D_{\text{res}}$. Instead report as fractions/percentages.

### 3. Semantic Fraction as Primary Metric

**Decision**: Report $f_{\text{sem}} = D_{\text{sem}} / D_{\text{total}}$ as the key integration metric.

**Rationale**:
- Intuitive: "What fraction of difference is explained by interpretable axes?"
- Bounded [0, 1]
- Directly answers: "Are semantic axes capturing the whole story?"

**Caveat**: This is an informal decomposition, not a rigorous variance decomposition (since density matrices are nonlinear in embeddings). But it's a useful heuristic.

### 4. Preserve Existing Per-Axis Analysis

**Decision**: `SubspaceDecomposition.per_axis_results` stores the existing `SemanticInterpretation` from v1.

**Rationale**:
- Per-axis effect sizes, p-values, and Wasserstein distances remain useful
- Subspace decomposition adds a global view, doesn't replace local view
- Users can drill down from "61% is semantic" to "which specific axes?"

## Success Criteria

**Conceptual**:
- ✓ Semantic axes feel like a natural part of density matrix framework, not bolted on
- ✓ Single unified analysis function replaces two separate calls
- ✓ Clear interpretation: semantic fraction tells you how much is explained

**Mathematical**:
- ✓ Gram decomposition is exact: $G = G_{\text{sem}} + G_{\text{res}}$
- ✓ Subspaces are orthogonal
- ✓ Both density matrices are valid (PSD, trace 1)

**Practical**:
- ✓ Results match standalone analysis when comparing per-axis to `per_axis_results`
- ✓ Semantic fraction in reasonable range (not 0% or 100% for typical cases)
- ✓ Summary output is clear and actionable

## Open Questions

1. **Axis orthogonalization**: Should we orthogonalize semantic axes before decomposition?
   - Pro: Ensures no double-counting if axes correlate
   - Con: Breaks interpretability (orthogonalized "professionalism" isn't pure professionalism)
   - **Recommendation**: Don't orthogonalize. Let correlations exist, they reflect reality. Check with `axis_correlations()` and warn if >0.7.

2. **Trace distance interpretation**: Is $f_{\text{sem}} = D_{\text{sem}} / D_{\text{total}}$ the right normalization?
   - Alternative: Use Gram matrix Frobenius norm instead of trace distance
   - **Recommendation**: Start with trace distance (it's the established metric), evaluate empirically

3. **Entropy interpretation**: What does $S_{\text{sem}}$ vs $S_{\text{res}}$ tell us?
   - If $S_{\text{sem}}$ is high: diversity is in interpretable dimensions
   - If $S_{\text{res}}$ is high: diversity is in unnamed dimensions
   - **Recommendation**: Include both in summary, with interpretation guidance

4. **Empty residual**: What if semantic axes span the full PCA space (k=50 axes)?
   - $G_{\text{res}} = 0$, $\rho_{\text{res}}$ undefined
   - **Recommendation**: Warn if $\text{trace}(G_{\text{res}}) < \epsilon$, report 100% semantic

## Timeline Estimate

- **Phase 1** (projection): ~2 hours (code + tests)
- **Phase 2** (subspace density matrices): ~2 hours (code + tests)
- **Phase 3** (decomposed metrics): ~2 hours (code + tests)
- **Phase 4** (unified analysis): ~3 hours (integration + end-to-end tests)
- **Phase 5** (documentation): ~2 hours (updates to guides + examples)
- **Total**: ~11 hours for complete implementation

Can be done incrementally with tests at each phase.

## Summary

This plan integrates semantic axes into the density matrix framework as **subspace projections**, making them a natural decomposition tool rather than a separate analysis path. The unified workflow provides:

1. **Detection**: Total trace distance quantifies distributional difference
2. **Decomposition**: Semantic fraction shows how much is explained by interpretable axes
3. **Interpretation**: Per-axis effect sizes identify what differs
4. **Residual analysis**: Entropy in residual subspace reveals unnamed variation

All from one function call, with rigorous mathematical foundation and streamlined conceptual flow.
