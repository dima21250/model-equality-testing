# Semantic Axes as Observables: A Theoretically Coherent Integration

## The Core Tension

The critique of SEMANTIC-SUBSPACE-INTEGRATION-PLAN.md revealed this fundamental problem:
- You want density matrices as the central representation (n×n sample space)
- Semantic axes live in embedding space (d-dimensional feature space)
- The previous plan tried to force them together via "decomposition" that doesn't mathematically hold

**But there IS a theoretically coherent way to integrate them.**

## The Aesthetic Goal

You're seeking **parsimony** - a single mathematical object (the density matrix) that unifies the analysis, not a collection of independent methods patched together.

This is a legitimate and elegant scientific aesthetic. The question is: how do we achieve it without forcing mathematics that doesn't work?

## The Solution: Semantic Axes as Observables

In quantum mechanics:
- **States** are density matrices: ρ
- **Observables** are Hermitian operators: A
- **Measurements** give expectation values: ⟨A⟩ = trace(ρA)

**Semantic axes should be observables in the density matrix formalism.**

This is not a metaphor - it's a rigorous mathematical construction.

### Constructing the Observable

A semantic axis v (unit vector in d-dimensional embedding space) induces an operator in sample space:

**Step 1: Project samples onto the semantic axis**

For each sample i, compute its projection onto axis v:
$$p_i = e_i \cdot v$$

This gives a vector of "semantic scores" for all n samples: $\vec{p} = [p_1, p_2, \ldots, p_n]$

**Step 2: Construct the observable as a diagonal operator**

The semantic observable is:
$$A_v = \text{diag}(p_1, p_2, \ldots, p_n) \in \mathbb{R}^{n \times n}$$

This is Hermitian (real diagonal matrix), so it's a valid quantum observable.

**Interpretation**: $A_v$ assigns a "semantic value" to each sample. When you "measure" this observable on a density matrix, you get the expected semantic score for that distribution.

### Measuring the Observable

**Expectation value** (mean semantic score for the distribution):
$$\langle A_v \rangle_\rho = \text{trace}(\rho A_v) = \sum_{i,j} \rho_{ij} (A_v)_{ij} = \sum_i \rho_{ii} p_i$$

Since ρ is a weighted similarity matrix (normalized), this is the weighted average semantic score.

**Variance** (diversity along the semantic axis):
$$\text{Var}_\rho(A_v) = \text{trace}(\rho A_v^2) - [\text{trace}(\rho A_v)]^2$$

**Standard uncertainty** (quantum standard deviation):
$$\sigma_\rho(A_v) = \sqrt{\text{Var}_\rho(A_v)}$$

**Difference between distributions**:
$$\Delta_v = \langle A_v \rangle_{\rho_A} - \langle A_v \rangle_{\rho_B} = \text{trace}(\rho_A A_v) - \text{trace}(\rho_B A_v)$$

This is the **shift in semantic axis v** between distributions A and B.

### Why This is Theoretically Coherent

1. **One central representation**: Density matrix ρ is the state
2. **Semantic axes are operators**: $A_v$, not a separate analysis
3. **Measurements follow quantum formalism**: $\text{trace}(\rho A)$
4. **No separate normalizations**: Everything flows from the single ρ
5. **Uncertainty relations**: You can compute commutators $[A_{\text{prof}}, A_{\text{tech}}]$ to see if semantic dimensions are "compatible observables"
6. **Well-defined probabilistic interpretation**: Expectation values have clear meaning

This is the **quantum measurement formalism applied to semantic questions**.

### What This Gives You

For each semantic axis and each distribution:

**Single-distribution measurements**:
- $\langle A_v \rangle_\rho$ - mean semantic score (where is the distribution centered on this axis?)
- $\sigma_\rho(A_v)$ - semantic diversity (how spread out along this axis?)

**Two-distribution comparisons**:
- $\Delta_v = \langle A_v \rangle_{\rho_A} - \langle A_v \rangle_{\rho_B}$ - semantic shift
- $\Delta\sigma_v = \sigma_{\rho_A}(A_v) - \sigma_{\rho_B}(A_v)$ - change in diversity

**Multi-axis relationships**:
- $[A_v, A_w] = A_v A_w - A_w A_v$ - commutator (do axes interfere?)
- Simultaneous measurability (Heisenberg-like uncertainty relations)

## Alternative: Spectral-Semantic Connection

Another theoretically unified approach that stays entirely within the density matrix framework:

### Eigendecomposition

$$\rho = \sum_i \lambda_i |\psi_i\rangle\langle\psi_i|$$

The eigenvectors $\psi_i$ are the "principal modes" of the distribution in sample space.

Each eigenvector is a weight vector over samples: $\psi_i \in \mathbb{R}^n$

### Map Eigenvectors to Embedding Space

Each eigenvector represents a "mode" - a weighted combination of samples. Map it back to embedding space:

$$\vec{m}_i = \sum_j \psi_{ij} e_j$$

This is the weighted average embedding for mode i.

### Project Modes onto Semantic Axes

$$s_{ik} = \vec{m}_i \cdot v_k$$

where:
- $s_{ik}$ = score of mode i on semantic axis k
- i indexes eigenmodes (sorted by eigenvalue)
- k indexes semantic axes

### Interpretation

**Example**: 
- Mode 1 (λ₁ = 0.45): s₁,prof = 0.85, s₁,tech = 0.12
- Mode 2 (λ₂ = 0.23): s₂,prof = -0.21, s₂,tech = 0.78

**Interpretation**: "The dominant mode of variation (45% of weight) is primarily professionalism. The second mode (23% of weight) is primarily technicality."

**For two-distribution comparison**:

Compare eigenspectra:
- Which modes changed in weight? (Δλᵢ)
- What are the semantic fingerprints of those modes? (sᵢₖ)

**Report**: "Distribution B increased weight on Mode 3 (technicality-heavy) and decreased weight on Mode 1 (professionalism-heavy)."

**Advantage**: You're analyzing the spectral structure of ρ through semantic lenses, staying entirely within the density matrix framework. No separate normalization, no parallel analysis.

## Proposed Architecture

Here's how to structure this to achieve your aesthetic:

### Core: Density Matrix Representation
```python
# Single central object
rho = density_matrix(embeddings, pca_dim=50)
```

### Layer 1: Distributional Metrics (unchanged)
```python
# Intrinsic properties of the density matrix
trace_distance = D(rho_A, rho_B)
entropy = von_neumann_entropy(rho)
divergence = quantum_relative_entropy(rho_A, rho_B)
```

### Layer 2: Semantic Observables (new, theoretically integrated)
```python
# Construct observables from semantic axes
A_prof = semantic_observable(embeddings, axis_professionalism)
A_tech = semantic_observable(embeddings, axis_technical)

# Measure expectations (mean semantic scores)
prof_A = measure(rho_A, A_prof)  # trace(rho_A @ A_prof)
prof_B = measure(rho_B, A_prof)
delta_prof = prof_A - prof_B

# Measure variance (diversity along axis)
var_prof_A = variance(rho_A, A_prof)
var_prof_B = variance(rho_B, A_prof)

# Measure uncertainty (standard deviation)
sigma_prof_A = uncertainty(rho_A, A_prof)
sigma_prof_B = uncertainty(rho_B, A_prof)
```

### Layer 3: Spectral-Semantic Analysis (optional, advanced)
```python
# Eigendecomposition of density matrix
eigenvalues, eigenvectors = eig(rho)

# Map eigenmodes to semantic space
semantic_fingerprints = eigenvectors_to_semantic(
    eigenvectors, embeddings, axes
)

# Interpretation: "Mode 1 (λ=0.45) is 80% professionalism, 20% technicality"
# Compare distributions: "B shifts weight from Mode 1 to Mode 3"
```

**Key point**: Everything flows from ρ. Semantic axes are operators acting on ρ, not a separate parallel analysis.

## API Design

### Data Structures

```python
@dataclass
class SemanticObservable:
    """
    A Hermitian operator representing a semantic axis.
    
    Constructed from a semantic axis (unit vector in embedding space)
    by projecting samples onto the axis and forming a diagonal operator.
    """
    axis: SemanticAxis           # The underlying semantic axis
    operator: np.ndarray         # (n, n) diagonal matrix
    projections: np.ndarray      # (n,) vector of sample projections
    n_samples: int
    
    def __matmul__(self, other):
        """Allow A @ B for operator composition."""
        return self.operator @ other.operator
    
    def commutator(self, other):
        """Compute [A, B] = AB - BA."""
        return self @ other - other @ self
```

```python
@dataclass
class SemanticMeasurement:
    """
    Result of measuring a semantic observable on a density matrix.
    
    Follows quantum measurement formalism: trace(ρA).
    """
    observable_name: str         # e.g., "professionalism-casualness"
    expectation: float           # ⟨A⟩ = trace(ρA)
    variance: float              # Var(A) = trace(ρA²) - ⟨A⟩²
    uncertainty: float           # σ = sqrt(Var)
    negative_pole: str           # "professionalism"
    positive_pole: str           # "casualness"
    
    def interpretation(self) -> str:
        """Human-readable interpretation."""
        direction = self.positive_pole if self.expectation > 0 else self.negative_pole
        return f"Mean: {self.expectation:.3f} toward {direction}, σ={self.uncertainty:.3f}"
```

```python
@dataclass
class SemanticComparison:
    """
    Comparison of two distributions via semantic observables.
    
    Unified analysis: density matrices + semantic measurements.
    """
    # Density matrix metrics (global)
    trace_distance: float
    entropy_a: float
    entropy_b: float
    rel_entropy: float
    
    # Per-axis measurements
    measurements_a: List[SemanticMeasurement]
    measurements_b: List[SemanticMeasurement]
    
    # Per-axis deltas
    semantic_shifts: List[Tuple[str, float, float]]  # (axis_name, Δ⟨A⟩, Δσ)
    
    def summary(self) -> str:
        """Unified summary: global metrics + semantic shifts."""
        pass
```

### Core Functions

```python
def semantic_observable(
    embeddings: np.ndarray,
    axis: SemanticAxis
) -> SemanticObservable:
    """
    Construct a Hermitian observable from a semantic axis.
    
    Args:
        embeddings: (n, d) sample embeddings
        axis: Semantic axis (unit vector in d-dimensional space)
    
    Returns:
        SemanticObservable: diagonal operator A = diag(e₁·v, e₂·v, ..., eₙ·v)
    
    The observable assigns each sample its projection onto the semantic axis.
    Measuring this observable on a density matrix gives the expected semantic score.
    """
    projections = embeddings @ axis.axis_vector  # (n,)
    operator = np.diag(projections)  # (n, n)
    return SemanticObservable(
        axis=axis,
        operator=operator,
        projections=projections,
        n_samples=len(embeddings)
    )
```

```python
def measure(
    rho: np.ndarray,
    observable: SemanticObservable
) -> SemanticMeasurement:
    """
    Measure a semantic observable on a density matrix.
    
    Follows quantum measurement formalism:
        Expectation: ⟨A⟩ = trace(ρA)
        Variance: Var(A) = trace(ρA²) - ⟨A⟩²
        Uncertainty: σ(A) = sqrt(Var(A))
    
    Args:
        rho: (n, n) density matrix
        observable: Semantic observable to measure
    
    Returns:
        SemanticMeasurement with expectation, variance, uncertainty
    """
    A = observable.operator
    
    # Expectation value
    expectation = np.trace(rho @ A)
    
    # Variance
    A_squared = A @ A
    variance = np.trace(rho @ A_squared) - expectation**2
    
    # Uncertainty (standard deviation)
    uncertainty = np.sqrt(max(0, variance))  # guard against numerical errors
    
    return SemanticMeasurement(
        observable_name=observable.axis.name,
        expectation=expectation,
        variance=variance,
        uncertainty=uncertainty,
        negative_pole=observable.axis.negative_pole,
        positive_pole=observable.axis.positive_pole
    )
```

```python
def compare_with_semantics(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    axes: List[SemanticAxis],
    pca_dim: int = 50
) -> SemanticComparison:
    """
    Unified comparison: density matrices + semantic measurements.
    
    This is the main entry point for integrated analysis.
    
    Workflow:
        1. Compute density matrices ρ_A and ρ_B (with PCA)
        2. Compute global metrics (trace distance, entropy, divergence)
        3. Construct semantic observables from axes
        4. Measure each observable on both density matrices
        5. Compute semantic shifts (Δ⟨A⟩, Δσ)
        6. Return unified SemanticComparison
    
    Args:
        embeddings_a: (n_a, d) embeddings for sample A
        embeddings_b: (n_b, d) embeddings for sample B
        axes: List of semantic axes
        pca_dim: PCA compression dimension
    
    Returns:
        SemanticComparison with all metrics and measurements
    """
    # Fit PCA on combined embeddings
    from model_equality_testing.src.quantum_metrics import fit_pca, pca_density_matrix
    
    pca = fit_pca(np.vstack([embeddings_a, embeddings_b]), k=pca_dim)
    
    # Compute density matrices
    rho_a = pca_density_matrix(embeddings_a, pca)
    rho_b = pca_density_matrix(embeddings_b, pca)
    
    # Global metrics
    from model_equality_testing.src.quantum_metrics import (
        trace_distance, von_neumann_entropy, quantum_relative_entropy
    )
    
    td = trace_distance(rho_a, rho_b)
    s_a = von_neumann_entropy(rho_a)
    s_b = von_neumann_entropy(rho_b)
    rel_ent = quantum_relative_entropy(rho_a, rho_b)
    
    # Transform embeddings to PCA space for observable construction
    emb_a_pca = pca.transform(embeddings_a)
    emb_b_pca = pca.transform(embeddings_b)
    
    # Project semantic axes into PCA space
    axes_pca = [project_axis_to_pca(axis, pca) for axis in axes]
    
    # Construct observables and measure
    measurements_a = []
    measurements_b = []
    semantic_shifts = []
    
    for axis_pca in axes_pca:
        # Construct observables for each sample
        obs_a = semantic_observable(emb_a_pca, axis_pca)
        obs_b = semantic_observable(emb_b_pca, axis_pca)
        
        # Measure on density matrices
        meas_a = measure(rho_a, obs_a)
        meas_b = measure(rho_b, obs_b)
        
        measurements_a.append(meas_a)
        measurements_b.append(meas_b)
        
        # Compute shifts
        delta_exp = meas_a.expectation - meas_b.expectation
        delta_unc = meas_a.uncertainty - meas_b.uncertainty
        semantic_shifts.append((axis_pca.name, delta_exp, delta_unc))
    
    return SemanticComparison(
        trace_distance=td,
        entropy_a=s_a,
        entropy_b=s_b,
        rel_entropy=rel_ent,
        measurements_a=measurements_a,
        measurements_b=measurements_b,
        semantic_shifts=semantic_shifts
    )
```

```python
def project_axis_to_pca(
    axis: SemanticAxis,
    pca: PCA
) -> SemanticAxis:
    """
    Project a semantic axis from full embedding space into PCA space.
    
    Args:
        axis: Semantic axis in full d-dimensional space (e.g., 768-D)
        pca: Fitted PCA model
    
    Returns:
        New SemanticAxis in PCA space (k-dimensional, e.g., 50-D)
    
    The projected axis is renormalized to unit length.
    """
    # Project: v_pca = U^T @ v where U is PCA components
    v_pca = pca.components_.T @ axis.axis_vector
    
    # Renormalize
    v_pca = v_pca / np.linalg.norm(v_pca)
    
    # Create new axis in PCA space
    return SemanticAxis(
        name=axis.name,
        negative_pole=axis.negative_pole,
        positive_pole=axis.positive_pole,
        axis_vector=v_pca,
        negative_centroid=pca.transform(axis.negative_centroid.reshape(1, -1)).flatten(),
        positive_centroid=pca.transform(axis.positive_centroid.reshape(1, -1)).flatten(),
        embedding_model=axis.embedding_model,
        embedding_dim=pca.n_components_,
        pole_distance=np.linalg.norm(v_pca),  # after projection
        metadata={**axis.metadata, 'pca_projected': True}
    )
```

### Summary Output Format

```
Semantic Observable Analysis
=============================

Distributional Metrics:
  Trace distance:       0.234
  von Neumann entropy:  Sample A = 3.82, Sample B = 4.15
  Relative entropy:     0.156

Semantic Measurements:

  professionalism ← → casualness
    Sample A: ⟨A⟩ = -0.32 toward professionalism, σ = 0.18
    Sample B: ⟨A⟩ = +0.22 toward casualness, σ = 0.21
    Shift: Δ⟨A⟩ = +0.54, Δσ = +0.03

  technical ← → layperson
    Sample A: ⟨A⟩ = +0.45 toward technical, σ = 0.15
    Sample B: ⟨A⟩ = +0.07 toward technical, σ = 0.19
    Shift: Δ⟨A⟩ = -0.38, Δσ = +0.04

  formal ← → informal
    Sample A: ⟨A⟩ = -0.12 toward formal, σ = 0.22
    Sample B: ⟨A⟩ = +0.09 toward informal, σ = 0.20
    Shift: Δ⟨A⟩ = +0.21, Δσ = -0.02

Interpretation: Distribution B shifted toward casualness (+0.54), away from 
technicality (-0.38), and slightly toward informality (+0.21). Diversity 
increased slightly along professionalism and technicality axes.
```

## The Aesthetic This Achieves

✓ **Unified representation**: Density matrix ρ is the core state  
✓ **Theoretically grounded**: Quantum measurement formalism  
✓ **No Frankenstein**: Semantic axes aren't "bolted on", they're observables  
✓ **Clean API**: `measure(rho, observable)` - one operation, clear semantics  
✓ **Extensible**: Add new semantic axes = add new observables, same formalism  
✓ **Rigorous**: Well-defined probabilistic interpretation (expectation values)  
✓ **No decomposition forcing**: Not claiming ρ = ρ_sem + ρ_res  

## What We Give Up (from the broken plan)

✗ "Fraction explained" decomposition - doesn't work mathematically  
✗ Separate ρ_semantic and ρ_residual - creates confusion  
✗ "61% captured by semantic axes" - this was circular anyway  
✗ Subspace density matrices - theoretically incoherent  

## What We Gain

✓ Theoretically rigorous integration  
✓ Clear probabilistic interpretation (expectation values)  
✓ **Variance along semantic axes** (new metric!) - diversity measurement  
✓ **Uncertainty quantification** - σ(A) tells you spread  
✓ Potential for **uncertainty relations** between semantic dimensions  
✓ Natural extension: **observable algebra**, commutators, simultaneous measurability  
✓ Spectral-semantic connection (eigenmodes through semantic lenses)  

## Comparison to Previous Approaches

### v1 (Implemented): Semantic Axes as Separate Analysis
- Density matrices: ρ_A, ρ_B → metrics
- Semantic axes: project embeddings, t-tests, effect sizes
- **Issue**: Feels "bolted on", no theoretical unity

### v2 (Broken Plan): Semantic Subspace Decomposition
- Separate ρ_sem and ρ_res with independent normalizations
- Report "fraction explained" from ratios
- **Issue**: Mathematically incoherent, forces decomposition that doesn't work

### v3 (This Proposal): Semantic Observables
- Single ρ as the quantum state
- Semantic axes as Hermitian operators A
- Measurements via trace(ρA) - quantum expectation values
- **Advantage**: Theoretically rigorous, unified formalism, no forced decomposition

## Implementation Notes

### Where to Add This Code

**File**: `model_equality_testing/src/semantic_axes.py` (extend existing module)

**New classes**:
- `SemanticObservable` - Hermitian operator from semantic axis
- `SemanticMeasurement` - Result of measuring observable on density matrix
- `SemanticComparison` - Unified analysis result

**New functions**:
- `semantic_observable(embeddings, axis)` - construct operator
- `measure(rho, observable)` - quantum measurement
- `compare_with_semantics(emb_a, emb_b, axes, pca_dim)` - main entry point
- `project_axis_to_pca(axis, pca)` - project axis into PCA space

**Integration**:
- Imports from `quantum_metrics.py`: `fit_pca`, `pca_density_matrix`, metrics
- Imports from existing `semantic_axes.py`: `SemanticAxis`, `load_axis_from_jsonl`
- No changes to existing code required

### Testing Strategy

1. **Observable construction**: Verify A is Hermitian (A = A^T)
2. **Expectation values**: Compare trace(ρA) to manual weighted average
3. **Variance formula**: Verify Var(A) = ⟨A²⟩ - ⟨A⟩² ≥ 0
4. **PCA projection**: Verify axis remains unit norm after projection
5. **End-to-end**: Load real data, compute measurements, verify interpretability

### Advanced Extensions (Future)

1. **Commutators**: `[A, B]` to test if semantic axes are compatible observables
2. **Uncertainty relations**: Heisenberg-like bounds on simultaneous measurability
3. **Spectral decomposition**: Map eigenmodes to semantic fingerprints
4. **Time evolution**: Track how measurements change across model versions
5. **Multi-axis observables**: Tensor products for joint measurements

## Summary

This proposal achieves the theoretical aesthetic you're seeking:

- **Density matrix as central representation** - the quantum state
- **Semantic axes as observables** - Hermitian operators, not separate analysis
- **Measurements via quantum formalism** - trace(ρA) for expectations
- **No Frankenstein** - everything flows from ρ, unified mathematical framework
- **Rigorous and extensible** - well-defined semantics, natural extensions

The key insight: **Don't try to decompose ρ. Instead, measure observables on ρ.**

This is how quantum mechanics works, and it maps perfectly to the semantic interpretation problem. Observables are questions you ask about the state. Measurements give you answers. All within one unified formalism.
