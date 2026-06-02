# Spectral-Semantic Analysis: Interpreting Eigenspectra Through Semantic Lenses

## Overview

This proposal presents an alternative to the semantic observables approach: instead of constructing operators from semantic axes, we **interpret the eigenmodes of the density matrix** using semantic axes.

**Core idea**: The eigendecomposition of ρ reveals the principal modes of distributional structure. We map these modes back to embedding space and project them onto semantic axes to understand *what* each mode represents semantically.

This achieves the same theoretical aesthetic (density matrix as central representation) through a different mathematical path.

## Mathematical Formulation

### Eigendecomposition of the Density Matrix

$$\rho = \sum_{i=1}^n \lambda_i |\psi_i\rangle\langle\psi_i|$$

where:
- $\lambda_i$ are eigenvalues (sorted descending): $\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_n$
- $|\psi_i\rangle$ are eigenvectors in sample space: $\psi_i \in \mathbb{R}^n$
- $\sum_i \lambda_i = 1$ (since trace(ρ) = 1)

**Interpretation**: 
- Each eigenvalue $\lambda_i$ is the "weight" of mode i
- Eigenvectors $\psi_i$ are orthonormal: $\langle\psi_i|\psi_j\rangle = \delta_{ij}$
- The density matrix is a weighted superposition of pure modes

### Mapping Eigenmodes to Embedding Space

Each eigenvector $\psi_i$ is a weight vector over the n samples. Map it back to embedding space:

$$\vec{m}_i = \sum_{j=1}^n \psi_{ij} \, e_j = E^T \psi_i$$

where:
- $E \in \mathbb{R}^{n \times d}$ is the embedding matrix (n samples, d dimensions)
- $\vec{m}_i \in \mathbb{R}^d$ is the "semantic fingerprint" of mode i
- This is a weighted average of embeddings, weighted by the eigenvector

**Intuition**: Mode i assigns weight $\psi_{ij}$ to sample j. The semantic fingerprint is the centroid of samples weighted by this mode.

**Note**: $\vec{m}_i$ is generally NOT unit norm. We can normalize if desired, or keep unnormalized to preserve magnitude information.

### Projecting Modes onto Semantic Axes

Given k semantic axes $\{v_1, v_2, \ldots, v_k\}$ (unit vectors in embedding space):

$$s_{ik} = \vec{m}_i \cdot v_k$$

This gives a **semantic signature matrix** $S \in \mathbb{R}^{n \times k}$ where:
- Rows: eigenmodes (sorted by eigenvalue)
- Columns: semantic axes
- Entry $s_{ik}$: "loading" of mode i on semantic axis k

**Normalization options**:
1. **Unnormalized**: $s_{ik} = \vec{m}_i \cdot v_k$ (preserves mode magnitude)
2. **Normalized**: $s_{ik} = \frac{\vec{m}_i}{||\vec{m}_i||} \cdot v_k$ (pure directional alignment)

**Recommendation**: Use normalized for semantic interpretation (cosine similarity semantics).

### Dominant Mode Analysis

The **dominant mode** (largest eigenvalue $\lambda_1$) captures the primary structure of the distribution.

Its semantic signature $[s_{1,1}, s_{1,2}, \ldots, s_{1,k}]$ tells you:
- Which semantic dimensions dominate the distribution
- The "semantic character" of the typical sample

**Example**:
- Mode 1 ($\lambda_1 = 0.45$): $s_{1,\text{prof}} = 0.85$, $s_{1,\text{tech}} = 0.12$, $s_{1,\text{formal}} = 0.50$
- **Interpretation**: "The dominant mode (45% of weight) is primarily professionalism with moderate formality"

### Two-Distribution Comparison

For distributions A and B, eigendecompose both:
- $\rho_A = \sum_i \lambda_i^A |\psi_i^A\rangle\langle\psi_i^A|$
- $\rho_B = \sum_i \lambda_i^B |\psi_i^B\rangle\langle\psi_i^B|$

Compute semantic signatures for both:
- $S_A[i,k] = \vec{m}_i^A \cdot v_k$
- $S_B[i,k] = \vec{m}_i^B \cdot v_k$

**Comparison strategies**:

**Strategy 1 - Compare dominant modes**:
- Top 5 modes for each distribution
- Show semantic signatures side-by-side
- Highlight shifts: "Mode 1 in A is professionalism-heavy; Mode 1 in B is technicality-heavy"

**Strategy 2 - Eigenvalue shifts**:
- Which modes gained/lost weight?
- What are their semantic characters?
- "Distribution B increased weight on technicality-modes and decreased weight on professionalism-modes"

**Strategy 3 - Weighted semantic average**:
$$\bar{s}_k^A = \sum_i \lambda_i^A \, s_{ik}^A$$

This is the "average semantic position" of the distribution, weighted by eigenvalues.

Compare: $\Delta\bar{s}_k = \bar{s}_k^A - \bar{s}_k^B$

**Connection to observables**: This is mathematically equivalent to the semantic observable expectation value! 
$$\bar{s}_k = \langle A_k \rangle = \text{trace}(\rho A_k)$$

So the spectral and observable approaches converge for expectation values.

## Why This is Theoretically Coherent

1. **Central representation**: Density matrix ρ is the state
2. **Natural decomposition**: Eigendecomposition is the spectral theorem - fundamental to linear algebra
3. **Semantic axes as interpretive layer**: Project the natural structure (eigenmodes) onto interpretable dimensions
4. **No forced decomposition**: We're not claiming ρ = ρ_sem + ρ_res; we're interpreting what ρ already contains
5. **Well-defined**: Every step is standard linear algebra + inner products
6. **Flexible**: Can focus on dominant modes, shifted modes, or weighted averages

**Key distinction from broken subspace plan**: We're not creating separate density matrices for subspaces. We're interpreting the eigenmodes of the **single** density matrix.

## What This Tells You

### Single Distribution Analysis

For a single density matrix ρ:

1. **Distributional structure**:
   - How many effective modes? (entropy, participation ratio)
   - Is it concentrated (one dominant mode) or diverse (many modes)?

2. **Semantic character of modes**:
   - Mode 1 (λ=0.45): professionalism-heavy
   - Mode 2 (λ=0.23): technicality-heavy
   - Mode 3 (λ=0.15): formality-heavy

3. **Overall semantic position** (weighted average):
   - $\bar{s}_{\text{prof}} = 0.42$ (moderately professional)
   - $\bar{s}_{\text{tech}} = 0.18$ (slightly technical)
   - $\bar{s}_{\text{formal}} = 0.31$ (moderately formal)

### Two-Distribution Comparison

For distributions A vs B:

1. **Structural changes**:
   - Entropy shift: $\Delta S = S(\rho_B) - S(\rho_A)$ (diversity change)
   - Trace distance: $D(\rho_A, \rho_B)$ (total difference)

2. **Eigenvalue shifts**:
   - Which modes gained/lost weight?
   - Mode 1 in A (λ=0.45) → Mode 2 in B (λ=0.23): "professionalism mode lost dominance"

3. **Semantic interpretation of shifted modes**:
   - "B shifted weight from professionalism-heavy modes to technicality-heavy modes"

4. **Weighted semantic shifts**:
   - $\Delta\bar{s}_{\text{prof}} = -0.34$ (less professional overall)
   - $\Delta\bar{s}_{\text{tech}} = +0.28$ (more technical overall)

## API Design

### Data Structures

```python
@dataclass
class EigenmodeSemantics:
    """
    Semantic signature of a single eigenmode.
    
    Maps an eigenmode (eigenvector of ρ) to semantic space.
    """
    mode_index: int                    # 0-indexed mode number (sorted by eigenvalue)
    eigenvalue: float                  # Weight of this mode
    semantic_fingerprint: np.ndarray   # (d,) weighted average embedding
    semantic_scores: Dict[str, float]  # axis_name → projection score
    semantic_signature_norm: float     # ||semantic_fingerprint||
    
    def dominant_axes(self, top_k: int = 3) -> List[Tuple[str, float]]:
        """Return top-k semantic axes by absolute score."""
        return sorted(self.semantic_scores.items(), 
                     key=lambda x: abs(x[1]), 
                     reverse=True)[:top_k]
    
    def interpretation(self) -> str:
        """Human-readable semantic character of this mode."""
        top_axes = self.dominant_axes(top_k=2)
        desc = ", ".join([f"{name}: {score:+.2f}" for name, score in top_axes])
        return f"Mode {self.mode_index} (λ={self.eigenvalue:.3f}): {desc}"
```

```python
@dataclass
class SpectralSemanticAnalysis:
    """
    Full spectral-semantic analysis of a single density matrix.
    
    Eigendecomposition + semantic interpretation of eigenmodes.
    """
    eigenvalues: np.ndarray              # (n,) sorted descending
    eigenvectors: np.ndarray             # (n, n) columns are eigenvectors
    mode_semantics: List[EigenmodeSemantics]  # One per mode
    
    # Global metrics
    entropy: float                       # von Neumann entropy
    effective_rank: float                # exp(entropy)
    participation_ratio: float           # 1 / Σλᵢ²
    
    # Weighted semantic position
    weighted_semantic_position: Dict[str, float]  # axis_name → Σλᵢ·sᵢₖ
    
    def dominant_modes(self, top_k: int = 5) -> List[EigenmodeSemantics]:
        """Return top-k modes by eigenvalue."""
        return self.mode_semantics[:top_k]
    
    def summary(self, top_k_modes: int = 5) -> str:
        """Human-readable summary of spectral-semantic structure."""
        pass
```

```python
@dataclass
class SpectralSemanticComparison:
    """
    Comparison of two distributions via spectral-semantic analysis.
    
    Unified analysis: density matrices → eigendecomposition → semantic interpretation.
    """
    # Individual analyses
    analysis_a: SpectralSemanticAnalysis
    analysis_b: SpectralSemanticAnalysis
    
    # Global comparison metrics
    trace_distance: float
    entropy_diff: float                  # S(B) - S(A)
    rel_entropy: float                   # S(A || B)
    
    # Eigenvalue comparison
    eigenvalue_shifts: np.ndarray        # λᵢᴮ - λᵢᴬ for each mode
    
    # Semantic position shifts
    semantic_position_shifts: Dict[str, float]  # axis_name → Δ(weighted position)
    
    # Mode alignment analysis (optional)
    mode_similarity: Optional[np.ndarray] = None  # (n, n) cosine similarity between eigenvectors
    
    def modes_with_largest_shifts(self, top_k: int = 5) -> List[Tuple[int, float, EigenmodeSemantics, EigenmodeSemantics]]:
        """
        Identify modes with largest eigenvalue shifts.
        
        Returns:
            List of (mode_index, Δλ, mode_A, mode_B) sorted by |Δλ|
        """
        pass
    
    def summary(self, top_k_modes: int = 5) -> str:
        """Unified summary: structural changes + semantic interpretation."""
        pass
```

### Core Functions

```python
def eigenmode_to_semantic(
    eigenvector: np.ndarray,
    embeddings: np.ndarray,
    axes: List[SemanticAxis],
    normalize: bool = True
) -> EigenmodeSemantics:
    """
    Map an eigenmode to semantic space.
    
    Args:
        eigenvector: (n,) weights over samples
        embeddings: (n, d) sample embeddings
        axes: List of semantic axes
        normalize: If True, normalize semantic fingerprint before projection
    
    Returns:
        EigenmodeSemantics with fingerprint and projections onto all axes
    
    Workflow:
        1. Compute semantic fingerprint: m = E^T ψ (weighted average embedding)
        2. Optionally normalize: m̂ = m / ||m||
        3. Project onto each semantic axis: s_k = m̂ · v_k
    """
    # Weighted average of embeddings
    semantic_fingerprint = embeddings.T @ eigenvector  # (d,)
    
    signature_norm = np.linalg.norm(semantic_fingerprint)
    
    # Normalize if requested
    if normalize and signature_norm > 1e-10:
        semantic_fingerprint_normalized = semantic_fingerprint / signature_norm
    else:
        semantic_fingerprint_normalized = semantic_fingerprint
    
    # Project onto semantic axes
    semantic_scores = {}
    for axis in axes:
        score = semantic_fingerprint_normalized @ axis.axis_vector
        semantic_scores[axis.name] = score
    
    return EigenmodeSemantics(
        mode_index=-1,  # Set by caller
        eigenvalue=-1.0,  # Set by caller
        semantic_fingerprint=semantic_fingerprint,
        semantic_scores=semantic_scores,
        semantic_signature_norm=signature_norm
    )
```

```python
def analyze_spectral_semantics(
    rho: np.ndarray,
    embeddings: np.ndarray,
    axes: List[SemanticAxis]
) -> SpectralSemanticAnalysis:
    """
    Full spectral-semantic analysis of a density matrix.
    
    Args:
        rho: (n, n) density matrix
        embeddings: (n, d) sample embeddings (in same space as axes)
        axes: List of semantic axes
    
    Returns:
        SpectralSemanticAnalysis with all eigenmodes interpreted semantically
    
    Workflow:
        1. Eigendecompose ρ
        2. Map each eigenmode to semantic space
        3. Compute global metrics (entropy, effective rank)
        4. Compute weighted semantic position (Σλᵢ·sᵢₖ for each axis)
    """
    # Eigendecomposition (sorted descending by eigenvalue)
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Map each eigenmode to semantic space
    mode_semantics = []
    for i in range(len(eigenvalues)):
        mode_sem = eigenmode_to_semantic(
            eigenvectors[:, i], embeddings, axes, normalize=True
        )
        mode_sem.mode_index = i
        mode_sem.eigenvalue = eigenvalues[i]
        mode_semantics.append(mode_sem)
    
    # Global metrics
    from model_equality_testing.src.quantum_metrics import von_neumann_entropy
    entropy = von_neumann_entropy(rho)
    effective_rank = np.exp(entropy)
    participation_ratio = 1.0 / np.sum(eigenvalues**2) if np.sum(eigenvalues**2) > 0 else 0
    
    # Weighted semantic position (expectation values)
    weighted_semantic_position = {}
    for axis in axes:
        weighted_pos = sum(
            mode.eigenvalue * mode.semantic_scores[axis.name]
            for mode in mode_semantics
        )
        weighted_semantic_position[axis.name] = weighted_pos
    
    return SpectralSemanticAnalysis(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        mode_semantics=mode_semantics,
        entropy=entropy,
        effective_rank=effective_rank,
        participation_ratio=participation_ratio,
        weighted_semantic_position=weighted_semantic_position
    )
```

```python
def compare_spectral_semantics(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    axes: List[SemanticAxis],
    pca_dim: int = 50,
    compute_mode_alignment: bool = False
) -> SpectralSemanticComparison:
    """
    Compare two distributions via spectral-semantic analysis.
    
    This is the main entry point for spectral-semantic comparison.
    
    Args:
        embeddings_a: (n_a, d) embeddings for sample A
        embeddings_b: (n_b, d) embeddings for sample B
        axes: List of semantic axes
        pca_dim: PCA compression dimension
        compute_mode_alignment: If True, compute cosine similarity between eigenmodes
    
    Returns:
        SpectralSemanticComparison with full analysis
    
    Workflow:
        1. Fit PCA on combined embeddings
        2. Compute density matrices ρ_A and ρ_B
        3. Compute global metrics (trace distance, entropy, divergence)
        4. Eigendecompose both density matrices
        5. Map all eigenmodes to semantic space
        6. Compare eigenvalue spectra
        7. Compare weighted semantic positions
        8. Optionally compute mode alignment matrix
    """
    from model_equality_testing.src.quantum_metrics import (
        fit_pca, pca_density_matrix, trace_distance, 
        von_neumann_entropy, quantum_relative_entropy
    )
    
    # Fit PCA
    pca = fit_pca(np.vstack([embeddings_a, embeddings_b]), k=pca_dim)
    
    # Transform to PCA space
    emb_a_pca = pca.transform(embeddings_a)
    emb_b_pca = pca.transform(embeddings_b)
    
    # Project axes into PCA space
    from model_equality_testing.src.semantic_axes import project_axis_to_pca
    axes_pca = [project_axis_to_pca(axis, pca) for axis in axes]
    
    # Compute density matrices
    rho_a = pca_density_matrix(embeddings_a, pca)
    rho_b = pca_density_matrix(embeddings_b, pca)
    
    # Global metrics
    td = trace_distance(rho_a, rho_b)
    s_a = von_neumann_entropy(rho_a)
    s_b = von_neumann_entropy(rho_b)
    rel_ent = quantum_relative_entropy(rho_a, rho_b)
    
    # Spectral-semantic analysis for each
    analysis_a = analyze_spectral_semantics(rho_a, emb_a_pca, axes_pca)
    analysis_b = analyze_spectral_semantics(rho_b, emb_b_pca, axes_pca)
    
    # Eigenvalue shifts (pad if needed)
    n = max(len(analysis_a.eigenvalues), len(analysis_b.eigenvalues))
    evals_a_padded = np.pad(analysis_a.eigenvalues, (0, n - len(analysis_a.eigenvalues)))
    evals_b_padded = np.pad(analysis_b.eigenvalues, (0, n - len(analysis_b.eigenvalues)))
    eigenvalue_shifts = evals_b_padded - evals_a_padded
    
    # Semantic position shifts
    semantic_position_shifts = {}
    for axis_name in analysis_a.weighted_semantic_position.keys():
        shift = (analysis_b.weighted_semantic_position[axis_name] - 
                analysis_a.weighted_semantic_position[axis_name])
        semantic_position_shifts[axis_name] = shift
    
    # Optional mode alignment
    mode_similarity = None
    if compute_mode_alignment:
        # Cosine similarity between eigenvectors
        mode_similarity = analysis_a.eigenvectors.T @ analysis_b.eigenvectors
    
    return SpectralSemanticComparison(
        analysis_a=analysis_a,
        analysis_b=analysis_b,
        trace_distance=td,
        entropy_diff=s_b - s_a,
        rel_entropy=rel_ent,
        eigenvalue_shifts=eigenvalue_shifts,
        semantic_position_shifts=semantic_position_shifts,
        mode_similarity=mode_similarity
    )
```

### Summary Output Format

```
Spectral-Semantic Analysis
==========================

Distributional Metrics:
  Trace distance:     0.234
  Entropy:            Sample A = 3.82, Sample B = 4.15 (Δ = +0.33)
  Relative entropy:   0.156

Sample A - Dominant Modes:
  Mode 0 (λ=0.387): professionalism: +0.82, formality: +0.45, technical: +0.15
  Mode 1 (λ=0.214): technical: +0.76, professionalism: +0.21, formality: -0.18
  Mode 2 (λ=0.152): formality: +0.71, professionalism: +0.38, technical: +0.05
  Mode 3 (λ=0.089): technical: -0.54, professionalism: +0.42, formality: +0.28
  Mode 4 (λ=0.061): professionalism: -0.61, formality: -0.45, technical: +0.12

  Effective rank: 8.7 modes (entropy: 2.16)
  Weighted semantic position: professionalism: +0.42, technical: +0.28, formality: +0.31

Sample B - Dominant Modes:
  Mode 0 (λ=0.412): technical: +0.79, professionalism: +0.18, formality: +0.22
  Mode 1 (λ=0.198): professionalism: +0.68, technical: +0.35, formality: +0.41
  Mode 2 (λ=0.143): formality: -0.65, professionalism: +0.51, technical: -0.23
  Mode 3 (λ=0.095): professionalism: -0.58, technical: +0.47, formality: +0.15
  Mode 4 (λ=0.071): technical: -0.61, formality: -0.52, professionalism: +0.08

  Effective rank: 10.3 modes (entropy: 2.33)
  Weighted semantic position: professionalism: +0.08, technical: +0.56, formality: +0.10

Spectral Comparison:
  Eigenvalue shifts (top modes):
    Mode 0: +0.025 (dominant mode slightly strengthened)
    Mode 1: -0.016
    Mode 2: -0.009
    Mode 3: +0.006
    Mode 4: +0.010

  Semantic character shifts:
    Mode 0: A was professionalism-heavy, B is technical-heavy (mode reoriented)
    Mode 1: A was technical-heavy, B is professionalism-heavy (mode reoriented)

  Weighted semantic position shifts:
    professionalism:  -0.34 (shifted toward casualness)
    technical:        +0.28 (shifted toward technical)
    formality:        -0.21 (shifted toward informality)

Interpretation: Distribution B exhibits higher diversity (entropy +0.33) with a 
structural reorganization: the dominant mode shifted from professionalism to 
technicality. Overall, B is more technical (+0.28), less professional (-0.34), 
and less formal (-0.21) than A.
```

## The Aesthetic This Achieves

✓ **Unified representation**: Density matrix ρ is the core state  
✓ **Natural decomposition**: Eigendecomposition is fundamental to ρ  
✓ **Semantic axes as interpretive layer**: No separate analysis, just interpretation  
✓ **Theoretically grounded**: Standard spectral theory + inner products  
✓ **No forced decomposition**: Not claiming ρ = ρ_sem + ρ_res  
✓ **Rich structural insight**: Understand *which modes* differ semantically  
✓ **Weighted semantic position**: Automatically gives expectation values  

## Relationship to Semantic Observables

The **weighted semantic position** in spectral analysis is mathematically equivalent to the **expectation value** in observable analysis:

**Spectral approach**:
$$\bar{s}_k = \sum_i \lambda_i \, s_{ik} = \sum_i \lambda_i \, (\vec{m}_i \cdot v_k)$$

**Observable approach**:
$$\langle A_k \rangle = \text{trace}(\rho A_k)$$

**These are the same!** (can be proven algebraically)

**What differs**:
- **Spectral**: Gives you the full mode-by-mode breakdown (which modes contribute to this average?)
- **Observable**: Gives you the expectation directly, plus variance/uncertainty

**They complement each other**:
- Use observables for: expectation values, variance, uncertainty quantification
- Use spectral for: mode-level interpretation, understanding structural shifts

## Comparison to Other Approaches

### v1 (Implemented): Semantic Axes as Separate Analysis
- Density matrices: ρ_A, ρ_B → metrics
- Semantic axes: project embeddings, t-tests, effect sizes
- **Issue**: Feels "bolted on", no theoretical unity

### v2 (Broken Plan): Semantic Subspace Decomposition
- Separate ρ_sem and ρ_res with independent normalizations
- Report "fraction explained" from ratios
- **Issue**: Mathematically incoherent, forces decomposition that doesn't work

### v3a (Observable Proposal): Semantic Observables
- Single ρ as quantum state
- Semantic axes as Hermitian operators A
- Measurements via trace(ρA)
- **Advantage**: Direct expectation values, variance, uncertainty

### v3b (This Proposal): Spectral-Semantic Analysis
- Single ρ as quantum state
- Eigendecomposition reveals modes
- Semantic interpretation of each mode
- **Advantage**: Mode-level insight, structural understanding

**Both v3a and v3b are theoretically coherent and achieve the unified aesthetic.**

## When to Use Each Approach

### Use Spectral-Semantic When:

✓ You want to understand **structural changes** (which modes shifted?)  
✓ You care about **mode-level interpretation** (what does each principal component mean semantically?)  
✓ You want to **diagnose reorganization** (did the dominant mode change character?)  
✓ You're interested in **diversity changes** (entropy, effective rank)  
✓ You want the **full breakdown** (how each mode contributes to the average)  

### Use Semantic Observables When:

✓ You want **direct expectation values** (mean semantic score)  
✓ You need **uncertainty quantification** (variance, standard deviation)  
✓ You want **simpler output** (just means and variances, not mode breakdown)  
✓ You're exploring **many semantic axes** (observable formalism is cleaner for large k)  
✓ You want to compute **commutators** (are axes compatible observables?)  

### Use Both When:

✓ You want **complete understanding**: observables for expectations/variance, spectral for mode interpretation  
✓ You're doing **comprehensive analysis**: use spectral to identify key modes, then observables for precise measurements  
✓ You want **validation**: weighted semantic position (spectral) should match expectation (observable)  

## Implementation Notes

### Where to Add This Code

**File**: `model_equality_testing/src/semantic_axes.py` (extend existing module)

**New classes**:
- `EigenmodeSemantics` - Semantic signature of one eigenmode
- `SpectralSemanticAnalysis` - Full analysis of one density matrix
- `SpectralSemanticComparison` - Comparison of two distributions

**New functions**:
- `eigenmode_to_semantic(eigenvector, embeddings, axes)` - Map mode to semantic space
- `analyze_spectral_semantics(rho, embeddings, axes)` - Full single-distribution analysis
- `compare_spectral_semantics(emb_a, emb_b, axes, pca_dim)` - Main entry point

**Integration**:
- Imports from `quantum_metrics.py`: eigendecomposition (via numpy), metrics
- Imports from existing `semantic_axes.py`: `SemanticAxis`, `project_axis_to_pca`
- No changes to existing code required

### Testing Strategy

1. **Eigenmode mapping**: Verify $\vec{m}_i = E^T \psi_i$ is computed correctly
2. **Weighted semantic position**: Verify $\sum_i \lambda_i s_{ik} = \langle A_k \rangle$ (matches observable)
3. **Normalization**: Test both normalized and unnormalized fingerprints
4. **Sorting**: Verify modes are sorted by eigenvalue descending
5. **End-to-end**: Load real data, interpret dominant modes, verify semantic signatures are sensible

### Computational Complexity

- Eigendecomposition: $O(n^3)$ for n×n density matrix
- Mode-to-semantic mapping: $O(ndk)$ for n modes, d dimensions, k axes
- Same complexity as observable approach (both do eigendecomposition)

### Advanced Extensions (Future)

1. **Mode tracking across versions**: Track how Mode 1 evolves across model updates
2. **Mode clustering**: Group modes by semantic similarity
3. **Semantic PCA**: Are the top semantic axes aligned with top eigenmodes?
4. **Time series**: Track eigenspectrum + semantics across many model versions
5. **Visualization**: Heatmap of semantic signature matrix S

## Example Use Case

### Comparing fp32 vs int8 Quantization

**Question**: How does int8 quantization change Llama-3-8B's output distribution?

**Spectral-Semantic Analysis**:

1. **Compute density matrices** (ρ_fp32, ρ_int8)
2. **Global metrics**: Trace distance = 0.18 (moderate difference)
3. **Entropy**: fp32 = 3.2, int8 = 3.5 (int8 more diverse)
4. **Eigendecomposition + semantic interpretation**:

   **fp32 dominant modes**:
   - Mode 0 (λ=0.42): professionalism +0.85, technical +0.32
   - Mode 1 (λ=0.21): technical +0.78, professionalism +0.15
   
   **int8 dominant modes**:
   - Mode 0 (λ=0.38): professionalism +0.71, technical +0.45
   - Mode 1 (λ=0.24): technical +0.81, professionalism +0.08

5. **Interpretation**:
   - Mode 0 weakened (0.42 → 0.38) and became less professionally-aligned
   - Mode 1 strengthened (0.21 → 0.24) and became more technically-focused
   - Overall: int8 is slightly less professional, slightly more technical, with higher diversity

**Insight**: The spectral view reveals that quantization didn't just shift the mean - it **restructured the distribution**, weakening the professionalism mode and strengthening the technicality mode.

This is information you couldn't get from just expectation values alone.

## Summary

This proposal achieves the theoretical aesthetic through **spectral decomposition + semantic interpretation**:

- **Density matrix as central representation** - the quantum state
- **Eigendecomposition reveals structure** - the natural modes
- **Semantic axes interpret modes** - what does each mode mean?
- **No Frankenstein** - everything flows from eigendecomposition of ρ
- **Rich insight** - understand which modes differ and what they represent

The key insight: **Don't create separate density matrices for subspaces. Instead, interpret the eigenmodes of the original density matrix using semantic lenses.**

This is theoretically rigorous, computationally efficient, and provides mode-level understanding unavailable from expectation values alone.
