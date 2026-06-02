# Semantic Stratification: A Coherent Integration

## The Problem

Both previous approaches failed:
- **Subspace decomposition**: Separate normalizations break mathematical decomposition
- **Diagonal observables**: Only use diagonal of ρ, ignore similarity structure, sample-dependent operators

**Root cause**: Trying to apply density matrices to different sample spaces (n_a ≠ n_b) while claiming to measure "the same" semantic quantity.

## The Solution: Stratification

**Core insight**: Instead of constructing separate observables for each sample, **stratify samples by semantic axes** and analyze density matrices within each stratum.

This achieves genuine integration:
- Density matrices remain the central analytical object
- Semantic axes define the stratification (partition samples into semantic bins)
- We analyze **how distributional structure differs across semantic strata**

## Mathematical Formulation

### Semantic Stratification

Given semantic axis v and samples with embeddings $\{e_1, \ldots, e_n\}$:

**Step 1**: Compute semantic projections
$$p_i = e_i \cdot v \quad \text{for } i = 1, \ldots, n$$

**Step 2**: Partition samples into K bins based on quantiles of projections

For K=3 (low/medium/high):
- Bin 1 (low): samples with $p_i < q_{33}$
- Bin 2 (medium): samples with $q_{33} \leq p_i < q_{67}$
- Bin 3 (high): samples with $p_i \geq q_{67}$

**Step 3**: For each bin k, construct density matrix over samples in that bin

$$\rho^{(k)} = \frac{G^{(k)}}{\text{trace}(G^{(k)})}$$

where $G^{(k)}$ is the Gram matrix restricted to samples in bin k.

### Two-Distribution Comparison

For distributions A and B, stratify both by the same semantic axis:

**Distribution A**:
- $\rho_A^{(1)}, \rho_A^{(2)}, \ldots, \rho_A^{(K)}$ (K density matrices, one per bin)
- Bin weights: $w_A^{(k)}$ = fraction of samples in bin k

**Distribution B**:
- $\rho_B^{(1)}, \rho_B^{(2)}, \ldots, \rho_B^{(K)}$ (K density matrices, one per bin)
- Bin weights: $w_B^{(k)}$ = fraction of samples in bin k

### Two Types of Difference

**1. Marginal difference** (semantic shift):

How did the distribution over bins change?

$$\Delta_{\text{marginal}} = \sum_k |w_A^{(k)} - w_B^{(k)}|$$

This measures **semantic shift**: Did distribution B move toward high or low values on this axis?

**2. Conditional difference** (within-bin structure):

For each bin k, how different are the internal structures?

$$D_k = D(\rho_A^{(k)}, \rho_B^{(k)})$$

This measures: **Among samples with similar semantic scores, how does their distributional structure differ?**

**3. Total difference decomposition**:

The total difference can be understood as:
- Marginal component: distributions shifted across semantic bins
- Conditional component: distributions differ within semantic bins

### Interpretation

**Example output**:

```
Semantic axis: professionalism ← → casualness

Marginal distribution (bin weights):
  Low professionalism:    A = 30%, B = 20%  (Δ = -10%)
  Medium professionalism: A = 40%, B = 35%  (Δ = -5%)
  High professionalism:   A = 30%, B = 45%  (Δ = +15%)
  
  Interpretation: Distribution B shifted toward casualness

Conditional differences (within-bin trace distances):
  Low bin:    D(ρ_A^low, ρ_B^low)    = 0.08 (minor structural difference)
  Medium bin: D(ρ_A^med, ρ_B^med)    = 0.12 (moderate structural difference)
  High bin:   D(ρ_A^high, ρ_B^high)  = 0.19 (major structural difference)
  
  Interpretation: Even among casual samples (high bin), A and B differ 
  substantially in their internal distributional structure

Conditional entropies (within-bin diversity):
  Low bin:    S(ρ_A^low) = 2.1, S(ρ_B^low) = 2.3 (B more diverse)
  Medium bin: S(ρ_A^med) = 2.8, S(ρ_B^med) = 2.7 (similar diversity)
  High bin:   S(ρ_A^high) = 3.2, S(ρ_B^high) = 3.8 (B much more diverse)
```

**Rich semantic interpretation**:
- Distribution B is more casual overall (marginal shift)
- The difference is largest among casual samples (conditional difference in high bin)
- Casual samples in B are more internally diverse (entropy comparison)

## Why This Works

### Theoretically Coherent

✓ **Density matrices are central**: Every measurement uses ρ (stratified versions)  
✓ **Semantic axes define structure**: Partition sample space along interpretable dimensions  
✓ **No separate normalizations**: Each stratum has its own properly normalized ρ  
✓ **Uses full Gram structure**: Within-bin density matrices use all pairwise similarities  
✓ **Fixed comparisons**: ρ_A^{(k)} vs ρ_B^{(k)} are density matrices over "similar semantic regions"  

### Conceptually Clean

- **Marginal analysis**: Where did the mass shift semantically? (bin weights)
- **Conditional analysis**: How did structure change within semantic regions? (trace distances)
- **Diversity analysis**: How diverse is each semantic slice? (entropies)

This is analogous to stratified statistical analysis, applied to density matrices.

### Interpretable

"Distribution B shifted toward casualness (marginal), and among casual samples, the internal structure became more diverse and different from A (conditional)."

This tells you:
1. **What changed** (semantic dimension: casualness)
2. **How much** (marginal shift: +15% in high bin)
3. **Where it matters** (conditional difference largest in high bin)

## Comparison to Failed Approaches

### vs. Subspace Decomposition
- **Subspace**: Tried to split ρ into ρ_sem + ρ_res with separate normalizations ✗
- **Stratification**: Creates multiple ρ^{(k)}, each properly normalized ✓

### vs. Diagonal Observables
- **Observables**: Only used diagonal of ρ, ignored similarities ✗
- **Stratification**: Uses full Gram structure within each bin ✓

### vs. "Bolted On" v1
- **v1**: Semantic projections separate from density matrices ✗
- **Stratification**: Semantic axes define structure for density matrix analysis ✓

## API Design

### Data Structures

```python
@dataclass
class SemanticStratum:
    """A single semantic bin with its density matrix."""
    bin_index: int
    bin_label: str              # "low", "medium", "high"
    bin_range: Tuple[float, float]  # (min_score, max_score)
    sample_indices: np.ndarray  # Which samples are in this bin
    weight: float               # Fraction of total samples
    rho: np.ndarray            # Density matrix for this bin
    entropy: float             # von Neumann entropy
    n_samples: int

@dataclass
class StratifiedDistribution:
    """Distribution stratified by a semantic axis."""
    axis_name: str
    strata: List[SemanticStratum]  # One per bin
    n_bins: int
    global_rho: np.ndarray         # Unstratified density matrix
    global_entropy: float
    
    def bin_weights(self) -> np.ndarray:
        return np.array([s.weight for s in self.strata])

@dataclass
class SemanticStratificationComparison:
    """Compare two distributions via semantic stratification."""
    axis_name: str
    negative_pole: str
    positive_pole: str
    
    # Stratified distributions
    dist_a: StratifiedDistribution
    dist_b: StratifiedDistribution
    
    # Global metrics (unstratified)
    global_trace_distance: float
    global_entropy_diff: float
    
    # Marginal difference (bin weight shifts)
    marginal_shift: np.ndarray      # w_B^(k) - w_A^(k) for each bin
    total_variation: float          # Σ |w_A^(k) - w_B^(k)| / 2
    
    # Conditional differences (within-bin)
    conditional_trace_distances: np.ndarray  # D(ρ_A^(k), ρ_B^(k))
    conditional_entropy_diffs: np.ndarray    # S(ρ_B^(k)) - S(ρ_A^(k))
    
    # Dominant bin (where conditional difference is largest)
    dominant_bin_index: int
    
    def summary(self) -> str:
        """Human-readable interpretation."""
        pass
```

### Core Functions

```python
def stratify_by_semantic_axis(
    embeddings: np.ndarray,
    axis: SemanticAxis,
    n_bins: int = 3,
    bin_method: str = "quantile"
) -> Tuple[List[np.ndarray], np.ndarray, List[Tuple[float, float]]]:
    """
    Partition samples into semantic bins.
    
    Args:
        embeddings: (n, d) sample embeddings
        axis: Semantic axis for stratification
        n_bins: Number of bins (default 3: low/med/high)
        bin_method: "quantile" or "equal_width"
    
    Returns:
        bin_indices: List of index arrays, one per bin
        projections: (n,) semantic scores
        bin_ranges: List of (min, max) for each bin
    """
    projections = embeddings @ axis.axis_vector
    
    if bin_method == "quantile":
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(projections, quantiles)
    else:  # equal_width
        bin_edges = np.linspace(projections.min(), projections.max(), n_bins + 1)
    
    bin_indices = []
    bin_ranges = []
    
    for i in range(n_bins):
        if i == n_bins - 1:
            # Last bin includes right edge
            mask = (projections >= bin_edges[i]) & (projections <= bin_edges[i+1])
        else:
            mask = (projections >= bin_edges[i]) & (projections < bin_edges[i+1])
        
        bin_indices.append(np.where(mask)[0])
        bin_ranges.append((bin_edges[i], bin_edges[i+1]))
    
    return bin_indices, projections, bin_ranges
```

```python
def stratified_distribution(
    embeddings: np.ndarray,
    axis: SemanticAxis,
    pca: Optional[PCA] = None,
    n_bins: int = 3
) -> StratifiedDistribution:
    """
    Create stratified distribution along semantic axis.
    
    Args:
        embeddings: (n, d) sample embeddings
        axis: Semantic axis for stratification
        pca: Optional PCA model (if None, use embeddings as-is)
        n_bins: Number of bins
    
    Returns:
        StratifiedDistribution with density matrix for each bin
    """
    from model_equality_testing.src.quantum_metrics import (
        density_matrix, pca_density_matrix, von_neumann_entropy
    )
    
    # Transform to PCA space if provided
    if pca is not None:
        embeddings_pca = pca.transform(embeddings)
        axis_pca = project_axis_to_pca(axis, pca)
    else:
        embeddings_pca = embeddings
        axis_pca = axis
    
    # Global (unstratified) density matrix
    if pca is not None:
        global_rho = pca_density_matrix(embeddings, pca)
    else:
        G = embeddings_pca @ embeddings_pca.T
        global_rho = G / np.trace(G)
    
    global_entropy = von_neumann_entropy(global_rho)
    
    # Stratify
    bin_indices, projections, bin_ranges = stratify_by_semantic_axis(
        embeddings_pca, axis_pca, n_bins
    )
    
    # Create stratum for each bin
    strata = []
    bin_labels = ["low", "medium", "high"] if n_bins == 3 else [f"bin_{i}" for i in range(n_bins)]
    
    for i, (indices, bin_range) in enumerate(zip(bin_indices, bin_ranges)):
        if len(indices) == 0:
            # Empty bin - skip or create dummy
            continue
        
        # Embeddings in this bin
        bin_embeddings = embeddings_pca[indices]
        
        # Density matrix for this bin
        G_bin = bin_embeddings @ bin_embeddings.T
        rho_bin = G_bin / np.trace(G_bin)
        
        # Entropy
        entropy_bin = von_neumann_entropy(rho_bin)
        
        stratum = SemanticStratum(
            bin_index=i,
            bin_label=bin_labels[i] if i < len(bin_labels) else f"bin_{i}",
            bin_range=bin_range,
            sample_indices=indices,
            weight=len(indices) / len(embeddings),
            rho=rho_bin,
            entropy=entropy_bin,
            n_samples=len(indices)
        )
        strata.append(stratum)
    
    return StratifiedDistribution(
        axis_name=axis.name,
        strata=strata,
        n_bins=n_bins,
        global_rho=global_rho,
        global_entropy=global_entropy
    )
```

```python
def compare_stratified(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    axis: SemanticAxis,
    pca_dim: int = 50,
    n_bins: int = 3
) -> SemanticStratificationComparison:
    """
    Compare distributions via semantic stratification.
    
    Main entry point for stratification analysis.
    
    Args:
        embeddings_a: (n_a, d) embeddings for distribution A
        embeddings_b: (n_b, d) embeddings for distribution B
        axis: Semantic axis for stratification
        pca_dim: PCA compression dimension
        n_bins: Number of semantic bins
    
    Returns:
        SemanticStratificationComparison with all metrics
    """
    from model_equality_testing.src.quantum_metrics import (
        fit_pca, trace_distance
    )
    
    # Fit PCA on combined embeddings
    pca = fit_pca(np.vstack([embeddings_a, embeddings_b]), k=pca_dim)
    
    # Stratify both distributions
    dist_a = stratified_distribution(embeddings_a, axis, pca, n_bins)
    dist_b = stratified_distribution(embeddings_b, axis, pca, n_bins)
    
    # Global metrics
    global_td = trace_distance(dist_a.global_rho, dist_b.global_rho)
    global_entropy_diff = dist_b.global_entropy - dist_a.global_entropy
    
    # Marginal shift (bin weight differences)
    weights_a = dist_a.bin_weights()
    weights_b = dist_b.bin_weights()
    marginal_shift = weights_b - weights_a
    total_variation = np.sum(np.abs(marginal_shift)) / 2
    
    # Conditional differences (within-bin)
    conditional_tds = []
    conditional_entropy_diffs = []
    
    for stratum_a, stratum_b in zip(dist_a.strata, dist_b.strata):
        # Trace distance between same bin in A vs B
        td = trace_distance(stratum_a.rho, stratum_b.rho)
        conditional_tds.append(td)
        
        # Entropy difference
        entropy_diff = stratum_b.entropy - stratum_a.entropy
        conditional_entropy_diffs.append(entropy_diff)
    
    conditional_tds = np.array(conditional_tds)
    conditional_entropy_diffs = np.array(conditional_entropy_diffs)
    
    # Dominant bin (largest conditional difference)
    dominant_bin_index = np.argmax(conditional_tds)
    
    return SemanticStratificationComparison(
        axis_name=axis.name,
        negative_pole=axis.negative_pole,
        positive_pole=axis.positive_pole,
        dist_a=dist_a,
        dist_b=dist_b,
        global_trace_distance=global_td,
        global_entropy_diff=global_entropy_diff,
        marginal_shift=marginal_shift,
        total_variation=total_variation,
        conditional_trace_distances=conditional_tds,
        conditional_entropy_diffs=conditional_entropy_diffs,
        dominant_bin_index=dominant_bin_index
    )
```

```python
def multi_axis_stratification(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    axes: List[SemanticAxis],
    pca_dim: int = 50,
    n_bins: int = 3
) -> List[SemanticStratificationComparison]:
    """
    Stratify along multiple semantic axes.
    
    Returns one stratification comparison per axis.
    """
    comparisons = []
    
    # Fit PCA once
    from model_equality_testing.src.quantum_metrics import fit_pca
    pca = fit_pca(np.vstack([embeddings_a, embeddings_b]), k=pca_dim)
    
    for axis in axes:
        comp = compare_stratified(embeddings_a, embeddings_b, axis, pca_dim, n_bins)
        comparisons.append(comp)
    
    return comparisons
```

### Summary Output

```
Semantic Stratification Analysis
=================================

Global Metrics:
  Trace distance:  0.234
  Entropy:         A = 3.82, B = 4.15 (Δ = +0.33)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Axis: professionalism ← → casualness

Marginal Distribution (bin weights):
  Low (professional):    A = 30%, B = 20%  (Δ = -10%)
  Medium:                A = 40%, B = 35%  (Δ = -5%)
  High (casual):         A = 30%, B = 45%  (Δ = +15%)
  
  Total variation distance: 0.15
  → Distribution B shifted toward casualness

Conditional Analysis (within-bin structure):
  Low bin:     D = 0.08, ΔS = +0.2  (minor difference, B slightly more diverse)
  Medium bin:  D = 0.12, ΔS = -0.1  (moderate difference, A more diverse)
  High bin:    D = 0.19, ΔS = +0.6  (★ major difference, B much more diverse)

  Dominant region: High (casual) bin
  → Among casual samples, A and B differ substantially (D=0.19)
  → Casual samples in B are much more internally diverse (ΔS=+0.6)

Interpretation: 
  Distribution B is more casual overall (15% more weight in high bin).
  The largest structural difference is among casual samples, where B exhibits
  much higher internal diversity than A.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Axis: technical ← → layperson
[similar structure for each axis...]
```

## The Aesthetic This Achieves

✓ **Density matrix as central representation**: Every measurement is a density matrix  
✓ **Semantic axes define structure**: Partition sample space along interpretable dimensions  
✓ **Uses full Gram structure**: Within-bin density matrices use all pairwise similarities  
✓ **Theoretically grounded**: Standard stratified analysis + quantum metrics  
✓ **No forced decomposition**: Each stratum is independently normalized (correctly)  
✓ **Rich interpretation**: Marginal (semantic shift) + conditional (structural difference within regions)  
✓ **No Frankenstein**: Unified framework where semantic axes structure density matrix analysis  

## What This Tells You That Other Approaches Don't

**Scenario**: Comparing fp32 vs int8 quantization on professionalism axis

**Simple projection** (v1): "int8 is 0.3 points more casual on average"

**Observables** (failed v3a): "int8 has ⟨A⟩ = +0.3, σ = 0.2" (just dressed-up mean/variance)

**Stratification** (this proposal):
- "int8 shifted 15% more weight to casual bin (marginal)"
- "Among casual samples, fp32 and int8 differ substantially (D=0.19)"
- "Casual int8 samples are much more diverse internally (ΔS=+0.6)"

**New insights**:
- WHERE the difference is concentrated semantically (casual region)
- HOW the structure differs within that region (diversity, trace distance)
- BOTH location shift (marginal) and structural change (conditional)

This is information you can't get from simple projections OR diagonal observables.

## Extensions

### 1. Joint Stratification
Stratify by multiple axes simultaneously:
- Create 2D grid: (low-prof, low-tech), (low-prof, high-tech), etc.
- Analyze marginal distributions and conditional structure
- Identify semantic "hot spots" where differences concentrate

### 2. Adaptive Binning
Choose bin boundaries to maximize separation:
- Find thresholds that best distinguish distributions A and B
- Report: "Difference is sharpest at professionalism score = 0.4"

### 3. Continuous Stratification
Instead of discrete bins, use kernel density estimation:
- Weight samples by semantic similarity to a target value
- Continuous version of stratification

### 4. Hierarchical Stratification
First stratify by axis 1, then within each bin, stratify by axis 2:
- Tree structure of semantic regions
- Most interpretable for small numbers of axes

## Computational Complexity

- Fit PCA: $O(n d^2 + d^3)$ - once
- Stratify: $O(n \log n)$ - sorting by projection
- Per-bin density matrix: $O(n_k^3)$ where $n_k$ is bin size
- For K bins: $O(K \cdot (n/K)^3) = O(n^3/K^2)$ total

**Advantage**: Smaller density matrices (n/K per bin) than full n×n matrix.

For n=100, K=3: Each bin has ~33 samples → 33³ = 36K operations per bin vs 100³ = 1M for full matrix.

**Stratification is actually faster than full density matrix!**

## Summary

Semantic stratification achieves true integration:

1. **Density matrices are central** - all measurements are quantum metrics on ρ
2. **Semantic axes define structure** - partition samples into interpretable regions
3. **Uses full similarity structure** - within-bin Gram matrices preserve all pairwise relationships
4. **Theoretically coherent** - no forced decompositions, proper normalizations
5. **Rich interpretation** - marginal shifts + conditional differences + diversity changes
6. **Computationally efficient** - smaller matrices per bin

**The key insight**: Don't try to make semantic axes into observables or decompose ρ. Instead, use semantic axes to **stratify the sample space**, then analyze density matrices within each stratum.

This is analogous to stratified statistical analysis (common in epidemiology, social science) applied to quantum distributional metrics. It's a natural, well-established framework that actually works mathematically and provides rich semantic interpretation.
