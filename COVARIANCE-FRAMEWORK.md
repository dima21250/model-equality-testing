# Covariance Matrix Framework: Origin and Metrics

## What Is a Covariance Matrix?

**Origin**: Classical multivariate statistics, not quantum mechanics.

For a dataset with d-dimensional samples $\{x_1, x_2, \ldots, x_n\} \in \mathbb{R}^d$:

**Step 1 - Center the data**:
$$\mu = \frac{1}{n} \sum_{i=1}^n x_i$$
$$\tilde{x}_i = x_i - \mu$$

**Step 2 - Covariance matrix**:
$$C = \frac{1}{n} \sum_{i=1}^n \tilde{x}_i \tilde{x}_i^T = \frac{1}{n} \tilde{X}^T \tilde{X}$$

where $\tilde{X}$ is the (n×d) centered data matrix.

**Result**: $C \in \mathbb{R}^{d \times d}$ is a symmetric positive semi-definite matrix.

**Interpretation**: 
- Diagonal C[i,i]: variance of feature i
- Off-diagonal C[i,j]: covariance between features i and j
- Captures second-order statistics (how features vary together)

## Relationship to Gram Matrix (Density Matrix)

**Gram matrix** (what you're currently using):
$$G = X X^T \in \mathbb{R}^{n \times n}$$
- Entry G[i,j] = $x_i^T x_j$ (similarity between samples i and j)
- Sample-space representation
- Size depends on sample count n

**Covariance matrix**:
$$C = X^T X \in \mathbb{R}^{d \times d}$$
- Entry C[i,j] = covariance between features i and j
- Feature-space representation  
- Size is always d×d (independent of sample count)

**They're transposes of the same operation!**

Both are positive semi-definite, both have eigendecompositions, both encode distributional structure - just in different spaces.

## What Framework Does This Come From?

### 1. Multivariate Statistics

Covariance matrices are fundamental in:
- **Principal Component Analysis (PCA)**: Eigendecomposition of C gives principal components
- **Factor Analysis**: Decompose C into common factors + noise
- **Linear Discriminant Analysis**: Between-class vs within-class covariance
- **Canonical Correlation Analysis**: Joint covariance of two variable sets

### 2. Gaussian Distributions

For multivariate Gaussian: $x \sim \mathcal{N}(\mu, C)$

The covariance matrix C **fully characterizes** the distribution (along with mean μ).

Log-likelihood:
$$\log p(x | \mu, C) = -\frac{1}{2}\left[ (x-\mu)^T C^{-1} (x-\mu) + \log|C| + d\log(2\pi) \right]$$

Many metrics assume Gaussian distributions and use covariance matrices.

### 3. Riemannian Geometry

The space of d×d positive definite matrices forms a **Riemannian manifold**.

This manifold has well-studied geometry:
- Tangent spaces
- Geodesics (shortest paths)
- Riemannian metrics
- Curvature

**Key insight**: Covariance matrices don't live in Euclidean space! 

Example: The average of two covariance matrices may not be a valid covariance matrix (could lose positive definiteness).

Proper operations respect the manifold structure.

### 4. Information Geometry

The manifold of Gaussian distributions (parameterized by μ and C) has:
- Fisher information metric
- Connections to KL divergence
- Natural gradient descent
- Exponential families

**Applications**:
- Brain-computer interfaces (covariance of EEG signals)
- Diffusion tensor imaging (medical imaging, covariance of water diffusion)
- Finance (covariance of asset returns)
- Robotics (uncertainty propagation)

## Available Metrics on Covariance Matrices

### 1. Euclidean Metrics (Simple but Flawed)

**Frobenius norm**:
$$d_F(C_A, C_B) = ||C_A - C_B||_F = \sqrt{\sum_{i,j} (C_A[i,j] - C_B[i,j])^2}$$

**Advantages**: 
- Simple to compute
- Convex

**Disadvantages**:
- Not invariant under linear transformations
- Treats matrices as flat Euclidean space (ignores manifold structure)
- Can give misleading results

### 2. Log-Euclidean Distance (Better)

$$d_{LE}(C_A, C_B) = ||\log(C_A) - \log(C_B)||_F$$

where log is the matrix logarithm.

**Advantages**:
- Respects positive definiteness (stays on manifold)
- Affine-invariant (invariant under linear transformations)
- Computationally efficient

**Disadvantages**:
- Requires C to be positive definite (not just semi-definite)
- Sensitive to small eigenvalues

**This is widely used** in diffusion tensor imaging and SPD matrix analysis.

### 3. Affine-Invariant Riemannian Distance (Gold Standard)

$$d_{AI}(C_A, C_B) = ||\log(C_A^{-1/2} C_B C_A^{-1/2})||_F = \sqrt{\sum_i (\log \lambda_i)^2}$$

where λᵢ are eigenvalues of $C_A^{-1} C_B$.

**Advantages**:
- True geodesic distance on the Riemannian manifold
- Affine-invariant
- Theoretically optimal

**Disadvantages**:
- More expensive to compute
- Requires C_A to be positive definite (invertible)

**This is the "correct" metric** from the Riemannian geometry perspective.

### 4. Stein Divergence (Information-Theoretic)

$$S(C_A, C_B) = \log \frac{|C_A + C_B|}{|C_A|^{1/2} |C_B|^{1/2}}$$

where |·| denotes determinant.

**Advantages**:
- Information-theoretic interpretation
- Symmetric
- Works for singular matrices (determinant can be zero if regularized)

**Disadvantages**:
- Not a metric (doesn't satisfy triangle inequality)
- Sensitive to conditioning

### 5. KL Divergence (Gaussian Assumption)

If you assume $p_A \sim \mathcal{N}(\mu_A, C_A)$ and $p_B \sim \mathcal{N}(\mu_B, C_B)$:

$$D_{KL}(p_A || p_B) = \frac{1}{2}\left[ \text{tr}(C_B^{-1} C_A) + (\mu_B - \mu_A)^T C_B^{-1} (\mu_B - \mu_A) - d + \log\frac{|C_B|}{|C_A|} \right]$$

**Advantages**:
- Direct probabilistic interpretation
- Connects to information theory

**Disadvantages**:
- Assumes Gaussian distributions (may not be valid for LLM outputs)
- Requires C_B to be invertible
- Asymmetric

### 6. Wasserstein Distance (Gaussian)

For Gaussians:

$$W_2^2(p_A, p_B) = ||\mu_A - \mu_B||^2 + \text{tr}(C_A + C_B - 2(C_A^{1/2} C_B C_A^{1/2})^{1/2})$$

**Advantages**:
- Optimal transport interpretation
- Metric (satisfies triangle inequality)
- Widely used in machine learning

**Disadvantages**:
- Expensive to compute (matrix square roots)
- Assumes Gaussian distributions

### 7. Eigenvalue-Based Metrics

**Trace ratio**:
$$r = \frac{\text{tr}(C_A)}{\text{tr}(C_B)}$$

**Determinant ratio** (volume ratio):
$$r = \frac{|C_A|}{|C_B|}$$

**Eigenvalue divergence**:
$$d_{eig}(C_A, C_B) = \sum_i |\lambda_i^A - \lambda_i^B|$$

where eigenvalues are sorted.

**Advantages**:
- Simple
- Interpretable (volume, total variance)

**Disadvantages**:
- Ignores eigenvector alignment
- Not full metrics (don't capture all structure)

## What Metrics Should You Use?

For comparing LLM output distributions via covariance matrices:

### Recommended Primary Metric: Log-Euclidean Distance

$$d_{LE}(C_A, C_B) = ||\log(C_A) - \log(C_B)||_F$$

**Why**:
- Theoretically grounded (respects manifold structure)
- Computationally efficient
- Well-studied in SPD matrix analysis
- Affine-invariant

**Implementation**:
```python
import scipy.linalg

def log_euclidean_distance(C_a, C_b):
    """Log-Euclidean distance between covariance matrices."""
    log_C_a = scipy.linalg.logm(C_a)
    log_C_b = scipy.linalg.logm(C_b)
    return np.linalg.norm(log_C_a - log_C_b, 'fro')
```

### Secondary Metric: Frobenius (for simplicity)

$$d_F(C_A, C_B) = ||C_A - C_B||_F$$

**Why**:
- Fast
- Easy to interpret
- Works as a rough approximation

### Tertiary: KL Divergence (if Gaussian assumption is reasonable)

**Why**:
- Connects to your existing framework (quantum relative entropy)
- Probabilistic interpretation
- Asymmetric (can measure both directions)

## Analogy to Your Current Metrics

| Gram/Density Matrix (n×n) | Covariance Matrix (d×d) |
|---|---|
| Trace distance: D(ρ_A, ρ_B) | Log-Euclidean: d_LE(C_A, C_B) |
| Von Neumann entropy: S(ρ) | Differential entropy: H(𝒩(0,C)) = ½log|C| + const |
| Quantum relative entropy: S(ρ_A\|\|ρ_B) | KL divergence: D_KL(𝒩(μ_A,C_A) \|\| 𝒩(μ_B,C_B)) |

**The structure is parallel!** Just in a different space.

## Semantic Integration with Covariance Matrices

**Variance along semantic axis v**:
$$\sigma_v^2 = v^T C v$$

This is the variance of projections onto axis v.

**Comparison**:
$$\Delta_v = (v^T C_A v) - (v^T C_B v)$$

Positive: Distribution B has more variance along axis v  
Negative: Distribution B has less variance along axis v

**Covariance between semantic axes v and w**:
$$\text{Cov}(v, w) = v^T C w$$

This measures how two semantic dimensions covary.

**Mahalanobis distance to mean along axis v**:

For a point x in the distribution:
$$(x - \mu)^T C^{-1} (x - \mu)$$

This is the "statistical distance" accounting for correlation structure.

## Advantages for Your Use Case

**1. Fixed dimensionality**: C is always d×d (50×50 after PCA), regardless of sample count

**2. Natural semantic integration**: 
- Project covariance: v^T C v
- No sample-count dependency
- Well-defined for different sample sizes

**3. Rich theory**: Decades of literature on covariance matrix comparison

**4. Multiple metrics**: Choose based on assumptions (Gaussian vs non-parametric, computational cost, etc.)

**5. Interpretability**: Variance, correlation, principal components all have clear meanings

## Disadvantages vs Density Matrices

**1. Only captures second-order statistics**:
- Covariance matrix only encodes mean and variance/covariance
- Doesn't capture higher-order moments (skewness, kurtosis)
- Assumes symmetric structure around the mean

**2. Less granular**:
- Density matrix: n×n, captures all pairwise sample similarities
- Covariance: d×d, captures feature covariances (aggregated over samples)

**3. Different interpretation**:
- Density matrix: "quantum state" (if you want that aesthetic)
- Covariance: classical statistical object

**4. Gaussian assumption** (for some metrics):
- KL divergence, Wasserstein assume Gaussian
- May not be valid for LLM outputs
- Log-Euclidean doesn't assume Gaussian (just uses matrix geometry)

## Summary

**Covariance matrices come from**:
- Multivariate statistics
- Riemannian geometry (manifold of SPD matrices)
- Information geometry

**They support**:
- Log-Euclidean distance (recommended primary)
- Frobenius norm (simple approximation)
- Affine-invariant Riemannian distance (theoretically optimal)
- KL divergence (if Gaussian assumption holds)
- Wasserstein distance (optimal transport)
- Many eigenvalue-based metrics

**For semantic integration**:
- Variance along axis: v^T C v
- Natural, well-defined, sample-size independent
- Multiple axes via quadratic forms

**Tradeoff**:
- ✓ Fixed dimensionality (solves sample-size problem)
- ✓ Rich metric theory
- ✓ Natural semantic integration
- ✗ Only second-order statistics (not full distribution)
- ✗ Different aesthetic (classical stats, not quantum)

**Is this a good fit for your goals?** It depends on whether:
1. You're okay moving from "quantum" to "classical" framing
2. Second-order statistics (variance, covariance) are sufficient
3. You value genuine semantic integration over specific quantum interpretation
