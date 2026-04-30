"""Test UMAP-based fix for trace distance sample-size artifact.

This script tests whether UMAP dimensionality reduction eliminates the
sample-size dependence while providing better discrimination than PCA.

Approach (Option 1 - fit on combined):
1. Combine embeddings from both distributions: combined = [A; B]
2. Fit UMAP on combined data
3. Transform A and B separately using the fitted projection
4. Construct k×k covariance matrices (fixed size)
5. Normalize to density matrices
6. Compute trace distance

Key differences from PCA:
- Non-linear projection (preserves manifold structure)
- Topological focus (preserves neighborhoods)
- May better capture subtle distribution shifts
"""

import numpy as np
from umap import UMAP


def compute_density_matrix_umap(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    n_components: int = 20,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
):
    """Construct density matrices via UMAP (Option 1: fit on combined).

    Args:
        embeddings_a: (N_a, d) array
        embeddings_b: (N_b, d) array
        n_components: Target dimensionality (k)
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        random_state: Random seed for reproducibility

    Returns:
        (rho_a, rho_b): Tuple of (k, k) density matrices
    """
    N_a, d = embeddings_a.shape
    N_b, _ = embeddings_b.shape

    # Combine embeddings
    combined = np.vstack([embeddings_a, embeddings_b])

    # Fit UMAP on combined data
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        verbose=False,
    )

    # Transform combined data
    reduced_combined = reducer.fit_transform(combined)

    # Split back
    reduced_a = reduced_combined[:N_a]
    reduced_b = reduced_combined[N_a:]

    # Construct covariance matrices
    cov_a = (reduced_a.T @ reduced_a) / N_a
    cov_b = (reduced_b.T @ reduced_b) / N_b

    # Normalize to density matrices
    trace_a = np.trace(cov_a)
    trace_b = np.trace(cov_b)

    if trace_a < 1e-10 or trace_b < 1e-10:
        raise ValueError(f"Covariance trace too small: {trace_a}, {trace_b}")

    rho_a = cov_a / trace_a
    rho_b = cov_b / trace_b

    return rho_a, rho_b


def trace_distance(rho_a: np.ndarray, rho_b: np.ndarray) -> float:
    """Compute trace distance."""
    diff = rho_a - rho_b
    eigenvalues = np.linalg.eigvalsh(diff)
    return 0.5 * np.sum(np.abs(eigenvalues))


def test_identical_distributions(n_components=20, n_neighbors=15):
    """Test 1: Two samples from the same Gaussian - UMAP version."""
    print("="*80)
    print(f"Test 1: Identical Distributions (UMAP k={n_components}, neighbors={n_neighbors})")
    print("="*80)

    np.random.seed(42)
    d = 768
    n_samples = 100

    embeddings_a = np.random.randn(n_samples, d)
    embeddings_b = np.random.randn(n_samples, d)

    # Normalize to unit vectors
    embeddings_a = embeddings_a / np.linalg.norm(embeddings_a, axis=1, keepdims=True)
    embeddings_b = embeddings_b / np.linalg.norm(embeddings_b, axis=1, keepdims=True)

    # Compute density matrices via UMAP
    rho_a, rho_b = compute_density_matrix_umap(
        embeddings_a, embeddings_b,
        n_components=n_components,
        n_neighbors=n_neighbors,
    )

    # Compute trace distance
    dist = trace_distance(rho_a, rho_b)

    print(f"Trace distance (UMAP): {dist:.6f}")
    print(f"Expected: Close to 0 (same distribution)")
    print(f"Status: {'✓ PASS' if dist < 0.1 else '✗ FAIL'}")
    print()

    return dist


def test_orthogonal_distributions(n_components=20, n_neighbors=15):
    """Test 2: Orthogonal subspaces - UMAP version."""
    print("="*80)
    print(f"Test 2: Orthogonal Distributions (UMAP k={n_components})")
    print("="*80)

    np.random.seed(42)
    d = 768
    n_samples = 100

    # Sample A: random vectors in first d/2 dimensions
    embeddings_a = np.zeros((n_samples, d))
    embeddings_a[:, :d//2] = np.random.randn(n_samples, d//2)

    # Sample B: random vectors in second d/2 dimensions
    embeddings_b = np.zeros((n_samples, d))
    embeddings_b[:, d//2:] = np.random.randn(n_samples, d//2)

    # Normalize
    embeddings_a = embeddings_a / np.linalg.norm(embeddings_a, axis=1, keepdims=True)
    embeddings_b = embeddings_b / np.linalg.norm(embeddings_b, axis=1, keepdims=True)

    # Compute density matrices via UMAP
    rho_a, rho_b = compute_density_matrix_umap(
        embeddings_a, embeddings_b,
        n_components=n_components,
        n_neighbors=n_neighbors,
    )

    # Compute trace distance
    dist = trace_distance(rho_a, rho_b)

    print(f"Trace distance (UMAP): {dist:.6f}")
    print(f"Expected: >0.3 (orthogonal subspaces)")
    print(f"Status: {'✓ PASS' if dist > 0.3 else '⚠️  May be compressed by UMAP'}")
    print()

    return dist


def test_gaussian_shift(n_components=20, n_neighbors=15):
    """Test 3: Shifted Gaussians - UMAP version."""
    print("="*80)
    print(f"Test 3: Shifted Gaussians (UMAP k={n_components})")
    print("="*80)

    np.random.seed(42)
    d = 768
    n_samples = 100

    # Sample A: N(0, I)
    embeddings_a = np.random.randn(n_samples, d)

    # Sample B: N(shift, I)
    shift = np.zeros(d)
    shift[0] = 2.0
    embeddings_b = np.random.randn(n_samples, d) + shift

    # Normalize
    embeddings_a = embeddings_a / np.linalg.norm(embeddings_a, axis=1, keepdims=True)
    embeddings_b = embeddings_b / np.linalg.norm(embeddings_b, axis=1, keepdims=True)

    # Compute density matrices via UMAP
    rho_a, rho_b = compute_density_matrix_umap(
        embeddings_a, embeddings_b,
        n_components=n_components,
        n_neighbors=n_neighbors,
    )

    # Compute trace distance
    dist = trace_distance(rho_a, rho_b)

    print(f"Trace distance (UMAP): {dist:.6f}")
    print(f"Expected: Between 0 and 1")
    print(f"Status: {'✓ PASS' if 0.0 < dist < 1.0 else '✗ FAIL'}")
    print()

    return dist


def test_sample_size_sensitivity(n_components=20, n_neighbors=15):
    """Test 4: CRITICAL - Does trace distance grow with N? (UMAP version)"""
    print("="*80)
    print(f"Test 4: Sample Size Sensitivity (UMAP k={n_components})")
    print("="*80)

    np.random.seed(42)
    d = 768
    sample_sizes = [50, 100, 200, 500, 1000]

    print("n_samples | Trace Distance (UMAP) | Status")
    print("-" * 55)

    distances = []
    for n in sample_sizes:
        embeddings_a = np.random.randn(n, d)
        embeddings_b = np.random.randn(n, d)

        embeddings_a = embeddings_a / np.linalg.norm(embeddings_a, axis=1, keepdims=True)
        embeddings_b = embeddings_b / np.linalg.norm(embeddings_b, axis=1, keepdims=True)

        rho_a, rho_b = compute_density_matrix_umap(
            embeddings_a, embeddings_b,
            n_components=n_components,
            n_neighbors=n_neighbors,
        )

        dist = trace_distance(rho_a, rho_b)
        distances.append(dist)

        status = "✓" if dist < 0.2 else "✗"
        print(f"{n:9d} | {dist:20.6f} | {status}")

    print()
    print(f"Trend: {distances[0]:.3f} → {distances[-1]:.3f}")

    # Success criteria
    is_increasing = distances[-1] > distances[0] * 1.5
    is_stable = distances[-1] < 0.3

    if is_stable and not is_increasing:
        print(f"Status: ✓ PASS - Distance stable across sample sizes")
    elif is_increasing:
        print(f"Status: ✗ FAIL - Distance still increases with N")
    else:
        print(f"Status: ⚠️  PARTIAL - Not increasing but absolute values high")

    print()

    return distances


def test_resampling_from_same_set(n_components=20, n_neighbors=15):
    """Test 5: Resampling from same pool - UMAP version."""
    print("="*80)
    print(f"Test 5: Resampling from Same Pool (UMAP k={n_components})")
    print("="*80)

    np.random.seed(42)
    d = 768
    pool_size = 1000
    n_samples = 100

    # Create a pool
    pool = np.random.randn(pool_size, d)
    pool = pool / np.linalg.norm(pool, axis=1, keepdims=True)

    # Sample twice from the same pool
    indices_a = np.random.choice(pool_size, size=n_samples, replace=True)
    indices_b = np.random.choice(pool_size, size=n_samples, replace=True)

    embeddings_a = pool[indices_a]
    embeddings_b = pool[indices_b]

    rho_a, rho_b = compute_density_matrix_umap(
        embeddings_a, embeddings_b,
        n_components=n_components,
        n_neighbors=n_neighbors,
    )

    dist = trace_distance(rho_a, rho_b)

    print(f"Trace distance (UMAP): {dist:.6f}")
    print(f"Expected: Close to 0 (same underlying pool)")
    print(f"Status: {'✓ PASS' if dist < 0.1 else '✗ FAIL'}")
    print()

    return dist


def compare_hyperparameters():
    """Test how choice of k and n_neighbors affects results."""
    print("="*80)
    print("Test 6: Effect of UMAP Hyperparameters")
    print("="*80)

    np.random.seed(42)
    d = 768
    n_samples = 100

    # Generate identical distribution samples
    embeddings_a = np.random.randn(n_samples, d)
    embeddings_b = np.random.randn(n_samples, d)
    embeddings_a = embeddings_a / np.linalg.norm(embeddings_a, axis=1, keepdims=True)
    embeddings_b = embeddings_b / np.linalg.norm(embeddings_b, axis=1, keepdims=True)

    print("\nVarying k (n_neighbors=15):")
    print("k (components) | Trace Distance (same dist)")
    print("-" * 50)

    for k in [5, 10, 20, 50, 100]:
        rho_a, rho_b = compute_density_matrix_umap(
            embeddings_a, embeddings_b,
            n_components=k,
            n_neighbors=15,
        )
        dist = trace_distance(rho_a, rho_b)
        print(f"{k:14d} | {dist:.6f}")

    print("\nVarying n_neighbors (k=20):")
    print("n_neighbors | Trace Distance (same dist)")
    print("-" * 50)

    for nn in [5, 10, 15, 30, 50]:
        rho_a, rho_b = compute_density_matrix_umap(
            embeddings_a, embeddings_b,
            n_components=20,
            n_neighbors=nn,
        )
        dist = trace_distance(rho_a, rho_b)
        print(f"{nn:11d} | {dist:.6f}")

    print()


def main():
    """Run all validation tests with UMAP-based approach."""
    print("\n" + "="*80)
    print("TRACE DISTANCE WITH UMAP FIX - VALIDATION SUITE")
    print("="*80 + "\n")

    # Default hyperparameters
    n_components = 20
    n_neighbors = 15

    print(f"Testing with k={n_components}, n_neighbors={n_neighbors}\n")

    results = {}

    results['identical'] = test_identical_distributions(n_components, n_neighbors)
    results['orthogonal'] = test_orthogonal_distributions(n_components, n_neighbors)
    results['shifted'] = test_gaussian_shift(n_components, n_neighbors)
    results['sample_size'] = test_sample_size_sensitivity(n_components, n_neighbors)
    results['resampling'] = test_resampling_from_same_set(n_components, n_neighbors)

    compare_hyperparameters()

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"1. Identical distributions: {results['identical']:.3f} (expect <0.1)")
    print(f"2. Orthogonal distributions: {results['orthogonal']:.3f} (expect >0.3)")
    print(f"3. Shifted Gaussians: {results['shifted']:.3f} (expect 0.0-1.0)")
    print(f"4. Sample size trend: {results['sample_size'][0]:.3f} → {results['sample_size'][-1]:.3f}")
    print(f"5. Resampling same pool: {results['resampling']:.3f} (expect <0.1)")
    print()

    # Comparison with PCA results
    print("="*80)
    print("COMPARISON: UMAP vs PCA")
    print("="*80)
    print("Test                    | PCA k=50 | UMAP k=20")
    print("-" * 60)
    print(f"Identical distributions | 0.004    | {results['identical']:.3f}")
    print(f"Sample size (50→1000)   | 0.005→0.002 | {results['sample_size'][0]:.3f}→{results['sample_size'][-1]:.3f}")
    print(f"Resampling same pool    | 0.019    | {results['resampling']:.3f}")
    print()

    # Diagnosis
    print("="*80)
    print("DIAGNOSIS: Does UMAP Fix the Artifact?")
    print("="*80)

    identical_ok = results['identical'] < 0.15
    resampling_ok = results['resampling'] < 0.15
    sample_size_ok = results['sample_size'][-1] < 0.3 and results['sample_size'][-1] < results['sample_size'][0] * 2

    if identical_ok and sample_size_ok and resampling_ok:
        print("✓ SUCCESS: UMAP approach fixes the sample-size artifact!")
        print()
        print("  Next step: Test on real LLM data (fp32 vs fp32 sanity check)")
    elif identical_ok or resampling_ok:
        print("⚠️  PARTIAL SUCCESS: Some improvement but issues remain")
        print()
        if not sample_size_ok:
            print(f"  ✗ Sample size dependence still present")
        if not identical_ok:
            print(f"  ✗ Identical distributions distance still high: {results['identical']:.3f}")
        if not resampling_ok:
            print(f"  ✗ Resampling variance still high: {results['resampling']:.3f}")
    else:
        print("✗ FAILURE: UMAP does not fix the fundamental issue")
        print()
        print("  Non-linear projection doesn't help.")
        print("  Problem likely lies in embedding space itself, not dimensionality reduction method.")

    print("="*80)


if __name__ == "__main__":
    main()
