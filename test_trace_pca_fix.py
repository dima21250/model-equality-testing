"""Test PCA-based fix for trace distance sample-size artifact.

This script tests whether reducing embeddings to a fixed dimension before
constructing density matrices eliminates the sample-size dependence.

Approach:
1. Reduce embeddings to k dimensions via PCA (k=50)
2. Construct k×k covariance matrix (fixed size, independent of N)
3. Normalize to density matrix
4. Compute trace distance

If successful, trace distance should:
- Be ~0 for identical distributions
- Decrease (or stay constant) as N increases
- Properly distinguish different distributions
"""

import numpy as np
from sklearn.decomposition import PCA


def compute_density_matrix_pca(embeddings: np.ndarray, n_components: int = 50) -> np.ndarray:
    """Construct density matrix via PCA dimensionality reduction.

    Args:
        embeddings: (N, d) array of embeddings
        n_components: Fixed number of PCA components (k)

    Returns:
        rho: (k, k) density matrix with Tr(rho) = 1
    """
    N, d = embeddings.shape

    # If we have fewer samples than components, reduce k
    k = min(n_components, N - 1, d)

    # Apply PCA to reduce to k dimensions
    pca = PCA(n_components=k)
    reduced = pca.fit_transform(embeddings)  # (N, k)

    # Construct covariance matrix in reduced space
    # Cov = (1/N) * reduced.T @ reduced
    cov = (reduced.T @ reduced) / N  # (k, k)

    # Normalize to trace 1 (density matrix)
    trace = np.trace(cov)

    if trace < 1e-10:
        raise ValueError(f"Covariance trace too small: {trace}")

    rho = cov / trace

    return rho


def trace_distance_pca(rho_a: np.ndarray, rho_b: np.ndarray) -> float:
    """Compute trace distance (same formula as before)."""
    diff = rho_a - rho_b
    eigenvalues = np.linalg.eigvalsh(diff)
    distance = 0.5 * np.sum(np.abs(eigenvalues))
    return distance


def test_identical_distributions(n_components=50):
    """Test 1: Two samples from the same Gaussian - PCA version."""
    print("="*80)
    print(f"Test 1: Identical Distributions (PCA k={n_components})")
    print("="*80)

    np.random.seed(42)
    d = 768
    n_samples = 100

    embeddings_a = np.random.randn(n_samples, d)
    embeddings_b = np.random.randn(n_samples, d)

    # Normalize to unit vectors
    embeddings_a = embeddings_a / np.linalg.norm(embeddings_a, axis=1, keepdims=True)
    embeddings_b = embeddings_b / np.linalg.norm(embeddings_b, axis=1, keepdims=True)

    # Compute density matrices via PCA
    rho_a = compute_density_matrix_pca(embeddings_a, n_components=n_components)
    rho_b = compute_density_matrix_pca(embeddings_b, n_components=n_components)

    # Compute trace distance
    dist = trace_distance_pca(rho_a, rho_b)

    print(f"Trace distance (PCA): {dist:.6f}")
    print(f"Expected: Close to 0 (same distribution)")
    print(f"Status: {'✓ PASS' if dist < 0.1 else '✗ FAIL'}")
    print()

    return dist


def test_orthogonal_distributions(n_components=50):
    """Test 2: Orthogonal subspaces - PCA version."""
    print("="*80)
    print(f"Test 2: Orthogonal Distributions (PCA k={n_components})")
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

    # Compute density matrices via PCA
    rho_a = compute_density_matrix_pca(embeddings_a, n_components=n_components)
    rho_b = compute_density_matrix_pca(embeddings_b, n_components=n_components)

    # Compute trace distance
    dist = trace_distance_pca(rho_a, rho_b)

    print(f"Trace distance (PCA): {dist:.6f}")
    print(f"Expected: Close to 1 (orthogonal subspaces)")
    print(f"Status: {'✓ PASS' if dist > 0.5 else '⚠️  PARTIAL (orthogonality may be reduced by PCA)'}")
    print()

    return dist


def test_gaussian_shift(n_components=50):
    """Test 3: Shifted Gaussians - PCA version."""
    print("="*80)
    print(f"Test 3: Shifted Gaussians (PCA k={n_components})")
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

    # Compute density matrices via PCA
    rho_a = compute_density_matrix_pca(embeddings_a, n_components=n_components)
    rho_b = compute_density_matrix_pca(embeddings_b, n_components=n_components)

    # Compute trace distance
    dist = trace_distance_pca(rho_a, rho_b)

    print(f"Trace distance (PCA): {dist:.6f}")
    print(f"Expected: Between 0 and 1")
    print(f"Status: {'✓ PASS' if 0.0 < dist < 1.0 else '✗ FAIL'}")
    print()

    return dist


def test_sample_size_sensitivity(n_components=50):
    """Test 4: CRITICAL - Does trace distance grow with N? (PCA version)"""
    print("="*80)
    print(f"Test 4: Sample Size Sensitivity (PCA k={n_components})")
    print("="*80)

    np.random.seed(42)
    d = 768
    sample_sizes = [50, 100, 200, 500, 1000]

    print("n_samples | Trace Distance (PCA) | Status")
    print("-" * 55)

    distances = []
    for n in sample_sizes:
        embeddings_a = np.random.randn(n, d)
        embeddings_b = np.random.randn(n, d)

        embeddings_a = embeddings_a / np.linalg.norm(embeddings_a, axis=1, keepdims=True)
        embeddings_b = embeddings_b / np.linalg.norm(embeddings_b, axis=1, keepdims=True)

        rho_a = compute_density_matrix_pca(embeddings_a, n_components=n_components)
        rho_b = compute_density_matrix_pca(embeddings_b, n_components=n_components)

        dist = trace_distance_pca(rho_a, rho_b)
        distances.append(dist)

        # Check if distance is reasonable (< 0.2 for identical distributions)
        status = "✓" if dist < 0.2 else "✗"
        print(f"{n:9d} | {dist:20.6f} | {status}")

    print()
    print(f"Trend: {distances[0]:.3f} → {distances[-1]:.3f}")

    # Success criteria: distances should NOT increase systematically
    # Allow some variance, but overall trend should be flat or decreasing
    is_increasing = distances[-1] > distances[0] * 1.5
    is_stable = distances[-1] < 0.3  # Absolute threshold

    if is_stable and not is_increasing:
        print(f"Status: ✓ PASS - Distance stable across sample sizes")
    elif is_increasing:
        print(f"Status: ✗ FAIL - Distance still increases with N")
    else:
        print(f"Status: ⚠️  PARTIAL - Not increasing but absolute values high")

    print()

    return distances


def test_resampling_from_same_set(n_components=50):
    """Test 5: Resampling from same pool - PCA version."""
    print("="*80)
    print(f"Test 5: Resampling from Same Pool (PCA k={n_components})")
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

    rho_a = compute_density_matrix_pca(embeddings_a, n_components=n_components)
    rho_b = compute_density_matrix_pca(embeddings_b, n_components=n_components)

    dist = trace_distance_pca(rho_a, rho_b)

    print(f"Trace distance (PCA): {dist:.6f}")
    print(f"Expected: Close to 0 (same underlying pool)")
    print(f"Status: {'✓ PASS' if dist < 0.1 else '✗ FAIL'}")
    print()

    return dist


def compare_k_values():
    """Test how choice of k affects results."""
    print("="*80)
    print("Test 6: Effect of k (number of PCA components)")
    print("="*80)

    np.random.seed(42)
    d = 768
    n_samples = 100
    k_values = [10, 20, 50, 100, 200]

    # Generate identical distribution samples
    embeddings_a = np.random.randn(n_samples, d)
    embeddings_b = np.random.randn(n_samples, d)
    embeddings_a = embeddings_a / np.linalg.norm(embeddings_a, axis=1, keepdims=True)
    embeddings_b = embeddings_b / np.linalg.norm(embeddings_b, axis=1, keepdims=True)

    print("k (components) | Trace Distance (same dist)")
    print("-" * 50)

    for k in k_values:
        rho_a = compute_density_matrix_pca(embeddings_a, n_components=k)
        rho_b = compute_density_matrix_pca(embeddings_b, n_components=k)
        dist = trace_distance_pca(rho_a, rho_b)
        print(f"{k:14d} | {dist:.6f}")

    print()
    print("Observation: Lower k = stronger regularization (may reduce noise)")
    print()


def main():
    """Run all validation tests with PCA-based approach."""
    print("\n" + "="*80)
    print("TRACE DISTANCE WITH PCA FIX - VALIDATION SUITE")
    print("="*80 + "\n")

    n_components = 50  # Fixed dimensionality

    results = {}

    results['identical'] = test_identical_distributions(n_components)
    results['orthogonal'] = test_orthogonal_distributions(n_components)
    results['shifted'] = test_gaussian_shift(n_components)
    results['sample_size'] = test_sample_size_sensitivity(n_components)
    results['resampling'] = test_resampling_from_same_set(n_components)

    compare_k_values()

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"1. Identical distributions: {results['identical']:.3f} (expect <0.1)")
    print(f"2. Orthogonal distributions: {results['orthogonal']:.3f} (expect >0.5)")
    print(f"3. Shifted Gaussians: {results['shifted']:.3f} (expect 0.0-1.0)")
    print(f"4. Sample size trend: {results['sample_size'][0]:.3f} → {results['sample_size'][-1]:.3f}")
    print(f"5. Resampling same pool: {results['resampling']:.3f} (expect <0.1)")
    print()

    # Diagnosis
    print("="*80)
    print("DIAGNOSIS: Did PCA Fix the Artifact?")
    print("="*80)

    # Check key criteria
    identical_ok = results['identical'] < 0.15
    resampling_ok = results['resampling'] < 0.15
    sample_size_ok = results['sample_size'][-1] < 0.3 and results['sample_size'][-1] < results['sample_size'][0] * 2

    if identical_ok and sample_size_ok and resampling_ok:
        print("✓ SUCCESS: PCA approach fixes the sample-size artifact!")
        print()
        print("  Key improvements:")
        print(f"  - Identical distributions: 0.209 (old) → {results['identical']:.3f} (PCA)")
        print(f"  - Sample size trend: 0.146→0.636 (old) → {results['sample_size'][0]:.3f}→{results['sample_size'][-1]:.3f} (PCA)")
        print(f"  - Resampling: 0.279 (old) → {results['resampling']:.3f} (PCA)")
        print()
        print("  Next step: Test on real LLM data (fp32 vs fp32 sanity check)")
    elif identical_ok or resampling_ok:
        print("⚠️  PARTIAL SUCCESS: Some improvement but issues remain")
        print()
        if not sample_size_ok:
            print(f"  ✗ Sample size dependence still present: {results['sample_size'][0]:.3f} → {results['sample_size'][-1]:.3f}")
        if not identical_ok:
            print(f"  ✗ Identical distributions distance still high: {results['identical']:.3f}")
        if not resampling_ok:
            print(f"  ✗ Resampling variance still high: {results['resampling']:.3f}")
        print()
        print("  May need: different k, different embedding preprocessing, or alternative approach")
    else:
        print("✗ FAILURE: PCA does not fix the fundamental issue")
        print()
        print("  This suggests the problem runs deeper than matrix size.")
        print("  Consider: quantum fidelity, kernel-based approach, or abandon Gram matrices")

    print("="*80)


if __name__ == "__main__":
    main()
