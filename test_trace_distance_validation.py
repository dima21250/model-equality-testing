"""Test trace distance implementation on synthetic data with known ground truth.

This script validates the quantum_metrics.py implementation by testing on:
1. Identical distributions (should give trace distance ≈ 0)
2. Orthogonal distributions (should give trace distance ≈ 1)
3. Partially overlapping distributions (should give 0 < trace distance < 1)

The goal is to determine if the high trace distances in fp32 vs fp32 comparisons
are due to implementation bugs or fundamental limitations.
"""

import numpy as np
from model_equality_testing.src.quantum_metrics import (
    compute_pip_matrix,
    normalize_to_density_matrix,
    trace_distance,
)


def test_identical_distributions():
    """Test 1: Two samples from the same Gaussian should give low trace distance."""
    print("="*80)
    print("Test 1: Identical Distributions (Same Gaussian)")
    print("="*80)

    np.random.seed(42)
    d = 768  # MPNet embedding dimension
    n_samples = 100

    # Generate two independent samples from N(0, I)
    embeddings_a = np.random.randn(n_samples, d)
    embeddings_b = np.random.randn(n_samples, d)

    # Normalize to unit vectors (like SBERT does)
    embeddings_a = embeddings_a / np.linalg.norm(embeddings_a, axis=1, keepdims=True)
    embeddings_b = embeddings_b / np.linalg.norm(embeddings_b, axis=1, keepdims=True)

    # Compute density matrices
    pip_a = compute_pip_matrix(embeddings_a)
    pip_b = compute_pip_matrix(embeddings_b)

    rho_a = normalize_to_density_matrix(pip_a)
    rho_b = normalize_to_density_matrix(pip_b)

    # Compute trace distance
    dist = trace_distance(rho_a, rho_b)

    print(f"Trace distance between two samples from N(0, I): {dist:.6f}")
    print(f"Expected: Close to 0 (since same distribution)")
    print(f"Status: {'✓ PASS' if dist < 0.1 else '✗ FAIL'}")
    print()

    return dist


def test_orthogonal_distributions():
    """Test 2: Samples from orthogonal subspaces should give trace distance ≈ 1."""
    print("="*80)
    print("Test 2: Orthogonal Distributions")
    print("="*80)

    np.random.seed(42)
    d = 768
    n_samples = 100

    # Create orthogonal subspaces
    # Sample A: random vectors in first d/2 dimensions
    embeddings_a = np.zeros((n_samples, d))
    embeddings_a[:, :d//2] = np.random.randn(n_samples, d//2)

    # Sample B: random vectors in second d/2 dimensions
    embeddings_b = np.zeros((n_samples, d))
    embeddings_b[:, d//2:] = np.random.randn(n_samples, d//2)

    # Normalize
    embeddings_a = embeddings_a / np.linalg.norm(embeddings_a, axis=1, keepdims=True)
    embeddings_b = embeddings_b / np.linalg.norm(embeddings_b, axis=1, keepdims=True)

    # Compute density matrices
    pip_a = compute_pip_matrix(embeddings_a)
    pip_b = compute_pip_matrix(embeddings_b)

    rho_a = normalize_to_density_matrix(pip_a)
    rho_b = normalize_to_density_matrix(pip_b)

    # Compute trace distance
    dist = trace_distance(rho_a, rho_b)

    print(f"Trace distance between orthogonal subspaces: {dist:.6f}")
    print(f"Expected: Close to 1 (maximum distinguishability)")
    print(f"Status: {'✓ PASS' if dist > 0.9 else '✗ FAIL'}")
    print()

    return dist


def test_gaussian_shift():
    """Test 3: Two Gaussians with different means - partial overlap."""
    print("="*80)
    print("Test 3: Shifted Gaussians (Partial Overlap)")
    print("="*80)

    np.random.seed(42)
    d = 768
    n_samples = 100

    # Sample A: N(0, I)
    embeddings_a = np.random.randn(n_samples, d)

    # Sample B: N(μ, I) where μ creates some separation
    shift = np.zeros(d)
    shift[0] = 2.0  # Shift along first dimension
    embeddings_b = np.random.randn(n_samples, d) + shift

    # Normalize
    embeddings_a = embeddings_a / np.linalg.norm(embeddings_a, axis=1, keepdims=True)
    embeddings_b = embeddings_b / np.linalg.norm(embeddings_b, axis=1, keepdims=True)

    # Compute density matrices
    pip_a = compute_pip_matrix(embeddings_a)
    pip_b = compute_pip_matrix(embeddings_b)

    rho_a = normalize_to_density_matrix(pip_a)
    rho_b = normalize_to_density_matrix(pip_b)

    # Compute trace distance
    dist = trace_distance(rho_a, rho_b)

    print(f"Trace distance between N(0, I) and N(shift, I): {dist:.6f}")
    print(f"Expected: Between 0 and 1 (moderate separation)")
    print(f"Status: {'✓ PASS' if 0.1 < dist < 0.9 else '✗ FAIL'}")
    print()

    return dist


def test_sample_size_sensitivity():
    """Test 4: How does trace distance change with sample size for identical distributions?"""
    print("="*80)
    print("Test 4: Sample Size Sensitivity (Identical Distributions)")
    print("="*80)

    np.random.seed(42)
    d = 768
    sample_sizes = [50, 100, 200, 500, 1000]

    print("n_samples | Trace Distance (same dist)")
    print("-" * 40)

    distances = []
    for n in sample_sizes:
        embeddings_a = np.random.randn(n, d)
        embeddings_b = np.random.randn(n, d)

        embeddings_a = embeddings_a / np.linalg.norm(embeddings_a, axis=1, keepdims=True)
        embeddings_b = embeddings_b / np.linalg.norm(embeddings_b, axis=1, keepdims=True)

        pip_a = compute_pip_matrix(embeddings_a)
        pip_b = compute_pip_matrix(embeddings_b)

        rho_a = normalize_to_density_matrix(pip_a)
        rho_b = normalize_to_density_matrix(pip_b)

        dist = trace_distance(rho_a, rho_b)
        distances.append(dist)

        print(f"{n:9d} | {dist:.6f}")

    print()
    print(f"Observation: Distance should DECREASE with more samples (better approximation)")
    print(f"Trend: {distances[0]:.3f} → {distances[-1]:.3f}")
    print(f"Status: {'✓ PASS' if distances[-1] < distances[0] else '✗ FAIL - distance increases!'}")
    print()

    return distances


def test_resampling_from_same_set():
    """Test 5: Resample from the EXACT same pool - should give very low distance."""
    print("="*80)
    print("Test 5: Resampling from Same Pool (Bootstrap)")
    print("="*80)

    np.random.seed(42)
    d = 768
    pool_size = 1000
    n_samples = 100

    # Create a pool of embeddings
    pool = np.random.randn(pool_size, d)
    pool = pool / np.linalg.norm(pool, axis=1, keepdims=True)

    # Sample twice from the same pool (with replacement)
    indices_a = np.random.choice(pool_size, size=n_samples, replace=True)
    indices_b = np.random.choice(pool_size, size=n_samples, replace=True)

    embeddings_a = pool[indices_a]
    embeddings_b = pool[indices_b]

    pip_a = compute_pip_matrix(embeddings_a)
    pip_b = compute_pip_matrix(embeddings_b)

    rho_a = normalize_to_density_matrix(pip_a)
    rho_b = normalize_to_density_matrix(pip_b)

    dist = trace_distance(rho_a, rho_b)

    print(f"Trace distance (resampling from same pool): {dist:.6f}")
    print(f"Expected: Should be VERY close to 0 (same underlying pool)")
    print(f"Status: {'✓ PASS' if dist < 0.05 else '✗ FAIL - high variance!'}")
    print()

    return dist


def main():
    """Run all validation tests."""
    print("\n" + "="*80)
    print("TRACE DISTANCE IMPLEMENTATION VALIDATION")
    print("="*80 + "\n")

    results = {}

    results['identical'] = test_identical_distributions()
    results['orthogonal'] = test_orthogonal_distributions()
    results['shifted'] = test_gaussian_shift()
    results['sample_size'] = test_sample_size_sensitivity()
    results['resampling'] = test_resampling_from_same_set()

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"1. Identical distributions: {results['identical']:.3f} (expect ~0)")
    print(f"2. Orthogonal distributions: {results['orthogonal']:.3f} (expect ~1)")
    print(f"3. Shifted Gaussians: {results['shifted']:.3f} (expect 0.1-0.9)")
    print(f"5. Resampling same pool: {results['resampling']:.3f} (expect <0.05)")
    print()

    # Diagnosis
    print("="*80)
    print("DIAGNOSIS")
    print("="*80)

    if results['identical'] > 0.3:
        print("⚠️  ISSUE DETECTED: Trace distance is HIGH even for identical distributions!")
        print("    → This explains the fp32 vs fp32 sanity check failure")
        print("    → Root cause: Finite sample artifact in high-dimensional space")
        print()
        print("    Likely causes:")
        print("    - Curse of dimensionality: N=100 samples in d=768 dimensions")
        print("    - PIP matrix structure depends on specific sample, not just distribution")
        print("    - Density matrix normalization doesn't account for sample size")
    elif results['resampling'] > 0.2:
        print("⚠️  ISSUE DETECTED: High variance when resampling from same pool!")
        print("    → Trace distance is sensitive to sampling noise")
    else:
        print("✓ Implementation appears mathematically correct")
        print("  → Issue may be specific to SBERT embeddings or LLM completions")

    print("="*80)


if __name__ == "__main__":
    main()
