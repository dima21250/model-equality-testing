"""test_quantum_basic.py

Basic unit tests for quantum metrics implementation.
Can be run without the full dataset using synthetic data.

Usage:
    python test_quantum_basic.py
"""

import numpy as np
import sys


def test_pip_matrix():
    """Test PIP matrix computation."""
    print("Testing PIP matrix computation...")

    from model_equality_testing.src.quantum_metrics import compute_pip_matrix

    # Create random embeddings
    embeddings = np.random.randn(10, 50)
    pip = compute_pip_matrix(embeddings)

    # Check shape
    assert pip.shape == (10, 10), f"Expected shape (10, 10), got {pip.shape}"

    # Check symmetry
    assert np.allclose(pip, pip.T), "PIP matrix should be symmetric"

    # Check diagonal (should be squared norms)
    expected_diag = np.sum(embeddings ** 2, axis=1)
    assert np.allclose(np.diag(pip), expected_diag), "Diagonal should be squared norms"

    print("  ✅ PIP matrix test passed\n")


def test_density_matrix():
    """Test density matrix normalization."""
    print("Testing density matrix normalization...")

    from model_equality_testing.src.quantum_metrics import (
        compute_pip_matrix,
        normalize_to_density_matrix,
    )

    # Create random embeddings
    embeddings = np.random.randn(10, 50)
    pip = compute_pip_matrix(embeddings)
    rho = normalize_to_density_matrix(pip)

    # Check trace is 1
    assert np.isclose(np.trace(rho), 1.0), f"Trace should be 1, got {np.trace(rho)}"

    # Check symmetry preserved
    assert np.allclose(rho, rho.T), "Density matrix should be symmetric"

    # Check positive semidefinite (all eigenvalues >= 0)
    eigenvalues = np.linalg.eigvalsh(rho)
    assert np.all(eigenvalues >= -1e-10), f"Eigenvalues should be non-negative, got min={eigenvalues.min()}"

    print("  ✅ Density matrix test passed\n")


def test_trace_distance():
    """Test trace distance computation."""
    print("Testing trace distance...")

    from model_equality_testing.src.quantum_metrics import (
        compute_pip_matrix,
        normalize_to_density_matrix,
        trace_distance,
    )

    # Test 1: Identical states should have distance 0
    embeddings = np.random.randn(10, 50)
    pip = compute_pip_matrix(embeddings)
    rho = normalize_to_density_matrix(pip)

    dist = trace_distance(rho, rho)
    assert np.isclose(dist, 0.0, atol=1e-8), f"Trace distance(ρ, ρ) should be 0, got {dist}"

    # Test 2: Orthogonal states should have distance 1
    rho_a = np.diag([1.0, 0.0])
    rho_b = np.diag([0.0, 1.0])
    dist = trace_distance(rho_a, rho_b)
    assert np.isclose(dist, 1.0, atol=1e-8), f"Orthogonal states should have distance 1, got {dist}"

    # Test 3: Distance should be in [0, 1]
    embeddings1 = np.random.randn(10, 50)
    embeddings2 = np.random.randn(10, 50)
    pip1 = compute_pip_matrix(embeddings1)
    pip2 = compute_pip_matrix(embeddings2)
    rho1 = normalize_to_density_matrix(pip1)
    rho2 = normalize_to_density_matrix(pip2)

    dist = trace_distance(rho1, rho2)
    assert 0 <= dist <= 1, f"Trace distance should be in [0, 1], got {dist}"

    print("  ✅ Trace distance test passed\n")


def test_von_neumann_entropy():
    """Test von Neumann entropy computation."""
    print("Testing von Neumann entropy...")

    from model_equality_testing.src.quantum_metrics import von_neumann_entropy

    # Test 1: Pure state (rank 1) has zero entropy
    rho_pure = np.array([[1.0, 0.0], [0.0, 0.0]])
    entropy = von_neumann_entropy(rho_pure)
    assert np.isclose(entropy, 0.0, atol=1e-8), f"Pure state should have entropy 0, got {entropy}"

    # Test 2: Maximally mixed state has entropy log(N)
    N = 4
    rho_mixed = np.eye(N) / N
    entropy = von_neumann_entropy(rho_mixed)
    expected = np.log(N)
    assert np.isclose(entropy, expected, atol=1e-8), f"Maximally mixed state should have entropy log({N})={expected}, got {entropy}"

    # Test 3: Entropy should be non-negative
    embeddings = np.random.randn(10, 50)
    from model_equality_testing.src.quantum_metrics import (
        compute_pip_matrix,
        normalize_to_density_matrix,
    )
    pip = compute_pip_matrix(embeddings)
    rho = normalize_to_density_matrix(pip)
    entropy = von_neumann_entropy(rho)
    assert entropy >= 0, f"Entropy should be non-negative, got {entropy}"

    print("  ✅ Von Neumann entropy test passed\n")


def test_embedding_generation():
    """Test embedding generation from sample data."""
    print("Testing embedding generation...")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  ⚠️  Skipping (sentence-transformers not installed)\n")
        return

    from model_equality_testing.src.embeddings import EmbeddingModel

    # Create embedding model
    embedder = EmbeddingModel("all-MiniLM-L6-v2", model_type="sbert")

    # Test encoding
    texts = ["Hello world", "Goodbye world", "Test sentence"]
    embeddings = embedder.encode(texts)

    # Check shape
    assert embeddings.shape[0] == 3, f"Expected 3 embeddings, got {embeddings.shape[0]}"
    assert embeddings.shape[1] == embedder.embedding_dim, \
        f"Expected dim {embedder.embedding_dim}, got {embeddings.shape[1]}"

    # Check embeddings are non-zero
    assert not np.allclose(embeddings, 0), "Embeddings should not be all zeros"

    # Check different texts produce different embeddings
    assert not np.allclose(embeddings[0], embeddings[1]), "Different texts should have different embeddings"

    print(f"  ✅ Embedding generation test passed (dim={embedder.embedding_dim})\n")


def test_quantum_test_functions():
    """Test quantum test functions with synthetic CompletionSample."""
    print("Testing quantum test functions...")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  ⚠️  Skipping (sentence-transformers not installed)\n")
        return

    import torch
    from model_equality_testing.distribution import CompletionSample
    from model_equality_testing.tests import (
        quantum_trace_distance,
        quantum_von_neumann_divergence,
    )

    # Create synthetic samples (unicode codepoints for short texts)
    def text_to_unicode(text):
        return [ord(c) for c in text]

    texts1 = ["Hello world", "Test sample", "Another text"]
    texts2 = ["Different text", "Other sample", "Third example"]

    # Convert to unicode arrays
    completions1 = np.array([text_to_unicode(t) + [-1] * (20 - len(t)) for t in texts1])
    completions2 = np.array([text_to_unicode(t) + [-1] * (20 - len(t)) for t in texts2])

    prompts1 = np.array([0, 0, 0])
    prompts2 = np.array([0, 0, 0])

    sample1 = CompletionSample(prompts=prompts1, completions=completions1, m=1)
    sample2 = CompletionSample(prompts=prompts2, completions=completions2, m=1)

    # Test trace distance
    dist = quantum_trace_distance(sample1, sample2, embedding_model="all-MiniLM-L6-v2")
    assert 0 <= dist <= 1, f"Trace distance should be in [0, 1], got {dist}"
    print(f"  - Trace distance: {dist:.6f}")

    # Test von Neumann divergence
    div = quantum_von_neumann_divergence(sample1, sample2, embedding_model="all-MiniLM-L6-v2")
    assert div >= 0, f"Von Neumann divergence should be non-negative, got {div}"
    print(f"  - Von Neumann divergence: {div:.6f}")

    # Test same sample should give distance close to 0
    dist_same = quantum_trace_distance(sample1, sample1, embedding_model="all-MiniLM-L6-v2")
    assert dist_same < 0.01, f"Same sample should have distance ~0, got {dist_same}"
    print(f"  - Trace distance (same sample): {dist_same:.6f}")

    print("  ✅ Quantum test functions passed\n")


def main():
    print("\n" + "="*80)
    print("QUANTUM METRICS BASIC TESTS")
    print("="*80 + "\n")

    tests = [
        test_pip_matrix,
        test_density_matrix,
        test_trace_distance,
        test_von_neumann_entropy,
        test_embedding_generation,
        test_quantum_test_functions,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ❌ Test failed: {e}\n")
            failed += 1
            import traceback
            traceback.print_exc()

    print("="*80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*80 + "\n")

    if failed > 0:
        print("Some tests failed. Please check the implementation.")
        sys.exit(1)
    else:
        print("All tests passed! Quantum metrics implementation is working correctly.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -e '.[quantum]'")
        print("2. Run validation: python validate_quantum.py --samples 50")
        print("3. Run comparison: python quantum_comparison.py --samples 100")
        sys.exit(0)


if __name__ == "__main__":
    main()
