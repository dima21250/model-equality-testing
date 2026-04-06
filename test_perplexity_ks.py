#!/usr/bin/env python3
"""
Test script for perplexity-based K-S test implementation.
"""

import numpy as np
import torch
from model_equality_testing.distribution import CompletionSample
from model_equality_testing.src.features import train_kenlm_from_sample, get_perplexity_scores
from model_equality_testing.src.tests import two_sample_perplexity_ks
from model_equality_testing.algorithm import run_two_sample_test
import tempfile
import os

def create_test_sample(n_samples=50, seed=42):
    """Create a simple test sample with unicode completions."""
    np.random.seed(seed)

    # Create simple text completions
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "A journey of a thousand miles begins with a single step",
        "To be or not to be that is the question",
        "All that glitters is not gold",
        "Actions speak louder than words",
    ]

    # Sample texts and convert to unicode codepoints
    sampled_texts = np.random.choice(texts, size=n_samples)

    # Convert to unicode codepoints (matching MET's unicode representation)
    max_len = max(len(t) for t in sampled_texts)
    completions = np.full((n_samples, max_len), -1, dtype=np.int32)

    for i, text in enumerate(sampled_texts):
        for j, char in enumerate(text):
            completions[i, j] = ord(char)

    # Random prompt indices (3 prompts)
    prompts = np.random.randint(0, 3, size=n_samples)

    return CompletionSample(
        prompts=prompts,
        completions=completions,
        m=3
    )

def test_basic_functionality():
    """Test basic training and scoring functionality."""
    print("=" * 60)
    print("Test 1: Basic KenLM Training and Scoring")
    print("=" * 60)

    # Create test sample
    print("Creating test sample...")
    sample = create_test_sample(n_samples=50)
    print(f"✓ Created sample with {sample.N} completions")

    # Train KenLM model
    with tempfile.NamedTemporaryFile(suffix='.arpa', delete=False) as f:
        model_path = f.name

    try:
        print(f"\nTraining KenLM model (3-gram)...")
        train_kenlm_from_sample(
            sample=sample,
            output_path=model_path,
            order=3,
            warn_on_degenerate=False
        )
        print(f"✓ Model trained successfully: {model_path}")

        # Verify model file exists
        assert os.path.exists(model_path), "Model file not created"
        print(f"✓ Model file exists ({os.path.getsize(model_path)} bytes)")

        # Get perplexity scores
        print("\nComputing perplexity scores...")
        scores = get_perplexity_scores(
            sample=sample,
            kenlm_model_path=model_path,
            warn_on_degenerate=False
        )
        print(f"✓ Computed {len(scores)} perplexity scores")
        print(f"  Score statistics: min={scores.min():.2f}, max={scores.max():.2f}, mean={scores.mean():.2f}")

        # Verify scores
        assert len(scores) == sample.N, "Wrong number of scores"
        assert np.all(scores > 0), "All scores should be positive"
        assert np.all(np.isfinite(scores)), "All scores should be finite"
        print("✓ All scores are valid (positive and finite)")

    finally:
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"\n✓ Cleaned up temporary model file")

    print("\n✅ Test 1 PASSED\n")

def test_ks_statistic():
    """Test the two-sample K-S test."""
    print("=" * 60)
    print("Test 2: Two-Sample K-S Test")
    print("=" * 60)

    # Create two samples (should be similar since from same distribution)
    print("Creating two samples from same distribution...")
    sample1 = create_test_sample(n_samples=50, seed=42)
    sample2 = create_test_sample(n_samples=50, seed=43)
    print(f"✓ Sample 1: {sample1.N} completions")
    print(f"✓ Sample 2: {sample2.N} completions")

    # Train model on combined data
    print("\nTraining KenLM model on combined data...")
    combined_prompts = torch.cat([sample1.prompt_sample, sample2.prompt_sample])
    combined_completions = torch.cat([sample1.completion_sample, sample2.completion_sample])
    combined_sample = CompletionSample(
        prompts=combined_prompts,
        completions=combined_completions,
        m=3
    )

    with tempfile.NamedTemporaryFile(suffix='.arpa', delete=False) as f:
        model_path = f.name

    try:
        train_kenlm_from_sample(
            sample=combined_sample,
            output_path=model_path,
            order=3,
            warn_on_degenerate=False
        )
        print(f"✓ Model trained on {combined_sample.N} completions")

        # Compute K-S statistic directly
        print("\nComputing K-S statistic...")
        statistic = two_sample_perplexity_ks(
            sample1=sample1,
            sample2=sample2,
            kenlm_model_path=model_path,
            warn_on_degenerate=False
        )
        print(f"✓ K-S statistic: {statistic:.6f}")

        # Statistic should be between 0 and 1
        assert 0 <= statistic <= 1, f"K-S statistic should be in [0,1], got {statistic}"
        print("✓ K-S statistic in valid range [0, 1]")

        # Since samples are from same distribution, statistic should be relatively small
        # (though not necessarily < 0.05 without permutation test)
        print(f"✓ Samples from same distribution yield K-S = {statistic:.6f}")

    finally:
        if os.path.exists(model_path):
            os.remove(model_path)
            print("\n✓ Cleaned up temporary model file")

    print("\n✅ Test 2 PASSED\n")

def test_registry_integration():
    """Test that the function is registered and callable via run_two_sample_test."""
    print("=" * 60)
    print("Test 3: Registry Integration")
    print("=" * 60)

    from model_equality_testing.src.registry import IMPLEMENTED_TESTS

    # Check registration
    print("Checking test registry...")
    assert "two_sample_perplexity_ks" in IMPLEMENTED_TESTS, "Test not registered"
    print("✓ two_sample_perplexity_ks found in IMPLEMENTED_TESTS")

    # Create samples
    sample1 = create_test_sample(n_samples=30, seed=42)
    sample2 = create_test_sample(n_samples=30, seed=43)

    # Train model
    with tempfile.NamedTemporaryFile(suffix='.arpa', delete=False) as f:
        model_path = f.name

    try:
        train_kenlm_from_sample(
            sample=sample1,
            output_path=model_path,
            order=3,
            warn_on_degenerate=False
        )

        # Test via run_two_sample_test
        print("\nCalling via run_two_sample_test...")
        pvalue, statistic = run_two_sample_test(
            sample1,
            sample2,
            stat_type="two_sample_perplexity_ks",
            pvalue_type="permutation_pvalue",
            kenlm_model_path=model_path,
            warn_on_degenerate=False,
            b=50  # Small number of permutations for speed
        )
        print(f"✓ K-S statistic: {statistic:.6f}")
        print(f"✓ Permutation p-value: {pvalue:.6f}")

        # Verify outputs
        assert isinstance(statistic, (int, float)), "Statistic should be numeric"
        assert isinstance(pvalue, (int, float)), "P-value should be numeric"
        assert 0 <= pvalue <= 1, "P-value should be in [0, 1]"
        print("✓ Output values are valid")

    finally:
        if os.path.exists(model_path):
            os.remove(model_path)
            print("\n✓ Cleaned up temporary model file")

    print("\n✅ Test 3 PASSED\n")

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PERPLEXITY-BASED K-S TEST - VERIFICATION")
    print("=" * 60 + "\n")

    try:
        test_basic_functionality()
        test_ks_statistic()
        test_registry_integration()

        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print("\nThe perplexity-based K-S test implementation is working correctly!")
        print("\nNext steps:")
        print("  1. Test with real MET dataset samples")
        print("  2. Compare performance against VADER K-S and MMD tests")
        print("  3. Experiment with different n-gram orders and granularities")
        print("  4. Test sentencepiece preprocessing (if model available)")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
