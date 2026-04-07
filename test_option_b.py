#!/usr/bin/env python3
"""
Test Option B: Generic two_sample_ks_statistic with feature functions
"""

import numpy as np
import torch
from functools import partial
from model_equality_testing.distribution import CompletionSample
from model_equality_testing.src.features import get_vader_scores, get_perplexity_scores
from model_equality_testing.src.tests import two_sample_ks_statistic
from model_equality_testing.algorithm import run_two_sample_test
from model_equality_testing.src.registry import IMPLEMENTED_TESTS
import tempfile
import os

def create_test_sample(n_samples=50, seed=42):
    """Create a simple test sample with unicode completions."""
    np.random.seed(seed)

    texts = [
        "The quick brown fox jumps over the lazy dog",
        "A journey of a thousand miles begins with a single step",
        "To be or not to be that is the question",
    ]

    sampled_texts = np.random.choice(texts, size=n_samples)
    max_len = max(len(t) for t in sampled_texts)
    completions = np.full((n_samples, max_len), -1, dtype=np.int32)

    for i, text in enumerate(sampled_texts):
        for j, char in enumerate(text):
            completions[i, j] = ord(char)

    prompts = np.random.randint(0, 3, size=n_samples)

    return CompletionSample(
        prompts=prompts,
        completions=completions,
        m=3
    )

def test_registry_has_generic_function():
    """Test that two_sample_ks_statistic is registered."""
    print("=" * 60)
    print("Test 1: Registry Check")
    print("=" * 60)

    assert "two_sample_ks_statistic" in IMPLEMENTED_TESTS, \
        "two_sample_ks_statistic not found in registry"
    print("✓ two_sample_ks_statistic is registered in IMPLEMENTED_TESTS")
    print()

def test_direct_usage_with_vader():
    """Test direct usage with VADER (default)."""
    print("=" * 60)
    print("Test 2: Direct Usage with VADER (Default)")
    print("=" * 60)

    sample1 = create_test_sample(n_samples=30, seed=42)
    sample2 = create_test_sample(n_samples=30, seed=43)

    # Use default feature_fn (VADER)
    stat = two_sample_ks_statistic(sample1, sample2)
    print(f"✓ K-S statistic with VADER: {stat:.6f}")
    assert 0 <= stat <= 1, "K-S statistic should be in [0, 1]"
    print()

def test_direct_usage_with_custom_feature():
    """Test with custom feature function."""
    print("=" * 60)
    print("Test 3: Direct Usage with Custom Feature (Text Length)")
    print("=" * 60)

    sample1 = create_test_sample(n_samples=30, seed=42)
    sample2 = create_test_sample(n_samples=30, seed=43)

    # Custom feature: text length
    def length_feature(sample):
        from model_equality_testing.src.features import decode_sample_to_strings
        texts = decode_sample_to_strings(sample)
        return np.array([len(text) for text in texts])

    stat = two_sample_ks_statistic(sample1, sample2, feature_fn=length_feature)
    print(f"✓ K-S statistic with text length: {stat:.6f}")
    assert 0 <= stat <= 1, "K-S statistic should be in [0, 1]"
    print()

def test_with_perplexity_via_partial():
    """Test with perplexity using functools.partial."""
    print("=" * 60)
    print("Test 4: Usage with Perplexity (via partial)")
    print("=" * 60)

    # Need to train a model first
    sample1 = create_test_sample(n_samples=30, seed=42)
    sample2 = create_test_sample(n_samples=30, seed=43)

    with tempfile.NamedTemporaryFile(suffix='.arpa', delete=False) as f:
        model_path = f.name

    try:
        # Train model
        from model_equality_testing.src.features import train_kenlm_from_sample
        print("Training KenLM model...")
        train_kenlm_from_sample(
            sample=sample1,
            output_path=model_path,
            order=3,
            warn_on_degenerate=False
        )
        print(f"✓ Model trained: {model_path}")

        # Create feature function with partial
        ppl_fn = partial(
            get_perplexity_scores,
            kenlm_model_path=model_path,
            granularity="word",
            warn_on_degenerate=False
        )

        # Use with two_sample_ks_statistic
        stat = two_sample_ks_statistic(sample1, sample2, feature_fn=ppl_fn)
        print(f"✓ K-S statistic with perplexity: {stat:.6f}")
        assert 0 <= stat <= 1, "K-S statistic should be in [0, 1]"

    finally:
        if os.path.exists(model_path):
            os.remove(model_path)
            print("✓ Cleaned up model file")

    print()

def test_via_run_two_sample_test():
    """Test via run_two_sample_test with feature_fn parameter."""
    print("=" * 60)
    print("Test 5: Via run_two_sample_test with feature_fn")
    print("=" * 60)

    sample1 = create_test_sample(n_samples=30, seed=42)
    sample2 = create_test_sample(n_samples=30, seed=43)

    with tempfile.NamedTemporaryFile(suffix='.arpa', delete=False) as f:
        model_path = f.name

    try:
        # Train model
        from model_equality_testing.src.features import train_kenlm_from_sample
        print("Training KenLM model...")
        train_kenlm_from_sample(
            sample=sample1,
            output_path=model_path,
            order=3,
            warn_on_degenerate=False
        )

        # Create feature function
        ppl_fn = partial(
            get_perplexity_scores,
            kenlm_model_path=model_path,
            warn_on_degenerate=False
        )

        # Use via run_two_sample_test
        print("Running test via run_two_sample_test...")
        pvalue, statistic = run_two_sample_test(
            sample1,
            sample2,
            stat_type="two_sample_ks_statistic",
            feature_fn=ppl_fn,
            pvalue_type="permutation_pvalue",
            b=50  # Small number for speed
        )

        print(f"✓ K-S statistic: {statistic:.6f}")
        print(f"✓ Permutation p-value: {pvalue:.6f}")
        assert 0 <= statistic <= 1, "Statistic should be in [0, 1]"
        assert 0 <= pvalue <= 1, "P-value should be in [0, 1]"

    finally:
        if os.path.exists(model_path):
            os.remove(model_path)
            print("✓ Cleaned up model file")

    print()

def test_compare_multiple_features():
    """Test comparing multiple feature extractors."""
    print("=" * 60)
    print("Test 6: Compare Multiple Features")
    print("=" * 60)

    sample1 = create_test_sample(n_samples=30, seed=42)
    sample2 = create_test_sample(n_samples=30, seed=43)

    with tempfile.NamedTemporaryFile(suffix='.arpa', delete=False) as f:
        model_path = f.name

    try:
        # Train model
        from model_equality_testing.src.features import train_kenlm_from_sample, decode_sample_to_strings
        train_kenlm_from_sample(
            sample=sample1,
            output_path=model_path,
            order=3,
            warn_on_degenerate=False
        )

        # Define multiple feature extractors
        features = {
            "VADER sentiment": get_vader_scores,
            "Perplexity (word)": partial(
                get_perplexity_scores,
                kenlm_model_path=model_path,
                granularity="word",
                warn_on_degenerate=False
            ),
            "Perplexity (char)": partial(
                get_perplexity_scores,
                kenlm_model_path=model_path,
                granularity="char",
                warn_on_degenerate=False
            ),
            "Text length": lambda s: np.array([len(t) for t in decode_sample_to_strings(s)]),
        }

        print("Comparing features:")
        print("-" * 50)
        for name, feature_fn in features.items():
            stat = two_sample_ks_statistic(sample1, sample2, feature_fn=feature_fn)
            print(f"  {name:20s}: K-S = {stat:.6f}")

        print("-" * 50)
        print("✓ All features computed successfully")

    finally:
        if os.path.exists(model_path):
            os.remove(model_path)

    print()

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("OPTION B TEST SUITE")
    print("Generic two_sample_ks_statistic with Feature Functions")
    print("=" * 60 + "\n")

    try:
        test_registry_has_generic_function()
        test_direct_usage_with_vader()
        test_direct_usage_with_custom_feature()
        test_with_perplexity_via_partial()
        test_via_run_two_sample_test()
        test_compare_multiple_features()

        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print("\nOption B is fully implemented!")
        print("\nKey capabilities:")
        print("  ✓ Generic two_sample_ks_statistic registered in IMPLEMENTED_TESTS")
        print("  ✓ Works with any feature extraction function")
        print("  ✓ Supports functools.partial for parameterized features")
        print("  ✓ Integrates with run_two_sample_test")
        print("  ✓ Easy to compare multiple feature extractors")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
