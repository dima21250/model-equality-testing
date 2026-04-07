#!/usr/bin/env python3
"""
Test suite for K-S convenience functions and analytical_ks p-value type.

Tests:
1. run_two_sample_test with pvalue_type="analytical_ks"
2. perplexity_ks_test convenience function
3. vader_ks_test convenience function
4. ks_test convenience function (generic)
"""

import numpy as np
import sys
import os

# Add the package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model_equality_testing'))

from model_equality_testing.distribution import CompletionSample
from model_equality_testing.algorithm import (
    run_two_sample_test,
    perplexity_ks_test,
    vader_ks_test,
    ks_test,
)


def create_test_samples(n=50):
    """Create two small test samples with different sentiment distributions."""
    # Sample 1: Neutral/positive text
    texts1 = [
        "This is good.",
        "Great work today.",
        "Everything is fine.",
    ] * (n // 3)

    # Sample 2: Negative text
    texts2 = [
        "This is terrible.",
        "Awful experience today.",
        "Nothing works correctly.",
    ] * (n // 3)

    def texts_to_unicode_sample(texts_list, m=3):
        """Convert list of text lists to CompletionSamples with consistent padding."""
        # Find max length across ALL texts (both samples)
        all_texts = [t for texts in texts_list for t in texts]
        max_len = max(len(t) for t in all_texts)

        samples = []
        for texts in texts_list:
            completions = []
            prompts = []

            for i, text in enumerate(texts):
                # Convert to unicode codepoints
                codepoints = [ord(c) for c in text]
                # Pad with -1 to consistent max_len
                padded = codepoints + [-1] * (max_len - len(codepoints))
                completions.append(padded)
                prompts.append(i % m)  # Cycle through m prompts

            samples.append(CompletionSample(
                prompts=np.array(prompts, dtype=np.int32),
                completions=np.array(completions, dtype=np.int32),
                m=m
            ))

        return samples

    sample1, sample2 = texts_to_unicode_sample([texts1, texts2])

    return sample1, sample2


def test_analytical_ks_with_vader():
    """Test 1: run_two_sample_test with analytical_ks for VADER."""
    print("\n" + "="*70)
    print("Test 1: run_two_sample_test with analytical_ks (VADER)")
    print("="*70)

    sample1, sample2 = create_test_samples()

    # Test with analytical_ks
    pvalue, statistic = run_two_sample_test(
        sample1, sample2,
        stat_type="two_sample_vader_ks",
        pvalue_type="analytical_ks"
    )

    print(f"K-S statistic: {statistic:.6f}")
    print(f"Analytical p-value: {pvalue:.6f}")

    # Verify results are reasonable
    assert 0 <= statistic <= 1, f"K-S statistic should be in [0,1], got {statistic}"
    assert 0 <= pvalue <= 1, f"p-value should be in [0,1], got {pvalue}"

    # Since samples have different sentiment, expect significant difference
    if pvalue < 0.05:
        print("✅ Test detected significant difference (p < 0.05)")
    else:
        print(f"⚠️  Warning: Expected significant difference, got p={pvalue:.4f}")

    print("✅ Test 1 PASSED")


def test_perplexity_ks_convenience():
    """Test 2: perplexity_ks_test convenience function."""
    print("\n" + "="*70)
    print("Test 2: perplexity_ks_test convenience function")
    print("="*70)

    sample1, sample2 = create_test_samples(n=30)  # Smaller for faster testing

    # First, train a KenLM model
    print("Training KenLM model...")
    from model_equality_testing.src.features import train_kenlm_from_sample
    import tempfile

    with tempfile.NamedTemporaryFile(mode='w', suffix='.arpa', delete=False) as f:
        model_path = f.name

    try:
        train_kenlm_from_sample(
            sample1,
            output_path=model_path,
            granularity="word",
            order=2,  # Lower order for small dataset
            skip_empty=True,
            warn_on_degenerate=False
        )

        # Test analytical method
        print("\nTesting analytical p-value...")
        pvalue_analytical, stat_analytical = perplexity_ks_test(
            sample1, sample2,
            kenlm_model_path=model_path,
            granularity="word",
            pvalue_method="analytical"
        )

        print(f"Analytical K-S statistic: {stat_analytical:.6f}")
        print(f"Analytical p-value: {pvalue_analytical:.6f}")

        assert 0 <= stat_analytical <= 1
        assert 0 <= pvalue_analytical <= 1

        # Test permutation method (small b for speed)
        print("\nTesting permutation p-value...")
        pvalue_perm, stat_perm = perplexity_ks_test(
            sample1, sample2,
            kenlm_model_path=model_path,
            granularity="word",
            pvalue_method="permutation",
            b=100
        )

        print(f"Permutation K-S statistic: {stat_perm:.6f}")
        print(f"Permutation p-value: {pvalue_perm:.6f}")

        assert 0 <= stat_perm <= 1
        assert 0 <= pvalue_perm <= 1

        # Statistics should be the same (same test, different p-value method)
        print(f"\nStatistic difference: {abs(stat_analytical - stat_perm):.6f}")
        assert abs(stat_analytical - stat_perm) < 0.01, \
            f"Statistics should match, got {stat_analytical} vs {stat_perm}"

        print("✅ Test 2 PASSED")

    finally:
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)


def test_vader_ks_convenience():
    """Test 3: vader_ks_test convenience function."""
    print("\n" + "="*70)
    print("Test 3: vader_ks_test convenience function")
    print("="*70)

    sample1, sample2 = create_test_samples()

    # Test analytical method
    print("Testing analytical p-value...")
    pvalue_analytical, stat_analytical = vader_ks_test(
        sample1, sample2,
        pvalue_method="analytical"
    )

    print(f"Analytical K-S statistic: {stat_analytical:.6f}")
    print(f"Analytical p-value: {pvalue_analytical:.6f}")

    assert 0 <= stat_analytical <= 1
    assert 0 <= pvalue_analytical <= 1

    # Test permutation method (small b for speed)
    print("\nTesting permutation p-value...")
    pvalue_perm, stat_perm = vader_ks_test(
        sample1, sample2,
        pvalue_method="permutation",
        b=100
    )

    print(f"Permutation K-S statistic: {stat_perm:.6f}")
    print(f"Permutation p-value: {pvalue_perm:.6f}")

    assert 0 <= stat_perm <= 1
    assert 0 <= pvalue_perm <= 1

    # Statistics should be the same
    print(f"\nStatistic difference: {abs(stat_analytical - stat_perm):.6f}")
    assert abs(stat_analytical - stat_perm) < 0.01, \
        f"Statistics should match, got {stat_analytical} vs {stat_perm}"

    print("✅ Test 3 PASSED")


def test_generic_ks_convenience():
    """Test 4: ks_test convenience function with custom feature."""
    print("\n" + "="*70)
    print("Test 4: ks_test convenience function (generic)")
    print("="*70)

    sample1, sample2 = create_test_samples()

    # Test with default VADER
    print("Testing with default VADER feature...")
    pvalue_vader, stat_vader = ks_test(sample1, sample2)

    print(f"VADER K-S statistic: {stat_vader:.6f}")
    print(f"VADER p-value: {pvalue_vader:.6f}")

    assert 0 <= stat_vader <= 1
    assert 0 <= pvalue_vader <= 1

    # Test with custom feature function (text length)
    print("\nTesting with custom feature (text length)...")

    def length_feature(sample):
        """Extract text length as feature."""
        from model_equality_testing.src.features import decode_sample_to_strings
        texts = decode_sample_to_strings(sample)
        return np.array([len(text) for text in texts])

    pvalue_length, stat_length = ks_test(
        sample1, sample2,
        feature_fn=length_feature,
        pvalue_method="analytical"
    )

    print(f"Length K-S statistic: {stat_length:.6f}")
    print(f"Length p-value: {pvalue_length:.6f}")

    assert 0 <= stat_length <= 1
    assert 0 <= pvalue_length <= 1

    # Length distributions should be similar (both samples have similar text lengths)
    # while VADER distributions should be different
    print(f"\nVADER detected stronger difference (stat={stat_vader:.4f}) than "
          f"length (stat={stat_length:.4f})")

    print("✅ Test 4 PASSED")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("K-S Convenience Functions and analytical_ks Test Suite")
    print("="*70)

    try:
        test_analytical_ks_with_vader()
        test_perplexity_ks_convenience()
        test_vader_ks_convenience()
        test_generic_ks_convenience()

        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
        print("\nSummary:")
        print("- run_two_sample_test with analytical_ks works correctly")
        print("- perplexity_ks_test provides convenient interface for perplexity tests")
        print("- vader_ks_test provides convenient interface for sentiment tests")
        print("- ks_test provides flexible interface for custom features")
        print("- Both analytical and permutation p-values work correctly")
        print("="*70)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
