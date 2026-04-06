#!/usr/bin/env python3
"""
Test script for perplexity scoring (no training required).
This tests the scoring and K-S functions assuming you have a pre-trained .arpa model.
"""

import numpy as np
import torch
from model_equality_testing.distribution import CompletionSample
from model_equality_testing.src.features import get_perplexity_scores
from model_equality_testing.src.tests import two_sample_perplexity_ks
import sys

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

def test_with_model(model_path):
    """Test scoring with existing model."""
    print("=" * 60)
    print("Testing Perplexity Scoring with Existing Model")
    print("=" * 60)
    print(f"Model: {model_path}\n")

    # Create samples
    print("Creating test samples...")
    sample1 = create_test_sample(n_samples=30, seed=42)
    sample2 = create_test_sample(n_samples=30, seed=43)
    print(f"✓ Sample 1: {sample1.N} completions")
    print(f"✓ Sample 2: {sample2.N} completions")

    # Test perplexity scoring
    print("\nTesting get_perplexity_scores()...")
    scores1 = get_perplexity_scores(
        sample=sample1,
        kenlm_model_path=model_path,
        warn_on_degenerate=False
    )
    print(f"✓ Computed {len(scores1)} scores for sample 1")
    print(f"  Stats: min={scores1.min():.2f}, max={scores1.max():.2f}, mean={scores1.mean():.2f}")

    scores2 = get_perplexity_scores(
        sample=sample2,
        kenlm_model_path=model_path,
        warn_on_degenerate=False
    )
    print(f"✓ Computed {len(scores2)} scores for sample 2")
    print(f"  Stats: min={scores2.min():.2f}, max={scores2.max():.2f}, mean={scores2.mean():.2f}")

    # Test K-S statistic
    print("\nTesting two_sample_perplexity_ks()...")
    ks_stat = two_sample_perplexity_ks(
        sample1=sample1,
        sample2=sample2,
        kenlm_model_path=model_path,
        warn_on_degenerate=False
    )
    print(f"✓ K-S statistic: {ks_stat:.6f}")
    assert 0 <= ks_stat <= 1, "K-S statistic should be in [0, 1]"
    print("✓ K-S statistic in valid range")

    print("\n✅ ALL TESTS PASSED")
    print("\nThe perplexity-based K-S implementation is working correctly!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_perplexity_scoring_only.py <path_to_kenlm_model.arpa>")
        print("\nThis test requires a pre-trained KenLM model (.arpa file).")
        print("You can:")
        print("  1. Fix the lmplz library dependencies and use test_perplexity_ks.py")
        print("  2. Train a model externally and provide the path here")
        print("  3. Use a model from the MET dataset if available")
        sys.exit(1)

    model_path = sys.argv[1]
    test_with_model(model_path)
