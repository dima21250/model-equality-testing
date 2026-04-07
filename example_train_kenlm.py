#!/usr/bin/env python3
"""
Example: Train a KenLM model from MET dataset and run perplexity K-S tests.

This script demonstrates:
1. Loading data from the MET dataset
2. Training a KenLM language model
3. Running perplexity-based K-S tests
4. SANITY CHECK: Comparing two independent fp32 samples (expect no difference)
5. ACTUAL TEST: Comparing fp32 vs int8 quantization (detect if quantization affects perplexity)
"""

import os
import sys

# Configuration
MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
PROMPTS = {"wikipedia_en": [0, 1, 2]}  # Use 3 prompts
L = 200  # Completion length
ROOT_DIR = "./data"
N_TRAIN = 500  # Samples for training (smaller for speed)
N_TEST = 100   # Samples for testing

def check_dataset():
    """Check if dataset is downloaded."""
    if not os.path.exists(ROOT_DIR):
        print(f"Dataset not found at {ROOT_DIR}")
        print("\nDownload the dataset first:")
        print("  python -c \"from model_equality_testing.dataset import download_dataset; download_dataset('./data')\"")
        print("\nWarning: Dataset is 37.1GB and will take time to download.")
        sys.exit(1)

def main():
    print("="*70)
    print("Example: Training KenLM Models from MET Dataset")
    print("="*70)

    check_dataset()

    from model_equality_testing.dataset import load_distribution
    from model_equality_testing.src.features import train_kenlm_from_sample
    from model_equality_testing.algorithm import perplexity_ks_test

    # Step 1: Load reference distribution (fp32)
    print("\n[1/5] Loading fp32 reference distribution...")
    fp32_dist = load_distribution(
        model=MODEL,
        prompt_ids=PROMPTS,
        L=L,
        source="fp32",
        load_in_unicode=True,  # Required for KenLM
        root_dir=ROOT_DIR
    )
    print(f"✓ Loaded distribution: {fp32_dist.m} prompts")

    # Step 2: Sample and train KenLM model
    print(f"\n[2/5] Sampling {N_TRAIN} completions for training...")
    fp32_train = fp32_dist.sample(n=N_TRAIN)
    print(f"✓ Sampled {fp32_train.N} completions")

    print("\n[3/5] Training KenLM language model...")
    print("  (This may take 10-30 seconds)")
    model_path = "fp32_reference.arpa"

    train_kenlm_from_sample(
        fp32_train,
        output_path=model_path,
        granularity="word",
        order=3,  # Lower order for speed with small dataset
        skip_empty=True,
        warn_on_degenerate=False  # Suppress warnings for cleaner output
    )
    print(f"✓ Model saved to {model_path}")

    # Step 3: Draw test samples
    print(f"\n[4/5] Drawing test samples ({N_TEST} each)...")

    # Draw first fp32 sample
    fp32_test = fp32_dist.sample(n=N_TEST)
    print(f"✓ fp32 test sample #1: {fp32_test.N} completions")

    # Load int8 quantization
    int8_dist = load_distribution(
        model=MODEL,
        prompt_ids=PROMPTS,
        L=L,
        source="int8",
        load_in_unicode=True,
        root_dir=ROOT_DIR
    )
    int8_test = int8_dist.sample(n=N_TEST)
    print(f"✓ int8 test sample: {int8_test.N} completions")
    print(f"\nNote: A second independent fp32 sample will be drawn for sanity check")

    # Step 4: Run perplexity K-S tests
    print("\n[5/5] Running perplexity-based K-S tests...")
    print("=" * 70)
    print("We'll run two tests:")
    print("  1. SANITY CHECK: fp32 vs fp32 (expect p-value >= 0.05)")
    print("  2. ACTUAL TEST: fp32 vs int8 (unknown outcome)")
    print("=" * 70)

    # Test 1: fp32 vs fp32 (SANITY CHECK - should NOT be significant)
    print("\n" + "─" * 70)
    print("Test 1: SANITY CHECK (fp32 vs fp32)")
    print("─" * 70)
    print("Purpose: Verify that independent samples from the SAME distribution")
    print("         show NO significant difference (validates our methodology)")
    print("")
    print("Expected: High p-value (>= 0.05) indicating no significant difference")
    print("")

    # Draw a second independent fp32 sample
    fp32_test2 = fp32_dist.sample(n=N_TEST)
    print(f"Sample 1: {fp32_test.N} fp32 completions")
    print(f"Sample 2: {fp32_test2.N} fp32 completions (independent draw)")
    print("")

    pvalue1, stat1 = perplexity_ks_test(
        fp32_test,
        fp32_test2,
        kenlm_model_path=model_path,
        granularity="word",
        pvalue_method="analytical"
    )

    print(f"Results:")
    print(f"  K-S statistic: {stat1:.4f}")
    print(f"  p-value:       {pvalue1:.4f}")
    print("")

    if pvalue1 >= 0.05:
        print("  ✓ ✓ ✓ SANITY CHECK PASSED ✓ ✓ ✓")
        print(f"  No significant difference detected (p={pvalue1:.4f} >= 0.05)")
        print("  This confirms our methodology is working correctly!")
        sanity_pass = True
    else:
        print("  ⚠ ⚠ ⚠ SANITY CHECK FAILED ⚠ ⚠ ⚠")
        print(f"  Unexpected significant difference (p={pvalue1:.4f} < 0.05)")
        print("  This suggests:")
        print("    - Sample size may be too small")
        print("    - Random chance (happens ~5% of the time)")
        print("    - Possible issue with the test setup")
        sanity_pass = False

    # Test 2: fp32 vs int8 (ACTUAL TEST - may or may not be significant)
    print("\n" + "─" * 70)
    print("Test 2: QUANTIZATION TEST (fp32 vs int8)")
    print("─" * 70)
    print("Purpose: Detect if int8 quantization changes perplexity distribution")
    print("")
    print("If p < 0.05: int8 significantly differs from fp32")
    print("If p >= 0.05: int8 statistically similar to fp32")
    print("")

    print(f"Sample 1: {fp32_test.N} fp32 completions")
    print(f"Sample 2: {int8_test.N} int8 completions")
    print("")

    pvalue2, stat2 = perplexity_ks_test(
        fp32_test,
        int8_test,
        kenlm_model_path=model_path,
        granularity="word",
        pvalue_method="analytical"
    )

    print(f"Results:")
    print(f"  K-S statistic: {stat2:.4f}")
    print(f"  p-value:       {pvalue2:.4f}")
    print("")

    if pvalue2 < 0.05:
        print("  ✗ Significant difference in perplexity distributions")
        print("  Interpretation:")
        print("    → int8 quantization AFFECTS language statistics")
        print("    → Completions have different perplexity patterns")
        print(f"    → Effect size: K-S statistic = {stat2:.4f}")
    else:
        print("  ✓ No significant difference in perplexity distributions")
        print("  Interpretation:")
        print("    → int8 quantization PRESERVES language statistics")
        print("    → Completions have similar perplexity patterns")
        print(f"    → int8 is statistically equivalent to fp32 (p={pvalue2:.4f})")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\nKenLM Model:")
    print(f"  File: {model_path}")
    print(f"  Training data: {N_TRAIN} fp32 completions")
    print(f"  Order: 3-gram")
    print(f"  Granularity: word-level")

    print(f"\nTest Results:")
    print(f"  ┌─ Sanity Check (fp32 vs fp32):")
    print(f"  │   K-S statistic: {stat1:.4f}")
    print(f"  │   p-value:       {pvalue1:.4f}")
    print(f"  │   Status:        {'✓ PASS' if pvalue1 >= 0.05 else '✗ FAIL'}")
    if not sanity_pass:
        print(f"  │   Note:         Sanity check failed - results may be unreliable")
    print(f"  │")
    print(f"  └─ Quantization Test (fp32 vs int8):")
    print(f"      K-S statistic: {stat2:.4f}")
    print(f"      p-value:       {pvalue2:.4f}")
    if pvalue2 < 0.05:
        print(f"      Result:        ✗ DIFFERENT (int8 affects perplexity)")
    else:
        print(f"      Result:        ✓ SIMILAR (int8 preserves perplexity)")

    print("\n" + "="*70)

    # Interpretation
    if sanity_pass and pvalue2 < 0.05:
        print("CONCLUSION: int8 quantization significantly changes perplexity")
        print("            distributions compared to fp32 baseline.")
    elif sanity_pass and pvalue2 >= 0.05:
        print("CONCLUSION: int8 quantization preserves perplexity distributions")
        print("            statistically equivalent to fp32 baseline.")
    elif not sanity_pass:
        print("CONCLUSION: Results are inconclusive due to failed sanity check.")
        print("            Consider increasing sample size or re-running the test.")

    print("="*70)

    # Optional: Show how to use with run_two_sample_test
    print("\nAlternative: Using run_two_sample_test framework:")
    print("```python")
    print("from model_equality_testing.algorithm import run_two_sample_test")
    print("")
    print("pvalue, stat = run_two_sample_test(")
    print("    fp32_test, int8_test,")
    print("    stat_type='two_sample_perplexity_ks',")
    print("    pvalue_type='analytical_ks',")
    print(f"    kenlm_model_path='{model_path}'")
    print(")")
    print("```")

    # Cleanup option
    print("\n" + "="*70)
    response = input("\nDelete the trained model? [y/N]: ").strip().lower()
    if response == 'y':
        os.remove(model_path)
        print(f"✓ Deleted {model_path}")
    else:
        print(f"Model kept at: {model_path}")
        print("You can reuse this model for future perplexity K-S tests!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
