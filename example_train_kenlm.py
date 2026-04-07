#!/usr/bin/env python3
"""
Example: Train a KenLM model from MET dataset and run perplexity K-S tests.

This script demonstrates:
1. Loading data from the MET dataset
2. Training a KenLM language model
3. Running perplexity-based K-S tests
4. Comparing different quantizations
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

    # Step 3: Test against different quantizations
    print(f"\n[4/5] Drawing test samples ({N_TEST} each)...")
    fp32_test = fp32_dist.sample(n=N_TEST)
    print(f"✓ fp32 test sample: {fp32_test.N} completions")

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

    # Step 4: Run perplexity K-S tests
    print("\n[5/5] Running perplexity-based K-S tests...")
    print("-" * 70)

    # Test 1: fp32 vs fp32 (should NOT be significant)
    print("\nTest 1: fp32 vs fp32 (sanity check)")
    fp32_test2 = fp32_dist.sample(n=N_TEST)
    pvalue1, stat1 = perplexity_ks_test(
        fp32_test,
        fp32_test2,
        kenlm_model_path=model_path,
        granularity="word",
        pvalue_method="analytical"
    )
    print(f"  K-S statistic: {stat1:.4f}")
    print(f"  p-value: {pvalue1:.4f}")
    if pvalue1 >= 0.05:
        print("  ✓ PASS: No significant difference (as expected)")
    else:
        print("  ⚠ WARNING: Unexpected significant difference")

    # Test 2: fp32 vs int8 (may or may not be significant)
    print("\nTest 2: fp32 vs int8 (quantization test)")
    pvalue2, stat2 = perplexity_ks_test(
        fp32_test,
        int8_test,
        kenlm_model_path=model_path,
        granularity="word",
        pvalue_method="analytical"
    )
    print(f"  K-S statistic: {stat2:.4f}")
    print(f"  p-value: {pvalue2:.4f}")
    if pvalue2 < 0.05:
        print("  ✗ Significant difference in perplexity distributions")
        print("    → int8 quantization affects language statistics")
    else:
        print("  ✓ No significant difference in perplexity distributions")
        print("    → int8 quantization preserves language statistics")

    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"Trained KenLM model: {model_path}")
    print(f"  - Trained on {N_TRAIN} fp32 completions")
    print(f"  - Order: 3-gram")
    print(f"  - Granularity: word-level")
    print(f"\nTest Results:")
    print(f"  fp32 vs fp32: K-S={stat1:.4f}, p={pvalue1:.4f} {'✓' if pvalue1 >= 0.05 else '✗'}")
    print(f"  fp32 vs int8: K-S={stat2:.4f}, p={pvalue2:.4f} {'✓' if pvalue2 >= 0.05 else '✗'}")
    print("\n" + "="*70)

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
