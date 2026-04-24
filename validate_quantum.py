"""validate_quantum.py

Validation experiments for quantum-inspired metrics.

Implements the validation strategy from FLAIR-ALTERNATIVE.md:
1. Known Differences (Sanity Check): Different models should show large trace distance
2. Known Similarities (Negative Control): Same model samples should show small trace distance
3. Controlled Perturbations: Metrics should correlate with quantization magnitude
4. Comparison to Baselines: Quantum metrics should provide complementary information

Usage:
    python validate_quantum.py --root_dir ./data --samples 100
"""

import argparse
import numpy as np
from typing import Dict, List, Tuple

from model_equality_testing.dataset import load_distribution
from model_equality_testing.tests import (
    quantum_trace_distance,
    quantum_von_neumann_divergence,
    mmd_hamming,
)
from model_equality_testing.utils import Stopwatch


def load_and_sample(
    model: str,
    source: str,
    prompt_ids: Dict[str, List[int]],
    L: int,
    n_samples: int,
    root_dir: str = "./data",
):
    """Helper to load distribution and draw samples."""
    dist = load_distribution(
        model=model,
        prompt_ids=prompt_ids,
        L=L,
        source=source,
        load_in_unicode=True,
        root_dir=root_dir,
    )
    return dist.sample(n=n_samples)


def sanity_check_known_differences(
    prompt_ids: Dict[str, List[int]],
    L: int,
    n_samples: int,
    root_dir: str,
    embedding_model: str,
) -> Dict[str, float]:
    """Test 1: Different models should show large trace distance.

    Compares Llama-3-8B vs Mistral-7B (very different architectures).
    Expected: trace_distance > 0.3 (moderate to large)
    """
    print("\n" + "="*80)
    print("TEST 1: Known Differences (Sanity Check)")
    print("="*80)
    print("Comparing: Llama-3-8B vs Mistral-7B (fp32)")
    print("Expected: LARGE trace distance (> 0.3)\n")

    with Stopwatch() as sw:
        sample1 = load_and_sample(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "fp32", prompt_ids, L, n_samples, root_dir
        )
        sample2 = load_and_sample(
            "mistralai/Mistral-7B-Instruct-v0.3",
            "fp32", prompt_ids, L, n_samples, root_dir
        )

        trace_dist = quantum_trace_distance(sample1, sample2, embedding_model=embedding_model)
        entropy_div = quantum_von_neumann_divergence(sample1, sample2, embedding_model=embedding_model)
        mmd_stat = mmd_hamming(sample1, sample2)

    print(f"Results (computed in {sw.time:.2f}s):")
    print(f"  - Trace distance:         {trace_dist:.6f}")
    print(f"  - Von Neumann divergence: {entropy_div:.6f}")
    print(f"  - MMD (baseline):         {mmd_stat:.6f}")

    if trace_dist > 0.3:
        print(f"  ✅ PASS: Trace distance {trace_dist:.4f} > 0.3 (different models detected)")
    else:
        print(f"  ⚠️  WARNING: Trace distance {trace_dist:.4f} < 0.3 (may not distinguish models)")

    print("="*80)

    return {
        "trace_distance": trace_dist,
        "entropy_divergence": entropy_div,
        "mmd": mmd_stat,
    }


def sanity_check_known_similarities(
    prompt_ids: Dict[str, List[int]],
    L: int,
    n_samples: int,
    root_dir: str,
    embedding_model: str,
) -> Dict[str, float]:
    """Test 2: Same model (different samples) should show small trace distance.

    Compares two independent samples from the same distribution.
    Expected: trace_distance < 0.15 (small, measuring sampling noise only)
    """
    print("\n" + "="*80)
    print("TEST 2: Known Similarities (Negative Control)")
    print("="*80)
    print("Comparing: Llama-3-8B (fp32) vs Llama-3-8B (fp32) - different samples")
    print("Expected: SMALL trace distance (< 0.15)\n")

    with Stopwatch() as sw:
        # Load distribution once
        dist = load_distribution(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            prompt_ids=prompt_ids,
            L=L,
            source="fp32",
            load_in_unicode=True,
            root_dir=root_dir,
        )

        # Draw two independent samples
        sample1 = dist.sample(n=n_samples)
        sample2 = dist.sample(n=n_samples)

        trace_dist = quantum_trace_distance(sample1, sample2, embedding_model=embedding_model)
        entropy_div = quantum_von_neumann_divergence(sample1, sample2, embedding_model=embedding_model)
        mmd_stat = mmd_hamming(sample1, sample2)

    print(f"Results (computed in {sw.time:.2f}s):")
    print(f"  - Trace distance:         {trace_dist:.6f}")
    print(f"  - Von Neumann divergence: {entropy_div:.6f}")
    print(f"  - MMD (baseline):         {mmd_stat:.6f}")

    if trace_dist < 0.15:
        print(f"  ✅ PASS: Trace distance {trace_dist:.4f} < 0.15 (same distribution detected)")
    else:
        print(f"  ⚠️  WARNING: Trace distance {trace_dist:.4f} > 0.15 (high sampling noise)")

    print("="*80)

    return {
        "trace_distance": trace_dist,
        "entropy_divergence": entropy_div,
        "mmd": mmd_stat,
    }


def controlled_perturbations(
    prompt_ids: Dict[str, List[int]],
    L: int,
    n_samples: int,
    root_dir: str,
    embedding_model: str,
) -> Dict[str, Tuple[float, float, float]]:
    """Test 3: Metrics should correlate with quantization magnitude.

    Compares fp32 vs {fp16, int8, nf4} to see if trace distance increases
    with more aggressive quantization.

    Expected ordering: D(fp32, fp16) < D(fp32, int8) < D(fp32, nf4)
    """
    print("\n" + "="*80)
    print("TEST 3: Controlled Perturbations (Quantization)")
    print("="*80)
    print("Comparing: FP32 vs {FP16, INT8, NF4}")
    print("Expected: Trace distance increases with quantization strength\n")

    # Load fp32 sample (reference)
    sample_fp32 = load_and_sample(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "fp32", prompt_ids, L, n_samples, root_dir
    )

    results = {}
    quantizations = ["fp16", "int8", "nf4"]

    for quant in quantizations:
        try:
            with Stopwatch() as sw:
                sample_quant = load_and_sample(
                    "meta-llama/Meta-Llama-3-8B-Instruct",
                    quant, prompt_ids, L, n_samples, root_dir
                )

                trace_dist = quantum_trace_distance(sample_fp32, sample_quant, embedding_model=embedding_model)
                entropy_div = quantum_von_neumann_divergence(sample_fp32, sample_quant, embedding_model=embedding_model)
                mmd_stat = mmd_hamming(sample_fp32, sample_quant)

            results[quant] = (trace_dist, entropy_div, mmd_stat)
            print(f"{quant.upper():>6} (computed in {sw.time:.2f}s):")
            print(f"  - Trace distance:         {trace_dist:.6f}")
            print(f"  - Von Neumann divergence: {entropy_div:.6f}")
            print(f"  - MMD (baseline):         {mmd_stat:.6f}")

        except Exception as e:
            print(f"{quant.upper():>6}: Skipped (data not available: {e})")
            results[quant] = (None, None, None)

    # Check monotonicity
    trace_dists = [v[0] for v in results.values() if v[0] is not None]
    if len(trace_dists) >= 2:
        print("\nMonotonicity check:")
        is_monotonic = all(trace_dists[i] <= trace_dists[i+1] for i in range(len(trace_dists)-1))
        if is_monotonic:
            print(f"  ✅ PASS: Trace distance increases with quantization strength")
        else:
            print(f"  ⚠️  WARNING: Non-monotonic trace distances: {trace_dists}")

    print("="*80)

    return results


def correlation_with_baselines(
    prompt_ids: Dict[str, List[int]],
    L: int,
    n_samples: int,
    root_dir: str,
    embedding_model: str,
    n_comparisons: int = 5,
) -> None:
    """Test 4: Quantum metrics should provide complementary information.

    Compares quantum metrics vs MMD across multiple distribution pairs.
    If correlation is too high (> 0.95), quantum metrics add little value.
    """
    print("\n" + "="*80)
    print("TEST 4: Correlation with Classical Baselines")
    print("="*80)
    print(f"Running {n_comparisons} comparisons to check correlation...")
    print("Expected: Moderate correlation (quantum metrics provide new information)\n")

    trace_dists = []
    mmd_stats = []

    # Define comparison pairs
    comparisons = [
        ("meta-llama/Meta-Llama-3-8B-Instruct", "fp32", "meta-llama/Meta-Llama-3-8B-Instruct", "int8"),
        ("meta-llama/Meta-Llama-3-8B-Instruct", "fp32", "meta-llama/Meta-Llama-3-8B-Instruct", "nf4"),
        ("meta-llama/Meta-Llama-3-8B-Instruct", "fp32", "mistralai/Mistral-7B-Instruct-v0.3", "fp32"),
    ]

    for i, (model1, src1, model2, src2) in enumerate(comparisons[:n_comparisons], 1):
        try:
            print(f"Comparison {i}: {model1}[{src1}] vs {model2}[{src2}]")

            sample1 = load_and_sample(model1, src1, prompt_ids, L, n_samples, root_dir)
            sample2 = load_and_sample(model2, src2, prompt_ids, L, n_samples, root_dir)

            trace_dist = quantum_trace_distance(sample1, sample2, embedding_model=embedding_model)
            mmd_stat = mmd_hamming(sample1, sample2)

            trace_dists.append(trace_dist)
            mmd_stats.append(mmd_stat)

            print(f"  Trace distance: {trace_dist:.6f}, MMD: {mmd_stat:.6f}\n")

        except Exception as e:
            print(f"  Skipped: {e}\n")

    if len(trace_dists) >= 2:
        correlation = np.corrcoef(trace_dists, mmd_stats)[0, 1]
        print(f"Correlation (Trace Distance vs MMD): {correlation:.4f}")

        if 0.3 <= abs(correlation) <= 0.85:
            print(f"  ✅ PASS: Moderate correlation {correlation:.4f} (complementary metrics)")
        elif abs(correlation) > 0.95:
            print(f"  ⚠️  WARNING: Very high correlation {correlation:.4f} (metrics may be redundant)")
        else:
            print(f"  ⚠️  WARNING: Very low correlation {correlation:.4f} (metrics may be uncorrelated)")
    else:
        print("  Not enough data points to compute correlation")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Validate quantum-inspired metrics on model equality testing dataset"
    )
    parser.add_argument(
        "--root_dir",
        default="./data",
        help="Root directory containing the dataset"
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["0", "1", "2"],
        help="Space-separated prompt IDs"
    )
    parser.add_argument(
        "--dataset",
        default="wikipedia_en",
        help="Dataset name (wikipedia_en, wikipedia_ru, etc.)"
    )
    parser.add_argument(
        "--L",
        type=int,
        default=500,
        help="Completion length (truncation)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples per distribution"
    )
    parser.add_argument(
        "--embedding_model",
        default="all-mpnet-base-v2",
        help="SBERT model name (all-mpnet-base-v2, all-MiniLM-L6-v2, etc.)"
    )
    parser.add_argument(
        "--skip_tests",
        nargs="*",
        default=[],
        choices=["1", "2", "3", "4"],
        help="Tests to skip (1=differences, 2=similarities, 3=perturbations, 4=correlation)"
    )

    args = parser.parse_args()

    prompt_ids = {args.dataset: [int(pid) for pid in args.prompts]}
    skip_tests = set(args.skip_tests)

    print("\n" + "="*80)
    print("QUANTUM METRICS VALIDATION SUITE")
    print("="*80)
    print(f"Dataset: {args.dataset}, Prompts: {args.prompts}")
    print(f"Samples per distribution: {args.samples}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Root directory: {args.root_dir}")
    print("="*80)

    # Run validation tests
    if "1" not in skip_tests:
        sanity_check_known_differences(
            prompt_ids, args.L, args.samples, args.root_dir, args.embedding_model
        )

    if "2" not in skip_tests:
        sanity_check_known_similarities(
            prompt_ids, args.L, args.samples, args.root_dir, args.embedding_model
        )

    if "3" not in skip_tests:
        controlled_perturbations(
            prompt_ids, args.L, args.samples, args.root_dir, args.embedding_model
        )

    if "4" not in skip_tests:
        correlation_with_baselines(
            prompt_ids, args.L, args.samples, args.root_dir, args.embedding_model,
            n_comparisons=3
        )

    print("\n" + "="*80)
    print("VALIDATION SUITE COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Review test results above")
    print("2. If all tests pass, quantum metrics are working correctly")
    print("3. Try running with different embedding models (--embedding_model)")
    print("4. Adjust sample size (--samples) to study statistical stability")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
