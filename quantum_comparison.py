"""quantum_comparison.py

Compare quantum-inspired metrics against classical statistical tests.

This script demonstrates the use of quantum metrics (trace distance, von Neumann
entropy divergence, quantum relative entropy) computed on SBERT embeddings,
and compares them to classical tests (MMD, VADER K-S) on the model equality
testing dataset.

Tests both equivalence (same model/source) and difference (different models/sources)
cases to evaluate discriminative power of quantum metrics.
"""

import argparse
from typing import Dict, List
import torch
import numpy as np

from model_equality_testing.dataset import load_distribution
from model_equality_testing.algorithm import run_two_sample_test
from model_equality_testing.tests import (
    quantum_von_neumann_divergence,
    quantum_relative_entropy_test,
    mmd_hamming,
    two_sample_vader_ks,
)
from model_equality_testing.utils import Stopwatch


def sample_distribution(
    model: str,
    prompt_ids: Dict[str, List[int]],
    L: int,
    source: str,
    n_samples: int,
    root_dir: str = "./data",
):
    """Load a distribution and draw n_samples completions."""
    dist = load_distribution(
        model=model,
        prompt_ids=prompt_ids,
        L=L,
        source=source,
        load_in_unicode=True,  # Quantum tests require unicode
        root_dir=root_dir,
    )
    return dist.sample(n=n_samples)


def compute_quantum_metrics(sample1, sample2, embedding_model="all-mpnet-base-v2", b=100):
    """Compute all quantum metrics with timing and statistical calibration.

    Args:
        sample1: First CompletionSample
        sample2: Second CompletionSample
        embedding_model: Name of SBERT model to use
        b: Number of permutations for p-value computation

    Returns:
        Dict mapping metric name to (statistic, pvalue, elapsed_time)
    """
    results = {}

    print("  Computing quantum metrics with permutation tests...")

    # Von Neumann entropy divergence with permutation p-value
    with Stopwatch() as sw:
        pvalue, entropy_div = run_two_sample_test(
            sample1, sample2,
            stat_type="quantum_von_neumann_divergence",
            pvalue_type="permutation_pvalue",
            b=b,
            embedding_model=embedding_model
        )
    results["Von Neumann Divergence"] = (entropy_div, pvalue, sw.time)
    print(f"    - Von Neumann divergence: {entropy_div:.6f}, p={pvalue:.4f} (took {sw.time:.2f}s)")

    # Quantum relative entropy (symmetric) - DISABLED due to rank mismatch issues
    # Consistently returns np.inf when rank(ρ) > rank(σ), which occurs frequently
    # with finite sample sizes. Needs further investigation or larger sample sizes.
    #
    # with Stopwatch() as sw:
    #     pvalue, qre = run_two_sample_test(
    #         sample1, sample2,
    #         stat_type="quantum_relative_entropy",
    #         pvalue_type="permutation_pvalue",
    #         b=b,
    #         embedding_model=embedding_model,
    #         symmetric=True
    #     )
    # results["QRE (symmetric)"] = (qre, pvalue, sw.time)
    # if qre == np.inf:
    #     print(f"    - QRE (symmetric): inf, p={pvalue:.4f} (took {sw.time:.2f}s)")
    # else:
    #     print(f"    - QRE (symmetric): {qre:.6f}, p={pvalue:.4f} (took {sw.time:.2f}s)")

    return results


def compute_classical_metrics(sample1, sample2):
    """Compute baseline classical metrics for comparison.

    Returns:
        Dict mapping metric name to (value, pvalue, elapsed_time)
    """
    results = {}

    print("  Computing classical metrics...")

    # MMD Hamming
    with Stopwatch() as sw:
        mmd_stat = mmd_hamming(sample1, sample2)
        pvalue, _ = run_two_sample_test(
            sample1, sample2,
            stat_type="mmd_hamming",
            pvalue_type="permutation_pvalue",
            b=100,
        )
    results["MMD (Hamming)"] = (mmd_stat, pvalue, sw.time)
    print(f"    - MMD (Hamming): {mmd_stat:.6f}, p={pvalue:.4f} (took {sw.time:.2f}s)")

    # VADER K-S
    try:
        with Stopwatch() as sw:
            vader_stat = two_sample_vader_ks(sample1, sample2)
        results["VADER K-S"] = (vader_stat, None, sw.time)
        print(f"    - VADER K-S: {vader_stat:.6f} (took {sw.time:.2f}s)")
    except ImportError:
        print("    - VADER K-S: Skipped (vaderSentiment not installed)")
        results["VADER K-S"] = (None, None, 0.0)

    return results


def run_comparison(
    model_a: str,
    model_b: str,
    source_a: str,
    source_b: str,
    prompt_ids: Dict[str, List[int]],
    L: int,
    n_samples: int,
    label: str,
    root_dir: str = "./data",
    embedding_model: str = "all-mpnet-base-v2",
    b: int = 100,
):
    """Run full comparison suite on a pair of distributions.

    Args:
        model_a: First model name
        model_b: Second model name
        source_a: Source for model A (fp32, int8, etc.)
        source_b: Source for model B
        prompt_ids: Dictionary mapping dataset names to prompt ID lists
        L: Completion length
        n_samples: Number of samples per distribution
        label: Descriptive label for this comparison
        root_dir: Root directory for dataset
        embedding_model: SBERT model name
        b: Number of permutations for p-value computation
    """
    print(f"\n{'='*80}")
    print(f"{label}")
    print(f"  Model A: {model_a} [{source_a}]")
    print(f"  Model B: {model_b} [{source_b}]")
    print(f"  Samples: {n_samples}, Prompts: {prompt_ids}")
    print(f"  Permutations: {b}")
    print(f"{'='*80}\n")

    # Load samples
    print("Loading distributions and sampling...")
    with Stopwatch() as sw:
        samp_a = sample_distribution(model_a, prompt_ids, L, source_a, n_samples, root_dir)
        samp_b = sample_distribution(model_b, prompt_ids, L, source_b, n_samples, root_dir)
    print(f"  Loaded in {sw.time:.2f}s\n")

    # Compute metrics
    quantum_results = compute_quantum_metrics(samp_a, samp_b, embedding_model=embedding_model, b=b)
    print()
    classical_results = compute_classical_metrics(samp_a, samp_b)

    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare quantum-inspired metrics against classical tests on LLM outputs."
    )
    parser.add_argument(
        "--model_a",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="First model for comparison"
    )
    parser.add_argument(
        "--model_b",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Second model for comparison (default: same as model_a)"
    )
    parser.add_argument(
        "--source_a",
        default="fp32",
        help="Source for model A (fp32, int8, nf4, etc.)"
    )
    parser.add_argument(
        "--source_b",
        default="int8",
        help="Source for model B (fp32, int8, nf4, etc.)"
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
        help="Number of completions per distribution"
    )
    parser.add_argument(
        "--root_dir",
        default="./data",
        help="Root directory for dataset"
    )
    parser.add_argument(
        "--embedding_model",
        default="all-mpnet-base-v2",
        help="SBERT model name (all-mpnet-base-v2, all-MiniLM-L6-v2, etc.)"
    )
    parser.add_argument(
        "--run_suite",
        action="store_true",
        help="Run full validation suite (equivalence + difference cases)"
    )
    parser.add_argument(
        "--b",
        type=int,
        default=100,
        help="Number of permutations for p-value computation (default: 100)"
    )

    args = parser.parse_args()

    prompt_ids = {args.dataset: [int(pid) for pid in args.prompts]}

    if args.run_suite:
        print("\n" + "="*80)
        print("QUANTUM METRICS VALIDATION SUITE")
        print("="*80)

        # Case 1: Same model, same source (should show SMALL differences)
        run_comparison(
            model_a=args.model_a,
            model_b=args.model_a,
            source_a=args.source_a,
            source_b=args.source_a,
            prompt_ids=prompt_ids,
            L=args.L,
            n_samples=args.samples,
            label="Case 1: Equivalence Test (Same Model, Same Source)",
            root_dir=args.root_dir,
            embedding_model=args.embedding_model,
            b=args.b,
        )

        # Case 2: Same model, different source (should show MODERATE differences)
        run_comparison(
            model_a=args.model_a,
            model_b=args.model_a,
            source_a="fp32",
            source_b="int8",
            prompt_ids=prompt_ids,
            L=args.L,
            n_samples=args.samples,
            label="Case 2: Quantization Test (FP32 vs INT8)",
            root_dir=args.root_dir,
            embedding_model=args.embedding_model,
            b=args.b,
        )

        # Case 3: Different models (should show LARGE differences)
        run_comparison(
            model_a="meta-llama/Meta-Llama-3-8B-Instruct",
            model_b="mistralai/Mistral-7B-Instruct-v0.3",
            source_a="fp32",
            source_b="fp32",
            prompt_ids=prompt_ids,
            L=args.L,
            n_samples=args.samples,
            label="Case 3: Model Difference Test (Llama-3-8B vs Mistral-7B)",
            root_dir=args.root_dir,
            embedding_model=args.embedding_model,
            b=args.b,
        )
    else:
        # Single comparison
        run_comparison(
            model_a=args.model_a,
            model_b=args.model_b,
            source_a=args.source_a,
            source_b=args.source_b,
            prompt_ids=prompt_ids,
            L=args.L,
            n_samples=args.samples,
            label="Quantum vs Classical Metrics Comparison",
            root_dir=args.root_dir,
            embedding_model=args.embedding_model,
            b=args.b,
        )


if __name__ == "__main__":
    main()
