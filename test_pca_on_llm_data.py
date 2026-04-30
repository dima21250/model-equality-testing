"""Test PCA-based trace distance on real LLM completion data.

This script applies the PCA fix to the quantum metrics and re-runs:
1. Sanity check: fp32 vs fp32 (should be low)
2. Quantization detection: fp32 vs int8 (should be higher)
3. Architecture difference: Llama-3-8B vs Mistral-7B (should be highest)

If successful, we've found an interpretable metric that goes beyond
binary hypothesis testing.
"""

import argparse
import numpy as np
from sklearn.decomposition import PCA
from typing import Dict, List

from model_equality_testing.dataset import load_distribution
from model_equality_testing.distribution import CompletionSample
from model_equality_testing.embeddings import embed_sample
from model_equality_testing.utils import Stopwatch


def compute_density_matrix_pca(embeddings: np.ndarray, n_components: int = 50) -> np.ndarray:
    """Construct density matrix via PCA (same as validation test)."""
    N, d = embeddings.shape
    k = min(n_components, N - 1, d)

    pca = PCA(n_components=k)
    reduced = pca.fit_transform(embeddings)
    cov = (reduced.T @ reduced) / N
    trace = np.trace(cov)

    if trace < 1e-10:
        raise ValueError(f"Covariance trace too small: {trace}")

    rho = cov / trace
    return rho


def trace_distance_pca(rho_a: np.ndarray, rho_b: np.ndarray) -> float:
    """Compute trace distance."""
    diff = rho_a - rho_b
    eigenvalues = np.linalg.eigvalsh(diff)
    distance = 0.5 * np.sum(np.abs(eigenvalues))
    return distance


def compute_trace_distance_pca_on_samples(
    sample1: CompletionSample,
    sample2: CompletionSample,
    embedding_model: str = "all-mpnet-base-v2",
    n_components: int = 50,
    batch_size: int = 32,
) -> float:
    """Compute PCA-based trace distance between two completion samples."""
    # Embed both samples
    print(f"  Embedding samples with {embedding_model}...")
    with Stopwatch() as sw:
        embeddings1 = embed_sample(sample1, model_name=embedding_model, batch_size=batch_size)
        embeddings2 = embed_sample(sample2, model_name=embedding_model, batch_size=batch_size)
    print(f"  Embedding took {sw.time:.2f}s")

    # Compute PCA-based density matrices
    print(f"  Computing density matrices (PCA k={n_components})...")
    rho1 = compute_density_matrix_pca(embeddings1, n_components=n_components)
    rho2 = compute_density_matrix_pca(embeddings2, n_components=n_components)

    # Compute trace distance
    dist = trace_distance_pca(rho1, rho2)

    return dist


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
        load_in_unicode=True,
        root_dir=root_dir,
    )
    return dist.sample(n=n_samples)


def run_test_case(
    label: str,
    model_a: str,
    model_b: str,
    source_a: str,
    source_b: str,
    prompt_ids: Dict[str, List[int]],
    L: int,
    n_samples: int,
    n_components: int = 50,
    root_dir: str = "./data",
    embedding_model: str = "all-mpnet-base-v2",
) -> float:
    """Run a single test case and return trace distance."""
    print("\n" + "="*80)
    print(label)
    print(f"  Model A: {model_a} [{source_a}]")
    print(f"  Model B: {model_b} [{source_b}]")
    print(f"  Samples: {n_samples}, Prompts: {prompt_ids}")
    print("="*80)

    # Load samples
    print("Loading distributions and sampling...")
    with Stopwatch() as sw:
        samp_a = sample_distribution(model_a, prompt_ids, L, source_a, n_samples, root_dir)
        samp_b = sample_distribution(model_b, prompt_ids, L, source_b, n_samples, root_dir)
    print(f"  Loaded in {sw.time:.2f}s")

    # Compute PCA-based trace distance
    with Stopwatch() as sw:
        dist = compute_trace_distance_pca_on_samples(
            samp_a, samp_b,
            embedding_model=embedding_model,
            n_components=n_components,
        )
    print()
    print(f"  Trace distance (PCA k={n_components}): {dist:.6f} (took {sw.time:.2f}s)")

    return dist


def main():
    parser = argparse.ArgumentParser(
        description="Test PCA-based trace distance on real LLM data"
    )
    parser.add_argument("--samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--k", type=int, default=50, help="Number of PCA components")
    parser.add_argument("--L", type=int, default=500, help="Completion length")
    parser.add_argument("--root_dir", default="./data", help="Dataset root directory")
    parser.add_argument(
        "--embedding_model",
        default="all-mpnet-base-v2",
        help="Sentence transformer model"
    )
    parser.add_argument(
        "--run_quick",
        action="store_true",
        help="Quick test with fewer samples"
    )

    args = parser.parse_args()

    n_samples = 50 if args.run_quick else args.samples
    prompt_ids = {"wikipedia_en": [0, 1, 2]}

    print("\n" + "="*80)
    print("PCA-BASED TRACE DISTANCE ON REAL LLM DATA")
    print("="*80)

    results = {}

    # Test 1: Sanity check - same model, same source
    results['fp32_vs_fp32'] = run_test_case(
        label="Test 1: SANITY CHECK (fp32 vs fp32)",
        model_a="meta-llama/Meta-Llama-3-8B-Instruct",
        model_b="meta-llama/Meta-Llama-3-8B-Instruct",
        source_a="fp32",
        source_b="fp32",
        prompt_ids=prompt_ids,
        L=args.L,
        n_samples=n_samples,
        n_components=args.k,
        root_dir=args.root_dir,
        embedding_model=args.embedding_model,
    )

    # Test 2: Quantization - same model, different source
    results['fp32_vs_int8'] = run_test_case(
        label="Test 2: QUANTIZATION (fp32 vs int8)",
        model_a="meta-llama/Meta-Llama-3-8B-Instruct",
        model_b="meta-llama/Meta-Llama-3-8B-Instruct",
        source_a="fp32",
        source_b="int8",
        prompt_ids=prompt_ids,
        L=args.L,
        n_samples=n_samples,
        n_components=args.k,
        root_dir=args.root_dir,
        embedding_model=args.embedding_model,
    )

    # Test 3: Different models
    results['llama_vs_mistral'] = run_test_case(
        label="Test 3: MODEL DIFFERENCE (Llama-3-8B vs Mistral-7B)",
        model_a="meta-llama/Meta-Llama-3-8B-Instruct",
        model_b="mistralai/Mistral-7B-Instruct-v0.3",
        source_a="fp32",
        source_b="fp32",
        prompt_ids=prompt_ids,
        L=args.L,
        n_samples=n_samples,
        n_components=args.k,
        root_dir=args.root_dir,
        embedding_model=args.embedding_model,
    )

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: PCA-Based Trace Distance Results")
    print("="*80)
    print(f"1. fp32 vs fp32 (same):      {results['fp32_vs_fp32']:.6f}")
    print(f"2. fp32 vs int8 (quant):     {results['fp32_vs_int8']:.6f}")
    print(f"3. Llama vs Mistral (diff):  {results['llama_vs_mistral']:.6f}")
    print()

    # Evaluation
    print("="*80)
    print("EVALUATION")
    print("="*80)

    sanity_ok = results['fp32_vs_fp32'] < 0.05
    quant_detected = results['fp32_vs_int8'] > results['fp32_vs_fp32'] * 2
    model_detected = results['llama_vs_mistral'] > results['fp32_vs_int8']

    print(f"Sanity check (fp32 vs fp32 low): {'✓ PASS' if sanity_ok else '✗ FAIL'}")
    print(f"Quantization detected: {'✓ PASS' if quant_detected else '✗ FAIL'}")
    print(f"Model difference detected: {'✓ PASS' if model_detected else '✗ FAIL'}")
    print()

    if sanity_ok and quant_detected:
        print("✓ SUCCESS: PCA-based trace distance works on real LLM data!")
        print()
        print("  Expected ordering: same < quantization < different models")
        print(f"  Actual ordering:   {results['fp32_vs_fp32']:.3f} < {results['fp32_vs_int8']:.3f} < {results['llama_vs_mistral']:.3f}")
        print()
        print("  This metric can now be used for:")
        print("  - Detecting distribution shifts")
        print("  - Quantifying magnitude of changes")
        print("  - Providing interpretability beyond binary hypothesis tests")
    elif sanity_ok:
        print("⚠️  PARTIAL: Sanity check passes but discrimination is weak")
        print()
        print("  May need to:")
        print("  - Adjust k (try different PCA dimensions)")
        print("  - Try different embedding models")
        print("  - Combine with Von Neumann divergence for better signal")
    else:
        print("✗ FAILURE: Sanity check fails - metric still unreliable")
        print()
        print("  PCA fix works on synthetic data but not real LLM completions")
        print("  This suggests embedding space structure is different from Gaussian")

    print("="*80)


if __name__ == "__main__":
    main()
