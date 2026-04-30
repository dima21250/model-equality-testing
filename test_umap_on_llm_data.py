"""Test UMAP-based trace distance on real LLM completion data.

After successful synthetic validation, test on actual model outputs.
"""

import argparse
import numpy as np
from umap import UMAP
from typing import Dict, List

from model_equality_testing.dataset import load_distribution
from model_equality_testing.distribution import CompletionSample
from model_equality_testing.embeddings import embed_sample
from model_equality_testing.utils import Stopwatch


def compute_density_matrix_umap(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    n_components: int = 20,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
):
    """Construct density matrices via UMAP (Option 1: fit on combined)."""
    N_a, d = embeddings_a.shape
    N_b, _ = embeddings_b.shape

    # Combine embeddings
    combined = np.vstack([embeddings_a, embeddings_b])

    # Fit UMAP on combined data
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        verbose=False,
    )

    reduced_combined = reducer.fit_transform(combined)

    # Split back
    reduced_a = reduced_combined[:N_a]
    reduced_b = reduced_combined[N_a:]

    # Construct covariance matrices
    cov_a = (reduced_a.T @ reduced_a) / N_a
    cov_b = (reduced_b.T @ reduced_b) / N_b

    # Normalize to density matrices
    trace_a = np.trace(cov_a)
    trace_b = np.trace(cov_b)

    if trace_a < 1e-10 or trace_b < 1e-10:
        raise ValueError(f"Covariance trace too small: {trace_a}, {trace_b}")

    return cov_a / trace_a, cov_b / trace_b


def trace_distance(rho_a: np.ndarray, rho_b: np.ndarray) -> float:
    """Compute trace distance."""
    diff = rho_a - rho_b
    eigenvalues = np.linalg.eigvalsh(diff)
    return 0.5 * np.sum(np.abs(eigenvalues))


def compute_trace_distance_umap_on_samples(
    sample1: CompletionSample,
    sample2: CompletionSample,
    embedding_model: str = "all-mpnet-base-v2",
    n_components: int = 20,
    n_neighbors: int = 15,
    batch_size: int = 32,
) -> float:
    """Compute UMAP-based trace distance between two completion samples."""
    # Embed both samples
    print(f"  Embedding samples with {embedding_model}...")
    with Stopwatch() as sw:
        embeddings1 = embed_sample(sample1, model_name=embedding_model, batch_size=batch_size)
        embeddings2 = embed_sample(sample2, model_name=embedding_model, batch_size=batch_size)
    print(f"  Embedding took {sw.time:.2f}s")

    # Compute UMAP-based density matrices
    print(f"  Computing density matrices (UMAP k={n_components}, neighbors={n_neighbors})...")
    rho1, rho2 = compute_density_matrix_umap(
        embeddings1, embeddings2,
        n_components=n_components,
        n_neighbors=n_neighbors,
    )

    # Compute trace distance
    dist = trace_distance(rho1, rho2)

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
    n_components: int = 20,
    n_neighbors: int = 15,
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

    # Compute UMAP-based trace distance
    with Stopwatch() as sw:
        dist = compute_trace_distance_umap_on_samples(
            samp_a, samp_b,
            embedding_model=embedding_model,
            n_components=n_components,
            n_neighbors=n_neighbors,
        )
    print()
    print(f"  Trace distance (UMAP k={n_components}): {dist:.6f} (took {sw.time:.2f}s)")

    return dist


def main():
    parser = argparse.ArgumentParser(
        description="Test UMAP-based trace distance on real LLM data"
    )
    parser.add_argument("--samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--k", type=int, default=20, help="Number of UMAP components")
    parser.add_argument("--neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--L", type=int, default=500, help="Completion length")
    parser.add_argument("--root_dir", default="./data", help="Dataset root directory")
    parser.add_argument(
        "--embedding_model",
        default="all-mpnet-base-v2",
        help="Sentence transformer model"
    )

    args = parser.parse_args()

    prompt_ids = {"wikipedia_en": [0, 1, 2]}

    print("\n" + "="*80)
    print("UMAP-BASED TRACE DISTANCE ON REAL LLM DATA")
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
        n_samples=args.samples,
        n_components=args.k,
        n_neighbors=args.neighbors,
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
        n_samples=args.samples,
        n_components=args.k,
        n_neighbors=args.neighbors,
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
        n_samples=args.samples,
        n_components=args.k,
        n_neighbors=args.neighbors,
        root_dir=args.root_dir,
        embedding_model=args.embedding_model,
    )

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: UMAP-Based Trace Distance Results")
    print("="*80)
    print(f"1. fp32 vs fp32 (same):      {results['fp32_vs_fp32']:.6f}")
    print(f"2. fp32 vs int8 (quant):     {results['fp32_vs_int8']:.6f}")
    print(f"3. Llama vs Mistral (diff):  {results['llama_vs_mistral']:.6f}")
    print()

    # Compute ratios
    ratio_quant = results['fp32_vs_int8'] / results['fp32_vs_fp32'] if results['fp32_vs_fp32'] > 0 else 0
    ratio_model = results['llama_vs_mistral'] / results['fp32_vs_fp32'] if results['fp32_vs_fp32'] > 0 else 0

    print(f"Quantization ratio (vs baseline): {ratio_quant:.2f}x")
    print(f"Model difference ratio (vs baseline): {ratio_model:.2f}x")
    print()

    # Comparison with PCA
    print("="*80)
    print("COMPARISON: UMAP vs PCA (k=5 for PCA)")
    print("="*80)
    print("Test            | PCA k=5 | UMAP k=20 | Better?")
    print("-" * 60)
    print(f"fp32 vs fp32    | 0.076   | {results['fp32_vs_fp32']:.3f}     | {'UMAP' if results['fp32_vs_fp32'] < 0.076 else 'PCA'}")
    print(f"fp32 vs int8    | 0.081   | {results['fp32_vs_int8']:.3f}     | {'UMAP' if results['fp32_vs_int8'] > 0.081 else 'PCA'}")
    print(f"Llama vs Mistral| 0.121   | {results['llama_vs_mistral']:.3f}     | {'UMAP' if results['llama_vs_mistral'] > 0.121 else 'PCA'}")
    print(f"Quant ratio     | 1.07x   | {ratio_quant:.2f}x    | {'UMAP' if ratio_quant > 1.07 else 'PCA'}")
    print()

    # Evaluation
    print("="*80)
    print("EVALUATION")
    print("="*80)

    sanity_ok = results['fp32_vs_fp32'] < 0.05
    quant_detected = results['fp32_vs_int8'] > results['fp32_vs_fp32'] * 1.5
    model_detected = results['llama_vs_mistral'] > results['fp32_vs_int8']
    strong_discrimination = ratio_quant > 2.0

    print(f"Sanity check (fp32 vs fp32 low): {'✓ PASS' if sanity_ok else '✗ FAIL'}")
    print(f"Quantization detected (>1.5× baseline): {'✓ PASS' if quant_detected else '✗ FAIL'}")
    print(f"Model difference detected: {'✓ PASS' if model_detected else '✗ FAIL'}")
    print(f"Strong discrimination (>2× baseline): {'✓ PASS' if strong_discrimination else '✗ FAIL'}")
    print()

    if sanity_ok and quant_detected and model_detected:
        print("✓ SUCCESS: UMAP-based trace distance works on real LLM data!")
        print()
        print("  Expected ordering: same < quantization < different models")
        print(f"  Actual ordering:   {results['fp32_vs_fp32']:.3f} < {results['fp32_vs_int8']:.3f} < {results['llama_vs_mistral']:.3f}")
        print()

        if strong_discrimination:
            print("  ✓ Strong discrimination - significantly better than PCA!")
        else:
            print(f"  ⚠️  Discrimination moderate ({ratio_quant:.1f}x) - better than PCA (1.07x) but not strong")

        print()
        print("  This metric can be used for:")
        print("  - Detecting distribution shifts")
        print("  - Quantifying magnitude of changes")
        print("  - Providing interpretability beyond binary hypothesis tests")
    elif sanity_ok and quant_detected:
        print("⚠️  PARTIAL: Good detection but ordering not perfect")
    elif sanity_ok:
        print("⚠️  PARTIAL: Sanity check passes but discrimination weak")
        print()
        print(f"  Quantization ratio only {ratio_quant:.2f}x (want >2x)")
        print("  May need to tune hyperparameters (k, n_neighbors)")
    else:
        print("✗ FAILURE: Sanity check fails - metric still unreliable")

    print("="*80)


if __name__ == "__main__":
    main()
