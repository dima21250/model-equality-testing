"""Test different k values for PCA-based trace distance.

Find the optimal number of PCA components that:
1. Keeps sanity check passing (fp32 vs fp32 low)
2. Maximizes discrimination (fp32 vs int8, Llama vs Mistral)
"""

import numpy as np
from sklearn.decomposition import PCA

from model_equality_testing.dataset import load_distribution
from model_equality_testing.embeddings import embed_sample
from model_equality_testing.utils import Stopwatch


def compute_density_matrix_pca(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    """Construct density matrix via PCA."""
    N, d = embeddings.shape
    k = min(n_components, N - 1, d)

    pca = PCA(n_components=k)
    reduced = pca.fit_transform(embeddings)
    cov = (reduced.T @ reduced) / N
    trace = np.trace(cov)

    if trace < 1e-10:
        raise ValueError(f"Covariance trace too small: {trace}")

    return cov / trace


def trace_distance_pca(rho_a: np.ndarray, rho_b: np.ndarray) -> float:
    """Compute trace distance."""
    diff = rho_a - rho_b
    eigenvalues = np.linalg.eigvalsh(diff)
    return 0.5 * np.sum(np.abs(eigenvalues))


def compute_distance_for_k(embeddings_a, embeddings_b, k):
    """Compute trace distance for given k."""
    rho_a = compute_density_matrix_pca(embeddings_a, n_components=k)
    rho_b = compute_density_matrix_pca(embeddings_b, n_components=k)
    return trace_distance_pca(rho_a, rho_b)


def main():
    print("\n" + "="*80)
    print("K-VALUE SWEEP FOR PCA-BASED TRACE DISTANCE")
    print("="*80 + "\n")

    # Load and embed samples once
    print("Loading samples...")
    prompt_ids = {"wikipedia_en": [0, 1, 2]}
    L = 500
    n_samples = 100

    # fp32 vs fp32 (sanity check)
    dist_fp32_1 = load_distribution(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        prompt_ids=prompt_ids,
        L=L,
        source="fp32",
        load_in_unicode=True,
    )
    dist_fp32_2 = load_distribution(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        prompt_ids=prompt_ids,
        L=L,
        source="fp32",
        load_in_unicode=True,
    )

    # fp32 vs int8 (quantization)
    dist_int8 = load_distribution(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        prompt_ids=prompt_ids,
        L=L,
        source="int8",
        load_in_unicode=True,
    )

    # Sample
    samp_fp32_1 = dist_fp32_1.sample(n=n_samples)
    samp_fp32_2 = dist_fp32_2.sample(n=n_samples)
    samp_int8 = dist_int8.sample(n=n_samples)

    print("Embedding samples...")
    emb_fp32_1 = embed_sample(samp_fp32_1, model_name="all-mpnet-base-v2", batch_size=32)
    emb_fp32_2 = embed_sample(samp_fp32_2, model_name="all-mpnet-base-v2", batch_size=32)
    emb_int8 = embed_sample(samp_int8, model_name="all-mpnet-base-v2", batch_size=32)

    print(f"Embeddings shape: {emb_fp32_1.shape}\n")

    # Test different k values
    k_values = [5, 10, 20, 30, 50, 75, 100, 150, 200]

    print("="*80)
    print("k    | fp32-fp32 | fp32-int8 | Ratio | Discrimination")
    print("="*80)

    best_k = None
    best_ratio = 0

    for k in k_values:
        d_same = compute_distance_for_k(emb_fp32_1, emb_fp32_2, k)
        d_quant = compute_distance_for_k(emb_fp32_1, emb_int8, k)

        ratio = d_quant / d_same if d_same > 0 else 0

        # Discrimination: want low baseline and high ratio
        # Penalize if baseline is too high (sanity check fails)
        sanity_ok = d_same < 0.1
        discrimination_score = ratio if sanity_ok else 0

        status = ""
        if discrimination_score > best_ratio:
            best_ratio = discrimination_score
            best_k = k
            status = " ← BEST"

        print(f"{k:4d} | {d_same:9.6f} | {d_quant:9.6f} | {ratio:5.2f}x | {discrimination_score:5.2f}{status}")

    print("="*80)
    print(f"\nOptimal k: {best_k} (ratio: {best_ratio:.2f}x)")
    print()

    # Test the best k
    if best_k:
        print(f"Testing k={best_k} on Llama vs Mistral...")

        dist_mistral = load_distribution(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            prompt_ids=prompt_ids,
            L=L,
            source="fp32",
            load_in_unicode=True,
        )
        samp_mistral = dist_mistral.sample(n=n_samples)
        emb_mistral = embed_sample(samp_mistral, model_name="all-mpnet-base-v2", batch_size=32)

        d_model = compute_distance_for_k(emb_fp32_1, emb_mistral, best_k)

        print(f"\nResults with k={best_k}:")
        print(f"  fp32 vs fp32:    {compute_distance_for_k(emb_fp32_1, emb_fp32_2, best_k):.6f}")
        print(f"  fp32 vs int8:    {compute_distance_for_k(emb_fp32_1, emb_int8, best_k):.6f}")
        print(f"  Llama vs Mistral: {d_model:.6f}")
        print()

        # Evaluate ordering
        d_same_best = compute_distance_for_k(emb_fp32_1, emb_fp32_2, best_k)
        d_quant_best = compute_distance_for_k(emb_fp32_1, emb_int8, best_k)

        correct_ordering = d_same_best < d_quant_best < d_model

        if correct_ordering:
            print("✓ Correct ordering: same < quantization < different models")
        else:
            print("✗ Ordering not monotonic - may need different approach")

    print("="*80)


if __name__ == "__main__":
    main()
