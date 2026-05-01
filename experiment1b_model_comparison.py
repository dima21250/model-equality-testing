"""Experiment 1b: Model Comparison with EmbeddingGemma-300M

Test whether EmbeddingGemma-300M embeddings can separate:
- Llama-3-8B vs Mistral-7B (different architectures)

If this works, it validates using semantic embeddings + quantum metrics
for interpretability of architecture differences.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Dict, List

from model_equality_testing.dataset import load_distribution
from model_equality_testing.distribution import CompletionSample
from model_equality_testing.utils import Stopwatch
from embeddinggemma_interface import get_embeddings, get_client


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


def unicode_to_string(codepoints: np.ndarray) -> str:
    """Convert array of unicode codepoints to string."""
    try:
        return ''.join(chr(int(cp)) for cp in codepoints if cp != -1)
    except (ValueError, OverflowError):
        return ''.join(chr(int(cp)) for cp in codepoints if cp != -1 and 0 <= cp <= 0x10FFFF)


def completions_to_strings(sample: CompletionSample) -> List[str]:
    """Convert CompletionSample (unicode codepoints) to list of strings."""
    strings = []
    completions = sample.completion_sample.numpy() if hasattr(sample.completion_sample, 'numpy') else sample.completion_sample

    for completion in completions:
        valid_codepoints = completion[completion != -1]
        text = unicode_to_string(valid_codepoints)
        strings.append(text)
    return strings


def run_tsne_comparison(
    samples_llama: CompletionSample,
    samples_mistral: CompletionSample,
    n_components: int = 2,
    perplexity: int = 30,
    random_state: int = 42,
):
    """Run t-SNE visualization comparing Llama vs Mistral."""
    print("\n" + "="*80)
    print("MODEL COMPARISON: Llama-3-8B vs Mistral-7B")
    print("="*80)

    # Convert to strings
    print("Converting completions to strings...")
    strings_llama = completions_to_strings(samples_llama)
    strings_mistral = completions_to_strings(samples_mistral)

    print(f"  Llama: {len(strings_llama)} completions")
    print(f"  Mistral: {len(strings_mistral)} completions")

    # Get embeddings
    print("\nGetting EmbeddingGemma-300M embeddings...")
    client = get_client()
    with Stopwatch() as sw:
        emb_llama = get_embeddings(strings_llama, client=client, batch_size=32, verbose=True)
        emb_mistral = get_embeddings(strings_mistral, client=client, batch_size=32, verbose=True)
    print(f"  Embedding completed in {sw.time:.2f}s")
    print(f"  Llama embedding shape: {emb_llama.shape}")
    print(f"  Mistral embedding shape: {emb_mistral.shape}")

    # Combine embeddings
    combined = np.vstack([emb_llama, emb_mistral])
    labels = ['Llama'] * len(emb_llama) + ['Mistral'] * len(emb_mistral)

    # Run t-SNE
    print(f"\nRunning t-SNE (perplexity={perplexity})...")
    with Stopwatch() as sw:
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            verbose=1
        )
        reduced = tsne.fit_transform(combined)
    print(f"  t-SNE completed in {sw.time:.2f}s")

    # Plot
    print("\nGenerating plot...")
    plt.figure(figsize=(12, 8))

    colors = {'Llama': '#1f77b4', 'Mistral': '#ff7f0e'}
    for model_name in ['Llama', 'Mistral']:
        mask = np.array([l == model_name for l in labels])
        plt.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            c=colors[model_name],
            label=f"{model_name}-7B" if model_name == "Mistral" else f"{model_name}-3-8B",
            alpha=0.7,
            s=80,
            edgecolors='black',
            linewidth=0.5
        )

    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.title('Semantic Embedding Space: Llama-3-8B vs Mistral-7B', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    filename = "model_comparison_llama_vs_mistral.png"
    plt.savefig(filename, dpi=150)
    print(f"  Plot saved to: {filename}")

    # Visual assessment
    print("\n" + "="*80)
    print("RESULT INTERPRETATION:")
    print("="*80)

    # Calculate separation metric
    llama_center = reduced[:len(emb_llama)].mean(axis=0)
    mistral_center = reduced[len(emb_llama):].mean(axis=0)
    center_distance = np.linalg.norm(llama_center - mistral_center)

    llama_spread = np.std(reduced[:len(emb_llama)], axis=0).mean()
    mistral_spread = np.std(reduced[len(emb_llama):], axis=0).mean()
    avg_spread = (llama_spread + mistral_spread) / 2

    separation_ratio = center_distance / avg_spread if avg_spread > 0 else 0

    print(f"Center-to-center distance: {center_distance:.3f}")
    print(f"Average within-cluster spread: {avg_spread:.3f}")
    print(f"Separation ratio: {separation_ratio:.2f}")
    print()

    if separation_ratio > 2.0:
        print("✓ STRONG SEPARATION: Models form distinct clusters!")
        print("  → EmbeddingGemma captures semantic differences between architectures")
        print("  → Quantum metrics can characterize WHAT differs semantically")
        print("  → Proceed to Experiment 2: Semantic characterization via quantum metrics")
    elif separation_ratio > 1.0:
        print("⚠ MODERATE SEPARATION: Some clustering but significant overlap")
        print("  → EmbeddingGemma captures some semantic differences")
        print("  → May still be useful for interpretability with quantum metrics")
    else:
        print("✗ NO SEPARATION: Models produce similar semantic embeddings")
        print("  → Different architectures don't produce semantically different outputs")
        print("  → Semantic embedding approach may not be useful here")

    print("="*80)

    return reduced, labels, separation_ratio


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 1b: Model comparison with EmbeddingGemma-300M"
    )
    parser.add_argument("--samples", type=int, default=100, help="Number of samples per model")
    parser.add_argument("--L", type=int, default=500, help="Completion length")
    parser.add_argument("--root_dir", default="./data", help="Dataset root directory")
    parser.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity")
    parser.add_argument("--dataset", default="wikipedia_en", help="Dataset name (wikipedia_en, humaneval, ultrachat, etc.)")
    parser.add_argument("--prompts", nargs="+", type=int, default=[0, 1, 2], help="Prompt IDs to use")

    args = parser.parse_args()

    prompt_ids = {args.dataset: args.prompts}

    print("\n" + "="*80)
    print("EXPERIMENT 1b: Model Comparison via Semantic Embeddings")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model A: meta-llama/Meta-Llama-3-8B-Instruct")
    print(f"  Model B: mistralai/Mistral-7B-Instruct-v0.3")
    print(f"  Dataset: {args.dataset}")
    print(f"  Prompt IDs: {args.prompts}")
    print(f"  Samples per model: {args.samples}")
    print(f"  Completion length: {args.L}")
    print(f"  t-SNE perplexity: {args.perplexity}")

    # Load distributions
    print("\n" + "="*80)
    print("Loading distributions...")
    print("="*80)

    with Stopwatch() as sw:
        samp_llama = sample_distribution(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            prompt_ids, args.L, "fp32", args.samples, args.root_dir
        )
        samp_mistral = sample_distribution(
            "mistralai/Mistral-7B-Instruct-v0.3",
            prompt_ids, args.L, "fp32", args.samples, args.root_dir
        )
    print(f"Loaded in {sw.time:.2f}s")

    # Run comparison
    reduced, labels, separation = run_tsne_comparison(
        samp_llama, samp_mistral,
        perplexity=args.perplexity,
    )

    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT 1b COMPLETE")
    print("="*80)
    print(f"\nSeparation ratio: {separation:.2f}")
    print(f"Generated plot: model_comparison_llama_vs_mistral.png")
    print("\nIf separation ratio > 2.0:")
    print("  → Ready for Experiment 2: Quantum metrics for semantic characterization")
    print("  → Goal: Understand HOW the models differ semantically")
    print("="*80)


if __name__ == "__main__":
    main()
