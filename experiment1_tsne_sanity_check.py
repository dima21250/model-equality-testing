"""Experiment 1: t-SNE Sanity Check for EmbeddingGemma-300M

Test whether EmbeddingGemma-300M embeddings can separate:
1. fp32 vs fp32 (should overlap - sanity check)
2. fp32 vs int8 (should separate if embedding captures quantization effects)

This is the critical first test to see if EmbeddingGemma is better than MPNet.
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
        # Handle invalid codepoints gracefully
        return ''.join(chr(int(cp)) for cp in codepoints if cp != -1 and 0 <= cp <= 0x10FFFF)


def completions_to_strings(sample: CompletionSample) -> List[str]:
    """Convert CompletionSample (unicode codepoints) to list of strings."""
    strings = []
    # Convert torch tensor to numpy if needed
    completions = sample.completion_sample.numpy() if hasattr(sample.completion_sample, 'numpy') else sample.completion_sample

    for completion in completions:
        # Remove padding (-1) and convert unicode codepoints to string
        valid_codepoints = completion[completion != -1]
        text = unicode_to_string(valid_codepoints)
        strings.append(text)
    return strings


def run_tsne_test(
    label: str,
    samples_a: CompletionSample,
    samples_b: CompletionSample,
    label_a: str,
    label_b: str,
    n_components: int = 2,
    perplexity: int = 30,
    random_state: int = 42,
):
    """Run t-SNE visualization comparing two sets of completions."""
    print("\n" + "="*80)
    print(label)
    print("="*80)

    # Convert to strings
    print("Converting completions to strings...")
    strings_a = completions_to_strings(samples_a)
    strings_b = completions_to_strings(samples_b)

    print(f"  Sample A: {len(strings_a)} completions")
    print(f"  Sample B: {len(strings_b)} completions")

    # Get embeddings
    print("\nGetting EmbeddingGemma-300M embeddings...")
    client = get_client()
    with Stopwatch() as sw:
        emb_a = get_embeddings(strings_a, client=client, batch_size=32, verbose=True)
        emb_b = get_embeddings(strings_b, client=client, batch_size=32, verbose=True)
    print(f"  Embedding completed in {sw.time:.2f}s")
    print(f"  Embedding shape A: {emb_a.shape}")
    print(f"  Embedding shape B: {emb_b.shape}")

    # Combine embeddings
    combined = np.vstack([emb_a, emb_b])
    labels = [label_a] * len(emb_a) + [label_b] * len(emb_b)

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
    plt.figure(figsize=(10, 8))

    # Plot each group with different colors
    colors = {'A': 'blue', 'B': 'red'}
    for label_name in ['A', 'B']:
        mask = np.array([l == label_name for l in labels])
        plt.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            c=colors[label_name],
            label=label_a if label_name == 'A' else label_b,
            alpha=0.6,
            s=50
        )

    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title(label)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    filename = label.lower().replace(" ", "_").replace(":", "").replace("(", "").replace(")", "") + ".png"
    plt.savefig(filename, dpi=150)
    print(f"  Plot saved to: {filename}")

    # Visual assessment
    print("\n" + "-"*80)
    print("VISUAL ASSESSMENT:")
    print(f"  Check {filename} to see if {label_a} and {label_b} form distinct clusters")
    print(f"  - Overlapping clouds → Embeddings don't capture difference")
    print(f"  - Separate clusters → Embeddings capture semantic difference ✓")
    print("-"*80)

    return reduced, labels


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 1: t-SNE sanity check for EmbeddingGemma-300M"
    )
    parser.add_argument("--samples", type=int, default=100, help="Number of samples per distribution")
    parser.add_argument("--L", type=int, default=500, help="Completion length")
    parser.add_argument("--root_dir", default="./data", help="Dataset root directory")
    parser.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity")

    args = parser.parse_args()

    prompt_ids = {"wikipedia_en": [0, 1, 2]}
    model = "meta-llama/Meta-Llama-3-8B-Instruct"

    print("\n" + "="*80)
    print("EXPERIMENT 1: EmbeddingGemma-300M t-SNE Sanity Check")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {model}")
    print(f"  Prompts: {prompt_ids}")
    print(f"  Samples per distribution: {args.samples}")
    print(f"  Completion length: {args.L}")
    print(f"  t-SNE perplexity: {args.perplexity}")

    # Test 1: Sanity check - fp32 vs fp32 (should overlap)
    print("\n" + "="*80)
    print("Loading distributions for Test 1: SANITY CHECK (fp32 vs fp32)")
    print("="*80)

    with Stopwatch() as sw:
        samp_fp32_1 = sample_distribution(model, prompt_ids, args.L, "fp32", args.samples, args.root_dir)
        samp_fp32_2 = sample_distribution(model, prompt_ids, args.L, "fp32", args.samples, args.root_dir)
    print(f"Loaded in {sw.time:.2f}s")

    run_tsne_test(
        label="Test 1: Sanity Check (fp32 vs fp32)",
        samples_a=samp_fp32_1,
        samples_b=samp_fp32_2,
        label_a="fp32 (sample 1)",
        label_b="fp32 (sample 2)",
        perplexity=args.perplexity,
    )

    # Test 2: fp32 vs int8 (should separate if EmbeddingGemma captures quantization)
    print("\n" + "="*80)
    print("Loading distributions for Test 2: QUANTIZATION (fp32 vs int8)")
    print("="*80)

    with Stopwatch() as sw:
        samp_int8 = sample_distribution(model, prompt_ids, args.L, "int8", args.samples, args.root_dir)
    print(f"Loaded in {sw.time:.2f}s")

    run_tsne_test(
        label="Test 2: Quantization Detection (fp32 vs int8)",
        samples_a=samp_fp32_1,  # Reuse first fp32 sample
        samples_b=samp_int8,
        label_a="fp32",
        label_b="int8",
        perplexity=args.perplexity,
    )

    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT 1 COMPLETE")
    print("="*80)
    print("\nGenerated plots:")
    print("  1. test_1_sanity_check_fp32_vs_fp32.png")
    print("  2. test_2_quantization_detection_fp32_vs_int8.png")
    print("\nNext steps:")
    print("  - Examine the plots visually")
    print("  - If Test 1 shows overlap ✓ and Test 2 shows separation ✓:")
    print("    → EmbeddingGemma is superior to MPNet!")
    print("    → Proceed to Experiment 2 (quantum metrics)")
    print("  - If Test 2 shows overlap ✗:")
    print("    → EmbeddingGemma has same limitations as MPNet")
    print("    → Consider alternative approaches")
    print("="*80)


if __name__ == "__main__":
    main()
