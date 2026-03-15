"""Analyze which prompts give the best KS test sensitivity.

For a given dataset and source comparison, test each prompt individually
to find which prompts maximize the KS statistic and minimize the p-value.
"""

import argparse
from typing import Dict, List
import numpy as np
from scipy.stats import ks_2samp
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from model_equality_testing.dataset import load_distribution


def analyze_prompt_sensitivity(
    model: str,
    dataset: str,
    source_a: str,
    source_b: str,
    num_prompts: int,
    L: int,
    n_samples: int,
    root_dir: str = "./data",
):
    """Test each prompt individually to find best KS sensitivity."""

    analyzer = SentimentIntensityAnalyzer()

    def get_vader_scores(sample):
        """Extract VADER compound scores from a sample."""
        data = sample.completion_sample.cpu().numpy()
        scores = []
        for row in data:
            text = "".join([chr(c) for c in row if c != -1])
            scores.append(analyzer.polarity_scores(text)["compound"])
        return scores

    results = []

    print(f"Testing prompts 0-{num_prompts-1} individually for {dataset}")
    print(f"Source A: {source_a}, Source B: {source_b}\n")

    for prompt_id in range(num_prompts):
        prompt_ids = {dataset: [prompt_id]}

        # Load distributions for this single prompt
        dist_a = load_distribution(
            model=model,
            prompt_ids=prompt_ids,
            L=L,
            source=source_a,
            root_dir=root_dir,
        )
        dist_b = load_distribution(
            model=model,
            prompt_ids=prompt_ids,
            L=L,
            source=source_b,
            root_dir=root_dir,
        )

        # Sample from both distributions
        samp_a = dist_a.sample(n=n_samples)
        samp_b = dist_b.sample(n=n_samples)

        # Get VADER sentiment scores
        scores_a = get_vader_scores(samp_a)
        scores_b = get_vader_scores(samp_b)

        # Compute KS statistic
        ks_res = ks_2samp(scores_a, scores_b)

        # Compute sentiment statistics
        mean_a = np.mean(scores_a)
        std_a = np.std(scores_a)
        mean_b = np.mean(scores_b)
        std_b = np.std(scores_b)

        results.append({
            'prompt_id': prompt_id,
            'ks_stat': ks_res.statistic,
            'ks_pvalue': ks_res.pvalue,
            'mean_a': mean_a,
            'std_a': std_a,
            'mean_b': mean_b,
            'std_b': std_b,
            'mean_diff': abs(mean_a - mean_b),
            'detected': ks_res.pvalue < 0.05,
        })

        print(f"Prompt {prompt_id:2d}: KS={ks_res.statistic:.4f}, p={ks_res.pvalue:.4f}, "
              f"mean_a={mean_a:+.3f}±{std_a:.3f}, mean_b={mean_b:+.3f}±{std_b:.3f}, "
              f"{'✅ DETECTED' if ks_res.pvalue < 0.05 else '❌ not detected'}")

    # Sort by KS statistic (descending)
    results_sorted = sorted(results, key=lambda x: x['ks_stat'], reverse=True)

    print("\n" + "="*80)
    print("TOP 5 PROMPTS BY KS STATISTIC:")
    print("="*80)
    for i, r in enumerate(results_sorted[:5], 1):
        print(f"{i}. Prompt {r['prompt_id']}: KS={r['ks_stat']:.4f}, p={r['ks_pvalue']:.4f}, "
              f"mean_diff={r['mean_diff']:.4f}")

    print("\n" + "="*80)
    print("PROMPTS WITH SIGNIFICANT DETECTION (p < 0.05):")
    print("="*80)
    detected = [r for r in results if r['detected']]
    if detected:
        for r in sorted(detected, key=lambda x: x['ks_pvalue']):
            print(f"Prompt {r['prompt_id']}: KS={r['ks_stat']:.4f}, p={r['ks_pvalue']:.4f}")
    else:
        print("None detected at α=0.05")

    print("\n" + "="*80)
    print("PROMPTS WITH HIGHEST SENTIMENT VARIANCE (std dev):")
    print("="*80)
    by_variance = sorted(results, key=lambda x: max(x['std_a'], x['std_b']), reverse=True)
    for i, r in enumerate(by_variance[:5], 1):
        max_std = max(r['std_a'], r['std_b'])
        print(f"{i}. Prompt {r['prompt_id']}: max_std={max_std:.4f}, KS={r['ks_stat']:.4f}")

    print("\n" + "="*80)
    print("RECOMMENDED PROMPT COMBINATION:")
    print("="*80)
    # Recommend top 3 by KS statistic
    top_3 = [r['prompt_id'] for r in results_sorted[:3]]
    print(f"Use prompts: {top_3}")
    print(f"\nCommand:")
    print(f"python mmd-ks-comparison.py \\")
    print(f"  --model_a {model} \\")
    print(f"  --model_b {model} \\")
    print(f"  --source_a {source_a} \\")
    print(f"  --source_b {source_b} \\")
    print(f"  --dataset {dataset} \\")
    print(f"  --prompts {' '.join(map(str, top_3))} \\")
    print(f"  --L {L} \\")
    print(f"  --samples {n_samples}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze which prompts maximize KS test sensitivity"
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Model name"
    )
    parser.add_argument(
        "--dataset",
        default="wikipedia_en",
        help="Dataset name"
    )
    parser.add_argument(
        "--source_a",
        default="fp32",
        help="First source"
    )
    parser.add_argument(
        "--source_b",
        default="watermark",
        help="Second source"
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=20,
        help="Number of prompts to test (0 to num_prompts-1)"
    )
    parser.add_argument(
        "--L",
        type=int,
        default=200,
        help="Completion length"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Number of samples per prompt"
    )
    parser.add_argument(
        "--root_dir",
        default="./data",
        help="Data directory"
    )

    args = parser.parse_args()

    analyze_prompt_sensitivity(
        model=args.model,
        dataset=args.dataset,
        source_a=args.source_a,
        source_b=args.source_b,
        num_prompts=args.num_prompts,
        L=args.L,
        n_samples=args.samples,
        root_dir=args.root_dir,
    )


if __name__ == "__main__":
    main()
