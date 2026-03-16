#!/usr/bin/env python3
"""
KS-only sanity check: compare identical model/source pairs for each wikipedia_en prompt
across fp32, nf4, int8, watermark to ensure no false rejections of the null hypothesis.
"""

import argparse
import os
from typing import Dict, List

import torch
from scipy.stats import ks_2samp

from model_equality_testing.dataset import load_distribution
from model_equality_testing.utils import Stopwatch

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError as e:
    raise ImportError("Please install vaderSentiment to use VADER‑based KS test: pip install vaderSentiment") from e


def sample_distribution(
    model: str,
    prompt_ids: Dict[str, List[int]],
    L: int,
    source: str,
    n_samples: int,
    root_dir: str = "./data",
):
    """Load a distribution and draw ``n_samples`` completions."""
    dist = load_distribution(
        model=model,
        prompt_ids=prompt_ids,
        L=L,
        source=source,
        root_dir=root_dir,
    )
    return dist.sample(n=n_samples)


def get_vader_scores(sample):
    """Return list of VADER compound scores for a CompletionSample."""
    analyzer = SentimentIntensityAnalyzer()
    data = sample.completion_sample.cpu().numpy()
    scores = []
    for row in data:
        text = "".join([chr(c) for c in row if c != -1])
        scores.append(analyzer.polarity_scores(text)["compound"])
    return scores


def compute_ks(sample1, sample2):
    """Two‑sample KS statistic using VADER sentiment scores, with timing."""
    with Stopwatch() as sw:
        scores1 = get_vader_scores(sample1)
        scores2 = get_vader_scores(sample2)
        ks_res = ks_2samp(scores1, scores2)
    return ks_res.statistic, ks_res.pvalue, sw.time


def main():
    parser = argparse.ArgumentParser(
        description="Run KS sanity checks for identical model/source across prompts and quantizations."
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Model identifier (same for both sides).",
    )
    parser.add_argument(
        "--dataset",
        default="wikipedia_en",
        help="Dataset name present in the data folder.",
    )
    parser.add_argument(
        "--prompts",
        type=int,
        nargs="+",
        default=list(range(0, 100)),
        help="List of prompt IDs to test (default 0-99).",
    )
    parser.add_argument(
        "--sources",
        type=str,
        nargs="+",
        default=["fp32", "nf4", "int8", "watermark"],
        help="Quantization/source identifiers to test.",
    )
    parser.add_argument(
        "--L",
        type=int,
        default=200,
        help="Maximum completion length (truncation).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Number of completions to draw per model.",
    )
    parser.add_argument(
        "--root_dir",
        default="./data",
        help="Directory where the dataset was extracted.",
    )
    parser.add_argument(
        "--output",
        default="ks-source-sanity-check.md",
        help="Markdown file to write results.",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) if os.path.dirname(args.output) else ".", exist_ok=True)

    # Prepare markdown content
    lines = []
    lines.append("# KS Sanity Check Results")
    lines.append("")
    lines.append(f"**Model:** {args.model}")
    lines.append(f"**Dataset:** {args.dataset}")
    lines.append(f"**Prompt IDs:** {args.prompts[0]}–{args.prompts[-1]} (total {len(args.prompts)})")
    lines.append(f"**Sources:** {', '.join(args.sources)}")
    lines.append(f"**Completion length (L):** {args.L}")
    lines.append(f"**Samples per comparison:** {args.samples}")
    lines.append("")
    lines.append("| Prompt ID | Source | KS statistic | p-value | Note |")
    lines.append("|-----------|--------|--------------|---------|------|")

    any_low_p = False

    for prompt_id in args.prompts:
        prompt_ids = {args.dataset: [prompt_id]}
        for source in args.sources:
            # Sample two independent draws from same distribution
            samp_a = sample_distribution(
                model=args.model,
                prompt_ids=prompt_ids,
                L=args.L,
                source=source,
                n_samples=args.samples,
                root_dir=args.root_dir,
            )
            samp_b = sample_distribution(
                model=args.model,
                prompt_ids=prompt_ids,
                L=args.L,
                source=source,
                n_samples=args.samples,
                root_dir=args.root_dir,
            )
            ks_stat, ks_p, elapsed = compute_ks(samp_a, samp_b)
            note = ""
            if ks_p < 0.05:
                note = "⚠️ Low p-value (possible false rejection)"
                any_low_p = True
            lines.append(f"| {prompt_id} | {source} | {ks_stat:.6f} | {ks_p:.6f} | {note} |")

    lines.append("")
    if any_low_p:
        lines.append("## ⚠️ Warning: Some comparisons yielded p < 0.05")
        lines.append("This may indicate false rejections of the null hypothesis.")
        lines.append("Consider increasing `--samples` or checking random seeds.")
    else:
        lines.append("## ✅ All KS tests produced p ≥ 0.05")
        lines.append("No false rejections detected in this sanity check.")

    with open(args.output, "w") as f:
        f.write("\n".join(lines))

    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
