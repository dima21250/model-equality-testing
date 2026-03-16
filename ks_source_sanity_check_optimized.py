#!/usr/bin/env python3
"""
KS sanity check for all wikipedia_en prompts (0-99) with identical model/source
across fp32, nf4, int8, watermark.
Optimized to load distribution once per (prompt, source) pair.
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


def load_distribution_once(
    model: str,
    prompt_ids: Dict[str, List[int]],
    L: int,
    source: str,
    root_dir: str = "./data",
):
    """Load a distribution (without sampling)."""
    return load_distribution(
        model=model,
        prompt_ids=prompt_ids,
        L=L,
        source=source,
        root_dir=root_dir,
    )


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
        description="Run KS sanity checks for identical model/source across all wikipedia_en prompts and quantizations."
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
        "--start-prompt",
        type=int,
        default=0,
        help="Start of prompt ID range (inclusive).",
    )
    parser.add_argument(
        "--end-prompt",
        type=int,
        default=99,
        help="End of prompt ID range (inclusive).",
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
        default=5,
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

    # Write header
    with open(args.output, "w") as f:
        f.write("# KS Sanity Check Results (All Prompts)\n\n")
        f.write(f"**Model:** {args.model}\n")
        f.write(f"**Dataset:** {args.dataset}\n")
        f.write(f"**Prompt IDs:** {args.start_prompt}–{args.end_prompt} (total {args.end_prompt - args.start_prompt + 1})\n")
        f.write(f"**Sources:** {', '.join(args.sources)}\n")
        f.write(f"**Completion length (L):** {args.L}\n")
        f.write(f"**Samples per comparison:** {args.samples}\n\n")
        f.write("| Prompt ID | Source | KS statistic | p-value | Note |\n")
        f.write("|-----------|--------|--------------|---------|------|\n")

    any_low_p = False

    # Open file for appending rows
    with open(args.output, "a") as f:
        for prompt_id in range(args.start_prompt, args.end_prompt + 1):
            prompt_ids = {args.dataset: [prompt_id]}
            for source in args.sources:
                # Load distribution once for this (prompt, source)
                dist = load_distribution_once(
                    model=args.model,
                    prompt_ids=prompt_ids,
                    L=args.L,
                    source=source,
                    root_dir=args.root_dir,
                )
                # Sample two independent draws
                samp_a = dist.sample(n=args.samples)
                samp_b = dist.sample(n=args.samples)
                ks_stat, ks_p, elapsed = compute_ks(samp_a, samp_b)
                note = ""
                if ks_p < 0.05:
                    note = "⚠️ Low p-value (possible false rejection)"
                    any_low_p = True
                f.write(f"| {prompt_id} | {source} | {ks_stat:.6f} | {ks_p:.6f} | {note} |\n")

    # Write footer
    with open(args.output, "a") as f:
        f.write("\n")
        if any_low_p:
            f.write("## ⚠️ Warning: Some comparisons yielded p < 0.05\n")
            f.write("This may indicate false rejections of the null hypothesis.\n")
            f.write("Consider increasing `--samples` or checking random seeds.\n")
        else:
            f.write("## ✅ All KS tests produced p ≥ 0.05\n")
            f.write("No false rejections detected in this sanity check.\n")

    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
