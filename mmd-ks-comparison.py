'''mmd-ks-comparison.py

Demonstrates a sanity‑check using both the MMD (hamming kernel) and the two‑sample
Kolmogorov‑Smirnov test on model output distributions.

The script loads two distributions using the utilities provided in the package:
* ``load_distribution`` – loads a ``DistributionFromDataset`` for a given model,
  prompt set and source (e.g. ``fp32`` or ``int8``).
* ``DistributionFromDataset.sample`` – draws i.i.d. completions.
* ``run_two_sample_test`` – permutation p‑value for the MMD statistic.
* ``scipy.stats.ks_2samp`` – analytical p‑value for the KS statistic.

It prints a concise summary for an *equivalence* case (same source) and a
*difference* case (different source).
'''

import argparse
from typing import Dict, List

import torch
from scipy.stats import ks_2samp

from model_equality_testing.dataset import load_distribution
from model_equality_testing.algorithm import run_two_sample_test
from model_equality_testing.tests import mmd_hamming


def sample_distribution(
    model: str,
    prompt_ids: Dict[str, List[int]],
    L: int,
    source: str,
    n_samples: int,
) -> "CompletionSample":
    """Load a distribution and draw ``n_samples`` completions.

    Parameters
    ----------
    model: str
        Model identifier compatible with the HuggingFace tokenizer.
    prompt_ids: dict
        Mapping of dataset name to a list of prompt IDs.
    L: int
        Completion length (truncation length).
    source: str
        One of the supported ``SOURCES`` (e.g. ``fp32`` or ``int8``).
    n_samples: int
        Number of completions to draw.
    """
    dist = load_distribution(
        model=model,
        prompt_ids=prompt_ids,
        L=L,
        source=source,
    )
    return dist.sample(n=n_samples)


def compute_mmd(sample1, sample2):
    """MMD statistic (hamming kernel) and permutation p‑value."""
    stat = mmd_hamming(sample1, sample2)
    pvalue, _ = run_two_sample_test(
        sample1,
        sample2,
        stat_type="mmd_hamming",
        pvalue_type="permutation_pvalue",
    )
    return stat, pvalue


def compute_ks(sample1, sample2):
    """Two‑sample KS statistic using VADER sentiment scores.

    This mirrors ``tests.two_sample_vader_ks`` for the statistic, but also
    returns the analytical p‑value obtained from ``scipy.stats.ks_2samp`` on the
    sentiment scores.
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError as e:
        raise ImportError("Please install vaderSentiment to use VADER‑based KS test: pip install vaderSentiment") from e

    analyzer = SentimentIntensityAnalyzer()

    def get_scores(sample):
        # Decode unicode integers back to a string (ignoring padding -1)
        data = sample.completion_sample.cpu().numpy()
        scores = []
        for row in data:
            text = "".join([chr(c) for c in row if c != -1])
            scores.append(analyzer.polarity_scores(text)["compound"])
        return scores

    scores1 = get_scores(sample1)
    scores2 = get_scores(sample2)
    ks_res = ks_2samp(scores1, scores2)
    return ks_res.statistic, ks_res.pvalue


def run_pair(
    model: str,
    prompt_ids: Dict[str, List[int]],
    L: int,
    source_a: str,
    source_b: str,
    n_samples: int,
    label: str,
):
    print(f"{label} (source: {source_a} vs {source_b}):")
    samp_a = sample_distribution(model, prompt_ids, L, source_a, n_samples)
    samp_b = sample_distribution(model, prompt_ids, L, source_b, n_samples)

    mmd_stat, mmd_p = compute_mmd(samp_a, samp_b)
    ks_stat, ks_p = compute_ks(samp_a, samp_b)

    print(f"  MMD statistic = {mmd_stat:.6f}, p‑value = {mmd_p:.4f}")
    print(f"  KS  statistic = {ks_stat:.6f}, p‑value = {ks_p:.4f}\n")


def main():
    parser = argparse.ArgumentParser(description="Compare MMD and KS tests on two model output distributions.")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model name for the tokenizer.")
    parser.add_argument("--prompts", nargs="+", default=["0", "1", "2"], help="Space‑separated list of prompt IDs (as strings).")
    parser.add_argument("--L", type=int, default=200, help="Maximum completion length (truncation).")
    parser.add_argument("--samples", type=int, default=500, help="Number of completions to draw per distribution.")
    parser.add_argument("--source_eq", default="fp32", help="Source for the equivalence pair (both sides).")
    parser.add_argument("--source_diff", default="int8", help="Source for the differing pair (second side).")
    args = parser.parse_args()

    # Build prompt mapping – use a single synthetic dataset name for simplicity.
    prompt_ids = {"synthetic": [int(pid) for pid in args.prompts]}

    # Equivalence sanity check (same source).
    run_pair(
        model=args.model,
        prompt_ids=prompt_ids,
        L=args.L,
        source_a=args.source_eq,
        source_b=args.source_eq,
        n_samples=args.samples,
        label="Equivalence check",
    )

    # Difference check (different sources).
    run_pair(
        model=args.model,
        prompt_ids=prompt_ids,
        L=args.L,
        source_a=args.source_eq,
        source_b=args.source_diff,
        n_samples=args.samples,
        label="Difference check",
    )


if __name__ == "__main__":
    main()
