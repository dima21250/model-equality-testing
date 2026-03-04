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
    root_dir: str = "./data",
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
    root_dir: str
        Directory where the dataset was extracted (default ``./data``).
    """
    dist = load_distribution(
        model=model,
        prompt_ids=prompt_ids,
        L=L,
        source=source,
        root_dir=root_dir,
    )
    return dist.sample(n=n_samples)


def compute_mmd(sample1, sample2):
    """MMD statistic (hamming kernel) and permutation p‑value, with timing.

    Returns a tuple ``(stat, pvalue, elapsed)`` where ``elapsed`` is the total
    wall‑clock time (seconds) spent computing the statistic and the p‑value.
    """
    from model_equality_testing.utils import Stopwatch
    with Stopwatch() as sw:
        stat = mmd_hamming(sample1, sample2)
        pvalue, _ = run_two_sample_test(
            sample1,
            sample2,
            stat_type="mmd_hamming",
            pvalue_type="permutation_pvalue",
        )
    return stat, pvalue, sw.time


def compute_ks(sample1, sample2):
    """Two‑sample KS statistic using VADER sentiment scores, with timing.

    Returns ``(statistic, pvalue, elapsed)`` where ``elapsed`` includes the
    feature extraction (sentiment scoring) and the call to ``ks_2samp``.
    """
    from model_equality_testing.utils import Stopwatch
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

    with Stopwatch() as sw:
        scores1 = get_scores(sample1)
        scores2 = get_scores(sample2)
        ks_res = ks_2samp(scores1, scores2)
    return ks_res.statistic, ks_res.pvalue, sw.time


def run_pair(
    model_a: str,
    model_b: str,
    prompt_ids: Dict[str, List[int]],
    L: int,
    source: str,
    n_samples: int,
    label: str,
    root_dir: str = "./data",
):
    print(f"{label} (model A: {model_a} vs model B: {model_b}, source: {source}):")
    samp_a = sample_distribution(model_a, prompt_ids, L, source, n_samples, root_dir=root_dir)
    samp_b = sample_distribution(model_b, prompt_ids, L, source, n_samples, root_dir=root_dir)

    mmd_stat, mmd_p, mmd_time = compute_mmd(samp_a, samp_b)
    ks_stat, ks_p, ks_time = compute_ks(samp_a, samp_b)

    print(f"  MMD statistic = {mmd_stat:.6f}, p‑value = {mmd_p:.4f}, elapsed = {mmd_time:.3f}s")
    print(f"  KS  statistic = {ks_stat:.6f}, p‑value = {ks_p:.4f}, elapsed = {ks_time:.3f}s\n")


def main():
    parser = argparse.ArgumentParser(description="Compare two LLMs on the same dataset using MMD and VADER‑based KS tests.")
    parser.add_argument("--model_a", default="meta-llama/Meta-Llama-3-8B-Instruct", help="First model name for the comparison.")
    parser.add_argument("--model_b", default="mistralai/Mistral-7B-Instruct-v0.3", help="Second model name for the comparison.")
    parser.add_argument("--prompts", nargs="+", default=["0", "1", "2"], help="Space‑separated list of prompt IDs (as strings).")
    parser.add_argument("--L", type=int, default=200, help="Maximum completion length (truncation).")
    parser.add_argument("--samples", type=int, default=500, help="Number of completions to draw per model.")
    parser.add_argument("--source", default="fp32", help="Source identifier (e.g., fp32, int8, ...) to use for both models.")
    parser.add_argument("--root_dir", default="./data", help="Directory where the dataset was extracted.")
    parser.add_argument("--dataset", default="wikipedia_en", help="Dataset name present in the data folder (e.g., wikipedia_en).")
    args = parser.parse_args()

    # Build prompt mapping – use the user‑provided dataset name.
    prompt_ids = {args.dataset: [int(pid) for pid in args.prompts]}

    # Single comparison between the two models on the same source.
    run_pair(
        model_a=args.model_a,
        model_b=args.model_b,
        prompt_ids=prompt_ids,
        L=args.L,
        source=args.source,
        n_samples=args.samples,
        label="Model comparison",
        root_dir=args.root_dir,
    )


if __name__ == "__main__":
    main()
