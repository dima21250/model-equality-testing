import numpy as np
import torch
from typing import Union, Tuple, List, Dict
from .registry import IMPLEMENTED_TESTS  # Import test registry
from model_equality_testing.distribution import (
    CompletionSample,
    DistributionFromDataset,
)
import tqdm
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from .features import get_vader_scores


def _plot_empirical_distribution(stats, ax=None, label="", **kwargs):
    """
    Given a numpy array of test stats, plots the empirical distribution using a histogram
    """
    if ax is None:
        plt.figure()
        ax = plt.gca()
    ax.hist(stats, bins="auto", **kwargs)  # Adjust bins as needed
    ax.set_xlabel("Test Statistic")
    ax.set_ylabel("Frequency")
    ax.set_title(
        f'Distribution of Test Statistic {"(" + label + ")" if len(label) else ""}'
    )
    return ax


##############################################
# Functions to simulate and compute p-values
##############################################


class EmpiricalPvalueCalculator:
    """
    Given an empirical sample of test statitics, provides a callable that returns the p-value of an observed test statistic.
    """

    def __init__(self, observed_stats: np.ndarray):
        """
        Args:
            observed_stats: a numpy array of shape (b,)
                where b is the number of bootstrap samples
        """
        self.stats = observed_stats

    def __call__(self, obs_stat: Union[float, np.ndarray, torch.Tensor]) -> float:
        # handle obs_stat: make sure it's a float
        if isinstance(obs_stat, (torch.Tensor, np.ndarray)):
            obs_stat = obs_stat.item()

        # compare to self.stats and average across the batch dimension (b)
        return np.mean((self.stats >= obs_stat), axis=0).item()


class AnalyticalKSPvalueCalculator:
    """
    Analytical KS p‑value calculator that uses a reference score distribution.
    """
    def __init__(self, reference_scores):
        self.reference_scores = reference_scores

    def __call__(self, obs_sample, feature_fn=get_vader_scores):
        obs_scores = feature_fn(obs_sample)
        # Classical two‑sample KS test yields a p‑value directly
        _, p_value = ks_2samp(self.reference_scores, obs_scores)
        return p_value


def one_sample_parametric_bootstrap_pvalue(
    null_dist: DistributionFromDataset,
    n: int,
    b=1000,
    plot=False,
    return_stats=False,
    stat_type="g_squared",
    **kwargs,
) -> Union[EmpiricalPvalueCalculator, Tuple[EmpiricalPvalueCalculator, np.ndarray]]:
    """
    Simulates the empirical distribution of the test statistic by repeatedly drawing samples
    from the null distribution and computing the test statistic.
    Args:
        null_dist: a distribution object from which to draw samples
        n: the size of the sample to take
        b: the number of times to draw samples and compute the test statistic
        plot: whether to plot the empirical distribution of the test statistics
        return_stats: whether to return the raw test statistics, in addition to
            the p-value calculator
        stat_type: the type of test statistic to compute as a string.
            Must be a key in IMPLEMENTED_TESTS
        **kwargs: additional arguments to pass to the test computation function
    """
    stats = []
    for _ in tqdm.tqdm(range(b), desc="Parametric bootstrap"):
        bootstrap_sample = null_dist.sample(n=n)
        stat = IMPLEMENTED_TESTS[stat_type](bootstrap_sample, null_dist, **kwargs)
        stats.append(stat)
    stats = np.array(stats)
    if stats.ndim == 1:
        stats = np.expand_dims(stats, 1)
    if stats.ndim == 2:
        stats = np.expand_dims(stats, 2)

    # plot the empirical distribution
    if plot:
        b, m, nstats = stats.shape
        assert m == 1, "Incorrect shape for plotting"
        for i in range(nstats):
            _plot_empirical_distribution(stats[:, :, i], label=f"{stat_type} dim {i}")

    get_pvalue = EmpiricalPvalueCalculator(stats)
    if return_stats:
        return get_pvalue, stats
    return get_pvalue


def two_sample_parametric_bootstrap_pvalue(
    null_dist: DistributionFromDataset,
    n1: int,
    n2: int,
    b=1000,
    plot=False,
    return_stats=False,
    stat_type="two_sample_L2",
    **kwargs,
) -> Union[EmpiricalPvalueCalculator, Tuple[EmpiricalPvalueCalculator, np.ndarray]]:
    """
    Simulates the empirical distribution of the test statistic by repeatedly drawing samples
    from the null distribution and computing the test statistic.
    Args:
        null_dist: a distribution object from which to draw samples
        n1: the size of the first sample to take
        n2: the size of the second sample to take
        b: the number of times to draw samples and compute the test statistic
        plot: whether to plot the empirical distribution of the test statistics
        return_stats: whether to return the raw test statistics, in addition to
            the p-value calculator
        stat_type: the type of test statistic to compute as a string.
            Must be a key in IMPLEMENTED_TESTS
        **kwargs: additional arguments to pass to the test computation function
    """
    stats = []
    for _ in tqdm.tqdm(range(b), desc="Parametric bootstrap"):
        sample1 = null_dist.sample(n=n1)
        sample2 = null_dist.sample(n=n2)
        stat = IMPLEMENTED_TESTS[stat_type](sample1, sample2, **kwargs)
        stats.append(stat)
    stats = np.array(stats)
    if stats.ndim == 1:
        stats = np.expand_dims(stats, 1)
    if stats.ndim == 2:
        stats = np.expand_dims(stats, 2)

    # plot the empirical distribution
    if plot:
        b, m, nstats = stats.shape
        assert m == 1, "Incorrect shape for plotting"
        for i in range(nstats):
            _plot_empirical_distribution(stats[:, :, i], label=f"{stat_type} dim {i}")

    get_pvalue = EmpiricalPvalueCalculator(stats)
    if return_stats:
        return get_pvalue, stats
    return get_pvalue


def two_sample_permutation_pvalue(
    sample1: CompletionSample,
    sample2: CompletionSample,
    b=1000,
    plot=False,
    return_stats=False,
    stat_type="two_sample_L2",
    **kwargs,
) -> Union[EmpiricalPvalueCalculator, Tuple[EmpiricalPvalueCalculator, np.ndarray]]:
    """
    Simulates the empirical distribution of the test statistic by repeatedly permuting the labels
    of the samples and computing the test statistic.
    Args:
        sample1: the first sample
        sample2: the second sample
        b: the number of times to draw samples and compute the test statistic
        plot: whether to plot the empirical distribution of the test statistics
        return_stats: whether to return the raw test statistics, in addition to
            the p-value calculator
        stat_type: the type of test statistic to compute as a string.
            Must be a key in IMPLEMENTED_TESTS
        **kwargs: additional arguments to pass to the test computation function
    """
    stats = []
    all_samples = torch.cat(
        [
            sample1.sequences,
            sample2.sequences,
        ],
        dim=0,
    )
    for _ in tqdm.tqdm(range(b), desc="Permutation bootstrap"):
        ix = torch.randperm(len(all_samples))
        permuted_sample1 = CompletionSample(
            prompts=all_samples[ix][: sample1.N, 0],
            completions=all_samples[ix][: sample1.N, 1:],
            m=sample1.m,
        )
        permuted_sample2 = CompletionSample(
            prompts=all_samples[ix][sample1.N :, 0],
            completions=all_samples[ix][sample1.N :, 1:],
            m=sample1.m,
        )

        stat = IMPLEMENTED_TESTS[stat_type](
            permuted_sample1, permuted_sample2, **kwargs
        )
        stats.append(stat)

    stats = np.array(stats)
    if stats.ndim == 1:
        stats = np.expand_dims(stats, 1)
    if stats.ndim == 2:
        stats = np.expand_dims(stats, 2)

    # plot the empirical distribution
    if plot:
        b, m, nstats = stats.shape
        assert m == 1, "Incorrect shape for plotting"
        for i in range(nstats):
            _plot_empirical_distribution(stats[:, :, i], label=f"{stat_type} dim {i}")

    get_pvalue = EmpiricalPvalueCalculator(stats)
    if return_stats:
        return get_pvalue, stats
    del stats
    return get_pvalue


QUANTUM_STAT_TYPES = frozenset({
    "quantum_trace_distance",
    "quantum_von_neumann_divergence",
    "quantum_relative_entropy",
})


def two_sample_embedding_permutation_pvalue(
    sample1: CompletionSample,
    sample2: CompletionSample,
    b=1000,
    plot=False,
    return_stats=False,
    stat_type="quantum_von_neumann_divergence",
    **kwargs,
) -> Union[EmpiricalPvalueCalculator, Tuple[EmpiricalPvalueCalculator, np.ndarray]]:
    """Permutation test optimized for embedding-based quantum metrics.

    Pre-computes SBERT embeddings once for the combined sample pool, then
    permutes embedding indices on each iteration instead of re-embedding.
    When pca_k is set, PCA is fit once on the combined pool so all
    permutations share the same basis.

    Args:
        sample1: First CompletionSample
        sample2: Second CompletionSample
        b: Number of permutations
        plot: Whether to plot the empirical distribution
        return_stats: Whether to return raw statistics
        stat_type: Must be a quantum stat type
        **kwargs: Passed to the test function (embedding_model, batch_size, pca_k, etc.)
    """
    from .embeddings import embed_sample

    embedding_model = kwargs.pop("embedding_model", "all-mpnet-base-v2")
    batch_size = kwargs.pop("batch_size", 32)
    pca_k = kwargs.get("pca_k", 50)

    # Embed all samples ONCE
    embeddings1 = embed_sample(sample1, model_name=embedding_model, batch_size=batch_size)
    embeddings2 = embed_sample(sample2, model_name=embedding_model, batch_size=batch_size)
    all_embeddings = np.concatenate([embeddings1, embeddings2], axis=0)
    n1 = len(embeddings1)

    # Fit PCA once on the combined pool so all permutations share the same basis
    fitted_pca = None
    if pca_k is not None:
        from .quantum_metrics import fit_pca
        fitted_pca = fit_pca(all_embeddings, k=pca_k)

    stats = []
    for _ in tqdm.tqdm(range(b), desc="Permutation bootstrap"):
        ix = np.random.permutation(len(all_embeddings))
        perm_emb1 = all_embeddings[ix[:n1]]
        perm_emb2 = all_embeddings[ix[n1:]]

        stat = IMPLEMENTED_TESTS[stat_type](
            sample1, sample2,
            _precomputed_embeddings=(perm_emb1, perm_emb2),
            _fitted_pca=fitted_pca,
            **kwargs,
        )
        stats.append(stat)

    stats = np.array(stats)
    if stats.ndim == 1:
        stats = np.expand_dims(stats, 1)
    if stats.ndim == 2:
        stats = np.expand_dims(stats, 2)

    if plot:
        b_len, m, nstats = stats.shape
        assert m == 1, "Incorrect shape for plotting"
        for i in range(nstats):
            _plot_empirical_distribution(stats[:, :, i], label=f"{stat_type} dim {i}")

    get_pvalue = EmpiricalPvalueCalculator(stats)
    if return_stats:
        return get_pvalue, stats
    return get_pvalue


###### map from name to function ######

IMPLEMENTED_PVALUES = {
    "one_sample_parametric_bootstrap": one_sample_parametric_bootstrap_pvalue,
    "two_sample_parametric_bootstrap": two_sample_parametric_bootstrap_pvalue,
    "two_sample_permutation": two_sample_permutation_pvalue,
    "two_sample_embedding_permutation": two_sample_embedding_permutation_pvalue,
}
