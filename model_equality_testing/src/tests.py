"""
Functions to compute test statistics given sample(s) and null distribution(s).
"""

import numpy as np
from model_equality_testing.utils import (
    Stopwatch,
    get_inv,
)
from model_equality_testing.src.features import get_vader_scores, get_perplexity_scores
import torch
from typing import Union, Tuple, List, Dict
from model_equality_testing.distribution import (
    CompletionSample,
    DistributionFromDataset,
)
from functools import lru_cache
from collections import Counter
from scipy.stats import ks_2samp

#######################
# Two sample tests
#######################

#### MMD tests ####


def _mmd(
    X: np.ndarray,
    Y: np.ndarray,
    get_kernel: callable,
    normalize=True,
    print_info=False,
) -> float:
    """
    Helper function to compute MMD test statistic.
    Handles normalization.
    Args:
        X: (n, L+1) numpy array of n sequences of length L.
            The first column X[:, 0] is an integer indicating the prompt.
        Y: (m, L+1) numpy array of m sequences of length L.
            The first column Y[:, 0] is an integer indicating the prompt.
        get_kernel: function that computes the kernel matrices K_XX, K_XY, K_YY given X, Y
        normalize: whether to normalize the kernel matrices
        print_info: whether to print time to compute kernels
    Returns:
        MMD test statistic
    """
    # Create a mask that is True if the prompts are different
    # When computing the kernel, we will zero out the entries where the mask is True
    # since we define the kernel to be 0 when the prompts are different
    prompts_x, prompts_y = X[:, 0], Y[:, 0]
    mask_XY = prompts_x[:, None] != prompts_y[None, :]
    mask_XX = prompts_x[:, None] != prompts_x[None, :]
    mask_YY = prompts_y[:, None] != prompts_y[None, :]

    # Call get_kernel to compute the kernel matrices
    with Stopwatch() as sw:
        K_XX, K_XY, K_YY = get_kernel(
            X[:, 1:], Y[:, 1:], mask_XX, mask_XY, mask_YY
        )  # remove prompt from seq
    if print_info:
        print("Time to compute kernels", sw.time)
    n_XX, n_XY, n_YY = K_XX.size, K_XY.size, K_YY.size

    # Zero out sequences from different prompts according to the mask
    K_XY[mask_XY] = 0
    n_XY -= mask_XY.sum()
    K_XX[mask_XX] = 0
    n_XX -= mask_XX.sum()
    K_YY[mask_YY] = 0
    n_YY -= mask_YY.sum()

    # Normalize the kernel matrices s.t. diagonal is 1
    if normalize:
        # kernel'[x, y] = kernel[x, y] / sqrt(kernel[x, x] * kernel[y, y])
        diagX = np.sqrt(np.diag(K_XX))
        diagY = np.sqrt(np.diag(K_YY))
        diagX[diagX == 0] = 1
        diagY[diagY == 0] = 1
        K_XX /= np.outer(diagX, diagX)
        K_YY /= np.outer(diagY, diagY)
        K_XY /= np.outer(diagX, diagY)

    # Zero out samples with themselves
    np.fill_diagonal(K_XX, 0)
    n_XX -= len(K_XX)
    np.fill_diagonal(K_YY, 0)
    n_YY -= len(K_YY)

    # Compute empirical MMD estimate
    return np.sum(K_XX) / n_XX - 2 * np.sum(K_XY) / n_XY + np.sum(K_YY) / n_YY


def _reflect_upper_triangular(K: np.ndarray) -> np.ndarray:
    """
    Helper function to reflect the upper triangular part of a matrix to the lower triangular part.
    Args:
        K: (n, n) numpy array with the diagonal + upper right part filled in
    Returns:
        (n, n) numpy array with the diagonal + upper right part filled in, and the lower left part
        filled in by reflecting the upper right part
    """
    np.fill_diagonal(K, K.diagonal() / 2)
    return K + K.T


def mmd_hamming(
    sample1: CompletionSample,
    sample2: CompletionSample,
) -> float:
    """
    MMD test statistic using K(x, y) = sum_i^L 1[x_i == y_i],
    i.e. whether the marginal densities match
    """

    def get_hamming_kernel(
        X: np.ndarray, Y: np.ndarray, *args, memory_threshold: int = 10000, **kwargs
    ):
        """
        Args:
            X: (n, L) numpy array of n sequences of length L
            Y: (m, L) numpy array of m sequences of length L
        Returns:
            K(X, X), K(X, Y), K(Y, Y) as a tuple
        """
        n, L = X.shape
        m, _ = Y.shape
        max_size_XX = n * n
        max_size_XY = n * m
        max_size_YY = m * m
        if max(max_size_XX, max_size_XY, max_size_YY) <= memory_threshold**2:
            K_XX = np.sum(X[:, None, :] == X[None, :, :], axis=-1).astype(float)
            K_XY = np.sum(X[:, None, :] == Y[None, :, :], axis=-1).astype(float)
            K_YY = np.sum(Y[:, None, :] == Y[None, :, :], axis=-1).astype(float)
        else:
            print("To save memory, computing Hamming using for loops")
            K_XX = np.zeros((n, n), dtype=float)
            K_XY = np.zeros((n, m), dtype=float)
            K_YY = np.zeros((m, m), dtype=float)
            for i in range(n):
                for j in range(i, n):
                    K_XX[i, j] = K_XX[j, i] = np.sum(X[i] == X[j])
            for i in range(n):
                for j in range(m):
                    K_XY[i, j] = K_XY[j, i] = np.sum(X[i] == Y[j])
            for i in range(m):
                for j in range(i, m):
                    K_YY[i, j] = K_YY[j, i] = np.sum(Y[i] == Y[j])
        return K_XX, K_XY, K_YY

    return _mmd(
        X=sample1.sequences.numpy(),
        Y=sample2.sequences.numpy(),
        get_kernel=get_hamming_kernel,
    )


@lru_cache(maxsize=10000)
def _get_kgrams(input_list: List[tuple], k: int, cumulative: bool = False) -> Counter:
    """
    Given a list of sequences, returns a Counter of all k-grams in the sequences.
    Args:
        input_list: tuple of sequences
        k: length of k-grams
        cum: whether to return all k-grams up to length k or just k-grams of length k
    """
    out = Counter()
    for ki in range(1, k + 1) if cumulative else [k]:
        out.update(zip(*(input_list[i:] for i in range(ki))))
    return out


@lru_cache(maxsize=100000)
def _compute_dot_product_counts(
    a: List[tuple], b: List[tuple], k: int, cumulative: bool
) -> int:
    r"""
    Given two sequences, computes the dot product of the counts of k-grams in the two sequences.
    $$
    \sum_{s \in a} \#(s \in a) \#(s \in b)
    $$
    Args:
        a: sequence
    """
    d1 = _get_kgrams(a, k, cumulative)
    d2 = _get_kgrams(b, k, cumulative)
    return sum(d1[key] * d2.get(key, 0) for key in d1)


def mmd_kspectrum(
    sample1: CompletionSample,
    sample2: CompletionSample,
    k: int = 5,
    cumulative: bool = True,
):
    r"""
    MMD test statistic using K(x, y) = \sum_{len-k substrings of len L} #(s in x) #(s in y)
    Args:
        sample1: CompletionSample
        sample2: CompletionSample
        k: length of k-grams
        cumulative: whether to use all k-grams up to length k or just k-grams of length k
    """

    def get_kspectrum_kernel(
        X: np.ndarray,
        Y: np.ndarray,
        mask_XX: np.ndarray = None,
        mask_XY: np.ndarray = None,
        mask_YY: np.ndarray = None,
    ):
        """
        Args:
            X: (n, L) numpy array of n sequences of length L
            Y: (m, L) numpy array of m sequences of length L
        Returns:
            K(X, X), K(X, Y), K(Y, Y) as a tuple
        """
        # functools caches require hashable inputs, so convert the 2D numpy arrays to lists of tuples
        Xp = list(map(tuple, X))
        Yp = list(map(tuple, Y))

        def _get_kernel(A, B, mask, diag=False):
            out = np.zeros((len(A), len(B)))
            for i in range(len(A)):
                for j in range(i if diag else 0, len(B)):
                    if mask is not None and mask[i, j]:
                        continue
                    ordered = (A[i], B[j]) if A[i] > B[j] else (B[j], A[i])
                    out[i, j] = _compute_dot_product_counts(*ordered, k, cumulative)
            return out

        K_XX = _get_kernel(Xp, Xp, mask_XX, diag=True)
        K_YY = _get_kernel(Yp, Yp, mask_YY, diag=True)
        K_XY = _get_kernel(Xp, Yp, mask_XY, diag=False)
        K_XX = _reflect_upper_triangular(K_XX)
        K_YY = _reflect_upper_triangular(K_YY)
        return K_XX, K_XY, K_YY

    return _mmd(
        X=sample1.sequences.numpy(),
        Y=sample2.sequences.numpy(),
        get_kernel=get_kspectrum_kernel,
    )


def mmd_all_subsequences(
    sample1: CompletionSample,
    sample2: CompletionSample,
):
    r"""
    Computes the all-subsequences MMD test statistic, which is the MMD test statistic using
    K(x, y) = \sum_{s \in x} \sum_{s \in y} 1[s in x and s in y]
    for all subsequences s of x and y.
    Args:
        sample1: CompletionSample
        sample2: CompletionSample
    """
    L = sample1.shape[-1]
    return mmd_kspectrum(sample1, sample2, k=L, cumulative=True)


#### Other two-sample tests ####


def _get_counts(sample1: torch.Tensor, sample2: torch.Tensor):
    """
    Given two 2D samples, finds the unique rows in the union of the two
    samples, and returns counts for each unique row in sample1, sample2.
    Args:
        sample1: (n, L) tensor where each row is a sequence
        sample2: (m, L) tensor where each row is a sequence
    Returns:
        unique_sequences: (k, L) tensor of unique sequences
        counts1_full: (k,) tensor of counts for each unique sequence in sample1
        counts2_full: (k,) tensor of counts for each unique sequence in sample2
    """
    unique1, counts1 = torch.unique(sample1, dim=0, return_counts=True)
    unique2, counts2 = torch.unique(sample2, dim=0, return_counts=True)
    all_sequences = torch.cat((unique1, unique2))
    unique_sequences, inverse = torch.unique(all_sequences, dim=0, return_inverse=True)

    counts1_full = torch.zeros(len(unique_sequences), dtype=torch.int64)
    counts2_full = torch.zeros(len(unique_sequences), dtype=torch.int64)
    counts1_full[inverse[: len(unique1)]] = counts1
    counts2_full[inverse[len(unique1) :]] = counts2
    return unique_sequences, counts1_full, counts2_full


def two_sample_chi_squared(
    sample1: CompletionSample,
    sample2: CompletionSample,
):
    r"""
    Computes the two-sample (centered) chi-squared test statistic, which has been modified for the imbalanced sample size case by
    Bhattacharya and Valiant (2015) and as cited in Balakrishnan & Wasserman (2017).
    $$
    \sum_{i=1}^k \frac{(N_2 c^1_i - N_1 c^2_i)^2}{c^1_i + c^2_i} - N_2^2 c^1_i - N_1^2 c^2_i
    $$
    where $c^1_i$ is the count of the $i$th unique sequence in sample1, $c^2_i$ is the count of the $i$th unique sequence in sample2,
    $N_1$ is the total number of sequences in sample1, and $N_2$ is the total number of sequences in sample2.

    References:
    - Bhattacharya and Valiant (2015) "Testing Closeness with Unequal Sized Samples" [step 2 in Alg 1]
        https://arxiv.org/abs/1504.04599
    - Balakrishnan & Wasserman (2017) "Hypothesis Testing for High-Dimensional Multinomials: A Selective Review"
        https://arxiv.org/abs/1712.06120

    Args:
        sample1: CompletionSample
        sample2: CompletionSample
    """
    _, c1, c2 = _get_counts(sample1.sequences, sample2.sequences)
    return np.nansum(
        (
            np.square(sample2.N * c1 - sample1.N * c2)
            - sample2.N**2 * c1
            - sample1.N**2 * c2
        )
        / (c1 + c2)
    )


def two_sample_L1(
    sample1: CompletionSample,
    sample2: CompletionSample,
):
    r"""
    Computes the two-sample L1 test statistic
    $$
    \sum_{i=1}^k |c^1_i / N_1 - c^2_i / N_2|
    $$
    where $c^1_i$ is the count of the $i$th unique sequence in sample1, $c^2_i$ is the count of the $i$th unique sequence in sample2,
    $N_1$ is the total number of sequences in sample1, and $N_2$ is the total number of sequences in sample2.

    References:
    - Balakrishnan & Wasserman (2017) "Hypothesis Testing for High-Dimensional Multinomials: A Selective Review"
        https://arxiv.org/abs/1712.06120
    """
    _, c1, c2 = _get_counts(sample1.sequences, sample2.sequences)
    return torch.sum(torch.abs(c1 / sample1.N - c2 / sample2.N)).item()


def two_sample_L2(
    sample1: CompletionSample,
    sample2: CompletionSample,
):
    r"""
    Computes the two-sample L2 test statistic
    $$
    \sum_{i=1}^k (c^1_i / N_1 - c^2_i / N_2)^2
    $$
    where $c^1_i$ is the count of the $i$th unique sequence in sample1, $c^2_i$ is the count of the $i$th unique sequence in sample2,
    $N_1$ is the total number of sequences in sample1, and $N_2$ is the total number of sequences in sample2.

    References:
    - Balakrishnan & Wasserman (2017) "Hypothesis Testing for High-Dimensional Multinomials: A Selective Review"
        https://arxiv.org/abs/1712.06120
    """
    _, c1, c2 = _get_counts(sample1.sequences, sample2.sequences)
    return torch.sum(torch.square(c1 / sample1.N - c2 / sample2.N)).item()


def two_sample_ks(
    sample1: CompletionSample,
    sample2: CompletionSample,
):
    """
    Computes the two-sample Kolmogorov-Smirnov test statistic. 
    Because the input data is sequences (categorical/high-dimensional), we 
    impose a lexicographical ordering on the unique sequences observed in the 
    combined sample to treat them as ordinal data.
    
    The test statistic is the maximum difference between the empirical cumulative 
    distribution functions (ECDFs) of the lexicographical ranks of the sequences.
    
    Args:
        sample1: CompletionSample
        sample2: CompletionSample
    Returns:
        float: KS test statistic
    """
    # Concatenate sequences to find a common sorting/ranking
    all_sequences = torch.cat([sample1.sequences, sample2.sequences], dim=0)
    
    # torch.unique with sorted=True (default) sorts the unique elements lexicographically
    # return_inverse gives us the index (rank) of each sequence in the sorted unique list
    _, inverse_indices = torch.unique(all_sequences, sorted=True, return_inverse=True, dim=0)
    
    # Split back into sample 1 and sample 2 ranks
    # Move to CPU for scipy
    ranks1 = inverse_indices[:sample1.N].float().cpu().numpy()
    ranks2 = inverse_indices[sample1.N:].float().cpu().numpy()
    
    # We perform the K-S test on these ranks
    # Note: ks_2samp calculates the max difference between ECDFs. 
    # Even though data is discrete (ranks), the statistic D is well-defined.
    return ks_2samp(ranks1, ranks2).statistic


def two_sample_ks_statistic(sample1, sample2, feature_fn=get_vader_scores):
    """
    Generic two-sample Kolmogorov-Smirnov test using custom feature extraction.

    This is a flexible function that computes the K-S statistic on any feature
    projection of the samples. It accepts a feature extraction function that
    converts CompletionSample objects to arrays of scalar scores.

    Args:
        sample1: First CompletionSample
        sample2: Second CompletionSample
        feature_fn: Function that takes a CompletionSample and returns array of scores.
            Default: get_vader_scores (sentiment analysis)

    Returns:
        K-S test statistic (float)

    Examples:
        >>> # Using VADER sentiment (default)
        >>> stat = two_sample_ks_statistic(sample1, sample2)

        >>> # Using perplexity scores with functools.partial
        >>> from functools import partial
        >>> from model_equality_testing.src.features import get_perplexity_scores
        >>> ppl_fn = partial(get_perplexity_scores, kenlm_model_path="model.arpa")
        >>> stat = two_sample_ks_statistic(sample1, sample2, feature_fn=ppl_fn)

        >>> # Via run_two_sample_test with perplexity
        >>> from model_equality_testing.algorithm import run_two_sample_test
        >>> pvalue, stat = run_two_sample_test(
        ...     sample1, sample2,
        ...     stat_type="two_sample_ks_statistic",
        ...     feature_fn=ppl_fn,
        ...     pvalue_type="permutation_pvalue",
        ...     b=1000
        ... )

        >>> # Custom feature function (e.g., text length)
        >>> def length_feature(sample):
        ...     from model_equality_testing.src.features import decode_sample_to_strings
        ...     texts = decode_sample_to_strings(sample)
        ...     return np.array([len(text) for text in texts])
        >>> stat = two_sample_ks_statistic(sample1, sample2, feature_fn=length_feature)

    See Also:
        two_sample_vader_ks: VADER sentiment-based K-S test
        two_sample_perplexity_ks: Perplexity-based K-S test
        get_vader_scores: Default feature function (sentiment)
        get_perplexity_scores: Perplexity feature function
    """
    scores1 = feature_fn(sample1)
    scores2 = feature_fn(sample2)
    statistic, _ = ks_2samp(scores1, scores2)
    return statistic


def two_sample_vader_ks(
    sample1: CompletionSample,
    sample2: CompletionSample,
):
    """
    Computes the two-sample Kolmogorov-Smirnov test statistic on VADER compound sentiment scores.
    Assumes that the sequences in the samples are Unicode codepoints (not token IDs).
    Requires the `vaderSentiment` package.
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        raise ImportError("Please install vaderSentiment to use this test: pip install vaderSentiment")

    analyzer = SentimentIntensityAnalyzer()

    def get_scores(sample):
        # Decode unicode integers back to string
        # sample.completion_sample is (N, L)
        scores = []
        # Ensure tensor is on CPU and numpy for iteration
        data = sample.completion_sample.cpu().numpy()
        for row in data:
            # Filter padding (-1) and convert to chars
            # Note: This assumes input was tokenized with tokenize_unicode which uses -1 padding
            text = "".join([chr(c) for c in row if c != -1])
            scores.append(analyzer.polarity_scores(text)["compound"])
        return scores

    scores1 = get_scores(sample1)
    scores2 = get_scores(sample2)

    return ks_2samp(scores1, scores2).statistic


def two_sample_perplexity_ks(
    sample1: CompletionSample,
    sample2: CompletionSample,
    kenlm_model_path: str,
    spm_model_path: str = None,
    granularity: str = "word",
    empty_text_score: Union[float, str] = "inf",
    zero_tokens_score: Union[float, str] = "inf",
    warn_on_degenerate: bool = True,
) -> float:
    """
    Computes two-sample K-S test statistic using KenLM perplexity scores.

    This test extracts perplexity scores from both samples using a KenLM language
    model, then computes the Kolmogorov-Smirnov test statistic on the distributions
    of perplexity scores.

    Requires kenlm package: pip install kenlm

    Args:
        sample1: First CompletionSample with unicode codepoint completions
        sample2: Second CompletionSample with unicode codepoint completions
        kenlm_model_path: Path to trained KenLM model (.arpa or .bin)
        spm_model_path: Optional path to sentencepiece model for text preprocessing
        granularity: "word" (split whitespace), "char" (character-level), or "token" (sentencepiece)
        empty_text_score: Score for empty strings - "inf", "nan", or float value (default: "inf")
        zero_tokens_score: Score for zero-token texts - "inf", "nan", or float value (default: "inf")
        warn_on_degenerate: Log warnings for degenerate cases (default: True)

    Returns:
        K-S test statistic (float)

    Examples:
        >>> # Basic usage with word-level perplexity
        >>> stat = two_sample_perplexity_ks(
        ...     sample1, sample2,
        ...     kenlm_model_path="model.arpa"
        ... )

        >>> # With sentencepiece preprocessing
        >>> stat = two_sample_perplexity_ks(
        ...     sample1, sample2,
        ...     kenlm_model_path="model.arpa",
        ...     spm_model_path="tokenizer.model",
        ...     granularity="token"
        ... )

    See Also:
        get_perplexity_scores: Feature extraction function
        train_kenlm_from_sample: Utility to train KenLM models from samples
        two_sample_vader_ks: Similar K-S test using sentiment scores
    """
    try:
        import kenlm
    except ImportError as e:
        raise ImportError(
            "Please install kenlm to use perplexity-based K-S tests: pip install kenlm"
        ) from e

    # Extract perplexity scores from both samples
    scores1 = get_perplexity_scores(
        sample1,
        kenlm_model_path=kenlm_model_path,
        spm_model_path=spm_model_path,
        granularity=granularity,
        empty_text_score=empty_text_score,
        zero_tokens_score=zero_tokens_score,
        warn_on_degenerate=warn_on_degenerate,
    )

    scores2 = get_perplexity_scores(
        sample2,
        kenlm_model_path=kenlm_model_path,
        spm_model_path=spm_model_path,
        granularity=granularity,
        empty_text_score=empty_text_score,
        zero_tokens_score=zero_tokens_score,
        warn_on_degenerate=warn_on_degenerate,
    )

    # Compute and return K-S statistic
    return ks_2samp(scores1, scores2).statistic


#######################
# Goodness of fit tests
#######################


def g_squared(
    sample: CompletionSample,
    null_dist: DistributionFromDataset,
):
    r"""
    Computes the G^2 / LRT test statistic
    $$
    -2 \sum_{i=1}^k o_i \log(p_i / m_i)
    $$
    where $o_i$ is the observed count of the $i$th unique sequence in the sample,
    $p_i$ is the completion probability of the $i$th unique sequence under the null distribution,
    and $m_i$ is the MLE of the completion probability of the $i$th unique sequence in the sample.

    Args:
        sample: CompletionSample
        null_dist: DistributionFromDataset
    """

    def _stat(p, m, o):
        # p = probabilities, m = mles, o = observed counts
        test_stat = np.log(p) - np.log(m)
        test_stat *= o
        test_stat = np.nansum(test_stat, axis=-1)
        return -2 * test_stat

    sequences, counts = torch.unique(
        sample.sequences,
        return_counts=True,
        dim=0,
    )
    probs = (
        null_dist.get_completion_probabilities(sequences)
        * null_dist.prompt_distribution[sequences[:, 0]]
    )
    return _stat(probs, counts / sample.N, counts)


def chi_squared(
    sample: CompletionSample,
    null_dist: DistributionFromDataset,
):
    r"""
    Computes the Pearson chi_squared test statistic
    $$
    \sum_{i=1}^k \frac{(o_i - n p_i)^2}{n p_i}
    $$
    where $o_i$ is the observed count of the $i$th unique sequence in the sample,
    $p_i$ is the completion probability of the $i$th unique sequence under the null distribution,
    and $n$ is the total number of sequences in the sample.

    Args:
        sample: CompletionSample
        null_dist: DistributionFromDataset
    """

    def _stat(seqs, probs, obs):
        hashmap = get_inv(tuple(map(tuple, seqs.numpy())))
        o, counts = torch.unique(obs, return_counts=True, dim=0)
        ixs = [hashmap[tuple(row.tolist())] for row in o]
        n = len(obs)
        obs_probs = probs[ixs]
        unobs_probs = probs[np.setdiff1d(np.arange(len(seqs)), ixs)]
        return (
            (np.square(counts - n * obs_probs) / (n * obs_probs)).nansum()
            + (np.square(0 - n * unobs_probs) / (n * unobs_probs)).nansum()
        ).item()

    sequences, probs = null_dist.get_all_joint_probabilities()
    return _stat(sequences, probs, sample.sequences)


def truncated_chi_squared(
    sample: CompletionSample,
    null_dist: DistributionFromDataset,
):
    r"""
    Computes the truncated chi_squared test statistic
    $$
    \sum_{i=1}^k \frac{(o_i - n p_i)^2 - o_i}{\max(p_i, 1/k)}
    $$
    where $o_i$ is the observed count of the $i$th unique sequence in the sample,
    $p_i$ is the completion probability of the $i$th unique sequence under the null distribution,
    and $n$ is the total number of sequences in the sample.

    References:
    - Balakrishnan & Wasserman (2017) "Hypothesis Testing for High-Dimensional Multinomials: A Selective Review"
        https://arxiv.org/abs/1712.06120

    Args:
        sample: CompletionSample
        null_dist: DistributionFromDataset
    """

    def _stat(seqs, probs, obs):
        hashmap = get_inv(tuple(map(tuple, seqs.numpy())))
        o, counts = torch.unique(obs, return_counts=True, dim=0)
        n = len(obs)
        ixs = [hashmap[tuple(row.tolist())] for row in o]
        obs_probs = probs[ixs]
        unobs_probs = probs[np.setdiff1d(np.arange(len(seqs)), ixs)]
        k = len(seqs)
        return (
            (
                (np.square(counts - n * obs_probs) - counts)
                / np.maximum(obs_probs, 1 / k * torch.ones_like(obs_probs))
            ).nansum()
            + (
                np.square(0 - n * unobs_probs)
                / np.maximum(unobs_probs, 1 / k * torch.ones_like(unobs_probs))
            ).nansum()
        ).item()

    sequences, probs = null_dist.get_all_joint_probabilities()
    return _stat(sequences, probs, sample.sequences)


def L1(
    sample: CompletionSample,
    null_dist: DistributionFromDataset,
):
    r"""
    Computes the L1 test statistic
    $$
    \sum_{i=1}^k |o_i - n p_i|
    $$

    Args:
        sample: CompletionSample
        null_dist: DistributionFromDataset
    """

    def _stat(seqs, probs, obs):
        hashmap = get_inv(tuple(map(tuple, seqs.numpy())))
        o, counts = torch.unique(obs, return_counts=True, dim=0)
        n = len(obs)
        ixs = [hashmap[tuple(row.tolist())] for row in o]
        obs_probs = probs[ixs]
        unobs_probs = probs[np.setdiff1d(np.arange(len(seqs)), ixs)]
        return (
            np.abs(counts - n * obs_probs).nansum()
            + np.abs(0 - n * unobs_probs).nansum()
        ).item()

    sequences, probs = null_dist.get_all_joint_probabilities()
    return _stat(sequences, probs, sample.sequences)


def L2(
    sample: CompletionSample,
    null_dist: DistributionFromDataset,
):
    r"""
    Computes the L2 test statistic
    $$
    \sum_{i=1}^k (o_i - n p_i)^2
    $$
    where $o_i$ is the observed count of the $i$th unique sequence in the sample,
    $p_i$ is the completion probability of the $i$th unique sequence under the null distribution,
    and $n$ is the total number of sequences in the sample.

    Args:
        sample: CompletionSample
        null_dist: DistributionFromDataset
    """

    def _stat(seqs, probs, obs):
        hashmap = get_inv(tuple(map(tuple, seqs.numpy())))
        o, counts = torch.unique(obs, return_counts=True, dim=0)
        n = len(obs)
        ixs = [hashmap[tuple(row.tolist())] for row in o]
        obs_probs = probs[ixs]
        unobs_probs = probs[np.setdiff1d(np.arange(len(seqs)), ixs)]
        return (
            np.square(counts - n * obs_probs).nansum()
            + np.square(0 - n * unobs_probs).nansum()
        ).item()

    sequences, probs = null_dist.get_all_joint_probabilities()
    return _stat(sequences, probs, sample.sequences)


###### map from name to function ######

IMPLEMENTED_TESTS = {
    "g_squared": g_squared,
    "chi_squared": chi_squared,
    "truncated_chi_squared": truncated_chi_squared,
    "L1": L1,
    "L2": L2,
    "two_sample_chi_squared": two_sample_chi_squared,
    "two_sample_L1": two_sample_L1,
    "two_sample_L2": two_sample_L2,
    "two_sample_ks": two_sample_ks,
    "two_sample_vader_ks": two_sample_vader_ks,
    "mmd_hamming": mmd_hamming,
    "mmd_kspectrum": mmd_kspectrum,
    "mmd_all_subsequences": mmd_all_subsequences,
}