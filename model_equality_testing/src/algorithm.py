from typing import Union, Tuple, List, Dict
# Lazy import of pvalue utilities to avoid circular imports; they are imported inside the functions that need them.

# Lazy import of IMPLEMENTED_TESTS moved inside functions to avoid circular imports
from model_equality_testing.distribution import (
    CompletionSample,
    DistributionFromDataset,
)


def _noop_pvalue(*args, **kwargs):
    return 1.0


def run_goodness_of_fit_test(
    sample: CompletionSample,
    null_dist: DistributionFromDataset,
    get_pvalue: Union[callable, 'EmpiricalPvalueCalculator'] = None,
    pvalue_type: str = "parametric_bootstrap",
    stat_type: str = "g_squared",
    b=1000,
    **kwargs,
) -> Tuple[float, float]:
    """Run a goodness‑of‑fit test.

    This function lazily imports the required p‑value utilities and the
    ``IMPLEMENTED_TESTS`` mapping to avoid circular imports.
    """
    # Lazy imports
    from .pvalue import (
        EmpiricalPvalueCalculator,
        one_sample_parametric_bootstrap_pvalue,
    )
    from .tests import IMPLEMENTED_TESTS

    if get_pvalue is None:
        if pvalue_type == "parametric_bootstrap":
            get_pvalue = one_sample_parametric_bootstrap_pvalue(
                null_dist=null_dist,
                n=sample.N,
                b=b,
                return_stats=False,
                stat_type=stat_type,
            )
        elif pvalue_type == "dummy":
            get_pvalue = _noop_pvalue
        else:
            raise ValueError("Unrecognized p-value type")

    statistic = IMPLEMENTED_TESTS[stat_type](sample, null_dist, **kwargs)
    pvalue = get_pvalue(statistic)
    if not isinstance(pvalue, float):
        pvalue = pvalue.item()
    return (pvalue, statistic)



def run_two_sample_test(
    sample: CompletionSample,
    other_sample: CompletionSample,
    null_dist: DistributionFromDataset = None,
    get_pvalue: Union[callable, 'EmpiricalPvalueCalculator'] = None,
    pvalue_type: str = "permutation_pvalue",
    stat_type: str = "two_sample_L2",
    b=1000,
    **kwargs,
) -> Tuple[float, float]:
    """Run a two‑sample test.

    This function lazily imports the required p‑value utilities and the
    ``IMPLEMENTED_TESTS`` mapping to avoid circular imports.

    Args:
        sample: First CompletionSample
        other_sample: Second CompletionSample
        null_dist: Optional null distribution (required for parametric_bootstrap)
        get_pvalue: Optional custom p-value calculator
        pvalue_type: P-value calculation method:
            - "permutation_pvalue": Permutation test (default)
            - "parametric_bootstrap": Bootstrap from null distribution
            - "analytical_ks": Analytical K-S p-value (only for K-S tests)
            - "dummy": Always returns 1.0
        stat_type: Test statistic name (must be in IMPLEMENTED_TESTS)
        b: Number of bootstrap/permutation samples
        **kwargs: Additional arguments passed to the test statistic function

    Returns:
        Tuple of (pvalue, statistic)
    """
    # Lazy imports
    from .pvalue import (
        EmpiricalPvalueCalculator,
        two_sample_permutation_pvalue,
        two_sample_parametric_bootstrap_pvalue,
        AnalyticalKSPvalueCalculator,
    )
    from .tests import IMPLEMENTED_TESTS

    if get_pvalue is None:
        if pvalue_type == "analytical_ks":
            # For K-S tests, compute both statistic and p-value efficiently in one call
            # This avoids computing feature scores twice
            from scipy.stats import ks_2samp
            from functools import partial

            # Determine feature function based on stat_type
            if stat_type == "two_sample_vader_ks":
                from .features import get_vader_scores
                feature_fn = get_vader_scores
            elif stat_type == "two_sample_perplexity_ks":
                from .features import get_perplexity_scores
                # Extract perplexity-specific parameters
                kenlm_model_path = kwargs.get('kenlm_model_path')
                if kenlm_model_path is None:
                    raise ValueError("kenlm_model_path required for two_sample_perplexity_ks")
                # Build feature function with perplexity parameters
                ppl_kwargs = {k: v for k, v in kwargs.items()
                             if k in ['kenlm_model_path', 'spm_model_path', 'granularity',
                                     'empty_text_score', 'zero_tokens_score', 'warn_on_degenerate']}
                feature_fn = partial(get_perplexity_scores, **ppl_kwargs)
            elif stat_type == "two_sample_ks_statistic":
                # Generic K-S test - use provided feature_fn or default to VADER
                feature_fn = kwargs.get('feature_fn')
                if feature_fn is None:
                    from .features import get_vader_scores
                    feature_fn = get_vader_scores
            else:
                raise ValueError(
                    f"analytical_ks only works with K-S test statistics "
                    f"(two_sample_vader_ks, two_sample_perplexity_ks, two_sample_ks_statistic), "
                    f"got {stat_type}"
                )

            # Compute features and run K-S test (single call for efficiency)
            scores1 = feature_fn(sample)
            scores2 = feature_fn(other_sample)
            statistic, pvalue = ks_2samp(scores1, scores2)

            if not isinstance(pvalue, float):
                pvalue = float(pvalue)
            if not isinstance(statistic, float):
                statistic = float(statistic)

            return (pvalue, statistic)

        elif pvalue_type == "permutation_pvalue":
            get_pvalue = two_sample_permutation_pvalue(
                sample, other_sample, b=b, stat_type=stat_type, **kwargs
            )
        elif pvalue_type == "parametric_bootstrap":
            assert (
                null_dist is not None
            ), "Must provide null distribution for parametric bootstrap"
            get_pvalue = two_sample_parametric_bootstrap_pvalue(
                null_dist=null_dist,
                n1=sample.N,
                n2=other_sample.N,
                b=b,
                stat_type=stat_type,
                **kwargs,
            )
        elif pvalue_type == "dummy":
            get_pvalue = _noop_pvalue
        else:
            raise ValueError("Unrecognized p-value type")

    statistic = IMPLEMENTED_TESTS[stat_type](sample, other_sample, **kwargs)
    # Try calling the custom p‑value calculator. It may expect a statistic (default) or a sample.
    try:
        pvalue = get_pvalue(statistic)
    except (TypeError, AttributeError):
        # Fallback: assume the calculator expects a CompletionSample (e.g., AnalyticalKSPvalueCalculator)
        pvalue = get_pvalue(other_sample)
    if not isinstance(pvalue, float):
        pvalue = pvalue.item()
    return (pvalue, statistic)


##############################################
# Convenience functions for K-S tests
##############################################


def perplexity_ks_test(
    sample1: CompletionSample,
    sample2: CompletionSample,
    kenlm_model_path: str,
    spm_model_path: str = None,
    granularity: str = "word",
    empty_text_score: Union[float, str] = "inf",
    zero_tokens_score: Union[float, str] = "inf",
    warn_on_degenerate: bool = True,
    pvalue_method: str = "analytical",
    b: int = 1000,
) -> Tuple[float, float]:
    """
    Convenience function for perplexity-based Kolmogorov-Smirnov two-sample test.

    This function provides a simple interface for running K-S tests on perplexity scores.
    It supports both analytical p-values (fast, based on scipy's implementation) and
    permutation-based p-values (more robust, but slower).

    Args:
        sample1: First CompletionSample with unicode codepoint completions
        sample2: Second CompletionSample with unicode codepoint completions
        kenlm_model_path: Path to trained KenLM model (.arpa or .bin)
        spm_model_path: Optional path to sentencepiece model for text preprocessing
        granularity: "word" (split whitespace), "char" (character-level), or "token" (sentencepiece)
        empty_text_score: Score for empty strings - "inf", "nan", or float value (default: "inf")
        zero_tokens_score: Score for zero-token texts - "inf", "nan", or float value (default: "inf")
        warn_on_degenerate: Log warnings for degenerate cases (default: True)
        pvalue_method: "analytical" (fast, uses scipy) or "permutation" (robust, slower)
        b: Number of permutations (only used if pvalue_method="permutation")

    Returns:
        Tuple of (pvalue, statistic)

    Examples:
        >>> # Quick test with analytical p-value
        >>> pvalue, stat = perplexity_ks_test(sample1, sample2, "model.arpa")
        >>> print(f"p-value: {pvalue:.4f}, K-S statistic: {stat:.4f}")

        >>> # More robust test with permutation p-value
        >>> pvalue, stat = perplexity_ks_test(
        ...     sample1, sample2, "model.arpa",
        ...     pvalue_method="permutation", b=1000
        ... )
    """
    if pvalue_method == "analytical":
        # Direct computation - most efficient
        from scipy.stats import ks_2samp
        from .features import get_perplexity_scores

        scores1 = get_perplexity_scores(
            sample1, kenlm_model_path, spm_model_path, granularity,
            empty_text_score, zero_tokens_score, warn_on_degenerate
        )
        scores2 = get_perplexity_scores(
            sample2, kenlm_model_path, spm_model_path, granularity,
            empty_text_score, zero_tokens_score, warn_on_degenerate
        )
        statistic, pvalue = ks_2samp(scores1, scores2)
        return (float(pvalue), float(statistic))

    elif pvalue_method == "permutation":
        # Use framework's permutation test
        return run_two_sample_test(
            sample1, sample2,
            stat_type="two_sample_perplexity_ks",
            pvalue_type="permutation_pvalue",
            kenlm_model_path=kenlm_model_path,
            spm_model_path=spm_model_path,
            granularity=granularity,
            empty_text_score=empty_text_score,
            zero_tokens_score=zero_tokens_score,
            warn_on_degenerate=warn_on_degenerate,
            b=b,
        )
    else:
        raise ValueError(f"Unknown pvalue_method: {pvalue_method}. Use 'analytical' or 'permutation'.")


def vader_ks_test(
    sample1: CompletionSample,
    sample2: CompletionSample,
    pvalue_method: str = "analytical",
    b: int = 1000,
) -> Tuple[float, float]:
    """
    Convenience function for VADER sentiment-based Kolmogorov-Smirnov two-sample test.

    This function provides a simple interface for running K-S tests on VADER compound
    sentiment scores. Requires the vaderSentiment package.

    Args:
        sample1: First CompletionSample with unicode codepoint completions
        sample2: Second CompletionSample with unicode codepoint completions
        pvalue_method: "analytical" (fast, uses scipy) or "permutation" (robust, slower)
        b: Number of permutations (only used if pvalue_method="permutation")

    Returns:
        Tuple of (pvalue, statistic)

    Examples:
        >>> # Quick test with analytical p-value
        >>> pvalue, stat = vader_ks_test(sample1, sample2)
        >>> print(f"p-value: {pvalue:.4f}, K-S statistic: {stat:.4f}")

        >>> # More robust test with permutation p-value
        >>> pvalue, stat = vader_ks_test(sample1, sample2, pvalue_method="permutation")
    """
    if pvalue_method == "analytical":
        # Direct computation - most efficient
        from scipy.stats import ks_2samp
        from .features import get_vader_scores

        scores1 = get_vader_scores(sample1)
        scores2 = get_vader_scores(sample2)
        statistic, pvalue = ks_2samp(scores1, scores2)
        return (float(pvalue), float(statistic))

    elif pvalue_method == "permutation":
        # Use framework's permutation test
        return run_two_sample_test(
            sample1, sample2,
            stat_type="two_sample_vader_ks",
            pvalue_type="permutation_pvalue",
            b=b,
        )
    else:
        raise ValueError(f"Unknown pvalue_method: {pvalue_method}. Use 'analytical' or 'permutation'.")


def ks_test(
    sample1: CompletionSample,
    sample2: CompletionSample,
    feature_fn=None,
    pvalue_method: str = "analytical",
    b: int = 1000,
) -> Tuple[float, float]:
    """
    Convenience function for generic Kolmogorov-Smirnov two-sample test with custom features.

    This function provides a flexible interface for running K-S tests on any feature
    projection of completion samples. If no feature function is provided, defaults to
    VADER sentiment scores.

    Args:
        sample1: First CompletionSample
        sample2: Second CompletionSample
        feature_fn: Function that takes a CompletionSample and returns array of scores.
            If None, defaults to get_vader_scores (sentiment analysis)
        pvalue_method: "analytical" (fast, uses scipy) or "permutation" (robust, slower)
        b: Number of permutations (only used if pvalue_method="permutation")

    Returns:
        Tuple of (pvalue, statistic)

    Examples:
        >>> # Using default VADER sentiment scores
        >>> pvalue, stat = ks_test(sample1, sample2)

        >>> # Using perplexity scores (via functools.partial)
        >>> from functools import partial
        >>> from model_equality_testing.src.features import get_perplexity_scores
        >>> ppl_fn = partial(get_perplexity_scores, kenlm_model_path="model.arpa")
        >>> pvalue, stat = ks_test(sample1, sample2, feature_fn=ppl_fn)

        >>> # Using custom feature function (e.g., text length)
        >>> def length_feature(sample):
        ...     from model_equality_testing.src.features import decode_sample_to_strings
        ...     texts = decode_sample_to_strings(sample)
        ...     return np.array([len(text) for text in texts])
        >>> pvalue, stat = ks_test(sample1, sample2, feature_fn=length_feature)

        >>> # With permutation p-value for robustness
        >>> pvalue, stat = ks_test(sample1, sample2, feature_fn=ppl_fn,
        ...                        pvalue_method="permutation", b=1000)
    """
    if feature_fn is None:
        from .features import get_vader_scores
        feature_fn = get_vader_scores

    if pvalue_method == "analytical":
        # Direct computation - most efficient
        from scipy.stats import ks_2samp

        scores1 = feature_fn(sample1)
        scores2 = feature_fn(sample2)
        statistic, pvalue = ks_2samp(scores1, scores2)
        return (float(pvalue), float(statistic))

    elif pvalue_method == "permutation":
        # Use framework's permutation test
        return run_two_sample_test(
            sample1, sample2,
            stat_type="two_sample_ks_statistic",
            pvalue_type="permutation_pvalue",
            feature_fn=feature_fn,
            b=b,
        )
    else:
        raise ValueError(f"Unknown pvalue_method: {pvalue_method}. Use 'analytical' or 'permutation'.")
