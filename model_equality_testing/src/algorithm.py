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
        if pvalue_type == "permutation_pvalue":
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
