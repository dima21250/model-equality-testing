# Registry for test statistic functions
# This module isolates the IMPLEMENTED_TESTS mapping to avoid circular imports.
# It imports the concrete test functions from `tests.py` and builds the dictionary.

from .tests import (
    g_squared,
    chi_squared,
    truncated_chi_squared,
    L1,
    L2,
    two_sample_chi_squared,
    two_sample_L1,
    two_sample_L2,
    two_sample_ks,
    two_sample_ks_statistic,
    two_sample_vader_ks,
    two_sample_perplexity_ks,
    mmd_hamming,
    mmd_kspectrum,
    mmd_all_subsequences,
    quantum_von_neumann_divergence,
    quantum_relative_entropy_test,
)

# Mapping from string name to test function
# Note: quantum_trace_distance removed from registry due to pathological
# sample-size dependence. The function remains available for direct use
# via tests.quantum_trace_distance(). See INITIAL-RESULTS.md.
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
    "two_sample_ks_statistic": two_sample_ks_statistic,
    "two_sample_vader_ks": two_sample_vader_ks,
    "two_sample_perplexity_ks": two_sample_perplexity_ks,
    "mmd_hamming": mmd_hamming,
    "mmd_kspectrum": mmd_kspectrum,
    "mmd_all_subsequences": mmd_all_subsequences,
    "quantum_von_neumann_divergence": quantum_von_neumann_divergence,
    "quantum_relative_entropy": quantum_relative_entropy_test,
}
