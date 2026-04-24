"""
Quantum-inspired metrics for comparing LLM output distributions.

This module implements quantum-inspired metrics computed on embeddings:
- Pairwise Inner Product (PIP) matrices
- Density matrix normalization
- Trace distance (quantum distinguishability)
- Von Neumann entropy (quantum uncertainty)
- Quantum Relative Entropy (QRE, quantum divergence)

These metrics operate on semantic embeddings and can capture higher-level
distributional differences compared to token-level or n-gram statistics.

References:
- Nielsen & Chuang, "Quantum Computation and Quantum Information"
- FLAIR-ALTERNATIVE.md (this repository)
"""

import numpy as np
import logging
from typing import Tuple


def compute_pip_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute Pairwise Inner Product (PIP) matrix from embeddings.

    The PIP matrix encodes pairwise similarities between all embeddings
    using inner products: PIP[i,j] = <e_i, e_j>.

    Args:
        embeddings: (N, d) array of N embeddings with dimension d

    Returns:
        PIP: (N, N) symmetric matrix where PIP[i,j] = <e_i, e_j>

    Examples:
        >>> embeddings = np.random.randn(100, 768)
        >>> pip = compute_pip_matrix(embeddings)
        >>> pip.shape
        (100, 100)
        >>> np.allclose(pip, pip.T)  # Symmetric
        True
    """
    return embeddings @ embeddings.T


def normalize_to_density_matrix(
    pip: np.ndarray,
    epsilon: float = 1e-10
) -> np.ndarray:
    """Normalize PIP matrix to a quantum density matrix.

    A density matrix must:
    1. Be Hermitian (for real matrices, symmetric)
    2. Have unit trace: Tr(ρ) = 1
    3. Be positive semidefinite (all eigenvalues >= 0)

    This function normalizes by trace: ρ = PIP / Tr(PIP).

    Args:
        pip: (N, N) PIP matrix (must be symmetric and positive semidefinite)
        epsilon: Minimum acceptable trace value (for numerical stability)

    Returns:
        rho: (N, N) normalized density matrix with Tr(ρ) = 1

    Raises:
        ValueError: If PIP trace is too small (< epsilon)

    Examples:
        >>> embeddings = np.random.randn(100, 768)
        >>> pip = compute_pip_matrix(embeddings)
        >>> rho = normalize_to_density_matrix(pip)
        >>> np.isclose(np.trace(rho), 1.0)
        True
    """
    trace = np.trace(pip)

    if trace < epsilon:
        raise ValueError(
            f"PIP matrix trace is too small: {trace:.2e} < {epsilon:.2e}. "
            "This may indicate degenerate embeddings (e.g., all zero vectors)."
        )

    rho = pip / trace
    return rho


def trace_distance(rho_a: np.ndarray, rho_b: np.ndarray) -> float:
    """Compute trace distance between two density matrices.

    The trace distance quantifies how distinguishable two quantum states are:
        D(ρ_A, ρ_B) = 0.5 * Tr(|ρ_A - ρ_B|)

    where |M| = sqrt(M† M) is the matrix absolute value (sum of absolute eigenvalues).

    Properties:
    - D ∈ [0, 1]
    - D(ρ, ρ) = 0 (identical states)
    - D(ρ_A, ρ_B) = 1 iff ρ_A and ρ_B have orthogonal support

    Args:
        rho_a: (N, N) density matrix for distribution A
        rho_b: (N, N) density matrix for distribution B

    Returns:
        Trace distance in [0, 1]

    Examples:
        >>> # Identical density matrices
        >>> rho = np.eye(10) / 10
        >>> trace_distance(rho, rho)
        0.0

        >>> # Orthogonal density matrices
        >>> rho_a = np.diag([1.0, 0.0])
        >>> rho_b = np.diag([0.0, 1.0])
        >>> trace_distance(rho_a, rho_b)
        1.0

    References:
        Nielsen & Chuang, "Quantum Computation and Quantum Information", Section 9.2
    """
    diff = rho_a - rho_b

    # Compute eigenvalues of the difference matrix
    # For Hermitian matrices, eigenvalues are real
    eigenvalues = np.linalg.eigvalsh(diff)

    # Trace distance is half the sum of absolute eigenvalues
    # This equals Tr(|ρ_A - ρ_B|) / 2
    distance = 0.5 * np.sum(np.abs(eigenvalues))

    return distance


def von_neumann_entropy(
    rho: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """Compute von Neumann entropy of a density matrix.

    The von Neumann entropy is the quantum analog of Shannon entropy:
        S(ρ) = -Tr(ρ log ρ) = -Σ λ_i log λ_i

    where λ_i are the eigenvalues of ρ.

    Properties:
    - S(ρ) >= 0
    - S(ρ) = 0 iff ρ is a pure state (rank 1)
    - S(ρ) is maximal for maximally mixed states

    Args:
        rho: (N, N) density matrix
        epsilon: Filter eigenvalues below this threshold (numerical zeros)

    Returns:
        Von Neumann entropy (non-negative)

    Examples:
        >>> # Pure state (rank 1) has zero entropy
        >>> rho_pure = np.array([[1.0, 0.0], [0.0, 0.0]])
        >>> von_neumann_entropy(rho_pure)
        0.0

        >>> # Maximally mixed state
        >>> rho_mixed = np.eye(4) / 4
        >>> entropy = von_neumann_entropy(rho_mixed)
        >>> np.isclose(entropy, np.log(4))  # log(N) for maximally mixed
        True

    References:
        Nielsen & Chuang, "Quantum Computation and Quantum Information", Section 11.3
    """
    # Compute eigenvalues (all real for Hermitian matrices)
    eigenvalues = np.linalg.eigvalsh(rho)

    # Filter out numerical zeros
    # Note: Eigenvalues can be negative due to numerical errors,
    # but should be non-negative in theory. We filter those too.
    positive_eigenvalues = eigenvalues[eigenvalues > epsilon]

    if len(positive_eigenvalues) == 0:
        # Degenerate case: all eigenvalues are zero (shouldn't happen for valid density matrix)
        logging.warning(
            "All eigenvalues below threshold. Density matrix may be degenerate."
        )
        return 0.0

    # Compute -Σ λ log λ
    # Use natural logarithm (could also use log2 for bits)
    entropy = -np.sum(positive_eigenvalues * np.log(positive_eigenvalues))

    return entropy


def quantum_relative_entropy(
    rho: np.ndarray,
    sigma: np.ndarray,
    epsilon: float = 1e-10,
    check_support: bool = True,
) -> float:
    """Compute quantum relative entropy (QRE) between two density matrices.

    The quantum relative entropy is defined as:
        S(ρ || σ) = Tr(ρ log ρ - ρ log σ)

    This can be computed via eigendecomposition:
        S(ρ || σ) = Σ_i λ_i (log λ_i - log μ_i)

    where λ_i are eigenvalues of ρ and μ_i are eigenvalues of σ.

    Properties:
    - S(ρ || σ) >= 0 (non-negative)
    - S(ρ || σ) = 0 iff ρ = σ
    - S(ρ || σ) = ∞ if supp(ρ) ⊄ supp(σ) (ρ has support outside σ's support)
    - S(ρ || σ) is NOT symmetric: S(ρ || σ) ≠ S(σ || ρ) in general

    Args:
        rho: (N, N) density matrix (reference distribution)
        sigma: (N, N) density matrix (comparison distribution)
        epsilon: Threshold for filtering small eigenvalues
        check_support: If True, return np.inf when supports don't overlap properly

    Returns:
        Quantum relative entropy (non-negative, can be np.inf)

    Examples:
        >>> # Identical states
        >>> rho = np.eye(3) / 3
        >>> quantum_relative_entropy(rho, rho)
        0.0

        >>> # Different states with overlapping support
        >>> rho = np.diag([0.5, 0.3, 0.2])
        >>> sigma = np.diag([0.4, 0.4, 0.2])
        >>> qre = quantum_relative_entropy(rho, sigma)
        >>> qre >= 0
        True

    References:
        Nielsen & Chuang, "Quantum Computation and Quantum Information", Section 11.3.2
        Wilde, "Quantum Information Theory", Section 11.7

    Notes:
        This implementation uses a simplified eigenvalue-based approach.
        For general (non-commuting) matrices, a more sophisticated method
        involving matrix logarithms is needed.
    """
    # Compute eigenvalues
    lambda_rho = np.linalg.eigvalsh(rho)
    lambda_sigma = np.linalg.eigvalsh(sigma)

    # Filter eigenvalues
    lambda_rho = lambda_rho[lambda_rho > epsilon]
    lambda_sigma = lambda_sigma[lambda_sigma > epsilon]

    # Check support condition: supp(rho) ⊆ supp(sigma)
    # In practice, this means checking if rho has eigenvalues in regions where sigma is zero
    if check_support:
        # Count effective ranks
        rank_rho = len(lambda_rho)
        rank_sigma = len(lambda_sigma)

        if rank_rho > rank_sigma:
            # rho has higher rank than sigma, likely violates support condition
            logging.warning(
                f"Rank mismatch: rank(ρ)={rank_rho} > rank(σ)={rank_sigma}. "
                "Support condition may be violated. Returning np.inf."
            )
            return np.inf

    # Compute S(ρ || σ) using the formula:
    # S(ρ || σ) ≈ Σ λ_i (log λ_i) - Σ λ_i (log μ_j)
    #
    # Note: This is an approximation that assumes the eigenspaces align.
    # For a more rigorous computation, we'd need to compute Tr(ρ log σ)
    # using matrix logarithms, which is more complex.

    term_rho = np.sum(lambda_rho * np.log(lambda_rho))

    # For the second term, we need to be careful about matching eigenvalues
    # In the simplified version, we assume both are diagonal in the same basis
    # and pair eigenvalues in order (this is an approximation)

    if len(lambda_sigma) < len(lambda_rho):
        # Pad sigma eigenvalues with small values to avoid dimension mismatch
        lambda_sigma_padded = np.concatenate([
            lambda_sigma,
            np.full(len(lambda_rho) - len(lambda_sigma), epsilon)
        ])
    else:
        lambda_sigma_padded = lambda_sigma[:len(lambda_rho)]

    # Sort eigenvalues in descending order for better pairing
    lambda_rho_sorted = np.sort(lambda_rho)[::-1]
    lambda_sigma_sorted = np.sort(lambda_sigma_padded)[::-1]

    # Check for near-zero sigma eigenvalues where rho is nonzero
    for i, (lr, ls) in enumerate(zip(lambda_rho_sorted, lambda_sigma_sorted)):
        if lr > epsilon and ls < epsilon:
            logging.warning(
                f"ρ eigenvalue {lr:.2e} paired with near-zero σ eigenvalue {ls:.2e}. "
                "Support condition violated. Returning np.inf."
            )
            return np.inf

    term_sigma = np.sum(lambda_rho_sorted * np.log(lambda_sigma_sorted))

    qre = term_rho - term_sigma

    # QRE should be non-negative; negative values indicate numerical errors
    if qre < -epsilon:
        logging.warning(
            f"Computed QRE is negative: {qre:.2e}. "
            "This indicates numerical instability. Returning 0.0."
        )
        return 0.0

    return max(qre, 0.0)  # Clip to non-negative
