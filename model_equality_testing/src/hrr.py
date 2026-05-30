"""
Holographic Reduced Representations (HRR) for compositional embedding.

Implements circular convolution binding to create prompt-aware embeddings
that encode *what the model said in response to what*. These HRR-bound
vectors live in the same Hilbert space as plain SBERT embeddings, so
existing density matrix and quantum metric machinery applies directly.

References:
    Plate, T. A. (1995). Holographic reduced representations.
    IEEE Transactions on Neural Networks, 6(3), 623-641.
"""

import numpy as np
from typing import Optional, Dict
from numpy.fft import fft, ifft


def circular_convolution(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bind two vectors via circular convolution.

    Computed efficiently in the Fourier domain:
        a ⊛ b = IFFT(FFT(a) ⊙ FFT(b))

    Args:
        a: (d,) vector
        b: (d,) vector

    Returns:
        (d,) bound vector
    """
    return np.real(ifft(fft(a) * fft(b)))


def circular_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Approximate inverse of circular convolution.

    Given bound = a ⊛ b, circular_correlation(a, bound) ≈ b (noisy).

    Args:
        a: (d,) cue vector
        b: (d,) bound vector

    Returns:
        (d,) approximate retrieval
    """
    return np.real(ifft(np.conj(fft(a)) * fft(b)))


def generate_role_vector(d: int, rng: np.random.Generator = None) -> np.ndarray:
    """Generate a random role vector for HRR binding.

    Components drawn i.i.d. from N(0, 1/d), which approximately preserves
    norms under circular convolution.

    Args:
        d: Dimensionality (must match embedding dimension)
        rng: NumPy random generator for reproducibility

    Returns:
        (d,) role vector
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.normal(0, 1.0 / np.sqrt(d), size=d)


def generate_prompt_id_vectors(
    prompt_ids: np.ndarray,
    d: int,
    seed: int = 42,
) -> Dict[int, np.ndarray]:
    """Generate a unique random vector for each prompt ID.

    Uses a seeded RNG so the same prompt ID always gets the same vector
    within a session. This is critical for permutation tests: the binding
    vectors must be fixed across permutations.

    Args:
        prompt_ids: Array of integer prompt IDs present in the sample
        d: Embedding dimensionality
        seed: Random seed for reproducibility

    Returns:
        Dict mapping prompt_id -> (d,) role vector
    """
    rng = np.random.default_rng(seed)
    unique_ids = np.unique(prompt_ids)
    return {pid: generate_role_vector(d, rng) for pid in unique_ids}


def bind_embeddings_to_prompts(
    embeddings: np.ndarray,
    prompt_ids: np.ndarray,
    prompt_vectors: Dict[int, np.ndarray],
) -> np.ndarray:
    """Bind each embedding to its prompt identity via circular convolution.

    For each completion embedding e_i with prompt p_i:
        hrr_i = prompt_vector[p_i] ⊛ e_i

    The resulting HRR vectors encode *what was said in response to which
    prompt*, not just what was said.

    Args:
        embeddings: (N, d) SBERT embeddings
        prompt_ids: (N,) integer prompt indices
        prompt_vectors: Dict mapping prompt_id -> (d,) role vector

    Returns:
        (N, d) HRR-bound embeddings
    """
    N, d = embeddings.shape
    bound = np.empty_like(embeddings)
    for i in range(N):
        pid = int(prompt_ids[i])
        bound[i] = circular_convolution(prompt_vectors[pid], embeddings[i])
    return bound


def hrr_embed_sample(
    sample,
    embeddings: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """Create HRR-bound embeddings from a CompletionSample.

    Takes pre-computed SBERT embeddings and binds each one to its prompt
    identity. This is the main entry point for integrating HRR into
    the quantum metrics pipeline.

    Args:
        sample: CompletionSample with .prompt_sample attribute
        embeddings: (N, d) pre-computed SBERT embeddings
        seed: Random seed for prompt vector generation

    Returns:
        (N, d) HRR-bound embeddings
    """
    prompt_ids = sample.prompt_sample.numpy()
    d = embeddings.shape[1]
    prompt_vectors = generate_prompt_id_vectors(prompt_ids, d, seed=seed)
    return bind_embeddings_to_prompts(embeddings, prompt_ids, prompt_vectors)


def effective_rank(rho: np.ndarray, epsilon: float = 1e-10) -> float:
    """Compute effective rank of a density matrix.

    The effective rank (also called participation ratio) measures how
    many eigenvalues contribute meaningfully:

        r_eff = exp(S(ρ))

    where S(ρ) is the von Neumann entropy. A density matrix with k
    equal eigenvalues has effective rank k.

    This is the key psychometric measure: it tells you how many
    distinct behavioral modes the model uses.

    Args:
        rho: (k, k) density matrix with Tr(ρ) = 1

    Returns:
        Effective rank (float >= 1.0)
    """
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > epsilon]
    entropy = -np.sum(eigenvalues * np.log(eigenvalues))
    return np.exp(entropy)


def prompt_sensitivity(
    embeddings: np.ndarray,
    sample,
    seed: int = 42,
    pca_k: int = 0,
) -> float:
    """Measure how much prompt binding changes the density matrix structure.

    Computes the ratio of effective ranks:
        sensitivity = effective_rank(ρ_HRR) / effective_rank(ρ_SBERT)

    Interpretation:
        > 1.0: Model differentiates responses across prompts (prompt-sensitive)
        ≈ 1.0: Model gives homogeneous responses regardless of prompt
        < 1.0: Binding compresses structure (unlikely with good prompts)

    Args:
        embeddings: (N, d) SBERT embeddings
        sample: CompletionSample
        seed: Random seed for HRR vectors
        pca_k: PCA components (0 for full-space)

    Returns:
        Sensitivity ratio (float)
    """
    from .quantum_metrics import full_density_matrix, pca_density_matrix, fit_pca

    hrr_emb = hrr_embed_sample(sample, embeddings, seed=seed)

    if pca_k > 0:
        pca = fit_pca(embeddings, k=pca_k)
        rho_sbert = pca_density_matrix(embeddings, pca)
        pca_hrr = fit_pca(hrr_emb, k=pca_k)
        rho_hrr = pca_density_matrix(hrr_emb, pca_hrr)
    else:
        rho_sbert = full_density_matrix(embeddings)
        rho_hrr = full_density_matrix(hrr_emb)

    rank_sbert = effective_rank(rho_sbert)
    rank_hrr = effective_rank(rho_hrr)

    return rank_hrr / rank_sbert
