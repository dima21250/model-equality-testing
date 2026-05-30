"""
Psychometric profiling of LLMs via HRR + quantum metrics.

Computes behavioral profiles from model completions: eigenspectrum shape,
prompt-response coupling, application alignment, and semantic stability.
These profiles characterize *how* a model responds, not just whether
it is correct — the distinction between a personality inventory and an
IQ test.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field

from model_equality_testing.distribution import CompletionSample
from .embeddings import embed_sample
from .quantum_metrics import (
    full_density_matrix,
    pca_density_matrix,
    fit_pca,
    trace_distance,
    von_neumann_entropy,
)
from .hrr import (
    hrr_embed_sample,
    effective_rank,
    prompt_sensitivity,
)


@dataclass
class BehavioralProfile:
    """Behavioral profile of an LLM based on its completion distribution.

    Attributes:
        eigenvalues: Sorted eigenvalues (descending) of the density matrix
        effective_rank: Number of distinct behavioral modes (exp(entropy))
        von_neumann_entropy: Entropy of the density matrix
        hrr_eigenvalues: Eigenvalues of the HRR-bound density matrix
        hrr_effective_rank: Effective rank with prompt binding
        prompt_sensitivity_ratio: hrr_effective_rank / effective_rank
        top_k_explained: Fraction of trace explained by top-k eigenvalues
    """
    eigenvalues: np.ndarray
    effective_rank: float
    von_neumann_entropy: float
    hrr_eigenvalues: np.ndarray
    hrr_effective_rank: float
    prompt_sensitivity_ratio: float
    top_k_explained: Dict[int, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Effective rank (SBERT):  {self.effective_rank:.1f}",
            f"Effective rank (HRR):   {self.hrr_effective_rank:.1f}",
            f"Prompt sensitivity:     {self.prompt_sensitivity_ratio:.2f}",
            f"Von Neumann entropy:    {self.von_neumann_entropy:.4f}",
        ]
        for k, frac in sorted(self.top_k_explained.items()):
            lines.append(f"Top-{k} eigenvalues:     {frac:.1%} of trace")
        return "\n".join(lines)


def compute_behavioral_profile(
    sample: CompletionSample,
    embedding_model: str = "all-mpnet-base-v2",
    batch_size: int = 32,
    pca_k: int = 0,
    hrr_seed: int = 42,
    top_k_values: List[int] = None,
) -> BehavioralProfile:
    """Compute a full behavioral profile for one model's completions.

    Args:
        sample: CompletionSample with unicode codepoint completions
        embedding_model: SBERT model name
        batch_size: Batch size for embedding
        pca_k: PCA components (0 for full-space)
        hrr_seed: Random seed for HRR prompt vectors
        top_k_values: List of k values for top-k explained variance
            (default: [3, 5, 10])

    Returns:
        BehavioralProfile dataclass
    """
    if top_k_values is None:
        top_k_values = [3, 5, 10]

    embeddings = embed_sample(sample, model_name=embedding_model, batch_size=batch_size)

    # SBERT density matrix
    if pca_k > 0:
        pca = fit_pca(embeddings, k=pca_k)
        rho = pca_density_matrix(embeddings, pca)
    else:
        rho = full_density_matrix(embeddings)

    eigs = np.sort(np.linalg.eigvalsh(rho))[::-1]
    eigs_positive = eigs[eigs > 1e-10]
    entropy = -np.sum(eigs_positive * np.log(eigs_positive))
    eff_rank = np.exp(entropy)

    # HRR density matrix
    hrr_emb = hrr_embed_sample(sample, embeddings, seed=hrr_seed)
    if pca_k > 0:
        pca_hrr = fit_pca(hrr_emb, k=pca_k)
        rho_hrr = pca_density_matrix(hrr_emb, pca_hrr)
    else:
        rho_hrr = full_density_matrix(hrr_emb)

    hrr_eigs = np.sort(np.linalg.eigvalsh(rho_hrr))[::-1]
    hrr_eigs_positive = hrr_eigs[hrr_eigs > 1e-10]
    hrr_entropy = -np.sum(hrr_eigs_positive * np.log(hrr_eigs_positive))
    hrr_eff_rank = np.exp(hrr_entropy)

    # Top-k explained
    top_k_explained = {}
    for k in top_k_values:
        if k <= len(eigs_positive):
            top_k_explained[k] = float(np.sum(eigs_positive[:k]))

    return BehavioralProfile(
        eigenvalues=eigs,
        effective_rank=float(eff_rank),
        von_neumann_entropy=float(entropy),
        hrr_eigenvalues=hrr_eigs,
        hrr_effective_rank=float(hrr_eff_rank),
        prompt_sensitivity_ratio=float(hrr_eff_rank / eff_rank),
        top_k_explained=top_k_explained,
    )


def compare_profiles(
    sample1: CompletionSample,
    sample2: CompletionSample,
    label1: str = "Model A",
    label2: str = "Model B",
    embedding_model: str = "all-mpnet-base-v2",
    batch_size: int = 32,
    pca_k: int = 0,
    hrr_seed: int = 42,
) -> Dict:
    """Compare behavioral profiles of two models.

    Computes profiles for both models and measures the distances between
    them in both SBERT and HRR spaces.

    Args:
        sample1: CompletionSample from first model
        sample2: CompletionSample from second model
        label1: Label for first model
        label2: Label for second model
        embedding_model: SBERT model name
        batch_size: Batch size for embedding
        pca_k: PCA components (0 for full-space)
        hrr_seed: Random seed for HRR prompt vectors

    Returns:
        Dict with profiles and comparison metrics
    """
    emb1 = embed_sample(sample1, model_name=embedding_model, batch_size=batch_size)
    emb2 = embed_sample(sample2, model_name=embedding_model, batch_size=batch_size)

    # SBERT density matrices
    if pca_k > 0:
        combined = np.concatenate([emb1, emb2], axis=0)
        pca = fit_pca(combined, k=pca_k)
        rho1 = pca_density_matrix(emb1, pca)
        rho2 = pca_density_matrix(emb2, pca)
    else:
        rho1 = full_density_matrix(emb1)
        rho2 = full_density_matrix(emb2)

    sbert_trace_dist = trace_distance(rho1, rho2)
    sbert_vn_div = abs(von_neumann_entropy(rho1) - von_neumann_entropy(rho2))

    # HRR density matrices (need shared prompt vectors)
    all_prompt_ids = np.concatenate([
        sample1.prompt_sample.numpy(),
        sample2.prompt_sample.numpy(),
    ])
    from .hrr import generate_prompt_id_vectors, bind_embeddings_to_prompts
    d = emb1.shape[1]
    prompt_vectors = generate_prompt_id_vectors(all_prompt_ids, d, seed=hrr_seed)

    hrr1 = bind_embeddings_to_prompts(emb1, sample1.prompt_sample.numpy(), prompt_vectors)
    hrr2 = bind_embeddings_to_prompts(emb2, sample2.prompt_sample.numpy(), prompt_vectors)

    if pca_k > 0:
        combined_hrr = np.concatenate([hrr1, hrr2], axis=0)
        pca_hrr = fit_pca(combined_hrr, k=pca_k)
        rho_hrr1 = pca_density_matrix(hrr1, pca_hrr)
        rho_hrr2 = pca_density_matrix(hrr2, pca_hrr)
    else:
        rho_hrr1 = full_density_matrix(hrr1)
        rho_hrr2 = full_density_matrix(hrr2)

    hrr_trace_dist = trace_distance(rho_hrr1, rho_hrr2)
    hrr_vn_div = abs(von_neumann_entropy(rho_hrr1) - von_neumann_entropy(rho_hrr2))

    # Per-model profiles
    def _quick_profile(rho, rho_hrr):
        eigs = np.sort(np.linalg.eigvalsh(rho))[::-1]
        eigs_p = eigs[eigs > 1e-10]
        ent = -np.sum(eigs_p * np.log(eigs_p))

        hrr_eigs = np.sort(np.linalg.eigvalsh(rho_hrr))[::-1]
        hrr_eigs_p = hrr_eigs[hrr_eigs > 1e-10]
        hrr_ent = -np.sum(hrr_eigs_p * np.log(hrr_eigs_p))

        return {
            "effective_rank": float(np.exp(ent)),
            "hrr_effective_rank": float(np.exp(hrr_ent)),
            "prompt_sensitivity": float(np.exp(hrr_ent) / np.exp(ent)),
            "von_neumann_entropy": float(ent),
        }

    return {
        label1: _quick_profile(rho1, rho_hrr1),
        label2: _quick_profile(rho2, rho_hrr2),
        "comparison": {
            "sbert_trace_distance": float(sbert_trace_dist),
            "sbert_vn_divergence": float(sbert_vn_div),
            "hrr_trace_distance": float(hrr_trace_dist),
            "hrr_vn_divergence": float(hrr_vn_div),
        },
    }


def alignment_score(
    candidate_sample: CompletionSample,
    reference_sample: CompletionSample,
    embedding_model: str = "all-mpnet-base-v2",
    batch_size: int = 32,
    pca_k: int = 0,
    use_hrr: bool = True,
    hrr_seed: int = 42,
) -> Dict[str, float]:
    """Measure how closely a candidate model's outputs align to a reference.

    The reference could be gold-standard outputs, human-written responses,
    or a known-good model. Returns trace distance (lower = more aligned)
    in both SBERT and HRR spaces.

    Args:
        candidate_sample: CompletionSample from the model being evaluated
        reference_sample: CompletionSample from the reference/gold standard
        embedding_model: SBERT model name
        batch_size: Batch size for embedding
        pca_k: PCA components (0 for full-space)
        use_hrr: Whether to also compute HRR-space alignment
        hrr_seed: Random seed for HRR prompt vectors

    Returns:
        Dict with alignment scores (trace distance, lower is better)
    """
    emb_cand = embed_sample(candidate_sample, model_name=embedding_model, batch_size=batch_size)
    emb_ref = embed_sample(reference_sample, model_name=embedding_model, batch_size=batch_size)

    if pca_k > 0:
        combined = np.concatenate([emb_cand, emb_ref], axis=0)
        pca = fit_pca(combined, k=pca_k)
        rho_cand = pca_density_matrix(emb_cand, pca)
        rho_ref = pca_density_matrix(emb_ref, pca)
    else:
        rho_cand = full_density_matrix(emb_cand)
        rho_ref = full_density_matrix(emb_ref)

    result = {
        "sbert_alignment": float(trace_distance(rho_cand, rho_ref)),
    }

    if use_hrr:
        all_prompt_ids = np.concatenate([
            candidate_sample.prompt_sample.numpy(),
            reference_sample.prompt_sample.numpy(),
        ])
        from .hrr import generate_prompt_id_vectors, bind_embeddings_to_prompts
        d = emb_cand.shape[1]
        prompt_vectors = generate_prompt_id_vectors(all_prompt_ids, d, seed=hrr_seed)

        hrr_cand = bind_embeddings_to_prompts(
            emb_cand, candidate_sample.prompt_sample.numpy(), prompt_vectors
        )
        hrr_ref = bind_embeddings_to_prompts(
            emb_ref, reference_sample.prompt_sample.numpy(), prompt_vectors
        )

        if pca_k > 0:
            combined_hrr = np.concatenate([hrr_cand, hrr_ref], axis=0)
            pca_hrr = fit_pca(combined_hrr, k=pca_k)
            rho_hrr_cand = pca_density_matrix(hrr_cand, pca_hrr)
            rho_hrr_ref = pca_density_matrix(hrr_ref, pca_hrr)
        else:
            rho_hrr_cand = full_density_matrix(hrr_cand)
            rho_hrr_ref = full_density_matrix(hrr_ref)

        result["hrr_alignment"] = float(trace_distance(rho_hrr_cand, rho_hrr_ref))

    return result


def semantic_stability(
    sample_original: CompletionSample,
    sample_paraphrased: CompletionSample,
    embedding_model: str = "all-mpnet-base-v2",
    batch_size: int = 32,
    pca_k: int = 0,
) -> Dict[str, float]:
    """Measure sensitivity of model outputs to prompt paraphrasing.

    Give the same prompts in original and paraphrased forms. Low trace
    distance means the model captures intent rather than surface form.

    Note: The two samples should use different prompt IDs (original vs
    paraphrased), so HRR binding is intentionally omitted here — we
    want to measure semantic similarity *despite* different prompts.

    Args:
        sample_original: Completions from original prompts
        sample_paraphrased: Completions from paraphrased prompts
        embedding_model: SBERT model name
        batch_size: Batch size for embedding
        pca_k: PCA components (0 for full-space)

    Returns:
        Dict with stability metrics
    """
    emb_orig = embed_sample(sample_original, model_name=embedding_model, batch_size=batch_size)
    emb_para = embed_sample(sample_paraphrased, model_name=embedding_model, batch_size=batch_size)

    if pca_k > 0:
        combined = np.concatenate([emb_orig, emb_para], axis=0)
        pca = fit_pca(combined, k=pca_k)
        rho_orig = pca_density_matrix(emb_orig, pca)
        rho_para = pca_density_matrix(emb_para, pca)
    else:
        rho_orig = full_density_matrix(emb_orig)
        rho_para = full_density_matrix(emb_para)

    return {
        "trace_distance": float(trace_distance(rho_orig, rho_para)),
        "vn_divergence": float(
            abs(von_neumann_entropy(rho_orig) - von_neumann_entropy(rho_para))
        ),
    }
