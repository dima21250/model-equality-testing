"""
Pre-trained embedding generation for quantum-inspired metrics.

This module provides a unified interface to pre-trained embedding models
(SBERT, FLAIR, BGE, etc.) for converting LLM text outputs into dense
semantic vectors. These embeddings are used to compute quantum-inspired
metrics like trace distance and quantum relative entropy.
"""

import numpy as np
import logging
from typing import List, Literal
from functools import lru_cache
from model_equality_testing.distribution import CompletionSample
from .features import decode_sample_to_strings


@lru_cache(maxsize=4)
def _load_sbert_model(model_name: str):
    """Cache loaded SBERT models to avoid reloading.

    Args:
        model_name: Name of the sentence-transformers model

    Returns:
        Loaded SentenceTransformer model

    Raises:
        ImportError: If sentence-transformers is not installed
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "Please install sentence-transformers to use embedding-based tests: "
            "pip install sentence-transformers"
        ) from e

    logging.info(f"Loading SBERT model: {model_name}")
    return SentenceTransformer(model_name)


class EmbeddingModel:
    """Wrapper for pre-trained embedding models.

    Provides a unified interface to different embedding model types
    (SBERT, FLAIR, BGE, etc.) with consistent encode() API.

    Examples:
        >>> # Create SBERT embedder
        >>> embedder = EmbeddingModel("all-mpnet-base-v2", model_type="sbert")
        >>> texts = ["Hello world", "Goodbye world"]
        >>> embeddings = embedder.encode(texts)
        >>> embeddings.shape
        (2, 768)
    """

    def __init__(
        self,
        model_name: str,
        model_type: Literal["sbert"] = "sbert"
    ):
        """Initialize embedding model.

        Args:
            model_name: Model identifier (e.g., "all-mpnet-base-v2")
            model_type: Type of embedding model. Currently only "sbert" is supported.

        Raises:
            ValueError: If model_type is not supported
        """
        self.model_name = model_name
        self.model_type = model_type

        if model_type == "sbert":
            self.model = _load_sbert_model(model_name)
        else:
            raise ValueError(
                f"Unsupported model_type: {model_type}. "
                "Currently only 'sbert' is supported."
            )

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        """Encode texts into dense embeddings.

        Args:
            texts: List of text strings to embed
            batch_size: Batch size for inference
            show_progress_bar: Whether to show progress bar during encoding
            convert_to_numpy: Whether to convert to numpy (vs torch tensor)

        Returns:
            Array of embeddings with shape (len(texts), embedding_dim)
        """
        if self.model_type == "sbert":
            return self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=convert_to_numpy,
            )
        else:
            raise NotImplementedError(
                f"encode() not implemented for model_type={self.model_type}"
            )

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        if self.model_type == "sbert":
            return self.model.get_sentence_embedding_dimension()
        else:
            raise NotImplementedError(
                f"embedding_dim not implemented for model_type={self.model_type}"
            )


def embed_sample(
    sample: CompletionSample,
    model_name: str = "all-mpnet-base-v2",
    model_type: str = "sbert",
    batch_size: int = 32,
    show_progress_bar: bool = False,
    warn_on_empty: bool = True,
) -> np.ndarray:
    """Embed a CompletionSample using a pre-trained model.

    Decodes the sample's completions to strings and embeds them using
    the specified pre-trained model. Empty texts (after filtering padding)
    are embedded as zero vectors.

    Args:
        sample: CompletionSample with unicode codepoint completions
        model_name: Name of the pre-trained model (default: all-mpnet-base-v2)
        model_type: Type of embedding model (default: "sbert")
        batch_size: Batch size for inference
        show_progress_bar: Whether to show progress during embedding
        warn_on_empty: Whether to log warnings for empty texts

    Returns:
        Embeddings array with shape (sample.N, embedding_dim)

    Examples:
        >>> # Embed a sample using SBERT
        >>> from model_equality_testing.dataset import load_distribution
        >>> dist = load_distribution(
        ...     model="meta-llama/Meta-Llama-3-8B-Instruct",
        ...     prompt_ids={"wikipedia_en": [0, 1, 2]},
        ...     L=500,
        ...     source="fp32",
        ...     load_in_unicode=True,
        ... )
        >>> sample = dist.sample(n=10)
        >>> embeddings = embed_sample(sample)
        >>> embeddings.shape
        (10, 768)

        >>> # Use a faster model
        >>> embeddings = embed_sample(sample, model_name="all-MiniLM-L6-v2")
        >>> embeddings.shape
        (10, 384)
    """
    # Decode completions to strings
    texts = decode_sample_to_strings(sample)

    # Create embedding model
    embedder = EmbeddingModel(model_name, model_type)

    # Handle empty texts
    empty_indices = []
    valid_texts = []
    for i, text in enumerate(texts):
        if len(text) == 0:
            empty_indices.append(i)
            if warn_on_empty:
                logging.warning(
                    f"Empty text at index {i}, will use zero vector for embedding"
                )
        else:
            valid_texts.append(text)

    # Encode valid texts
    if len(valid_texts) > 0:
        valid_embeddings = embedder.encode(
            valid_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )
    else:
        # All texts are empty
        valid_embeddings = np.zeros((0, embedder.embedding_dim))

    # Reconstruct full embedding array with zero vectors for empty texts
    embeddings = np.zeros((sample.N, embedder.embedding_dim))
    valid_idx = 0
    for i in range(sample.N):
        if i not in empty_indices:
            embeddings[i] = valid_embeddings[valid_idx]
            valid_idx += 1

    return embeddings
