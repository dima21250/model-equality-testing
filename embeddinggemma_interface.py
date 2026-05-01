"""Interface to EmbeddingGemma-300M via OpenAI-compatible API.

This module provides functions to get embeddings from the NIST cluster's
EmbeddingGemma-300M model using the OpenAI Python client.

Usage:
    from embeddinggemma_interface import get_embeddings, get_embedding_dim

    embeddings = get_embeddings(["text1", "text2", "text3"])
    # Returns: (3, dim) numpy array
"""

import os
import numpy as np
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_client():
    """Get OpenAI client configured for EmbeddingGemma-300M endpoint."""
    from openai import OpenAI

    api_base = os.getenv("OPENAI_API_BASE", "https://rchat.nist.gov/api")
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment. "
            "Make sure .env file contains the API key."
        )

    return OpenAI(base_url=api_base, api_key=api_key)


def get_model_name() -> str:
    """Get the embedding model name from environment."""
    return os.getenv("EMBEDDING_MODEL_NAME", "embeddinggemma-300m")


def get_embedding(text: str, client=None) -> np.ndarray:
    """Get embedding for a single text.

    Args:
        text: Input text to embed
        client: OpenAI client (created automatically if None)

    Returns:
        1-D numpy array containing the embedding
    """
    if client is None:
        client = get_client()

    model_name = get_model_name()

    try:
        response = client.embeddings.create(model=model_name, input=text)
        return np.array(response.data[0].embedding)
    except Exception as exc:
        print(f"Embedding error for '{text[:50]}...': {exc}")
        return np.zeros(0)


def get_embeddings(
    texts: List[str],
    client=None,
    batch_size: int = 32,
    verbose: bool = True
) -> np.ndarray:
    """Get embeddings for a list of texts.

    Args:
        texts: List of input texts to embed
        client: OpenAI client (created automatically if None)
        batch_size: Number of texts to process at once (for progress tracking)
        verbose: Whether to print progress

    Returns:
        (len(texts), embedding_dim) numpy array
    """
    if client is None:
        client = get_client()

    model_name = get_model_name()
    embeddings = []

    # Process in batches for progress tracking
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        if verbose:
            print(f"  Embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}...")

        for text in batch:
            vec = get_embedding(text, client=client)
            embeddings.append(vec)

    if not embeddings:
        return np.zeros((0, 0))

    # Stack into matrix
    try:
        return np.vstack(embeddings)
    except ValueError:
        # Handle case where some embeddings failed
        dim = embeddings[0].shape[0] if embeddings[0].size else 0
        return np.zeros((len(embeddings), dim))


def get_embedding_dim(client=None) -> int:
    """Get the embedding dimension by querying with a test string.

    Args:
        client: OpenAI client (created automatically if None)

    Returns:
        Embedding dimension (e.g., 768, 1024, 2048)
    """
    test_vec = get_embedding("test", client=client)
    return test_vec.shape[0] if test_vec.size > 0 else 0


if __name__ == "__main__":
    # Quick test
    print("Testing EmbeddingGemma-300M interface...")

    # Test single embedding
    print("\n1. Single embedding test:")
    vec = get_embedding("Hello world")
    print(f"   Embedding dimension: {vec.shape[0]}")
    print(f"   First 5 values: {vec[:5]}")

    # Test batch embeddings
    print("\n2. Batch embedding test:")
    test_texts = ["The cat sat on the mat", "Machine learning is fascinating", "Python is a programming language"]
    batch_emb = get_embeddings(test_texts, batch_size=2, verbose=True)
    print(f"   Batch shape: {batch_emb.shape}")

    print("\n✓ Interface test complete!")
