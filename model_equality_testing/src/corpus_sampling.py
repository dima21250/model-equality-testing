import pandas as pd
from typing import NamedTuple, List
import torch
from model_equality_testing.distribution import CompletionSample


class CorpusRatio(NamedTuple):
    """Ratios for the three corpus domains.

    The default ratio follows the specification 5 (reviews) : 3 (news) : 3 (tweets).
    """
    reviews: int = 5
    news: int = 3
    tweets: int = 3


def sample_corpus(
    models: List[str],
    data: dict,
    k: int = 1,
) -> dict:
    """Create subsamples of the loaded corpus for the given models.

    The loaded ``data`` dictionary is expected to contain a ``"samples"`` key
    mapping model identifiers to ``CompletionSample`` objects. This function
    extracts a subsample following the ``CorpusRatio`` (5 reviews, 3 news, 3
    tweets) multiplied by ``k``.

    Parameters
    ----------
    models: List[str]
        Model identifiers for which to draw samples.
    data: dict
        The full data dictionary loaded from the pickle.
    k: int, optional
        Multiplicative factor for the base ratio. ``k=1`` yields the base ratio
        of ``CorpusRatio``; ``k=2`` doubles the number of samples per category.

    Returns
    -------
    dict
        A dictionary mapping each model name to a new ``CompletionSample``
        containing the subsampled prompts and completions.
    """
    if "samples" not in data:
        raise ValueError("Loaded data does not contain a 'samples' key.")

    samples = data["samples"]
    ratio = CorpusRatio()
    subsampled = {}
    for model in models:
        if model not in samples:
            raise KeyError(f"Model '{model}' not found in loaded samples.")
        orig = samples[model]
        # Assume prompts are encoded as integers 0=reviews,1=news,2=tweets
        prompt_tensor = orig.prompt_sample
        completion_tensor = orig.completion_sample
        # Gather indices for each category
        idx_reviews = (prompt_tensor == 0).nonzero(as_tuple=True)[0]
        idx_news = (prompt_tensor == 1).nonzero(as_tuple=True)[0]
        idx_tweets = (prompt_tensor == 2).nonzero(as_tuple=True)[0]
        # Helper to sample without replacement, fallback to replacement if needed
        def sample_indices(idxs, n):
            if len(idxs) == 0:
                return torch.tensor([], dtype=torch.long)
            if len(idxs) < n:
                # sample with replacement to satisfy count
                return idxs[torch.randint(len(idxs), (n,))]
            perm = torch.randperm(len(idxs))[:n]
            return idxs[perm]
        sel_reviews = sample_indices(idx_reviews, k * ratio.reviews)
        sel_news = sample_indices(idx_news, k * ratio.news)
        sel_tweets = sample_indices(idx_tweets, k * ratio.tweets)
        # Concatenate selected indices
        selected_idx = torch.cat([sel_reviews, sel_news, sel_tweets])
        # Preserve original order (optional)
        selected_idx, _ = torch.sort(selected_idx)
        sub_prompts = prompt_tensor[selected_idx]
        sub_completions = completion_tensor[selected_idx]
        # Create new CompletionSample preserving the same number of prompts (m)
        sub_sample = CompletionSample(sub_prompts, sub_completions, orig.m)
        subsampled[model] = sub_sample
    return subsampled
