import numpy as np
import logging
from typing import Union
from functools import lru_cache

def decode_sample_to_strings(sample):
    """
    Decode a sample's completion_sample (unicode codepoints) to strings.

    Args:
        sample: Sample object with completion_sample attribute containing unicode codepoints

    Returns:
        List of decoded strings
    """
    decoded_strings = []
    for i in range(sample.N):
        codepoints = sample.completion_sample[i].tolist()
        valid_codepoints = [c for c in codepoints if c != -1]
        text = "".join([chr(c) for c in valid_codepoints])
        decoded_strings.append(text)
    return decoded_strings

def get_vader_scores(sample):
    """
    Layman: This turns a pile of text into a pile of numbers (sentiment).
    """
    # Assuming a hypothetical vader implementation
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    # Extract strings from the numeric completion sample
    # (Simplified: assumes you've decoded the unicode codepoints back to strings)
    decoded_strings = decode_sample_to_strings(sample)
    return np.array([analyzer.polarity_scores(s)['compound'] for s in decoded_strings])


@lru_cache(maxsize=2)
def _load_kenlm_model(model_path: str):
    """Cache loaded KenLM models to avoid reloading."""
    try:
        import kenlm
    except ImportError as e:
        raise ImportError(
            "Please install kenlm to use perplexity-based tests: pip install kenlm"
        ) from e
    return kenlm.Model(model_path)


@lru_cache(maxsize=2)
def _load_sentencepiece_model(spm_model_path: str):
    """Cache loaded SentencePiece models to avoid reloading."""
    try:
        import sentencepiece as spm
    except ImportError as e:
        raise ImportError(
            "Please install sentencepiece to use tokenization: pip install sentencepiece"
        ) from e
    sp = spm.SentencePieceProcessor(model_file=spm_model_path)
    return sp


def _parse_score_value(score_str: Union[float, str]) -> float:
    """Convert score configuration string to float value."""
    if isinstance(score_str, (int, float)):
        return float(score_str)
    if score_str == "inf":
        return float('inf')
    elif score_str == "nan":
        return np.nan
    else:
        try:
            return float(score_str)
        except ValueError:
            raise ValueError(f"Invalid score value: {score_str}. Expected 'inf', 'nan', or numeric value.")


def get_perplexity_scores(
    sample,
    kenlm_model_path: str,
    spm_model_path: str = None,
    granularity: str = "word",
    empty_text_score: Union[float, str] = "inf",
    zero_tokens_score: Union[float, str] = "inf",
    warn_on_degenerate: bool = True,
) -> np.ndarray:
    """
    Extract perplexity scores from sample completions using a KenLM model.

    Args:
        sample: CompletionSample with unicode codepoint completions
        kenlm_model_path: Path to .arpa or .bin KenLM model file
        spm_model_path: Optional path to sentencepiece model for preprocessing
        granularity: "word" (split whitespace), "char" (character-level), or "token" (sentencepiece)
        empty_text_score: Score for empty strings - "inf", "nan", or float value
        zero_tokens_score: Score when text exists but has zero tokens/words
        warn_on_degenerate: Log warnings for degenerate cases

    Returns:
        Array of perplexity scores (one per sample)

    Examples:
        >>> # Basic usage with word-level perplexity
        >>> scores = get_perplexity_scores(sample, "model.arpa")

        >>> # With sentencepiece preprocessing
        >>> scores = get_perplexity_scores(
        ...     sample,
        ...     kenlm_model_path="model.arpa",
        ...     spm_model_path="tokenizer.model",
        ...     granularity="token"
        ... )
    """
    # Load KenLM model (cached)
    model = _load_kenlm_model(kenlm_model_path)

    # Parse score configuration
    empty_score = _parse_score_value(empty_text_score)
    zero_score = _parse_score_value(zero_tokens_score)

    # Decode completions to strings
    decoded_strings = decode_sample_to_strings(sample)

    # Load sentencepiece if needed
    sp = None
    if spm_model_path is not None:
        sp = _load_sentencepiece_model(spm_model_path)
        if granularity not in ["token"]:
            logging.warning(
                f"sentencepiece model provided but granularity='{granularity}'. "
                "Consider using granularity='token' for sentencepiece preprocessing."
            )

    scores = []
    for text in decoded_strings:
        # Handle empty text
        if len(text) == 0:
            if warn_on_degenerate:
                logging.warning("Empty text encountered, using configured empty_text_score")
            scores.append(empty_score)
            continue

        # Preprocess text based on granularity
        if sp is not None and granularity == "token":
            # Sentencepiece tokenization
            tokens = sp.encode(text, out_type=str)
            processed_text = " ".join(tokens)
            num_tokens = len(tokens)
        elif granularity == "word":
            # Word-level: split on whitespace
            processed_text = text
            num_tokens = len(text.split())
        elif granularity == "char":
            # Character-level
            processed_text = text
            num_tokens = len(text)
        else:
            raise ValueError(
                f"Invalid granularity: {granularity}. "
                "Expected 'word', 'char', or 'token' (with spm_model_path)."
            )

        # Handle zero tokens after preprocessing
        if num_tokens == 0:
            if warn_on_degenerate:
                logging.warning(
                    f"Zero tokens after preprocessing (granularity={granularity}), "
                    "using configured zero_tokens_score"
                )
            scores.append(zero_score)
            continue

        # Compute perplexity
        # KenLM score returns log10 probability
        # Perplexity = 10^(-log_prob / num_tokens)
        log_prob = model.score(processed_text)
        perplexity = 10 ** (-log_prob / num_tokens)
        scores.append(perplexity)

    return np.array(scores)


def train_kenlm_from_sample(
    sample,
    output_path: str,
    spm_model_path: str = None,
    granularity: str = "word",
    order: int = 5,
    skip_empty: bool = True,
    warn_on_degenerate: bool = True,
    tmp_dir: str = None,
) -> None:
    """
    Train a KenLM language model from CompletionSample data.

    Requires the `lmplz` binary from KenLM to be installed separately.
    Installation instructions:
    - Ubuntu/Debian: apt-get install kenlm
    - macOS: brew install kenlm
    - Or build from source: https://github.com/kpu/kenlm

    Args:
        sample: CompletionSample to train from
        output_path: Where to save the trained .arpa model
        spm_model_path: Optional sentencepiece model for preprocessing
        granularity: "word", "char", or "token" - must match scoring usage
        order: N-gram order (default 5)
        skip_empty: Skip empty/degenerate texts during training
        warn_on_degenerate: Log warnings for skipped texts
        tmp_dir: Directory for temporary files (default: system temp)

    Raises:
        RuntimeError: If lmplz binary not found

    Examples:
        >>> # Train a 3-gram model from sample data
        >>> train_kenlm_from_sample(
        ...     sample,
        ...     output_path="model.arpa",
        ...     order=3
        ... )

        >>> # Train with sentencepiece preprocessing
        >>> train_kenlm_from_sample(
        ...     sample,
        ...     output_path="model.arpa",
        ...     spm_model_path="tokenizer.model",
        ...     granularity="token",
        ...     order=5
        ... )
    """
    import shutil
    import subprocess
    import tempfile
    import os

    # Check if lmplz is available
    if shutil.which('lmplz') is None:
        raise RuntimeError(
            "lmplz binary not found. Please install KenLM binaries:\n"
            "  - Ubuntu/Debian: apt-get install kenlm\n"
            "  - macOS: Build from source (https://github.com/kpu/kenlm)\n"
            "  - Or use a pre-trained KenLM model with get_perplexity_scores()"
        )

    # Try running lmplz to verify it works
    # Note: lmplz --help exits with code 1, but that's normal
    try:
        result = subprocess.run(
            ['lmplz', '--help'],
            capture_output=True,
            text=True,
            timeout=5
        )
        # Check if the output looks like lmplz help text
        if "Builds unpruned language models" not in result.stderr and \
           "Builds unpruned language models" not in result.stdout:
            raise RuntimeError(
                f"lmplz binary found but not working properly.\n"
                f"Exit code: {result.returncode}\n"
                f"Output: {result.stdout[:200]}\n"
                f"Error: {result.stderr[:200]}\n\n"
                "This is often caused by missing/incompatible boost libraries.\n"
                "Workaround: Use a pre-trained KenLM model instead:\n"
                "  - Train externally: https://github.com/kpu/kenlm#using-the-lmplz-binary\n"
                "  - Or download a pre-trained model"
            )
    except subprocess.TimeoutExpired:
        raise RuntimeError("lmplz binary found but hangs when executed")

    # Load sentencepiece if needed
    sp = None
    if spm_model_path is not None:
        sp = _load_sentencepiece_model(spm_model_path)

    # Decode completions to strings
    decoded_strings = decode_sample_to_strings(sample)

    # Preprocess texts
    processed_texts = []
    skipped_count = 0

    for text in decoded_strings:
        # Handle empty text
        if len(text) == 0:
            if skip_empty:
                skipped_count += 1
                continue
            else:
                processed_texts.append("")
                continue

        # Preprocess based on granularity
        if sp is not None and granularity == "token":
            tokens = sp.encode(text, out_type=str)
            if len(tokens) == 0 and skip_empty:
                skipped_count += 1
                continue
            processed_text = " ".join(tokens)
        elif granularity == "word":
            words = text.split()
            if len(words) == 0 and skip_empty:
                skipped_count += 1
                continue
            processed_text = text
        elif granularity == "char":
            processed_text = text
        else:
            raise ValueError(
                f"Invalid granularity: {granularity}. "
                "Expected 'word', 'char', or 'token' (with spm_model_path)."
            )

        processed_texts.append(processed_text)

    if warn_on_degenerate and skipped_count > 0:
        logging.warning(f"Skipped {skipped_count} empty/degenerate texts during training")

    if len(processed_texts) == 0:
        raise ValueError("No valid texts remaining after preprocessing. Cannot train KenLM model.")

    # Write corpus to temporary file
    temp_file = None
    try:
        # Create temporary file
        fd, temp_file = tempfile.mkstemp(
            suffix=".txt",
            dir=tmp_dir,
            text=True
        )

        # Write corpus (one text per line)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            for text in processed_texts:
                f.write(text + '\n')

        # Run lmplz
        cmd = [
            'lmplz',
            '-o', str(order),
            '--text', temp_file,
            '--arpa', output_path,
            '--discount_fallback'  # Handle small/artificial datasets
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"lmplz training failed with return code {result.returncode}.\n"
                f"stderr: {result.stderr}"
            )

        logging.info(f"Successfully trained KenLM model: {output_path}")

    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)
