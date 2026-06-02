# Kolmogorov-Smirnov Tests for Model Equality Testing

This document describes the K-S test functionality in the Model Equality Testing framework, including both the flexible framework interface and convenient wrapper functions.

## Overview

The K-S (Kolmogorov-Smirnov) two-sample test is a non-parametric test that compares the distributions of two samples. In the context of model equality testing, K-S tests can detect distributional differences in **derived features** of model completions, such as:

- **Sentiment scores** (VADER compound scores)
- **Perplexity scores** (KenLM language model scores)
- **Custom features** (text length, lexical diversity, etc.)

This is different from the core MET tests (MMD, chi-squared, L1/L2) which operate directly on completion sequences.

## Two Ways to Run K-S Tests

### 1. Convenience Functions (Recommended for Quick Analysis)

Simple, direct interface for common use cases:

```python
from model_equality_testing.algorithm import (
    vader_ks_test,
    perplexity_ks_test,
    ks_test
)

# VADER sentiment-based K-S test
pvalue, stat = vader_ks_test(sample1, sample2)

# Perplexity-based K-S test
pvalue, stat = perplexity_ks_test(sample1, sample2, kenlm_model_path="model.arpa")

# Generic K-S test with custom features
pvalue, stat = ks_test(sample1, sample2, feature_fn=my_feature_function)
```

### 2. Framework Interface (More Control and Flexibility)

Use `run_two_sample_test()` for integration with the full framework:

```python
from model_equality_testing.algorithm import run_two_sample_test

# VADER K-S test via framework
pvalue, stat = run_two_sample_test(
    sample1, sample2,
    stat_type="two_sample_vader_ks",
    pvalue_type="analytical_ks"  # or "permutation_pvalue"
)

# Perplexity K-S test via framework
pvalue, stat = run_two_sample_test(
    sample1, sample2,
    stat_type="two_sample_perplexity_ks",
    pvalue_type="analytical_ks",
    kenlm_model_path="model.arpa",
    granularity="word"
)
```

## P-value Methods

All K-S tests support two p-value calculation methods:

### Analytical (Fast, Recommended for Exploration)

Uses scipy's built-in analytical p-value calculation. This is efficient because it computes both the K-S statistic and p-value in a single call.

**Advantages:**
- Very fast (no resampling required)
- Exact under the null hypothesis
- Suitable for large samples

**When to use:**
- Exploratory analysis
- Quick hypothesis testing
- When computational resources are limited

```python
# Convenience function
pvalue, stat = vader_ks_test(sample1, sample2, pvalue_method="analytical")

# Framework interface
pvalue, stat = run_two_sample_test(
    sample1, sample2,
    stat_type="two_sample_vader_ks",
    pvalue_type="analytical_ks"
)
```

### Permutation (Robust, Slower)

Uses permutation resampling to empirically estimate the p-value distribution under the null hypothesis.

**Advantages:**
- More robust to distributional assumptions
- Can capture complex null distributions
- Consistent with other MET tests

**When to use:**
- Final/publication-ready results
- When you need robustness
- When you have sufficient compute

```python
# Convenience function
pvalue, stat = vader_ks_test(sample1, sample2, pvalue_method="permutation", b=1000)

# Framework interface
pvalue, stat = run_two_sample_test(
    sample1, sample2,
    stat_type="two_sample_vader_ks",
    pvalue_type="permutation_pvalue",
    b=1000
)
```

## Available K-S Tests

### 1. VADER Sentiment-based K-S Test

Tests for differences in sentiment distributions using VADER compound scores.

**Requirements:** `pip install vaderSentiment`

**Use case:** Detecting if two models produce completions with different emotional tone or sentiment patterns.

```python
from model_equality_testing.algorithm import vader_ks_test

# Quick test with analytical p-value
pvalue, stat = vader_ks_test(sample1, sample2)
print(f"p-value: {pvalue:.4f}, K-S statistic: {stat:.4f}")

# Robust test with permutation p-value
pvalue, stat = vader_ks_test(
    sample1, sample2,
    pvalue_method="permutation",
    b=1000
)
```

**Example output:**
```
p-value: 0.0023, K-S statistic: 0.2450
```

### 2. Perplexity-based K-S Test

Tests for differences in perplexity distributions using KenLM language models.

**Requirements:** `pip install kenlm sentencepiece` (sentencepiece optional)

**Use case:** Detecting if two models produce completions with different linguistic patterns, fluency, or conformity to expected language statistics.

```python
from model_equality_testing.algorithm import perplexity_ks_test

# Quick test with analytical p-value
pvalue, stat = perplexity_ks_test(
    sample1, sample2,
    kenlm_model_path="model.arpa",
    granularity="word"
)

# With custom preprocessing
pvalue, stat = perplexity_ks_test(
    sample1, sample2,
    kenlm_model_path="model.arpa",
    spm_model_path="tokenizer.model",  # Use sentencepiece
    granularity="token",
    empty_text_score="nan",  # Handle empty texts
    pvalue_method="permutation",
    b=1000
)
```

**Parameters:**
- `kenlm_model_path`: Path to trained KenLM model (.arpa or .bin)
- `spm_model_path`: Optional sentencepiece model for tokenization
- `granularity`: "word" (whitespace split), "char" (character-level), or "token" (sentencepiece)
- `empty_text_score`: Score for empty strings ("inf", "nan", or float)
- `zero_tokens_score`: Score for zero-token texts ("inf", "nan", or float)
- `warn_on_degenerate`: Whether to warn about degenerate cases

**Training a KenLM model from samples:**
```python
from model_equality_testing.src.features import train_kenlm_from_sample

# Train on a reference sample
train_kenlm_from_sample(
    reference_sample,
    output_path="model.arpa",
    granularity="word",
    order=5,  # n-gram order
    skip_empty=True
)

# Then use it for testing
pvalue, stat = perplexity_ks_test(sample1, sample2, kenlm_model_path="model.arpa")
```

### 3. Generic K-S Test with Custom Features

Tests for differences in any custom feature distribution.

**Use case:** Experimenting with novel features like text length, lexical diversity, specific token counts, etc.

```python
from model_equality_testing.algorithm import ks_test
import numpy as np

# Default: uses VADER sentiment
pvalue, stat = ks_test(sample1, sample2)

# Custom feature: text length
def length_feature(sample):
    from model_equality_testing.src.features import decode_sample_to_strings
    texts = decode_sample_to_strings(sample)
    return np.array([len(text) for text in texts])

pvalue, stat = ks_test(sample1, sample2, feature_fn=length_feature)

# Custom feature: average word length
def avg_word_length_feature(sample):
    from model_equality_testing.src.features import decode_sample_to_strings
    texts = decode_sample_to_strings(sample)
    scores = []
    for text in texts:
        words = text.split()
        if words:
            scores.append(np.mean([len(w) for w in words]))
        else:
            scores.append(0.0)
    return np.array(scores)

pvalue, stat = ks_test(sample1, sample2, feature_fn=avg_word_length_feature)

# Using functools.partial for parameterized features
from functools import partial
from model_equality_testing.src.features import get_perplexity_scores

ppl_fn = partial(get_perplexity_scores, kenlm_model_path="model.arpa", granularity="word")
pvalue, stat = ks_test(sample1, sample2, feature_fn=ppl_fn)
```

**Feature function requirements:**
- Input: `CompletionSample` object
- Output: `np.ndarray` of scalar scores (one per completion)
- Shape: `(N,)` where N is the number of completions in the sample

## Framework Integration: The Generic K-S Interface

For maximum flexibility, use `two_sample_ks_statistic` with custom feature functions via `run_two_sample_test`:

```python
from model_equality_testing.algorithm import run_two_sample_test
from functools import partial

# Define your feature function
from model_equality_testing.src.features import get_perplexity_scores
feature_fn = partial(get_perplexity_scores, kenlm_model_path="model.arpa")

# Use with analytical p-value
pvalue, stat = run_two_sample_test(
    sample1, sample2,
    stat_type="two_sample_ks_statistic",
    pvalue_type="analytical_ks",
    feature_fn=feature_fn
)

# Use with permutation p-value
pvalue, stat = run_two_sample_test(
    sample1, sample2,
    stat_type="two_sample_ks_statistic",
    pvalue_type="permutation_pvalue",
    feature_fn=feature_fn,
    b=1000
)
```

This approach is documented in detail in [`OPTION_B_USAGE.md`](OPTION_B_USAGE.md).

## Complete Example: Comparing Two Models

```python
from model_equality_testing.dataset import load_distribution
from model_equality_testing.algorithm import vader_ks_test, perplexity_ks_test

# Load samples from the dataset
p1 = load_distribution(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    prompt_ids={"wikipedia_en": [0, 1, 2]},
    L=200,
    source="fp32",
    load_in_unicode=True
)

p2 = load_distribution(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    prompt_ids={"wikipedia_en": [0, 1, 2]},
    L=200,
    source="int8",
    load_in_unicode=True
)

# Draw samples
sample1 = p1.sample(n=500)
sample2 = p2.sample(n=500)

# Test 1: Sentiment distribution difference
print("VADER K-S Test:")
pvalue_vader, stat_vader = vader_ks_test(sample1, sample2, pvalue_method="analytical")
print(f"  K-S statistic: {stat_vader:.4f}")
print(f"  p-value: {pvalue_vader:.4f}")
if pvalue_vader < 0.05:
    print("  ✓ Significant difference in sentiment distributions")
else:
    print("  ✗ No significant difference in sentiment distributions")

# Test 2: Perplexity distribution difference
# First train a KenLM model on a corpus
from model_equality_testing.src.features import train_kenlm_from_sample
train_kenlm_from_sample(sample1, output_path="ref_model.arpa", order=5)

print("\nPerplexity K-S Test:")
pvalue_ppl, stat_ppl = perplexity_ks_test(
    sample1, sample2,
    kenlm_model_path="ref_model.arpa",
    pvalue_method="analytical"
)
print(f"  K-S statistic: {stat_ppl:.4f}")
print(f"  p-value: {pvalue_ppl:.4f}")
if pvalue_ppl < 0.05:
    print("  ✓ Significant difference in perplexity distributions")
else:
    print("  ✗ No significant difference in perplexity distributions")
```

## Performance Comparison

| Method | Sample Size | b (permutations) | Time |
|--------|-------------|------------------|------|
| VADER (analytical) | 500 | N/A | ~0.1s |
| VADER (permutation) | 500 | 1000 | ~10s |
| Perplexity (analytical) | 500 | N/A | ~1s |
| Perplexity (permutation) | 500 | 1000 | ~60s |

**Note:** Perplexity tests are slower due to KenLM scoring overhead. The analytical method is ~10-60x faster than permutation.

## When to Use K-S Tests vs Core MET Tests

**Use K-S tests when:**
- You want to test specific hypotheses about derived features (sentiment, fluency, etc.)
- You have a small sample size (K-S works well with small n)
- You want fast exploratory analysis (analytical p-values)
- You want to combine multiple complementary tests

**Use core MET tests (MMD, chi-squared, L1/L2) when:**
- You want to detect any distributional difference in completions
- You care about the full sequence distribution
- You're conducting the primary equality test (not a supplementary analysis)

**Best practice:** Use both! K-S tests can provide interpretable insights about *why* distributions differ, while core MET tests detect *that* they differ.

## Available Test Statistics

The following K-S test statistics are registered in `IMPLEMENTED_TESTS`:

- `"two_sample_vader_ks"` - VADER sentiment-based K-S test
- `"two_sample_perplexity_ks"` - Perplexity-based K-S test
- `"two_sample_ks_statistic"` - Generic K-S test with custom features
- `"two_sample_ks"` - Legacy K-S test on lexicographical ordering of sequences (not recommended for most use cases)

## See Also

- [`OPTION_B_USAGE.md`](OPTION_B_USAGE.md) - Detailed guide for using the generic K-S interface
- [`LESSONS-LEARNED.md`](LESSONS-LEARNED.md) - Implementation details and troubleshooting
- [Main README](README.md) - Package overview and installation
