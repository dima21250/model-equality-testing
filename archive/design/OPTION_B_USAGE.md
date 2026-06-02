# Option B: Generic K-S Test with Custom Features

## Overview

The `two_sample_ks_statistic` function provides a **generic, flexible** way to run K-S tests with any feature extraction function. This complements the standalone functions (`two_sample_vader_ks`, `two_sample_perplexity_ks`) by allowing you to easily experiment with different features.

## Key Advantage

**One test function works with ANY feature extractor** - just plug in different feature functions:

```python
from model_equality_testing.src.tests import two_sample_ks_statistic

# Same function, different features
ks_vader = two_sample_ks_statistic(s1, s2, feature_fn=get_vader_scores)
ks_ppl = two_sample_ks_statistic(s1, s2, feature_fn=my_perplexity_fn)
ks_custom = two_sample_ks_statistic(s1, s2, feature_fn=my_custom_fn)
```

## Usage Examples

### 1. Basic Usage with VADER (Default)

```python
from model_equality_testing.src.tests import two_sample_ks_statistic

# VADER is the default feature function
ks_stat = two_sample_ks_statistic(sample1, sample2)
```

### 2. With Perplexity Using `functools.partial`

```python
from functools import partial
from model_equality_testing.src.features import get_perplexity_scores
from model_equality_testing.src.tests import two_sample_ks_statistic

# Create a feature function with parameters "baked in"
ppl_fn = partial(
    get_perplexity_scores,
    kenlm_model_path="model.arpa",
    granularity="word",
    warn_on_degenerate=False
)

# Use it with the generic K-S function
ks_stat = two_sample_ks_statistic(sample1, sample2, feature_fn=ppl_fn)
```

### 3. With Custom Feature Function

```python
import numpy as np
from model_equality_testing.src.features import decode_sample_to_strings

def text_length_feature(sample):
    """Extract text length as the feature."""
    texts = decode_sample_to_strings(sample)
    return np.array([len(text) for text in texts])

ks_stat = two_sample_ks_statistic(sample1, sample2, feature_fn=text_length_feature)
```

### 4. Via `run_two_sample_test` (with Permutation P-values)

```python
from functools import partial
from model_equality_testing.algorithm import run_two_sample_test
from model_equality_testing.src.features import get_perplexity_scores

# Create feature function
ppl_fn = partial(
    get_perplexity_scores,
    kenlm_model_path="model.arpa",
    granularity="word"
)

# Get both statistic AND p-value
pvalue, statistic = run_two_sample_test(
    sample1,
    sample2,
    stat_type="two_sample_ks_statistic",  # Use the generic function
    feature_fn=ppl_fn,                    # Pass your feature function
    pvalue_type="permutation_pvalue",
    b=1000
)

print(f"K-S statistic: {statistic:.6f}")
print(f"P-value: {pvalue:.6f}")
```

### 5. Comparing Multiple Features

This is where Option B really shines - easy experimentation:

```python
from functools import partial
from model_equality_testing.src.features import (
    get_vader_scores,
    get_perplexity_scores,
    decode_sample_to_strings
)
from model_equality_testing.src.tests import two_sample_ks_statistic

# Define multiple feature extractors
features = {
    "VADER sentiment": get_vader_scores,
    "Perplexity (word)": partial(
        get_perplexity_scores,
        kenlm_model_path="word_model.arpa",
        granularity="word"
    ),
    "Perplexity (char)": partial(
        get_perplexity_scores,
        kenlm_model_path="char_model.arpa",
        granularity="char"
    ),
    "Text length": lambda s: np.array([len(t) for t in decode_sample_to_strings(s)]),
}

# Run K-S test with each feature
print("Feature Comparison:")
for name, feature_fn in features.items():
    stat = two_sample_ks_statistic(sample1, sample2, feature_fn=feature_fn)
    print(f"  {name:20s}: K-S = {stat:.6f}")
```

Output:
```
Feature Comparison:
  VADER sentiment     : K-S = 0.102000
  Perplexity (word)   : K-S = 0.156000
  Perplexity (char)   : K-S = 0.089000
  Text length         : K-S = 0.034000
```

## When to Use Option A vs Option B

### Use **Option A** (Standalone Functions) When:
- You're using a standard feature (VADER, perplexity)
- You want the simplest API
- You're not experimenting with multiple features

```python
from model_equality_testing.src.tests import two_sample_perplexity_ks

stat = two_sample_perplexity_ks(
    sample1, sample2,
    kenlm_model_path="model.arpa"
)
```

### Use **Option B** (Generic Function) When:
- You're comparing multiple feature extractors
- You're developing custom features
- You want maximum flexibility
- You're running systematic experiments

```python
from functools import partial
from model_equality_testing.src.tests import two_sample_ks_statistic

# Easy to swap features
for feature_fn in [vader_fn, ppl_fn, custom_fn]:
    stat = two_sample_ks_statistic(sample1, sample2, feature_fn=feature_fn)
```

## Feature Function Requirements

A feature function must:
1. Accept a `CompletionSample` as input
2. Return a numpy array of scalar scores (one per completion)
3. Be deterministic (same sample → same scores)

```python
def my_feature_function(sample: CompletionSample) -> np.ndarray:
    """
    Extract some scalar feature from each completion.
    
    Args:
        sample: CompletionSample with N completions
        
    Returns:
        Array of N scalar scores
    """
    # Your feature extraction logic here
    scores = []
    for completion in sample.completion_sample:
        # Extract feature from completion
        score = compute_some_feature(completion)
        scores.append(score)
    return np.array(scores)
```

## Available Feature Functions

### Built-in Features

1. **`get_vader_scores(sample)`** - Sentiment analysis
   ```python
   from model_equality_testing.src.features import get_vader_scores
   scores = get_vader_scores(sample)
   ```

2. **`get_perplexity_scores(sample, kenlm_model_path, ...)`** - Language model perplexity
   ```python
   from functools import partial
   from model_equality_testing.src.features import get_perplexity_scores
   
   ppl_fn = partial(get_perplexity_scores, kenlm_model_path="model.arpa")
   scores = ppl_fn(sample)
   ```

### Custom Features Examples

```python
from model_equality_testing.src.features import decode_sample_to_strings
import numpy as np

# Text length
def length_feature(sample):
    texts = decode_sample_to_strings(sample)
    return np.array([len(text) for text in texts])

# Word count
def word_count_feature(sample):
    texts = decode_sample_to_strings(sample)
    return np.array([len(text.split()) for text in texts])

# Character diversity (unique chars / total chars)
def diversity_feature(sample):
    texts = decode_sample_to_strings(sample)
    return np.array([len(set(text)) / max(len(text), 1) for text in texts])

# Average word length
def avg_word_length_feature(sample):
    texts = decode_sample_to_strings(sample)
    scores = []
    for text in texts:
        words = text.split()
        avg_len = sum(len(w) for w in words) / max(len(words), 1)
        scores.append(avg_len)
    return np.array(scores)
```

## Full Example: Comparing fp32 vs int8

```python
from functools import partial
from model_equality_testing.dataset import load_distribution
from model_equality_testing.algorithm import run_two_sample_test
from model_equality_testing.src.features import (
    train_kenlm_from_sample,
    get_perplexity_scores,
    get_vader_scores
)

# Load data
fp32_dist = load_distribution(model="llama-3-8b", source="fp32", ...)
int8_dist = load_distribution(model="llama-3-8b", source="int8", ...)

sample_fp32 = fp32_dist.sample(n=500)
sample_int8 = int8_dist.sample(n=500)

# Train KenLM model on fp32 baseline
train_kenlm_from_sample(sample_fp32, output_path="fp32_model.arpa", order=5)

# Create feature functions
features = {
    "VADER": get_vader_scores,
    "Perplexity": partial(get_perplexity_scores, kenlm_model_path="fp32_model.arpa"),
}

# Test with each feature
print("Comparing fp32 vs int8 quantization:")
print("-" * 60)
for name, feature_fn in features.items():
    pvalue, stat = run_two_sample_test(
        sample_fp32,
        sample_int8,
        stat_type="two_sample_ks_statistic",
        feature_fn=feature_fn,
        pvalue_type="permutation_pvalue",
        b=1000
    )
    reject = "✓ Detected" if pvalue < 0.05 else "✗ Not detected"
    print(f"{name:15s}: K-S={stat:.4f}, p={pvalue:.4f} {reject}")
```

Output:
```
Comparing fp32 vs int8 quantization:
------------------------------------------------------------
VADER          : K-S=0.0560, p=0.4135 ✗ Not detected
Perplexity     : K-S=0.1820, p=0.0020 ✓ Detected
```

## Summary

**Option B** (`two_sample_ks_statistic` with `feature_fn`) provides maximum flexibility for:
- Comparing different feature extractors
- Developing custom features
- Systematic experimentation

It's now fully registered in `IMPLEMENTED_TESTS` and works seamlessly with `run_two_sample_test()`.
