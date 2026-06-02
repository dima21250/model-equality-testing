# Training KenLM Models from MET Dataset

This guide shows how to train KenLM language models using data from the Model Equality Testing dataset.

## Prerequisites

```bash
# Install required packages
pip install kenlm sentencepiece  # sentencepiece optional

# Download the MET dataset (37.1GB)
python -c "from model_equality_testing.dataset import download_dataset; download_dataset('./data')"
```

## Quick Start: Train from a Distribution

The simplest approach is to load a distribution and train directly:

```python
from model_equality_testing.dataset import load_distribution
from model_equality_testing.src.features import train_kenlm_from_sample

# Load a distribution
dist = load_distribution(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    prompt_ids={"wikipedia_en": [0, 1, 2, 3, 4]},  # 5 prompts
    L=200,  # completion length
    source="fp32",
    load_in_unicode=True,  # IMPORTANT: must be True for KenLM
    root_dir="./data"
)

# Sample completions for training
sample = dist.sample(n=1000)  # 1000 completions (5 prompts × 200 each)

# Train a 5-gram KenLM model
train_kenlm_from_sample(
    sample,
    output_path="llama3_8b_fp32.arpa",
    granularity="word",  # word-level model
    order=5,  # 5-gram
    skip_empty=True
)

print("✓ Model trained and saved to llama3_8b_fp32.arpa")
```

## Common Training Scenarios

### 1. Train from Multiple Prompt Sets

Combine different prompt sets to create a more diverse training corpus:

```python
from model_equality_testing.dataset import load_distribution
from model_equality_testing.src.features import train_kenlm_from_sample

# Load distribution with multiple datasets
dist = load_distribution(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    prompt_ids={
        "wikipedia_en": [0, 1, 2, 3, 4],      # 5 prompts
        "wikipedia_es": [0, 1, 2],            # 3 prompts
        "xsum": [0, 1]                        # 2 prompts
    },
    L=200,
    source="fp32",
    load_in_unicode=True,
    root_dir="./data"
)

# Sample from the combined distribution
sample = dist.sample(n=2000)

# Train the model
train_kenlm_from_sample(
    sample,
    output_path="llama3_8b_multilingual.arpa",
    granularity="word",
    order=5
)
```

### 2. Train a Reference Model for Two-Sample Tests

For perplexity-based K-S tests, train on a reference distribution:

```python
from model_equality_testing.dataset import load_distribution
from model_equality_testing.src.features import train_kenlm_from_sample
from model_equality_testing.algorithm import perplexity_ks_test

# Load reference distribution (e.g., fp32 baseline)
ref_dist = load_distribution(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    prompt_ids={"wikipedia_en": [0, 1, 2]},
    L=200,
    source="fp32",
    load_in_unicode=True,
    root_dir="./data"
)

# Sample and train reference model
ref_sample = ref_dist.sample(n=5000)
train_kenlm_from_sample(
    ref_sample,
    output_path="reference_model.arpa",
    granularity="word",
    order=5
)

# Now test if quantized model matches reference
test_dist = load_distribution(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    prompt_ids={"wikipedia_en": [0, 1, 2]},
    L=200,
    source="int8",
    load_in_unicode=True,
    root_dir="./data"
)

test_sample = test_dist.sample(n=500)

# Run perplexity K-S test
pvalue, stat = perplexity_ks_test(
    ref_sample[:500],  # Use subset for fair comparison
    test_sample,
    kenlm_model_path="reference_model.arpa"
)

print(f"K-S statistic: {stat:.4f}, p-value: {pvalue:.4f}")
if pvalue < 0.05:
    print("✗ Significant difference detected between fp32 and int8")
else:
    print("✓ No significant difference in perplexity distributions")
```

### 3. Using Corpus Sampling for K-S Tests

For balanced training across different prompt types:

```python
from model_equality_testing.src.corpus_sampling import sample_corpus
from model_equality_testing.src.features import train_kenlm_from_sample
import pickle

# Load pre-collected corpus data
# (Assumes you have a pickle file with corpus samples structured like corpus_sampling.py expects)
with open("corpus_data.pkl", "rb") as f:
    data = pickle.load(f)

# Sample with corpus ratios (5 reviews : 3 news : 3 tweets)
models = ["meta-llama/Meta-Llama-3-8B-Instruct"]
samples = sample_corpus(models, data, k=10)  # k=10 → 50 reviews, 30 news, 30 tweets

# Train from the balanced corpus
train_kenlm_from_sample(
    samples[models[0]],
    output_path="balanced_corpus.arpa",
    granularity="word",
    order=5
)
```

## Training Parameters

### Granularity Options

**Word-level (default, recommended):**
```python
train_kenlm_from_sample(
    sample,
    output_path="model.arpa",
    granularity="word"  # Splits on whitespace
)
```

**Character-level:**
```python
train_kenlm_from_sample(
    sample,
    output_path="char_model.arpa",
    granularity="char"  # Each character is a token
)
```

**Sentencepiece tokenization:**
```python
# First train a sentencepiece model
import sentencepiece as spm

# ... train sentencepiece model (see sentencepiece docs) ...

# Then use it for KenLM training
train_kenlm_from_sample(
    sample,
    output_path="sp_model.arpa",
    spm_model_path="tokenizer.model",
    granularity="token"
)
```

### N-gram Order

The `order` parameter controls the n-gram order:

```python
# Unigram model (order=1) - fast but less context
train_kenlm_from_sample(sample, output_path="1gram.arpa", order=1)

# Trigram model (order=3) - good balance
train_kenlm_from_sample(sample, output_path="3gram.arpa", order=3)

# 5-gram model (order=5) - more context, larger model
train_kenlm_from_sample(sample, output_path="5gram.arpa", order=5)
```

**Recommendations:**
- Order 3: Fast training, good for small datasets (<1000 samples)
- Order 5: Standard choice, good for most use cases
- Order 7+: Only for very large datasets (>10000 samples)

### Handling Degenerate Cases

```python
train_kenlm_from_sample(
    sample,
    output_path="model.arpa",
    skip_empty=True,        # Skip empty texts (recommended)
    warn_on_degenerate=True # Log warnings about skipped texts
)
```

## Data Size Recommendations

| Dataset Size | Order | Use Case |
|--------------|-------|----------|
| 100-500 | 3 | Quick testing, small prompts |
| 500-2000 | 5 | Standard K-S tests |
| 2000-10000 | 5-7 | Robust reference models |
| 10000+ | 7+ | Production-quality LMs |

## Complete Example: Comparing Quantizations

```python
from model_equality_testing.dataset import load_distribution
from model_equality_testing.src.features import train_kenlm_from_sample
from model_equality_testing.algorithm import perplexity_ks_test

# Configuration
MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
PROMPTS = {"wikipedia_en": list(range(10))}  # 10 prompts
L = 200
N_TRAIN = 2000
N_TEST = 500

# 1. Load fp32 reference distribution
print("Loading fp32 reference distribution...")
fp32_dist = load_distribution(
    model=MODEL,
    prompt_ids=PROMPTS,
    L=L,
    source="fp32",
    load_in_unicode=True,
    root_dir="./data"
)

# 2. Train reference KenLM model
print("Training reference KenLM model...")
fp32_train = fp32_dist.sample(n=N_TRAIN)
train_kenlm_from_sample(
    fp32_train,
    output_path="fp32_reference.arpa",
    granularity="word",
    order=5,
    skip_empty=True
)

# 3. Test each quantization
for quantization in ["int8", "nf4"]:
    print(f"\nTesting {quantization}...")
    
    # Load quantized distribution
    quant_dist = load_distribution(
        model=MODEL,
        prompt_ids=PROMPTS,
        L=L,
        source=quantization,
        load_in_unicode=True,
        root_dir="./data"
    )
    
    # Draw test samples
    fp32_test = fp32_dist.sample(n=N_TEST)
    quant_test = quant_dist.sample(n=N_TEST)
    
    # Run perplexity K-S test
    pvalue, stat = perplexity_ks_test(
        fp32_test,
        quant_test,
        kenlm_model_path="fp32_reference.arpa",
        pvalue_method="analytical"
    )
    
    print(f"  K-S statistic: {stat:.4f}")
    print(f"  p-value: {pvalue:.4f}")
    
    if pvalue < 0.05:
        print(f"  ✗ {quantization} significantly different from fp32")
    else:
        print(f"  ✓ {quantization} statistically similar to fp32")
```

## Optimizing KenLM Models

### Convert to Binary Format (Faster Loading)

```python
import subprocess

# After training, convert .arpa to binary .bin
subprocess.run([
    "build_binary",
    "model.arpa",
    "model.bin"
])

# Use the binary model (much faster to load)
pvalue, stat = perplexity_ks_test(
    sample1, sample2,
    kenlm_model_path="model.bin"  # Binary format
)
```

**Note:** `build_binary` comes with KenLM. Binary models load ~10-100x faster.

### Pruning Large Models

For very large models, you can prune low-probability n-grams:

```bash
# Prune n-grams with count threshold
lmplz -o 5 --text corpus.txt --arpa model.arpa --prune 0 0 1
#                                                      ^^^^^^
#                                                      unigram bigram trigram thresholds
```

See [KenLM documentation](https://kheafield.com/code/kenlm/) for details.

## Troubleshooting

### "Could not calculate Kneser-Ney discounts"

**Problem:** Dataset too small for modified Kneser-Ney smoothing.

**Solution:** The code already includes `--discount_fallback` flag. If still failing:
- Increase sample size (try n=500+)
- Reduce order (try order=3 instead of 5)
- Use character-level granularity (more tokens)

### "lmplz binary not found"

**Problem:** KenLM binaries not installed.

**Solution:**
```bash
# macOS: Build from source
git clone https://github.com/kpu/kenlm.git
cd kenlm
mkdir build && cd build
cmake ..
make -j 4
# Then add build/bin to PATH or copy binaries to /usr/local/bin
```

### Memory Issues

**Problem:** Training very large models causes OOM.

**Solutions:**
- Use `--memory 50%` flag with lmplz (requires direct lmplz call, not via train_kenlm_from_sample)
- Reduce sample size
- Reduce n-gram order
- Use pruning

## Best Practices

1. **Train on representative data**: Use diverse prompts that cover your test scenarios
2. **Use word-level granularity**: Usually gives best perplexity discrimination
3. **Start with order=5**: Good balance of context and model size
4. **Train on reference model**: For two-sample tests, train on the baseline (e.g., fp32)
5. **Sample enough data**: Aim for 1000+ completions for robust models
6. **Convert to binary**: For repeated use, convert .arpa to .bin for faster loading
7. **Skip empty texts**: Use `skip_empty=True` to avoid training artifacts

## See Also

- [`KS_TESTS.md`](KS_TESTS.md) - Complete guide to K-S testing in MET
- [`LESSONS-LEARNED.md`](LESSONS-LEARNED.md) - Troubleshooting KenLM issues
- [KenLM Documentation](https://kheafield.com/code/kenlm/) - Official KenLM docs
