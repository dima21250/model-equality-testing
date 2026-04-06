# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the codebase for the research paper "Model Equality Testing: Which model is this API serving?" ([arxiv.org/abs/2410.20247](https://arxiv.org/abs/2410.20247)). It provides both a Python package for conducting statistical tests to detect if a black-box LLM API has changed its underlying model, and experimental code to reproduce the paper's results.

The core problem: Users access LLMs through APIs, but providers may quantize, watermark, or finetune models without notification. This package enables users to statistically test whether two distributions of model completions are the same.

## Repository Structure

**Main package** (`model_equality_testing/src/`):
- `algorithm.py` - High-level API: `run_two_sample_test()` and `run_goodness_of_fit_test()`
- `distribution.py` - Core data structures: `CompletionSample` (represents samples) and `DistributionFromDataset` (distributions to sample from)
- `tests.py` - Test statistic implementations (MMD variants, chi-squared, L1/L2, KS tests)
- `pvalue.py` - P-value calculators: permutation tests, parametric bootstrap, analytical KS
- `dataset.py` - Loading the 1.6M completion dataset from the paper
- `registry.py` - Mapping of string names to test functions (avoids circular imports)
- `utils.py` - Utility functions (unicode tokenization, padding, etc.)
- `corpus_sampling.py` - Corpus sampling utilities for K-S tests
- `features.py` - Feature extraction (e.g., VADER sentiment scores for K-S tests)

**Experiments** (`experiments/`):
- `experiments/sampling/` - Code to collect samples from local models and APIs (used to generate the dataset)
- `experiments/testing/` - Power simulation code for evaluating test performance
- `experiments/prompts.py` - Prompt generation utilities

**Root directory**: Various analysis scripts for exploratory work (K-S sanity checks, MMD-KS comparisons, prompt sensitivity analysis)

## Installation and Development

Install the package in development mode:
```bash
cd model_equality_testing
pip install -e .
```

The package requires: numpy, torch, matplotlib, python-dotenv

To download the full dataset (37.1GB):
```python
# First install gdown: pip install gdown
from model_equality_testing.dataset import download_dataset
download_dataset(root_dir="./data")
```

## Key Architectural Patterns

**Test registry pattern**: To avoid circular imports between `algorithm.py` and `tests.py`, test functions are registered in `registry.py` which maintains the `IMPLEMENTED_TESTS` dictionary mapping names like `"mmd_hamming"` to actual functions.

**Lazy imports**: `algorithm.py` imports p-value utilities and test registry inside functions rather than at module level to prevent circular dependencies.

**Sample representation**: Completions are represented as integer arrays - either token IDs (for local models) or Unicode codepoints (for API strings). All processing uses `-1` as the padding token. The `CompletionSample` class bundles prompts (indices), completions (padded arrays), and metadata.

**Distribution abstraction**: `DistributionFromDataset` wraps pre-collected samples and provides a `draw_completion_sample(n)` method. This enables both goodness-of-fit tests (against a reference distribution) and two-sample tests (between two sample sets).

**P-value flexibility**: Tests accept either a `pvalue_type` string (`"permutation_pvalue"`, `"parametric_bootstrap"`) or a custom p-value calculator function via `get_pvalue` parameter.

## Common Development Tasks

**Run a two-sample test**:
```python
from model_equality_testing.distribution import CompletionSample
from model_equality_testing.algorithm import run_two_sample_test

# Prepare samples (see README.md for full example)
sample1 = CompletionSample(prompts=..., completions=..., m=num_prompts)
sample2 = CompletionSample(prompts=..., completions=..., m=num_prompts)

pvalue, statistic = run_two_sample_test(
    sample1, sample2,
    pvalue_type="permutation_pvalue",
    stat_type="mmd_hamming",  # or "two_sample_L2", "two_sample_ks", etc.
    b=100  # number of permutations
)
```

**Available test statistics**:
- Goodness-of-fit: `g_squared`, `chi_squared`, `truncated_chi_squared`, `L1`, `L2`
- Two-sample: `two_sample_chi_squared`, `two_sample_L1`, `two_sample_L2`, `two_sample_ks`, `two_sample_vader_ks`, `mmd_hamming`, `mmd_kspectrum`, `mmd_all_subsequences`

**Load distributions from the dataset**:
```python
from model_equality_testing.dataset import load_distribution

p = load_distribution(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    prompt_ids={"wikipedia_ru": [0, 3, 10]},
    L=1000,  # pad/truncate to this length
    source="fp32",  # or 'nf4', 'int8', 'amazon', 'azure', etc.
    load_in_unicode=True,  # use unicode codepoints instead of tokens
    root_dir="./data"
)
```

**Working with local vs API samples**:
- Local samples (fp32, fp16, int8, nf4, watermark): stored as token IDs, can be loaded in either token or unicode space
- API samples (anyscale, amazon, fireworks, etc.): stored as strings, only loaded in unicode space
- When loading local samples in unicode, special tokens are skipped and each character becomes its Unicode codepoint
- Padding is always represented as `-1`

## Dataset Organization

The dataset contains ~1.6M completions across:
- 5 language models (Llama-3-8B, Llama-3-70B, Llama-2-7B, Mistral-7B, Mixtral-8x7B)
- Various sources: local quantizations (fp32, fp16, int8, nf4) and API providers (amazon, azure, fireworks, etc.)
- 540 prompts total (100 dev set with logprobs, 440 test set)

See the [Google Drive spreadsheet](https://docs.google.com/spreadsheets/d/1T9aPZHK1xxfxogrHYaHqvW0Blqi-XJx2rgOPktWcN0w/edit?usp=sharing) for full catalog of available samples.

## Reproducing Paper Experiments

The `experiments/` directory contains the original code used to generate results in the paper. Note that APIs have evolved since data collection (July-August 2024), so behavior may differ.

**Generate samples from a local model**:
```bash
# See experiments/sampling/cache_local_samples.py
# Loads models via transformers and generates completions
```

**Simulate test power**:
```bash
# See experiments/testing/simulate_two_sample_power.py
# Runs power simulations comparing different test statistics
```

## Important Notes

- Completion samples are processed to only include text up to the first `<eos>` token. Tokens after `<eos>` are replaced with pad tokens.
- When a model tokenizer doesn't specify a pad token, the `<eos>` token is used as padding.
- API samples are returned without special tokens.
- The MMD tests use prompts to define the kernel: completions from different prompts have kernel value 0.
- Test functions in `tests.py` expect `CompletionSample` objects where completions include the prompt index in the first column internally, but this is handled by the `CompletionSample` class automatically.
