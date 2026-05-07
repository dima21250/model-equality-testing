# quantum_comparison.py

Compares quantum-inspired metrics against classical statistical tests on LLM output distributions. The script operates in a two-tier framework: classical tests detect whether token-level distributions differ, and quantum metrics characterize whether those differences extend to the semantic level.

## Prerequisites

```bash
cd model_equality_testing
pip install -e ".[quantum,sentiment]"
```

This installs the core package plus `sentence-transformers`, `scikit-learn`, and `vaderSentiment`. The dataset (37.1 GB) must be downloaded separately:

```python
from model_equality_testing.dataset import download_dataset
download_dataset(root_dir="./data")
```

## Modes of Operation

### 1. Single Comparison (default)

Compare two models or sources on the same set of prompts.

```bash
# FP32 vs INT8 quantization on Llama-3-8B, prompt 0
python quantum_comparison.py --prompts 0 --source_a fp32 --source_b int8 --samples 200

# Two different models
python quantum_comparison.py --model_a meta-llama/Meta-Llama-3-8B-Instruct \
                             --model_b mistralai/Mistral-7B-Instruct-v0.3 \
                             --source_a fp32 --source_b fp32 --samples 200
```

Both samples are drawn from the same prompt(s). The question being asked: "Does changing the model or quantization change the output distribution?"

### 2. Prompt Comparison (`--prompt_comparison`)

Compare completions from two different prompts on the same model and source. This serves as a positive control -- different prompts should produce semantically distinct outputs, so the quantum metrics should detect the difference.

```bash
# Prompt 0 vs prompt 5 on fp32 Llama-3-8B
python quantum_comparison.py --prompt_comparison 0 5 --samples 200

# With a different model or source
python quantum_comparison.py --prompt_comparison 0 5 --source_a int8 --samples 200
```

Both samples use `--model_a` and `--source_a` (the model/source is held constant). The question being asked: "Can the quantum metrics detect a genuine semantic difference?"

### 3. Validation Suite (`--run_suite`)

Runs three comparisons automatically to validate metric behavior across a range of expected effect sizes.

```bash
python quantum_comparison.py --run_suite --prompts 0 --samples 200
```

This runs:
- **Case 1 -- Equivalence**: Same model, same source (expect no significant differences)
- **Case 2 -- Quantization**: FP32 vs INT8 (expect token-level but not semantic differences)
- **Case 3 -- Model difference**: Llama-3-8B vs Mistral-7B (expect large differences everywhere)

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_a` | `meta-llama/Meta-Llama-3-8B-Instruct` | First model |
| `--model_b` | `meta-llama/Meta-Llama-3-8B-Instruct` | Second model (ignored in `--prompt_comparison` mode) |
| `--source_a` | `fp32` | Source for model A (`fp32`, `int8`, `nf4`, etc.) |
| `--source_b` | `int8` | Source for model B (ignored in `--prompt_comparison` mode) |
| `--prompts` | `0 1 2` | Space-separated prompt IDs |
| `--dataset` | `wikipedia_en` | Dataset name (`wikipedia_en`, `wikipedia_ru`, etc.) |
| `--L` | `500` | Completion length (truncation in characters) |
| `--samples` | `100` | Number of completions per distribution |
| `--b` | `100` | Number of permutations for p-value computation |
| `--pca_k` | `50` | Number of PCA components for density matrix construction |
| `--embedding_model` | `all-mpnet-base-v2` | Sentence-transformers model for embeddings |
| `--root_dir` | `./data` | Root directory for the dataset |
| `--run_suite` | off | Run the three-case validation suite |
| `--prompt_comparison` | off | Two prompt IDs to compare (positive control mode) |

## Metrics

The script computes five metrics, divided into two tiers.

### Classical Metrics (Token-Level)

**MMD (Hamming)** -- Maximum Mean Discrepancy with Hamming kernel. Compares the character-level distributions directly. Highly sensitive to any token-level change, including those that don't affect meaning (e.g., whitespace, punctuation). p-value via permutation test.

**VADER K-S** -- Kolmogorov-Smirnov test on VADER compound sentiment scores. Compares the sentiment distributions of the two samples. Detects shifts in emotional tone. p-value via analytical KS distribution (scipy).

### Quantum Metrics (Semantic Level)

All three quantum metrics operate on density matrices constructed from SBERT embeddings. Completions are embedded into 768-dimensional space (all-mpnet-base-v2), projected to a shared k-dimensional PCA basis (default k=50), and the k x k covariance matrix is normalized to unit trace to form the density matrix.

**Trace Distance** -- `D(rho_A, rho_B) = 0.5 * Tr(|rho_A - rho_B|)`. Measures how *distinguishable* two semantic distributions are. Compares the full shape of the distributions in embedding space. Ranges from 0 (identical) to 1 (orthogonal support). Answers: "Are these the same semantic distribution?"

**Von Neumann Divergence** -- `|S(rho_A) - S(rho_B)|` where `S(rho) = -Tr(rho log rho)`. Measures whether two distributions have similar *diversity* or *spread*. Computes entropy independently for each density matrix, then takes the absolute difference. Does not compare content directly -- only the degree of variety. Answers: "Do these distributions have the same level of semantic diversity?"

**QRE (symmetric)** -- `S(rho_A || rho_B) + S(rho_B || rho_A)`. Symmetrized quantum relative entropy. Measures the *information loss* when approximating one distribution with the other. Captures both shape and spread differences. The quantum analog of symmetrized KL divergence. Answers: "How much information is lost by treating one distribution as the other?"

All quantum p-values are computed via permutation tests: sample labels are shuffled, the statistic is recomputed for each permutation, and the p-value is the fraction of permuted statistics that exceed the observed value.

## Interpreting Results

### Reading the Output

Each run prints a table like:

```
    - Trace distance: 0.219735, p=0.3900 (took 5.52s)
    - Von Neumann divergence: 0.034498, p=0.4900 (took 2.37s)
    - QRE (symmetric): 0.029843, p=0.0400 (took 2.44s)
    ...
    - MMD (Hamming): 0.002173, p=0.0000 (took 12.34s)
    - VADER K-S: 0.030000, p=0.0300 (took 0.05s)
```

The **statistic** is the raw metric value. The **p-value** indicates whether the observed statistic is larger than expected under the null hypothesis (the two samples come from the same distribution). A p-value below 0.05 suggests the distributions differ; above 0.05 suggests no detectable difference.

### Context: Quantization (FP32 vs INT8)

The central question: does quantization change what the model says, or only how it says it?

**Expected pattern**:
- MMD (Hamming): significant (p < 0.05). Quantization changes the token-level distribution.
- Quantum metrics: non-significant (p > 0.05). The semantic distribution is preserved.

This pattern supports the conclusion that INT8 quantization changes tokens but preserves semantics. The model produces detectably different character sequences, but the *meaning* of its outputs is statistically indistinguishable.

If quantum metrics *are* significant, it suggests quantization is affecting the semantic content -- a more serious degradation than mere syntactic variation.

### Context: Prompt Comparison (Positive Control)

The question: can the quantum metrics detect genuine semantic differences when they exist?

**Expected pattern**:
- MMD (Hamming): significant. Different prompts produce different text.
- Trace distance: significant. The semantic distributions have different shapes.
- QRE (symmetric): significant. There is information loss when treating one as the other.
- Von Neumann divergence: likely *non-significant*. Different prompts may produce outputs with similar diversity even though the content differs.

The von Neumann divergence result is particularly informative. Its non-significance here is *not* a failure -- it confirms that the metric measures *diversity*, not *content*. Two prompts can elicit very different topics while maintaining similar levels of output variety. This validates the metric's interpretation in the quantization context: when von Neumann divergence is non-significant for FP32 vs INT8, it genuinely means the diversity is preserved, not that the metric is insensitive.

### Context: Different Models

**Expected pattern**: all metrics significant. Different models produce fundamentally different output distributions at every level.

If quantum metrics are *not* significant for truly different models, it suggests the embedding model or PCA projection is too coarse to distinguish them.

### Summary: What Each Metric Tells You

| Metric | Sensitive to content? | Sensitive to diversity? | Role |
|--------|----------------------|------------------------|------|
| MMD (Hamming) | Yes (tokens) | Yes (tokens) | Detect any token-level change |
| VADER K-S | Partially (sentiment) | Partially | Detect sentiment shifts |
| Trace distance | Yes (semantics) | Yes | Semantic distribution shape |
| Von Neumann div | No | Yes | Semantic diversity only |
| QRE (symmetric) | Yes (semantics) | Yes | Comprehensive semantic divergence |

## Experimental Design Guidance

### Prompt Selection

Use single prompts for quantum metrics. Running several single-prompt experiments independently (e.g., prompt 0, then prompt 3, then prompt 10) and checking consistency across prompts provides stronger evidence than pooling multiple prompts into one test. Multi-prompt pooling introduces prompt-driven semantic variation that can drown out the signal of interest. See `quantum-prompt-strategy.md` for the full rationale.

### Sample Size

Recommended: n=200 or above. Larger samples give more stable density matrix estimates but increase computation time linearly for the permutation tests. The PCA projection (k=50) keeps density matrix operations fast regardless of n.

### Permutations

The default b=100 is sufficient for exploratory work. For publication-quality p-values, use b=1000 or higher. At b=100 the minimum reportable p-value is 0.01 (i.e., p=0.0000 in the output means p < 0.01).

### PCA Components

The default k=50 is a standard choice in the embedding literature, capturing roughly 40-50% of total variance. This is defensible in review because:
- It is large enough to preserve meaningful semantic structure
- It is small enough to regularize the density matrix (avoids overfitting to noise in the tail components)
- It is consistent with established dimensionality reduction practice for sentence embeddings

To verify the choice is appropriate for your data, check `pca.explained_variance_ratio_.sum()` (available programmatically via `fit_pca()` in `quantum_metrics.py`).

## Technical Details

### Density Matrix Construction

Embeddings are projected to a shared k-dimensional PCA basis fit on the *combined* pool (both samples together). For each sample, the k x k covariance matrix in the reduced space is computed and normalized to unit trace:

```
X_reduced = PCA.transform(embeddings)    # (N, k)
cov = X_reduced.T @ X_reduced / N        # (k, k)
rho = cov / Tr(cov)                       # unit trace
```

This approach fixes two problems with the earlier N x N Gram matrix method:
1. Both density matrices live in the same k-dimensional Hilbert space (shared PCA basis)
2. Matrix dimension is fixed at k x k regardless of sample size N, eliminating sample-size dependence

### Permutation Test Performance

SBERT embedding inference (the expensive step) runs once on the combined pool. PCA is fit once. Each permutation only shuffles indices and recomputes k x k matrix operations, giving ~1000+ permutations/second.

### Legacy Mode

To use the original N x N Gram matrix approach (not recommended), pass `--pca_k 0`. Note that this reintroduces the Hilbert space mismatch and sample-size dependence documented in `INITIAL-RESULTS.md`.
