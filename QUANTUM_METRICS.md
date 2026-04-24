# Quantum-Inspired Metrics for LLM Output Comparison

This implementation provides quantum-inspired metrics for comparing LLM output distributions, based on the approach described in `FLAIR-ALTERNATIVE.md`.

## Overview

The quantum metrics framework uses **pre-trained SBERT embeddings** to map LLM outputs into a shared semantic space, then computes quantum-inspired divergence measures:

- **Trace Distance**: Measures distinguishability between distributions (analogous to total variation distance)
- **Von Neumann Entropy Divergence**: Measures difference in output diversity
- **Quantum Relative Entropy (QRE)**: Measures information loss when approximating one distribution with another

## Installation

Install the quantum metrics dependencies:

```bash
cd model_equality_testing
pip install -e ".[quantum]"
```

This installs `sentence-transformers` for embedding generation.

## Quick Start

### Basic Usage

```python
from model_equality_testing.dataset import load_distribution
from model_equality_testing.algorithm import run_two_sample_test

# Load two distributions
dist_fp32 = load_distribution(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    prompt_ids={"wikipedia_en": [0, 1, 2]},
    L=500,
    source="fp32",
    load_in_unicode=True,  # Important: quantum tests require unicode
)

dist_int8 = load_distribution(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    prompt_ids={"wikipedia_en": [0, 1, 2]},
    L=500,
    source="int8",
    load_in_unicode=True,
)

# Sample
sample1 = dist_fp32.sample(n=100)
sample2 = dist_int8.sample(n=100)

# Run quantum trace distance test
pvalue, stat = run_two_sample_test(
    sample1, sample2,
    stat_type="quantum_trace_distance",
    pvalue_type="permutation_pvalue",
    b=100,
    embedding_model="all-mpnet-base-v2"  # Optional parameter
)

print(f"Trace distance: {stat:.4f}, p-value: {pvalue:.4f}")
```

### Direct Metric Computation

```python
from model_equality_testing.tests import (
    quantum_trace_distance,
    quantum_von_neumann_divergence,
    quantum_relative_entropy_test,
)

# Compute metrics directly (no p-values)
trace_dist = quantum_trace_distance(sample1, sample2)
entropy_div = quantum_von_neumann_divergence(sample1, sample2)
qre = quantum_relative_entropy_test(sample1, sample2, symmetric=True)

print(f"Trace distance: {trace_dist:.6f}")
print(f"Entropy divergence: {entropy_div:.6f}")
print(f"QRE (symmetric): {qre:.6f}")
```

## Example Scripts

### `quantum_comparison.py`

Compare quantum metrics against classical tests (MMD, VADER K-S):

```bash
# Single comparison (fp32 vs int8)
python quantum_comparison.py \
  --model_a "meta-llama/Meta-Llama-3-8B-Instruct" \
  --source_a fp32 \
  --source_b int8 \
  --samples 100

# Run full validation suite (equivalence + difference cases)
python quantum_comparison.py --run_suite --samples 200
```

### `validate_quantum.py`

Run validation experiments to verify quantum metrics work correctly:

```bash
python validate_quantum.py \
  --root_dir ./data \
  --samples 100 \
  --embedding_model all-mpnet-base-v2
```

This runs four validation tests:
1. **Known Differences**: Different models should show large trace distance
2. **Known Similarities**: Same model (different samples) should show small trace distance
3. **Controlled Perturbations**: Metrics should increase with quantization strength (fp32 < int8 < nf4)
4. **Correlation with Baselines**: Quantum metrics should provide complementary information to MMD

## Available Quantum Tests

### `quantum_trace_distance`

**Measures**: Distinguishability between distributions  
**Range**: [0, 1]  
**Interpretation**:
- 0 = Identical distributions
- 1 = Completely orthogonal distributions

**Example**:
```python
stat = quantum_trace_distance(sample1, sample2, embedding_model="all-mpnet-base-v2")
```

### `quantum_von_neumann_divergence`

**Measures**: Difference in output diversity/entropy  
**Range**: [0, ∞)  
**Interpretation**:
- 0 = Same diversity
- Large values = One distribution is more diverse than the other

**Example**:
```python
stat = quantum_von_neumann_divergence(sample1, sample2)
```

### `quantum_relative_entropy`

**Measures**: Information loss when approximating one distribution with another  
**Range**: [0, ∞) (can be infinite)  
**Interpretation**:
- 0 = Identical distributions
- Large values = Distributions are very different
- ∞ = Non-overlapping support

**Example**:
```python
# Symmetric version (like Jensen-Shannon divergence)
stat = quantum_relative_entropy_test(sample1, sample2, symmetric=True)

# Asymmetric version (directional)
stat = quantum_relative_entropy_test(sample1, sample2, symmetric=False)
```

## Embedding Models

The quantum metrics use pre-trained sentence transformers. Recommended models:

| Model | Embedding Dim | Speed | Use Case |
|-------|--------------|-------|----------|
| `all-mpnet-base-v2` | 768 | Medium | **Recommended** - Best quality |
| `all-MiniLM-L6-v2` | 384 | Fast | Quick experiments, limited compute |
| `all-distilroberta-v1` | 768 | Medium | Alternative to mpnet |

**Usage**:
```python
# Use a faster model
stat = quantum_trace_distance(
    sample1, sample2,
    embedding_model="all-MiniLM-L6-v2"
)
```

## How It Works

### Pipeline

1. **Decode completions to strings**
   - Unicode codepoints → text strings
   
2. **Embed with SBERT**
   - Texts → dense semantic vectors (e.g., 768-dimensional)
   - All texts mapped to the same space (automatic alignment)

3. **Construct PIP matrices**
   - Compute pairwise inner products: PIP[i,j] = ⟨e_i, e_j⟩
   - Result: N×N similarity matrix

4. **Normalize to density matrices**
   - ρ = PIP / Tr(PIP)
   - Ensures Tr(ρ) = 1 (quantum state normalization)

5. **Compute quantum metrics**
   - Trace distance: D(ρ_A, ρ_B) = 0.5 × Tr(|ρ_A - ρ_B|)
   - Von Neumann entropy: S(ρ) = -Tr(ρ log ρ) = -Σ λ_i log λ_i
   - QRE: S(ρ || σ) = Tr(ρ log ρ - ρ log σ)

### Why This Works

**Pre-trained embeddings solve the alignment problem**:
- ❌ **Separate autoencoders**: Each learns different coordinate systems → alignment required
- ✅ **SBERT**: Fixed, pre-trained semantic space → automatic alignment

**Advantages over classical tests**:
- Operates on semantic similarity, not token-level statistics
- Captures higher-level distributional differences
- No training required (just inference)
- Well-validated in NLP tasks

## Validation Results

Expected behavior based on validation experiments:

| Scenario | Trace Distance | Von Neumann Div | MMD |
|----------|---------------|-----------------|-----|
| Same model, same source | < 0.15 | Small | Small |
| Same model, fp32 vs int8 | 0.15 - 0.35 | Moderate | Moderate |
| Same model, fp32 vs nf4 | 0.25 - 0.45 | Moderate-Large | Moderate |
| Different models | > 0.35 | Large | Large |

*Note: Exact values depend on sample size, prompts, and embedding model.*

## Performance Considerations

### Sample Size

- **Recommended**: N = 100-500 samples per distribution
- **Memory**: PIP matrices are N×N (grows quadratically)
- **Large N (> 1000)**: Consider subsampling or batched computation

### Computational Cost

**Embedding generation is the bottleneck**:
- SBERT inference: ~10-50 samples/second (GPU) or ~1-5 samples/second (CPU)
- Quantum metric computation: Fast (linear algebra on N×N matrices)

**Optimization tips**:
- Use a faster embedding model (`all-MiniLM-L6-v2`)
- Use batch_size=32 or higher for GPU inference
- Cache embeddings if re-running with same samples

### Example Timing

On a modern CPU with 100 samples:
- Embedding (all-mpnet-base-v2): ~15-30 seconds
- Trace distance computation: < 1 second
- **Total**: ~15-30 seconds per comparison

With GPU: ~3-5 seconds total

## Troubleshooting

### `ImportError: sentence-transformers not installed`

Install quantum dependencies:
```bash
pip install -e ".[quantum]"
```

### `ValueError: PIP matrix trace is too small`

This indicates degenerate embeddings (all zero vectors). Causes:
- All texts are empty after filtering padding
- Embedding model failed to load
- Numerical underflow (very rare)

**Solution**: Check that `load_in_unicode=True` and texts are non-empty.

### `np.inf` returned by QRE

The quantum relative entropy is infinite when supports don't overlap.
This is **mathematically correct** behavior, indicating very different distributions.

**Interpretation**: The distributions are so different that they occupy non-overlapping regions of the semantic space.

### Trace distance close to 1.0 for all comparisons

Possible causes:
- Embedding model not loaded correctly
- Wrong embedding model for the task (try a different model)
- Sample size too small (increase to N=200+)

## References

- **FLAIR-ALTERNATIVE.md**: Detailed explanation of the quantum metrics approach
- **Nielsen & Chuang**: "Quantum Computation and Quantum Information" (quantum metric theory)
- **Sentence-BERT paper**: Reimers & Gurevych (2019), "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"

## Citation

If you use quantum metrics in your research:

```bibtex
@misc{gao2024model,
    title={Model Equality Testing: Which model is this API serving?},
    author={Gao, Irena and Liang, Percy and Guestrin, Carlos},
    journal={arXiv preprint},
    year={2024}
}
```

And cite Sentence-BERT if using SBERT embeddings:

```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Irene",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    year = "2019",
}
```
