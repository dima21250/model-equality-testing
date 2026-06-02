# Perplexity-based K-S Test Implementation - Summary

## ✅ Implementation Complete

The perplexity-based K-S test has been successfully implemented for the Model Equality Testing (MET) framework!

### Files Modified

1. **`model_equality_testing/src/features.py`**
   - ✅ Added `get_perplexity_scores()` - Extract perplexity scores using KenLM
   - ✅ Added `train_kenlm_from_sample()` - Train KenLM models from CompletionSample data
   - ✅ Added helper functions for model caching (`_load_kenlm_model`, `_load_sentencepiece_model`)
   - ✅ Added score parsing utility (`_parse_score_value`)

2. **`model_equality_testing/src/tests.py`**
   - ✅ Added `two_sample_perplexity_ks()` - K-S test using perplexity scores
   - ✅ Updated imports to include `get_perplexity_scores`

3. **`model_equality_testing/src/registry.py`**
   - ✅ Registered `two_sample_perplexity_ks` in `IMPLEMENTED_TESTS`
   - ✅ Added import for the new test function

4. **`model_equality_testing/pyproject.toml`**
   - ✅ Already configured with optional `[perplexity]` dependencies

### Verification Status

- ✅ All Python files compile without syntax errors
- ✅ All imports work correctly
- ✅ Function is properly registered and accessible via `run_two_sample_test(..., stat_type="two_sample_perplexity_ks")`
- ⚠️ System dependency issue: `lmplz` binary needs boost libraries reinstalled

## ⚠️ Known Issue: lmplz Dependencies

The `lmplz` binary requires boost C++ libraries. Current error:

```
Library not loaded: /opt/homebrew/opt/boost/lib/libboost_system.dylib
```

### Fix Required

```bash
# Option 1: Reinstall boost
brew reinstall boost

# Option 2: Link boost libraries
brew link boost --overwrite

# Option 3: Rebuild KenLM against current boost
# Follow instructions at https://github.com/kpu/kenlm
```

## 🎯 Key Features Implemented

### Maximum Configurability

**Text Preprocessing:**
- ✅ Raw text mode (default)
- ✅ Sentencepiece tokenization (when `spm_model_path` provided)
- ✅ Word-level granularity (split on whitespace)
- ✅ Character-level granularity
- ✅ Token-level granularity (with sentencepiece)

**Degenerate Case Handling:**
- ✅ Configurable `empty_text_score` ("inf", "nan", or custom float)
- ✅ Configurable `zero_tokens_score` ("inf", "nan", or custom float)
- ✅ Optional warning logging (`warn_on_degenerate` flag)

**Model Management:**
- ✅ Automatic model caching via `@lru_cache` (avoids reloading)
- ✅ Support for both `.arpa` and `.bin` KenLM models

## 📖 Usage Examples

### 1. Train a KenLM Model from MET Data

```python
from model_equality_testing.dataset import load_distribution
from model_equality_testing.src.features import train_kenlm_from_sample

# Load sample data
dist = load_distribution(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    prompt_ids={"wikipedia_en": [0, 1, 2]},
    L=200,
    source="fp32",
    root_dir="./data"
)
sample = dist.sample(n=1000)

# Train 5-gram model
train_kenlm_from_sample(
    sample=sample,
    output_path="llama3_fp32.arpa",
    order=5,  # 5-gram model
    skip_empty=True,
    warn_on_degenerate=True
)
```

### 2. Compute Perplexity Scores

```python
from model_equality_testing.src.features import get_perplexity_scores

# Basic usage (word-level perplexity)
scores = get_perplexity_scores(
    sample=my_sample,
    kenlm_model_path="llama3_fp32.arpa"
)

# With sentencepiece preprocessing
scores = get_perplexity_scores(
    sample=my_sample,
    kenlm_model_path="llama3_fp32.arpa",
    spm_model_path="tokenizer.model",
    granularity="token"
)

# Character-level perplexity
scores = get_perplexity_scores(
    sample=my_sample,
    kenlm_model_path="llama3_fp32.arpa",
    granularity="char"
)
```

### 3. Run K-S Test via Registry

```python
from model_equality_testing.algorithm import run_two_sample_test

# Load two samples to compare
sample_fp32 = fp32_dist.sample(n=500)
sample_int8 = int8_dist.sample(n=500)

# Run perplexity-based K-S test
pvalue, statistic = run_two_sample_test(
    sample_fp32,
    sample_int8,
    stat_type="two_sample_perplexity_ks",
    kenlm_model_path="llama3_fp32.arpa",
    pvalue_type="permutation_pvalue",
    b=1000,  # number of permutations
    warn_on_degenerate=False
)

print(f"K-S statistic: {statistic:.6f}")
print(f"p-value: {pvalue:.6f}")
print(f"Reject null hypothesis? {pvalue < 0.05}")
```

### 4. Direct K-S Test Function

```python
from model_equality_testing.src.tests import two_sample_perplexity_ks

# Compute K-S statistic directly
ks_stat = two_sample_perplexity_ks(
    sample1=sample_fp32,
    sample2=sample_int8,
    kenlm_model_path="llama3_fp32.arpa",
    granularity="word",
    empty_text_score="inf",
    zero_tokens_score="inf",
    warn_on_degenerate=True
)
```

## 🧪 Testing

### Test Scripts Created

1. **`test_perplexity_ks.py`** - Full integration test (requires working `lmplz`)
   - Tests training, scoring, and K-S computation
   - Validates registry integration
   - Checks edge cases

2. **`test_perplexity_scoring_only.py`** - Scoring test (requires pre-trained model)
   - Tests scoring and K-S without training
   - Usage: `python test_perplexity_scoring_only.py <model.arpa>`

### Running Tests

Once boost dependencies are fixed:

```bash
# Full test suite
python test_perplexity_ks.py

# Or with existing model
python test_perplexity_scoring_only.py path/to/model.arpa
```

## 📋 Next Steps

### Immediate (Fix Dependencies)

1. **Fix boost library dependencies**
   ```bash
   brew reinstall boost
   # or
   brew link boost --overwrite
   ```

2. **Verify lmplz works**
   ```bash
   lmplz --help
   ```

3. **Run test suite**
   ```bash
   python test_perplexity_ks.py
   ```

### Experimental Validation

4. **Train models from MET dataset**
   - Train KenLM models for each quantization level (fp32, int8, nf4)
   - Use same prompt set for fair comparison

5. **Compare test performance**
   - Perplexity K-S vs. VADER K-S
   - Perplexity K-S vs. MMD (hamming, k-spectrum)
   - Measure power to distinguish quantizations

6. **Evaluate different configurations**
   - N-gram orders: 3-gram vs 5-gram vs 7-gram
   - Granularity: word-level vs character-level
   - Training corpus size: impact on sensitivity

7. **Test with real monitoring scenario**
   - Train baseline model from API samples at T₀
   - Collect new samples at T₁
   - Test if perplexity distribution shifted

### Advanced Features (Future)

8. **Sentencepiece integration testing** (if you have/train SPM models)
   - Train sentencepiece tokenizer on completions
   - Compare token-level vs word-level perplexity

9. **Cross-model experiments**
   - Does perplexity under Llama-3 model distinguish Mistral quantizations?
   - Cross-architecture sensitivity analysis

10. **Performance optimization**
    - Profile large-scale experiments
    - Consider parallel processing for scoring
    - Binary KenLM models for faster loading

## 🔍 Research Questions to Explore

1. **Sensitivity**: Is perplexity-based K-S more sensitive to quantization than sentiment?

2. **Corpus requirements**: How much training data needed for reliable perplexity estimates?

3. **Domain transfer**: Does a model trained on Wikipedia completions work for other domains?

4. **Optimal n-gram order**: Does higher-order (5-gram, 7-gram) improve detection power?

5. **Baseline choice**: Should you train on fp32, the API baseline, or mixed sources?

6. **Combination strategies**: Can you combine perplexity + VADER scores for better power?

## 📚 Documentation

### Installation for Users

Add to README or documentation:

```bash
# Install with perplexity support
pip install -e ".[perplexity]"

# Install KenLM binaries (macOS)
brew install kenlm

# Or Ubuntu/Debian
sudo apt-get install kenlm

# Verify installation
lmplz --help
python -c "import kenlm; print('KenLM ready')"
```

### API Reference

All functions are fully documented with:
- Detailed docstrings
- Parameter descriptions
- Return types
- Usage examples
- Error handling notes

See:
- `model_equality_testing/src/features.py` - Training and scoring functions
- `model_equality_testing/src/tests.py` - K-S test implementation

## 🎉 Summary

The perplexity-based K-S test is **fully implemented and ready to use** once the system dependencies are resolved. The implementation provides:

- **Flexibility**: Multiple preprocessing modes, granularities, and configurations
- **Robustness**: Proper error handling, degenerate case handling, model caching
- **Integration**: Registered in test registry, works with existing MET infrastructure
- **Documentation**: Comprehensive docstrings and examples

This opens up a new dimension for model equality testing, complementing the existing MMD and VADER-based approaches with perplexity-based distribution comparison!
