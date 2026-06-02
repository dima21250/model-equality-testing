# Complete Workflow Example: Line-by-Line Explanation

This document provides a detailed walkthrough of the perplexity-based K-S test workflow for detecting distributional differences in LLM outputs.

## The Complete Example

```python
from model_equality_testing.dataset import load_distribution
from model_equality_testing.src.features import train_kenlm_from_sample
from model_equality_testing.algorithm import perplexity_ks_test

# Load fp32 baseline
fp32 = load_distribution(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    prompt_ids={"wikipedia_en": list(range(10))},  # 10 prompts
    L=200,
    source="fp32",
    load_in_unicode=True,
    root_dir="./data"
)

# Train reference model
train = fp32.sample(n=2000)
train_kenlm_from_sample(train, output_path="ref.arpa", order=5)

# Test quantization
int8 = load_distribution(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    prompt_ids={"wikipedia_en": list(range(10))},
    L=200,
    source="int8",
    load_in_unicode=True,
    root_dir="./data"
)

# Compare
fp32_test = fp32.sample(n=500)
int8_test = int8.sample(n=500)

pvalue, stat = perplexity_ks_test(fp32_test, int8_test, kenlm_model_path="ref.arpa")
print(f"K-S stat: {stat:.4f}, p-value: {pvalue:.4f}")
```

---

## Imports

```python
from model_equality_testing.dataset import load_distribution
from model_equality_testing.src.features import train_kenlm_from_sample
from model_equality_testing.algorithm import perplexity_ks_test
```

**What they do:**
- `load_distribution` - Loads pre-collected model completions from the MET dataset
- `train_kenlm_from_sample` - Trains a KenLM language model from completion samples
- `perplexity_ks_test` - Convenience function for running K-S tests on perplexity scores

---

## Step 1: Load the fp32 Baseline Distribution

```python
fp32 = load_distribution(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    prompt_ids={"wikipedia_en": list(range(10))},  # 10 prompts
    L=200,
    source="fp32",
    load_in_unicode=True,
    root_dir="./data"
)
```

**Line-by-line:**

- **`model="meta-llama/Meta-Llama-3-8B-Instruct"`** - Which model's completions to load
  
- **`prompt_ids={"wikipedia_en": list(range(10))}`** - Which prompts to use:
  - `"wikipedia_en"` = dataset name (English Wikipedia prompts)
  - `list(range(10))` = prompt IDs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - So we're loading 10 different prompts
  
- **`L=200`** - Completion length:
  - Each completion will be padded/truncated to exactly 200 tokens/characters
  - All completions have uniform length for easier processing
  
- **`source="fp32"`** - Which version of the model to load from:
  - `"fp32"` = 32-bit floating point (baseline, no quantization)
  - Other options: `"int8"`, `"nf4"`, `"azure"`, `"amazon"`, etc.
  
- **`load_in_unicode=True`** - **CRITICAL for KenLM:**
  - `True` = Convert token IDs to Unicode codepoints (for text-based analysis)
  - `False` = Keep as token IDs (for sequence-based tests like MMD)
  - KenLM needs actual text, not token IDs, so this must be True
  
- **`root_dir="./data"`** - Where the MET dataset is stored on disk

**What gets returned:**

`fp32` is a **`DistributionFromDataset`** object that contains:
- Pre-loaded completions for all 10 prompts (15,000 completions per prompt = 150,000 total)
- Ability to sample from these completions
- Metadata about prompts and completion probabilities

**Under the hood:**
```python
# Conceptually, fp32 contains something like:
fp32.k = {
    0: torch.Tensor([150000, 200]),  # Prompt 0's completions
    1: torch.Tensor([150000, 200]),  # Prompt 1's completions
    # ... 10 prompts total
}
fp32.m = 10  # Number of prompts
```

---

## Step 2: Sample Training Data

```python
train = fp32.sample(n=2000)
```

**What this does:**
- Draws **2000 random completions** from the fp32 distribution
- Samples uniformly across all 10 prompts (so ~200 completions per prompt)
- Returns a **`CompletionSample`** object

**What `train` contains:**

```python
train.N = 2000  # Total number of completions
train.m = 10    # Number of prompts
train.prompts = np.array([0, 0, 0, ..., 9, 9, 9])  # Shape: (2000,)
                # Which prompt each completion came from

train.completions = np.array([              # Shape: (2000, 200)
    [72, 101, 108, 108, 111, -1, -1, ...],  # "Hello" (unicode codepoints + padding)
    [87, 111, 114, 108, 100, -1, -1, ...],  # "World"
    # ... 2000 total completions, each padded to length 200
])
```

**Key insight:** Each completion is an array of Unicode codepoints (integers representing characters):
- `72` = 'H'
- `101` = 'e'
- `-1` = padding (end of text)

---

## Step 3: Train KenLM Model

```python
train_kenlm_from_sample(train, output_path="ref.arpa", order=5)
```

**What this does:**

1. **Decodes Unicode codepoints back to text:**
   ```python
   # Internally converts:
   [72, 101, 108, 108, 111, -1, -1] → "Hello"
   ```

2. **Writes text to temporary file:**
   ```
   Hello
   World
   This is a completion
   ... (2000 lines total)
   ```

3. **Calls lmplz binary:**
   ```bash
   lmplz -o 5 --text /tmp/corpus.txt --arpa ref.arpa --discount_fallback
   ```
   - `-o 5` = train 5-gram model (looks at sequences of 5 words)
   - `--text` = input corpus
   - `--arpa` = output ARPA format model
   - `--discount_fallback` = handle small datasets gracefully

4. **Saves trained model to `ref.arpa`:**
   - This is a statistical language model that has learned probability distributions of word sequences from the fp32 completions
   - File format: ARPA (ASCII text with n-gram probabilities)

**What the model learns:**
```
# Example n-grams and their log probabilities
-1.234 "the"
-2.567 "the cat"
-3.890 "the cat sat"
... (thousands of n-grams)
```

**Why 5-gram?**
- Captures context of up to 5 consecutive words
- Balance between context (higher order) and generalization (lower order)

---

## Step 4: Load int8 Quantized Distribution

```python
int8 = load_distribution(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    prompt_ids={"wikipedia_en": list(range(10))},  # Same 10 prompts
    L=200,
    source="int8",  # ← ONLY DIFFERENCE: quantized version
    load_in_unicode=True,
    root_dir="./data"
)
```

**What's different:**
- `source="int8"` - Loads completions from the 8-bit quantized version of the model
- Same model, same prompts, but the completions were generated by a quantized version

**The question we're answering:**
*"Does int8 quantization change the language statistics of the model's outputs?"*

---

## Step 5: Sample Test Data

```python
fp32_test = fp32.sample(n=500)
int8_test = int8.sample(n=500)
```

**What this does:**
- Draws **500 new completions** from fp32 (disjoint from the 2000 training samples)
- Draws **500 completions** from int8
- These are our test sets for comparison

**Why separate train/test?**
- We trained the KenLM model on `train` (2000 fp32 samples)
- We test on fresh samples to avoid overfitting/bias
- Fair comparison: both fp32_test and int8_test are scored against the same reference model

**Data structure:**
```python
fp32_test.N = 500  # 500 completions
fp32_test.m = 10   # Still 10 prompts
fp32_test.completions.shape = (500, 200)  # Unicode codepoints

int8_test.N = 500
int8_test.m = 10
int8_test.completions.shape = (500, 200)
```

---

## Step 6: Run Perplexity K-S Test

```python
pvalue, stat = perplexity_ks_test(fp32_test, int8_test, kenlm_model_path="ref.arpa")
```

**What happens under the hood:**

### 6a. Compute perplexity scores for fp32_test

```python
# For each of the 500 fp32 completions:
scores_fp32 = []
for completion in fp32_test.completions:
    text = decode_to_string(completion)  # e.g., "Hello world"
    
    # Score with KenLM model
    log_prob = model.score(text)  # e.g., -12.345
    num_words = len(text.split())  # e.g., 2
    
    # Calculate perplexity: 10^(-log_prob / num_words)
    perplexity = 10 ** (-log_prob / num_words)
    scores_fp32.append(perplexity)

# Result: scores_fp32 = [15.2, 23.7, 18.9, ..., 20.1]  (500 numbers)
```

**Perplexity interpretation:**
- Lower perplexity = more predictable/fluent text
- Higher perplexity = more surprising/unusual text
- "How surprised is the model by this text?"

### 6b. Compute perplexity scores for int8_test

```python
# Same process for int8 completions:
scores_int8 = []
for completion in int8_test.completions:
    text = decode_to_string(completion)
    log_prob = model.score(text)
    num_words = len(text.split())
    perplexity = 10 ** (-log_prob / num_words)
    scores_int8.append(perplexity)

# Result: scores_int8 = [16.1, 22.3, 19.5, ..., 21.7]  (500 numbers)
```

### 6c. Run K-S test on the two distributions

```python
from scipy.stats import ks_2samp

# Compare the two distributions of perplexity scores
statistic, pvalue = ks_2samp(scores_fp32, scores_int8)
```

**What K-S statistic measures:**
- Maximum difference between the cumulative distribution functions (CDFs)
- Range: [0, 1]
- 0 = distributions are identical
- 1 = distributions are completely different

**Visual analogy:**
```
Cumulative Distribution Functions (CDFs):

1.0 ┤                    ╭────
    │                 ╭──╯
0.5 ┤           ╭────╯  fp32
    │      ╭───╯
0.0 ┤─────╯
    └─────────────────────────→ Perplexity
    10    15    20    25    30

1.0 ┤                       ╭────
    │                  ╭───╯
0.5 ┤            ╭────╯  int8  (slightly shifted)
    │       ╭───╯
0.0 ┤──────╯
    └─────────────────────────→ Perplexity
    10    15    20    25    30

K-S statistic = max vertical distance between the curves
```

**What p-value tells us:**
- Probability of seeing this difference by random chance
- p < 0.05: "Significant difference - int8 changes perplexity distribution"
- p ≥ 0.05: "No significant difference - int8 preserves perplexity distribution"

---

## Step 7: Interpret Results

```python
print(f"K-S stat: {stat:.4f}, p-value: {pvalue:.4f}")
```

**Example outputs:**

**Scenario 1: No significant difference**
```
K-S stat: 0.0520, p-value: 0.3214
```
- Small K-S statistic (≈5% max difference in CDFs)
- High p-value (32% chance this is random)
- **Conclusion:** int8 quantization does NOT significantly affect perplexity distributions

**Scenario 2: Significant difference**
```
K-S stat: 0.2150, p-value: 0.0003
```
- Large K-S statistic (≈21% max difference in CDFs)
- Low p-value (0.03% chance this is random)
- **Conclusion:** int8 quantization DOES significantly affect perplexity distributions

---

## The Complete Data Flow

```
1. Load Data
   ├─ fp32.sample(2000) → train (CompletionSample)
   │                       ├─ prompts: [0,0,...,9,9]
   │                       └─ completions: [[72,101,...], ...] (unicode)
   │
   └─ int8 distribution loaded

2. Train KenLM
   train → decode → "Hello\nWorld\n..." → lmplz → ref.arpa
                                                    │
                    ┌───────────────────────────────┘
                    ▼
3. Sample Test Data
   ├─ fp32.sample(500) → fp32_test
   └─ int8.sample(500) → int8_test

4. Score with KenLM
   ├─ fp32_test → ref.arpa → [15.2, 23.7, ...] (500 perplexities)
   └─ int8_test → ref.arpa → [16.1, 22.3, ...] (500 perplexities)

5. K-S Test
   [15.2, 23.7, ...] ──┐
                       ├─→ ks_2samp() → (statistic, pvalue)
   [16.1, 22.3, ...] ──┘

6. Decision
   if pvalue < 0.05:
       "int8 changes language statistics"
   else:
       "int8 preserves language statistics"
```

---

## Why This Workflow?

1. **Train on reference (fp32)** - Establishes baseline language statistics
2. **Test both distributions** - Fair comparison using same reference
3. **Use perplexity** - Interpretable metric (language fluency/predictability)
4. **K-S test** - Detects distributional differences in perplexity
5. **Result** - Statistical answer to "Does quantization affect language quality?"

This is much more interpretable than sequence-based tests (MMD, chi-squared) because perplexity has a clear meaning!

---

## Key Takeaways

### About the Data
- **Unicode codepoints**: Text is represented as integer arrays where each integer is a character's Unicode value
- **Padding with -1**: All sequences padded to uniform length for efficient processing
- **Prompt structure**: Each completion is tagged with which prompt it came from

### About KenLM Training
- **Text-based**: Must use `load_in_unicode=True` to convert token IDs to text
- **N-gram models**: Learn statistical patterns in word sequences
- **ARPA format**: Human-readable format with n-gram probabilities
- **Discount fallback**: Handles small datasets gracefully

### About Perplexity
- **Definition**: `10^(-log_prob / num_words)` - exponential of negative average log probability
- **Interpretation**: How "surprised" the model is by the text
- **Lower is better**: More predictable/fluent text has lower perplexity
- **Reference model**: Score text against a reference LM (not the generator)

### About K-S Tests
- **Non-parametric**: No assumptions about distribution shapes
- **Sensitive**: Can detect subtle differences in distributions
- **Two modes**: Analytical (fast) or permutation (robust)
- **Interpretable**: Compare scalar features, not high-dimensional sequences

### About the Workflow Design
- **Separate train/test**: Avoid overfitting - train LM on one set, test on another
- **Same reference**: Both samples scored against same KenLM model for fair comparison
- **Representative sampling**: Use enough prompts and samples for robust statistics
- **Clear interpretation**: Perplexity difference → linguistic difference

---

## Common Questions

**Q: Why not train on int8 instead of fp32?**

A: We want to test if int8 matches the fp32 baseline. Training on fp32 establishes "normal" language statistics, then we see if int8 deviates.

**Q: Can I use the same samples for training and testing?**

A: Not recommended - this can create bias. The KenLM model will assign artificially good scores to text it was trained on.

**Q: How many samples do I need?**

A: 
- Training: 1000+ for order=5, 500+ for order=3
- Testing: 500+ per distribution for reliable K-S test
- More prompts = more diversity = better generalization

**Q: What if I don't have the MET dataset?**

A: You can use `train_kenlm_from_sample()` on any `CompletionSample` object. Just ensure it has `load_in_unicode=True` if loading from token IDs.

**Q: Should I use analytical or permutation p-values?**

A: 
- Analytical: Fast, good for exploration
- Permutation: Robust, good for final results
- Both give similar results for K-S tests

**Q: What's a "significant" K-S statistic?**

A: Focus on the p-value, not the statistic. p < 0.05 = significant. The statistic magnitude depends on sample size and effect size.

---

## See Also

- [`KS_TESTS.md`](KS_TESTS.md) - Complete guide to K-S testing
- [`TRAINING_KENLM_MODELS.md`](TRAINING_KENLM_MODELS.md) - KenLM training details
- [`example_train_kenlm.py`](example_train_kenlm.py) - Executable example script
- [`OPTION_B_USAGE.md`](OPTION_B_USAGE.md) - Generic K-S with custom features
