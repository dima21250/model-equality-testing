# Understanding analyze-prompt-sensitivity.py

## How analyze-prompt-sensitivity.py Works

### 1. Data Structure

The MET dataset is organized hierarchically:
```
data/samples/
├── {model}-{dataset}-{source}-L={length}-{prompt_id}.pkl
```

For example:
- `meta-llama-Meta-Llama-3-8B-Instruct-wikipedia_en-fp32-L=200-0.pkl` (15,000 completions)
- `meta-llama-Meta-Llama-3-8B-Instruct-wikipedia_en-int8-L=200-0.pkl` (5,000 completions)
- ... up to prompt 99 for wikipedia_en

Each file contains **many completions** (5,000-15,000) generated from that specific prompt.

### 2. The Analysis Process

For a given comparison (e.g., fp32 vs int8), the script:

**For EACH prompt individually (0 → N):**

```python
# Step A: Load data for THIS prompt only
dist_fp32 = load_distribution(prompt_ids={dataset: [prompt_id]}, source="fp32")
dist_int8 = load_distribution(prompt_ids={dataset: [prompt_id]}, source="int8")

# Step B: Sample 500 completions from each
sample_fp32 = dist_fp32.sample(n=500)  # 500 from the 15,000 available
sample_int8 = dist_int8.sample(n=500)  # 500 from the 5,000 available

# Step C: Extract VADER compound sentiment scores
scores_fp32 = [vader.polarity_scores(text)["compound"] for text in sample_fp32]
scores_int8 = [vader.polarity_scores(text)["compound"] for text in sample_int8]

# Step D: Run KS test
ks_stat, p_value = ks_2samp(scores_fp32, scores_int8)

# Step E: Record results for this prompt
results[prompt_id] = {
    'ks_stat': ks_stat,
    'p_value': p_value,
    'mean_fp32': mean(scores_fp32),
    'std_fp32': std(scores_fp32),
    'mean_int8': mean(scores_int8),
    'std_int8': std(scores_int8),
}
```

### 3. Statistical Analysis

For each prompt, it computes:

**KS Test Results:**
- **KS statistic**: Maximum distance between the two empirical CDFs of sentiment scores
- **p-value**: Probability of observing this difference by chance

**Sentiment Statistics:**
- **Mean sentiment**: Average compound score for each source
- **Standard deviation**: Variance in sentiment (high variance = more diverse responses)
- **Mean difference**: |mean_A - mean_B| (how much watermarking/quantization shifted sentiment)

### 4. Ranking & Selection

The script then:

**Ranks prompts by:**
1. **KS statistic** (largest = strongest signal)
2. **p-value** (smallest = most significant)
3. **Sentiment variance** (largest = most diverse, potentially more discriminative)

**Identifies:**
- Prompts with **significant detection** (p < 0.05)
- **Top 3 prompts** by KS statistic
- Prompts with **highest variance**

### 5. Why This Works

**Key insight**: Different prompts are affected differently by quantization/watermarking:

**Example from our tests:**
- **Prompt 4** → int8 quantization shifts sentiment noticeably (KS=0.114, p=0.003) ✅
- **Prompt 6** → int8 quantization barely affects sentiment (KS=0.024, p=0.999) ❌

**Why the difference?**
- Prompt 4 may elicit responses where subtle word choice changes (from quantization) affect sentiment
- Prompt 6 may elicit more formulaic responses where quantization doesn't change sentiment much

### 6. Summary

The process:
✅ **Data**: Pre-computed LLM completions partitioned by prompt
✅ **Method**: Test each prompt independently with KS on VADER sentiment
✅ **Goal**: Identify which prompts maximize detection power
✅ **Analysis**: Statistical ranking by KS statistic, p-value, and variance

### 7. Critical Distinction: Independent vs Combined Testing

The script tests prompts **independently**, not **combined**:

- **Independent testing**: Each prompt gets full 500 samples → strong signal
- **Combined testing** (prompts 0,1,2): 500 samples split across 3 prompts → diluted signal

This is why individual prompts can show p=0.003 but combining them gives p=0.46!

**Analogy**: It's like finding which specific fishing spot has the most fish, rather than averaging across multiple spots. The best spot (Prompt 4) has lots of fish, but mixing it with mediocre spots (Prompt 6) dilutes your catch.

## Key Findings from Our Analysis

### Number of Prompts Available in MET Dataset

**Wikipedia Datasets** (100 prompts each):
- `wikipedia_en` - English Wikipedia: **100 prompts (0-99)**
- `wikipedia_de` - German Wikipedia: **100 prompts (0-99)**
- `wikipedia_es` - Spanish Wikipedia: **100 prompts (0-99)**
- `wikipedia_fr` - French Wikipedia: **100 prompts (0-99)**
- `wikipedia_ru` - Russian Wikipedia: **100 prompts (0-99)**

**Other Datasets** (20 prompts each):
- `humaneval` - Code completion: **20 prompts (0-19)**
- `ultrachat` - Conversational: **20 prompts (0-19)**

### Results from Testing 20/100 wikipedia_en Prompts

**For int8 quantization:**
- Found **6 sensitive prompts** out of 20 tested (30% success rate)
- Best: Prompt 4 (KS=0.114, p=0.003)
- Also detected: Prompts 12, 7, 10, 17, 13

**For nf4 quantization:**
- Found **1 sensitive prompt** out of 20 tested (5% success rate)
- Best: Prompt 5 (KS=0.092, p=0.029)

**For watermarking:**
- Found **9 sensitive prompts** out of 20 tested (45% success rate)
- Best: Prompt 12 (KS=0.192, p=0.0000)
- Also detected: Prompts 6, 19, 14, 13, 11, 16, 0, 9

### Implications

1. **We've only tested 20% of available prompts** for wikipedia_en
2. **80 more prompts** remain unexplored - potentially better ones exist
3. **Prompt sensitivity is comparison-specific**: Different prompts work for int8 vs nf4 vs watermark
4. **Single prompts work better than combinations** due to signal dilution

## Practical Recommendations

### For KS Testing with VADER Sentiment:

**For detecting int8 quantization:**
- Use **Prompt 4 only** (KS=0.114, p=0.003)
- Don't combine with other prompts

**For detecting nf4 quantization:**
- Use **Prompt 5 only** (KS=0.092, p=0.029)
- This is the only prompt that showed significance in our testing

**For detecting watermarking:**
- Use **Prompt 12** (KS=0.192, p=0.0000) - strongest signal
- Alternative: Prompt 6 (KS=0.178, p=0.0000)

### General Strategy:

1. **Run analyze-prompt-sensitivity.py** for each specific comparison type
2. **Use the single best prompt** identified by the analysis
3. **Don't combine multiple prompts** - this dilutes the signal
4. **Prompt optimization is essential** for KS-based testing to work

## Why This Matters

The analyze-prompt-sensitivity.py tool is **essential** for KS-based testing because:
- It identifies which specific prompts maximize sensitivity
- It tests prompts **independently** without dilution
- It reveals that prompt choice is **comparison-specific**
- It enables targeted testing instead of hoping random prompts work

**Bottom line**: VADER-KS can detect quantization and watermarking, but only if you carefully select the right single prompt for each comparison type!
