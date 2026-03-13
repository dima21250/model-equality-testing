# MMD vs VADER-KS Test Comparison Results

## Objective
Compare the sensitivity and computational efficiency of two statistical tests for detecting distributional differences in LLM outputs:
- **MMD (Maximum Mean Discrepancy)** with hamming kernel
- **VADER-based Kolmogorov-Smirnov** two-sample test

## Test Configuration
- **Model**: `meta-llama/Meta-Llama-3-8B-Instruct`
- **Dataset**: `wikipedia_en`
- **Prompts**: 0, 1, 2
- **Completion Length (L)**: 200
- **Samples per distribution**: 500

## Results Summary

| Source Comparison | MMD Statistic | MMD p-value | MMD Time (s) | KS Statistic | KS p-value | KS Time (s) | KS Detected? |
|-------------------|---------------|-------------|--------------|--------------|------------|-------------|--------------|
| **fp32 vs int8** | 0.004339 | 0.0000 | 130.104 | 0.056000 | 0.4135 | 0.056 | ❌ No |
| **fp32 vs nf4** | 0.003545 | 0.0000 | 129.848 | 0.044000 | 0.7189 | 0.058 | ❌ No |
| **fp32 vs watermark** | 0.010374 | 0.0000 | 130.085 | 0.102000 | 0.0110 | 0.058 | ✅ Yes |

*Significance threshold: α = 0.05*

## Detailed Results

### 1. fp32 vs int8 (32-bit → 8-bit quantization)

**MMD Test:**
- Statistic: 0.004339
- p-value: 0.0000 (< 0.05)
- Elapsed time: 130.104s
- Result: ✅ **Detected difference**

**VADER KS Test:**
- Statistic: 0.056000
- p-value: 0.4135 (>> 0.05)
- Elapsed time: 0.056s
- Result: ❌ **Failed to detect difference**

**Speed-up**: ~2,323x faster (but failed to detect)

---

### 2. fp32 vs nf4 (32-bit → 4-bit quantization)

**MMD Test:**
- Statistic: 0.003545
- p-value: 0.0000 (< 0.05)
- Elapsed time: 129.848s
- Result: ✅ **Detected difference**

**VADER KS Test:**
- Statistic: 0.044000
- p-value: 0.7189 (>> 0.05)
- Elapsed time: 0.058s
- Result: ❌ **Failed to detect difference**

**Speed-up**: ~2,239x faster (but failed to detect)

---

### 3. fp32 vs watermark

**MMD Test:**
- Statistic: 0.010374
- p-value: 0.0000 (< 0.05)
- Elapsed time: 130.085s
- Result: ✅ **Detected difference**

**VADER KS Test:**
- Statistic: 0.102000
- p-value: 0.0110 (< 0.05)
- Elapsed time: 0.058s
- Result: ✅ **Detected difference**

**Speed-up**: ~2,243x faster (successfully detected)

---

## Key Findings

### 1. Sensitivity Comparison
- **MMD**: Successfully detected distributional differences in all three comparisons (int8, nf4, watermark)
- **VADER-KS**: Only detected watermarking; failed to detect quantization effects (int8, nf4)
- The KS test appears to lose too much information when projecting text distributions to sentiment scores

### 2. Effect Size Analysis
The KS statistic magnitude correlates with detectability:
- **Watermark**: 0.102 (detected, p=0.0110)
- **int8 quantization**: 0.056 (not detected, p=0.4135)
- **nf4 quantization**: 0.044 (not detected, p=0.7189)

Watermarking produces a **2.3x larger** KS statistic than int8 quantization, suggesting it has a more substantial impact on sentiment distributions.

### 3. Computational Efficiency
- **KS Test**: Consistently ~0.056-0.058 seconds
- **MMD Test**: Consistently ~130 seconds (1000 permutation bootstraps)
- **Speed-up**: ~2,200-2,300x faster

### 4. Implications

**Quantization vs. Watermarking:**
- **Quantization** (int8, nf4) changes model precision but may preserve relative sentiment patterns, making differences undetectable via sentiment-based tests
- **Watermarking** actively modifies token selection to encode information, which creates measurable shifts in sentiment distributions

**When to use each test:**
- **MMD**: When you need high sensitivity and can afford the computational cost (~2 minutes per comparison)
- **VADER-KS**: When you need fast screening for large distributional shifts and can accept lower sensitivity

**VADER projection limitations:**
- The VADER sentiment projection collapses entire text distributions into single scalar values (compound sentiment)
- This lossy compression discards granular distributional information needed to detect subtle quantization effects
- More informative feature projections might improve KS test sensitivity

## Conclusion

The comparison validates the sensitivity vs. speed trade-off:
- **MMD** is the gold standard for detecting distributional differences but is computationally expensive
- **VADER-based KS** is ~2,200x faster but only detects large, semantically-meaningful shifts
- For quantization testing, MMD is necessary; for watermark detection, the fast KS test is sufficient

## Command Used

```bash
# fp32 vs int8
python mmd-ks-comparison.py \
  --model_a meta-llama/Meta-Llama-3-8B-Instruct \
  --model_b meta-llama/Meta-Llama-3-8B-Instruct \
  --source_a fp32 \
  --source_b int8 \
  --dataset wikipedia_en \
  --prompts 0 1 2 \
  --L 200 \
  --samples 500

# fp32 vs nf4
python mmd-ks-comparison.py \
  --model_a meta-llama/Meta-Llama-3-8B-Instruct \
  --model_b meta-llama/Meta-Llama-3-8B-Instruct \
  --source_a fp32 \
  --source_b nf4 \
  --dataset wikipedia_en \
  --prompts 0 1 2 \
  --L 200 \
  --samples 500

# fp32 vs watermark
python mmd-ks-comparison.py \
  --model_a meta-llama/Meta-Llama-3-8B-Instruct \
  --model_b meta-llama/Meta-Llama-3-8B-Instruct \
  --source_a fp32 \
  --source_b watermark \
  --dataset wikipedia_en \
  --prompts 0 1 2 \
  --L 200 \
  --samples 500
```

## Date
March 13, 2026
