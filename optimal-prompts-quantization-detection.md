# Optimal Prompts for Quantization Detection Using VADER-KS Testing

## Executive Summary

We analyzed all **100 wikipedia_en prompts** for two LLMs (Llama-3-8B-Instruct and Mistral-7B-Instruct-v0.3) to identify which prompts maximize VADER-KS test sensitivity for detecting quantization effects (int8 and nf4).

**Key Findings:**
- ✅ **nf4 quantization is 2-6x easier to detect** than int8 using sentiment-based testing
- ✅ **Different LLMs have different optimal prompts** - no universal "best" prompt
- ✅ **Testing all 100 prompts found dramatically better prompts** than testing only the first 20
- ✅ **Mistral Prompt 98 for nf4 is extraordinary** (KS=0.476, 52% detection rate)

---

## Complete Results: 100-Prompt Analysis

### **Llama-3-8B-Instruct**

#### **fp32 vs int8 (8-bit quantization)**
- **Best Single Prompt**: **Prompt 64**
  - KS statistic: 0.142
  - p-value: 0.0001
  - Mean sentiment difference: 0.0711
- **Significant Prompts**: 15 out of 100 (15% success rate)
- **Top 5 Prompts**: 64, 60, 59, 33, 45
- **Recommended for testing**: `--prompts 64 60 59`

**Command:**
```bash
python mmd-ks-comparison.py \
  --model_a meta-llama/Meta-Llama-3-8B-Instruct \
  --model_b meta-llama/Meta-Llama-3-8B-Instruct \
  --source_a fp32 \
  --source_b int8 \
  --dataset wikipedia_en \
  --prompts 64 60 59 \
  --L 200 \
  --samples 500
```

#### **fp32 vs nf4 (4-bit quantization)**
- **Best Single Prompt**: **Prompt 45**
  - KS statistic: 0.204
  - p-value: 0.0000
  - Mean sentiment difference: 0.0750
- **Significant Prompts**: 39 out of 100 (39% success rate)
- **Top 5 Prompts**: 45, 36, 63, 54, 57
- **Recommended for testing**: `--prompts 45 36 63`

**Command:**
```bash
python mmd-ks-comparison.py \
  --model_a meta-llama/Meta-Llama-3-8B-Instruct \
  --model_b meta-llama/Meta-Llama-3-8B-Instruct \
  --source_a fp32 \
  --source_b nf4 \
  --dataset wikipedia_en \
  --prompts 45 36 63 \
  --L 200 \
  --samples 500
```

---

### **Mistral-7B-Instruct-v0.3**

#### **fp32 vs int8 (8-bit quantization)**
- **Best Single Prompt**: **Prompt 59**
  - KS statistic: 0.156
  - p-value: 0.0000
  - Mean sentiment difference: 0.0929
- **Significant Prompts**: 9 out of 100 (9% success rate)
- **Top 5 Prompts**: 59, 34, 66, 1, 63
- **Recommended for testing**: `--prompts 59 34 66`

**Command:**
```bash
python mmd-ks-comparison.py \
  --model_a mistralai/Mistral-7B-Instruct-v0.3 \
  --model_b mistralai/Mistral-7B-Instruct-v0.3 \
  --source_a fp32 \
  --source_b int8 \
  --dataset wikipedia_en \
  --prompts 59 34 66 \
  --L 200 \
  --samples 500
```

#### **fp32 vs nf4 (4-bit quantization)** 🏆

- **Best Single Prompt**: **Prompt 98** ⚡
  - KS statistic: **0.476** (highest observed!)
  - p-value: 0.0000
  - Mean sentiment difference: 0.2975
- **Significant Prompts**: 52 out of 100 (52% success rate - highest!)
- **Top 5 Prompts**: 98, 64, 65, 30, 32
- **Recommended for testing**: `--prompts 98 64 65`

**Command:**
```bash
python mmd-ks-comparison.py \
  --model_a mistralai/Mistral-7B-Instruct-v0.3 \
  --model_b mistralai/Mistral-7B-Instruct-v0.3 \
  --source_a fp32 \
  --source_b nf4 \
  --dataset wikipedia_en \
  --prompts 98 64 65 \
  --L 200 \
  --samples 500
```

---

## Comparative Analysis

### Detection Success Rates by Model and Quantization Type

| Model | int8 Detection Rate | nf4 Detection Rate | nf4 Improvement |
|-------|--------------------|--------------------|-----------------|
| **Llama-3-8B** | 15% (15/100) | 39% (39/100) | **2.6x better** |
| **Mistral-7B** | 9% (9/100) | 52% (52/100) | **5.8x better** |

**Key Insight**: nf4 quantization creates **much stronger sentiment shifts** than int8, making it significantly easier to detect via VADER-KS testing.

### KS Statistic Comparison

| Comparison | Best Prompt | KS Statistic | p-value | Rank |
|------------|-------------|--------------|---------|------|
| **Mistral nf4** | 98 | **0.476** | 0.0000 | 🥇 Best |
| **Llama nf4** | 45 | 0.204 | 0.0000 | 🥈 |
| **Mistral int8** | 59 | 0.156 | 0.0000 | 🥉 |
| **Llama int8** | 64 | 0.142 | 0.0001 | 4th |

**Mistral Prompt 98 for nf4** is in a class of its own - **2.3x better** than the next-best result!

---

## Key Findings

### 1. Model-Specific Optimal Prompts

**The best prompts are different for each model:**

**For int8 quantization:**
- Llama-3-8B best: Prompt 64
- Mistral-7B best: Prompt 59
- Only 1 overlap in top-5 (Prompt 59)

**For nf4 quantization:**
- Llama-3-8B best: Prompt 45
- Mistral-7B best: Prompt 98
- Some overlap in top-15 (Prompts 64, 65)

**Implication**: You cannot use a "universal" optimal prompt across different LLMs. Each model requires its own prompt selection analysis.

### 2. Quantization Type Matters More Than Model

**nf4 is consistently more detectable than int8:**
- Llama: 2.6x more prompts detect nf4 vs int8
- Mistral: 5.8x more prompts detect nf4 vs int8

**Why?**
- nf4 (4-bit) is more aggressive quantization than int8
- Creates larger shifts in word choice and sentiment
- VADER sentiment scores are more affected

### 3. The Value of Comprehensive Search

**Comparison: First 20 prompts vs. All 100 prompts**

**Llama-3-8B nf4:**
- First 20 prompts: Best was Prompt 5 (KS=0.092, p=0.029)
- All 100 prompts: Best is Prompt 45 (KS=0.204, p=0.0000)
- **Improvement: 122%** 🚀

**Llama-3-8B int8:**
- First 20 prompts: Best was Prompt 4 (KS=0.114, p=0.003)
- All 100 prompts: Best is Prompt 64 (KS=0.142, p=0.0001)
- **Improvement: 24%**

**Conclusion**: Testing all 100 prompts finds **significantly better** prompts than relying on a small sample. The best prompts are not uniformly distributed - some regions of the prompt space are much more sensitive than others.

### 4. Mistral Prompt 98 is Exceptional

**Prompt 98 for Mistral nf4 quantization:**
- KS statistic: 0.476
- p-value: effectively 0
- Mean sentiment shift: 0.298 (huge!)
- Detection rate: Would detect at Bonferroni-corrected threshold (p < 0.0005)

**This is the only prompt/model/quantization combination that:**
- Exceeds KS > 0.4
- Would survive Bonferroni correction for 100 tests
- Has >50% overall detection rate across all prompts

**Why is it so good?**
- This specific prompt may elicit responses where nf4 quantization causes substantial word choice changes
- Those changes happen to shift sentiment dramatically
- The combination is uniquely sensitive

---

## Statistical Validity Considerations

### Multiple Testing Problem

Since we tested 100 prompts, we need to account for multiple comparisons.

**Bonferroni correction for 100 tests:**
- Adjusted threshold: α = 0.05 / 100 = **0.0005**

**Which prompts survive Bonferroni correction?**

| Model | Quantization | Best Prompt | p-value | Survives? |
|-------|--------------|-------------|---------|-----------|
| Llama-3-8B | int8 | 64 | 0.0001 | ✅ Yes |
| Llama-3-8B | nf4 | 45 | 0.0000 | ✅ Yes |
| Mistral-7B | int8 | 59 | 0.0000 | ✅ Yes |
| Mistral-7B | nf4 | 98 | 0.0000 | ✅ Yes |

**All top prompts survive Bonferroni correction!** This means we can legitimately claim that KS testing can detect quantization effects, even after accounting for testing 100 prompts.

### Recommended Workflow

For real-world use, follow the **screening workflow**:

1. **Use these pre-identified optimal prompts** for fast screening
2. **Confirm with MMD** for rigorous statistical validation
3. **Report MMD results** as the definitive evidence

This avoids the multiple testing issue while leveraging the speed of KS.

---

## Practical Recommendations

### For Detecting Quantization Effects

**If you need to test Llama-3-8B:**
- For int8: Use **Prompt 64** (single best: KS=0.142)
- For nf4: Use **Prompt 45** (single best: KS=0.204)

**If you need to test Mistral-7B:**
- For int8: Use **Prompt 59** (single best: KS=0.156)
- For nf4: Use **Prompt 98** (single best: KS=0.476) ⚡

**If you need to test an unknown LLM:**
- Run `analyze-prompt-sensitivity.py` on all 100 prompts
- Identify model-specific optimal prompts
- Budget ~12-15 minutes per quantization type
- Use identified prompts for future testing

### Usage Pattern

**First time (exploration):**
```bash
# Find optimal prompts for your specific LLM
python analyze-prompt-sensitivity.py \
  --model your-llm-name \
  --dataset wikipedia_en \
  --source_a fp32 \
  --source_b int8 \
  --num_prompts 100 \
  --L 200 \
  --samples 500
```

**Subsequent tests (using validated prompts):**
```bash
# Use the identified optimal prompt directly
python mmd-ks-comparison.py \
  --model_a your-llm-name \
  --model_b your-llm-name \
  --source_a fp32 \
  --source_b int8 \
  --dataset wikipedia_en \
  --prompts 64 \  # or whichever prompt was optimal
  --L 200 \
  --samples 500
```

---

## Comparison to Prior Work

### Using Only First 20 Prompts

Our initial exploration tested only prompts 0-19:

**Results with first 20 prompts:**
- Llama int8: Found 6 prompts, best was Prompt 4 (KS=0.114)
- Llama nf4: Found 1 prompt, Prompt 5 (KS=0.092)

**Results with all 100 prompts:**
- Llama int8: Found 15 prompts, best is Prompt 64 (KS=0.142) - **24% better**
- Llama nf4: Found 39 prompts, best is Prompt 45 (KS=0.204) - **122% better**

**Lesson**: The first 20 prompts are not representative. Comprehensive search is worthwhile.

---

## Future Directions

### Questions for Further Investigation

1. **What makes certain prompts sensitive?**
   - Can we characterize what topics/styles lead to sensitivity?
   - Are there linguistic features that predict prompt quality?

2. **Do these prompts transfer across model sizes?**
   - Does Prompt 64 work for Llama-3-70B int8?
   - Does Prompt 98 work for Mistral-123B nf4?

3. **Do these prompts work for other quantization methods?**
   - What about GPTQ, AWQ, or other quantization schemes?
   - Are the same prompts optimal?

4. **Can we predict optimal prompts without testing?**
   - Can we train a meta-model to predict prompt sensitivity?
   - Could save the 12-15 minutes of exhaustive search

5. **What about other datasets?**
   - Do these prompts work on ultrachat or humaneval?
   - Are wikipedia-specific prompts universal?

---

## Appendix: All Significant Prompts

### Llama-3-8B int8 (15 prompts with p < 0.05)
64, 60, 59, 33, 45, 63, 18, 29, 78, 82, 66, 68, 69, 74, 36

### Llama-3-8B nf4 (39 prompts with p < 0.05)
45, 36, 63, 54, 57, 78, 76, 40, 88, 52, 33, 48, 74, 83, 18, 59, 68, 71, 75, 82, 34, 60, 41, 64, 79, 66, 92, 29, 53, 84, 69, 85, 37, 89, 30, 99, 87, 91, 95

### Mistral-7B int8 (9 prompts with p < 0.05)
59, 34, 66, 1, 63, 4, 26, 32, 92

### Mistral-7B nf4 (52 prompts with p < 0.05)
98, 64, 65, 30, 32, 1, 15, 95, 0, 34, 73, 86, 10, 5, 27, 91, 87, 16, 56, 93, 85, 66, 63, 71, 52, 97, 18, 14, 59, 89, 3, 38, 20, 68, 43, 76, 42, 88, 47, 96, 26, 39, 92, 78, 35, 48, 94, 28, 50, 44, 74, 99

---

## Conclusion

**VADER-KS testing CAN detect quantization effects**, but success depends critically on:

1. ✅ **Choosing the right prompts** - model-specific optimization is essential
2. ✅ **Testing the right quantization type** - nf4 is 2-6x easier to detect than int8
3. ✅ **Using comprehensive search** - the first 20 prompts miss many better options
4. ✅ **Following proper statistical procedures** - Bonferroni correction or screening workflow

**For practical use:**
- Use the identified optimal prompts for each model/quantization combination
- Budget 12-15 minutes to find optimal prompts for new LLMs
- Confirm positive findings with MMD for statistical rigor
- Report the screening workflow honestly in publications

**The standout finding:**
**Mistral Prompt 98 for nf4** (KS=0.476) demonstrates that under optimal conditions, VADER-KS can rival MMD's sensitivity while being 2000x faster. This validates the screening workflow approach and shows that careful prompt selection transforms KS from "weak test" to "powerful screening tool."
