# KS-MMD Screening Workflow for LLM Comparison

## The Problem: Speed vs. Rigor Trade-off

When comparing LLM outputs, we have two statistical tests available:

| Test | Speed | Sensitivity | Statistical Issues |
|------|-------|-------------|-------------------|
| **MMD (Hamming)** | ~130-190s per comparison | High - detects all differences | Single joint test - no correction needed |
| **VADER-KS** | ~0.06-0.08s per comparison | Lower - misses quantization | Multiple testing problem when screening prompts |

**The Challenge**:
- MMD is ~2,000-3,000x slower than KS
- But KS has limited sensitivity and requires careful prompt selection
- Testing many prompts with KS creates a multiple comparisons problem

## The Multiple Testing Problem

### Why Correction is Needed

If you test 100 prompts at α=0.05 to find the "best" one:
- **Expected false positives**: ~5 prompts (5%) will show p<0.05 just by random chance
- Even with **identical LLMs**, you'll find "significant" prompts!
- Selecting the "best" p-value after testing invalidates statistical guarantees

### Bonferroni Correction

For N=100 prompts, the corrected threshold is:
```
α_corrected = 0.05 / 100 = 0.0005
```

**Impact on our results:**
- **int8 quantization**: Prompt 4 (p=0.003) → **FAILS** ❌
- **nf4 quantization**: Prompt 5 (p=0.029) → **FAILS** ❌
- **Watermarking**: Prompt 12 (p=0.0000) → **PASSES** ✅

With proper correction, KS alone **cannot claim to detect quantization**.

## The Solution: Screening Workflow

**Key Insight**: Use KS as a **fast screening tool**, then confirm with rigorous MMD testing.

This approach:
- ✅ Avoids the multiple testing problem (MMD is the confirmatory test)
- ✅ Leverages KS's speed advantage for efficient exploration
- ✅ Provides statistically valid conclusions via MMD
- ✅ Reduces computational cost compared to blind MMD testing

### Conceptual Framework

```
┌─────────────────────────────────────────────────────────┐
│                    SCREENING PHASE                       │
│         (Fast, exploratory - no p-value claims)         │
├─────────────────────────────────────────────────────────┤
│  1. Run VADER-KS on all 100 prompts                     │
│  2. Identify "interesting" candidates (p < 0.05)        │
│  3. Rank by KS statistic                                │
│                                                          │
│  Time: ~100 minutes (1 min/prompt)                      │
│  Output: List of candidate prompts                      │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                  CONFIRMATION PHASE                      │
│        (Rigorous, definitive - valid p-values)          │
├─────────────────────────────────────────────────────────┤
│  1. Run MMD on top candidate prompts                    │
│  2. MMD tests joint distribution - no correction needed │
│  3. Report MMD p-values as confirmatory evidence        │
│                                                          │
│  Time: ~3 minutes per test                              │
│  Output: Statistically valid conclusion                 │
└─────────────────────────────────────────────────────────┘
```

## Detailed Workflow

### Step 1: Screening with KS (Exploratory)

**Goal**: Quickly identify which prompts might show sensitivity to the comparison of interest

**Command**:
```bash
python analyze-prompt-sensitivity.py \
  --dataset wikipedia_en \
  --source_a fp32 \
  --source_b int8 \
  --num_prompts 100 \
  --L 200 \
  --samples 500
```

**Output**:
```
TOP 5 PROMPTS BY KS STATISTIC:
1. Prompt 4: KS=0.1140, p=0.0030
2. Prompt 12: KS=0.1000, p=0.0134
3. Prompt 7: KS=0.0960, p=0.0199
4. Prompt 10: KS=0.0940, p=0.0241
5. Prompt 17: KS=0.0900, p=0.0348
```

**Interpretation**:
- These prompts are **candidates** for further investigation
- The p-values are **not corrected** - this is exploratory
- We're generating hypotheses, not testing them

**Time**: ~100 minutes for 100 prompts

### Step 2: Confirmation with MMD (Rigorous)

**Goal**: Definitively test whether distributions differ using the most sensitive prompts

**Command** (using top candidates from screening):
```bash
python mmd-ks-comparison.py \
  --model_a meta-llama/Meta-Llama-3-8B-Instruct \
  --model_b meta-llama/Meta-Llama-3-8B-Instruct \
  --source_a fp32 \
  --source_b int8 \
  --dataset wikipedia_en \
  --prompts 4 12 7 \
  --L 200 \
  --samples 500
```

**Output**:
```
MMD statistic = 0.000838, p-value = 0.0290, elapsed = 155.163s
KS  statistic = 0.024000, p-value = 0.9988, elapsed = 0.077s
```

**Interpretation**:
- **MMD p-value (0.029)** is the confirmatory evidence
- This is a **single test** of the joint distribution - no multiple testing issue
- The KS result is reported for comparison but not relied upon

**Time**: ~3 minutes

### Step 3: Reporting

**Report the workflow honestly**:

```markdown
We used a two-phase screening workflow:

1. **Screening Phase**: We tested 100 prompts using VADER-KS to identify
   candidates with potential sensitivity (p < 0.05, uncorrected). This
   exploratory analysis identified 6 promising prompts.

2. **Confirmation Phase**: We ran MMD tests on the top 3 candidate prompts
   (4, 12, 7) identified in screening. MMD confirmed significant differences
   between fp32 and int8 sources (p = 0.029).

Total time: 103 minutes (screening + confirmation)
vs. ~500 minutes for blind MMD testing of all prompt combinations
```

## Computational Efficiency Analysis

### Scenario: Testing one source comparison

**Naive approach (blind MMD testing)**:
- Test all reasonable prompt combinations
- Example: Test 10 different 3-prompt combinations
- Time: 10 × 3 min = **30 minutes**

**Screening workflow**:
- Screen 100 prompts with KS: **100 minutes**
- Confirm best combination with MMD: **3 minutes**
- Total: **103 minutes**

**Not faster in this case!**

### Scenario: Testing multiple comparisons

**Testing 10 different source pairs** (e.g., fp32 vs int8, fp32 vs nf4, etc.)

**Naive approach**:
- 10 comparisons × 30 minutes = **300 minutes**

**Screening workflow**:
- Screen each comparison: 10 × 100 min = 1000 minutes
- Confirm each: 10 × 3 min = 30 minutes
- Total: **1030 minutes**

**Much slower!**

### Scenario: Screening NEW comparisons with cached prompt rankings

**Key insight**: Once you know which prompts are sensitive for a given modification type, you can reuse that knowledge!

**First time** (discovering sensitive prompts):
- Screen 100 prompts: 100 min
- Confirm: 3 min
- **Total: 103 minutes**

**Subsequent tests** (using known-good prompts):
- No screening needed - use Prompt 4 for int8, Prompt 5 for nf4, etc.
- Just run MMD on the pre-identified prompt: **3 minutes**

**This is where the workflow pays off**: After initial exploration, you can quickly test new data using the validated prompts.

## When to Use This Workflow

### ✅ **Good Use Cases**:

1. **Initial exploration** of a new comparison type
   - Don't know which prompts are sensitive
   - Want to understand the landscape
   - Building a catalog of sensitive prompts

2. **Resource-constrained settings**
   - Can't afford to run MMD on everything
   - Need to prioritize which comparisons to investigate deeply
   - Want to filter out clearly identical distributions

3. **Building a test suite**
   - Creating validated prompts for future use
   - One-time cost of screening, ongoing benefit of known-good prompts
   - Documentation of which prompts work for which comparisons

### ❌ **Poor Use Cases**:

1. **Single comparison with unknown prompts**
   - Screening is slower than just running MMD
   - Better to test a few reasonable prompt combinations directly

2. **When you already know good prompts**
   - Skip screening, go straight to MMD confirmation
   - Use validated prompts from prior work

3. **When you need maximum sensitivity**
   - Just use MMD from the start
   - KS screening might miss subtle differences

## Statistical Validity

### What You CAN Claim:

✅ "We used VADER-KS screening to identify candidate prompts, then confirmed with MMD (p = 0.029)"

✅ "KS screening reduced the search space from 100 prompts to 6 candidates, which we validated with MMD"

✅ "Our workflow combines the speed of KS screening with the rigor of MMD confirmation"

### What You CANNOT Claim:

❌ "KS detected quantization effects (p = 0.003)" ← Uncorrected p-value from screening

❌ "Testing 100 prompts confirmed differences" ← Unless you apply Bonferroni correction

❌ "KS is sufficient for detecting quantization" ← Need MMD for confirmation

### The Key Principle:

**Screening is exploratory; MMD is confirmatory.**

- KS p-values during screening are **not reported as evidence**
- Only MMD p-values from the confirmation phase are **statistically valid conclusions**
- The workflow is about **efficiency**, not replacing rigorous testing

## Alternative Approaches (If You Want to Claim KS Detection)

If you want to make claims based on KS testing alone, you have these options:

### **Option 1: Bonferroni Correction**
```python
# Test all 100 prompts
# Only claim detection if ANY prompt has p < 0.05/100 = 0.0005
# Very conservative, low power
```

### **Option 2: Train/Test Split**
```python
# Use prompts 0-49 to find best prompt (exploration)
# Test that prompt on prompts 50-99 (confirmation)
# p-value from held-out prompts is valid
```

### **Option 3: Pre-registration**
```python
# Before seeing data, specify: "We will test prompts 4, 12, 7"
# Test only those 3 prompts
# Correction: α = 0.05 / 3 = 0.0167
# Prompt 4 (p=0.003) would PASS
```

### **Option 4: False Discovery Rate Control**
```python
# Use Benjamini-Hochberg instead of Bonferroni
# Less conservative, controls proportion of false discoveries
# Better power than Bonferroni
```

## Practical Example

### Testing fp32 vs int8 Quantization

**Phase 1: Screening (100 prompts)**
```bash
python analyze-prompt-sensitivity.py \
  --source_a fp32 \
  --source_b int8 \
  --num_prompts 100
```

**Results**:
- 6 prompts with p < 0.05 (uncorrected)
- Top candidate: Prompt 4 (KS=0.114, p=0.003)

**Phase 2: Confirmation (best 3 prompts)**
```bash
python mmd-ks-comparison.py \
  --source_a fp32 \
  --source_b int8 \
  --prompts 4 12 7
```

**Results**:
- MMD: p = 0.029 ✅ **Significant at α=0.05**
- **Conclusion**: int8 quantization produces detectably different distributions

**Time**: 103 minutes total

**For Future int8 Tests**:
- Just use prompts 4, 12, 7 directly
- Skip screening phase
- **3 minutes** per test

## Summary

### The Screening Workflow:

1. **Use VADER-KS** to quickly scan many prompts (exploratory)
2. **Identify candidates** with potential sensitivity
3. **Confirm with MMD** on top candidates (rigorous)
4. **Report MMD results** as the definitive evidence

### Benefits:

- ✅ **Statistically valid**: MMD provides confirmation without multiple testing issues
- ✅ **Efficient exploration**: KS helps identify promising prompts quickly
- ✅ **Honest reporting**: Clear distinction between exploratory and confirmatory phases
- ✅ **Reusable knowledge**: Build a catalog of validated prompts for future tests
- ✅ **Practical value**: ~97% time savings on subsequent tests using validated prompts

### Key Insight:

**KS is not a replacement for MMD - it's a tool to make MMD testing more efficient.**

The workflow leverages the 2000x speed advantage of KS for exploration while maintaining the statistical rigor of MMD for confirmation.
