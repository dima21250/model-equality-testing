# Choosing Sentinel Prompts for Black-Box LLM Monitoring

## The Core Problem

**Scenario**: You have a black-box LLM via API and want to detect if the vendor changes the model (e.g., switches quantization to reduce costs).

**What you need**: Prompts that are "sensitive" to model changes

**What you have**: Only today's black-box API responses

**What you can't do**: Test "sensitivity" because that requires comparing two different models/quantizations

**Implication**: You **cannot truly optimize** sentinel prompt selection without already having a change to detect!

---

## Why You Can't Run analyze-prompt-sensitivity.py

The prompt sensitivity analysis requires TWO distributions to compare:
- Source A (e.g., fp32)
- Source B (e.g., int8)

It identifies prompts where the KS test shows:
```python
ks_stat, p_value = ks_2samp(scores_A[prompt], scores_B[prompt])
# "Sensitive" prompts have low p_value
```

**But with a black-box API:**
- You only have ONE version today
- You can't compare "before" vs "after" until "after" happens
- You can't identify sensitive prompts without the comparison

**This is a fundamental limitation.**

---

## Practical Solutions

### **Option 1: Use ALL Prompts with Composite/MMD Test (Recommended)**

**Strategy**: Don't pre-select prompts at all. Test the full distribution.

**Workflow:**

```python
# Today: Collect baseline with ALL 100 prompts
baseline = collect_prompts(api="your-black-box", prompts=range(100))

# Later: Collect monitoring with ALL 100 prompts
monitoring = collect_prompts(api="your-black-box", prompts=range(100))

# Test: Use MMD or composite statistic (single test!)
p_value = run_mmd_test(baseline, monitoring)
# OR
composite_stat = mean([ks_2samp(baseline[i], monitoring[i]).statistic
                       for i in range(100)])
```

**Advantages:**
- ✅ **No prompt selection needed** - avoids the problem entirely
- ✅ **Maximum power** - uses all available information
- ✅ **No multiple testing issue** - single omnibus test
- ✅ **Detects unexpected changes** - doesn't assume what will change

**Disadvantages:**
- ❌ Need to collect 100 prompts at monitoring time (~10 minutes)
- ❌ More complex to implement composite test null distribution

**Implementation:**

```bash
# Baseline collection (one-time)
python collect-baseline.py \
  --api your-black-box-endpoint \
  --prompts 0-99 \
  --samples 500 \
  --output baseline.pkl

# Monitoring (weekly/monthly)
python collect-monitoring.py \
  --api your-black-box-endpoint \
  --prompts 0-99 \
  --samples 500 \
  --output monitoring-week-N.pkl

# Detection test (single test, no prompt selection)
python detect-change.py \
  --baseline baseline.pkl \
  --monitoring monitoring-week-N.pkl \
  --test mmd \
  --alpha 0.05
```

**Cost Analysis:**
- Baseline collection: One-time, ~10 minutes
- Monitoring frequency: Weekly or monthly
- Per-check cost: ~10 minutes
- Annual cost (monthly): ~2 hours/year

**This is probably your best option** since you can't optimize prompt selection anyway.

---

### **Option 2: Transfer Learning from Existing Results**

**Strategy**: Use the prompts we found optimal for Llama/Mistral, hoping they generalize to your LLM.

**Workflow:**

```python
# Use our validated "sensitive" prompts from 100-prompt analysis
sentinel_prompts = [45, 59, 64, 98]  # From our Llama/Mistral analysis

# Collect baseline
baseline = collect_prompts(api="your-black-box", prompts=sentinel_prompts)

# Monitor with same prompts
monitoring = collect_prompts(api="your-black-box", prompts=sentinel_prompts)

# Test with Bonferroni correction
for prompt in sentinel_prompts:
    p = ks_2samp(baseline[prompt], monitoring[prompt]).pvalue
    if p < 0.0125:  # 0.05/4 (Bonferroni correction)
        alert(f"Change detected in prompt {prompt}")
```

**Which prompts to use:**

From our 100-prompt analysis:
- **Prompt 45**: Best for Llama-3-8B nf4 (KS=0.204)
- **Prompt 59**: Best for Mistral-7B int8 (KS=0.156)
- **Prompt 64**: Best for Llama-3-8B int8 (KS=0.142), also good for Mistral nf4
- **Prompt 98**: Best for Mistral-7B nf4 (KS=0.476, extraordinary!)

These prompts showed high sensitivity across different models and quantization types.

**Advantages:**
- ✅ **Fast monitoring** - only 4 prompts (~30 seconds)
- ✅ **Statistically valid** - Bonferroni correction applied
- ✅ **Based on empirical evidence** - these prompts worked for real LLMs
- ✅ **Covers multiple scenarios** - sensitive to different quantization types

**Disadvantages:**
- ❌ **May not transfer** - your LLM might be architecturally different
- ❌ **No guarantee** - we found these for Llama/Mistral, not your model
- ❌ **Might miss changes** - if your LLM's quantization affects different prompts

**Risk Assessment:**
- **Low risk if**: Your black-box LLM is Llama-based or Mistral-based
- **Medium risk if**: Your LLM is a different open-source model (e.g., Falcon, MPT)
- **High risk if**: Your LLM is a proprietary architecture (e.g., GPT-4, Claude, Gemini)

**When to use this:**
- You need fast, frequent monitoring (e.g., hourly/daily)
- You're confident your LLM is similar to Llama/Mistral
- You're willing to accept some risk of missing changes

---

### **Option 3: Heuristic Selection - High Variance Prompts**

**Strategy**: Choose prompts that TODAY show high sentiment variance, hoping variance correlates with sensitivity.

**Workflow:**

```python
# Today: Collect responses for all 100 prompts
baseline = collect_prompts(api="your-black-box", prompts=range(100))

# Analyze variance in sentiment for each prompt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

variances = {}
for prompt_id in range(100):
    scores = [analyzer.polarity_scores(text)["compound"]
              for text in baseline[prompt_id]]
    variances[prompt_id] = np.std(scores)

# Choose top 4 highest-variance prompts as sentinels
sentinel_prompts = sorted(variances.keys(),
                         key=lambda p: variances[p],
                         reverse=True)[:4]

print(f"Sentinel prompts: {sentinel_prompts}")
# e.g., [28, 14, 34, 95]

# Later: Monitor using these prompts
monitoring = collect_prompts(api="your-black-box", prompts=sentinel_prompts)

# Test with Bonferroni correction
for prompt_id in sentinel_prompts:
    p = ks_2samp(baseline[prompt_id], monitoring[prompt_id]).pvalue
    if p < 0.0125:  # 0.05/4
        alert(f"Change detected in prompt {prompt_id}")
```

**Rationale:**
- High-variance prompts elicit diverse responses
- More diverse responses = more opportunity for changes to be detectable
- Our analysis showed high-variance prompts *sometimes* correlate with sensitivity

**Evidence from our analysis:**

From the 100-prompt study, prompts with highest variance:

**Llama-3-8B int8:**
- Prompt 28: max_std=0.552, but KS=0.024 (❌ not sensitive)
- Prompt 14: max_std=0.512, KS=0.080 (borderline)
- Prompt 13: max_std=0.467, KS=0.086 (✅ p=0.0495)

**Mixed results**: High variance sometimes predicts sensitivity, sometimes doesn't.

**Advantages:**
- ✅ **Customized to YOUR LLM** - based on its actual behavior
- ✅ **No transfer assumptions** - doesn't assume similarity to Llama/Mistral
- ✅ **Fast monitoring** - only 4 prompts
- ✅ **Data-driven** - uses your baseline data

**Disadvantages:**
- ❌ **Heuristic only** - variance doesn't guarantee sensitivity
- ❌ **Not validated** - no proof this works reliably
- ❌ **Might miss changes** - low-variance prompts could still be sensitive
- ❌ **Requires full baseline collection** - need 100 prompts upfront anyway

**When to use this:**
- You want a data-driven approach
- You're willing to accept heuristic uncertainty
- You collected full 100-prompt baseline anyway

---

### **Option 4: Stratified Sampling Across Sentiment Ranges**

**Strategy**: Choose prompts that cover the sentiment spectrum.

**Workflow:**

```python
# Collect baseline
baseline = collect_prompts(api="your-black-box", prompts=range(100))

# Compute mean sentiment for each prompt
mean_sentiments = {}
for prompt_id in range(100):
    scores = [analyzer.polarity_scores(text)["compound"]
              for text in baseline[prompt_id]]
    mean_sentiments[prompt_id] = np.mean(scores)

# Choose one from each quartile
sorted_prompts = sorted(mean_sentiments.keys(),
                       key=lambda p: mean_sentiments[p])

sentinel_prompts = [
    sorted_prompts[25],   # Most negative (~25th percentile)
    sorted_prompts[40],   # Somewhat negative (~40th percentile)
    sorted_prompts[60],   # Somewhat positive (~60th percentile)
    sorted_prompts[90],   # Most positive (~90th percentile)
]

print(f"Stratified sentinels: {sentinel_prompts}")
# e.g., [87, 45, 23, 64]
```

**Rationale:**
- Cover different types of content
- If quantization affects sentiment, you'll catch it somewhere
- Diversification strategy - don't put all eggs in one basket

**Advantages:**
- ✅ **Broad coverage** - different sentiment regions
- ✅ **Customized** - based on your LLM's responses
- ✅ **Intuitive** - makes sense from first principles

**Disadvantages:**
- ❌ **No guarantee** - just as arbitrary as random selection
- ❌ **Might miss changes** in the middle ranges
- ❌ **Assumes sentiment matters** - quantization might affect other aspects

**When to use this:**
- You want a principled diversification strategy
- You believe sentiment coverage is important
- You lack other guidance for selection

---

### **Option 5: Use-Case Driven Selection**

**Strategy**: Choose prompts similar to your actual application queries.

**Workflow:**

```python
# If your app does Q&A about products:
sentinel_prompts = [
    prompt_about_electronics,
    prompt_about_clothing,
    prompt_about_food,
    prompt_about_services,
]

# Or if you do customer support:
sentinel_prompts = [
    prompt_technical_question,
    prompt_billing_question,
    prompt_complaint_handling,
    prompt_feature_request,
]

# These are the prompts that matter to YOUR use case
```

**Rationale:**
- You care most about detecting changes that affect YOUR application
- Who cares if Prompt 98 is sensitive if you never use it?
- Business value > statistical optimality

**Advantages:**
- ✅ **Aligned with business value** - detects changes that matter to you
- ✅ **Practical** - focuses on your actual use case
- ✅ **Actionable** - if change detected, you know it affects your app
- ✅ **Explainable** - easy to justify to stakeholders

**Disadvantages:**
- ❌ **Narrow** - might miss broader model changes
- ❌ **Still no guarantee** of sensitivity
- ❌ **Might not cover edge cases**

**When to use this:**
- You have specific use cases you care about
- Your app uses a narrow domain of queries
- Business impact is the primary concern

---

## Comparison of Approaches

| Approach | Speed | Power | Validity | Setup Cost | Risk of Missing Changes |
|----------|-------|-------|----------|------------|------------------------|
| **All-100 + MMD** | Slow (10m) | Highest | Perfect | Low | Lowest |
| **Transfer (45,59,64,98)** | Fast (30s) | Medium | Valid* | None | Medium |
| **High Variance** | Fast (30s) | Unknown | Valid* | Medium | High |
| **Stratified Sentiment** | Fast (30s) | Unknown | Valid* | Medium | High |
| **Use-Case Driven** | Fast (30s) | Unknown | Valid* | Low | Medium** |

*Valid if Bonferroni correction applied
**Lower if use-case prompts happen to be sensitive

---

## My Honest Recommendation

Given that you **cannot truly optimize sentinel selection** without having two versions to compare:

### **Recommended: All-100 with MMD**

**Accept that you can't optimize upfront, so don't try:**

```python
# Baseline collection (one-time)
baseline_all = collect_prompts(api, prompts=range(100), n_samples=500)
save(baseline_all, "baseline.pkl")

# Monitoring (weekly/monthly)
monitoring_all = collect_prompts(api, prompts=range(100), n_samples=500)

# Detection test (single test, no prompt selection)
mmd_p = run_mmd_test(baseline_all, monitoring_all)

if mmd_p < 0.05:
    ALERT("Model changed!")
```

**Why this is best:**

1. **No arbitrary choices** - you don't pretend to know which prompts are "best"
2. **Maximum power** - uses all data
3. **Statistically valid** - single test, no correction needed
4. **Detects any change** - doesn't assume what will change
5. **Future-proof** - works regardless of how the vendor changes the model

**Cost:**
- ~10 minutes per monitoring check
- If you monitor monthly, that's 2 hours/year
- If you monitor weekly, that's 9 hours/year

**This is totally reasonable for production monitoring.**

**Time savings over blind sentinel selection:**
- Arbitrary sentinels: might miss the change entirely → ∞ time to detect!
- All-100: guaranteed to detect if there's a detectable change

---

### **If You MUST Use Sentinels: Pragmatic Compromise**

**Combine transfer learning + variance validation:**

```python
# Start with our empirically validated prompts
candidates = [45, 59, 64, 98]

# Collect baseline for all 100 (you need this anyway for variance check)
baseline = collect_prompts(api, prompts=range(100))

# Check which of our prompts have reasonable variance in YOUR LLM
variances = {}
for p in candidates:
    scores = [get_vader_score(text) for text in baseline[p]]
    variances[p] = np.std(scores)

# Validate that chosen prompts have sufficient variance
sentinel_prompts = []
for prompt in candidates:
    if variances[prompt] > 0.15:  # Reasonable threshold for diversity
        sentinel_prompts.append(prompt)
        print(f"✓ Prompt {prompt}: variance={variances[prompt]:.3f}")
    else:
        print(f"✗ Prompt {prompt}: variance={variances[prompt]:.3f} (too low)")
        # Replace with highest-variance unused prompt
        all_variances = {p: np.std([get_vader_score(text)
                                    for text in baseline[p]])
                        for p in range(100)}
        replacement = max(set(range(100)) - set(sentinel_prompts) - set(candidates),
                         key=lambda p: all_variances[p])
        sentinel_prompts.append(replacement)
        print(f"  → Replaced with Prompt {replacement}: variance={all_variances[replacement]:.3f}")

print(f"\nFinal sentinel prompts: {sentinel_prompts}")
```

**This combines:**
- Our empirical findings (prompts 45, 59, 64, 98 worked for Llama/Mistral)
- Your LLM's characteristics (variance validation)
- A principled replacement strategy (highest variance as tiebreaker)

**Example output:**
```
✓ Prompt 45: variance=0.287
✓ Prompt 59: variance=0.243
✓ Prompt 64: variance=0.193
✗ Prompt 98: variance=0.089 (too low)
  → Replaced with Prompt 34: variance=0.512

Final sentinel prompts: [45, 59, 64, 34]
```

---

## The Uncomfortable Truth

**You cannot optimally choose sentinel prompts with only today's data.**

Your options are:

1. **Don't choose** - use all 100 prompts with MMD (**best**)
2. **Educated guess** - use our validated prompts (**reasonable**)
3. **Heuristic** - use high-variance prompts (**okay**)
4. **Stratified** - cover sentiment spectrum (**okay**)
5. **Use-case** - focus on your domain (**practical**)
6. **Random** - honestly not much worse than heuristics!

The "scientifically correct" answer is **Option 1: don't pre-select**. Use the full distribution and avoid the problem entirely.

---

## Decision Tree

```
Do you need sub-minute monitoring frequency?
│
├─ YES → Use sentinels
│   │
│   ├─ Is your LLM Llama/Mistral-based?
│   │   ├─ YES → Transfer learning (prompts 45,59,64,98)
│   │   └─ NO → High-variance or use-case driven
│   │
│   └─ Can you tolerate 10-30 seconds?
│       └─ Consider 10-20 prompts instead of 4
│
└─ NO → Use all 100 prompts with MMD
    └─ This is the scientifically correct approach
```

---

## Implementation Checklist

### For All-100 + MMD Approach:

- [ ] Collect baseline with all 100 prompts (one-time, ~10 min)
- [ ] Store baseline data securely
- [ ] Set up monitoring schedule (weekly/monthly)
- [ ] Implement MMD test or composite statistic
- [ ] Define alert threshold (α = 0.05)
- [ ] Set up alerting mechanism (email, Slack, etc.)
- [ ] Document baseline collection date and LLM version

### For Sentinel Approach:

- [ ] Choose selection method (transfer, variance, use-case, etc.)
- [ ] Collect baseline for chosen prompts
- [ ] **Also collect full 100 prompts** (for future investigation)
- [ ] Calculate Bonferroni threshold (α = 0.05 / n_sentinels)
- [ ] Set up monitoring schedule
- [ ] Define escalation: if sentinel alerts, run full MMD
- [ ] Document prompt selection rationale
- [ ] Store both sentinel AND full baseline

---

## What to Do If Change Detected

1. **Verify it's not a false alarm**
   - Run multiple monitoring samples
   - Check if change persists

2. **Investigate the change**
   - Collect data with all 100 prompts
   - Run comprehensive MMD test
   - Identify which prompts show largest differences

3. **Assess impact**
   - Test on your actual use cases
   - Measure quality degradation (if any)
   - Determine if action is needed

4. **Document and decide**
   - Log the detected change
   - Update baseline if change is permanent
   - Notify vendor if change violates SLA

5. **Update monitoring**
   - New baseline = current state
   - Continue monitoring for future changes

---

## Conclusion

**The fundamental limitation**: Without two versions to compare, you cannot optimize sentinel selection.

**The pragmatic solution**: Either use all 100 prompts (best) or transfer our findings (reasonable compromise).

**The honest assessment**: Don't pretend you can optimize what you can't measure. Be transparent about limitations.

**The recommendation**: If you can afford 10 minutes per check, use all 100 prompts with MMD. The scientific rigor is worth it.
