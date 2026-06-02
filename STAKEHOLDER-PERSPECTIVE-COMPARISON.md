# Stakeholder Perspective: Permutation Testing vs Bootstrap Confidence Intervals

## Overview

When implementing statistical inference for density matrix metrics, there are two main approaches: permutation testing (Option 1) and bootstrap confidence intervals (Option 2). This document explains how stakeholders will interpret and use results from each approach.

The fundamental difference is about **what question gets answered** and **how you communicate uncertainty**.

---

## Option 1: Permutation Testing

### What the stakeholder receives:

```
Trace distance: 0.2341
P-value: 0.003
Conclusion: Distributions are significantly different (p < 0.05)
```

### The question answered:

**"Are these two LLMs statistically different, or could this just be random sampling variation?"**

### Decision framing:

**Binary (yes/no)**:
- p < 0.05 → "Yes, they're different"
- p ≥ 0.05 → "No significant difference detected"

### Stakeholder interpretation:

- ✓ **Easy to understand**: "Is there a real difference?" → Yes or No
- ✓ **Clear decision rule**: Standard significance threshold (p < 0.05)
- ✓ **Familiar framework**: Everyone knows p-values (even if misinterpret them)
- ✗ **Loses magnitude uncertainty**: Doesn't tell you the range of plausible effect sizes
- ✗ **Binary thinking**: Encourages "significant vs. not significant" without nuance

### Example stakeholder conversation:

> **Stakeholder**: "Did the quantization change the model?"
> 
> **You**: "Yes, with high confidence. The p-value is 0.003, meaning there's only a 0.3% chance we'd see a difference this large if the model hadn't changed. We can confidently say the model is different."
>
> **Stakeholder**: "How different?"
>
> **You**: "The trace distance is 0.23, which indicates moderate distinguishability." (But you still have no uncertainty around that 0.23)

---

## Option 2: Bootstrap Confidence Intervals

### What the stakeholder receives:

```
Trace distance: 0.2341
95% Confidence Interval: [0.1875, 0.2894]
Conclusion: Significantly different from zero (CI excludes 0)
```

### The question answered:

**"How different are these LLMs, and what's the range of plausible values accounting for sampling uncertainty?"**

### Decision framing:

**Range-based**:
- CI excludes 0 → "Different"
- CI includes 0 → "Could be the same"
- Width of CI → "How confident we are"

### Stakeholder interpretation:

- ✓ **Quantifies uncertainty**: Not just "0.23" but "somewhere between 0.19 and 0.29"
- ✓ **More informative**: Shows precision of the estimate
- ✓ **Nuanced**: Can say "definitely different, but effect size uncertain"
- ✗ **Harder to explain**: What does "95% CI" mean? (Most people get this wrong)
- ✗ **No standard threshold**: Is [0.18, 0.29] a "big" difference? Still unclear

### Example stakeholder conversation:

> **Stakeholder**: "Did the quantization change the model?"
> 
> **You**: "Yes. The trace distance is 0.23, and we're 95% confident the true value is between 0.19 and 0.29. Since this range doesn't include zero, we can conclude the distributions differ."
>
> **Stakeholder**: "Is that a big difference?"
>
> **You**: "It's moderate. The effect could be as low as 0.19 (small-to-moderate) or as high as 0.29 (moderate), but we can rule out 'no difference' or 'very large difference'."

---

## Side-by-Side Comparison

| Aspect | Permutation Test (Option 1) | Bootstrap CI (Option 2) |
|--------|---------------------------|------------------------|
| **Output** | td = 0.23, p = 0.003 | td = 0.23, 95% CI [0.19, 0.29] |
| **Primary question** | "Is it significant?" | "What's the plausible range?" |
| **Decision** | Binary (sig / not sig) | Based on CI excluding threshold |
| **Uncertainty** | About the null hypothesis | About the effect size |
| **Ease of communication** | High (p < 0.05 is familiar) | Medium (CI needs explanation) |
| **Nuance** | Low (binary) | High (shows precision) |
| **Follow-up question** | "How big is the effect?" | "Is this big enough to matter?" |

---

## Real-World Scenarios

### Scenario 1: Executive Summary

**Permutation test version**:
> "We tested whether quantization changed the model. The difference is statistically significant (p < 0.001). The model has changed."

**Bootstrap CI version**:
> "We tested whether quantization changed the model. The trace distance is 0.23 (95% CI: 0.19–0.29), indicating a moderate difference. We can rule out no change or large changes."

**Which is clearer?** Depends on audience. Executives may prefer the first (yes/no answer). Technical reviewers may prefer the second (quantified uncertainty).

---

### Scenario 2: Borderline Case

Imagine: td = 0.08, p = 0.06, 95% CI = [0.04, 0.13]

**Permutation test says**:
> "No significant difference (p = 0.06 > 0.05). We cannot reject the null hypothesis."

**Bootstrap CI says**:
> "Trace distance is 0.08 (95% CI: 0.04–0.13). There appears to be a small difference, though the effect could be as low as 0.04 or as high as 0.13."

**Which is more useful?** 

The permutation test gives a binary "not significant" which might be misleading (p = 0.06 is close to 0.05, and there's clearly some effect).

The CI shows there IS an effect (CI doesn't include 0), just a small one with uncertainty. More informative for decision-making.

**Key difference**: p-value emphasizes "evidence against null", CI emphasizes "magnitude and precision of effect."

---

### Scenario 3: Large Sample (High Precision)

Imagine: n = 1000, td = 0.05, p < 0.001, 95% CI = [0.048, 0.052]

**Permutation test says**:
> "Highly significant (p < 0.001). Distributions are different."

**Bootstrap CI says**:
> "Trace distance is 0.05 (95% CI: 0.048–0.052). Distributions differ, but the effect is very small and precisely estimated."

**Which is more useful?**

The permutation test might overstate importance ("highly significant!") when the effect is tiny.

The CI makes clear: yes, there's a difference, but it's small (0.05) and we're very precise about it (tight CI).

**Key difference**: With large n, small effects become "significant" but may not be meaningful. CI helps distinguish "statistically significant" from "practically significant."

---

## What Stakeholders Actually Want

### 1. "Did the model change?" → **Permutation test**
Binary decision, clear threshold.

### 2. "How much did it change?" → **Bootstrap CI**
Effect size with uncertainty bounds.

### 3. "Should we be concerned?" → **Both**
- Permutation test: Is it real?
- Bootstrap CI: Is it big enough to matter?

---

## Communication Templates

### Permutation Test Report:

```
Model Comparison: fp32 vs int8

Trace Distance: 0.2341
P-value: 0.003

Interpretation:
The distributions are statistically significantly different (p < 0.05).
There is less than a 0.3% probability this difference occurred by chance.

Conclusion: The int8 quantization has changed the model's output distribution.
```

**Tone**: Definitive, binary, decision-oriented

---

### Bootstrap CI Report:

```
Model Comparison: fp32 vs int8

Trace Distance: 0.2341
95% Confidence Interval: [0.1875, 0.2894]

Interpretation:
The best estimate of the difference is 0.23.
We are 95% confident the true difference is between 0.19 and 0.29.
This represents a moderate level of distinguishability.

Conclusion: The int8 quantization has changed the model's output distribution,
with a moderate effect size.
```

**Tone**: Nuanced, uncertainty-aware, magnitude-oriented

---

### Combined Report (Best of Both):

```
Model Comparison: fp32 vs int8

Trace Distance: 0.2341
95% Confidence Interval: [0.1875, 0.2894]
P-value: 0.003

Interpretation:
- Statistical significance: Yes, highly significant (p = 0.003)
- Effect magnitude: Moderate (0.23 on a 0-1 scale)
- Precision: We're 95% confident the true effect is between 0.19 and 0.29

Conclusion: The int8 quantization has significantly changed the model's output
distribution, with a moderate and precisely estimated effect.
```

**Tone**: Comprehensive, balances decision and estimation

---

## Which Should You Choose?

### Choose Permutation Test (Option 1) if:

- ✓ Stakeholders want binary decisions ("Did it change?" yes/no)
- ✓ Standard hypothesis testing framework is expected
- ✓ Audience is familiar with p-values
- ✓ Primary concern is "Is this real or just noise?"
- ✓ You're submitting to a journal that expects p-values

**Best for**: Conforming to standard statistical inference practices, clear decision-making

### Choose Bootstrap CI (Option 2) if:

- ✓ Stakeholders want to know "How much uncertainty is there?"
- ✓ Effect size precision matters
- ✓ You want to avoid binary thinking
- ✓ Audience is sophisticated (understands CIs)
- ✓ You're exploring, not just confirming

**Best for**: Estimation-focused research, nuanced understanding, avoiding p-value misinterpretation

### Choose Both if:

- ✓ You want comprehensive reporting
- ✓ Different stakeholders have different needs
- ✓ You're writing a research paper (show robustness)
- ✓ The computational cost is acceptable

**Best for**: Complete statistical reporting, satisfying multiple audiences

---

## The Fundamental Difference

### Permutation Test Philosophy:
**"Is there an effect?"**
- Hypothesis testing
- Binary decision
- Focus on evidence against null

### Bootstrap CI Philosophy:
**"What is the effect?"**
- Estimation
- Continuous understanding
- Focus on magnitude and precision

---

## Recommended Approach: Implement Both

**Implement both, report both**.

```python
# Compute both
pvalue, td_obs = trace_distance_test(emb_a, emb_b, b=1000)
td_point, (ci_lower, ci_upper) = trace_distance_with_ci(emb_a, emb_b, n_boot=1000)

# Report both
print(f"Trace distance: {td_obs:.4f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"P-value: {pvalue:.4f}")

# Interpret
if pvalue < 0.05:
    if td_obs < 0.1:
        print("Statistically significant but small effect")
    elif td_obs < 0.3:
        print("Statistically significant with moderate effect")
    else:
        print("Statistically significant with large effect")
    
    ci_width = ci_upper - ci_lower
    if ci_width < 0.05:
        print("Precisely estimated (narrow CI)")
    else:
        print("Some uncertainty in effect size (wide CI)")
else:
    print("No significant difference detected")
```

This gives stakeholders:
- **Binary decision** (p-value)
- **Effect magnitude** (trace distance)
- **Uncertainty** (CI width)
- **Precision** (CI bounds)

**Best of both worlds**.

---

## Summary

From a stakeholder perspective:

**Permutation testing** answers: **"Is it real?"**
- Clear yes/no decision
- Familiar p-value framework
- Easier to communicate to non-technical audiences
- Misses uncertainty in effect magnitude

**Bootstrap CI** answers: **"How big is it, really?"**
- Quantifies uncertainty in the effect size
- More nuanced understanding
- Better for technical audiences
- Requires explaining confidence intervals

**Both together** provide complete picture:
- Is the effect real? (p-value)
- How big is it? (point estimate)
- How certain are we? (CI width)
- What's the plausible range? (CI bounds)

**Recommendation**: Implement both methods and report all statistics for comprehensive inference.
