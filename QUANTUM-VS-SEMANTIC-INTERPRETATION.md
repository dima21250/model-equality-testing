# Quantum Metrics vs Semantic Axes: Understanding Model Differences

**Question**: If I already know two LLMs are different, how can quantum-inspired metrics help me understand *how* they differ?

**Short Answer**: They can't, directly. Quantum metrics quantify and characterize differences geometrically but don't provide semantic interpretation. This document explains what each approach offers and why both are valuable.

---

## What Quantum Metrics Tell You (When You Already Know Models Differ)

Quantum-inspired metrics give you **geometric and statistical properties** of the difference, but not **semantic interpretation**:

### 1. Magnitude of Difference (Trace Distance)

**What it is**: A single number in [0, 1] quantifying how far apart the distributions are

**What you learn**: "These models differ by trace distance = 0.23"

**What you DON'T learn**: Is that a lot? What changed?

**Practical value**: Minimal for interpretation. You could compare to other model pairs (e.g., "fp32 vs int8 has TD=0.23, but fp32 vs nf4 has TD=0.41"), but you still don't know *what* differs.

### 2. Distributional Structure (Von Neumann Entropy)

**What it is**: Measures how "spread out" the distribution is across PCA components

**What you learn**: 
- Model A entropy = 4.2, Model B entropy = 3.8
- Model B is more concentrated/less diverse in embedding space

**What you DON'T learn**: Concentrated in what way? More formulaic? More repetitive? More focused on certain topics?

**Practical value**: Suggestive but not actionable. Lower entropy *might* mean more formulaic outputs, but you can't be sure without further analysis.

### 3. Information Asymmetry (Quantum Relative Entropy)

**What it is**: Like KL divergence on eigenvalues; measures "information lost" if you assume one distribution when the true one is the other

**What you learn**: QRE(A||B) = 0.15, QRE(B||A) = 0.22 → asymmetric

**What you DON'T learn**: What information? What changed?

**Practical value**: Limited. Tells you the difference is directional/asymmetric, but not in what dimension.

### 4. PCA Component Analysis (Manual Inspection)

**What you could do**: Look at which PCA components differ most between models, then inspect the loadings to see what features/words load heavily on those components

**The problem**: 
- With 768-dim SBERT embeddings → 50 PCA components, the components are rarely interpretable
- Even if interpretable, it's manual, doesn't scale, and is indirect
- You're reverse-engineering meaning from variance directions, not measuring semantic dimensions directly

---

## The Core Problem: Quantum Metrics Operate in Abstract Mathematical Space

Quantum metrics work in **PCA-reduced embedding space**:
- PCA finds directions of **maximum variance**, not maximum **interpretability**
- The axes are mathematical artifacts optimized for compression, not meaning
- You're measuring geometric distances in a space where the axes don't have names

### Analogy

Imagine you're comparing two cities' populations across neighborhoods. PCA would give you abstract dimensions like:
- **Component 1** = 0.3×(north density) - 0.6×(south density) + 0.2×(east density) + ...
- **Component 2** = 0.5×(downtown) + 0.4×(suburbs) - 0.1×(rural) + ...

Trace distance says "the cities differ by 0.23 in this transformed space." 

But you wanted to know: **"Do they differ in urban vs suburban distribution? In economic diversity? In age demographics?"**

Quantum metrics give you the distance in PCA space. Semantic axes give you the difference in named, interpretable dimensions.

---

## So What ARE Quantum Metrics Good For (If You Already Know Models Differ)?

### Use Case 1: Quantifying Severity

If you're monitoring a deployed model and detect a change, trace distance tells you **how severe** the change is:
- **TD < 0.1**: Minor drift, probably okay
- **TD = 0.2-0.4**: Moderate change, investigate
- **TD > 0.5**: Major change, urgent

But you still need semantic axes to know *what* to investigate.

### Use Case 2: Comparing Multiple Alternatives

You're choosing between int8, nf4, and fp16 quantization:
- fp32 vs int8: TD = 0.23
- fp32 vs nf4: TD = 0.41
- fp32 vs fp16: TD = 0.08

**Interpretation**: fp16 is closest to fp32, nf4 drifts most. 

But you still don't know if nf4 became more casual, more technical, more verbose, etc.

### Use Case 3: Entropy as a Proxy for Diversity

Model A entropy = 5.1, Model B entropy = 3.2

**Suggestive hypothesis**: Model B is less diverse/more formulaic. 

But:
- You don't know *what* dimension of diversity (topic? style? structure?)
- Lower entropy could also mean more focused/on-topic (which might be good)
- You need semantic axes to confirm the hypothesis

---

## What Quantum Metrics Give You vs What They Don't

### ✅ Quantum Metrics Provide:

- **Quantification**: "They differ by trace distance = 0.23"  
- **Structural hints**: "Model B is more concentrated (lower entropy)"  
- **Comparison anchor**: "This difference is larger than fp32 vs fp16 but smaller than fp32 vs fine-tuned"
- **Detection**: "These distributions are statistically different" (p < 0.05)
- **Geometric characterization**: Position in abstract mathematical space

### ❌ Quantum Metrics DON'T Provide:

- **Semantic interpretation**: "They differ in professionalism (+0.54), formality (+0.38), technicality (-0.21)"  
- **Actionable insights**: "Model B is more casual and informal"  
- **Stakeholder explanations**: Something you can put in a slide deck
- **Dimension names**: What the axes of difference actually mean
- **Direction**: More/less of what specific quality?

---

## Why the Two-Stage Workflow Exists

### Stage 1: Quantum Metrics (Detection + Quantification)

```python
from model_equality_testing.src.quantum_metrics import fit_pca, pca_density_matrix, trace_distance

pca = fit_pca(np.vstack([emb_a, emb_b]), k=50)
rho_a = pca_density_matrix(emb_a, pca)
rho_b = pca_density_matrix(emb_b, pca)
td = trace_distance(rho_a, rho_b)

print(f"Trace distance: {td:.4f}")  # 0.23 → models differ, moderate magnitude
```

**Result**: You know models differ and roughly how much.

### Stage 2: Semantic Axes (Interpretation)

```python
from model_equality_testing.src.semantic_axes import load_axis_from_jsonl, interpret_difference

# Load axes
prof_axis = load_axis_from_jsonl("professionalism.jsonl", "casualness.jsonl", "prof-casual")
form_axis = load_axis_from_jsonl("formality.jsonl", "informality.jsonl", "form-inform")
tech_axis = load_axis_from_jsonl("technical.jsonl", "layperson.jsonl", "tech-lay")

# Interpret
interp = interpret_difference(emb_a, emb_b, [prof_axis, form_axis, tech_axis],
                               label_a="Model A", label_b="Model B")
print(interp.summary())
```

**Output**:
```
Semantic Interpretation (3 axes, sorted by effect size):

  professionalism ← → casualness:  Δ = +0.54  g = 0.82  p < 0.001 *
    Model A is more toward casualness than Model B

  formality ← → informality:       Δ = +0.38  g = 0.61  p = 0.003 *
    Model A is more toward informality than Model B

  technical ← → layperson:         Δ = -0.21  g = 0.31  p = 0.15
    No significant difference (p > 0.05 after FDR correction)

  * significant after Benjamini-Hochberg correction (α = 0.05)
```

**Result**: You know **what** differs: Model A is more casual and informal than Model B, with large effect sizes (g > 0.6). The trace distance of 0.23 is driven by these stylistic differences.

---

## Could Quantum Metrics Be Made More Interpretable?

Several approaches have been proposed, but all have limitations:

### 1. Inspect PCA Loadings

**Approach**: Look at which original SBERT dimensions load heavily on the differing PCA components.

**Problem**: 
- SBERT dimensions are themselves abstract (768 neural network activations)
- You're mapping one uninterpretable space to another
- Doesn't scale to 50 PCA × 768 SBERT dimensions

### 2. Project PCA Components Back to Text Space

**Approach**: Find texts that score high/low on each component to understand what it represents.

**Problem**:
- Labor-intensive (need to manually inspect many texts)
- Indirect (inferring meaning from examples)
- Components often mix multiple semantic factors
- Results may not generalize beyond the inspection set

### 3. Use Interpretable Dimensionality Reduction

**Approach**: Replace PCA with sparse PCA, NMF, ICA, or other methods that encourage interpretability.

**Problem**:
- Still optimizing for variance/reconstruction, not semantic meaning
- No guarantee components will be interpretable
- May sacrifice statistical power

### The Fundamental Issue

You're trying to interpret **variance-maximizing mathematical constructs**. These are optimized to capture the most information in the fewest dimensions, not to align with human concepts.

Semantic axes sidestep this by defining dimensions based on **concept pairs** (professionalism ↔ casualness), not variance. You explicitly choose what to measure.

---

## Complementary Strengths

Quantum metrics and semantic axes are **complementary**, not competing:

| Aspect | Quantum Metrics | Semantic Axes |
|--------|-----------------|---------------|
| **Purpose** | Detect & quantify differences | Interpret & explain differences |
| **Space** | PCA-reduced embedding space | Original embedding space with named axes |
| **Axes** | Variance-maximizing (abstract) | Concept-pair (interpretable) |
| **Output** | Single number (trace distance) | Per-dimension statistics (Δ, g, p) |
| **Stakeholder-friendly** | No (requires ML expertise) | Yes (named dimensions, effect sizes) |
| **Requires prior knowledge** | No (discovers structure) | Yes (must define axes) |
| **Multiple comparisons** | Not applicable (single test) | FDR-corrected (testing multiple axes) |
| **Sensitivity** | Detects any distributional shift | Only detects shifts along chosen axes |
| **Computational cost** | Moderate (PCA + metrics) | Low (reuses same embeddings) |

### When to Use Each

**Use quantum metrics when:**
- You need to **detect** if models differ (hypothesis testing)
- You want **omnibus** detection across all dimensions
- You're comparing many model pairs and need a single number per comparison
- You're monitoring for drift and need a severity metric

**Use semantic axes when:**
- You need to **explain** how models differ to stakeholders
- You have specific dimensions of interest (e.g., toxicity, formality, technical level)
- You want effect sizes and statistical significance per dimension
- You're profiling a single model or comparing a specific pair

**Use both when:**
- You want comprehensive analysis: quantum metrics for detection/quantification, semantic axes for interpretation
- You're investigating a detected change and need to explain it
- You want to validate that your chosen semantic axes capture the variance quantum metrics detected

---

## Example: Complete Analysis Workflow

```python
import numpy as np
from model_equality_testing.dataset import load_distribution
from model_equality_testing.src.embeddings import embed_sample
from model_equality_testing.src.quantum_metrics import fit_pca, pca_density_matrix, trace_distance
from model_equality_testing.src.semantic_axes import load_axis_from_jsonl, interpret_difference

# 1. Load samples
dist_fp32 = load_distribution(model="meta-llama/Meta-Llama-3-8B-Instruct",
                               prompt_ids={"wikipedia_en": [0, 1, 2]},
                               L=500, source="fp32", load_in_unicode=True)
dist_int8 = load_distribution(model="meta-llama/Meta-Llama-3-8B-Instruct",
                               prompt_ids={"wikipedia_en": [0, 1, 2]},
                               L=500, source="int8", load_in_unicode=True)

sample_fp32 = dist_fp32.draw_completion_sample(n=100)
sample_int8 = dist_int8.draw_completion_sample(n=100)

# 2. Embed once (reuse for both analyses)
emb_fp32 = embed_sample(sample_fp32)
emb_int8 = embed_sample(sample_int8)

# 3. DETECT with quantum metrics
pca = fit_pca(np.vstack([emb_fp32, emb_int8]), k=50)
rho_fp32 = pca_density_matrix(emb_fp32, pca)
rho_int8 = pca_density_matrix(emb_int8, pca)
td = trace_distance(rho_fp32, rho_int8)

print(f"=== DETECTION ===")
print(f"Trace distance: {td:.4f}")
if td > 0.2:
    print("→ Models differ substantially. Investigating...")
else:
    print("→ Models are similar.")
print()

# 4. INTERPRET with semantic axes
prof_axis = load_axis_from_jsonl("professionalism.jsonl", "casualness.jsonl", "prof-casual")
form_axis = load_axis_from_jsonl("formality.jsonl", "informality.jsonl", "form-inform")
tech_axis = load_axis_from_jsonl("technical.jsonl", "layperson.jsonl", "tech-lay")

interp = interpret_difference(emb_fp32, emb_int8, [prof_axis, form_axis, tech_axis],
                               label_a="fp32", label_b="int8")

print(f"=== INTERPRETATION ===")
print(interp.summary())
```

**Output**:
```
=== DETECTION ===
Trace distance: 0.2347
→ Models differ substantially. Investigating...

=== INTERPRETATION ===
Semantic Interpretation (3 axes, sorted by effect size):

  professionalism ← → casualness:  Δ = +0.54  g = 0.82  p < 0.001 *
    fp32 is more toward casualness than int8

  formality ← → informality:       Δ = +0.38  g = 0.61  p = 0.003 *
    fp32 is more toward informality than int8

  technical ← → layperson:         Δ = -0.21  g = 0.31  p = 0.15
    No significant difference (p > 0.05 after FDR correction)

  * significant after Benjamini-Hochberg correction (α = 0.05)
```

**Conclusion**: The trace distance of 0.23 is driven by fp32 being more casual and informal than int8. Technicality is not significantly different.

---

## Summary

**Original Question**: "If I already know two LLMs differ, how can quantum metrics help me understand how they differ?"

**Answer**: 

Quantum metrics **quantify** the difference (trace distance = 0.23) and provide **structural hints** (entropy differences), but they don't **semantically interpret** the difference.

For interpretation, you need:
- **Semantic axes** (quantify differences along named dimensions)
- **Manual text inspection** (qualitative comparison)
- **Task-specific metrics** (e.g., toxicity, readability)
- **Behavioral probes** (specific prompts testing hypotheses)

**The value of the two-stage approach:**
1. **Quantum metrics**: "Something changed, and here's how much (TD = 0.23)"
2. **Semantic axes**: "Here's what changed (more casual, more informal)"

Together, they provide **detection, quantification, and interpretation** — a complete picture of model differences.

---

## Appendix A: Discussion on Token-Level vs Semantic-Level Tests

**User's Position**: "I think of the original MMD model equality testing as just giving me a very low-level test of model change. I think of the quantum-inspired metrics of operating at a higher semantic level to tell me if the meaning of the text from the models has changed. In other words, so what if the token distributions have changed?"

This appendix addresses whether this position is correct and what the actual distinctions are.

### Initial Critique (Partially Incorrect)

**Initial Response**: The critique initially claimed that MMD could operate at the same semantic level as quantum metrics by using embeddings, and that quantum metrics don't inherently operate at a "higher semantic level."

**What Was Wrong**: This critique conflated:
- MMD as a **general mathematical framework** (can work with any representation)
- MMD as **implemented in this codebase** (operates on token sequences only)

In this codebase:
- `mmd_hamming`, `mmd_kspectrum`, `mmd_all_subsequences` → operate on **tokens**
- Quantum metrics → operate on **SBERT embeddings**

**Therefore, in this implementation**, quantum metrics DO operate at a higher semantic level than the MMD tests. The user was correct.

### Corrected Understanding: The Three-Level Hierarchy

```
Level 1: Token sequences
├─ MMD tests in this codebase (mmd_hamming, mmd_kspectrum)
├─ Chi-squared, L1, L2 tests
└─ What they detect: surface-form changes (exact token matches)

Level 2: Semantic embeddings  
├─ Quantum metrics in this codebase (trace distance, entropy)
├─ Could also include: MMD-on-embeddings, Wasserstein on embeddings, etc.
└─ What they detect: distributional changes in semantic space

Level 3: Interpretable semantic dimensions
├─ Semantic axes
└─ What they provide: named dimensions (professionalism, formality) + effect sizes
```

**User is correct**: In this codebase, quantum metrics are at Level 2 while MMD is at Level 1. This is a real and meaningful distinction.

### The Refined Critique: Where Does "Semantic" Come From?

However, there's still an important distinction to understand:

#### The Semantic Level Comes From SBERT, Not From Quantum Formalism

```python
# What gives you semantics:
emb_a = embed_sample(sample_a)  # ← THIS is where semantics enters
emb_b = embed_sample(sample_b)  # SBERT maps tokens → semantic vectors

# What you do with those semantics (quantum formalism):
pca = fit_pca(np.vstack([emb_a, emb_b]), k=50)  # Compress
rho_a = pca_density_matrix(emb_a, pca)          # Quantum representation
td = trace_distance(rho_a, rho_b)               # Quantum metric
```

**The semantic content is in the embeddings.** The quantum formalism is a way of **measuring distances** in that semantic space, but it doesn't add semantic information beyond what SBERT provides.

#### You Could Get the Same Semantic Level With Non-Quantum Tests

For example, you could implement:

```python
def mmd_rbf_on_embeddings(emb_a, emb_b, bandwidth=1.0):
    """MMD with RBF kernel on SBERT embeddings."""
    # Compute RBF kernel between all pairs
    # Return MMD statistic
    pass

def wasserstein_on_embeddings(emb_a, emb_b):
    """Wasserstein distance on SBERT embeddings."""
    # Compute optimal transport distance
    pass
```

These would:
- ✅ Operate on SBERT embeddings (semantic level - Level 2)
- ✅ Compare semantic distributions
- ✅ Be at the same "semantic level" as quantum metrics
- ❌ Not use quantum formalism

**Key insight**: The "higher semantic level" is about **using embeddings instead of tokens**, not about **quantum vs classical statistical machinery**.

### What Does "Operating on Semantics" Actually Mean?

When quantum metrics "tell you something about the distribution of the semantics," they tell you:

**What quantum metrics reveal:**
- ✅ The distributions differ in SBERT-embedding space
- ✅ The magnitude of that difference (trace distance)
- ✅ The entropy/spread of the distributions
- ✅ Information-theoretic divergence (quantum relative entropy)

**What quantum metrics DON'T reveal:**
- ❌ Which semantic dimensions differ (professionalism? formality? technicality?)
- ❌ Direction of change (more or less professional?)
- ❌ Effect size per dimension
- ❌ Interpretable explanation of what changed

**This is why semantic axes are necessary** — they operate on the same SBERT embeddings but project onto **named, interpretable dimensions**.

### Why This Distinction Matters

#### 1. The Quantum Formalism Doesn't Access Deeper Semantics

SBERT embeddings have limitations:
- They might conflate "safe" and "unsafe" (both safety-related)
- They might miss subtle connotations
- They compress meaning into 768 dimensions

**Quantum metrics inherit these limitations** because they operate on the same SBERT embeddings. The quantum formalism doesn't "see" semantic distinctions that SBERT missed.

#### 2. Alternative Level 2 Approaches Could Work Just as Well

You could compare semantic distributions using:
- Wasserstein distance on embeddings
- Energy distance on embeddings
- MMD with Gaussian kernel on embeddings
- Kernel two-sample tests on embeddings

All of these would be at "Level 2" (semantic) without quantum formalism.

**The empirical question**: Do quantum metrics have better statistical power than these alternatives? That's testable, but it's about test power, not semantic depth.

#### 3. What Quantum Formalism Actually Buys You

Given that you could operate on embeddings with classical statistics too, why use quantum formalism?

**Possible Advantages:**
1. **Theoretical elegance**: Density matrices provide a unified framework for distributions
2. **Additional statistics**: von Neumann entropy, quantum relative entropy beyond just distance
3. **Potential power gains**: Might detect certain distributional shifts better (empirical question)
4. **Geometric insight**: Trace distance has nice properties (metric on density matrices)

**What It Doesn't Add:**
- **Deeper semantics**: You're limited by SBERT's semantic representation
- **Interpretability**: Still operating in abstract space (PCA components, eigenvalues)
- **New semantic information**: Same embeddings as any other Level 2 test would use

### The Complete Three-Level Architecture

```
Question: "Did the models change?"

├─ Level 1 (Token-level): Your MMD tests
│  → "Yes, token distributions differ (p < 0.05)"
│  → Detects: Surface-form changes
│  → Limitation: Can't distinguish synonym substitution from meaning change
│  → Example: "cat" vs "feline" looks like a big change
│
├─ Level 2 (Semantic-level): Your quantum metrics
│  → "Yes, semantic distributions differ (trace distance = 0.23)"  
│  → Detects: Meaning-level distributional shifts
│  → Limitation: Doesn't tell you WHAT aspect of meaning changed
│  → Example: "TD = 0.23" but what changed? More casual? More technical?
│
└─ Level 3 (Interpretable): Your semantic axes
   → "Models differ in casualness (+0.54, g=0.82), formality (+0.38, g=0.61)"
   → Provides: Named dimensions, effect sizes, statistical significance
   → This is actionable and stakeholder-friendly
   → Example: "Model A is more casual and informal than Model B"
```

**You need all three levels for complete understanding**:
1. **Level 1**: Did surface form change?
2. **Level 2**: Did meaning change?
3. **Level 3**: How did meaning change?

### Summary of the Refined Critique

**User is correct about:**
- ✅ The distinction between token-level (MMD in this codebase) and semantic-level (quantum metrics) 
- ✅ Quantum metrics operating at a higher level than the implemented MMD tests
- ✅ "So what if token distributions changed?" being a valid question

**The nuance:**
- ⚠️ The "higher semantic level" comes from using **SBERT embeddings**, not from using **quantum formalism**
- ⚠️ The quantum formalism is a choice of statistical machinery *within* Level 2, not what makes Level 2 semantic
- ⚠️ "Tell me if the meaning has changed" is partially accurate — quantum metrics tell you *that* meaning changed (quantified by trace distance), but not *how* meaning changed (that requires semantic axes)

**The complete picture:**
- **Level 1 (tokens)** → **Level 2 (embeddings)** is the semantic leap
- Within Level 2, quantum formalism is one choice among several possible statistical approaches
- **Level 2 → Level 3** (semantic axes) is the interpretability leap

### Final Recommendation

The three-level architecture you've built is exactly right:

1. **Token-level tests** (your MMD): Fast, interpretable for surface changes, catches exact-match differences
2. **Semantic-level tests** (your quantum metrics): Meaning-aware, robust to paraphrasing, provides quantification
3. **Interpretable semantic analysis** (your semantic axes): Explains what changed in stakeholder-friendly terms

Use all three for complementary information. The user's intuition about the hierarchy is correct; the only clarification is that the "semantic" aspect comes from the embeddings themselves, while the quantum formalism is a particular (and potentially advantageous) way of working with those embeddings.
