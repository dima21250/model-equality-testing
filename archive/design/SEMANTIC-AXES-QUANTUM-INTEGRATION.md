# Integrating Semantic Axes with Quantum-Inspired Metrics

This document explains how the **semantic axes framework** (from `~/Code/semantic-pole`) integrates with **quantum-inspired density matrices** for interpretable LLM behavioral profiling.

**Key insight**: Semantic axes provide interpretable dimensions; quantum metrics measure distributional structure in those dimensions.

---

## Overview

The two frameworks complement each other:

- **Semantic axes** (from `semantic-pole`): Define explicit, interpretable dimensions via pole corpora → SBERT → projection
- **Quantum metrics** (from `model-equality-testing`): Measure distributional structure via density matrices, effective rank, trace distance

**The integration**: Apply quantum metrics to psychometric vectors derived from semantic axis projections, yielding interpretable distributional analysis.

---

## Three Integration Options

### Option 1: Quantum Metrics for Detection, Axes for Interpretation

**Pipeline:**
1. Build density matrices ρ_A and ρ_B in full SBERT space (768-dim)
2. Use quantum metrics to **detect** differences (trace distance, effective rank)
3. When you find a significant difference, use semantic axes to **interpret** what changed

**Example:**
```python
# Detection in full SBERT space
trace_dist = trace_distance(rho_A, rho_B)  # 0.23 — models differ
effective_rank_A = 12.3  # Model A uses ~12 semantic modes
effective_rank_B = 8.1   # Model B uses ~8 modes

# Interpretation (post-hoc)
delta_prof = project(model_A, v_prof) - project(model_B, v_prof)  # +0.54
delta_form = project(model_A, v_form) - project(model_B, v_form)  # +0.38
delta_tech = project(model_A, v_tech) - project(model_B, v_tech)  # -0.21

# Stakeholder message
"Models differ (trace distance 0.23). The difference is primarily in 
professionalism (+0.54) and formality (+0.38), not technical depth (-0.21)."
```

**Pros:**
- Keeps quantum metrics in validated full SBERT space
- Sensitive detection (uses all 768 dimensions)

**Cons:**
- Interpretation is post-hoc, not integrated into the metric
- Quantum metrics operate on uninterpretable SBERT latent space
- Two-step process (detect, then interpret separately)

---

### Option 2: Psychometric Density Matrices ⭐ RECOMMENDED

**Pipeline:**
1. Define K semantic axes (professionalism, formality, creativity, technical depth, etc.)
2. Project each text onto all K axes → K-dimensional **psychometric vector**
3. Build density matrix in K-dimensional **interpretable psychometric space**
4. Apply quantum metrics to this space

**Example:**
```python
# Step 1: Define semantic axes
axes = {
    'professionalism': load_axis('prof-casual'),
    'formality': load_axis('formal-informal'), 
    'technical_depth': load_axis('technical-layperson'),
    'creativity': load_axis('creative-literal'),
    'conciseness': load_axis('concise-verbose')
}  # K = 5 dimensions

# Step 2: Project each text onto all axes → 5D psychometric vector
def psychometric_embed(text):
    sbert_emb = encode(text)  # 768-dim SBERT embedding
    return np.array([project(sbert_emb, axis_vec) for axis_vec in axes.values()])  # 5-dim

psychometric_vectors = [psychometric_embed(text) for text in model_outputs]
# Each vector: [prof_score, form_score, tech_score, creat_score, concise_score]
# Example: [+0.72, +0.65, -0.15, -0.42, +0.33]

# Step 3: Build K×K density matrix in psychometric space
# Normalize each vector to unit length (quantum state)
normalized_vectors = [v / np.linalg.norm(v) for v in psychometric_vectors]
rho_psychometric = (1/N) * sum(np.outer(psi, psi) for psi in normalized_vectors)
# Shape: (5, 5) — interpretable dimensions!

# Step 4: Quantum metrics on psychometric space
effective_rank = np.exp(-np.sum(eigenvalues * np.log(eigenvalues)))
# Interpretation: "Model uses 3.2 distinct psychometric modes out of 5 dimensions"

trace_dist = trace_distance(rho_A_psych, rho_B_psych)
# Interpretation: "Models differ by 0.18 in psychometric space"

quantum_entropy = -np.sum(eigenvalues * np.log(eigenvalues))
# Interpretation: "Psychometric diversity = 1.16 nats"

# Step 5: Interpret eigenvectors (psychometric modes)
eigenvalues, eigenvectors = np.linalg.eigh(rho_psychometric)
mode_1 = dict(zip(axes.keys(), eigenvectors[:, -1]))
# Example: {
#   'professionalism': 0.82, 
#   'formality': 0.51, 
#   'technical_depth': 0.12, 
#   'creativity': -0.15, 
#   'conciseness': 0.08
# }
# Interpretation: "Dominant mode is highly professional + formal, low creativity"
```

**Stakeholder Output:**
```
Model A Profile:
- Effective rank: 3.2 modes (out of 5 psychometric dimensions)
- Quantum entropy: 1.16 nats
- Dominant mode (52% of outputs): professional-formal (prof: 0.82, form: 0.51, creativity: -0.15)
- Secondary mode (28% of outputs): technical-concise (tech: 0.78, concise: 0.61, prof: 0.21)
- Minor mode (14% of outputs): creative-verbose (creat: 0.71, concise: -0.68, prof: -0.15)

Model B Profile:
- Effective rank: 2.1 modes
- Quantum entropy: 0.74 nats
- Dominant mode (68% of outputs): casual-creative (prof: -0.62, creat: 0.75, form: -0.41)
- Secondary mode (22% of outputs): technical-concise (tech: 0.80, concise: 0.58)

Trace Distance: 0.18 (psychometric space)
Interpretation: Models differ significantly. Model A is more professional, formal, and diverse 
(3.2 modes vs 2.1). Model B is more casual and creative but less diverse.
```

**Pros:**
- **Interpretable quantum metrics**: Effective rank = "How many psychometric modes?" (modes have named dimensions)
- **Interpretable eigenvectors**: Weighted combinations of semantic axes ("professional-formal mode" vs "creative-casual mode")
- **Stakeholder-aligned**: Axes designed for dimensions stakeholders care about
- **Lower dimensional**: K=5-10 axes vs 768-dim SBERT or 50-dim PCA
- **Bridges both projects**: `semantic-pole` generates axes, `model-equality-testing` provides quantum metrics
- **Solves PCA interpretability problem**: "What is PC3?" → "This is the technical-concise mode"

**Cons:**
- **Information loss**: Axes are projections, not full embedding space (may miss orthogonal structure)
- **Axis design matters**: Bad axis choices → miss important behavioral dimensions
- **Assumes axes span the space**: If model behavior varies on unmodeled dimensions, metrics will miss it

**Mitigation:**
- Start with 5-10 axes covering broad behavioral dimensions
- Validate: compare psychometric space trace distance to full SBERT space trace distance
- If they diverge, consider adding axes to cover the gap

---

### Option 3: Hybrid — Both Spaces

**Pipeline:**
1. Compute quantum metrics in **both** full SBERT space and psychometric axis space
2. Compare: Do they tell the same story?

**Use cases:**
- **Full SBERT space**: Sensitive detection (did anything change?)
- **Psychometric space**: Interpretable profiling (what changed?)

**Example:**
```python
# Detection in full 768-dim SBERT space
rho_A_full = density_matrix(sbert_embeddings_A)  # 768×768 (or PCA reduced)
rho_B_full = density_matrix(sbert_embeddings_B)
trace_dist_full = trace_distance(rho_A_full, rho_B_full)  # 0.23

# Interpretation in 5-dim psychometric space
rho_A_psych = psychometric_density_matrix(model_A_outputs, axes)  # 5×5
rho_B_psych = psychometric_density_matrix(model_B_outputs, axes)  # 5×5
trace_dist_psych = trace_distance(rho_A_psych, rho_B_psych)  # 0.18

# Compare
if trace_dist_full > threshold and trace_dist_psych < threshold:
    # Models differ, but NOT along stakeholder-relevant dimensions
    message = (
        "Models differ (trace dist 0.23 in full space), but the difference is not "
        "captured by professionalism/formality/creativity axes. Consider defining "
        "additional axes to cover the divergence."
    )
elif trace_dist_full > threshold and trace_dist_psych > threshold:
    # Models differ in interpretable ways
    message = (
        "Models differ (trace dist 0.23 full space, 0.18 psychometric space). "
        "The difference is primarily in professionalism and formality modes."
    )
elif trace_dist_full < threshold:
    # Models are the same
    message = "Models are statistically equivalent in both full and psychometric space."
```

**Pros:**
- **Comprehensive**: Catches both interpretable and non-interpretable differences
- **Diagnostic**: Gap between full and psychometric trace distance indicates unmeasured dimensions
- **Conservative**: Full space for sensitive detection, psychometric for interpretation

**Cons:**
- **More computation**: Two density matrix constructions, two metric calculations
- **Complexity**: Two numbers to report and reconcile

---

## Concrete Example: Professionalism Analysis

### Setup

```python
from semantic_pole import load_axis
from model_equality_testing.src.profiling import psychometric_density_matrix
import numpy as np

# Define 5 semantic axes using semantic-pole framework
axes = {
    'professionalism': load_axis('professionalism-casualness'),
    'formality': load_axis('formality-informality'), 
    'technical_depth': load_axis('technical-layperson'),
    'creativity': load_axis('creative-literal'),
    'conciseness': load_axis('concise-verbose')
}

# Collect model outputs
model_A_outputs = [...]  # 1000 completions from Model A
model_B_outputs = [...]  # 1000 completions from Model B
```

### Psychometric Embedding

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer('all-mpnet-base-v2')

def psychometric_embed(text, axes):
    """Project text onto semantic axes to get psychometric vector."""
    sbert_emb = encoder.encode(text)  # 768-dim
    scores = []
    for axis_name, axis_vec in axes.items():
        # Project onto axis (assume axis_vec is unit vector)
        score = np.dot(sbert_emb, axis_vec)
        scores.append(score)
    return np.array(scores)  # K-dim psychometric vector

# Embed all outputs
psych_A = np.array([psychometric_embed(text, axes) for text in model_A_outputs])
psych_B = np.array([psychometric_embed(text, axes) for text in model_B_outputs])

# Example psychometric vector for one output
print(psych_A[0])
# [+0.72, +0.65, -0.15, -0.42, +0.33]
# Interpretation: highly professional, highly formal, not technical, 
#                 not creative, moderately concise
```

### Build Density Matrices

```python
def psychometric_density_matrix(psychometric_vectors):
    """Build density matrix in psychometric space."""
    # Normalize each vector to unit length (quantum state requirement)
    normalized = psychometric_vectors / np.linalg.norm(
        psychometric_vectors, axis=1, keepdims=True
    )
    
    # Density matrix: ρ = (1/N) Σ |ψ_i⟩⟨ψ_i|
    N = len(normalized)
    rho = sum(np.outer(psi, psi) for psi in normalized) / N
    return rho

rho_A = psychometric_density_matrix(psych_A)  # 5×5
rho_B = psychometric_density_matrix(psych_B)  # 5×5
```

### Quantum Metrics

```python
def effective_rank(rho):
    """Compute effective rank: exp(von Neumann entropy)."""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # filter numerical zeros
    entropy = -np.sum(eigenvalues * np.log(eigenvalues))
    return np.exp(entropy)

def trace_distance(rho1, rho2):
    """Compute trace distance: (1/2) * trace(|rho1 - rho2|)."""
    delta = rho1 - rho2
    singular_values = np.linalg.svd(delta, compute_uv=False)
    return 0.5 * np.sum(np.abs(singular_values))

# Compute metrics
eff_rank_A = effective_rank(rho_A)  # 3.2 modes
eff_rank_B = effective_rank(rho_B)  # 2.1 modes
trace_dist = trace_distance(rho_A, rho_B)  # 0.18

print(f"Model A effective rank: {eff_rank_A:.2f} psychometric modes")
print(f"Model B effective rank: {eff_rank_B:.2f} psychometric modes")
print(f"Trace distance: {trace_dist:.3f}")
```

### Interpret Modes

```python
def interpret_psychometric_modes(rho, axes, top_k=3):
    """Extract and interpret dominant psychometric modes."""
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    modes = []
    for k in range(top_k):
        mode_vector = eigenvectors[:, k]
        mode_dict = dict(zip(axes.keys(), mode_vector))
        weight = eigenvalues[k]
        modes.append({
            'weight': weight,
            'components': mode_dict,
            'interpretation': interpret_mode(mode_dict)
        })
    return modes

def interpret_mode(mode_dict):
    """Generate human-readable interpretation of a mode."""
    # Find dominant components (|coefficient| > 0.3)
    dominant = {k: v for k, v in mode_dict.items() if abs(v) > 0.3}
    
    positive = [f"{k} ({v:.2f})" for k, v in dominant.items() if v > 0]
    negative = [f"not {k} ({-v:.2f})" for k, v in dominant.items() if v < 0]
    
    parts = []
    if positive:
        parts.append(" + ".join(positive))
    if negative:
        parts.append(" + ".join(negative))
    
    return " ; ".join(parts) if parts else "mixed mode"

# Interpret Model A
modes_A = interpret_psychometric_modes(rho_A, axes)
print("\nModel A Psychometric Modes:")
for i, mode in enumerate(modes_A):
    print(f"  Mode {i+1} ({mode['weight']*100:.1f}% of outputs):")
    print(f"    {mode['interpretation']}")

# Example output:
# Model A Psychometric Modes:
#   Mode 1 (52.3% of outputs):
#     professionalism (0.82) + formality (0.51) ; not creativity (0.15)
#   Mode 2 (27.8% of outputs):
#     technical_depth (0.78) + conciseness (0.61)
#   Mode 3 (13.6% of outputs):
#     creativity (0.71) ; not conciseness (0.68)
```

### Stakeholder Report

```python
def generate_stakeholder_report(rho_A, rho_B, axes):
    """Generate interpretable stakeholder report."""
    eff_rank_A = effective_rank(rho_A)
    eff_rank_B = effective_rank(rho_B)
    trace_dist = trace_distance(rho_A, rho_B)
    
    modes_A = interpret_psychometric_modes(rho_A, axes, top_k=3)
    modes_B = interpret_psychometric_modes(rho_B, axes, top_k=3)
    
    report = f"""
# Psychometric Profile Comparison

## Summary
- **Trace Distance**: {trace_dist:.3f} (models differ significantly)
- **Model A Diversity**: {eff_rank_A:.2f} psychometric modes
- **Model B Diversity**: {eff_rank_B:.2f} psychometric modes

## Model A Behavioral Modes
"""
    for i, mode in enumerate(modes_A):
        report += f"\n{i+1}. **{mode['interpretation']}** ({mode['weight']*100:.1f}% of outputs)"
    
    report += "\n\n## Model B Behavioral Modes\n"
    for i, mode in enumerate(modes_B):
        report += f"\n{i+1}. **{mode['interpretation']}** ({mode['weight']*100:.1f}% of outputs)"
    
    report += f"""

## Interpretation
Model A uses {eff_rank_A:.1f} distinct behavioral modes compared to Model B's {eff_rank_B:.1f} modes.
The {trace_dist:.3f} trace distance indicates substantial differences in psychometric profiles.

Model A is characterized by: {modes_A[0]['interpretation']}
Model B is characterized by: {modes_B[0]['interpretation']}
"""
    return report

print(generate_stakeholder_report(rho_A, rho_B, axes))
```

---

## Recommended Approach: Option 2 (Psychometric Density Matrices)

**Why this is the most compelling integration:**

### 1. Solves the PCA Interpretability Problem

**Current limitation** (from your existing work):
- PCA eigenvectors are uninterpretable: "What is PC3?"
- Effective rank counts modes, but you can't name them

**Psychometric solution**:
- Eigenvectors are weighted combinations of **named semantic axes**
- Mode 1: "professional-formal communication" (prof: 0.82, form: 0.51)
- Mode 2: "technical-concise explanation" (tech: 0.78, concise: 0.61)
- Mode 3: "creative-casual expression" (creat: 0.71, casual: 0.65)

### 2. Meaningful Effective Rank

**Current**: "Model uses 12.3 modes" → What are they?

**Psychometric**: "Model uses 3.2 psychometric modes out of 5 defined dimensions"
- Which modes? Look at eigenvectors
- What dimensions? Professionalism, formality, technical depth, creativity, conciseness (you defined them)

### 3. Stakeholder-Aligned Metrics

**Design axes for what stakeholders care about:**
- Business application → professionalism, formality, clarity, actionability
- Technical documentation → technical depth, precision, completeness, accessibility
- Creative writing → creativity, originality, style, emotional impact
- Customer support → empathy, clarity, helpfulness, professionalism

**Quantum metrics measure structure in stakeholder-relevant space**, not arbitrary latent dimensions.

### 4. Bridges Your Two Projects

**Workflow:**
1. Use `semantic-pole` to generate axis corpora with convergence testing
2. Project model outputs onto axes → psychometric vectors
3. Use `model-equality-testing` quantum metrics on psychometric space
4. Get interpretable distributional analysis

This is a **natural synthesis** of your two research directions.

### 5. Validation Path

**Built-in validation from semantic axes:**
- Pole corpora have convergence testing (split-half stability)
- Multi-LLM interleaving reduces artifacts
- Axes have explicit semantic grounding

**Additional validation for psychometric density matrices:**
- **Coverage test**: Does trace distance in psychometric space match trace distance in full SBERT space?
- **Axis sufficiency**: Do K axes capture most behavioral variation?
- **Mode stability**: Are eigenvectors stable under resampling?

---

## Implementation Roadmap

### Phase 1: Proof of Concept (1-2 days)

1. **Generate 5 semantic axes** with `semantic-pole`:
   - professionalism ↔ casualness
   - formality ↔ informality
   - technical ↔ layperson
   - creative ↔ literal
   - concise ↔ verbose

2. **Collect test samples**:
   - Model A: 500 completions
   - Model B: 500 completions (same prompts, different model/quantization)

3. **Implement psychometric embedding**:
   - Project each completion onto all 5 axes
   - Store as 5-dimensional psychometric vectors

4. **Build psychometric density matrices**:
   - Normalize vectors to unit length
   - Compute ρ = (1/N) Σ |ψ⟩⟨ψ|

5. **Compute quantum metrics**:
   - Effective rank (both models)
   - Trace distance
   - von Neumann entropy

6. **Interpret modes**:
   - Eigendecomposition
   - Label modes by dominant axis components
   - Generate stakeholder report

### Phase 2: Validation (3-5 days)

1. **Coverage test**:
   - Compare psychometric trace distance to full SBERT trace distance
   - If they diverge, investigate what dimensions are missing

2. **Axis sufficiency**:
   - Vary K (number of axes): 3, 5, 10, 15
   - Plot: effective rank vs K
   - Find elbow: where do additional axes stop adding information?

3. **Mode stability**:
   - Bootstrap resampling (1000 iterations)
   - Compute eigenvectors for each bootstrap sample
   - Check: are mode interpretations stable?

4. **Human alignment**:
   - Ask human raters to judge model outputs on the 5 dimensions
   - Compare: human ratings vs psychometric projections
   - Validate: do the axes measure what we think they measure?

### Phase 3: Application (ongoing)

1. **Model comparison**:
   - Compare fp32 vs int8 quantization in psychometric space
   - Report: "Quantization reduces professionalism by 0.15 units, increases diversity from 3.2 to 3.8 modes"

2. **API provider comparison**:
   - Compare amazon vs azure vs fireworks Llama-3-8B
   - Report: "Providers differ in psychometric space (trace distance 0.23), primarily on formality and conciseness modes"

3. **Prompt sensitivity**:
   - Measure: does psychometric profile change across prompt types?
   - Report: "Model is stable on professionalism (σ=0.08) but varies on creativity (σ=0.42) across prompts"

4. **Stakeholder dashboards**:
   - Visualize: radar plots of psychometric profiles
   - Compare: models, quantizations, providers, prompt types
   - Track: how profiles shift over time (API drift)

---

## Comparison to Prior Approaches

| Approach | Space | Dimensions | Interpretability | Validation |
|----------|-------|-----------|------------------|------------|
| **Full SBERT density matrix** | 768-dim latent | 768 or PCA-reduced (50) | Low (latent space) | High (SBERT is validated) |
| **PCA density matrix** | PCA space | 50-100 PCs | Very low ("what is PC3?") | Medium (explains variance, not meaning) |
| **HRR prompt binding** | 768-dim rotated | 768 | Low (random rotations) | None (V1-V4 missing) |
| **Psychometric density matrix** ⭐ | Semantic axis projections | 5-15 axes | **High (named dimensions)** | **High (axes + convergence)** |

**Psychometric density matrices** combine the best of both worlds:
- **Quantum distributional sophistication** (from density matrices)
- **Human-interpretable dimensions** (from semantic axes)

---

## Related Documents

- **SEMANTIC-AXES-VS-HRR-COMPARISON.md** — Why semantic axes beats HRR for psychometric profiling
- **HRR-quantum-metrics.md** — Critique of HRR prompt binding approach
- **CLAUDE.md** (`~/Code/semantic-pole`) — Corpus generation framework for semantic axes
- **profiling.py** (`model_equality_testing/src/`) — Behavioral profiling utilities

---

## Next Steps

1. **Implement psychometric embedding function** in `model_equality_testing/src/profiling.py`
2. **Generate 5 proof-of-concept axes** with `semantic-pole`
3. **Run psychometric density matrix analysis** on existing dataset (fp32 vs int8 Llama-3-8B)
4. **Generate stakeholder report** with interpretable modes
5. **Validate**: coverage test, axis sufficiency, mode stability
6. **Write paper section** on psychometric quantum profiling

This integration bridges your two projects and provides a principled, interpretable approach to LLM behavioral profiling using quantum-inspired metrics.
