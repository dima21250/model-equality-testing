# Semantic Axes vs. HRR for LLM Analysis: A Comparative Assessment

This document compares two approaches to analyzing LLM behavior in structured semantic space: the **semantic axes** approach (from `~/Code/semantic-pole`) and the **HRR-based prompt binding** approach (explored in `HRR-quantum-metrics.md`).

---

## Executive Summary

**Semantic axes** is the superior approach for psychometric profiling of LLMs. It is:
- **Simpler** — no compressed tensor products, no lossy retrieval, no binding noise
- **More interpretable** — axes have clear semantic meaning defined by human-chosen poles
- **Empirically grounded** — poles are validated by generating diverse corpora and measuring convergence
- **Flexible** — axes can be designed for any dimension stakeholders care about
- **Already validated** — uses established SBERT embeddings without additional transformations

HRR adds complexity and noise without clear benefit. The problems HRR was designed to solve (fixed-dimensional composition of multiple roles) don't exist in this use case.

---

## Side-by-Side Comparison

| Dimension | Semantic Axes | HRR Prompt Binding |
|-----------|--------------|-------------------|
| **What it encodes** | Position along human-interpretable dimensions (professionalism, formality, etc.) | Prompt group membership via random rotations |
| **Semantic grounding** | Explicit — each axis has named poles with generated corpora | Implicit — "prompt-awareness" via arbitrary random vectors |
| **Interpretability** | High — "Model A is 0.3 units more professional than Model B" | Low — "Effective rank increased from 12 to 19" (what are the 19 modes?) |
| **Validation** | Built-in — convergence mode tests split-half stability | Missing — requires V1-V4 experiments that don't exist |
| **Signal-to-noise** | Clean — no transformations beyond SBERT | Noisy — circular convolution adds structured noise |
| **Inner products** | Preserved — semantic similarity is preserved | Destroyed — similar responses to different prompts become orthogonal |
| **Complexity** | Low — corpus generation + SBERT + projection | High — corpus generation + SBERT + HRR binding + density matrices |
| **Prompt dependence** | Explicit and controllable — axes define the measurement space | Hidden and arbitrary — random vectors assigned by seed |
| **Stakeholder communication** | "78% toward professional end of axis" | "Prompt sensitivity ratio = 1.52" |
| **Research maturity** | Established in NLP (e.g., word2vec analogies, bias detection) | Exploratory — no prior use for LLM profiling |

---

## Conceptual Comparison

### Semantic Axes: Explicit Dimensions

**Core idea**: Define semantic dimensions explicitly by their poles, generate representative corpora, embed, and project model outputs onto the resulting axis.

**Pipeline**:
```
1. Define axis: "professionalism" ↔ "casualness"
2. Generate diverse corpora for each pole using `semantic-pole`
   - Multi-LLM interleaving for robustness
   - Convergence mode to ensure split-half stability
   - Metadata tracking for reproducibility
3. Embed both poles with SBERT → centroids C₁, C₂
4. Define axis vector: v = C₂ - C₁
5. Project any text onto axis: score = ⟨SBERT(text), v⟩ / ‖v‖
6. Interpret: positive = toward C₂, negative = toward C₁
```

**Example output**:
- "Model A outputs score +0.68 on professionalism axis (68% toward professional pole)"
- "Model B outputs score -0.21 (21% toward casual pole)"
- "Quantization (fp32→int8) shifts professionalism by -0.15 units"

**Key strength**: The measurement space is **explicitly designed** for what stakeholders care about. If you want to measure formality, technical depth, creativity, or bias, you design an axis for it.

### HRR: Implicit Prompt-Response Coupling

**Core idea**: Bind each embedding to its prompt via circular convolution with a random vector, then analyze the resulting density matrix.

**Pipeline**:
```
1. Collect completions with prompt labels
2. Embed with SBERT → {e₁, e₂, ..., eₙ}
3. For each prompt p, generate random vector rₚ (seeded)
4. Bind: hrr_i = rₚ ⊛ SBERT(completion_i)
5. Build density matrix: ρ_HRR = (1/N) Σ |hrr_i⟩⟨hrr_i|
6. Compute effective rank, compare to ρ_SBERT
7. Interpret: ratio > 1 → "prompt-sensitive" (unverified)
```

**Example output** (fabricated, not validated):
- "Model A effective rank = 12 (SBERT), 19 (HRR), sensitivity = 1.58"
- "Model B effective rank = 8 (SBERT), 9 (HRR), sensitivity = 1.12"
- "Model A is more prompt-sensitive" (but: what does this mean for stakeholders?)

**Key weakness**: The measurement space is **implicitly defined by random vectors**. It's unclear what the 19 "modes" are, or why sensitivity = 1.58 matters.

---

## What Problems Does Each Solve?

### Semantic Axes Solves:

1. **"How does this model behave along dimensions stakeholders care about?"**
   - Professionalism, formality, technical depth, creativity, bias, sentiment
   - Axes can be tailored to any application domain

2. **"How does a model change when quantized/finetuned/prompted differently?"**
   - Measure shift along pre-defined axes
   - Example: "INT8 quantization reduces formality by 0.12 units"

3. **"Which model is better suited for my application?"**
   - Project candidate outputs onto application-relevant axes
   - Example: "For technical documentation, Model A scores +0.85 on technical-depth axis, Model B scores +0.42"

4. **"Is this axis stable and reproducible?"**
   - Convergence mode with split-half testing
   - Multi-LLM interleaving to reduce corpus artifacts

### HRR Prompt Binding Solves:

1. **"Does binding preserve compositional structure in fixed dimensions?"**
   - Yes, that's what HRR was designed for (Plate 1995)
   - But this problem doesn't exist in the LLM profiling use case

2. **"Can I encode multiple prompt-response pairs in a single vector?"**
   - Yes, via superposition: s = Σ (prompt_k ⊛ response_k)
   - But we don't need this — we have N vectors, not one

3. **Unclear: "Does the model differentiate responses across prompts?"**
   - This is what prompt sensitivity claims to measure
   - But V1 null test (required validation) hasn't been run
   - Simpler baselines (per-prompt analysis) may suffice

---

## Analytic Power

### What Semantic Axes Enables

**Profiling**: Project model outputs onto multiple axes to build a behavioral fingerprint.

```python
axes = {
    "professionalism": load_axis("prof-casual"),
    "formality": load_axis("formal-informal"),
    "technical_depth": load_axis("technical-layperson"),
    "creativity": load_axis("creative-literal"),
}

profile_A = {axis_name: project_onto_axis(model_a_outputs, axis) 
             for axis_name, axis in axes.items()}
# {'professionalism': +0.68, 'formality': +0.52, 
#  'technical_depth': -0.15, 'creativity': -0.42}
```

**Stakeholder message**: "Model A is highly professional (+0.68) and formal (+0.52), but not technical (-0.15) and not creative (-0.42). For business communication, it's a good fit. For technical writing, consider Model B."

**Comparison**: Measure Δ along each axis.

```python
delta = {axis: profile_B[axis] - profile_A[axis] for axis in axes}
# {'professionalism': -0.23, 'formality': -0.18,
#  'technical_depth': +0.61, 'creativity': +0.38}
```

**Stakeholder message**: "Model B is less professional and formal, but significantly more technical and creative. The quantization shifted behavior toward technical creativity."

**Alignment**: Measure distance from a reference along each axis.

```python
reference_profile = {...}  # From gold-standard or human outputs
alignment = {axis: abs(candidate[axis] - reference[axis]) 
             for axis in axes}
# {'professionalism': 0.08, 'formality': 0.12, ...}
```

**Stakeholder message**: "Model A aligns closely with your reference on professionalism (0.08 units away) but diverges on creativity (0.31 units). Model B is the reverse."

### What HRR Enables

**Effective rank comparison**: Compare ρ_SBERT and ρ_HRR.

```python
rank_sbert = effective_rank(density_matrix(sbert_embeddings))
rank_hrr = effective_rank(density_matrix(hrr_embeddings))
sensitivity = rank_hrr / rank_sbert
# 12.3, 18.7, 1.52
```

**Stakeholder message**: "Model A uses 12 semantic modes without prompt binding, 19 with prompt binding. This suggests... [interpretation unclear without validation]."

**Problem**: Without V1-V4 validation, we don't know if this measures real structure or binding artifacts. Without interpretable modes, stakeholders can't act on it.

---

## Interpretability for Stakeholders

### Semantic Axes

**Question**: "Is this model suitable for formal business communication?"

**Answer**: "Model A scores +0.72 on the professionalism axis and +0.65 on the formality axis, placing it in the 92nd percentile for business communication compared to our benchmark corpus. Model B scores +0.18 and +0.21 — still professional but less formal."

**Actionable**: Yes — stakeholder knows which model to choose.

### HRR

**Question**: "Is this model suitable for formal business communication?"

**Answer**: "Model A has effective rank 12.3 and prompt sensitivity 1.52. Model B has effective rank 8.1 and prompt sensitivity 1.14."

**Actionable**: No — stakeholder doesn't know what effective rank or sensitivity mean for their application. They'd need to ask "what do these numbers imply about formality?" and the answer requires either:
- Empirical grounding (V4: show effective rank correlates with formality)
- Or falling back to semantic axes anyway

---

## Validation and Robustness

### Semantic Axes

**Built-in validation**:
- **Convergence mode**: Generates samples in rounds, tests split-half stability with bootstrap CIs
- **Multi-LLM interleaving**: Reduces model-specific artifacts by round-robin across endpoints
- **Explicit diversity mechanisms**: Anti-repetition context, high temperature, varied system prompts

**Empirical checks**:
- Pole corpora should cluster (low intra-pole variance, high inter-pole distance)
- Axes should be orthogonal or interpretably correlated (e.g., "professionalism" and "formality" correlate)
- Projections should be stable under resampling (bootstrap confidence intervals)

**Existing tooling**: The `semantic-pole` CLI has convergence reporting, metadata tracking, and Miller-friendly JSONL output for post-hoc analysis.

### HRR

**Missing validation**:
- **V1 null test**: Does fake prompt labeling produce sensitivity > 1? (tests for artifacts)
- **V2 baseline**: Do per-prompt density matrices give the same information? (tests necessity)
- **V3 seed sensitivity**: Does changing the seed change conclusions? (tests reliability)
- **V4 grounding**: Does effective rank correlate with observable behavior? (tests interpretability)

**No tooling**: Validation experiments are defined but not implemented.

**Current status**: Code exists, but claims are unverified.

---

## Complexity Cost-Benefit

### Semantic Axes

**Costs**:
- Must design axes (choose poles, write prompts)
- Must generate corpora (LLM API calls, ~100-300 samples per pole)
- Must embed corpora once to compute centroids

**Benefits**:
- Explicit, interpretable measurement space
- Tailored to stakeholder's application
- No additional math beyond SBERT + projection
- Well-validated (existing SBERT model, standard geometry)

**Verdict**: Complexity is **domain design** (what axes matter?), not **mathematical machinery**.

### HRR

**Costs**:
- All costs of semantic axes (need prompts + corpora anyway)
- **Plus**: Circular convolution binding (FFT overhead)
- **Plus**: Density matrix construction (eigendecomposition)
- **Plus**: Effective rank computation
- **Plus**: Validation experiments V1-V4 (not yet run)
- **Plus**: Interpretation of abstract metrics (effective rank, sensitivity ratio)

**Benefits**:
- Fixed dimensionality (doesn't matter — we're not composing many roles)
- Prompt-awareness (unclear if this is signal or artifact)
- Possible: detects cross-prompt interactions that per-prompt analysis misses (unverified)

**Verdict**: Complexity is **mathematical machinery** for unclear gain. The benefit must be demonstrated empirically, and it hasn't been.

---

## Recommended Approach

### Use Semantic Axes

**For**:
- Psychometric profiling ("what is this model like?")
- Application-specific alignment ("does this model fit my use case?")
- Interpretable comparisons ("how does quantization change behavior?")
- Stakeholder communication ("Model A is 68% toward professional")

**How**:
1. Identify dimensions stakeholders care about (professionalism, creativity, bias, etc.)
2. Design axes by defining poles and generating diverse corpora with `semantic-pole`
3. Embed poles with SBERT, compute centroids, define axis vectors
4. Project candidate model outputs onto axes
5. Report positions and deltas in interpretable units

**Advantages over HRR**:
- No binding noise
- No arbitrary random vectors
- No unvalidated effective rank claims
- Direct stakeholder interpretability

### Consider Per-Prompt Density Matrices

**If prompt-specific analysis is needed**, the simpler approach:
1. Build separate ρ_p for each prompt p
2. Compute per-prompt effective ranks, entropies, trace distances
3. Compare distributions across prompts or models

**Advantages over HRR**:
- Exact (no compression)
- Interpretable (each prompt is its own measurement)
- No seed dependence
- No PCA basis inconsistency issues

**Disadvantage**:
- Doesn't capture cross-prompt structure (if it exists and matters)

### Do Not Use HRR

**Unless**:
- V1 null test shows sensitivity is not artifactual
- V2 baseline comparison shows HRR adds information beyond per-prompt analysis
- V3 seed sensitivity shows results are reproducible
- V4 grounding shows metrics correlate with observable behaviors stakeholders care about

**Currently**: None of these are true. HRR is exploratory code with unverified claims.

---

## Case Study: Professionalism Measurement

Let's compare how each approach would answer: "Is Model A more professional than Model B?"

### Semantic Axes Approach

**Setup**:
1. Generate `professionalism.jsonl` (100 samples: "Please review the attached quarterly report...")
2. Generate `casualness.jsonl` (100 samples: "wanna grab lunch later?")
3. Embed both, compute centroids C_prof, C_casual
4. Axis vector: v_prof = C_prof - C_casual

**Measurement**:
```python
score_A = project(model_a_outputs, v_prof)  # +0.72
score_B = project(model_b_outputs, v_prof)  # +0.18
```

**Interpretation**: Model A outputs are 0.72 units toward the professional pole; Model B outputs are 0.18 units toward professional. Model A is 0.54 units more professional.

**Stakeholder message**: "On a scale from casual (-1) to professional (+1), Model A scores +0.72 and Model B scores +0.18. Model A is significantly more professional."

**Validation**: Check that pole corpora cluster, axis is stable under resampling, scores correlate with human ratings of professionalism.

### HRR Approach

**Setup**:
1. Collect outputs from Model A and Model B on multiple prompts
2. Embed with SBERT
3. Bind to prompt IDs via HRR
4. Build ρ_SBERT and ρ_HRR for each model
5. Compute effective ranks

**Measurement**:
```python
rank_A_sbert = 12.3, rank_A_hrr = 18.7, sensitivity_A = 1.52
rank_B_sbert = 8.1,  rank_B_hrr = 9.2,  sensitivity_B = 1.14
```

**Interpretation**: Model A has higher effective rank (12 vs 8) and higher prompt sensitivity (1.52 vs 1.14).

**Stakeholder message**: "Model A uses more behavioral modes and differentiates more across prompts. This suggests... [what does this say about professionalism? Unclear.]"

**To connect to professionalism**: Would need V4 validation showing effective rank or sensitivity correlates with professionalism scores. Without that, you're back to measuring professionalism via semantic axes anyway.

---

## Conclusion

**Semantic axes** is the right tool for psychometric LLM profiling:
- Explicit, interpretable dimensions
- Direct stakeholder communication
- Empirically grounded via corpus generation + convergence testing
- Simpler pipeline (no binding transformations)
- Already validated (SBERT + projection)

**HRR** adds complexity without demonstrated benefit:
- Implicit dimensions (random binding vectors)
- Unclear stakeholder value (what does sensitivity = 1.52 mean?)
- Unvalidated claims (V1-V4 experiments don't exist)
- Additional noise (circular convolution is lossy)
- Destroys semantic similarity across prompts (inner products not preserved)

**Recommendation**: Use semantic axes for all LLM profiling work. If prompt-specific analysis is needed, use per-prompt density matrices (simpler than HRR, exact instead of lossy). Reserve HRR for use cases where you actually need to compose many roles in fixed dimensions (which this is not).
