# HRR for Structured Source Comparison via LLM-as-Judge Labels

This document presents an alternative use case for HRR: encoding multi-dimensional LLM-as-judge rubric labels for structured source comparison. This differs substantially from the prompt-binding approach critiqued in `HRR-quantum-metrics.md`.

**Source**: ChatGPT conversation discussing HRR for source-comparison analysis
**Date**: 2026-05-30

---

## Summary

This conversation proposes using **HRR (Holographic Reduced Representations) for structured source comparison** using LLM-as-judge labels—a different use case than the prompt-binding approach we critiqued.

### The Core Idea

**Problem**: When comparing two information sources (Source A vs Source B), stakeholders don't just want to know "the embedding distributions differ"—they want to know **how the sources differ along interpretable dimensions** like:
- Framing (risk vs. solution-oriented)
- Actionability (concrete recommendations vs. abstract analysis)
- Evidence quality
- Audience (technical vs. strategic)
- Topic coverage vs. topic framing

**Proposed Solution**: Use LLM-as-judge to label each text with a rubric across multiple dimensions, then encode these labels as HRR annotation vectors that can be combined with text embeddings.

### The HRR Encoding

Each text gets labeled by an LLM judge across dimensions like:
```
tone = alarmist
risk_type = legal
actionability = low
evidence_quality = weak
```

These are encoded as an HRR vector:
```
a_i = r_tone ⊛ v_alarmist + r_risk ⊛ v_legal + r_actionability ⊛ v_low + r_evidence ⊛ v_weak
```

Then combined with the text embedding:
```
m_i = α*h_i + β*a_i
```
where `h_i` is the SBERT embedding and `a_i` is the annotation structure.

### What HRR Enables (According to the Proposal)

1. **Source differences in stakeholder-relevant dimensions**
   - Compare sources by annotation centroids: `Δ = ā_A - ā_B`
   - Unbind specific dimensions: `r_actionability^-1 ⊛ Δ` tells you which actionability labels distinguish sources
   - Output: "Source A is more associated with low actionability, high uncertainty, legal risk"

2. **Separate "what they talk about" from "how they frame it"**
   - Two sources may discuss the same topics but frame them differently
   - Raw embeddings: "Both discuss cybersecurity"
   - HRR labels: "A frames it as risk/compliance, B frames it as implementation/cost"

3. **Clean provenance encoding**
   - Bind labels to metadata: judge model, rubric version, confidence scores
   - Support statements like: "The source difference is stable across two judge prompts"

4. **Similarity search over "action profiles"**
   - Query: "high urgency + low evidence + regulatory risk + no recommendation"
   - Retrieve examples matching this profile from both sources
   - Show stakeholders concrete representative cases

5. **Identify source-specific blind spots**
   - "Source A identifies risks but rarely recommends mitigations"
   - "Source B gives recommendations but weakly supports them with evidence"
   - Actionable: "Ask Source A for more recommendations, ask Source B for stronger evidence"

6. **Structured divergence measurement**
   - Compare sources within topic clusters to avoid the confound: "they look different because they cover different topics"
   - Measure: overall divergence, per-dimension divergence, topic-conditioned divergence

7. **Better stakeholder summaries**
   - Instead of "embeddings differ," output: "Source A has high risk + low actionability + weak evidence + institutional framing; Source B has moderate risk + high actionability + implementation detail + user-impact framing"

### The Proposed Workflow

1. Embed all texts (SBERT)
2. Cluster by semantic topic
3. Use LLM-as-judge to label with fixed rubric
4. Encode judge labels as HRR annotation vectors
5. Create hybrid text + annotation vectors
6. Compare Source A vs Source B globally, within topics, across rubrics
7. Decode largest differences into human-readable labels
8. Retrieve representative examples
9. Convert patterns into stakeholder recommendations

### Key Claimed Advantage

**HRR vs. simple label tables**:
- Simple counts: "Source A: 42% high risk, Source B: 25% high risk" → good for dashboards
- HRR annotation vectors: good for **compositional, searchable profiles** that support:
  - Similarity search over label structures
  - Nearest examples by annotation profile
  - Centroid differences
  - Role-based probing (unbinding specific dimensions)
  - Hybrid text/annotation retrieval

---

## Comparison to Prompt-Binding HRR (Our Critique)

This is a **substantially different use case** from what we critiqued in `HRR-quantum-metrics.md`:

| Aspect | Our Critique (Prompt Binding) | This Proposal (Judge Labels) |
|--------|-------------------------------|------------------------------|
| **Number of roles** | 1 (prompt ⊛ content) | Many (tone ⊛ value, risk ⊛ value, actionability ⊛ value, ...) |
| **Compositional depth** | Shallow (categorical label) | Deep (multi-dimensional rubric) |
| **Use of retrieval** | No—only binding, never unbinding | Yes—unbind to probe specific dimensions |
| **Semantic vs. interpretive** | Semantic (prompt group membership) | Interpretive (LLM judge dimensions) |
| **Stakeholder framing** | Abstract ("effective rank increased") | Concrete ("Source A is risk-focused, Source B is action-oriented") |
| **What it encodes** | "This response came from prompt k" | "This text has tone=X, risk=Y, actionability=Z" |

### Why This Use Case Is More Compelling

1. **Actually compositional**: Encoding 5-10 rubric dimensions in one vector is what HRR was designed for
2. **Uses retrieval**: Unbinding `r_dimension^-1 ⊛ Δ` to probe specific dimensions leverages HRR's key feature
3. **Clear stakeholder value**: "Source A frames as risk, Source B frames as action" is directly interpretable
4. **Addresses real need**: Comparing sources along structured dimensions that stakeholders care about

### Critiques That Still Apply

Even for this better use case, validation is still required:

1. **Signal-to-noise**: Binding is still lossy; does the noise outweigh the structure?

2. **Validation needed**: Does this actually work better than alternatives?
   - **Alternative 1**: Structured metadata + filtering (e.g., SQL-style queries over label columns)
   - **Alternative 2**: Multi-dimensional embeddings (one SBERT vector per rubric dimension)
   - **Alternative 3**: Simple label counts + topic clustering
   - **Alternative 4**: Concatenated text + label feature vectors

3. **Seed sensitivity**: Random role vectors—are results stable across seeds?

4. **Empirical comparison needed**: Show that HRR hybrid vectors outperform simpler baselines on real source-comparison tasks

5. **Interpretability of unbinding**: When you unbind `r_actionability^-1 ⊛ Δ`, the result is noisy—how noisy is acceptable for stakeholder interpretation?

6. **Complexity cost**: Is the mathematical machinery worth the interpretive gain compared to structured metadata tables?

### Recommended Validation

If pursuing this approach, build a small proof-of-concept comparing:

1. **HRR annotation vectors** (this proposal)
2. **Simple label counts + topic clusters** (baseline)
3. **Separate SBERT embeddings per rubric dimension** (multi-embedding baseline)
4. **Concatenated text + label feature vectors** (feature concatenation baseline)

...on a real source-comparison task with ground truth stakeholder judgments about which analysis is most useful.

**Success criteria**:
- HRR retrieval/probing produces more actionable insights than baselines
- Results are stable across random seeds
- Unbinding noise is acceptable for stakeholder interpretation
- The added complexity is justified by empirical gains

---

## Original ChatGPT Conversation

Below is the full text of the conversation discussing HRR for LLM-as-judge label encoding:

---

Using HRR here buys you a way to turn LLM-as-judge labels into analyzable structured signals, not just metadata attached to embeddings.

For stakeholder-facing source comparison, that matters because you usually do not want only:
"Source A and Source B have different embedding distributions."

You want something more like:
"Source A is more likely to frame the topic as regulatory risk, uses more uncertainty language, and is judged less action-oriented; Source B is more operational, more solution-oriented, and has more implementation detail."

HRR can help bridge that gap.

### The core setup

Suppose each text item has:
- text embedding
- source = A or B
- LLM-as-judge labels
- confidence scores
- rubric dimensions
- possibly paragraph/section roles
- possibly time or topic

A plain text embedding captures semantic similarity. But the judge labels capture interpretive dimensions that stakeholders care about.

For example:
- tone = alarmist
- recommendation = absent
- risk_type = legal
- audience = technical
- actionability = low
- specificity = high
- sentiment = negative
- evidence_quality = weak

An HRR-style representation lets you encode these as structured role-value bindings:

a_i = r_tone ∗ v_alarmist + r_risk ∗ v_legal + r_actionability ∗ v_low + r_evidence ∗ v_weak

Then you can combine that with the text embedding:

m_i = α h_i + β a_i

where h_i is the text embedding mapped into the HRR space, and a_i is the annotation structure.

### What HRR buys you analytically

#### 1. You get source differences in stakeholder-relevant dimensions

Instead of comparing sources only by raw embedding geometry, you can compare them by structured judge dimensions.

For each source, you can compute an average annotation vector:

ā_A = (1/n_A) Σ_{i∈A} a_i
ā_B = (1/n_B) Σ_{i∈B} a_i

Then examine the difference:

Δ = ā_A − ā_B

That difference vector can be probed by unbinding with role vectors.

For example:

r_actionability^{-1} ∗ Δ

tells you which actionability labels distinguish the sources.

So you can produce statements like:

Compared with Source B, Source A is more associated with:
- low actionability
- high uncertainty
- legal/regulatory risk
- weak evidence

That is much more useful than saying the embedding centroids differ.

#### 2. You can separate "what they talk about" from "how they frame it"

This is a major advantage.

Two sources may discuss the same topics but frame them differently.

Raw embeddings may say:
Source A and Source B both discuss cybersecurity procurement.

HRR labels may say:
Source A frames it as risk and compliance.
Source B frames it as implementation and cost reduction.

That gives stakeholders actionable contrast:
Same topic, different framing.

This is often exactly what people want from source-comparison analysis.

#### 3. You can encode provenance cleanly

Because your labels come from an LLM judge, you may care about:
- which judge model
- which prompt/rubric
- which source
- which label dimension
- which confidence score
- which time period

HRR lets you bind labels to provenance:

a_i = r_judge ∗ v_gpt-judge + r_rubric ∗ v_actionability-rubric + r_source ∗ v_A + r_dimension ∗ v_risk

More usefully, you can encode something like:

r_judge1 ∗ r_risk ∗ v_legal

versus:

r_judge2 ∗ r_risk ∗ v_technical

That helps if you later compare judge agreement, prompt sensitivity, or label stability.

For stakeholders, this supports caveats like:
The source difference is stable across two judge prompts.

or:
The apparent difference depends heavily on the rubric wording.

That is important if the results are going to drive decisions.

#### 4. You can do similarity search over "action profiles"

This may be one of the most practical benefits.

You can search not just for semantically similar documents, but for documents with similar decision-relevant profiles.

For example, you could query for:

high urgency + low evidence + regulatory risk + no clear recommendation

and retrieve examples from both sources.

That lets you show stakeholders concrete representative cases:
Here are Source A documents that fit the problematic pattern.
Here are the closest Source B analogs.
Here is how their judge profiles differ.

This moves the analysis from abstract statistics to inspectable examples.

#### 5. You can identify source-specific blind spots

Suppose you define stakeholder-relevant dimensions such as:
- problem clarity
- recommended action
- evidence strength
- implementation feasibility
- cost impact
- risk severity
- audience specificity

For each source, you can estimate which combinations are overrepresented.

Examples:
- Source A often identifies risks but rarely recommends mitigations.
- Source B often gives recommendations but weakly supports them with evidence.
- Source A has more strategic framing; Source B has more operational framing.
- Source B discusses implementation, but mostly for already-mature technologies.
- Source A emphasizes harms to institutions; Source B emphasizes harms to users.

That is actionable because it suggests different interventions:
- Ask Source A for more concrete recommendations.
- Ask Source B for stronger evidence citations.
- Use Source A for risk discovery and Source B for implementation planning.
- Flag items where urgency is high but evidence is weak.

#### 6. You can measure structured divergence

With HRR annotation vectors, you can compute distances between source profiles:
- overall annotation divergence
- per-rubric-dimension divergence
- topic-conditioned divergence
- time-conditioned divergence
- judge-conditioned divergence

For example:
Within the same topic cluster, Source A has higher risk framing and lower actionability.

That is better than global comparison because it avoids a common confound:
The sources look different only because they cover different topics.

A strong analysis would compare:
- topic difference
- label/framing difference
- topic-conditioned label difference

HRR gives you a compact way to represent the label/framing side.

#### 7. You can produce better stakeholder summaries

The HRR layer can support outputs like:

| Stakeholder question | HRR-enabled analysis |
|---------------------|---------------------|
| What is different between the sources? | Compare source-level annotation centroids |
| Are they covering different topics or framing the same topics differently? | Compare embedding clusters vs annotation profiles |
| Which differences are actionable? | Probe role-bound labels like actionability, risk, recommendation |
| Can you show examples? | Retrieve nearest examples by annotation profile |
| Are findings robust? | Compare across judges, prompts, or rubrics |
| What should we do? | Translate overrepresented label patterns into interventions |

### A concrete workflow

I would structure the analysis like this:

1. Embed all texts.
2. Cluster or stratify by semantic topic.
3. Use LLM-as-judge to label each item with a fixed rubric.
4. Encode judge labels as HRR annotation vectors.
5. Create separate vectors:
   - raw text embedding
   - HRR annotation vector
   - hybrid text + annotation vector
6. Compare Source A vs Source B:
   - globally
   - within topic clusters
   - across time
   - across rubric dimensions
7. Decode the largest HRR differences into human-readable labels.
8. Retrieve representative examples for each difference.
9. Convert recurring patterns into stakeholder recommendations.

The key step is not merely building HRR vectors. It is using them to generate contrastive profiles.

### What it buys you over simpler methods

A simple table of labels can already answer some questions. For example, you can count:
- Source A: 42% high risk
- Source B: 25% high risk

So HRR is not automatically better for every analysis.

HRR becomes more valuable when you need combinations like:
source + topic + judge + rubric dimension + label + confidence

or when you want vector operations such as:
- similarity search over label structures
- nearest examples by annotation profile
- centroid differences
- role-based probing
- hybrid text/annotation retrieval

In other words:
- flat label counts → good for simple dashboards
- HRR annotation vectors → good for structured, compositional, searchable profiles

### The biggest practical payoff

For stakeholder-actionable analysis, HRR lets you say:

These sources differ not only in semantic content, but in recurring judgment patterns.

For example:

Source A is closer to the profile:
  high risk + low actionability + weak evidence + institutional framing

Source B is closer to the profile:
  moderate risk + high actionability + implementation detail + user-impact framing

Then you can attach representative examples and recommendations:

Action: Use Source A for early risk discovery, but require follow-up synthesis before decision meetings.

Action: Use Source B for operational planning, but add evidence-quality checks.

Action: Flag documents matching high-urgency/low-evidence profiles for human review.

That is the real value: HRR gives you a compact mathematical representation that can support comparison, retrieval, probing, and explanation over structured annotations attached to embedded text.

---

## Related Documents

- **HRR-quantum-metrics.md** — Critique of HRR for prompt binding (the use case we rejected)
- **HRR-CODE-CRITIQUE.md** — Implementation issues in the prompt-binding code
- **SEMANTIC-AXES-VS-HRR-COMPARISON.md** — Why semantic axes is superior for psychometric profiling
- **CLAUDE.md** (semantic-pole project) — The corpus generation framework for semantic axes

## Status

**Exploratory**. This judge-labels use case is more compelling than prompt-binding, but still requires:
- Empirical validation against simpler baselines
- Proof that unbinding noise is acceptable
- Demonstration of stakeholder value on real tasks

If validated, this could be a legitimate application of HRR for structured source comparison where compositional encoding of multi-dimensional rubrics provides value beyond simple metadata tables.
