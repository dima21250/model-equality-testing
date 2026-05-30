# Holographic Reduced Representations and Quantum-Inspired Metrics

## Background: HRR

Holographic Reduced Representation (HRR) is a framework developed by Tony Plate (1995) for encoding compositional structure in fixed-dimensional distributed vectors.

**Core operations:**

- **Circular convolution** (⊛) binds two vectors into one of the same dimensionality. Given vectors **a** and **b** of dimension *d*, their binding **a** ⊛ **b** is also dimension *d*.
- **Circular correlation** is the approximate inverse — it retrieves a noisy version of **b** from the bound pair given **a** as a cue.

HRR solves the problem of representing structured, relational information (like "John loves Mary") in a single fixed-width vector without concatenation or growing dimensionality. Multiple bindings can be superimposed by addition and individual role-filler pairs can still be decoded.

HRR sits in the family of Vector Symbolic Architectures (VSAs) alongside Tensor Product Representations (Smolensky), Binary Spatter Codes (Kanerva), and Multiply-Add-Permute. HRR's advantage is that binding doesn't increase dimensionality (unlike outer-product tensor representations). Its disadvantage is lossy retrieval — it trades exact decoding for fixed-size representation.

## HRR as a Hilbert Space

HRR vectors live in R^d (or C^d in the frequency domain), which is a finite-dimensional Hilbert space. The standard inner product ⟨a, b⟩ = Σ aᵢbᵢ provides linearity, conjugate symmetry, positive-definiteness, and completeness (automatic in finite dimensions). This means the existing density matrix construction — `ρ = E_norm.T @ E_norm / N`, i.e., ρ = (1/N) Σ |ψᵢ⟩⟨ψᵢ| — is immediately valid on HRR vectors.

### Fourier Domain Structure

Circular convolution becomes element-wise multiplication under DFT:

    F(a ⊛ b) = F(a) ⊙ F(b)

The DFT is a unitary transformation — it preserves inner products (Parseval's theorem). This means we can move freely between the time/spatial domain and the frequency domain without breaking the Hilbert space structure. In the Fourier basis, binding is *diagonal*: each frequency component is independent.

### Compressed Tensor Product

In standard quantum mechanics, composing two systems uses the tensor product: H_A ⊗ H_B, which has dimension d_A × d_B — exponential growth. Circular convolution is a **compressed tensor product**: it maps H × H → H (same dimension d). Plate showed this compression acts like a random projection of the full tensor product, which is why retrieval is approximate but statistically reliable in high dimensions.

## Connections to the Quantum Metrics Framework

### 1. Density Matrices from HRR Vectors Are Legitimate Density Operators

Trace distance, von Neumann entropy, and QRE all remain well-defined on density matrices constructed from HRR vectors. No mathematical modification is needed.

### 2. HRR Density Matrices Encode Different Information than SBERT Density Matrices

An SBERT density matrix captures the distribution of completions in semantic space. An HRR density matrix, where each vector is `prompt_role ⊛ sbert(completion)`, captures the distribution of *prompt-bound semantic states*. The eigenspectrum of ρ_HRR reflects compositional structure that ρ_SBERT collapses.

### 3. Approximate Unitarity

When Plate-style random vectors are used for role bindings (components drawn i.i.d. from N(0, 1/d)), binding approximately preserves norms: ‖a ⊛ b‖ ≈ ‖a‖ · ‖b‖. The approximation tightens as d grows. The L2-normalization step in `full_density_matrix` absorbs the residual error.

### 4. Fourier-Domain Eigenstructure

Since binding is diagonal in the Fourier basis, eigenvalues of density matrices built from HRR vectors decompose along frequency components. Low frequencies capture coarse compositional patterns; high frequencies capture fine-grained binding details. This could enable band-limited "observables" that measure specific structural aspects — e.g., "is the prompt-response relationship preserved under quantization?" vs. "is the raw semantic content preserved?"

## Potential Directions

### A. Compositional Binding as a Richer Embedding

Each completion currently maps to a single SBERT vector. HRR could encode structured relationships within a completion — e.g., bind prompt identity with semantic content:

    representation = prompt_role ⊛ prompt_embedding + response_role ⊛ response_embedding

This produces a single fixed-width vector that preserves *which part contributed what*, rather than flattening everything into one pooled embedding. Density matrices would then encode compositional structure, not just aggregate semantics.

### B. Superposition as a Density Matrix Analog

HRR's core move — superimposing multiple bound vectors by addition — is structurally analogous to computing ρ = (1/N) Σ |ψᵢ⟩⟨ψᵢ|. The superposition s = Σ (role_k ⊛ filler_k) is a compressed representation of the full mixture. Building density matrices from HRR-encoded vectors could capture **role-filler distributional structure** that SBERT embeddings alone miss (e.g., "does this model use different vocabulary for different prompt types?").

### C. Binding for Prompt-Conditional Analysis

The current permutation tests pool all prompt responses together. HRR could bind each completion to its prompt identity before constructing the density matrix, creating a representation that is simultaneously prompt-aware and fixed-dimensional. This is relevant because results already show prompt choice matters (en[0] vs en[5] gives trace distance 0.470) — HRR binding could factor out prompt effects from model effects without splitting the sample.

## Toward Psychometric Profiling of LLMs

### Motivation

The current framework answers "are these two models different?" The deeper goal is "what *is* this model like?" — characterization, not just discrimination. HRR + quantum metrics can get there because the density matrix eigenstructure becomes a **behavioral profile** of the model, not just a comparison statistic.

The aim is to give stakeholders concrete, application-specific evidence about LLM suitability without running exhaustive benchmarks. LLMs have a unique advantage over human psychometric subjects: we can generate thousands of responses per prompt, giving us **population-level statistics from a single "individual."**

### The Psychometric Parallel

In psychometrics, the process is:

1. Give a battery of test items (= prompts)
2. Observe responses from an individual (= completions from one model)
3. Build a covariance matrix of item responses (≈ density matrix)
4. Factor-analyze to extract latent traits (≈ eigendecomposition of ρ)
5. Characterize the individual by factor scores (≈ eigenvalues + eigenvectors)

Without HRR, the density matrix captures "the distribution of what the model says." With HRR binding, it captures "the distribution of what the model says **in response to what.**" That distinction is exactly what separates a generic semantic profile from a psychometric one — in psychometrics, item identity matters. The same answer means different things depending on which question elicited it.

When `prompt_id ⊛ sbert(completion)` is used as the binding, the resulting density matrix's eigenstructure decomposes into modes that reflect **prompt-response coupling patterns**, not just semantic content.

### What This Tells Stakeholders

#### 1. Behavioral Diversity Profile (Eigenspectrum Shape)

The eigenvalues of ρ tell you how many distinct "behavioral modes" the model uses. A model with a few dominant eigenvalues and a sharp dropoff is **concentrated** — it has a small repertoire of response patterns. A model with a flatter spectrum is **diffuse** — it draws from a wider behavioral range.

*Stakeholder language:* "Model A generates responses across 8 distinct semantic modes for your prompt set; Model B concentrates on 3. If your application needs varied, creative outputs, Model A is better suited. If you need predictable, consistent responses, Model B is preferable."

This is analogous to bandwidth in personality psychology — some people have narrow, consistent behavioral patterns; others are more situationally adaptive.

#### 2. Prompt-Response Coupling Strength (Effective Rank of ρ_HRR vs ρ_SBERT)

Compare the effective rank (number of significant eigenvalues) of density matrices built from plain SBERT vectors vs. HRR-bound vectors. If HRR increases the effective rank substantially, the model is **prompt-sensitive** — it genuinely differentiates its responses across prompts. If the effective rank barely changes, the model gives **homogeneous responses regardless of prompt**.

*Stakeholder language:* "Model A strongly differentiates between your prompt categories; Model B gives similar responses regardless of what you ask. For an application requiring precise prompt-following, Model A is measurably better."

#### 3. Application-Specific Alignment (Trace Distance from a Reference)

Construct a **reference density matrix** representing what ideal outputs look like for the stakeholder's application — either from curated examples, a gold-standard model, or human-written responses. Then measure each candidate model's trace distance from that reference.

*Stakeholder language:* "Model A is 0.12 trace distance from your reference outputs; Model B is 0.31. Model A's output distribution is substantially closer to what you want."

This replaces exhaustive benchmarking with a distributional alignment score tailored to the specific application.

#### 4. Semantic Stability (Sensitivity to Prompt Perturbation)

Give the same prompt in paraphrased forms. Measure trace distance between the density matrices of original vs. paraphrased prompt responses. A model with low trace distance is **robust** — it captures intent rather than surface form. High trace distance means it is **brittle** to phrasing.

*Stakeholder language:* "Model A's outputs shift minimally when prompts are rephrased (trace distance 0.05); Model B is 4x more sensitive (0.21). For user-facing applications where you can't control exact phrasing, Model A is more reliable."

#### 5. Cross-Domain Transfer Profile (Block Structure in the Fourier Domain)

Using HRR's Fourier decomposition, analyze whether a model's behavioral modes cluster by prompt domain (technical vs. creative vs. factual) or cut across domains. A model with strong block structure has **domain-specific behaviors**. One with diffuse structure has a **unified behavioral style**.

*Stakeholder language:* "Model A writes technical content differently than creative content — it adapts its style to the domain. Model B uses a consistent voice across all domains. For a multi-purpose application, Model A better matches how human experts shift register."

### What This Gives You That Benchmarks Don't

Benchmarks measure **accuracy on fixed tasks** — they tell you what a model gets right. This approach measures **distributional behavior across your specific prompts** — it tells you *how* the model responds, not just whether it is correct. That is the psychometric distinction: an IQ test tells you how smart someone is; a personality inventory tells you how they will behave in context.

Instead of "Model A scores 85% on MMLU," the output is: "Model A's behavioral profile is a better geometric fit to your application's requirements, and here is why, decomposed into interpretable factors."

## Critical Assessment

The following critique examines whether HRR actually adds value to the quantum metrics framework, organized from most fundamental to most practical.

### 1. Binding Encodes Group Membership, Not Compositional Structure

The claim that HRR captures "what the model says *in response to what*" is misleading. Circular convolution with a **random** prompt vector does not encode the semantic relationship between prompt and response. It encodes **group membership**: "this response came from prompt k."

Convolving all completions from prompt p₁ with random vector r₁ and all from prompt p₂ with r₂ is approximately equivalent to applying a different random rotation to each prompt's cluster of embeddings — categorical scrambling, not compositional structure.

A truly compositional representation would need the binding to reflect something about *how the prompt relates to the response*. Using the prompt's SBERT embedding as the binding vector (instead of a random one) would at least tie the binding to prompt semantics, but even then the interpretation of circular convolution between two SBERT vectors is unclear.

### 2. Prompt Sensitivity Ratio May Be Artifactual

This is the most serious concern. Take *any* set of vectors, apply different random transformations to subsets, and the effective rank will increase — the subsets are mechanically spread into non-overlapping regions. The `prompt_sensitivity_ratio` may be > 1.0 **by construction**, regardless of whether the model actually differentiates across prompts.

**Critical falsifiability test:** Take completions from a single prompt, randomly assign fake prompt labels, run the HRR binding. If the sensitivity ratio is still > 1.0, the measure is an artifact of the method, not a property of the model. This test must be run before making any claims.

### 3. Simpler Alternatives May Achieve the Same Goals

If the goal is prompt-conditional analysis, existing tools may suffice:

- **Per-prompt density matrices**: Build separate ρ for each prompt, compare eigenspectra. Interpretable, exact, no noise added.
- **Block-diagonal construction**: Stack per-prompt density matrices along the diagonal. Effective rank of this block structure directly measures prompt differentiation.
- **Conditional trace distance**: Compute trace distance *per prompt*, then average. This tells whether quantization affects responses differently depending on the prompt.

All of these are exact, interpretable, and do not sacrifice signal-to-noise. HRR's advantage — fixed dimensionality — matters when composing many roles simultaneously (agent ⊛ action ⊛ object ⊛ modifier). With a single binding (prompt ⊛ content), it is a compression tool with nothing to compress.

### 4. Inner Products Are Not Preserved Under Binding

Circular convolution approximately preserves *norms* but does **not** preserve *inner products* between different vectors. If two completions are semantically similar (large ⟨a, b⟩), their bound versions ⟨r₁ ⊛ a, r₂ ⊛ b⟩ are approximately zero when r₁ ≠ r₂ (different prompts). Consequences:

- Two semantically identical responses to different prompts become orthogonal after binding.
- The density matrix geometry changes fundamentally — it no longer reflects semantic similarity across prompts.
- The information that "this model gives similar responses regardless of prompt" (itself a meaningful finding) is explicitly destroyed.

L2-normalization absorbs norm distortion but cannot fix the inner product problem. The density matrices are *valid* operators, but they encode a **different metric space** than SBERT density matrices, and that metric space has not been characterized.

### 5. Fourier Domain Interpretability Claim Is Unfounded

The claim that "low frequencies capture coarse compositional patterns; high frequencies capture fine-grained binding details" is true for signals with inherent spatial or temporal structure (images, audio). HRR vectors have **no such structure** — the dimension ordering is arbitrary. Permuting the dimensions of an SBERT embedding before binding produces a completely different Fourier decomposition with equally valid but completely different "frequency components."

The band-limited observables idea has no mathematical basis in this context. It should be either dropped or demonstrated empirically by showing that specific frequency bands correlate with interpretable properties.

### 6. Signal-to-Noise Relative to Effect Sizes

The fp32-vs-int8 trace distance is 0.154 — a subtle effect. Circular convolution adds noise proportional to the number of bindings and inversely proportional to √d. With d=768 and a single binding, per-component noise is small, but it accumulates across the eigenspectrum.

Worse: the noise is **correlated within each prompt group** (same binding vector) and **uncorrelated across prompt groups**. This creates a structured noise pattern that could masquerade as a real signal in the eigenspectrum.

### 7. The Psychometric Analogy Is Strained

In psychometrics:

- Test items are **designed** to probe specific latent constructs (e.g., extraversion, conscientiousness).
- The covariance matrix reveals trait structure *because items were selected for discriminant validity*.
- Factor analysis is meaningful because items have known loadings on known constructs.

The prompts in the current dataset are drawn from Wikipedia passages — they were not designed to probe orthogonal behavioral dimensions of an LLM. The "factor structure" of the density matrix reflects the *prompts chosen*, not stable latent traits of the model. Change the prompts, and the "personality" changes.

For the psychometric framing to be rigorous, one would need to:

- Design prompts that systematically probe different behavioral dimensions (formality, creativity, accuracy, verbosity, etc.)
- Validate that eigenvalues are stable across samples (test-retest reliability)
- Show that eigenvalue patterns correlate with downstream task performance (predictive validity)

This is a full research program, not a consequence of adding HRR.

### 8. Stakeholder Claims Are Ungrounded

Every example output in the profiling code is fabricated. Claims like "Model A generates responses across 8 distinct semantic modes" are only meaningful if:

- Effective rank is stable across random samples (low variance under resampling)
- The number "8" is interpretable (what are the 8 modes? can they be named?)
- The number correlates with something a stakeholder cares about

Currently, effective rank is a number that comes out of an eigendecomposition. It has not been established that it means anything actionable. A stakeholder hearing "8 modes vs 3 modes" will ask "what are the modes?" and that question cannot yet be answered.

### 9. Reproducibility Depends on Arbitrary Choices

- **Seed sensitivity**: Different seeds produce different prompt vectors, different density matrices, different metrics. If variance across seeds is large relative to the effect size, the method is unreliable.
- **Prompt set sensitivity**: Different prompt selections produce different profiles. Is a "behavioral profile" a property of the model, or of the model-prompt interaction? Psychometrics has this problem too (test form effects), but addresses it with item response theory.
- **Embedding model sensitivity**: Different SBERT models produce different semantic spaces and different HRR binding geometries. A profile might be an artifact of all-mpnet-base-v2, not a genuine model property.

### 10. What Does HRR Actually Add?

The existing quantum metrics already detect quantization effects, distinguish between models, and characterize distributional differences. The concrete question: **what empirical finding would HRR enable that the current framework cannot produce?**

If the answer is "prompt sensitivity measurement," then simpler approaches (per-prompt density matrices) may already do this better. If the answer is "richer behavioral profiles," then the enrichment must be shown to be real signal, not noise.

## Constructive Path Forward

The core intuition — that prompt-conditioned distributional analysis tells you more than unconditioned analysis — is probably right. The question is whether HRR is the right tool for it, or whether simpler alternatives suffice. Before investing further, the following validation steps are needed.

### Required Validation Experiments

#### V1. Null Test for Prompt Sensitivity Artifact

Take completions from a single prompt, randomly assign fake prompt labels, run HRR binding, compute the sensitivity ratio. If ratio > 1.0, the measure is artifactual by construction.

**Pass criterion**: Sensitivity ratio ≈ 1.0 under fake labels, significantly > 1.0 under real labels.

#### V2. Comparison to Simple Baselines

Compute per-prompt density matrices and block-diagonal construction. Compare the information gained to HRR-bound density matrices. If the simpler approach gives the same or better discrimination with less noise, HRR is not earning its complexity.

**Pass criterion**: HRR provides information not recoverable from per-prompt analysis (e.g., interactions between prompt groups that per-prompt analysis misses).

#### V3. Seed Sensitivity Analysis

Run the same comparison (fp32 vs int8) with 10 different random seeds for prompt vectors. Measure variance of each metric across seeds.

**Pass criterion**: Seed-to-seed variance is small relative to the effect sizes of interest (e.g., < 10% of the fp32-vs-int8 trace distance).

#### V4. Ground One Stakeholder Claim

Pick one measure (e.g., effective rank). Show it is stable under resampling (bootstrap confidence intervals). Show it correlates with at least one observable behavior (response diversity measured by distinct n-grams, prompt-following accuracy, etc.).

**Pass criterion**: Significant correlation (p < 0.05) between effective rank and at least one independent measure of model behavior.

### Decision Framework

- If HRR passes V1-V4: proceed with the HRR-enhanced profiling approach.
- If HRR fails V1 (sensitivity is artifactual): abandon HRR binding, pursue prompt-conditional analysis via per-prompt density matrices instead.
- If HRR passes V1 but fails V2 (simpler baselines are equivalent): abandon HRR in favor of the simpler approach.
- If HRR passes V1-V2 but fails V3 (seed-sensitive): investigate whether using prompt SBERT embeddings instead of random vectors as binding vectors improves stability.
- If HRR passes V1-V3 but fails V4 (ungrounded claims): revise the stakeholder framing; report metrics without interpretive claims until grounding is established.

### Open Questions (Revised)

**Superposition vs. mixed state structure:** HRR superposition (s = Σ roleₖ ⊛ fillerₖ) produces a single vector — a *pure state* in the Hilbert space that internally encodes a mixture. When a density matrix is built over N such superposition vectors, the result is a mixed state of composition-encoding pure states. That is a richer object than either a standard mixed state or a standard superposition. What does the eigenspectrum of that object mean for the statistical tests?

**Prompt SBERT vectors as an alternative to random binding vectors:** If random prompt vectors create only categorical scrambling, using the prompt's own SBERT embedding as the binding vector might encode genuine prompt-response semantic relationships. This would make the binding non-arbitrary but introduces a different concern: the binding vector and the response vector live in the same semantic space, and circular convolution of two SBERT vectors has no established interpretation.

**Prompt-conditional analysis without HRR:** The per-prompt density matrix approach avoids all of HRR's issues. The tradeoff is that it requires aggregation across prompts (e.g., averaging per-prompt trace distances), which may miss cross-prompt interactions. Whether such interactions exist and matter is an empirical question.

## Code Sketch

The implementation adds two modules to `model_equality_testing/src/`:

- `hrr.py` — HRR primitives (circular convolution, prompt binding, effective rank)
- `profiling.py` — Stakeholder-facing measures (behavioral profiles, comparison, alignment, stability)

These integrate with the existing codebase by operating on the same `CompletionSample` objects and `embed_sample()` pipeline, then feeding HRR-bound embeddings into the existing density matrix / quantum metrics machinery.

### Module: `hrr.py`

Core HRR operations and prompt-aware embedding construction.

```python
from model_equality_testing.src.hrr import (
    circular_convolution,       # a ⊛ b via FFT
    circular_correlation,       # approximate inverse
    generate_role_vector,       # random N(0, 1/d) vector
    generate_prompt_id_vectors, # seeded per-prompt vectors
    bind_embeddings_to_prompts, # main binding: prompt_vec ⊛ sbert_embedding
    hrr_embed_sample,           # entry point: CompletionSample + embeddings → HRR
    effective_rank,             # exp(von Neumann entropy) of density matrix
    prompt_sensitivity,         # ratio of HRR/SBERT effective ranks
)
```

Key design decisions:

- **Prompt vectors are seeded** (`seed=42` by default) so the same prompt ID always gets the same binding vector. This is critical for permutation tests.
- **Binding is per-completion**: each embedding is convolved with its prompt's role vector individually, producing an (N, d) array that feeds directly into `full_density_matrix()` or `pca_density_matrix()`.
- **No new dependencies**: only NumPy's FFT, which is already available.

### Module: `profiling.py`

Stakeholder-facing behavioral measures built on HRR + quantum metrics.

```python
from model_equality_testing.src.profiling import (
    compute_behavioral_profile,  # full profile for one model
    compare_profiles,            # side-by-side comparison of two models
    alignment_score,             # trace distance from a reference
    semantic_stability,          # sensitivity to prompt paraphrasing
)
```

#### Example: Profile a Single Model

```python
from model_equality_testing.dataset import load_distribution
from model_equality_testing.src.profiling import compute_behavioral_profile

dist = load_distribution(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    prompt_ids={"wikipedia_en": [0, 1, 2, 3, 4]},
    L=500, source="fp32", load_in_unicode=True,
)
sample = dist.sample(n=1000)

profile = compute_behavioral_profile(sample, pca_k=0)
print(profile.summary())
# Effective rank (SBERT):  12.3
# Effective rank (HRR):   18.7
# Prompt sensitivity:     1.52
# Von Neumann entropy:    2.5100
# Top-3 eigenvalues:      45.2% of trace
# Top-5 eigenvalues:      61.8% of trace
# Top-10 eigenvalues:     82.1% of trace
```

Interpretation: This model uses ~12 distinct semantic modes, but ~19 when accounting for prompt-response structure (sensitivity 1.52 > 1.0 → the model genuinely differentiates across prompts).

#### Example: Compare Two Models for an Application

```python
from model_equality_testing.src.profiling import compare_profiles

dist_a = load_distribution(model="meta-llama/Meta-Llama-3-8B-Instruct", ...)
dist_b = load_distribution(model="mistralai/Mistral-7B-Instruct-v0.3", ...)
sample_a = dist_a.sample(n=1000)
sample_b = dist_b.sample(n=1000)

results = compare_profiles(
    sample_a, sample_b,
    label1="Llama-3-8B", label2="Mistral-7B",
    pca_k=0,
)

print(results["Llama-3-8B"])
# {'effective_rank': 12.3, 'hrr_effective_rank': 18.7,
#  'prompt_sensitivity': 1.52, 'von_neumann_entropy': 2.51}

print(results["Mistral-7B"])
# {'effective_rank': 8.1, 'hrr_effective_rank': 9.2,
#  'prompt_sensitivity': 1.14, 'von_neumann_entropy': 2.09}

print(results["comparison"])
# {'sbert_trace_distance': 0.35, 'sbert_vn_divergence': 0.42,
#  'hrr_trace_distance': 0.41, 'hrr_vn_divergence': 0.38}
```

Stakeholder summary: "Llama-3-8B uses a broader repertoire of response patterns (effective rank 12 vs 8) and differentiates more across prompts (sensitivity 1.52 vs 1.14). For applications needing varied, prompt-sensitive responses, Llama-3-8B is the better fit."

#### Example: Alignment to a Reference

```python
from model_equality_testing.src.profiling import alignment_score

# reference_sample could be from a gold-standard model or human-written responses
scores = alignment_score(candidate_sample, reference_sample, pca_k=0)

print(scores)
# {'sbert_alignment': 0.12, 'hrr_alignment': 0.15}
```

Lower trace distance = closer to the reference distribution. The HRR alignment captures whether the model matches the reference's *prompt-response coupling pattern*, not just its aggregate semantics.

#### Example: Semantic Stability Under Paraphrasing

```python
from model_equality_testing.src.profiling import semantic_stability

# sample_orig: completions from original prompts
# sample_para: completions from paraphrased versions of same prompts
stability = semantic_stability(sample_orig, sample_para, pca_k=0)

print(stability)
# {'trace_distance': 0.05, 'vn_divergence': 0.02}
```

Low values mean the model captures intent rather than surface form — it is robust to prompt phrasing.

### Integration with Existing Permutation Tests

The HRR-bound embeddings can be used with the existing `multi_quantum_permutation_test` by pre-computing HRR embeddings and passing them via the `_precomputed_embeddings` parameter. The prompt vectors must be generated once and held fixed across all permutations.

### Data Flow

```
CompletionSample
    │
    ├─ decode_sample_to_strings() → texts
    │       │
    │       └─ SBERT.encode() → embeddings (N, 768)
    │               │
    │               ├─ [existing path] → full_density_matrix() → ρ_SBERT
    │               │
    │               └─ [new HRR path]
    │                       │
    │                       ├─ generate_prompt_id_vectors() → {pid: role_vec}
    │                       │
    │                       └─ bind_embeddings_to_prompts() → HRR embeddings (N, 768)
    │                               │
    │                               └─ full_density_matrix() → ρ_HRR
    │
    └─ .prompt_sample → prompt IDs (used by HRR binding)
```

## References

- Plate, T. A. (1995). Holographic reduced representations. *IEEE Transactions on Neural Networks*, 6(3), 623–641.
- Nielsen, M. A., & Chuang, I. L. (2010). *Quantum Computation and Quantum Information*. Cambridge University Press.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019*.
- Kanerva, P. (2009). Hyperdimensional computing: An introduction to computing in distributed representation with high-dimensional random vectors. *Cognitive Computation*, 1(2), 139–159.
