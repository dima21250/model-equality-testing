# Critique: Semantic Axes Implementation Plan

**Document being critiqued**: `SEMANTIC-AXES-IMPLEMENTATION-PLAN.md`

**Bottom line**: The plan is a solid starting point with clean separation of concerns, but it has gaps in statistical methodology, underspecified behavior for edge cases, an incomplete serialization design, and missing consideration of axis quality validation. Several design choices need justification or revision.

---

## What the Plan Gets Right

✅ **Single new file, no changes to existing code** — clean, low-risk addition  
✅ **Reuses `EmbeddingModel` from `embeddings.py`** — consistent SBERT usage  
✅ **Follows `_precomputed_embeddings` pattern** — avoids redundant embedding  
✅ **Interpretation-only framing** — doesn't try to replace PCA or density matrices  
✅ **Phased implementation** — core → interpretation → serialization is a sensible order  
✅ **No new dependencies** — scipy and sentence-transformers already in the project  

---

## Detailed Critiques

### 1. Axis Direction Convention Is Ambiguous

**The plan says**: pole_a is "negative direction," pole_b is "positive direction." Axis vector points from A to B.

**Problem**: Which pole is A and which is B?

- `load_axis_from_jsonl(pole_a_path="professionalism.jsonl", pole_b_path="casualness.jsonl")`
- This makes professionalism = negative, casualness = positive
- So a "professional" text gets a **negative** score
- Is that intuitive? Stakeholders might expect "high professionalism" to be positive

**The convention is arbitrary but must be documented clearly.** The user will trip over this if the sign convention isn't immediately obvious from the output.

**Recommendations**:
- Document the convention explicitly in the dataclass docstring
- In `summary()`, always name both poles: "Sample A is +0.54 toward casualness (away from professionalism)" rather than just "Δ = +0.54"
- Consider: should the user specify which pole is "positive" in the config, independent of argument order?

---

### 2. Centroid Computation Is Naive

**The plan says**: `centroid = mean(embeddings, axis=0)`

**Problems**:

#### 2a. Outlier sensitivity
- Arithmetic mean is sensitive to outliers
- One bad corpus generation (e.g., LLM hallucinates off-topic) shifts the centroid
- Semantic-pole's convergence mode mitigates this, but fixed-count mode doesn't

**Alternative**: Use geometric median (medoid) or trimmed mean. But the arithmetic mean is standard practice in the embedding bias literature (Bolukbasi et al. 2016, Caliskan et al. 2017), so it's defensible.

**Recommendation**: Keep arithmetic mean but add a note about outlier sensitivity. Optionally add a `centroid_method` parameter for future experimentation.

#### 2b. Should embeddings be L2-normalized before computing centroids?

- SBERT embeddings from `all-mpnet-base-v2` are already normalized by default
- But this is model-dependent. Some models don't normalize
- The plan doesn't address whether to normalize before centroid computation

**Recommendation**: After embedding, explicitly L2-normalize each vector before computing the centroid. This ensures cosine-similarity semantics regardless of the SBERT model. Document this choice.

#### 2c. Centroid of normalized vectors vs normalization of centroid

- If you normalize each embedding, then compute the mean, the centroid is NOT on the unit sphere (it's inside the sphere, shorter than unit length)
- The axis vector is then normalized to unit length
- This is fine for the axis direction, but the centroid itself isn't a "representative embedding"
- Doesn't affect correctness, but worth noting

---

### 3. Axis Quality Validation Is Missing

**The plan has no validation step for axis quality.** You compute the axis and trust it.

**What could go wrong**:

#### 3a. Poles don't separate well
- If the two pole corpora have overlapping SBERT embeddings, the axis vector is noisy
- Centroid distance between poles could be tiny
- The axis would project everything near zero (no discriminating power)

**Recommendation**: After computing the axis, report:
- **Inter-pole distance**: `||centroid_b - centroid_a||` — how far apart are the poles?
- **Intra-pole variance**: Average distance of embeddings from their centroid — how tight is each cluster?
- **Separation ratio**: inter-pole distance / mean(intra-pole variance) — analogous to Fisher's discriminant ratio. Higher = better axis.

Store these in `metadata` and optionally warn if separation is low.

#### 3b. Corpus too small
- With 10 samples per pole, the centroid is unstable
- With 1000 samples, it's robust
- The plan doesn't mention minimum corpus size

**Recommendation**: Warn if either pole has fewer than ~50 samples. Mention that convergence-mode corpora from semantic-pole are preferred.

#### 3c. Axis stability
- Different random subsets of the corpus should produce similar axes
- The plan doesn't include bootstrap confidence intervals for the axis direction

**Recommendation**: Optionally compute bootstrap CI for the centroid and axis vector direction. This is expensive (re-embed subsets), so make it opt-in. But the user should at least be able to check axis stability.

---

### 4. Statistical Methodology Gaps

#### 4a. Multiple comparisons problem

**The plan computes t-tests for each axis independently.** With 10 axes, you get 10 p-values. At α=0.05, you expect 0.5 false positives by chance.

**The plan doesn't address multiple comparisons correction.**

**Recommendation**: Apply Bonferroni correction or Benjamini-Hochberg FDR. Report both raw and corrected p-values. Mention the number of comparisons in the summary.

#### 4b. Independence assumption of t-test

**The t-test assumes independence of observations.** But LLM completions from the same prompt may be correlated (if using multiple completions per prompt).

**The plan doesn't address this.** The `CompletionSample` has a `prompt_sample` attribute tracking which prompt each completion belongs to.

**Severity**: Moderate. Correlated observations inflate the t-statistic (more false positives). The degree depends on how many completions per prompt and how variable the prompt effect is.

**Recommendations**:
- Document the independence assumption
- For rigorous analysis, consider a mixed-effects model or cluster-robust standard errors (treating prompt as a cluster)
- Alternatively, use a permutation test on axis projections (shuffle sample labels, recompute delta, build null distribution) — this is non-parametric and doesn't assume independence

#### 4c. Cohen's d pooled standard deviation

**The plan uses pooled standard deviation for Cohen's d.** This assumes equal variances in both groups (homoscedasticity).

**If variances differ** (e.g., one model is more variable than another on an axis), Glass's Δ or Hedges' g may be more appropriate.

**Recommendation**: Use Hedges' g (bias-corrected Cohen's d) instead of raw Cohen's d. The correction is small but standard:
```
g = d × (1 - 3/(4(n_a + n_b) - 9))
```

#### 4d. Wasserstein distance interpretation

**The plan includes Wasserstein distance per axis but doesn't discuss what the values mean.** Unlike Cohen's d (which has conventional thresholds: 0.2 small, 0.5 medium, 0.8 large), Wasserstein distance is in the units of the axis projection scores, which are dot products in SBERT space. These units are not inherently interpretable.

**Recommendation**: Either provide context for interpreting Wasserstein values (e.g., compare to inter-pole distance) or deprioritize it in favor of Cohen's d, which has well-established effect size conventions.

---

### 5. Serialization Design Is Underspecified

#### 5a. npz format is fragile

**The plan uses `.npz` for serialization.** This works but has issues:
- No schema versioning — if you add fields later, old files break
- Metadata stored as a JSON string inside an ndarray is awkward
- Not human-inspectable (binary format)

**Recommendation**: Use a clearer structure:
- Store a `version` field (integer, start at 1)
- Arrays: `axis_vector`, `pole_a_centroid`, `pole_b_centroid`
- Metadata: JSON string with `name`, `pole_a_name`, `pole_b_name`, `embedding_model`, `embedding_dim`, and freeform `metadata`
- On load, check version and handle migration

#### 5b. Single axis vs multiple axes

**The plan has `save_axes(axes, path)` saving a list.** But `load_axis_from_jsonl()` returns a single axis.

**Inconsistency**: The serialization API saves/loads lists, but the construction API produces singles. Should there be `save_axis()` (singular) too?

**Recommendation**: Support both:
- `save_axis(axis, path)` / `load_axis(path)` for single axes
- `save_axes(axes, path)` / `load_axes(path)` for collections
- Or: always save/load lists (a single axis is a list of length 1)

#### 5c. When to serialize

**The plan doesn't discuss when the user should serialize.** Embedding 400 texts takes ~10-30 seconds with SBERT. This is noticeable but not terrible for interactive use.

**Recommendation**: Make serialization optional but recommended. The `load_axis_from_jsonl()` function works without it. Mention in docs that for repeated analysis, save axes after first computation.

---

### 6. `load_axes_from_config` Has Design Issues

#### 6a. Pole pairing assumption

**The plan says**: "Pair poles (first = pole_a, second = pole_b)."

**Problem**: What if the config has 4 poles? (e.g., professionalism, casualness, formality, informality)

- Does it pair (1,2) and (3,4)?
- Does it pair all combinations?
- The semantic-pole config format has a flat `poles` array with no explicit pairing

**The config format doesn't encode which poles form an axis.** It's designed for corpus generation, not axis definition.

**Recommendation**: Either:
1. **Require exactly 2 poles per config** — document that each config file defines one axis with two poles. This is the simplest and matches existing configs like `professionalism-axis.json`.
2. **Add an explicit pairing mechanism** — e.g., a `pairs` field in the config. But this changes the semantic-pole format, which the user may not want.
3. **Drop `load_axes_from_config` entirely** — just use `load_axis_from_jsonl` with explicit paths. The config loading is a convenience that may not be worth the complexity.

Option 1 is probably best: one config = one axis = two poles.

#### 6b. Path resolution

**The plan says**: "Resolve output_file paths relative to config file location."

**Problem**: In the actual semantic-pole configs, `output_file` paths are like `"../output/professionalism.jsonl"` — relative to the project file. But if the user copies the config to a different directory, the paths break.

**Recommendation**: Resolve relative to the config file's directory (as stated), but warn if the resolved path doesn't exist. Provide a clear error message with the resolved absolute path.

---

### 7. The Summary Format Needs Work

#### 7a. Sign interpretation

**The summary says**: "Sample A is more toward casualness (+0.54)"

**Problem**: Does +0.54 mean Sample A scored +0.54 on the axis, or that the delta (A - B) is +0.54? The summary conflates the two.

If delta = mean_a - mean_b = +0.54, that means Sample A scores higher (more toward pole_b). But "more toward casualness" only makes sense if pole_b IS casualness.

**Recommendation**: Be explicit:
```
professionalism ← → casualness:  Δ = +0.54 (A more casual)  d = 0.82  p < 0.001
```

This shows both poles, the delta direction, and a plain-English interpretation.

#### 7b. Significance threshold

**The summary uses p > 0.05 as the threshold for "no significant difference."** This is conventional but arbitrary. With many samples, even tiny effects become significant. Cohen's d is more informative.

**Recommendation**: Report both, and in the summary, highlight axes where BOTH p < 0.05 AND |d| > 0.2 (small effect). This filters out statistically significant but practically meaningless differences.

#### 7c. Raw scores are uninterpretable

**Projection scores are dot products in SBERT space.** A score of +0.54 has no inherent meaning. Is that a lot? A little?

**Recommendation**: Provide context by reporting scores relative to the inter-pole distance:
```
normalized_score = projection_score / ||centroid_b - centroid_a||
```
A normalized score of 0.0 = at pole A centroid, 1.0 = at pole B centroid, 0.5 = midpoint. This is more interpretable than raw dot products.

---

### 8. Edge Cases Not Addressed

#### 8a. Empty or very short texts
- `embed_sample()` already handles empty texts (zero vectors)
- But what if ALL texts are empty? The centroid is the zero vector. The axis vector is undefined.
- What if only a few texts are empty? They pull the centroid toward the origin.

**Recommendation**: Filter out empty texts (and their zero-vector embeddings) before computing centroids. Warn if a significant fraction are empty.

#### 8b. Identical pole corpora
- If pole_a and pole_b have the same text (or very similar embeddings), the centroid difference is near zero
- `axis_vector / ||axis_vector||` would divide by near-zero
- Result: numerically unstable axis

**Recommendation**: Check `||centroid_b - centroid_a||` and raise an error if it's below a threshold (e.g., 1e-6). This catches the "poles don't separate" case from critique #3.

#### 8c. SBERT model mismatch
- Axes computed with `all-mpnet-base-v2` are incompatible with embeddings from `all-MiniLM-L6-v2`
- The plan mentions dimension validation but not model identity validation

**Recommendation**: Store `embedding_model` in the axis. When projecting, check that the model name matches. If dimensions match but model names differ, warn (same dimension doesn't guarantee compatible embedding spaces).

#### 8d. Very large corpora
- Embedding 10,000 texts takes minutes
- The plan doesn't discuss progress reporting or memory management

**Recommendation**: Pass `show_progress_bar=True` when embedding large corpora. Consider batching the centroid computation if memory is tight (but for typical corpus sizes of 100-1000, this isn't needed).

---

### 9. Missing: Axis Orthogonality Analysis

**The plan treats axes as independent.** But if professionalism and formality are correlated (which they likely are), the axes are NOT orthogonal.

**Consequences**:
- A model that's "more professional" will also show up as "more formal" — not because it's independently more formal, but because the axes overlap
- The per-axis interpretation overstates the number of independent differences

**Recommendation**: After loading multiple axes, compute the cosine similarity matrix between axis vectors:
```python
def axis_correlation_matrix(axes: List[SemanticAxis]) -> np.ndarray:
    vectors = np.array([axis.axis_vector for axis in axes])
    return vectors @ vectors.T  # cosine sim (unit vectors)
```

Report this alongside interpretation results. If two axes correlate above 0.7, warn that they measure overlapping dimensions.

Optionally, offer Gram-Schmidt orthogonalization — but this destroys interpretability (orthogonalized axes are no longer aligned with the original poles), so it should be opt-in and the trade-off documented.

---

### 10. Missing: Confidence Intervals for Means

**The plan reports mean scores but no confidence intervals.** The standard error of the mean is:
```
SEM = std / sqrt(n)
```

**With n=100, SEM is 10% of std.** With n=1000, it's 3%. The confidence interval tells the user how much to trust the mean.

**Recommendation**: Add `sem_a`, `sem_b` (standard error of mean) to `AxisProjectionResult`. Optionally compute bootstrap CIs. At minimum, the summary should show CIs:
```
professionalism ← → casualness:
  Sample A: 0.32 ± 0.04 (95% CI)
  Sample B: -0.22 ± 0.05
  Δ = +0.54  d = 0.82  p < 0.001
```

---

### 11. Missing: Per-Prompt Analysis

**The plan projects all completions onto axes and computes aggregate statistics.** But completions from different prompts may behave differently on an axis.

**Example**: A model might be professional on business-related prompts but casual on creative prompts. The aggregate "professionalism = +0.3" hides this prompt-level variation.

**The existing `CompletionSample` tracks which prompt each completion belongs to** (via `prompt_sample`). This information is available but unused.

**Recommendation**: Optionally support per-prompt analysis:
- Group completions by prompt
- Compute per-prompt mean projections
- Report: overall mean, per-prompt means, variance across prompts
- This connects to the "prompt sensitivity" concept from the quantum metrics work

This is a stretch goal, not a Phase 1 requirement. But the data structure should accommodate it (don't discard prompt labels during projection).

---

### 12. Missing: Visualization Support

**The plan produces text summaries but no plots.** For notebook-friendly analysis, users will want:

- **Per-axis distribution plots**: Overlapping histograms or KDE plots of projections for Sample A vs Sample B on each axis
- **Radar/spider plots**: Multi-axis profile comparison (axes as spokes, mean projections as radii)
- **Axis correlation heatmap**: Cosine similarity between axis vectors

**Recommendation**: Add a `plot` parameter or separate plotting functions. Even a minimal `plot_axis_distributions(interpretation, axis_name)` using matplotlib would be valuable. But this can be Phase 4 (after core functionality works).

---

### 13. Integration Example Has Import Path Issues

**The plan shows**:
```python
from model_equality_testing.src.embeddings import embed_sample
from model_equality_testing.src.quantum_metrics import ...
from model_equality_testing.src.semantic_axes import ...
```

**Problem**: The existing codebase uses relative imports within the package (e.g., `from .embeddings import EmbeddingModel`). External users import from the top-level package. Check how `__init__.py` exports modules.

**The existing `__init__.py` exports**: algorithm, dataset, distribution, utils, pvalue, tests. It does NOT export embeddings or quantum_metrics directly.

**Recommendation**: Either:
1. Add `semantic_axes` to `__init__.py` exports
2. Or document the correct import path for external users
3. The integration example should use whatever import pattern the existing codebase uses

---

### 14. The `interpret_difference` API Takes Raw Lists

**The plan has**: `interpret_difference(embeddings_a, embeddings_b, axes)` where `axes` is a `List[SemanticAxis]`.

**But earlier it defines `SemanticAxesSet`** in the data structures discussion. The actual function signature should accept `SemanticAxesSet` or `List[SemanticAxis]` — pick one.

**Recommendation**: Accept `List[SemanticAxis]` (simpler). Drop `SemanticAxesSet` unless it adds functionality beyond a list (e.g., correlation analysis, model validation). A plain list is sufficient for Phase 1.

If `SemanticAxesSet` stays, it should be the input type for `interpret_difference()` to enforce model consistency.

---

### 15. Missing: What Happens When Detection Shows No Difference?

**The plan assumes quantum metrics find a difference that needs interpreting.** But what if trace distance is near zero?

**The user will still want to run semantic axes** — to confirm that models are equivalent across all interpretable dimensions.

**This works fine with the current design** (semantic interpretation doesn't require a prior detection result). But the plan should explicitly state: "Semantic axes can be used independently of quantum metrics for profiling a single model or comparing models regardless of quantum detection results."

---

### 16. Missing: Single-Model Profiling

**The plan focuses on two-sample comparison.** But semantic axes are also useful for profiling a single model: "Model A's outputs are high-professionalism, medium-formality, low-creativity."

**This requires only**: Project embeddings onto axes, compute means and standard deviations. No comparison needed.

**Recommendation**: Add a `profile_sample(embeddings, axes)` function that returns per-axis mean, std, and median for a single sample. This is simpler than interpretation and equally useful.

---

## Summary of Issues

### Must Fix (Before Implementation)

1. ❌ **Axis direction convention** — document clearly, improve summary format (Critique #1)
2. ❌ **Multiple comparisons** — apply Bonferroni or FDR correction (Critique #4a)
3. ❌ **Pole pairing in config loader** — require exactly 2 poles or drop `load_axes_from_config` (Critique #6a)
4. ❌ **Edge case: near-zero inter-pole distance** — check and error (Critique #8b)
5. ❌ **Model mismatch validation** — check model name, not just dimension (Critique #8c)
6. ❌ **Serialization versioning** — add version field to npz format (Critique #5a)

### Should Fix (High Value)

7. ⚠️ **Axis quality metrics** — inter-pole distance, intra-pole variance, separation ratio (Critique #3a)
8. ⚠️ **Normalize scores** — divide by inter-pole distance for interpretable [0,1] scale (Critique #7c)
9. ⚠️ **Confidence intervals** — add SEM to results (Critique #10)
10. ⚠️ **Axis correlation matrix** — warn about non-orthogonal axes (Critique #9)
11. ⚠️ **L2-normalize embeddings** before centroid computation (Critique #2b)

### Nice to Have (Phase 2+)

12. ⚠️ **Single-model profiling** — `profile_sample()` function (Critique #16)
13. ⚠️ **Per-prompt analysis** — group by prompt, report prompt-level variation (Critique #11)
14. ⚠️ **Visualization** — distribution plots, radar plots (Critique #12)
15. ⚠️ **Bootstrap CIs for axis stability** — opt-in validation (Critique #3c)
16. ⚠️ **Hedges' g** instead of raw Cohen's d (Critique #4c)
17. ⚠️ **Permutation test** alternative to t-test for non-independent observations (Critique #4b)

### Minor/Cosmetic

18. ⚠️ **Import path consistency** — match existing codebase patterns (Critique #13)
19. ⚠️ **API consistency** — `List[SemanticAxis]` vs `SemanticAxesSet` — pick one (Critique #14)
20. ⚠️ **Document independent use** — axes work without prior quantum detection (Critique #15)

---

## Conclusion

**The plan is architecturally sound** — single new file, clean separation of concerns, reuse of existing embedding infrastructure, no changes to quantum metrics code.

**The main gaps are**:
1. **Statistical rigor** — multiple comparisons, independence assumptions, effect size conventions
2. **Axis validation** — no quality checks on the computed axes
3. **Interpretability of output** — raw dot products are unintuitive; normalize to inter-pole distance
4. **Edge cases** — near-zero separation, model mismatch, empty corpora

**Recommended changes before implementation**:
- Add axis quality metrics (separation ratio) to `SemanticAxis.metadata`
- Add multiple comparisons correction to `SemanticInterpretation`
- Normalize projection scores by inter-pole distance for interpretability
- Add model name validation (not just dimension check)
- Improve summary format to show both pole names and plain-English direction
- Handle edge cases (near-zero axis, empty texts, model mismatch)
- Drop `SemanticAxesSet` in favor of plain `List[SemanticAxis]` for simplicity

With these fixes, the plan becomes a robust, defensible implementation for semantic axes as an interpretation layer.
