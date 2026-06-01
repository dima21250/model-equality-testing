# Critique: Semantic Axes Implementation Plan v2

**Document being critiqued**: `SEMANTIC-AXES-IMPLEMENTATION-PLAN-v2.md`

**Bottom line**: The v2 plan addressed the major statistical and edge-case issues from the v1 critique (Welch's t-test, Hedges' g, FDR correction, pole separation validation, L2 normalization). What remains are mostly design-level questions about scope, naming, and a few subtle technical issues that could bite during implementation.

---

## What v2 Fixed Well

✅ **Welch's t-test** instead of Student's (no equal-variance assumption)  
✅ **Hedges' g** instead of raw Cohen's d (bias-corrected)  
✅ **Benjamini-Hochberg FDR** across axes (multiple comparisons)  
✅ **L2-normalize embeddings** before centroid computation  
✅ **`pole_distance` validation** with error on near-zero separation  
✅ **SEM** in results  
✅ **`negative_pole`/`positive_pole`** naming (clear convention)  
✅ **Summary format** shows both poles with direction  
✅ **`axis_correlations()`** with warning on high correlation  
✅ **Model name validation** (warn on mismatch)  

---

## Remaining Issues

### 1. L2 Normalization Creates a Consistency Question

**The plan says**: "L2-normalize each embedding before computing centroid."

**But**: When the user runs quantum metrics, `embed_sample()` returns **raw** SBERT embeddings (not L2-normalized). The density matrix construction in `quantum_metrics.py` has its own normalization:
- `full_density_matrix()` L2-normalizes internally
- `pca_density_matrix()` does NOT normalize — it just projects

**The question**: When the user passes precomputed embeddings to `interpret_difference()`, are those embeddings raw or normalized?

```python
emb_a = embed_sample(sample_a)  # raw SBERT embeddings
# ... quantum metrics use emb_a ...
# ... semantic axes also receive emb_a ...
```

If `embed_sample()` returns raw embeddings (which it does), and `interpret_difference()` receives those raw embeddings, but the axis was built from L2-normalized embeddings — the projection scores will be inconsistent.

**The fix options**:
1. **Normalize inside `project_onto_axis()`**: Always L2-normalize incoming embeddings before projecting. This ensures consistency regardless of what the caller passes. But it means the caller can't control normalization.
2. **Normalize inside `load_axis_from_jsonl()` AND `project_onto_axis()`**: Both sides normalize. Consistent.
3. **Don't normalize at all**: Use raw SBERT embeddings everywhere. Simpler. `all-mpnet-base-v2` already returns near-unit-norm vectors (cosine similarity is the default similarity metric for this model), so the practical difference is small.

**Recommendation**: Option 2 is safest. Normalize in both places. Document that all operations happen in cosine-similarity space. The plan should state explicitly whether `project_onto_axis()` normalizes incoming embeddings.

---

### 2. The `cohens_d` Field Name Is Misleading

**The plan says**: The field is named `cohens_d` but contains Hedges' g.

This will confuse anyone reading the code or the output. Cohen's d and Hedges' g are related but distinct. Naming the field `cohens_d` while computing Hedges' g is a naming lie.

**Recommendation**: Either:
- Name the field `hedges_g` (accurate)
- Name the field `effect_size` (generic, correct for either)
- Name the field `cohens_d` and actually compute Cohen's d (simpler, the bias correction is tiny for n > 30)

Pick one and be consistent. The summary output says "d = 0.82" — if the value is Hedges' g, it should say "g = 0.82" or just "effect = 0.82".

---

### 3. Hedges' g Formula Is Underspecified

**The plan says**: "Hedges' g (bias-corrected Cohen's d) using Welch's formula (does not assume equal variance)."

**Problem**: There are two issues being conflated:

1. **Cohen's d vs Hedges' g**: This is about bias correction. Hedges' g multiplies Cohen's d by a correction factor `J = 1 - 3/(4(n_a + n_b) - 9)`. This is a small adjustment for small samples.

2. **Pooled SD vs separate SDs**: This is about the equal-variance assumption. Cohen's d traditionally uses pooled SD (assumes equal variance). The "Welch" analog would use a different denominator.

**The standard formulas**:
- **Cohen's d (pooled)**: `d = (mean_a - mean_b) / s_pooled` where `s_pooled = sqrt(((n_a-1)*s_a² + (n_b-1)*s_b²) / (n_a + n_b - 2))`
- **Glass's Δ**: Uses only one group's SD as denominator
- **Cohen's d (average)**: `d = (mean_a - mean_b) / sqrt((s_a² + s_b²) / 2)` — doesn't assume equal variance
- **Hedges' g**: Any of the above × correction factor J

**Recommendation**: Since we're already using Welch's t-test (not assuming equal variance), use the **average SD** denominator for the effect size: `sqrt((s_a² + s_b²) / 2)`. Then apply the Hedges correction if desired. Specify this formula explicitly in the plan so there's no ambiguity during implementation.

---

### 4. BH-FDR Implementation Detail

**The plan says**: "implement directly (sort p-values, multiply by n_tests/rank)."

**The BH procedure is slightly more involved than this**:

```python
def benjamini_hochberg(pvalues):
    n = len(pvalues)
    sorted_indices = np.argsort(pvalues)
    sorted_pvalues = pvalues[sorted_indices]
    corrected = sorted_pvalues * n / np.arange(1, n + 1)
    # Enforce monotonicity: corrected[i] >= corrected[i-1]
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    # Cap at 1.0
    corrected = np.minimum(corrected, 1.0)
    # Unsort
    result = np.empty_like(corrected)
    result[sorted_indices] = corrected
    return result
```

**The monotonicity enforcement step is critical** — without it, you can get corrected p-values that are smaller than the raw p-values for some ranks. The plan mentions "verify BH correction is monotonic" in verification but doesn't mention the enforcement step in the implementation.

**Recommendation**: Include the monotonicity enforcement in the plan description, or reference a specific implementation (e.g., `statsmodels.stats.multitest.multipletests` with `method='fdr_bh'`). If implementing from scratch, the enforce-then-cap pattern above is the standard.

---

### 5. What Does `project_onto_axis` Do With the Embeddings Model Name?

**The plan says**: "Validates embedding model name matches (warn on mismatch)."

**But `project_onto_axis` takes `embeddings: np.ndarray`** — a raw numpy array. It has no way to know what model produced those embeddings. The model name is lost by the time you have a numpy array.

**The only function that knows the model name is `interpret_samples()`**, which takes CompletionSample objects and calls `embed_sample()`.

**Options**:
1. **Add an `embedding_model` parameter to `project_onto_axis()`** — but this is awkward (the caller has to pass a string alongside the array)
2. **Only validate in `interpret_samples()`** — where the model name is known
3. **Remove model validation from `project_onto_axis()`** — it's the wrong abstraction level for this check

**Recommendation**: Option 3. Validate model name only in `interpret_samples()` (the high-level function). `project_onto_axis()` is a low-level function that takes numpy arrays — it should validate dimensions only.

---

### 6. The `metadata` Dict Is a Bag of Untyped Data

**The plan says** `metadata: dict` stores "paths, corpus sizes, timestamp, pole_distance."

**Problem**: `pole_distance` is already a typed field on `SemanticAxis`. If it's also in `metadata`, it's duplicated. What exactly goes in `metadata`?

**Recommendation**: Define the metadata schema explicitly:
```python
metadata: dict  # Contains:
    # "negative_pole_path": str — file path to negative pole JSONL
    # "positive_pole_path": str — file path to positive pole JSONL
    # "negative_pole_samples": int — number of texts in negative pole corpus
    # "positive_pole_samples": int — number of texts in positive pole corpus
    # "created_at": str — ISO 8601 timestamp
```

Don't duplicate `pole_distance` in metadata since it's already a first-class field. Consider whether metadata should be a typed dataclass instead of a dict — but for v1 a dict with documented keys is fine.

---

### 7. Serialization Format Doesn't Handle Multiple Axes Well

**The plan says**: `save_axes(axes: List, path)` saves a list to `.npz`.

**Problem**: How do you store N axes of different metadata in a single `.npz`? The natural approach would be:

```python
np.savez(path,
    axis_vectors=np.stack([a.axis_vector for a in axes]),     # (K, d)
    neg_centroids=np.stack([a.negative_centroid for a in axes]),  # (K, d)
    pos_centroids=np.stack([a.positive_centroid for a in axes]),  # (K, d)
    metadata=json.dumps([...])  # JSON string with all per-axis metadata
)
```

This works if all axes have the same `embedding_dim` (which they should if they use the same model). But it conflates K axes into matrices, making it impossible to load a single axis from a multi-axis file.

**Alternative**: Use one `.npz` file per axis. `save_axes()` takes a directory path and saves `axis_name.npz` for each. `load_axes(dir_path)` loads all `.npz` files from the directory.

**Recommendation**: Either approach works. The plan should specify which one. Per-axis files are simpler and allow selective loading. A single file is more portable. Pick one.

---

### 8. `axis_quality()` Is Thin

**The plan says** `axis_quality(axis) → dict` returns `pole_distance` and corpus sizes.

**These are already on the `SemanticAxis` object** (`pole_distance` as a field, corpus sizes in `metadata`). The function adds no value — the user can read `axis.pole_distance` directly.

**What would be actually useful**:
- **Intra-pole variance**: Average pairwise distance within each pole corpus. But this requires access to the raw embeddings, which aren't stored on the axis (only centroids are).
- **Separation ratio**: inter-pole distance / mean intra-pole std. Same problem — needs raw embeddings.

**Recommendation**: Either:
1. **Compute quality metrics during `load_axis_from_jsonl()`** and store them in metadata (since the raw embeddings are available at that point). Then `axis_quality()` just returns them.
2. **Drop `axis_quality()` entirely** — the user reads `axis.pole_distance` and `axis.metadata['negative_pole_samples']` directly. Less API surface, same information.

Option 1 is better if you want separation ratio. Option 2 is simpler.

---

### 9. The Plan Doesn't Address What `interpret_samples` Does With the `axes` Parameter

**`interpret_samples(sample_a, sample_b, axes, _precomputed_embeddings=None)`**

**What type is `axes`?** The plan alternates between `List[SemanticAxis]` and mentioning `SemanticAxesSet` (from the v1 plan, which was dropped). In the v2 plan, `axes` appears to be `List[SemanticAxis]` everywhere, but it's never stated explicitly.

**Recommendation**: State the type. `axes: List[SemanticAxis]` is fine. No need for a wrapper class.

---

### 10. The Summary Doesn't Include Sample Labels

**The summary says**: "Sample A is more toward casualness."

**But the user didn't call them "Sample A" and "Sample B."** They're "fp32" and "int8", or "Llama-3-8B on Amazon" and "Llama-3-8B on Azure."

**The plan doesn't accept sample labels.** The interpretation functions take embedding arrays, which have no names.

**Recommendation**: Add optional `label_a` and `label_b` parameters to `interpret_difference()` and `interpret_samples()`, defaulting to "Sample A" / "Sample B". The summary then shows:

```
fp32 ← → int8 on professionalism ← → casualness:  Δ = +0.54
  fp32 is more toward casualness
```

This is a small but significant usability improvement.

---

### 11. The Summary Direction Logic Is Ambiguous

**When delta is positive**: "Sample A is more toward [positive_pole]"

**When delta is negative**: "Sample A is more toward [negative_pole]"

**But delta = mean_a - mean_b.** So:
- delta > 0 means A scored higher (more positive = more toward positive_pole) ✓
- delta < 0 means A scored lower (more negative = more toward negative_pole) ✓

**This is correct**, but the summary could also be stated from B's perspective: "Sample B is more toward [negative_pole]." Which perspective is more intuitive?

**Recommendation**: State from the perspective of the *larger* difference: "fp32 is more professional" rather than "int8 is more casual." Both are true, but the former is often more natural. Or show both perspectives. Either way, document the convention.

---

### 12. No Guidance on Axis Design

**The plan covers loading and using axes but doesn't address which axes to create.** The user needs to:
1. Decide which dimensions matter for their analysis
2. Define poles for each dimension
3. Generate corpora via semantic-pole
4. Load axes into model-equality-testing

**Steps 1-3 happen in the semantic-pole project.** But the plan should at least mention a recommended starter set of axes for LLM evaluation:
- Professionalism ↔ Casualness
- Formality ↔ Informality
- Technical ↔ Layperson
- Conciseness ↔ Verbosity
- Creativity ↔ Literalness

**Recommendation**: Not a code issue. But the plan (or the module docstring) should mention that axis design is the user's responsibility, and point to semantic-pole for axis generation. A brief note about what makes a good axis pair (clear contrast, well-generated corpora, convergence-tested) would be valuable.

---

### 13. No Discussion of Score Magnitude Interpretation

**The v1 critique raised**: "Raw dot products are uninterpretable. A score of +0.54 has no inherent meaning."

**v2 didn't address this.** The plan still reports raw dot products.

**The v1 critique suggested**: Normalize by pole distance so scores are on a [0, 1] scale relative to the inter-pole distance.

**Arguments for raw scores**: They're consistent with SBERT cosine similarity space. Users familiar with embeddings expect this.

**Arguments for normalized scores**: "0.54 out of a pole distance of 0.8" is more informative than "0.54 in SBERT dot-product units." The inter-pole distance provides the natural scale.

**Recommendation**: Report both raw and normalized in `AxisProjectionResult`:
- `mean_a`: raw projection score
- `normalized_mean_a`: `mean_a / pole_distance` — fraction of the way from midpoint to positive pole

Or keep raw only and let the summary format add context by referencing pole distance. Either way, the plan should make a decision.

---

### 14. Verification Test 4 Has a Dependency Problem

**The plan says**: "Load professionalism/casualness corpora from `~/Code/semantic-pole/output/`."

**Problem**: This creates a hard dependency on the semantic-pole project being present at a specific path. If someone else clones model-equality-testing, the test fails.

**Recommendation**: Either:
1. **Copy a small test fixture** (e.g., 20 records per pole) into the model-equality-testing repo under `tests/fixtures/`
2. **Mark the test as integration-only** (skip if semantic-pole path doesn't exist)
3. **Generate synthetic JSONL** in the test setup (random sentences, embed, verify the pipeline works)

Option 1 or 3 for unit tests. Option 2 for integration tests that use real corpora.

---

### 15. Missing: What Happens With a Single Axis?

**When `axes` has length 1**, BH-FDR correction is trivial (corrected p-value = raw p-value). This is correct behavior, but it should be handled explicitly rather than as a degenerate case of the general algorithm.

**No code issue** — just verify the BH implementation handles K=1 gracefully.

---

### 16. Missing: Thread Safety of `EmbeddingModel`

**`load_axis_from_jsonl()`** creates an `EmbeddingModel` instance (which uses `@lru_cache` for model loading). If multiple axes are being loaded concurrently (e.g., in a notebook with async), the LRU cache is shared.

**Not a real problem for this use case** — axes are loaded sequentially. But if the user loads axes in a parallel loop, they'll share the SBERT model (which is fine, since SBERT inference is stateless).

**No action needed** — just noting for completeness.

---

### 17. The Integration Example Has a Subtle Bug

**The plan shows**:
```python
from model_equality_testing.src.embeddings import embed_sample
```

**But `embed_sample` takes a `CompletionSample`** — which stores completions as unicode codepoints in a tensor. The user's workflow would be:

```python
# Load distribution → draw sample → embed → quantum detect → semantic interpret
dist_a = load_distribution(model=..., source="fp32", ...)
sample_a = dist_a.draw_completion_sample(n=100)
emb_a = embed_sample(sample_a)
```

**The example skips the distribution loading step.** This is fine for illustration, but a user unfamiliar with the codebase would be confused about where `sample_a` comes from.

**Recommendation**: Include the full pipeline from `load_distribution` through to `interpret_difference()` in the example, or at minimum add a comment: "# sample_a is a CompletionSample from load_distribution()".

---

## Summary of Issues

### Must Fix (Before Implementation)

1. ❌ **L2 normalization consistency** — specify whether `project_onto_axis()` normalizes incoming embeddings (Critique #1)
2. ❌ **Field naming** — `cohens_d` should be `hedges_g` or `effect_size` if it contains Hedges' g (Critique #2)
3. ❌ **Effect size formula** — specify the exact denominator (pooled, average, or Glass's) (Critique #3)
4. ❌ **BH monotonicity enforcement** — include `np.minimum.accumulate` step in description (Critique #4)
5. ❌ **Model validation scope** — remove from `project_onto_axis()`, keep only in `interpret_samples()` (Critique #5)

### Should Fix (High Value)

6. ⚠️ **Sample labels** — add `label_a`/`label_b` parameters for readable summaries (Critique #10)
7. ⚠️ **Metadata schema** — document what goes in the dict, don't duplicate `pole_distance` (Critique #6)
8. ⚠️ **Serialization granularity** — decide: one file per axis vs one file for all (Critique #7)
9. ⚠️ **`axis_quality()` value** — either compute intra-pole variance during loading or drop the function (Critique #8)
10. ⚠️ **Test fixture** — don't depend on `~/Code/semantic-pole/` for tests (Critique #14)

### Minor / Cosmetic

11. ⚠️ **Type annotation** — explicitly state `axes: List[SemanticAxis]` in all signatures (Critique #9)
12. ⚠️ **Summary direction** — document which perspective the direction statement takes (Critique #11)
13. ⚠️ **Score normalization** — decide whether to report raw or normalized-by-pole-distance scores (Critique #13)
14. ⚠️ **Integration example completeness** — show full pipeline from `load_distribution` (Critique #17)
15. ⚠️ **Axis design guidance** — mention in module docstring, point to semantic-pole (Critique #12)

---

## Comparison: v1 vs v2

| Issue | v1 Status | v2 Status |
|-------|-----------|-----------|
| Equal-variance t-test | ❌ Used Student's t | ✅ Welch's t-test |
| Effect size bias | ❌ Raw Cohen's d | ✅ Hedges' g (but naming is wrong) |
| Multiple comparisons | ❌ Not addressed | ✅ BH-FDR correction |
| L2 normalization | ❌ Not addressed | ✅ Added (but consistency unclear) |
| Pole separation validation | ❌ Not addressed | ✅ Error if < 1e-6 |
| SEM in results | ❌ Not addressed | ✅ Added |
| Axis correlation warnings | ❌ Not addressed | ✅ `axis_correlations()` |
| Model name validation | ❌ Not addressed | ✅ Added (but wrong scope) |
| Direction convention | ❌ Ambiguous | ✅ negative_pole / positive_pole |
| Summary format | ❌ Unclear | ✅ Shows both poles |
| Score normalization | ❌ Not addressed | ❌ Still not addressed |
| Sample labels | ❌ Not addressed | ❌ Still not addressed |
| Serialization versioning | ❌ Not addressed | ✅ Added version field |
| Axis quality metrics | ❌ Not addressed | ⚠️ Thin (just reads metadata) |
| Test fixtures | ❌ Hard dependency | ❌ Still hard dependency |

**v2 fixed 10 of 16 v1 issues.** 5 remain unaddressed, 1 was partially addressed.

---

## Conclusion

**The v2 plan is substantially improved** over v1. The statistical methodology is sound (Welch's t, Hedges' g, BH-FDR), edge cases are handled (pole separation, empty texts, model mismatch), and the architecture is clean (single file, no changes to existing code).

**The remaining issues are mostly design polish**, not fundamental problems:
- Normalization consistency between axis construction and projection
- Naming accuracy (`cohens_d` vs `hedges_g`)
- Effect size formula specification
- BH implementation detail
- Sample labels for usability

**None of these are blockers.** They can be resolved during implementation. The plan is ready to implement with these notes as guidance.

**Overall assessment**: Good plan. Fix the 5 "must fix" items during implementation, incorporate the "should fix" items where practical, and defer the rest to iteration.
