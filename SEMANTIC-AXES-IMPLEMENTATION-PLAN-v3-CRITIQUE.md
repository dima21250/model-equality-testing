# Critique: Semantic Axes Implementation Plan v3

**Document being critiqued**: `SEMANTIC-AXES-IMPLEMENTATION-PLAN-v3.md`

**Bottom line**: v3 is a solid, implementable plan. The major statistical, normalization, and naming issues from v2 are resolved. What remains are mostly second-order concerns: a few subtle mathematical edge cases, some API ergonomics questions, and practical issues that will surface during implementation. Nothing here is a blocker.

---

## What v3 Fixed From v2

✅ L2 normalization in both axis construction AND projection  
✅ `hedges_g` field name (not `cohens_d`)  
✅ Explicit effect size formula with average-SD denominator  
✅ BH-FDR with monotonicity enforcement shown inline  
✅ Model validation scoped to `interpret_samples()` only  
✅ Sample labels (`label_a`/`label_b`) throughout  
✅ Metadata schema documented, no duplication of `pole_distance`  
✅ One-file-per-axis serialization  
✅ Intra-pole std and separation ratio computed during loading  
✅ Test fixtures synthetic, no hard dependency on semantic-pole  
✅ All function signatures typed  
✅ Summary uses both-pole notation with sample labels  
✅ Full integration example from `load_distribution` through summary  
✅ Axis design guidance in Context section  
✅ K=1 BH passthrough noted  

---

## Remaining Issues

### 1. L2 Normalization of Centroids Creates Non-Unit Centroids

**The plan says**: L2-normalize each embedding, then compute `centroid = mean(normalized_embeddings, axis=0)`.

**Consequence**: The centroid itself is NOT a unit vector. It's the arithmetic mean of points on the unit sphere, which lies *inside* the sphere. Its norm depends on how spread out the pole corpus is:
- Tight cluster → centroid norm ≈ 1.0
- Spread-out cluster → centroid norm << 1.0

**The axis vector is then**: `(centroid_pos - centroid_neg) / ||centroid_pos - centroid_neg||`, which IS a unit vector. So the axis direction is correct.

**But `pole_distance`** = `||centroid_pos - centroid_neg||` is computed from these non-unit centroids. The maximum possible pole distance is 2.0 (antipodal centroids), and the typical range is much smaller because centroids shrink toward the origin.

**Is this a problem?** Not for the axis vector (direction is correct). But `pole_distance` and `separation_ratio` depend on centroid norms. If one pole's corpus is more spread out, its centroid is shorter, which could make `pole_distance` artificially small.

**Severity**: Low. The axis vector (the thing that matters) is unaffected. The quality metrics are somewhat influenced, but they're diagnostic, not operational. Worth documenting that `pole_distance` is in "mean-of-unit-vectors" space, not pure cosine space.

---

### 2. "Intra-Pole Standard Deviation" Definition Is Ambiguous

**The plan says**: "mean L2 distance of each embedding from its centroid."

**This is NOT standard deviation.** Standard deviation is the root-mean-square of deviations:
```
std = sqrt(mean(||x_i - centroid||²))
```

Mean L2 distance is:
```
mean_dist = mean(||x_i - centroid||)
```

These are different (Jensen's inequality: `mean(||.||) ≤ sqrt(mean(||.||²))`).

**The field is named `_intra_std`** but the description says "mean L2 distance." This is a naming mismatch.

**Recommendation**: Pick one:
- Use actual standard deviation (`sqrt(mean(||x_i - c||²))`) and keep the name `_intra_std`
- Use mean L2 distance and rename to `_intra_mean_dist`

Either works for the separation ratio. Just be consistent.

---

### 3. Separation Ratio Denominator Could Be Zero

**The plan says**: `separation_ratio = pole_distance / mean(intra_std_neg, intra_std_pos)`

**If a pole corpus has zero variance** (all embeddings are identical after L2 normalization), `intra_std = 0`. If both poles have zero variance, the denominator is zero.

**When would this happen?**
- Corpus with 1 sample (std undefined, but the plan warns on < 50 samples)
- Corpus where all texts embed to the same vector (unlikely but possible with degenerate prompts)

**Severity**: Very low. The 50-sample warning makes this nearly impossible in practice. But division by zero is a crash, not just bad output.

**Recommendation**: Guard the division: `separation_ratio = pole_distance / max(mean_intra, epsilon)` with `epsilon = 1e-10`.

---

### 4. The Hedges' g J Correction Has a Division-by-Zero Edge Case

**The formula**: `J = 1 - 3 / (4 * (n_a + n_b) - 9)`

**If `n_a + n_b ≤ 2`** (e.g., n_a=1, n_b=1), the denominator `4*2 - 9 = -1` produces `J = 1 - 3/(-1) = 4`, which is nonsensical.

**If `n_a + n_b = 3`**, the denominator is `4*3 - 9 = 3`, so `J = 0`, which zeros out the effect size.

**When would this happen?** Only with tiny samples. The plan warns when corpus size < 50, but sample sizes for interpretation (`n_a`, `n_b`) are different — they're the LLM completion counts, not the pole corpus sizes. A user could call `interpret_difference` with 2 embeddings per group.

**Severity**: Low. Extreme edge case. But `n_a + n_b < 4` produces mathematically invalid J values.

**Recommendation**: Guard: if `n_a + n_b < 4`, set `J = 1.0` (skip bias correction; with tiny samples the correction is unreliable anyway). Or require `n_a ≥ 2` and `n_b ≥ 2`.

---

### 5. The Average-SD Denominator and Welch's t-Test Use Different Assumptions

**Hedges' g denominator**: `s_avg = sqrt((s_a² + s_b²) / 2)` — weights both groups equally regardless of sample size.

**Welch's t-test denominator**: `sqrt(s_a²/n_a + s_b²/n_b)` — weights by sample size.

These are subtly different. When `n_a ≠ n_b`, the t-test accounts for the unequal precision of the two means, but the effect size doesn't.

**Is this a problem?** Not really — effect sizes and test statistics serve different purposes. Cohen's d / Hedges' g intentionally ignores sample size (it's a *population* effect size estimate). The t-test accounts for sample size in its precision estimate. Using different denominators for effect size vs significance is standard practice.

**Severity**: None. This is correct behavior. But it's worth noting that a small Hedges' g with a tiny p-value means "small effect, but we're very confident it's real" (large sample), while a large Hedges' g with a large p-value means "large effect, but we're not confident" (small sample).

---

### 6. Wasserstein Distance Units

**The plan includes Wasserstein distance** but doesn't discuss its units or how to interpret values.

Wasserstein distance on 1D projections is in the same units as the projection scores (dot products after L2 normalization). Unlike Hedges' g (which is standardized and has conventional thresholds), Wasserstein distance has no inherent scale.

**When is Wasserstein more informative than Hedges' g?**
- When distributions are non-Gaussian (multimodal, skewed)
- When the full distributional shape matters, not just the mean shift
- When one distribution has outliers that inflate the standard deviation

**When is it less informative?**
- When distributions are roughly Gaussian (Wasserstein ≈ |mean_a - mean_b| + term from variance difference)
- When the user doesn't know what "0.12 Wasserstein units" means

**Recommendation**: Consider whether Wasserstein actually earns its place. It adds a number to every result that most users won't know how to interpret. If kept, add a brief note in the summary about what large vs small values mean relative to the axis. Or normalize it by pole distance.

---

### 7. `save_axes` Directory Semantics

**The plan says**: `save_axes(axes, directory)` saves each axis as `{directory}/{axis.name}.npz`.

**What if the directory doesn't exist?** Create it? Error?

**What if an axis with the same name already exists in the directory?** Overwrite silently? Error? Warn?

**What if two axes in the list have the same `name`?** The second write overwrites the first.

**Recommendation**: Create the directory if it doesn't exist. Overwrite existing files (this is a save operation, not an append). Validate that axis names are unique in the list before saving (error on duplicates).

---

### 8. `load_axes` Directory Ordering

**The plan says**: `load_axes(directory)` loads all `.npz` files from the directory.

**In what order?** `os.listdir()` returns arbitrary order on some filesystems. If the user expects axes in a specific order (e.g., alphabetical, or the order they were saved), they'll get inconsistent behavior.

**Recommendation**: Sort by filename (alphabetical). Document this. Alternatively, return a dict keyed by axis name instead of a list — but the rest of the API uses lists.

---

### 9. The Integration Example May Have an Import Path Issue

**The example uses**:
```python
from model_equality_testing.src.embeddings import embed_sample
from model_equality_testing.src.semantic_axes import load_axis_from_jsonl, interpret_difference
```

**But the package's `__init__.py`** exports `algorithm`, `dataset`, `distribution`, `utils`, `pvalue`, `tests` — NOT `embeddings` or `quantum_metrics` directly.

**The `.src.` in the import path** is unusual. Most Python packages use `from model_equality_testing.embeddings import ...` (without `.src.`), with `__init__.py` re-exporting.

**Recommendation**: Check how existing code imports these modules (look at notebooks, scripts). Match the convention. If users import via `from model_equality_testing.src.X`, keep that. If they import via `from model_equality_testing.X`, add `semantic_axes` to `__init__.py`.

This is a cosmetic issue — either path works with `pip install -e .` — but inconsistency will confuse users.

---

### 10. No `__repr__` or `__str__` on Data Structures

**`SemanticAxis` and `AxisProjectionResult` are dataclasses.** Python dataclasses auto-generate `__repr__`, but for numpy array fields, the repr is noisy:

```python
SemanticAxis(name='prof-casual', ..., axis_vector=array([0.012, -0.034, ...768 values...]), ...)
```

**In a notebook**, printing an axis dumps hundreds of floats.

**Recommendation**: Override `__repr__` to show a compact representation:
```python
SemanticAxis(name='prof-casual', negative='professionalism', positive='casualness', dim=768, pole_distance=0.83, separation_ratio=2.14)
```

This is a quality-of-life issue, not functional.

---

### 11. No Mechanism to Inspect Raw Projections

**`interpret_difference` returns `AxisProjectionResult` with summary statistics** (means, stds, effect sizes). But the user cannot access the raw (N,) arrays of projection scores.

**Why this matters**: For exploratory analysis in notebooks, users want to:
- Plot histograms of projections
- Identify outlier texts (which completion scored extremely high/low on professionalism?)
- Compute custom statistics beyond what `AxisProjectionResult` provides

**Currently**: The user must call `project_onto_axis()` separately to get raw scores, then also call `interpret_difference()` for statistics. Two calls, two normalizations, two projections.

**Recommendation**: Either:
1. **Add raw projections to `AxisProjectionResult`**: `projections_a: np.ndarray`, `projections_b: np.ndarray`. This increases memory usage but makes the result self-contained.
2. **Return projections alongside interpretation**: `interpret_difference()` returns `(SemanticInterpretation, dict_of_projections)`. Awkward API.
3. **Accept the two-call pattern**: Users call `project_onto_axes()` for raw data and `interpret_difference()` for statistics. Document this.

Option 3 is simplest and avoids storing large arrays in the result dataclass. Option 1 is most convenient. Either is fine — just decide.

---

### 12. Single-Model Profiling Is Mentioned But Not Designed

**The Context says**: "They can also be used independently to profile a single model."

**But no function exists for this.** `interpret_difference()` requires two sets of embeddings. To profile one model, the user must call `project_onto_axes()` and compute their own means/stds.

**Recommendation**: Either:
1. **Add `profile(embeddings, axes, label="Model") → dict`** — returns per-axis mean, std, SEM, median for a single sample. Simple, useful, complements the two-sample comparison.
2. **Remove the claim** from Context if you don't plan to implement it in v1.

Option 2 is fine for now — `project_onto_axes()` gives the raw data, and computing `mean()` and `std()` is trivial. But if you mention the capability, users expect a function for it.

---

### 13. Axis Name Collision in Serialization

**`save_axes(axes, directory)`** saves as `{axis.name}.npz`. If axis names contain characters invalid in filenames (e.g., `/`, `\`, `:`, spaces), the save will fail or create unexpected paths.

**Example**: `axis.name = "professionalism/casualness"` → tries to create `directory/professionalism/casualness.npz`, which is a nested path.

**Recommendation**: Sanitize axis names for filenames: replace non-alphanumeric characters (except `-` and `_`) with `_`. Or use a slug derived from the name.

---

### 14. The Plan Doesn't Discuss Logging

**The plan mentions warnings** (e.g., "Warn if separation_ratio < 1.0", "Warn if corpus has fewer than 50 samples", "Warn on model mismatch").

**How are these warnings emitted?** Options:
- Python `logging.warning()` — standard, configurable, matches `embeddings.py` which uses `logging.info()` and `logging.warning()`
- Python `warnings.warn()` — shows once by default, can be filtered
- Print to stderr — simplest but not configurable

**Recommendation**: Use `logging.warning()` to match the existing pattern in `embeddings.py`. Import `logging` at the top of the module.

---

### 15. What If the JSONL Has No `metadata.concept` Field?

**The plan says**: "Extract pole names from `metadata.concept` in the first record of each file."

**What if the JSONL was generated outside semantic-pole?** The user said JSONL-only, but the specific JSONL schema is semantic-pole's. If someone hand-writes a JSONL file with `{"text": "..."}` and no metadata, the extraction fails.

**Recommendation**: Fall back gracefully: if `metadata.concept` is missing, derive the pole name from the filename (e.g., `professionalism.jsonl` → `"professionalism"`). This makes the function robust to JSONL files that have the `text` field but not the full semantic-pole metadata.

---

### 16. The BH-FDR Code Has a Subtle Type Issue

**The inline code uses**: `np.empty_like(corrected)` to create the result array.

If `pvalues` is a Python list (not a numpy array), `np.argsort` will work (it converts internally), but `np.empty_like` on the intermediate `corrected` array will be fine since it's already numpy by that point.

**However**: The code assumes `pvalues` is indexable by a numpy index array (`result[sorted_indices] = corrected`). If `pvalues` is a list, this fails.

**Recommendation**: Start the BH function with `pvalues = np.asarray(pvalues)` to ensure numpy throughout.

---

## Summary of Issues

### Should Fix (Before or During Implementation)

1. ⚠️ **Intra-std naming mismatch** — "mean L2 distance" ≠ standard deviation; pick one formula and match the name (Critique #2)
2. ⚠️ **Division-by-zero in separation ratio** — guard with epsilon when intra-std is zero (Critique #3)
3. ⚠️ **Hedges' g with tiny samples** — guard J correction when n_a + n_b < 4 (Critique #4)
4. ⚠️ **Filename sanitization** — sanitize axis names before using as filenames (Critique #13)
5. ⚠️ **Logging convention** — use `logging.warning()` to match `embeddings.py` (Critique #14)
6. ⚠️ **Missing `metadata.concept` fallback** — derive pole name from filename if metadata absent (Critique #15)
7. ⚠️ **BH type safety** — `np.asarray(pvalues)` at top of function (Critique #16)

### Nice to Have

8. ⚠️ **`__repr__` for dataclasses** — compact repr that doesn't dump 768-dim arrays (Critique #10)
9. ⚠️ **`load_axes` ordering** — sort by filename for deterministic order (Critique #8)
10. ⚠️ **`save_axes` directory/duplicate handling** — create dir, error on duplicate names (Critique #7)
11. ⚠️ **Import path consistency** — verify against existing codebase import conventions (Critique #9)
12. ⚠️ **Single-model profiling** — either add `profile()` function or remove the claim from Context (Critique #12)

### Informational (No Action Required)

13. ℹ️ **Non-unit centroids** — documented consequence of L2-normalizing before averaging, doesn't affect axis direction (Critique #1)
14. ℹ️ **Different denominators for g vs t** — standard practice, correct behavior (Critique #5)
15. ℹ️ **Wasserstein interpretation** — units are unclear to stakeholders, but the metric is valid (Critique #6)
16. ℹ️ **Raw projection access** — users can call `project_onto_axes()` separately; two-call pattern is acceptable (Critique #11)

---

## Comparison: v1 → v2 → v3

| Issue | v1 | v2 | v3 |
|-------|----|----|-----|
| Equal-variance t-test | ❌ | ✅ Welch's | ✅ |
| Effect size bias | ❌ | ✅ Hedges' g (wrong name) | ✅ (correct name + formula) |
| Multiple comparisons | ❌ | ✅ BH-FDR | ✅ (+ monotonicity) |
| L2 normalization | ❌ | ✅ (consistency unclear) | ✅ (both sides) |
| Pole separation | ❌ | ✅ | ✅ (+ separation ratio) |
| SEM | ❌ | ✅ | ✅ |
| Axis correlations | ❌ | ✅ | ✅ |
| Model validation scope | ❌ | ✅ (wrong scope) | ✅ (correct scope) |
| Direction convention | ❌ | ✅ | ✅ |
| Sample labels | ❌ | ❌ | ✅ |
| Metadata schema | ❌ | ❌ | ✅ (+ quality metrics) |
| Serialization | ❌ | ⚠️ | ✅ (per-axis files) |
| Test fixtures | ❌ | ❌ | ✅ (synthetic) |
| Integration example | ❌ | ❌ | ✅ (full pipeline) |
| Intra-pole std naming | — | — | ⚠️ NEW |
| Division guards | — | — | ⚠️ NEW |
| Filename sanitization | — | — | ⚠️ NEW |

**v3 resolved all 15 issues from v2.** The 7 "should fix" items above are implementation-level details, not design flaws. The 5 "nice to have" items are polish.

---

## Conclusion

**v3 is ready for implementation.** The architecture is clean, the statistical methodology is sound, edge cases are addressed, and the API is well-designed.

The remaining issues (7 "should fix") are all implementable as small guards, fallbacks, or name choices within the code itself — they don't require plan changes. Handle them during implementation:

- Add epsilon guards for division-by-zero cases
- Pick either `sqrt(mean(||.||²))` or `mean(||.||)` for intra-pole dispersion and name accordingly
- Sanitize filenames
- Use `logging.warning()`
- Fall back to filename-derived pole names
- Cast to numpy in BH function

**No further plan revisions are needed.** Proceed to implementation.
