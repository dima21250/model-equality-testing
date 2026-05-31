# Code Critique: HRR Implementation

This document provides a thorough critique of the `hrr.py` and `profiling.py` modules, examining correctness, performance, design, and integration issues.

---

## `hrr.py` Critique

### Correctness Issues

#### 1. **Missing Dimension Validation**

```python
def circular_convolution(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.real(ifft(fft(a) * fft(b)))
```

**Problem**: No check that `a` and `b` have the same shape. If they don't, `fft(a) * fft(b)` will broadcast incorrectly or raise a cryptic error.

**Fix**: Add shape validation:
```python
if a.shape != b.shape:
    raise ValueError(f"Shape mismatch: a{a.shape} != b{b.shape}")
if a.ndim != 1:
    raise ValueError(f"Expected 1D vectors, got {a.ndim}D arrays")
```

#### 2. **No Norm Preservation Check**

The docstring claims circular convolution "approximately preserves norms," but this is only true when binding vectors are N(0, 1/√d). There's no verification that the actual norm distortion is acceptable.

**Fix**: Add a debug mode that logs norm ratios:
```python
if __debug__:
    norm_a, norm_b, norm_result = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(result)
    expected = norm_a * norm_b
    if abs(norm_result - expected) / expected > 0.1:
        logging.warning(f"Norm distortion: expected {expected:.3f}, got {norm_result:.3f}")
```

#### 3. **`circular_correlation` Is Unused**

This function is defined but never called anywhere in the codebase. If retrieval is not being tested, it suggests the binding is one-way only — which undermines the claim that HRR encodes compositional structure (retrieval is how you verify the structure is there).

**Decision**: Either remove it (YAGNI), or add a unit test that verifies `circular_correlation(a, circular_convolution(a, b))` is close to `b` with bounded error.

#### 4. **Seeded RNG Does Not Guarantee Cross-Session Stability**

```python
def generate_prompt_id_vectors(prompt_ids, d, seed=42):
    rng = np.random.default_rng(seed)
    unique_ids = np.unique(prompt_ids)
    return {pid: generate_role_vector(d, rng) for pid in unique_ids}
```

**Problem**: The order of `unique_ids` affects which random vectors are assigned to which prompts. `np.unique()` sorts by default, so this is actually stable, but it's fragile: if `np.unique` ever changes behavior or if `prompt_ids` is passed as a set instead of an array, the mapping changes.

**Fix**: Explicitly iterate in sorted order and document the dependency:
```python
# CRITICAL: iterate in sorted order so seed produces consistent mappings
for pid in sorted(unique_ids):
    prompt_vectors[pid] = generate_role_vector(d, rng)
```

#### 5. **Type Annotation Lies: `sample` Is Not Typed**

```python
def hrr_embed_sample(sample, embeddings: np.ndarray, seed: int = 42) -> np.ndarray:
```

`sample` has no type annotation but the code assumes it has a `.prompt_sample.numpy()` method (i.e., it's a `CompletionSample`). This breaks static analysis and makes the function contract unclear.

**Fix**:
```python
from model_equality_testing.distribution import CompletionSample

def hrr_embed_sample(
    sample: CompletionSample,
    embeddings: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
```

But this introduces a circular import risk if `hrr.py` is imported early. Better: use `TYPE_CHECKING`:
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from model_equality_testing.distribution import CompletionSample

def hrr_embed_sample(sample: 'CompletionSample', ...):
```

---

### Performance Issues

#### 6. **O(N) Loop in `bind_embeddings_to_prompts`**

```python
for i in range(N):
    pid = int(prompt_ids[i])
    bound[i] = circular_convolution(prompt_vectors[pid], embeddings[i])
```

This is an explicit Python loop calling FFT N times. For N=1000, d=768, this is ~1000 FFTs. With batching, this could be reduced to one FFT per unique prompt ID.

**Vectorized alternative**:
```python
unique_pids = np.unique(prompt_ids)
for pid in unique_pids:
    mask = prompt_ids == pid
    # FFT the prompt vector once
    prompt_fft = fft(prompt_vectors[pid])
    # Vectorized FFT over all embeddings with this prompt
    emb_fft = fft(embeddings[mask], axis=-1)
    bound[mask] = np.real(ifft(prompt_fft * emb_fft, axis=-1))
```

This reduces ~1000 FFTs to ~5 FFTs (for 5 unique prompts) + vectorized operations. Speedup: ~100x for large N.

#### 7. **Redundant PCA Fitting in `prompt_sensitivity`**

```python
if pca_k > 0:
    pca = fit_pca(embeddings, k=pca_k)
    rho_sbert = pca_density_matrix(embeddings, pca)
    pca_hrr = fit_pca(hrr_emb, k=pca_k)  # <-- separate PCA
    rho_hrr = pca_density_matrix(hrr_emb, pca_hrr)
```

This fits PCA twice, independently, so the two density matrices live in **different subspaces**. The ratio of their effective ranks is meaningless — you're comparing apples to oranges in different coordinate systems.

If the intent is to measure prompt sensitivity, both should be in the same PCA basis (fit on concatenated embeddings). Otherwise the ratio reflects PCA basis differences, not prompt structure.

**Fix**: Fit one PCA on the combined pool (like `compare_profiles` does), or document explicitly that they're in different spaces.

---

### Design Issues

#### 8. **`effective_rank` Duplicates `von_neumann_entropy` Logic**

```python
def effective_rank(rho):
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > epsilon]
    entropy = -np.sum(eigenvalues * np.log(eigenvalues))
    return np.exp(entropy)
```

This recomputes von Neumann entropy, which already exists in `quantum_metrics.py`. Duplication means:
- Two copies of the same numerical logic (risk of divergence)
- If `von_neumann_entropy` gets a bug fix, `effective_rank` doesn't

**Fix**: Reuse the existing function:
```python
from .quantum_metrics import von_neumann_entropy

def effective_rank(rho: np.ndarray) -> float:
    return np.exp(von_neumann_entropy(rho))
```

#### 9. **Hardcoded Epsilon in Multiple Places**

The threshold `1e-10` appears in:
- `effective_rank` (line 167)
- `prompt_sensitivity` indirectly via `effective_rank`
- `profiling.py` functions (lines 99, 100, 112, etc.)

If numerical stability ever requires changing this (e.g., for different precision floats), you have to hunt down all instances.

**Fix**: Define a module-level constant:
```python
EIGENVALUE_THRESHOLD = 1e-10
```

Or better: inherit it from `quantum_metrics.py` if defined there.

#### 10. **No Input Validation in `bind_embeddings_to_prompts`**

Assumes:
- All prompt IDs in `prompt_ids` exist as keys in `prompt_vectors`
- `len(prompt_ids) == len(embeddings)`

If either is violated, you get a KeyError or shape mismatch deep in the FFT.

**Fix**: Add checks:
```python
if len(embeddings) != len(prompt_ids):
    raise ValueError(f"Length mismatch: {len(embeddings)} embeddings, {len(prompt_ids)} prompt IDs")
missing = set(prompt_ids) - set(prompt_vectors.keys())
if missing:
    raise KeyError(f"Prompt IDs {missing} not in prompt_vectors")
```

---

## `profiling.py` Critique

### Correctness Issues

#### 11. **PCA Basis Inconsistency in `compute_behavioral_profile`**

```python
if pca_k > 0:
    pca = fit_pca(embeddings, k=pca_k)
    rho = pca_density_matrix(embeddings, pca)
    pca_hrr = fit_pca(hrr_emb, k=pca_k)  # Different basis!
    rho_hrr = pca_density_matrix(hrr_emb, pca_hrr)
```

Same issue as in `hrr.py`: the two density matrices live in different PCA subspaces. The `prompt_sensitivity_ratio` is comparing effective ranks in non-comparable spaces.

**Fix**: Either:
- Fit one PCA on concatenated `[embeddings, hrr_emb]`
- Document explicitly that they're in different spaces and explain why that's valid (if it is)

#### 12. **Duplicated Eigendecomposition Logic Everywhere**

The pattern appears in 4+ places:
```python
eigs = np.sort(np.linalg.eigvalsh(rho))[::-1]
eigs_positive = eigs[eigs > 1e-10]
entropy = -np.sum(eigs_positive * np.log(eigs_positive))
eff_rank = np.exp(entropy)
```

This is the definition of `effective_rank`, but it's re-implemented inline instead of calling the function. Violates DRY.

**Fix**: Refactor to call `effective_rank(rho)` from `hrr.py`.

#### 13. **`_quick_profile` Helper Is Quadruply Nested**

In `compare_profiles`, there's a nested function `_quick_profile` defined at line 202. It's only called twice, but it contains the same duplicated eigendecomposition logic. This is a sign the abstraction is wrong.

**Fix**: Make `_quick_profile` a module-level function, or better yet, replace its body with calls to existing functions (`effective_rank`, `von_neumann_entropy`).

#### 14. **Prompt Vectors Must Be Shared for Fair Comparison**

`compare_profiles` correctly generates shared prompt vectors:
```python
all_prompt_ids = np.concatenate([sample1.prompt_sample.numpy(), sample2.prompt_sample.numpy()])
prompt_vectors = generate_prompt_id_vectors(all_prompt_ids, d, seed=hrr_seed)
```

But `alignment_score` does the same thing. This logic should be factored out — it's a critical invariant that if two samples share prompt IDs, they must use the same binding vectors.

**Fix**: Create a helper:
```python
def get_shared_prompt_vectors(sample1, sample2, d, seed):
    all_ids = np.concatenate([sample1.prompt_sample.numpy(), sample2.prompt_sample.numpy()])
    return generate_prompt_id_vectors(all_ids, d, seed=seed)
```

---

### Design Issues

#### 15. **BehavioralProfile Dataclass Mixes Data and Presentation**

```python
def summary(self) -> str:
    lines = [f"Effective rank (SBERT):  {self.effective_rank:.1f}", ...]
    return "\n".join(lines)
```

This hardcodes formatting in the data object. If you want JSON output or a different format, you can't reuse the dataclass.

**Better design**: Keep `BehavioralProfile` pure data, add a separate `format_profile(profile: BehavioralProfile) -> str` function.

#### 16. **Misleading Function Name: `alignment_score`**

The function returns a **dictionary** with keys `"sbert_alignment"` and `"hrr_alignment"`. The name suggests it returns a single scalar score.

**Fix**: Rename to `compute_alignment_scores` (plural) or `measure_alignment`.

#### 17. **Boolean Flag for Optional Computation**

```python
def alignment_score(..., use_hrr: bool = True):
```

This is an anti-pattern. If `use_hrr=False`, the function still imports HRR modules and generates prompt vectors, only to not use them. The conditional should be outside the function, not inside.

**Better design**:
```python
def sbert_alignment(candidate, reference, ...): ...
def hrr_alignment(candidate, reference, ...): ...
def full_alignment(candidate, reference, ...):
    return {
        "sbert": sbert_alignment(...),
        "hrr": hrr_alignment(...),
    }
```

Now callers can choose which to call, and the functions are pure.

#### 18. **Inconsistent Return Types**

- `compute_behavioral_profile` returns a `BehavioralProfile` dataclass
- `compare_profiles` returns a raw `Dict`
- `alignment_score` returns a raw `Dict[str, float]`
- `semantic_stability` returns a raw `Dict[str, float]`

There's no consistent API. Dataclasses are used for one function and plain dicts for the rest.

**Fix**: Either use dataclasses everywhere (type-safe, self-documenting) or plain dicts everywhere (simpler, but less safe). Mixed is the worst of both.

---

### Performance Issues

#### 19. **Embedding Computed Twice in `compare_profiles`**

`compare_profiles` calls `embed_sample` for both samples, then `_quick_profile` recomputes eigendecompositions. But if the caller already has embeddings (e.g., from a prior call to `compute_behavioral_profile`), they're re-computed.

**Fix**: Accept optional `precomputed_embeddings` like the quantum test functions do.

#### 20. **No Caching for Repeated Calls**

If you call `compare_profiles` twice on the same samples (e.g., with different seeds to test stability), embeddings are regenerated both times. SBERT inference is the bottleneck (~15-30 seconds for N=1000), and it's deterministic.

**Fix**: Either:
- Document that callers should cache embeddings themselves
- Use `functools.lru_cache` (careful: sample objects might not be hashable)
- Accept embeddings as optional parameters

---

### Integration Issues

#### 21. **Circular Import Risk**

`profiling.py` imports from `hrr.py`, which imports from `quantum_metrics.py` (inside `prompt_sensitivity`), which might import from `distribution.py`. If any of these imports `profiling.py` in the future, you have a cycle.

**Fix**: Keep imports lazy (inside functions) for cross-module dependencies, or restructure so dependencies flow one-way:
```
distribution.py → embeddings.py → quantum_metrics.py → hrr.py → profiling.py
```

#### 22. **No Tests**

Neither module has unit tests. Critical properties to test:
- `circular_convolution(a, b)` preserves approximate norm
- `circular_correlation` retrieves approximately correctly
- Effective rank is deterministic under resampling (bootstrap test)
- Seed stability: same seed → same prompt vectors
- Validation V1 (null test): fake prompt labels → sensitivity ≈ 1.0

**Fix**: Add a `tests/test_hrr.py` and `tests/test_profiling.py` with at least:
- Norm preservation test
- Retrieval error test
- Seed determinism test
- Fake label sensitivity test (V1)

#### 23. **No Logging**

When HRR binding is slow (N=10,000), there's no progress indicator. When PCA is fitted, there's no indication of what basis was chosen. When effective ranks differ wildly, there's no warning.

**Fix**: Add strategic `logging.info` calls:
```python
logging.info(f"Binding {N} embeddings to {len(unique_pids)} unique prompts...")
logging.info(f"Effective ranks: SBERT={rank_sbert:.2f}, HRR={rank_hrr:.2f}, ratio={ratio:.2f}")
```

---

## Missing Functionality

#### 24. **No Validation Experiment Implementations**

The critique document defines V1-V4 validation experiments, but none are implemented. These should exist as scripts or test functions.

**Fix**: Add `experiments/validate_hrr.py` with:
```python
def validate_v1_null_test(sample): ...
def validate_v2_baseline_comparison(sample1, sample2): ...
def validate_v3_seed_sensitivity(sample1, sample2, n_seeds=10): ...
def validate_v4_grounding(sample): ...
```

#### 24. **No Per-Prompt Baseline**

The critique suggests per-prompt density matrices as a simpler alternative. This isn't implemented for comparison.

**Fix**: Add to `profiling.py`:
```python
def per_prompt_analysis(sample, embedding_model="all-mpnet-base-v2"):
    """Compute per-prompt density matrices and effective ranks."""
    embeddings = embed_sample(sample, ...)
    unique_prompts = np.unique(sample.prompt_sample.numpy())
    results = {}
    for pid in unique_prompts:
        mask = sample.prompt_sample.numpy() == pid
        emb_p = embeddings[mask]
        rho_p = full_density_matrix(emb_p)
        results[pid] = {
            "effective_rank": effective_rank(rho_p),
            "entropy": von_neumann_entropy(rho_p),
        }
    return results
```

---

## Documentation Issues

#### 26. **Docstrings Claim Unverified Properties**

`prompt_sensitivity` docstring:
```
Interpretation:
    > 1.0: Model differentiates responses across prompts (prompt-sensitive)
```

This interpretation is unverified. The critique argues it might be artifactual. The docstring should say:
```
Interpretation (WARNING: requires validation via null test V1):
    > 1.0: MAY indicate model differentiates responses, OR may be artifact
```

#### 27. **No Examples That Run**

The markdown examples show fabricated outputs:
```python
# Effective rank (SBERT):  12.3
```

These numbers are made up. A reader can't verify the code works.

**Fix**: Add a minimal runnable example in the docstring or a separate `examples/hrr_demo.py` that uses real data from the test set.

---

## Summary: Code Quality Tiers

| Category | Issue Count | Severity |
|----------|-------------|----------|
| Correctness | 5 | **High** — will cause silent wrong results or crashes |
| Performance | 5 | Medium — works but 10-100x slower than necessary |
| Design | 10 | Medium — makes future changes harder, violates DRY/SOLID |
| Integration | 5 | Low-Medium — testing, logging, imports |
| Missing | 3 | High — validations V1-V4 don't exist, which blocks the entire research claim |

**Most critical fixes** (blocking research validity):
1. Implement V1-V4 validation experiments (issue #24)
2. Fix PCA basis inconsistency (#11, #7) — currently comparing incomparable spaces
3. Add V1 null test to verify sensitivity isn't artifactual (#2 from critique doc)
4. Vectorize `bind_embeddings_to_prompts` (#6) — currently unusably slow for N > 5000

**Next tier** (correctness without which results are suspect):
5. Reuse `von_neumann_entropy` instead of duplicating (#8)
6. Add input validation (#1, #4, #10)
7. Add unit tests for norm preservation, retrieval, determinism (#22)

**Nice-to-have** (maintainability):
8. Fix type annotations (#5)
9. Refactor duplicated eigendecomposition logic (#12)
10. Consistent return types (dataclasses vs dicts) (#18)
