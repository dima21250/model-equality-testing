# Plan for Removing HRR Code Sketches from Codebase

**Goal**: Remove HRR-related code from the codebase while preserving useful quantum metrics and documentation of the exploration.

**Rationale**: Based on our comprehensive critique, HRR for prompt binding is not a valuable approach. Semantic axes are the primary contribution, and quantum metrics should be applied to full SBERT space or semantic axis projections, not HRR-bound embeddings.

---

## Files to Remove

### Code Files (Delete Completely)

1. **`model_equality_testing/src/hrr.py`**
   - Contains: `circular_convolution`, `generate_prompt_id_vectors`, `bind_embeddings_to_prompts`, `hrr_embed_sample`, `effective_rank`, `prompt_sensitivity`
   - Reason: HRR prompt binding is fundamentally flawed (per critiques)
   - Action: **DELETE**

2. **`model_equality_testing/src/profiling.py`**
   - Contains: HRR-based behavioral profiling (depends entirely on hrr.py)
   - Imports: `from .hrr import hrr_embed_sample, effective_rank, prompt_sensitivity`
   - Reason: Built on top of HRR, which we're removing
   - Action: **DELETE** (will be replaced with semantic axis profiling later if needed)

---

## Files to Keep (Documentation of Exploration)

These markdown files document our exploration and critique of HRR. They're valuable for showing the research process and why we chose not to pursue HRR.

### Keep in Repository

1. **`HRR-quantum-metrics.md`** - Original framework exploration + 10 critiques
2. **`HRR-CODE-CRITIQUE.md`** - 27 implementation issues
3. **`HRR-JUDGE-LABELS-ALTERNATIVE.md`** - Alternative use case (judge labels)
4. **`HRR-JUDGE-LABELS-CRITIQUE.md`** - Critique of judge labels use case
5. **`SEMANTIC-AXES-VS-HRR-COMPARISON.md`** - Why semantic axes is superior
6. **`SEMANTIC-AXES-QUANTUM-INTEGRATION.md`** - Integration proposal (Option 2 uses HRR, but Options 1 & 3 don't)
7. **`SEMANTIC-AXES-QUANTUM-INTEGRATION-CRITIQUE.md`** - Critique of psychometric density matrices
8. **`PHD-THESIS-CONTRIBUTION-ANALYSIS.md`** - Thesis framing analysis
9. **`QUANTUM-METRICS-HONEST-ASSESSMENT.md`** - Honest assessment of quantum metrics value
10. **`SEMANTIC-AXES-AS-PRIMARY-CONTRIBUTION.md`** - Why semantic axes matter more

**Reason to keep**: These documents show:
- We explored HRR thoroughly
- We critically evaluated it (not blindly accepting)
- We have clear reasons for not using it
- The research process was rigorous

This is valuable for:
- PhD defense (shows critical thinking)
- Future reference (why didn't we use HRR?)
- Other researchers (saves them from repeating this exploration)

**Optional**: Create a `docs/explorations/` directory and move HRR docs there to signal they're archived research, not active work.

---

## Files to Preserve and Keep Active

### Keep as Core Framework

1. **`model_equality_testing/src/quantum_metrics.py`**
   - Contains: `full_density_matrix`, `pca_density_matrix`, `fit_pca`, `trace_distance`, `von_neumann_entropy`, `effective_rank`
   - Reason: These are useful for full SBERT space analysis (not HRR)
   - Action: **KEEP** (core quantum metrics, not HRR-specific)

2. **`model_equality_testing/src/embeddings.py`**
   - Contains: SBERT embedding utilities
   - Reason: Core functionality for semantic space analysis
   - Action: **KEEP**

3. **All other existing code** (algorithm.py, tests.py, distribution.py, etc.)
   - Reason: Core model equality testing framework
   - Action: **KEEP**

---

## Verification Steps

### 1. Check for Dependencies

Before deleting, verify no other code depends on HRR modules:

```bash
# Search for imports of hrr or profiling
grep -r "from.*hrr import\|import.*hrr" model_equality_testing/
grep -r "from.*profiling import\|import.*profiling" model_equality_testing/

# Search for function calls to HRR functions
grep -r "circular_convolution\|hrr_embed_sample\|bind_embeddings_to_prompts\|generate_prompt_id_vectors\|prompt_sensitivity" model_equality_testing/ experiments/ *.py *.ipynb

# Search for imports in notebooks
find . -name "*.ipynb" -exec grep -l "hrr\|profiling" {} \;
```

If anything outside `hrr.py` and `profiling.py` imports these modules, those imports need to be removed first.

---

### 2. Check for References in Documentation

```bash
# Check if CLAUDE.md or README references HRR code
grep -i "hrr\|profiling.py" CLAUDE.md README.md
```

If CLAUDE.md mentions HRR code, remove those references (they were added as exploratory code sketches).

---

## Execution Plan

### Step 1: Verify No Dependencies
```bash
# Check for imports
grep -r "from.*hrr import\|import.*hrr" model_equality_testing/ experiments/
grep -r "from.*profiling import\|import.*profiling" model_equality_testing/ experiments/

# Check for function calls
grep -r "hrr_embed_sample\|bind_embeddings_to_prompts\|prompt_sensitivity" --include="*.py" --exclude-dir=".ipynb_checkpoints" .
```

Expected result: Only `hrr.py` and `profiling.py` should appear.

---

### Step 2: Remove Code Files
```bash
# Delete HRR module
rm model_equality_testing/src/hrr.py

# Delete profiling module (depends on HRR)
rm model_equality_testing/src/profiling.py
```

---

### Step 3: Clean Up Checkpoint Files (if any)
```bash
# Remove any .ipynb_checkpoints containing HRR code
find model_equality_testing/src/.ipynb_checkpoints -name "*hrr*" -delete
find model_equality_testing/src/.ipynb_checkpoints -name "*profiling*" -delete
```

---

### Step 4: Optional - Organize Documentation

Create an archive directory for HRR exploration docs:

```bash
# Create exploration archive directory
mkdir -p docs/explorations/hrr

# Move HRR exploration docs there
mv HRR-quantum-metrics.md docs/explorations/hrr/
mv HRR-CODE-CRITIQUE.md docs/explorations/hrr/
mv HRR-JUDGE-LABELS-ALTERNATIVE.md docs/explorations/hrr/
mv HRR-JUDGE-LABELS-CRITIQUE.md docs/explorations/hrr/
mv SEMANTIC-AXES-VS-HRR-COMPARISON.md docs/explorations/hrr/

# Keep active framework docs in root
# (SEMANTIC-AXES-QUANTUM-INTEGRATION.md and critiques stay in root)
```

Alternative: Keep all docs in root (they're part of the research record).

---

### Step 5: Update CLAUDE.md

Remove any references to `hrr.py` or `profiling.py` from CLAUDE.md.

Check:
```bash
grep -i "hrr.py\|profiling.py" CLAUDE.md
```

If found, edit CLAUDE.md to remove those sections.

---

### Step 6: Verify Clean State

```bash
# Verify files are gone
ls model_equality_testing/src/hrr.py 2>/dev/null && echo "ERROR: hrr.py still exists" || echo "✓ hrr.py deleted"
ls model_equality_testing/src/profiling.py 2>/dev/null && echo "ERROR: profiling.py still exists" || echo "✓ profiling.py deleted"

# Verify no imports remain
grep -r "from.*hrr import\|import.*hrr" model_equality_testing/ && echo "ERROR: HRR imports remain" || echo "✓ No HRR imports"
grep -r "from.*profiling import\|import.*profiling" model_equality_testing/ && echo "ERROR: profiling imports remain" || echo "✓ No profiling imports"

# Verify quantum_metrics.py still exists (we keep this)
ls model_equality_testing/src/quantum_metrics.py && echo "✓ quantum_metrics.py preserved" || echo "ERROR: quantum_metrics.py missing"
```

---

### Step 7: Git Commit

```bash
# Stage deletions
git add -u model_equality_testing/src/hrr.py
git add -u model_equality_testing/src/profiling.py

# Optional: Stage doc reorganization if you moved files
# git add docs/explorations/hrr/

# Commit with clear message
git commit -m "Remove HRR code sketches from codebase

After comprehensive critique (see HRR-CODE-CRITIQUE.md and 
SEMANTIC-AXES-VS-HRR-COMPARISON.md), we've concluded that HRR 
for prompt binding is not a valuable approach.

Removed:
- model_equality_testing/src/hrr.py (HRR binding implementation)
- model_equality_testing/src/profiling.py (HRR-based behavioral profiling)

Preserved:
- quantum_metrics.py (useful for full SBERT space, not HRR-specific)
- All HRR exploration markdown docs (valuable research record)

Rationale: Semantic axes (from semantic-pole framework) are the 
primary contribution for interpretable LLM profiling. Quantum metrics 
should be applied to full SBERT space or semantic axis projections, 
not HRR-bound embeddings.

See SEMANTIC-AXES-AS-PRIMARY-CONTRIBUTION.md for thesis framing."
```

---

## What Remains After Removal

### Core Framework (Active)
- **Quantum metrics**: `quantum_metrics.py` for density matrices on SBERT embeddings
- **Embeddings**: `embeddings.py` for SBERT encoding
- **Model equality testing**: All existing token-space tests (MMD, chi-squared, KS)
- **Dataset utilities**: `dataset.py`, `distribution.py`

### Research Documentation (Preserved)
- **HRR exploration docs**: Record of what we tried and why we rejected it
- **Semantic axes docs**: Framework for interpretable LLM profiling
- **Integration proposals**: Options for combining quantum metrics + semantic axes
- **Honest assessments**: Critical evaluation of quantum metrics value

### Ready for Future Work
- **Semantic axis profiling**: Can be implemented cleanly without HRR
  - Project onto axes → get psychometric vectors
  - Apply quantum metrics to psychometric space (Option 2)
  - Or use classical stats (recommended per critique)
- **Full SBERT analysis**: Apply quantum metrics to 768-dim embeddings (Option 1 or 3)

---

## Summary

**Delete**:
- ❌ `hrr.py` (flawed approach)
- ❌ `profiling.py` (depends on hrr.py)

**Keep**:
- ✅ `quantum_metrics.py` (useful for non-HRR analysis)
- ✅ All markdown docs (research record)
- ✅ All other existing code

**Result**: Clean codebase with quantum metrics ready for semantic axes or full SBERT analysis, with complete documentation of why HRR was explored and rejected.

This aligns with our conclusion: **Semantic axes are the primary contribution**, quantum metrics are a tool choice, and HRR adds complexity without benefit.
