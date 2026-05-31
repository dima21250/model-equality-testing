# HRR Exploration Archive

This directory contains documentation of our exploration of Holographic Reduced Representations (HRR) for LLM behavioral analysis.

**Status**: Explored and rejected as of 2026-05-30

## Summary

We thoroughly explored using HRR for LLM profiling in three use cases:

1. **Prompt binding** (`HRR-quantum-metrics.md`) - Bind completions to prompt IDs via circular convolution
2. **Psychometric profiling** (`SEMANTIC-AXES-VS-HRR-COMPARISON.md`) - Compare HRR to semantic axes approach
3. **Judge labels** (`HRR-JUDGE-LABELS-ALTERNATIVE.md`) - Encode LLM-as-judge labels via HRR

After comprehensive critique, we concluded that **HRR adds complexity without benefit** for all three use cases:

- **Prompt binding**: Encodes group membership, not compositional structure; destroys inner products; prompt sensitivity likely artifactual
- **Psychometric profiling**: Semantic axes (from `semantic-pole` framework) are superior - more interpretable, validated, simpler
- **Judge labels**: Structured metadata (SQL/pandas) is cleaner, faster, exact (no noise), and more interpretable than HRR

## Documents in This Archive

### Core Exploration
- **HRR-quantum-metrics.md** - Initial framework proposal + 10 fundamental critiques + validation plan (V1-V4)
- **HRR-CODE-CRITIQUE.md** - 27 implementation issues in the code sketches

### Alternative Use Cases
- **HRR-JUDGE-LABELS-ALTERNATIVE.md** - Proposal for encoding multi-dimensional judge labels
- **HRR-JUDGE-LABELS-CRITIQUE.md** - 15-point critique showing structured metadata is superior

### Comparative Analysis
- **SEMANTIC-AXES-VS-HRR-COMPARISON.md** - Side-by-side comparison showing semantic axes are better for psychometric profiling

## Key Findings

### What HRR Doesn't Provide
- ❌ Not more interpretable (adds abstraction)
- ❌ Not compositional for our use cases (prompt binding encodes groups, not structure)
- ❌ Not validated (no empirical evidence it works)
- ❌ Not simpler (circular convolution + noise vs direct operations)

### What Works Better
- ✅ **Semantic axes** for interpretable dimensions (professionalism, formality, etc.)
- ✅ **Quantum metrics on full SBERT space** for distributional comparison
- ✅ **Structured metadata** for judge labels (pandas DataFrames, not compressed vectors)
- ✅ **Classical statistics** for low-dimensional interpretable analysis

## Recommended Approach

For LLM behavioral profiling:

1. **Use semantic axes** (from `semantic-pole` framework) for interpretable dimensions
2. **Apply quantum metrics to full SBERT embeddings** for distributional comparison
3. **Use classical statistics** (means, t-tests, effect sizes) for interpretable dimensions
4. **Avoid HRR** unless future validation shows it outperforms simpler baselines (which it doesn't)

## Research Value

This exploration demonstrates:
- Thorough investigation of alternative approaches
- Critical evaluation (not blindly accepting complex methods)
- Clear reasoning for methodological choices
- Intellectual honesty (rejecting our own initial ideas when evidence doesn't support them)

These documents are preserved as part of the research record showing the critical thinking process.

## Related Active Work

- **Semantic-pole framework**: `~/Code/semantic-pole` (corpus generation for interpretable axes)
- **Quantum metrics**: `model_equality_testing/src/quantum_metrics.py` (for full SBERT space)
- **Integration proposals**: See root-level markdown files for semantic axes + quantum metrics integration

---

**Last updated**: 2026-05-30  
**Conclusion**: HRR explored, critiqued, and archived. Focus on semantic axes as primary contribution.
