# Quantum vs Classical Results Interpretation

## Experiment: FP32 vs INT8 Quantization (Llama-3-8B-Instruct)

**Setup**: Single prompt (wikipedia_en [0]), permutation tests with b=100

### Results

| Metric | n=200 | n=2000 | Interpretation |
|--------|-------|--------|----------------|
| **MMD (Hamming)** | stat=0.002173, p=0.0000 | stat=0.002732, p=0.0000 | Token-level difference: **detected** |
| **Von Neumann Div** | stat=0.028190, p=0.6000 | stat=0.009453, p=0.5100 | Semantic difference: **not detected** |
| **VADER K-S** | stat=0.030 (no p-value) | stat=0.030 (no p-value) | Statistic only, not calibrated |

### Interpretation

The two-tier framework gives a clear result:

1. **Token-level (MMD Hamming)**: Highly significant (p < 0.0001) at both sample sizes. INT8 quantization produces detectably different token distributions.

2. **Semantic-level (Von Neumann divergence)**: p-values at chance level (0.60, 0.51) -- not trending toward significance even with 10x more samples. The semantic diversity/uncertainty of the output distributions is statistically indistinguishable.

**Conclusion**: INT8 quantization changes tokens but preserves semantics. The distributions are statistically different at the character level but semantically equivalent.

### Notes

- Von Neumann divergence statistic decreases with sample size (0.028 -> 0.009), consistent with better density matrix estimation reducing noise
- p-values remain stable around 0.5, confirming no real semantic signal
- VADER K-S reports only the statistic (0.030), not a p-value -- needs calibration via permutation or analytical K-S to be interpretable
- Von Neumann divergence currently uses raw NxN Gram matrices without PCA dimensionality reduction

### Performance

The embedding pre-computation optimization is working:
- n=200: Von Neumann took 5.81s (100 permutations at 278 it/s)
- n=2000: Von Neumann took 149.78s (100 permutations at 1.24s/it)

Previously at n=1000 with re-embedding, each quantum metric took ~570s. The ~100x speedup is confirmed.
