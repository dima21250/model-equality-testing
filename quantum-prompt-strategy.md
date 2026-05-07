# Prompt Strategy for Quantum Metrics

## Question

Should quantum metric experiments use single prompts or multiple prompts?

## Recommendation: Single Prompts

Stick to single prompts for quantum metrics. Run several single-prompt comparisons independently (e.g., prompt 0, then prompt 3, then prompt 10) and check consistency across prompts.

## Rationale

### Single prompt (preferred for quantum metrics)

Each sample responds to the same input, so detected differences are purely in the model's generation behavior -- not confounded by prompt variation. Cleaner signal for measuring quantization/architecture effects.

### Multiple prompts (problematic for quantum metrics)

Mixes two sources of variation:

1. How the model responds differently to different prompts
2. How the model's generation differs between versions (the signal we want)

The MMD tests handle this correctly (kernel value = 0 for completions from different prompts), but the quantum metrics build a single density matrix from all embeddings regardless of prompt. Prompt-driven semantic variation dominates the embedding space and can drown out subtle quantization effects.

### Why this matters given known limitations

INITIAL-RESULTS.md shows von Neumann divergence discrimination already weakens with more data. Adding prompt variation would likely make it worse by introducing semantic diversity that has nothing to do with the distribution shift being tested.

## Approach

- **Quantum metrics**: Run single-prompt experiments. Test multiple prompts separately and compare results for consistency.
- **MMD (Hamming)**: Can use multiple prompts (handles them correctly by design via prompt-conditional kernel). Compare single-prompt vs multi-prompt results to verify p-values are consistent.
- **Interpretation**: If quantum metrics give consistent results across individual prompts, that's stronger evidence than a single multi-prompt pooled result.
