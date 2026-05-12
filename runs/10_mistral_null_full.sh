#!/bin/bash
# Context 5b: Mistral-7B FP32 vs FP32 null baseline, full space (no PCA)
# Tests that same model/source produces non-significant results in full-dimensional mode
# Model: mistralai/Mistral-7B-Instruct-v0.3
# Dataset: wikipedia_en, prompt 0
# Density matrix: full space (no PCA, pca_k=0)
# Samples: 1000, Permutations: 5000

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTFILE="runs/results/10_mistral_null_full_${TIMESTAMP}.txt"
mkdir -p runs/results

echo "Run started: $(date)" | tee "$OUTFILE"
echo "Script: $0" | tee -a "$OUTFILE"
echo "" | tee -a "$OUTFILE"

python quantum_comparison.py \
    --model_a mistralai/Mistral-7B-Instruct-v0.3 \
    --model_b mistralai/Mistral-7B-Instruct-v0.3 \
    --source_a fp32 \
    --source_b fp32 \
    --prompts 0 \
    --dataset wikipedia_en \
    --samples 1000 \
    --b 5000 \
    --pca_k 0 \
    2>&1 | tee -a "$OUTFILE"

echo "" | tee -a "$OUTFILE"
echo "Run finished: $(date)" | tee -a "$OUTFILE"
