#!/bin/bash
# Context 1: Llama-3-8B FP32 vs INT8 quantization, PCA k=50
# Tests whether INT8 quantization changes the semantic distribution
# Model: meta-llama/Meta-Llama-3-8B-Instruct
# Dataset: wikipedia_en, prompt 0
# Density matrix: PCA k=50 (50x50)
# Samples: 1000, Permutations: 5000

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTFILE="runs/results/01_llama_fp32_vs_int8_pca50_${TIMESTAMP}.txt"
mkdir -p runs/results

echo "Run started: $(date)" | tee "$OUTFILE"
echo "Script: $0" | tee -a "$OUTFILE"
echo "" | tee -a "$OUTFILE"

python quantum_comparison.py \
    --model_a meta-llama/Meta-Llama-3-8B-Instruct \
    --model_b meta-llama/Meta-Llama-3-8B-Instruct \
    --source_a fp32 \
    --source_b int8 \
    --prompts 0 \
    --dataset wikipedia_en \
    --samples 1000 \
    --b 5000 \
    --pca_k 50 \
    2>&1 | tee -a "$OUTFILE"

echo "" | tee -a "$OUTFILE"
echo "Run finished: $(date)" | tee -a "$OUTFILE"
