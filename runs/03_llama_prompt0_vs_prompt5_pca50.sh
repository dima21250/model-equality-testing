#!/bin/bash
# Context 3: Llama-3-8B prompt 0 vs prompt 5, PCA k=50
# Tests whether different prompts produce detectably different distributions (positive control)
# Model: meta-llama/Meta-Llama-3-8B-Instruct (fp32)
# Dataset: wikipedia_en, prompts 0 and 5
# Density matrix: PCA k=50 (50x50)
# Samples: 1000, Permutations: 5000

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTFILE="runs/results/03_llama_prompt0_vs_prompt5_pca50_${TIMESTAMP}.txt"
mkdir -p runs/results

echo "Run started: $(date)" | tee "$OUTFILE"
echo "Script: $0" | tee -a "$OUTFILE"
echo "" | tee -a "$OUTFILE"

python quantum_comparison.py \
    --model_a meta-llama/Meta-Llama-3-8B-Instruct \
    --source_a fp32 \
    --dataset wikipedia_en \
    --samples 1000 \
    --b 5000 \
    --pca_k 50 \
    --prompt_comparison 0 5 \
    2>&1 | tee -a "$OUTFILE"

echo "" | tee -a "$OUTFILE"
echo "Run finished: $(date)" | tee -a "$OUTFILE"
