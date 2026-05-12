#!/bin/bash
# Context 4: Llama-3-8B English vs Russian cross-language, PCA k=50
# Tests whether different languages produce detectably different distributions
# Model: meta-llama/Meta-Llama-3-8B-Instruct (fp32)
# Datasets: wikipedia_en prompt 0 vs wikipedia_ru prompt 0
# Density matrix: PCA k=50 (50x50)
# Samples: 1000, Permutations: 5000

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTFILE="runs/results/04_llama_en_vs_ru_pca50_${TIMESTAMP}.txt"
mkdir -p runs/results

echo "Run started: $(date)" | tee "$OUTFILE"
echo "Script: $0" | tee -a "$OUTFILE"
echo "" | tee -a "$OUTFILE"

python quantum_comparison.py \
    --model_a meta-llama/Meta-Llama-3-8B-Instruct \
    --source_a fp32 \
    --dataset wikipedia_en \
    --dataset_b wikipedia_ru \
    --samples 1000 \
    --b 5000 \
    --pca_k 50 \
    --prompt_comparison 0 0 \
    2>&1 | tee -a "$OUTFILE"

echo "" | tee -a "$OUTFILE"
echo "Run finished: $(date)" | tee -a "$OUTFILE"
