"""Analyze per-model completion characteristics to understand convergence."""

import numpy as np
from model_equality_testing.dataset import load_distribution

def get_completion_stats(language_code, model, prompts=[0, 1, 2], n_samples=100):
    """Get detailed completion statistics."""
    prompt_ids = {language_code: prompts}
    
    dist = load_distribution(
        model=model,
        prompt_ids=prompt_ids,
        L=500,
        source="fp32",
        load_in_unicode=True,
        root_dir="./data",
    )
    
    sample = dist.sample(n=n_samples)
    completions = sample.completion_sample.numpy() if hasattr(sample.completion_sample, 'numpy') else sample.completion_sample
    
    lengths = []
    for completion in completions:
        valid_codepoints = completion[completion != -1]
        text = ''.join(chr(int(cp)) for cp in valid_codepoints if 0 <= cp <= 0x10FFFF)
        lengths.append(len(text))
    
    return {
        'mean': np.mean(lengths),
        'std': np.std(lengths),
    }

languages = {
    'wikipedia_en': 'English',
    'wikipedia_fr': 'French', 
    'wikipedia_de': 'German',
    'wikipedia_es': 'Spanish',
    'wikipedia_ru': 'Russian'
}

models = {
    'meta-llama/Meta-Llama-3-8B-Instruct': 'Llama',
    'mistralai/Mistral-7B-Instruct-v0.3': 'Mistral'
}

separations = {
    'wikipedia_en': 0.40,
    'wikipedia_fr': 0.19,
    'wikipedia_de': 0.71,
    'wikipedia_es': 1.00,
    'wikipedia_ru': 0.40
}

print("="*80)
print("MODEL DIFFERENCES: Does Mistral show special French optimization?")
print("="*80)
print()

results = {}
for lang_code, lang_name in languages.items():
    print(f"{lang_name}:")
    
    for model_id, model_name in models.items():
        stats = get_completion_stats(lang_code, model_id)
        results[(lang_code, model_id)] = stats
        print(f"  {model_name:8s}: {stats['mean']:.1f} chars (std={stats['std']:.1f})")
    
    # Calculate model difference
    llama_mean = results[(lang_code, 'meta-llama/Meta-Llama-3-8B-Instruct')]['mean']
    mistral_mean = results[(lang_code, 'mistralai/Mistral-7B-Instruct-v0.3')]['mean']
    diff = llama_mean - mistral_mean
    diff_pct = (diff / llama_mean) * 100
    
    llama_std = results[(lang_code, 'meta-llama/Meta-Llama-3-8B-Instruct')]['std']
    mistral_std = results[(lang_code, 'mistralai/Mistral-7B-Instruct-v0.3')]['std']
    
    separation = separations[lang_code]
    
    print(f"  Difference: Llama {diff:+.1f} chars longer ({diff_pct:+.1f}%)")
    print(f"  Llama std: {llama_std:.1f} | Mistral std: {mistral_std:.1f}")
    print(f"  Separation: {separation:.2f}")
    print()

print("="*80)
print("ANALYSIS: Length Consistency vs Semantic Convergence")
print("="*80)
print()

summary = []
for lang_code, lang_name in languages.items():
    llama_mean = results[(lang_code, 'meta-llama/Meta-Llama-3-8B-Instruct')]['mean']
    mistral_mean = results[(lang_code, 'mistralai/Mistral-7B-Instruct-v0.3')]['mean']
    llama_std = results[(lang_code, 'meta-llama/Meta-Llama-3-8B-Instruct')]['std']
    mistral_std = results[(lang_code, 'mistralai/Mistral-7B-Instruct-v0.3')]['std']
    
    diff = abs(llama_mean - mistral_mean)
    avg_std = (llama_std + mistral_std) / 2
    separation = separations[lang_code]
    
    summary.append((lang_name, diff, avg_std, separation))

# Sort by convergence
summary.sort(key=lambda x: x[3])

print(f"{'Language':<10s} {'Length Diff':>12s} {'Avg Std':>10s} {'Separation':>12s}")
print("-" * 50)
for lang_name, diff, avg_std, separation in summary:
    print(f"{lang_name:<10s} {diff:>10.1f} ch {avg_std:>8.1f} {separation:>11.2f}")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

# French-specific analysis
fr_llama = results[('wikipedia_fr', 'meta-llama/Meta-Llama-3-8B-Instruct')]['mean']
fr_mistral = results[('wikipedia_fr', 'mistralai/Mistral-7B-Instruct-v0.3')]['mean']
fr_diff = abs(fr_llama - fr_mistral)

es_llama = results[('wikipedia_es', 'meta-llama/Meta-Llama-3-8B-Instruct')]['mean']
es_mistral = results[('wikipedia_es', 'mistralai/Mistral-7B-Instruct-v0.3')]['mean']
es_diff = abs(es_llama - es_mistral)

print()
print("1. French convergence (0.19):")
print(f"   - Llama produces {fr_llama:.1f} char completions")
print(f"   - Mistral produces {fr_mistral:.1f} char completions")
print(f"   - Difference: {fr_diff:.1f} chars")
print()
print("2. Spanish divergence (1.00):")
print(f"   - Llama produces {es_llama:.1f} char completions")
print(f"   - Mistral produces {es_mistral:.1f} char completions")
print(f"   - Difference: {es_diff:.1f} chars")
print()
print("3. Does Mistral have special French optimization?")

# Check if Mistral's French std is notably lower
fr_mistral_std = results[('wikipedia_fr', 'mistralai/Mistral-7B-Instruct-v0.3')]['std']
fr_llama_std = results[('wikipedia_fr', 'meta-llama/Meta-Llama-3-8B-Instruct')]['std']

other_langs = ['wikipedia_en', 'wikipedia_de', 'wikipedia_es', 'wikipedia_ru']
avg_mistral_std = np.mean([results[(l, 'mistralai/Mistral-7B-Instruct-v0.3')]['std'] for l in other_langs])

print(f"   - Mistral French std: {fr_mistral_std:.1f}")
print(f"   - Mistral avg std (other langs): {avg_mistral_std:.1f}")
print(f"   - Llama French std: {fr_llama_std:.1f}")
print()

if fr_mistral_std < avg_mistral_std * 0.8:
    print("   ✓ Mistral shows MORE consistency in French than other languages")
    print("     → Evidence of special French optimization")
else:
    print("   ✗ Mistral's French consistency similar to other languages")
    print("     → Convergence likely due to both models being good at French")

print()
print("4. Hypothesis: Training data standardization")
print(f"   - Languages with LOW model length difference: High convergence?")
print(f"   - Languages with HIGH model length difference: Low convergence?")
print()

# Correlation between length difference and separation
length_diffs = [x[1] for x in summary]
seps = [x[3] for x in summary]
correlation = np.corrcoef(length_diffs, seps)[0, 1]
print(f"   Correlation (length diff vs separation): {correlation:.3f}")

if correlation > 0.5:
    print("   ✓ Models that produce similar lengths also converge semantically")
    print("     → Both converge to similar language 'norms'")
elif correlation < -0.5:
    print("   ✗ Negative correlation - unexpected pattern")
else:
    print("   ~ Weak correlation - length similarity doesn't predict convergence")

print("="*80)
