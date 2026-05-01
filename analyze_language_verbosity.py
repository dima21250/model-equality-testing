"""Analyze completion lengths across languages to test verbosity hypothesis."""

import numpy as np
from model_equality_testing.dataset import load_distribution

def analyze_language_stats(language_code, model, prompts=[0, 1, 2], n_samples=100):
    """Load samples and compute length statistics."""
    prompt_ids = {language_code: prompts}
    
    # Load distribution
    dist = load_distribution(
        model=model,
        prompt_ids=prompt_ids,
        L=500,
        source="fp32",
        load_in_unicode=True,
        root_dir="./data",
    )
    
    # Sample
    sample = dist.sample(n=n_samples)
    
    # Convert to strings and measure lengths
    completions = sample.completion_sample.numpy() if hasattr(sample.completion_sample, 'numpy') else sample.completion_sample
    
    lengths = []
    for completion in completions:
        valid_codepoints = completion[completion != -1]
        text = ''.join(chr(int(cp)) for cp in valid_codepoints if 0 <= cp <= 0x10FFFF)
        lengths.append(len(text))
    
    return {
        'mean': np.mean(lengths),
        'std': np.std(lengths),
        'median': np.median(lengths),
        'min': np.min(lengths),
        'max': np.max(lengths)
    }

# Test both models across all languages
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

# Separation ratios from experiments
separations = {
    'wikipedia_en': 0.40,
    'wikipedia_fr': 0.19,
    'wikipedia_de': 0.71,
    'wikipedia_es': 1.00,
    'wikipedia_ru': 0.40
}

print("="*80)
print("VERBOSITY ANALYSIS: Completion Lengths vs Semantic Convergence")
print("="*80)
print()

results = {}
for lang_code, lang_name in languages.items():
    print(f"\n{lang_name} ({lang_code}):")
    print("-" * 40)
    
    for model_id, model_name in models.items():
        stats = analyze_language_stats(lang_code, model_id)
        results[(lang_code, model_id)] = stats
        print(f"  {model_name:8s}: mean={stats['mean']:.1f} chars (std={stats['std']:.1f})")
    
    # Average across models
    avg_length = np.mean([results[(lang_code, m)]['mean'] for m in models.keys()])
    separation = separations[lang_code]
    print(f"  Average:  {avg_length:.1f} chars")
    print(f"  Separation ratio: {separation:.2f}")

print("\n" + "="*80)
print("SUMMARY: Verbosity vs Convergence")
print("="*80)

summary = []
for lang_code, lang_name in languages.items():
    avg_length = np.mean([results[(lang_code, m)]['mean'] for m in models.keys()])
    separation = separations[lang_code]
    summary.append((lang_name, avg_length, separation))

# Sort by separation
summary.sort(key=lambda x: x[2])

print("\nRanked by semantic convergence (low separation = high convergence):")
print()
print(f"{'Language':<10s} {'Avg Length':>12s} {'Separation':>12s} {'Verbosity Rank':>15s}")
print("-" * 55)

# Add verbosity ranking
summary_with_rank = sorted(summary, key=lambda x: x[1], reverse=True)
verbosity_ranks = {lang: idx+1 for idx, (lang, _, _) in enumerate(summary_with_rank)}

for lang_name, avg_length, separation in summary:
    v_rank = verbosity_ranks[lang_name]
    print(f"{lang_name:<10s} {avg_length:>10.1f} ch {separation:>11.2f}  {v_rank:>13d}/5")

print("\n" + "="*80)
print("HYPOTHESIS TEST: Does verbosity explain convergence?")
print("="*80)

# Calculate correlation
lengths = [x[1] for x in summary]
seps = [x[2] for x in summary]
correlation = np.corrcoef(lengths, seps)[0, 1]

print(f"\nCorrelation between length and separation: {correlation:.3f}")
print()
if correlation < -0.5:
    print("✓ STRONG NEGATIVE correlation: More verbose → more convergence")
    print("  → Linguistic constraint hypothesis SUPPORTED")
elif correlation > 0.5:
    print("✗ STRONG POSITIVE correlation: More verbose → more divergence") 
    print("  → Linguistic constraint hypothesis REJECTED")
else:
    print("~ WEAK correlation: Verbosity doesn't strongly predict convergence")
    print("  → Organizational bias likely primary factor")
print("="*80)

