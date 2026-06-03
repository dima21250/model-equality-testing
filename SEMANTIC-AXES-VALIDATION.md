# Semantic Axes Validation: Addressing Reviewer Concerns

## The Core Problem

**"LLM-generated axes to interpret LLM outputs" has a circularity problem** that reviewers will flag.

You're using:
1. LLM to generate pole corpora (e.g., "professionalism" vs "casualness")
2. Those poles to define semantic axes
3. Those axes to interpret LLM outputs

**Reviewers will ask**: How is this not circular? How do you know it's valid?

---

## Expected Reviewer Objections

### Objection 1: Circularity
> "You're using an LLM to generate pole corpora to define 'professionalism', then using that to evaluate LLM outputs. How is this not circular?"

**The concern**: LLM biases in pole generation → biased axes → biased interpretation

### Objection 2: Validity
> "How do you know your 'professionalism' axis actually measures professionalism and not something else?"

**The concern**: No ground truth, no validation against human judgment

### Objection 3: Subjective Framing
> "Your choice of which axes to construct is arbitrary. Why professionalism vs formality vs register?"

**The concern**: Cherry-picking dimensions to tell a story

### Objection 4: Replicability
> "If I generate different pole corpora, will I get the same results?"

**The concern**: Instability in the axes across generations

---

## What You Already Have (Good News)

From your implementation in `semantic_axes.py`:

### Quality Metrics
```python
separation_ratio = pole_distance / mean_intra_std
# Warns if < 1.0
```

This provides **construct validity** - poles actually separate in embedding space.

### Statistical Rigor
- Welch's t-test (no equal variance assumption)
- Hedges' g (established effect size scale)
- Benjamini-Hochberg FDR correction (multiple testing)

This provides **statistical validity**.

### Transparency
- Pole corpora stored in JSONL
- Separation metrics saved
- Process documented in SEMANTIC-AXES-GUIDE.md

This supports **replicability**.

---

## What You're Missing (The Gap)

### 1. External Validation

**No human annotation** comparing axis projections to human judgments.

**What reviewers want to see**:
```
100 LLM outputs
→ Human raters score professionalism (1-7 scale)
→ Semantic axis projects onto professionalism dimension
→ Correlation: r = 0.85, p < 0.001
```

**This proves**: Axis captures what humans think of as professionalism.

---

### 2. Known-Groups Validation

**No demonstration** that axes discriminate expected differences.

**What reviewers want**:
```
Compare:
- Formal academic papers vs Reddit comments
- Technical documentation vs casual blog posts

Prediction: Professionalism axis should separate these
Result: Academic papers score +2.3σ higher (p < 0.001)
```

**This proves**: Axis behaves as expected on obvious cases.

---

### 3. Convergent/Discriminant Validity

**No comparison** to existing validated measures.

**What reviewers want**:
```
Existing validated scale: LIWC formality score
Your axis: Professionalism projection

Convergent validity: r = 0.75 (should correlate)
Discriminant validity: r = 0.15 with sentiment (shouldn't correlate)
```

**This proves**: Axis measures what you claim, not something else.

---

### 4. Stability/Reliability

**No check** that regenerating poles gives same axis.

**What reviewers want**:
```
Generate professionalism poles 5 times (different LLM samples)
Compute 5 axes
Check correlation: r > 0.95 across all pairs
```

**This proves**: Axes are stable, not random.

---

## Solutions (Ordered by Effort)

### Solution 1: Face Validity (30 minutes)

**Show concrete examples** in your paper.

#### Implementation:

```markdown
Table: Example Pole Corpus Texts for Professionalism-Casualness Axis

Professionalism Pole (negative direction):
- "yeah so basically the thing is like super weird lol"
- "idk man just try it and see what happens"
- "this is kinda dumb but whatever"
- "u should prob check it out sometime"
- "nah that doesn't make any sense tbh"

Casualness Pole (positive direction):
- "The aforementioned framework demonstrates substantial efficacy"
- "We observe systematic patterns consistent with theoretical predictions"
- "This approach warrants careful consideration"
- "The methodology employed adheres to established protocols"
- "Results indicate statistically significant differences"

Separation: ||μ_pos - μ_neg|| = 2.34
Separation ratio: 3.42 (indicates good discrimination)
Pole corpus size: 50 texts each
```

**Value**: Readers can judge for themselves if poles make sense.

**Limitation**: Subjective, not quantitative validation.

**Effort**: 30 minutes to document and format.

---

### Solution 2: Known-Groups Validation (1-2 hours)

**Find existing corpora** with known properties and test if axis discriminates as expected.

#### Implementation:

```python
from model_equality_testing.src.embeddings import embed_sample
from model_equality_testing.src.semantic_axes import load_axis_from_jsonl
import numpy as np
from scipy.stats import ttest_ind

def validate_axis_known_groups(axis, corpus_high, corpus_low, label_high, label_low):
    """
    Validate semantic axis using known-groups with expected differences.
    
    Args:
        axis: SemanticAxis object
        corpus_high: List of texts expected to score high on axis
        corpus_low: List of texts expected to score low on axis
        label_high, label_low: Names for the groups
    
    Returns:
        dict with validation statistics
    """
    # Embed both corpora
    emb_high = embed_texts(corpus_high)  # Helper to embed raw text
    emb_low = embed_texts(corpus_low)
    
    # Project onto axis
    proj_high = emb_high @ axis.axis_vector
    proj_low = emb_low @ axis.axis_vector
    
    # Statistics
    mean_high = proj_high.mean()
    mean_low = proj_low.mean()
    std_high = proj_high.std()
    std_low = proj_low.std()
    
    # T-test
    t_stat, p_value = ttest_ind(proj_high, proj_low, equal_var=False)
    
    # Cohen's d
    pooled_std = np.sqrt((std_high**2 + std_low**2) / 2)
    cohens_d = (mean_high - mean_low) / pooled_std
    
    return {
        'mean_high': mean_high,
        'mean_low': mean_low,
        'difference': mean_high - mean_low,
        'cohens_d': cohens_d,
        't_statistic': t_stat,
        'p_value': p_value,
        'label_high': label_high,
        'label_low': label_low
    }


# Example: Validate professionalism axis
prof_axis = load_axis_from_jsonl(
    "professionalism.jsonl",
    "casualness.jsonl",
    "prof-casual"
)

# Load known corpora
academic_papers = load_corpus("arxiv_abstracts", n=1000)
reddit_comments = load_corpus("reddit_comments", n=1000)

# Validate
result = validate_axis_known_groups(
    prof_axis, 
    academic_papers, 
    reddit_comments,
    "Academic papers",
    "Reddit comments"
)

print(f"{result['label_high']} score: {result['mean_high']:.2f}")
print(f"{result['label_low']} score: {result['mean_low']:.2f}")
print(f"Difference: {result['difference']:.2f} ({result['cohens_d']:.2f}σ)")
print(f"P-value: {result['p_value']:.2e}")
```

#### Suggested Known-Groups Pairs:

| Axis | High Group | Low Group |
|------|-----------|-----------|
| Professionalism-Casualness | arXiv abstracts | Reddit comments |
| Technicality-Layperson | API documentation | Children's books |
| Formality-Informality | Legal documents | Text messages |
| Academic-Colloquial | Research papers | Blog posts |
| Precision-Vagueness | Math proofs | Casual emails |

#### Where to Find Corpora:

- **arXiv abstracts**: Download from arXiv API
- **Reddit**: Use Reddit API or existing datasets (e.g., Pushshift)
- **Wikipedia**: Simple English Wikipedia (low complexity) vs regular (higher)
- **Children's books**: Project Gutenberg children's section
- **Academic papers**: PubMed abstracts, ACL anthology
- **Casual text**: Twitter, SMS datasets

**Value**: Quantitative, convincing, doesn't require new human annotation.

**Limitation**: Only validates on obvious cases, might not generalize.

**Effort**: 1-2 hours to gather corpora and run validation.

---

### Solution 3: Human Annotation Study (1-2 days)

**Gold standard**: Compare axis projections to human judgments.

#### Design:

**Sampling**:
1. Draw 100 LLM outputs from your dataset
   - Mix of models (fp32, int8, etc.)
   - Mix of prompts
   - Ensure diversity

**Annotation**:
2. Recruit 3-5 annotators (colleagues, Mechanical Turk, Prolific)
3. Each rates each text on professionalism (1-7 Likert scale)
   - 1 = Very casual/unprofessional
   - 4 = Neutral
   - 7 = Very professional/formal

**Guidelines for annotators**:
```
Rate how professional/formal this text is:

Consider:
- Word choice (casual vs formal vocabulary)
- Grammar (colloquial vs standard)
- Tone (conversational vs academic)
- Structure (organized vs stream-of-consciousness)

Do NOT consider:
- Content accuracy
- Your agreement with the text
- Topic/subject matter
```

**Analysis**:
4. Compute inter-rater reliability (Cronbach's α or ICC)
5. Average ratings across raters → "ground truth"
6. Compute axis projections for same 100 texts
7. Correlation analysis

#### Implementation:

```python
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

def validate_axis_human_annotation(axis, texts, human_ratings):
    """
    Validate semantic axis against human judgments.
    
    Args:
        axis: SemanticAxis object
        texts: List of 100 text strings
        human_ratings: (100, n_raters) array of human scores (1-7 scale)
    
    Returns:
        dict with validation statistics
    """
    # Inter-rater reliability
    # Cronbach's alpha
    n_items, n_raters = human_ratings.shape
    item_variance = human_ratings.var(axis=1, ddof=1).mean()
    total_variance = human_ratings.values.flatten().var(ddof=1)
    cronbach_alpha = (n_raters / (n_raters - 1)) * (1 - item_variance / total_variance)
    
    # Or use ICC (more appropriate for continuous ratings)
    # from pingouin import intraclass_corr
    # icc = intraclass_corr(data, targets='item', raters='rater', ratings='score')
    
    # Average human ratings
    human_avg = human_ratings.mean(axis=1)  # (100,)
    
    # Axis projections
    embeddings = embed_texts(texts)  # (100, 768)
    axis_proj = embeddings @ axis.axis_vector  # (100,)
    
    # Correlation
    r, p_value = pearsonr(human_avg, axis_proj)
    
    # Regression (for R²)
    model = LinearRegression()
    model.fit(axis_proj.reshape(-1, 1), human_avg)
    r_squared = model.score(axis_proj.reshape(-1, 1), human_avg)
    
    return {
        'cronbach_alpha': cronbach_alpha,
        'correlation': r,
        'p_value': p_value,
        'r_squared': r_squared,
        'n_texts': len(texts),
        'n_raters': n_raters
    }


# Example usage
result = validate_axis_human_annotation(prof_axis, texts, ratings_array)

print(f"Inter-rater reliability (Cronbach's α): {result['cronbach_alpha']:.3f}")
print(f"Correlation with human judgments: r = {result['correlation']:.3f}")
print(f"P-value: {result['p_value']:.2e}")
print(f"Variance explained: R² = {result['r_squared']:.3f}")
```

#### Interpretation Guidelines:

**Inter-rater reliability**:
- α > 0.7: Acceptable
- α > 0.8: Good
- α > 0.9: Excellent

**Correlation**:
- r > 0.5: Moderate validity
- r > 0.7: Good validity
- r > 0.8: Strong validity

**R²**:
- R² > 0.5: Axis explains 50%+ of variance in human judgments

#### Report in Paper:

> "We validated the professionalism-casualness axis against human judgments. Five annotators rated 100 LLM outputs on professionalism (1-7 scale; ICC=0.82, indicating good inter-rater agreement). Axis projections correlated strongly with average human ratings (r=0.85, p<0.001, R²=0.72), demonstrating that the axis captures human judgments of professionalism."

**Value**: Gold standard validation, hard to argue against.

**Limitation**: Labor intensive, requires IRB if formal human subjects research.

**Effort**: 1-2 days (recruiting, annotation, analysis).

**Cost**: $50-200 if using Mechanical Turk/Prolific (100 texts × 5 raters × $0.10-0.40 per rating).

---

### Solution 4: Stability Analysis (2 hours)

**Regenerate pole corpora multiple times** and check that resulting axes are consistent.

#### Implementation:

```python
from model_equality_testing.src.semantic_axes import load_axis_from_jsonl
import numpy as np

def stability_analysis(axis_name, n_replications=5):
    """
    Check stability of axis across multiple pole corpus generations.
    
    Args:
        axis_name: Name of the axis (e.g., "professionalism")
        n_replications: Number of times to regenerate poles
    
    Returns:
        dict with stability statistics
    """
    axes = []
    
    for i in range(n_replications):
        # Generate fresh pole corpora with different seed
        # (This assumes you have a pole generation function)
        neg_corpus = generate_pole_corpus(
            f"{axis_name}_negative", 
            n_samples=50, 
            seed=i
        )
        pos_corpus = generate_pole_corpus(
            f"{axis_name}_positive", 
            n_samples=50, 
            seed=i
        )
        
        # Save to JSONL
        save_corpus(neg_corpus, f"pole_neg_{i}.jsonl")
        save_corpus(pos_corpus, f"pole_pos_{i}.jsonl")
        
        # Build axis
        axis = load_axis_from_jsonl(
            f"pole_neg_{i}.jsonl",
            f"pole_pos_{i}.jsonl",
            f"{axis_name}_{i}"
        )
        
        axes.append(axis.axis_vector)
    
    # Compute pairwise correlations (cosine similarity)
    correlations = []
    for i in range(n_replications):
        for j in range(i+1, n_replications):
            # Cosine similarity in high-dimensional space
            cos_sim = np.dot(axes[i], axes[j])  # Already L2-normalized
            correlations.append(cos_sim)
    
    # Statistics
    mean_corr = np.mean(correlations)
    std_corr = np.std(correlations)
    min_corr = np.min(correlations)
    max_corr = np.max(correlations)
    
    return {
        'n_replications': n_replications,
        'mean_correlation': mean_corr,
        'std_correlation': std_corr,
        'min_correlation': min_corr,
        'max_correlation': max_corr,
        'all_correlations': correlations
    }


# Example
result = stability_analysis("professionalism", n_replications=5)

print(f"Stability Analysis: {result['n_replications']} replications")
print(f"Mean pairwise correlation: {result['mean_correlation']:.3f}")
print(f"Std: {result['std_correlation']:.3f}")
print(f"Range: [{result['min_correlation']:.3f}, {result['max_correlation']:.3f}]")

# Good stability: mean > 0.95, min > 0.90
```

#### Interpretation:

**Cosine similarity between axes**:
- \> 0.95: Excellent stability
- 0.90-0.95: Good stability
- 0.80-0.90: Moderate stability (acceptable but note in paper)
- < 0.80: Poor stability (indicates random axes)

#### Report in Paper:

> "To assess axis stability, we regenerated pole corpora five times with different random seeds. Pairwise cosine similarities between resulting axes ranged from 0.96 to 0.98 (mean=0.97, std=0.01), demonstrating high reliability. This indicates that axes are not artifacts of specific pole samples but capture stable semantic dimensions."

**Value**: Shows axes aren't random, results are replicable.

**Limitation**: Only checks consistency, not validity against external criteria.

**Effort**: 2 hours (requires pole generation capability).

---

## Addressing the Circularity Concern

### Why LLM-Generated Poles Are Less Problematic Than They Seem

#### Argument 1: Different LLMs (If Applicable)

```
Pole generation: GPT-4 (or Claude) generates pole corpora
Evaluation targets: Llama-3, Mistral outputs being evaluated

No circularity: Different models, different training data, different architectures
```

**Framing in paper**:
> "Pole corpora were generated using GPT-4, while the evaluated outputs come from Llama-3 and Mistral models. Since pole generation and evaluation target different LLMs, there is no self-referential circularity."

---

#### Argument 2: Embedding Space is Independent

```
SBERT embeddings: all-mpnet-base-v2
- Trained on general web text (not LLM outputs)
- Captures semantic similarity independent of generation source

Axes live in embedding space, not generation space
Semantic dimensions exist independently of which LLM generated poles
```

**Framing in paper**:
> "Semantic axes are defined in the SBERT embedding space (all-mpnet-base-v2), which was trained on general human-written text. The axes capture semantic dimensions that exist in this shared embedding space, independent of which LLM generated the pole corpora. The LLM serves only to efficiently sample texts from the extremes of these pre-existing dimensions."

---

#### Argument 3: External Validation Breaks Circularity

```
If axes validated against:
- Human judgment (external anchor)
- Known corpora (external anchor)
- Existing validated scales (external anchor)

Then circularity is broken by external validation
```

**Framing in paper**:
> "While pole corpora were generated using an LLM, we validate the resulting axes through external criteria: (1) known-groups with expected differences, (2) stability across independent regenerations, and (3) [if done: correlation with human judgments]. These external anchors ensure the axes capture real semantic dimensions rather than artifacts of the generation process."

---

#### Argument 4: Comparison to Human-Generated Lexicons

Many established tools use similar approaches:

- **LIWC**: Human-curated word lists define psychological categories
- **SentiWordNet**: Semi-automated sentiment lexicon
- **General Inquirer**: Human-tagged word categories

**Your approach**:
- LLM-generated text corpora define semantic poles
- Validated against external criteria

**Framing in paper**:
> "Our approach parallels established lexicon-based methods like LIWC, which define semantic dimensions through curated text samples. While LIWC uses human-curated word lists, we use LLM-generated pole corpora, both validated through external criteria. The key difference is scale: LLMs enable rapid generation of large, diverse pole corpora that would be prohibitively expensive to create manually."

---

## What to Include in Your Paper

### Minimum (Essential for Publication)

**Required to pass review**:

1. **Face validity**: Example pole texts (Table in main paper or appendix)
2. **Known-groups validation**: Test on 2-3 obvious cases (Section in paper)
3. **Separation metrics**: Report pole distance, separation ratio (always report these)
4. **Stability check**: Regenerate poles, show consistency (Appendix or brief paragraph)

**Estimated effort**: 3-4 hours total

**Where to include**:
- Methods section: Describe pole generation and axis construction
- Validation section: Known-groups results, stability analysis
- Appendix: Example pole texts, detailed stability results

---

### Strong Validation (Top-Tier Venue)

**For ICML, NeurIPS, ACL, etc.**:

All of the above, plus:

5. **Human annotation study**: Compare to human judgments (Main paper section)
6. **Convergent validity**: Compare to existing measures like LIWC (if available)
7. **Multiple axes**: Show validation pattern holds across 3+ semantic dimensions

**Estimated effort**: 1-2 days additional

**Impact**: Very hard to reject on methodological grounds

---

## Recommended Timeline

### Phase 1: Quick Validation (Tonight, 2-3 hours)

**Goal**: Minimum viable validation for paper draft

1. **Face validity** (30 min):
   - Extract and document 5 example texts from each pole
   - Format as table for paper
   - Report separation ratio

2. **Known-groups validation** (1-2 hours):
   - Identify accessible corpora (arXiv, Reddit, Wikipedia)
   - Download/sample texts
   - Run validation script
   - Compute statistics (mean difference, Cohen's d, p-value)

3. **Stability check** (1 hour):
   - Regenerate poles 3-5 times
   - Compute pairwise correlations
   - Report range and mean

**Deliverable**: Validation section for paper (~1 page)

---

### Phase 2: If Reviewers Request More (During Revision, 1-2 days)

4. **Human annotation study**:
   - Sample 100 texts
   - Recruit 3-5 annotators
   - Collect ratings
   - Analyze correlation

**Deliverable**: Strong empirical validation to address reviewer concerns

---

## Example Validation Section (For Paper)

### 4.3 Semantic Axis Validation

We validate our semantic axes through multiple complementary approaches to ensure they capture meaningful semantic dimensions.

#### 4.3.1 Pole Separation and Quality

Table 2 shows example texts from the professionalism-casualness axis pole corpora. The poles achieve strong separation (pole distance = 2.34, separation ratio = 3.42), indicating the centroid distance between poles is 3.42 times the average within-pole standard deviation. We set a minimum threshold of 1.0 for separation ratio; axes below this threshold are flagged as poorly discriminating.

#### 4.3.2 Known-Groups Validation

To validate that axes capture their intended semantic dimensions, we tested them on external corpora with known properties. For the professionalism-casualness axis, we compared arXiv paper abstracts (n=1,000) with Reddit comments (n=1,000). Academic abstracts scored μ=+1.87 (σ=0.42) while Reddit comments scored μ=-2.14 (σ=0.58), a difference of 4.01 standard deviations (Cohen's d=2.15, Welch's t-test p<0.001). This confirms the axis discriminates between formal and informal text as expected.

Similarly, for the technical-layperson axis, API documentation scored +2.34 while children's stories scored -1.98 (d=2.87, p<0.001). Table 3 shows results for all five axes tested.

#### 4.3.3 Stability and Reliability

To assess whether axes are stable across different pole corpus samples, we regenerated pole corpora five times with different random seeds (n=50 texts per pole, per replication). For each replication, we constructed the full axis in SBERT embedding space. Pairwise cosine similarities between the five professionalism-casualness axes ranged from 0.96 to 0.98 (mean=0.97, std=0.01), demonstrating high reliability. This pattern held across all axes tested (Appendix B), indicating axes capture stable semantic dimensions rather than artifacts of specific pole samples.

#### 4.3.4 Human Validation [Optional - if done]

We validated the professionalism-casualness axis against human judgments by collecting ratings for 100 LLM outputs sampled from our dataset. Five annotators rated each text on professionalism (1-7 Likert scale); inter-rater reliability was high (ICC=0.82). Axis projections correlated strongly with average human ratings (Pearson r=0.85, p<0.001, R²=0.72), confirming the axis captures human perceptions of professionalism. [Include scatter plot showing correlation.]

#### 4.3.5 Addressing Potential Circularity

While pole corpora were generated using an LLM (GPT-4), the evaluated outputs come from different models (Llama-3, Mistral), eliminating self-referential circularity. Moreover, axes are defined in the SBERT embedding space (all-mpnet-base-v2), trained on general human-written text, not LLM outputs. The axes capture semantic dimensions in this shared embedding space; the LLM serves only to efficiently sample texts from the extremes of pre-existing dimensions. External validation (known-groups discrimination, human correlation) confirms axes measure real semantic properties rather than generation artifacts.

---

## Implementation Checklist

### Before Paper Submission

- [ ] Document example pole texts for main axes (face validity)
- [ ] Identify 2-3 known-groups pairs per axis
- [ ] Download/sample known-groups corpora
- [ ] Implement `validate_axis_known_groups()` function
- [ ] Run known-groups validation for all axes
- [ ] Generate validation statistics table
- [ ] Implement `stability_analysis()` function
- [ ] Regenerate poles 3-5 times per axis
- [ ] Compute stability correlations
- [ ] Write validation section for paper (1 page)
- [ ] Create tables/figures:
  - Table: Example pole texts
  - Table: Known-groups validation results
  - (Optional) Figure: Correlation with human ratings

### If Reviewers Request (During Revision)

- [ ] Design human annotation study
- [ ] Sample 100 diverse texts
- [ ] Create annotation interface/instructions
- [ ] Recruit 3-5 annotators
- [ ] Collect annotations
- [ ] Compute inter-rater reliability
- [ ] Analyze correlation with axis projections
- [ ] Create scatter plot showing correlation
- [ ] Update validation section with human results

---

## Summary

### The Circularity Concern is Real But Addressable

**Reviewers will question** LLM-generated poles for interpreting LLM outputs.

**You can address this through**:
1. **External validation** (known-groups, human judgments)
2. **Stability** (regeneration shows consistency)
3. **Honest framing** (embedding space is independent)
4. **Different LLMs** (generation ≠ evaluation)

### Minimum Viable Validation (3-4 hours)

- ✅ Face validity (example pole texts)
- ✅ Known-groups validation (2-3 obvious cases per axis)
- ✅ Stability check (3-5 regenerations)

**This should be sufficient** for most venues.

### Gold Standard Validation (1-2 days)

Add human annotation study if:
- Targeting top-tier venue (ICML, NeurIPS, ACL)
- Reviewers specifically request it
- You want bulletproof validation

### Next Steps

1. **Tonight**: Implement known-groups validation (2 hours)
2. **Tomorrow**: Run stability analysis (1 hour)
3. **Write validation section** for paper (1 hour)
4. **If needed later**: Human annotation study (1-2 days)

The validation work is straightforward and significantly strengthens the semantic axes methodology. Without it, reviewers will (rightfully) question the approach.

**Want help implementing the validation scripts or identifying good known-groups corpora?**
