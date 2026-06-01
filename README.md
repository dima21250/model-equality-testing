# Model Equality Testing: Which Model Is This API Serving?

[Paper link](https://arxiv.org/abs/2410.20247) | [Dataset link](https://drive.google.com/drive/folders/1TgxlUp3n-BFh-A6ARA42jkvxkv7Leccv?usp=drive_link) | [Twitter announcement](https://x.com/irena_gao/status/1851273269690908777)


Users often interact with large language models through black-box inference APIs, both for closed- and open-weight models (e.g., Llama models are popularly accessed via Amazon Bedrock and Azure AI Studios). 
In order to cut costs or add functionality, API providers may quantize, watermark, or finetune the underlying model, changing the output distribution — often without notifying users. How can we detect if an API has changed for our particular task using only sample access?

<center><img src = "hero.png" width=600></center>


We formalize this problem as Model Equality Testing, a two-sample testing problem where the user collects samples from the API and a reference distribution, and conducts a statistical test to see if the two distributions are the same. Unlike current approaches that simply compare numbers on standard benchmarks, this approach is specific to a user’s distribution of task prompts, is applicable to tasks without automated evaluation metrics, and can be more powerful at distinguishing distributions.


<center><img src = "setup.png" width=400></center>

To enable users to test APIs on their own tasks, we open-source a Python package here. Additionally, to encourage future research into this problem, we also release a dataset of 1 million LLM completions that can be used to learn / evaluate more powerful tests.

## Installation
To run Model Equality Testing on your own samples, we recommend using pip to install the package:

```
pip install model-equality-testing
```

The package provides functions to run the tests discussed in the paper on your samples. This includes functions to compute test statistics and simulate p-values.

An example of how to use the package to test two samples is below, and additional examples will be continually added to `demo.ipynb`.

```python
import numpy as np

########## Example data ###############
sampled_prompts_1 = np.array([0, 1, 0]) # integers representing which prompt was selected
corresponding_completions_1 = [
    "...a time to be born and a time to die",
    "'Laughter,' I said, 'is madness.'",
    "...a time to weep and a time to laugh",
] # corresponding completions
sampled_prompts_2 = np.array([0, 0, 1]) # integers representing which prompt was selected
corresponding_completions_2 = [
    "...a time to mourn and a time to dance",
    "...a time to embrace and a time to refrain from embracing",
    "I said to myself, 'Come now, I will test you'",
] # corresponding completions

######### Testing code ################
# Tokenize the string completions as unicode codepoints
# and pad both completion arrays to a shared maximum length of 200 chars
from model_equality_testing.utils import tokenize_unicode
corresponding_completions_1 = tokenize_unicode(corresponding_completions_1)
corresponding_completions_1 = pad_to_length(corresponding_completions_1, L=200) 
corresponding_completions_2 = tokenize_unicode(corresponding_completions_2)
corresponding_completions_2 = pad_to_length(corresponding_completions_2, L=200) 


# Wrap these as CompletionSample objects
# m is the total number of prompts supported by the distribution
from model_equality_testing.distribution import CompletionSample

sample1 = CompletionSample(prompts=sampled_prompts_1, completions=corresponding_completions_1, m=2)
sample2 = CompletionSample(prompts=sampled_prompts_2, completions=corresponding_completions_2, m=2)

from model_equality_testing.algorithm import run_two_sample_test

# Run the two-sample test
pvalue, test_statistic = run_two_sample_test(
    sample1,
    sample2,
    pvalue_type="permutation_pvalue", # use the permutation procedure to compute the p-value
    stat_type="mmd_hamming", # use the MMD with Hamming kernel as the test statistic
    b=100, # number of permutations
)
print(f"p-value: {pvalue}, test statistic: {test_statistic}")
print("Should we reject P = Q?", pvalue < 0.05)
```

## Dataset
To enable future research on better tests for Model Equality Testing, we release a dataset of LLM completions, including samples used in the paper experiments. At a high level, this dataset includes 1.6M completion samples collected across 5 language models, each served by various sources (e.g. in `fp32` and `int8` precisions, as well as by various inference API providers, e.g. `amazon` and `azure`). These completions are collected for a fixed set of 540 prompts. For 100 of these prompts (the "dev set"), we additionally collect logprobs for each completion under the fp32 model.

The data (and a spreadsheet documenting its contents) are hosted as a 37.1GB `dataset.zip` file [here](https://drive.google.com/drive/folders/1TgxlUp3n-BFh-A6ARA42jkvxkv7Leccv?usp=drive_link). For convenience, we provide a function in the `model-equality-testing` package to automatically download and unzip the dataset.

```python
# make sure to first install gdown 
# ! pip install gdown
from model_equality_testing.dataset import download_dataset
download_dataset(root_dir="./data") # will download to ./data
```

You can also download just the samples (dev/test set) or just the logprobs (dev set); make sure to set these inside `{root_dir}/samples/` and `{root_dir}/logprobs` paths respectively. These can be found as separate zip files in the Google Drive link above.

Once downloaded, you can load the dataset using the function `load_distribution`, which returns a `DistributionFromDataset` object.

```python
# load a distribution object representing the joint distribution
# where prompts come from Wikipedia (Ru) with prompt ids 0, 3, 10
# and Wikipedia (De) with prompt id 5
# and completions come from meta-llama/Meta-Llama-3-8B-Instruct
from model_equality_testing.dataset import load_distribution
p = load_distribution(
    model="meta-llama/Meta-Llama-3-8B-Instruct", # model
    prompt_ids={"wikipedia_ru": [0, 3, 10], "wikipedia_de": [5]}, # prompts
    L=1000, # number of characters to pad / truncate to
    source="fp32", # or replace with 'nf4', 'int8', 'amazon', etc.
    load_in_unicode=True, # instead of tokens
    root_dir="./data",
)
```

[This spreadsheet in the Google Drive](https://docs.google.com/spreadsheets/d/1T9aPZHK1xxfxogrHYaHqvW0Blqi-XJx2rgOPktWcN0w/edit?usp=sharing) catalogs all samples present in the dataset, including for which prompt they were collected, under which language model, and served by which source. 

At a high level, samples are split into those from local sources (`fp32`, `fp16`, `int8`, `nf4`, `watermark`) vs. APIs (`anyscale`, `amazon`, `fireworks`, `replicate`, `deepinfra`, `groq`,  `perplexity`, `together`, `azure`).
Local samples are saved as `.pkl` files containing numpy arrays of token IDs (integers).
API samples are saved as `.pkl` files containing dictionaries; the samples themselves are strings returned directly by the API.

* Note that the data loading code in our package does additional postprocessing on local samples, e.g. replacing tokens to the right of the first `<eos>` token with pad tokens, and padding to the max observed length. This is because our desired behavior is to only test strings up to the first `<eos>` token. When the model tokenizer does not specify a pad token, we set it to the `<eos>` token. For convenience, we've preprocessed local samples before saving in the zip files above, but we include the processing code in the `model_equality_testing.dataset` module.
* To the best of our knowledge, API samples were all returned without special tokens.
* Local samples may be loaded in unicode or token space; API samples can only be loaded in unicode. When loading samples in unicode, samples are batch decoded skipping special tokens, and each character is represented by its integer Unicode codepoint. Padding is represented as `-1`.

Additional details about dataset collection can be found in Appendix B.1 in [the paper](https://arxiv.org/abs/2410.20247). 

## Reproducing paper experiments

In `experiments/`, we include the code used to produce the experiments shown in the paper, including the code to generate the dataset (`experiments/sampling`) and the code to simulate power (`experiments/testing`).

Note that APIs are actively evolving: many APIs have changed behavior since when we used these scripts to collect samples between July and August 2024. For full details documenting the dates we queried each API for the samples in our dataset, see Appendix B.1 in [the paper](https://arxiv.org/abs/2410.20247).

## Semantic Axes Interpretation

Beyond detecting *if* two models differ, you may want to understand *what* differs. The `semantic_axes` module provides an interpretation layer that explains distributional differences using semantic dimensions.

### What are Semantic Axes?

A semantic axis is a unit vector in SBERT embedding space pointing from one concept (negative pole, e.g., "professionalism") to another (positive pole, e.g., "casualness"). By projecting LLM outputs onto these axes, you can quantify and compare models along interpretable dimensions like formality, technicality, sentiment, etc.

**Key features:**
- Statistical rigor: Welch's t-test, Hedges' g effect sizes, Benjamini-Hochberg FDR correction
- Reuses embeddings: Works with the same SBERT embeddings used by quantum metrics
- Quality metrics: Separation ratio and intra-pole statistics validate axis quality
- Flexible: Use independently or as an interpretation layer after quantum detection

### Quick Example

```python
import numpy as np
from model_equality_testing.dataset import load_distribution
from model_equality_testing.src.embeddings import embed_sample
from model_equality_testing.src.semantic_axes import load_axis_from_jsonl, interpret_difference

# 1. Load samples from two model sources
dist_fp32 = load_distribution(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    prompt_ids={"wikipedia_en": [0, 1, 2]},
    L=500, source="fp32", load_in_unicode=True,
)
dist_int8 = load_distribution(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    prompt_ids={"wikipedia_en": [0, 1, 2]},
    L=500, source="int8", load_in_unicode=True,
)
sample_fp32 = dist_fp32.draw_completion_sample(n=100)
sample_int8 = dist_int8.draw_completion_sample(n=100)

# 2. Embed samples (reuse for both quantum metrics and semantic axes)
emb_fp32 = embed_sample(sample_fp32)
emb_int8 = embed_sample(sample_int8)

# 3. Load semantic axis from pole corpora (see semantic-pole project)
prof_axis = load_axis_from_jsonl(
    "/path/to/professionalism.jsonl",
    "/path/to/casualness.jsonl",
    "professionalism-casualness"
)

# 4. Interpret the difference
interp = interpret_difference(
    emb_fp32, emb_int8, [prof_axis],
    label_a="fp32", label_b="int8"
)
print(interp.summary())
```

**Output:**
```
Semantic Interpretation (1 axes, sorted by effect size):

  professionalism ← → casualness:  Δ = +0.54  g = 0.82  p < 0.001 *
    fp32 is more toward casualness than int8

  * significant after Benjamini-Hochberg correction (α = 0.05)
```

### Integration with Quantum Metrics

Semantic axes complement quantum-inspired metrics by providing interpretability:

```python
from model_equality_testing.src.quantum_metrics import fit_pca, pca_density_matrix, trace_distance

# 1. Detect difference with quantum metrics
pca = fit_pca(np.vstack([emb_fp32, emb_int8]), k=50)
rho_fp32 = pca_density_matrix(emb_fp32, pca)
rho_int8 = pca_density_matrix(emb_int8, pca)
td = trace_distance(rho_fp32, rho_int8)
print(f"Trace distance: {td:.4f}")  # e.g., 0.23 → models differ

# 2. Interpret what differs with semantic axes
interp = interpret_difference(emb_fp32, emb_int8, [prof_axis, form_axis, tech_axis],
                               label_a="fp32", label_b="int8")
print(interp.summary())  # Shows professionalism +0.54, formality +0.38, etc.
```

### Creating Semantic Axes

Semantic axes are built from **pole corpora** — collections of texts exemplifying each pole of a dimension. Use the [`semantic-pole`](https://github.com/yourusername/semantic-pole) project to generate pole corpora via LLM synthesis.

Each pole corpus is a JSONL file with records like:
```json
{"text": "Pursuant to our discussion...", "metadata": {"concept": "professionalism"}}
{"text": "With regard to the aforementioned...", "metadata": {"concept": "professionalism"}}
```

The `load_axis_from_jsonl()` function embeds these texts, L2-normalizes them, computes centroids, and constructs a unit vector axis with quality metrics (separation ratio, intra-pole variance).

For detailed guidance on axis design, interpretation, and quality metrics, see `SEMANTIC-AXES-GUIDE.md`.

## Citation

If you use our dataset or code, please cite this work as
```bibtex
@misc{gao2024model,
    title={Model Equality Testing: Which model is this API serving?},
    author={Gao, Irena and Liang, Percy and Guestrin, Carlos},
    journal={arXiv preprint},
    year={2024}
}
```