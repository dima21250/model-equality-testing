Model comparison (model A: meta-llama/Meta-Llama-3-8B-Instruct [fp32] vs model B: mistralai/Mistral-7B-Instruct-v0.3 [fp32]):
Loading dataset from source: fp32
Prompt IDs: {'wikipedia_en': [0, 1, 2]}
Initializing DistributionFromDataset; loading k...
	Time to load distribution 0 w/ 15000 entries: 1.0995006669982104
	Time to load distribution 1 w/ 15000 entries: 0.9313212909910362
	Time to load distribution 2 w/ 15000 entries: 1.1018825409992132
Done initializing
Loading dataset from source: fp32
Prompt IDs: {'wikipedia_en': [0, 1, 2]}
Initializing DistributionFromDataset; loading k...
	Time to load distribution 0 w/ 15000 entries: 1.0935248750029132
	Time to load distribution 1 w/ 15000 entries: 1.1560282499995083
	Time to load distribution 2 w/ 15000 entries: 1.2234595409972826
Done initializing
  MMD statistic = 0.085477, p‑value = 0.0000, elapsed = 613.603s
  KS  statistic = 0.302000, p‑value = 0.0000, elapsed = 0.145s

