Model comparison (model A: meta-llama/Meta-Llama-3-8B-Instruct [fp32] vs model B: mistralai/Mistral-7B-Instruct-v0.3 [fp32]):
Loading dataset from source: fp32
Prompt IDs: {'wikipedia_en': [0, 1, 2]}
Initializing DistributionFromDataset; loading k...
	Time to load distribution 0 w/ 15000 entries: 1.0925608749967068
	Time to load distribution 1 w/ 15000 entries: 0.9286687080020783
	Time to load distribution 2 w/ 15000 entries: 1.1031718750018626
Done initializing
Loading dataset from source: fp32
Prompt IDs: {'wikipedia_en': [0, 1, 2]}
Initializing DistributionFromDataset; loading k...
	Time to load distribution 0 w/ 15000 entries: 1.1049844580120407
	Time to load distribution 1 w/ 15000 entries: 1.1618619169894373
	Time to load distribution 2 w/ 15000 entries: 1.2269343750085682
Done initializing
  MMD statistic = 0.085517, p‑value = 0.0000, elapsed = 616.160s
  KS  statistic = 0.263000, p‑value = 0.0000, elapsed = 0.146s

