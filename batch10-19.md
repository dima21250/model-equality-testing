# KS Sanity Check Results (All Prompts)

**Model:** meta-llama/Meta-Llama-3-8B-Instruct
**Dataset:** wikipedia_en
**Prompt IDs:** 10–19 (total 10)
**Sources:** fp32, nf4, int8, watermark
**Completion length (L):** 200
**Samples per comparison:** 100

| Prompt ID | Source | KS statistic | p-value | Note |
|-----------|--------|--------------|---------|------|
| 10 | fp32 | 0.100000 | 0.702057 |  |
| 10 | nf4 | 0.190000 | 0.053902 |  |
| 10 | int8 | 0.330000 | 0.000032 | ⚠️ Low p-value (possible false rejection) |
| 10 | watermark | 0.160000 | 0.154839 |  |
| 11 | fp32 | 0.070000 | 0.968410 |  |
| 11 | nf4 | 0.050000 | 0.999689 |  |
| 11 | int8 | 0.130000 | 0.368188 |  |
| 11 | watermark | 0.100000 | 0.702057 |  |
| 12 | fp32 | 0.060000 | 0.994236 |  |
| 12 | nf4 | 0.090000 | 0.815415 |  |
| 12 | int8 | 0.090000 | 0.815415 |  |
| 12 | watermark | 0.080000 | 0.908411 |  |
| 13 | fp32 | 0.060000 | 0.994236 |  |
| 13 | nf4 | 0.080000 | 0.908411 |  |
| 13 | int8 | 0.200000 | 0.036384 | ⚠️ Low p-value (possible false rejection) |
| 13 | watermark | 0.080000 | 0.908411 |  |
| 14 | fp32 | 0.120000 | 0.469506 |  |
| 14 | nf4 | 0.080000 | 0.908411 |  |
| 14 | int8 | 0.100000 | 0.702057 |  |
| 14 | watermark | 0.100000 | 0.702057 |  |
| 15 | fp32 | 0.150000 | 0.211170 |  |
| 15 | nf4 | 0.070000 | 0.968410 |  |
| 15 | int8 | 0.090000 | 0.815415 |  |
| 15 | watermark | 0.090000 | 0.815415 |  |
| 16 | fp32 | 0.110000 | 0.583009 |  |
| 16 | nf4 | 0.080000 | 0.908411 |  |
| 16 | int8 | 0.080000 | 0.908411 |  |
| 16 | watermark | 0.110000 | 0.583009 |  |
| 17 | fp32 | 0.090000 | 0.815415 |  |
| 17 | nf4 | 0.240000 | 0.006134 | ⚠️ Low p-value (possible false rejection) |
| 17 | int8 | 0.080000 | 0.908411 |  |
| 17 | watermark | 0.150000 | 0.211170 |  |
| 18 | fp32 | 0.050000 | 0.999689 |  |
| 18 | nf4 | 0.090000 | 0.815415 |  |
| 18 | int8 | 0.110000 | 0.583009 |  |
| 18 | watermark | 0.100000 | 0.702057 |  |
| 19 | fp32 | 0.110000 | 0.583009 |  |
| 19 | nf4 | 0.150000 | 0.211170 |  |
| 19 | int8 | 0.120000 | 0.469506 |  |
| 19 | watermark | 0.070000 | 0.968410 |  |

## ⚠️ Warning: Some comparisons yielded p < 0.05
This may indicate false rejections of the null hypothesis.
Consider increasing `--samples` or checking random seeds.
