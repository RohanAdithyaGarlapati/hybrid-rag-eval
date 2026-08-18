# hybrid-rag-eval experiment report

- Commit SHA: `unknown`
- Dataset SHA256: `878b47f05f8b6a13b0e131d3f73f82fd8fde3b500d98340c8f31cbe84b218996`
- Embedder: `sentence-transformers/all-MiniLM-L6-v2` (dimension 384)
- Embedder is_semantic: **True**
- Answerable questions: 50  |  Unanswerable: 10
- Baseline: `overlapping / lexical` (BM25)
- Paired comparison metric: `mrr`  |  seed: 0

## Overall metrics by variant

| chunking | mode | recall@1 | recall@3 | recall@5 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|---|
| fixed | lexical | 0.920 | 0.940 | 0.980 | 1.000 | 0.938 | 0.952 |
| fixed | dense | 0.800 | 0.980 | 0.980 | 1.000 | 0.886 | 0.915 |
| fixed | hybrid-rrf | 0.920 | 0.980 | 1.000 | 1.000 | 0.947 | 0.960 |
| fixed | hybrid-normalized | 0.920 | 0.960 | 1.000 | 1.000 | 0.946 | 0.959 |
| overlapping | lexical | 0.920 | 0.940 | 0.980 | 1.000 | 0.940 | 0.954 |
| overlapping | dense | 0.840 | 0.980 | 0.980 | 1.000 | 0.910 | 0.933 |
| overlapping | hybrid-rrf | 0.900 | 0.980 | 1.000 | 1.000 | 0.937 | 0.953 |
| overlapping | hybrid-normalized | 0.920 | 0.960 | 1.000 | 1.000 | 0.946 | 0.959 |
| semantic | lexical | 0.960 | 0.980 | 0.980 | 1.000 | 0.969 | 0.976 |
| semantic | dense | 0.860 | 0.960 | 0.980 | 1.000 | 0.915 | 0.936 |
| semantic | hybrid-rrf | 0.940 | 0.980 | 1.000 | 1.000 | 0.961 | 0.970 |
| semantic | hybrid-normalized | 0.960 | 0.960 | 1.000 | 1.000 | 0.970 | 0.977 |

## By question kind (recall@5 / MRR)

| chunking | mode | lexical recall@5 | lexical MRR | paraphrase recall@5 | paraphrase MRR |
|---|---|---|---|---|---|
| fixed | lexical | 1.000 | 1.000 | 0.960 | 0.875 |
| fixed | dense | 1.000 | 0.980 | 0.960 | 0.792 |
| fixed | hybrid-rrf | 1.000 | 1.000 | 1.000 | 0.895 |
| fixed | hybrid-normalized | 1.000 | 1.000 | 1.000 | 0.891 |
| overlapping | lexical | 1.000 | 1.000 | 0.960 | 0.880 |
| overlapping | dense | 1.000 | 0.980 | 0.960 | 0.840 |
| overlapping | hybrid-rrf | 1.000 | 1.000 | 1.000 | 0.875 |
| overlapping | hybrid-normalized | 1.000 | 1.000 | 1.000 | 0.891 |
| semantic | lexical | 1.000 | 1.000 | 0.960 | 0.937 |
| semantic | dense | 1.000 | 0.980 | 0.960 | 0.850 |
| semantic | hybrid-rrf | 1.000 | 1.000 | 1.000 | 0.921 |
| semantic | hybrid-normalized | 1.000 | 1.000 | 1.000 | 0.940 |

## Paired comparisons vs BM25 baseline (Holm corrected)

| variant | test | p (raw) | p (Holm) | effect r | label | mean diff | 95% CI |
|---|---|---|---|---|---|---|---|
| semantic / hybrid-normalized | wilcoxon | 0.068 | 0.747 | 1.000 | large | 0.030 | [0.001, 0.072] |
| fixed / lexical | wilcoxon | 0.655 | 1.000 | -0.333 | medium | -0.002 | [-0.010, 0.003] |
| fixed / dense | wilcoxon | 0.106 | 1.000 | -0.564 | large | -0.054 | [-0.117, 0.010] |
| fixed / hybrid-rrf | wilcoxon | 0.581 | 1.000 | 0.300 | medium | 0.007 | [-0.023, 0.037] |
| fixed / hybrid-normalized | wilcoxon | 0.180 | 1.000 | 1.000 | large | 0.006 | [0.000, 0.014] |
| overlapping / dense | wilcoxon | 0.507 | 1.000 | -0.244 | small | -0.030 | [-0.097, 0.040] |
| overlapping / hybrid-rrf | wilcoxon | 1.000 | 1.000 | 0.000 | negligible | -0.003 | [-0.027, 0.017] |
| overlapping / hybrid-normalized | wilcoxon | 0.180 | 1.000 | 1.000 | large | 0.006 | [0.000, 0.014] |
| semantic / lexical | wilcoxon | 0.109 | 1.000 | 1.000 | large | 0.029 | [0.000, 0.071] |
| semantic / dense | wilcoxon | 0.511 | 1.000 | -0.244 | small | -0.025 | [-0.092, 0.044] |
| semantic / hybrid-rrf | wilcoxon | 0.357 | 1.000 | 0.500 | large | 0.021 | [-0.020, 0.067] |

## Abstention operating point

- Chosen threshold (dense cosine): **0.387**
- Correct abstention rate (unanswerable caught): 1.000
- False abstention rate (answerable wrongly refused): 0.060
- Youden J (correct minus false): 0.940

Abstention is thresholded on dense cosine similarity, not on the fused RRF score, because RRF scores are position only and carry no absolute confidence signal.
