# hybrid-rag-eval

A hybrid (lexical plus dense) retrieval system with a statistical evaluation harness,
built to be read as a portfolio piece: every module is small, tested, and honest about
what it does. It ships a hand authored 40 document handbook corpus on ML systems
engineering, a 50 question labeled query set split into lexical and paraphrase halves,
10 unanswerable questions for abstention calibration, and an experiment grid that reports
recall, MRR, and nDCG with paired significance testing and multiple comparison correction.

## Honest statement on the reported numbers

The evaluation supports two embedders:

* `SentenceTransformerEmbedder` wrapping `all-MiniLM-L6-v2` (semantic, `is_semantic=True`).
* `HashingEmbedder`, a dependency free character n gram hashing fallback (`is_semantic=False`).

**The numbers committed in `reports/experiment_report.md` were produced by the non semantic
`HashingEmbedder` fallback.** The environment used to build and validate this repository had
no access to download the `sentence-transformers` weights, so `build_embedder("auto")` warned
loudly and fell back to hashing, exactly as designed. The harness ran fully end to end on the
fallback: dataset build, 75 passing tests, the complete experiment grid, and the CI quality gate.

Because the fallback matches on character overlap rather than meaning, the **paraphrase split
is weaker than it would be with a real dense encoder**. That is expected and is not a bug. It
shows up clearly in the by kind table: the lexical split is essentially solved while dense
retrieval on paraphrases lags. To reproduce with the semantic model, install the extra and run
on a machine with network access to the model hub:

```bash
pip install -e ".[semantic]"
python scripts/eval_gate.py   # build_embedder("auto") will now prefer all-MiniLM-L6-v2
```

The report is always stamped with the embedder name and its `is_semantic` flag, so no run can be
mistaken for the other.

### Headline numbers from the committed run (HashingEmbedder fallback, seed 0)

Overall recall@5 / MRR / nDCG@10 for a few variants:

| chunking | mode | recall@5 | MRR | nDCG@10 |
|---|---|---|---|---|
| overlapping | lexical (BM25 baseline) | 0.980 | 0.940 | 0.954 |
| overlapping | hybrid-rrf | 0.940 | 0.878 | 0.902 |
| overlapping | dense | 0.880 | 0.847 | 0.873 |
| semantic | lexical | 0.980 | 0.969 | 0.976 |

By question kind (recall@5), overlapping chunking: lexical questions 1.000, paraphrase questions
0.960 for BM25 and 0.760 for dense. Abstention operating point: threshold 0.260 on dense cosine
similarity, catching 100 percent of unanswerable questions at a 26 percent false abstention rate.
See `reports/experiment_report.md` for the full grid, the paired Wilcoxon comparisons with
Holm-Bonferroni correction, and bootstrap confidence intervals.

With BM25 as such a strong lexical baseline on this corpus, none of the fallback variants beat it
after Holm correction. That is the honest result for a non semantic embedder and is precisely the
gap a real dense model is expected to close on the paraphrase split.

## Architecture

```
data/build_dataset.py      Hand authored corpus + questions; validates every gold id.
src/hybridrag/
  chunking.py              fixed / overlapping / semantic chunking -> frozen Chunk dataclasses.
  bm25.py                  Inverted index + BM25 from scratch, positional phrase search.
  embeddings.py            Embedder protocol; SentenceTransformer + Hashing; build_embedder("auto").
  vectorstore.py           L2 normalized matrix, cosine search via argpartition, metadata filter.
  fusion.py                Reciprocal rank fusion and normalized score fusion.
  retriever.py             HybridRetriever (lexical/dense/hybrid), abstention, context builder.
  metrics.py               recall@k, hit@k, precision@k, MRR, nDCG, per query + aggregate.
  stats.py                 Wilcoxon (MWU fallback), rank biserial, bootstrap CI, Holm-Bonferroni.
  judge.py                 Anthropic LLM as judge (pinned model + versioned prompt), skips w/o key.
  evaluate.py              Experiment grid, stats, abstention sweep, Markdown report.
  api.py                   FastAPI /health /search /answer with Pydantic validation.
  cli.py                   index / query / evaluate subcommands.
tests/                     75 tests (chunking, bm25, embeddings, fusion, metrics, stats, retriever, api).
scripts/eval_gate.py       CI quality gate on recall@5.
.github/workflows/eval.yml Lint + tests + evaluation gate.
```

Key design choices worth calling out:

* **Abstention is thresholded on dense cosine similarity, not on the fused RRF score.** RRF scores
  are a function of rank position only and carry no absolute confidence, so they are the wrong
  signal for deciding whether to answer. The dense cosine of the best chunk is a real confidence.
* **Context is built strongest first** to mitigate the lost in the middle effect.
* **The candidate pool is wider than k** before fusion, so fusion has material to reorder.
* **Everything reproducible is seeded and stamped**: the report carries the commit SHA, the dataset
  SHA256, and the embedder identity.

## Requirements and Python version

The project targets **Python 3.12** (`requires-python = ">=3.12"` in `pyproject.toml`). The sandbox
used to author and validate it ran **CPython 3.10** because 3.12 could not be fetched there; the
code is written to run on 3.10+ and the test suite passes on it. On 3.12 nothing changes. Tests use
`pythonpath = ["src"]` so no install is required to run them.

## How to run

```bash
# 1. Install (dev tools included)
pip install -e ".[dev]"

# 2. Build the dataset (writes corpus.jsonl, questions.jsonl, unanswerable.jsonl)
python data/build_dataset.py

# 3. Run the tests
pytest -q

# 4. Run the full evaluation and write the report
python -m hybridrag.evaluate --out reports/experiment_report.md

# 5. Run the CI quality gate locally
python scripts/eval_gate.py

# CLI
python -m hybridrag.cli index --strategy overlapping
python -m hybridrag.cli query "how does bm25 rank documents" --mode hybrid-rrf
python -m hybridrag.cli evaluate

# API
uvicorn hybridrag.api:app --reload    # then GET /health, POST /search, POST /answer
```

## LLM as judge

`judge.py` scores faithfulness and answer relevance through the Anthropic API against a pinned model
(`claude-3-5-sonnet-20241022`) and a versioned prompt (`faithfulness-relevance-v1`). It reads
`ANTHROPIC_API_KEY` and, when the key or SDK is absent, reports itself unavailable and skips cleanly
rather than failing. No judge call is made in the committed evaluation run.

## License

MIT.
