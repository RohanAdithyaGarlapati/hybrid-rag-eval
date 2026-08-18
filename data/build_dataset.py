"""Build the hybrid-rag-eval handbook dataset.

Emits three JSONL files into the same directory as this script:

* ``corpus.jsonl``        -- 40 hand-authored handbook documents on ML systems engineering.
* ``questions.jsonl``     -- 50 labeled questions, 25 "lexical" and 25 "paraphrase".
* ``unanswerable.jsonl``  -- 10 questions with no supporting document (abstention calibration).

Every question carries ``kind`` and a list of ``gold`` document ids. The builder
validates that every gold id exists in the corpus and refuses to write otherwise.

Run with:  python data/build_dataset.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Corpus: 40 documents. Each is a compact handbook entry rich in its own terms.
# ---------------------------------------------------------------------------

DOCS: list[dict[str, str]] = [
    {
        "id": "doc-inverted-index",
        "title": "Inverted Indexes",
        "text": (
            "An inverted index maps every term to a postings list of the documents that "
            "contain it. Each posting records the document id, the term frequency, and often "
            "the positions where the term appears. Query evaluation intersects or unions "
            "postings lists to find candidate documents quickly. The index is the backbone of "
            "any lexical search engine and is built once during ingestion and reused for all queries."
        ),
    },
    {
        "id": "doc-bm25",
        "title": "BM25 Ranking",
        "text": (
            "BM25 is a probabilistic ranking function that scores a document from the term "
            "frequency, the inverse document frequency, and a length normalization component. "
            "The k1 parameter controls term frequency saturation and the b parameter controls how "
            "strongly document length is normalized. Rare terms receive a higher inverse document "
            "frequency weight, so matching an uncommon query term contributes more to the score."
        ),
    },
    {
        "id": "doc-dense-retrieval",
        "title": "Dense Retrieval",
        "text": (
            "Dense retrieval encodes queries and documents into fixed length vectors with a neural "
            "encoder and ranks by vector similarity such as cosine. Because it matches on meaning "
            "rather than exact tokens, dense retrieval can find relevant passages that share no "
            "surface words with the query. It complements lexical search, which excels when the "
            "query and document use the same vocabulary."
        ),
    },
    {
        "id": "doc-rrf",
        "title": "Reciprocal Rank Fusion",
        "text": (
            "Reciprocal rank fusion combines several ranked lists into one by summing one over a "
            "constant k plus the rank of each item in each list. The constant k, commonly sixty, "
            "damps the influence of the very top positions and makes the fusion robust to score "
            "scale differences. Because it uses only ranks and not raw scores, reciprocal rank "
            "fusion needs no score calibration across retrievers."
        ),
    },
    {
        "id": "doc-chunking",
        "title": "Chunking Strategies",
        "text": (
            "Chunking splits a long document into smaller passages so that retrieval returns "
            "focused context. Fixed chunking cuts every N tokens, overlapping chunking slides a "
            "window with a configurable overlap so context is not lost at boundaries, and semantic "
            "chunking packs whole sentences up to a target size with a hard maximum. The chunk size "
            "trades recall against precision and directly affects the context budget."
        ),
    },
    {
        "id": "doc-recall-at-k",
        "title": "Recall at K",
        "text": (
            "Recall at k measures the fraction of relevant documents that appear in the top k "
            "retrieved results. It answers whether the retriever surfaced the gold document at all "
            "within a cutoff. Recall at k is a set membership metric and ignores the exact rank of "
            "the relevant item, so it is often paired with a rank aware metric for a fuller picture."
        ),
    },
    {
        "id": "doc-mrr",
        "title": "Mean Reciprocal Rank",
        "text": (
            "Mean reciprocal rank averages the reciprocal of the rank of the first relevant result "
            "across all queries. If the gold document is returned at rank one the reciprocal rank is "
            "one, at rank two it is one half, and so on. Mean reciprocal rank rewards putting the "
            "first correct answer as high as possible and is a natural metric for known item search."
        ),
    },
    {
        "id": "doc-ndcg",
        "title": "Normalized Discounted Cumulative Gain",
        "text": (
            "Normalized discounted cumulative gain rewards placing highly relevant documents near "
            "the top of the ranking. Each result contributes a gain discounted by a logarithm of its "
            "position, and the sum is normalized by the ideal ordering so scores fall between zero "
            "and one. Unlike recall, discounted cumulative gain is sensitive to the exact order of "
            "graded relevance labels."
        ),
    },
    {
        "id": "doc-run-variance",
        "title": "Run to Run Variance",
        "text": (
            "Run to run variance is the spread in a metric when the same evaluation is repeated with "
            "different random seeds, shuffles, or nondeterministic kernels. Ignoring this variance "
            "leads teams to celebrate noise as improvement. Reporting the mean together with a "
            "confidence interval over several runs distinguishes a real gain from random fluctuation."
        ),
    },
    {
        "id": "doc-mann-whitney",
        "title": "Mann-Whitney U Test",
        "text": (
            "The Mann-Whitney U test is a nonparametric test for whether one of two independent "
            "samples tends to produce larger values than the other. It ranks the pooled observations "
            "and compares the sum of ranks, making no assumption of normality. It is the unpaired "
            "counterpart to the Wilcoxon signed rank test and is useful when scores are skewed."
        ),
    },
    {
        "id": "doc-effect-size",
        "title": "Effect Size",
        "text": (
            "Effect size quantifies the magnitude of a difference independent of sample size, unlike "
            "a p value which only reports whether a difference is unlikely under the null hypothesis. "
            "The rank biserial correlation is a convenient effect size for rank based tests and is "
            "often labeled negligible, small, medium, or large by conventional thresholds."
        ),
    },
    {
        "id": "doc-holm-bonferroni",
        "title": "Holm-Bonferroni Correction",
        "text": (
            "The Holm-Bonferroni correction controls the family wise error rate when many hypotheses "
            "are tested at once. It sorts the p values in ascending order and multiplies each by the "
            "number of remaining tests in a step down fashion, enforcing monotonically nondecreasing "
            "adjusted values. It is uniformly more powerful than the plain Bonferroni correction."
        ),
    },
    {
        "id": "doc-bootstrap-ci",
        "title": "Bootstrap Confidence Intervals",
        "text": (
            "A bootstrap confidence interval resamples the observed data with replacement many times, "
            "recomputes the statistic on each resample, and reads the interval from percentiles of "
            "the resulting distribution. It needs no closed form and adapts to skewed statistics. "
            "Seeding the resampler makes the interval reproducible across runs."
        ),
    },
    {
        "id": "doc-paired-design",
        "title": "Paired Experimental Design",
        "text": (
            "A paired design evaluates two systems on the exact same queries so that per query "
            "difficulty cancels out. Because each query contributes a matched pair, a paired test has "
            "more statistical power than comparing two independent groups. Retrieval evaluation is "
            "naturally paired since every variant answers the identical question set."
        ),
    },
    {
        "id": "doc-llm-judge",
        "title": "LLM as Judge",
        "text": (
            "Using a language model as a judge scores generated answers along rubrics such as "
            "correctness or helpfulness when human labels are scarce. The judge is prompted with the "
            "question, the answer, and the supporting context and returns a graded verdict. Pinning "
            "the judge model and versioning its prompt are essential for reproducible scores."
        ),
    },
    {
        "id": "doc-faithfulness",
        "title": "Faithfulness",
        "text": (
            "Faithfulness measures whether every claim in a generated answer is supported by the "
            "retrieved context rather than invented. An unfaithful answer hallucinates facts that the "
            "sources never stated. Faithfulness is scored by checking each claim against the provided "
            "passages, and a low score signals grounding failures in the generation step."
        ),
    },
    {
        "id": "doc-answer-relevance",
        "title": "Answer Relevance",
        "text": (
            "Answer relevance measures how directly a response addresses the user question rather "
            "than drifting into tangential material. A relevant answer stays on topic and covers what "
            "was asked, while an irrelevant one may be factually true yet off point. It is scored "
            "separately from faithfulness because an answer can be grounded but still miss the question."
        ),
    },
    {
        "id": "doc-abstention",
        "title": "Abstention",
        "text": (
            "Abstention is the deliberate choice to decline to answer when retrieval confidence is "
            "too low, preferring to say I do not know over guessing. A good abstention policy fires on "
            "questions the corpus cannot support while rarely refusing answerable ones. The tradeoff "
            "between correct abstention and false abstention is tuned with a confidence threshold."
        ),
    },
    {
        "id": "doc-guardrails",
        "title": "Guardrails",
        "text": (
            "Guardrails are validation layers that constrain a system input and output to safe and "
            "expected forms. In retrieval they reject empty queries, cap result counts, and block "
            "answers when grounding is missing. Guardrails are enforced before and after generation so "
            "that malformed or unsupported responses never reach the user."
        ),
    },
    {
        "id": "doc-prompt-versioning",
        "title": "Prompt Versioning",
        "text": (
            "Prompt versioning tracks every change to a prompt template with an identifier so results "
            "can be tied to the exact wording that produced them. Without a pinned prompt version an "
            "evaluation cannot be reproduced, since a small edit can shift model behavior. Versioned "
            "prompts are stored alongside code and referenced in every report."
        ),
    },
    {
        "id": "doc-context-budget",
        "title": "Context Budgeting",
        "text": (
            "Context budgeting allocates the limited token window among the system prompt, retrieved "
            "passages, and the answer. Packing too many chunks wastes tokens and can bury the useful "
            "passage, while too few starves the model of evidence. A budget picks how many chunks and "
            "how large each may be to stay within the window."
        ),
    },
    {
        "id": "doc-lost-in-middle",
        "title": "Lost in the Middle",
        "text": (
            "The lost in the middle effect is the tendency of language models to attend less to "
            "information placed in the center of a long context. Relevant passages buried between many "
            "others are more likely to be ignored than those at the start or end. Ordering the "
            "strongest evidence first mitigates the effect and improves grounded answers."
        ),
    },
    {
        "id": "doc-reranking",
        "title": "Reranking",
        "text": (
            "Reranking applies a slower and more accurate model to reorder an initial candidate list "
            "from a fast retriever. A cross encoder scores each query and document pair jointly, "
            "capturing interactions that a single vector similarity misses. Reranking only the top "
            "candidates keeps latency bounded while sharply improving the final ordering."
        ),
    },
    {
        "id": "doc-ann-search",
        "title": "Approximate Nearest Neighbor Search",
        "text": (
            "Approximate nearest neighbor search finds vectors close to a query without scanning "
            "every point, trading a little recall for large speed gains. Graph indexes and inverted "
            "file structures prune the search space so latency stays low at scale. The approximation "
            "quality is tuned by parameters that balance recall against query time."
        ),
    },
    {
        "id": "doc-vector-store",
        "title": "Vector Stores",
        "text": (
            "A vector store persists embeddings together with metadata and serves similarity queries "
            "with optional filters. It supports adding, deleting, and searching vectors and often "
            "keeps them normalized so cosine similarity reduces to a dot product. Metadata filters let "
            "a query restrict results to a subset such as a tenant or a document type."
        ),
    },
    {
        "id": "doc-idempotent-ingestion",
        "title": "Idempotent Ingestion",
        "text": (
            "Idempotent ingestion guarantees that ingesting the same document twice leaves the index "
            "unchanged, usually by keying records on a stable content hash. This makes pipelines safe "
            "to retry after a failure without creating duplicates. Upserts replace an existing record "
            "in place so reprocessing is harmless."
        ),
    },
    {
        "id": "doc-columnar-storage",
        "title": "Columnar Storage",
        "text": (
            "Columnar storage lays out data by column rather than by row so that analytical scans read "
            "only the fields they need. Storing a column together enables strong compression and "
            "vectorized processing. Formats like this speed aggregations over large tables while "
            "penalizing row at a time point lookups."
        ),
    },
    {
        "id": "doc-latency-percentiles",
        "title": "Latency Percentiles",
        "text": (
            "Latency percentiles describe the tail of a response time distribution rather than only "
            "its average. The ninety fifth and ninety ninth percentiles reveal the slow requests that "
            "a mean hides, and the tail often dominates user experience. Tracking percentiles over "
            "time catches regressions that an average would mask."
        ),
    },
    {
        "id": "doc-caching",
        "title": "Caching",
        "text": (
            "Caching stores the result of an expensive computation so repeated requests are served "
            "without redoing the work. In retrieval, caching embeddings and frequent query results "
            "cuts latency and cost. A cache needs an eviction policy and a way to invalidate entries "
            "when the underlying index changes."
        ),
    },
    {
        "id": "doc-batching",
        "title": "Batching",
        "text": (
            "Batching groups many small requests into one larger call to amortize fixed overhead and "
            "raise throughput. Encoding a batch of texts through a model at once is far more efficient "
            "than one at a time because it saturates the hardware. Batching trades a little added "
            "latency for a large gain in overall throughput."
        ),
    },
    {
        "id": "doc-ci-gates",
        "title": "Continuous Integration Gates",
        "text": (
            "A continuous integration gate blocks a change from merging unless automated checks pass. "
            "For a retrieval system the gate can run the evaluation and fail the build if recall drops "
            "below a committed threshold. Gating on a quality metric prevents silent regressions from "
            "reaching production."
        ),
    },
    {
        "id": "doc-reproducible-reports",
        "title": "Reproducible Reports",
        "text": (
            "A reproducible report stamps every result with the commit hash, the dataset hash, and the "
            "configuration that produced it. Anyone can then regenerate the exact numbers from the "
            "same inputs. Reproducibility turns an evaluation from a one off claim into an auditable "
            "artifact."
        ),
    },
    {
        "id": "doc-seeding",
        "title": "Seeding Randomness",
        "text": (
            "Seeding fixes the state of every random number generator so a run can be repeated "
            "exactly. Without a seed, shuffles and resamples vary from run to run and results cannot "
            "be compared. A single controlled seed threaded through the pipeline is the foundation of "
            "reproducible experiments."
        ),
    },
    {
        "id": "doc-stemming",
        "title": "Stemming",
        "text": (
            "Stemming reduces inflected words to a common root so that related forms match during "
            "lexical search. A conservative suffix stemmer strips endings like ing, ed, and plural s "
            "while avoiding aggressive changes that merge unrelated words. Stemming raises recall by "
            "letting run match running and index match indexes."
        ),
    },
    {
        "id": "doc-tokenization",
        "title": "Tokenization",
        "text": (
            "Tokenization splits raw text into the units a model or index consumes, whether words, "
            "subwords, or characters. Consistent tokenization at ingestion and query time is essential "
            "or the same term will fail to match itself. Subword tokenization keeps the vocabulary "
            "small while still representing rare words."
        ),
    },
    {
        "id": "doc-query-expansion",
        "title": "Query Expansion",
        "text": (
            "Query expansion adds related terms or synonyms to a query so that documents using "
            "different wording are still retrieved. It bridges the vocabulary gap that hurts pure "
            "lexical matching. Expansion can be drawn from a thesaurus, from pseudo relevance "
            "feedback, or from a generative model."
        ),
    },
    {
        "id": "doc-index-compression",
        "title": "Index Compression",
        "text": (
            "Index compression shrinks postings lists with techniques like delta encoding of document "
            "ids and variable length integers. Smaller postings fit in memory and are faster to scan "
            "because less data moves from disk. Compression trades a little decode cost for a large "
            "reduction in index size."
        ),
    },
    {
        "id": "doc-sharding",
        "title": "Sharding and Replication",
        "text": (
            "Sharding partitions an index across machines so no single node holds all the data, while "
            "replication keeps copies of each shard for availability and read throughput. A query "
            "fans out to every shard and the partial results are merged. Together sharding and "
            "replication let a search system scale in both size and traffic."
        ),
    },
    {
        "id": "doc-quantization",
        "title": "Embedding Quantization",
        "text": (
            "Embedding quantization stores vectors at lower precision, such as eight bit integers or "
            "product quantized codes, to cut memory and speed distance computation. The compression "
            "introduces a small error that slightly lowers recall. Quantization is central to serving "
            "billions of embeddings within a fixed memory budget."
        ),
    },
    {
        "id": "doc-drift-monitoring",
        "title": "Data Drift Monitoring",
        "text": (
            "Data drift monitoring watches for shifts in the distribution of incoming queries or "
            "documents relative to the data a system was evaluated on. Undetected drift silently "
            "degrades retrieval quality as the world changes. Alerting on distribution distance lets a "
            "team retrain or reindex before users notice."
        ),
    },
]

# ---------------------------------------------------------------------------
# Lexical questions: wording deliberately overlaps the gold document's terms.
# ---------------------------------------------------------------------------

LEXICAL_QUESTIONS: list[dict] = [
    {"id": "lex-01", "text": "What does a postings list in an inverted index record for each term?", "gold": ["doc-inverted-index"]},
    {"id": "lex-02", "text": "How do the k1 and b parameters affect BM25 term frequency saturation and length normalization?", "gold": ["doc-bm25"]},
    {"id": "lex-03", "text": "How does dense retrieval encode queries and documents into vectors and rank by cosine similarity?", "gold": ["doc-dense-retrieval"]},
    {"id": "lex-04", "text": "Why does reciprocal rank fusion use the constant k of sixty when summing over ranks?", "gold": ["doc-rrf"]},
    {"id": "lex-05", "text": "What is the difference between fixed, overlapping, and semantic chunking strategies?", "gold": ["doc-chunking"]},
    {"id": "lex-06", "text": "What fraction does recall at k measure among the top k retrieved results?", "gold": ["doc-recall-at-k"]},
    {"id": "lex-07", "text": "How does mean reciprocal rank average the reciprocal of the first relevant rank?", "gold": ["doc-mrr"]},
    {"id": "lex-08", "text": "How does normalized discounted cumulative gain discount gain by the logarithm of position?", "gold": ["doc-ndcg"]},
    {"id": "lex-09", "text": "What is run to run variance across different random seeds and shuffles?", "gold": ["doc-run-variance"]},
    {"id": "lex-10", "text": "How does the Mann-Whitney U test rank pooled observations without assuming normality?", "gold": ["doc-mann-whitney"]},
    {"id": "lex-11", "text": "How does the rank biserial correlation label effect size as negligible, small, medium, or large?", "gold": ["doc-effect-size"]},
    {"id": "lex-12", "text": "How does the Holm-Bonferroni correction control the family wise error rate in a step down fashion?", "gold": ["doc-holm-bonferroni"]},
    {"id": "lex-13", "text": "How does a bootstrap confidence interval resample data with replacement and read percentiles?", "gold": ["doc-bootstrap-ci"]},
    {"id": "lex-14", "text": "Why does a paired design evaluate two systems on the same queries for more statistical power?", "gold": ["doc-paired-design"]},
    {"id": "lex-15", "text": "Why must the judge model be pinned and its prompt versioned when using an LLM as a judge?", "gold": ["doc-llm-judge"]},
    {"id": "lex-16", "text": "How is faithfulness scored by checking each claim against the retrieved context?", "gold": ["doc-faithfulness"]},
    {"id": "lex-17", "text": "How does the abstention policy trade correct abstention against false abstention with a confidence threshold?", "gold": ["doc-abstention"]},
    {"id": "lex-18", "text": "How does reranking use a cross encoder to score query and document pairs jointly?", "gold": ["doc-reranking"]},
    {"id": "lex-19", "text": "How does approximate nearest neighbor search trade recall for speed using graph indexes?", "gold": ["doc-ann-search"]},
    {"id": "lex-20", "text": "How does a vector store keep embeddings normalized so cosine similarity reduces to a dot product?", "gold": ["doc-vector-store"]},
    {"id": "lex-21", "text": "How does idempotent ingestion key records on a content hash to avoid duplicates?", "gold": ["doc-idempotent-ingestion"]},
    {"id": "lex-22", "text": "Why do the ninety fifth and ninety ninth latency percentiles reveal slow tail requests?", "gold": ["doc-latency-percentiles"]},
    {"id": "lex-23", "text": "How does a continuous integration gate fail the build if recall drops below a threshold?", "gold": ["doc-ci-gates"]},
    {"id": "lex-24", "text": "How does a conservative suffix stemmer strip endings like ing, ed, and plural s?", "gold": ["doc-stemming"]},
    {"id": "lex-25", "text": "Why does batching group many requests into one call to raise throughput?", "gold": ["doc-batching"]},
]

# ---------------------------------------------------------------------------
# Paraphrase questions: ask about the same concept while avoiding the gold
# document's distinctive vocabulary, so lexical overlap is minimal.
# ---------------------------------------------------------------------------

PARAPHRASE_QUESTIONS: list[dict] = [
    {"id": "par-01", "text": "Which lookup structure lets an engine jump straight to the files that mention a word?", "gold": ["doc-inverted-index"]},
    {"id": "par-02", "text": "Which classic keyword scorer weighs rare words more and dampens repeated ones?", "gold": ["doc-bm25"]},
    {"id": "par-03", "text": "How can a system find passages that mean the same thing even with no shared words?", "gold": ["doc-dense-retrieval"]},
    {"id": "par-04", "text": "What method blends several ordered result lists using only positions rather than raw scores?", "gold": ["doc-rrf"]},
    {"id": "par-05", "text": "How should a long article be cut into smaller pieces for focused lookup?", "gold": ["doc-chunking"]},
    {"id": "par-06", "text": "What tells you whether the right answer showed up anywhere in the first few hits?", "gold": ["doc-recall-at-k"]},
    {"id": "par-07", "text": "What single number rewards putting the first correct hit as high as possible?", "gold": ["doc-mrr"]},
    {"id": "par-08", "text": "Which graded quality score cares about the exact order of the best items near the top?", "gold": ["doc-ndcg"]},
    {"id": "par-09", "text": "Why can the same benchmark give slightly different outcomes each time you run it?", "gold": ["doc-run-variance"]},
    {"id": "par-10", "text": "Which distribution free comparison checks if one unpaired group tends to score higher?", "gold": ["doc-mann-whitney"]},
    {"id": "par-11", "text": "How do you express how big a difference is, separate from whether it is unlikely by chance?", "gold": ["doc-effect-size"]},
    {"id": "par-12", "text": "When running many comparisons at once, how do you keep the chance of a false alarm in check?", "gold": ["doc-holm-bonferroni"]},
    {"id": "par-13", "text": "How can you build an uncertainty range by drawing samples over and over from your own data?", "gold": ["doc-bootstrap-ci"]},
    {"id": "par-14", "text": "Why is it stronger to test two approaches on the identical set of questions?", "gold": ["doc-paired-design"]},
    {"id": "par-15", "text": "How can an automated model grade generated responses when human labels are missing?", "gold": ["doc-llm-judge"]},
    {"id": "par-16", "text": "How do you tell whether a written reply invented facts the sources never contained?", "gold": ["doc-faithfulness"]},
    {"id": "par-17", "text": "How do you tell whether a reply actually answered what was asked instead of wandering?", "gold": ["doc-answer-relevance"]},
    {"id": "par-18", "text": "When should a system say it does not know rather than guess an unsupported reply?", "gold": ["doc-abstention"]},
    {"id": "par-19", "text": "What validation layers reject bad input and block ungrounded output before it reaches a user?", "gold": ["doc-guardrails"]},
    {"id": "par-20", "text": "Why should every wording change to a template be tracked with an identifier for reproducibility?", "gold": ["doc-prompt-versioning"]},
    {"id": "par-21", "text": "How do you decide how many passages fit inside a limited model window?", "gold": ["doc-context-budget"]},
    {"id": "par-22", "text": "Why do models often overlook evidence placed in the center of a very long prompt?", "gold": ["doc-lost-in-middle"]},
    {"id": "par-23", "text": "How do column oriented layouts speed up scans that touch only a few fields?", "gold": ["doc-columnar-storage"]},
    {"id": "par-24", "text": "How can storing an expensive result avoid repeating the same work on later requests?", "gold": ["doc-caching"]},
    {"id": "par-25", "text": "Why fix the state of the random generators before starting an experiment?", "gold": ["doc-seeding"]},
]

# ---------------------------------------------------------------------------
# Unanswerable questions: plausible but outside the corpus, for abstention.
# ---------------------------------------------------------------------------

UNANSWERABLE_QUESTIONS: list[dict] = [
    {"id": "una-01", "text": "How does the credit assignment problem work in deep reinforcement learning?", "gold": []},
    {"id": "una-02", "text": "What noise schedule should I use to train a diffusion image generator?", "gold": []},
    {"id": "una-03", "text": "How do I fine tune a large language model with low rank adapters?", "gold": []},
    {"id": "una-04", "text": "What is the best way to stabilize training of a generative adversarial network?", "gold": []},
    {"id": "una-05", "text": "How does a convolutional neural network detect edges in an image?", "gold": []},
    {"id": "una-06", "text": "What database isolation level prevents phantom reads in a payments ledger?", "gold": []},
    {"id": "una-07", "text": "How do I configure Kubernetes horizontal pod autoscaling for a web service?", "gold": []},
    {"id": "una-08", "text": "What is the time complexity of Dijkstra's shortest path algorithm with a heap?", "gold": []},
    {"id": "una-09", "text": "How does gradient clipping prevent exploding gradients in recurrent networks?", "gold": []},
    {"id": "una-10", "text": "What are the tradeoffs between OAuth and SAML for enterprise single sign on?", "gold": []},
]


def _validate(corpus_ids: set[str], questions: list[dict]) -> None:
    for q in questions:
        for gid in q["gold"]:
            if gid not in corpus_ids:
                raise ValueError(f"Question {q['id']} references unknown gold id {gid!r}")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(out_dir: Path | None = None) -> dict[str, Path]:
    out_dir = out_dir or Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_ids = {d["id"] for d in DOCS}
    if len(corpus_ids) != len(DOCS):
        raise ValueError("Duplicate document ids in corpus")

    lexical = [dict(q, kind="lexical") for q in LEXICAL_QUESTIONS]
    paraphrase = [dict(q, kind="paraphrase") for q in PARAPHRASE_QUESTIONS]
    unanswerable = [dict(q, kind="unanswerable") for q in UNANSWERABLE_QUESTIONS]
    answerable = lexical + paraphrase

    # Hard validation: every gold id must exist.
    _validate(corpus_ids, answerable)

    if len(DOCS) != 40:
        raise ValueError(f"Expected 40 documents, found {len(DOCS)}")
    if len(lexical) != 25 or len(paraphrase) != 25:
        raise ValueError("Expected 25 lexical and 25 paraphrase questions")
    if len(unanswerable) != 10:
        raise ValueError("Expected 10 unanswerable questions")

    corpus_path = out_dir / "corpus.jsonl"
    questions_path = out_dir / "questions.jsonl"
    unanswerable_path = out_dir / "unanswerable.jsonl"

    _write_jsonl(corpus_path, DOCS)
    _write_jsonl(questions_path, answerable)
    _write_jsonl(unanswerable_path, unanswerable)

    return {
        "corpus": corpus_path,
        "questions": questions_path,
        "unanswerable": unanswerable_path,
    }


def dataset_hash(corpus_path: Path) -> str:
    return hashlib.sha256(corpus_path.read_bytes()).hexdigest()


def main() -> int:
    paths = build()
    h = dataset_hash(paths["corpus"])
    print(f"Wrote {len(DOCS)} documents to {paths['corpus'].name}")
    print(f"Wrote {len(LEXICAL_QUESTIONS) + len(PARAPHRASE_QUESTIONS)} answerable questions "
          f"({len(LEXICAL_QUESTIONS)} lexical, {len(PARAPHRASE_QUESTIONS)} paraphrase) to {paths['questions'].name}")
    print(f"Wrote {len(UNANSWERABLE_QUESTIONS)} unanswerable questions to {paths['unanswerable'].name}")
    print(f"corpus sha256: {h}")
    print("All gold ids validated against the corpus.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
