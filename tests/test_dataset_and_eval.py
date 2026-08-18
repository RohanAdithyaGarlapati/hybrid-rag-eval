import warnings

from hybridrag.dataset import load_dataset
from hybridrag.embeddings import HashingEmbedder
from hybridrag.evaluate import CHUNKINGS, MODES, render_report, run_experiment
from hybridrag.judge import AnthropicJudge


def test_dataset_shapes_and_gold_ids():
    ds = load_dataset()
    corpus_ids = {d["id"] for d in ds["corpus"]}
    assert len(ds["corpus"]) == 40
    assert len(ds["questions"]) == 50
    assert len(ds["unanswerable"]) == 10
    kinds = {q["kind"] for q in ds["questions"]}
    assert kinds == {"lexical", "paraphrase"}
    # Every gold id must exist in the corpus.
    for q in ds["questions"]:
        for gid in q["gold"]:
            assert gid in corpus_ids


def test_experiment_runs_and_reports_all_variants():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_experiment(embedder=HashingEmbedder(dimension=256), seed=0)
    # Grid completeness.
    for strategy in CHUNKINGS:
        for mode in MODES:
            assert (strategy, mode) in results["variants"]
    # A sanity floor: the best lexical variant should retrieve well on this corpus.
    best_recall5 = max(results["variants"][(s, "lexical")].overall["recall@5"] for s in CHUNKINGS)
    assert best_recall5 > 0.7
    # Report renders and is stamped with provenance.
    report = render_report(results)
    assert "Dataset SHA256" in report
    assert "is_semantic" in report


def test_judge_skips_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    judge = AnthropicJudge()
    assert judge.available is False
    assert judge.score("q", "context", "answer") is None
    assert judge.reason_unavailable is not None
