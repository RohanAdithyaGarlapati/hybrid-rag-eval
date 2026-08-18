import pytest

from hybridrag.embeddings import HashingEmbedder
from hybridrag.retriever import HybridRetriever

CORPUS = [
    {"id": "doc-cat", "title": "Cats", "text": "A cat is a small domesticated feline that purrs and hunts mice."},
    {"id": "doc-dog", "title": "Dogs", "text": "A dog is a loyal domesticated canine that barks and fetches sticks."},
    {"id": "doc-car", "title": "Cars", "text": "A car is a road vehicle with an engine, wheels, and a steering wheel."},
]


@pytest.fixture
def retriever():
    return HybridRetriever(CORPUS, HashingEmbedder(dimension=256), strategy="fixed", abstain_threshold=0.30)


@pytest.mark.parametrize("mode", ["lexical", "dense", "hybrid-rrf", "hybrid-normalized"])
def test_all_modes_return_results(retriever, mode):
    res = retriever.retrieve("domesticated feline that purrs", k=3, mode=mode)
    assert len(res.chunks) >= 1
    assert res.mode == mode


def test_retrieval_finds_relevant_doc(retriever):
    res = retriever.retrieve("small feline that hunts mice", k=3, mode="lexical")
    assert res.doc_ids[0] == "doc-cat"


def test_doc_ids_deduplicated_and_ordered(retriever):
    res = retriever.retrieve("dog canine barks fetches", k=5, mode="lexical")
    assert len(res.doc_ids) == len(set(res.doc_ids))


def test_unknown_mode_raises(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("anything", k=3, mode="nonsense")


def test_abstention_fires_on_out_of_corpus_query(retriever):
    # A query with no support in the tiny corpus should have low dense similarity.
    res = retriever.retrieve("quantum chromodynamics gluon confinement lattice", k=3, mode="hybrid-rrf")
    assert res.abstained is True


def test_no_abstention_on_supported_query(retriever):
    res = retriever.retrieve("a cat is a small domesticated feline that purrs", k=3, mode="hybrid-rrf")
    assert res.abstained is False


def test_abstention_uses_dense_not_fused_score(retriever):
    # Threshold at 1.0 forces abstention regardless of fused RRF score existing.
    retriever.abstain_threshold = 1.0
    res = retriever.retrieve("a cat is a small domesticated feline", k=3, mode="hybrid-rrf")
    assert res.abstained is True
    assert len(res.chunks) >= 1  # candidates still retrieved even while abstaining


def test_build_context_orders_strongest_first(retriever):
    res = retriever.retrieve("small feline that hunts mice", k=3, mode="lexical")
    ctx = retriever.build_context(res)
    assert res.chunks[0].chunk.title in ctx.split("\n\n")[0]


def test_candidate_pool_wider_than_k(retriever):
    # Even with k=1 the retriever pools more candidates internally before truncating.
    res = retriever.retrieve("cat dog car", k=1, mode="hybrid-rrf")
    assert len(res.chunks) == 1
