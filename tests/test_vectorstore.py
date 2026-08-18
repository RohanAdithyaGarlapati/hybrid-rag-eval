import numpy as np

from hybridrag.vectorstore import VectorStore


def _store():
    store = VectorStore(dimension=3)
    store.add(
        ids=["a", "b", "c"],
        vectors=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
        metadata=[{"g": "x"}, {"g": "y"}, {"g": "x"}],
    )
    return store


def test_cosine_search_orders_by_similarity():
    store = _store()
    hits = store.search(np.array([1.0, 0.0, 0.0]), k=3)
    assert hits[0].id == "a"
    assert abs(hits[0].score - 1.0) < 1e-9


def test_search_respects_k():
    store = _store()
    assert len(store.search(np.array([1.0, 0.0, 0.0]), k=2)) == 2


def test_filtered_search_only_returns_matches():
    store = _store()
    hits = store.filtered_search(np.array([1.0, 1.0, 0.0]), k=5, predicate=lambda m: m["g"] == "x")
    assert {h.id for h in hits} <= {"a", "c"}


def test_add_rejects_wrong_dimension():
    store = VectorStore(dimension=3)
    try:
        store.add(["a"], np.array([[1.0, 0.0]]))
        raised = False
    except ValueError:
        raised = True
    assert raised
