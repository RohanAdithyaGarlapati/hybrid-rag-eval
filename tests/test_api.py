import warnings

from fastapi.testclient import TestClient  # TestClient is httpx based

from hybridrag.api import app

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    client = TestClient(app)


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["n_chunks"] > 0
    assert "is_semantic" in body


def test_search_returns_hits():
    resp = client.post("/search", json={"query": "inverted index postings list", "k": 5, "mode": "lexical"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "lexical"
    assert len(body["hits"]) >= 1
    assert body["hits"][0]["rank"] == 1


def test_search_rejects_empty_query():
    resp = client.post("/search", json={"query": "", "k": 5})
    assert resp.status_code == 422  # Pydantic min_length validation


def test_search_rejects_bad_mode():
    resp = client.post("/search", json={"query": "hello", "mode": "banana"})
    assert resp.status_code == 422


def test_answer_endpoint_returns_context():
    resp = client.post("/answer", json={"query": "how does bm25 rank documents", "k": 5, "mode": "hybrid-rrf"})
    assert resp.status_code == 200
    body = resp.json()
    assert "context" in body
    assert "doc_ids" in body
