import numpy as np

from hybridrag.embeddings import Embedder, HashingEmbedder, build_embedder


def test_hashing_embedder_is_l2_normalized():
    emb = HashingEmbedder(dimension=256)
    vecs = emb.encode(["inverted index", "bm25 ranking", "dense retrieval"])
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-9)


def test_hashing_embedder_deterministic():
    emb = HashingEmbedder(dimension=256)
    a = emb.encode(["reproducible seeding"])
    b = emb.encode(["reproducible seeding"])
    assert np.array_equal(a, b)


def test_hashing_embedder_reports_not_semantic():
    emb = HashingEmbedder()
    assert emb.is_semantic is False
    assert emb.dimension > 0
    assert isinstance(emb, Embedder)  # satisfies the runtime checkable protocol


def test_hashing_embedder_dimension_respected():
    emb = HashingEmbedder(dimension=128)
    vecs = emb.encode(["a", "b"])
    assert vecs.shape == (2, 128)


def test_similar_texts_more_similar_than_dissimilar():
    emb = HashingEmbedder(dimension=512)
    v = emb.encode(["inverted index postings list", "inverted index posting lists", "banana smoothie recipe"])
    sim_close = float(v[0] @ v[1])
    sim_far = float(v[0] @ v[2])
    assert sim_close > sim_far


def test_build_embedder_auto_falls_back_without_warning_crash():
    # sentence-transformers is not installed here, so auto must fall back cleanly.
    emb = build_embedder("auto")
    assert emb.dimension > 0
    # Either it loaded the real model (semantic) or fell back to hashing (not semantic).
    assert isinstance(emb.is_semantic, bool)


def test_build_embedder_hashing_explicit():
    emb = build_embedder("hashing", dimension=64)
    assert emb.is_semantic is False
    assert emb.dimension == 64


def test_empty_encode_returns_empty_matrix():
    emb = HashingEmbedder(dimension=32)
    out = emb.encode([])
    assert out.shape == (0, 32)
