import math

from hybridrag.bm25 import BM25Index, stem, tokenize


def _index():
    idx = BM25Index()
    idx.add("a", "the cat sat on the mat")
    idx.add("b", "the dog sat on the log")
    idx.add("c", "cats and dogs are common pets in many homes today indeed")
    idx.add("d", "quantum entanglement is a physics phenomenon")
    return idx


def test_stemmer_conservative():
    assert stem("running") == "runn"  # strips -ing
    assert stem("indexes") == "index"  # strips -es
    assert stem("cats") == "cat"  # strips plural -s
    assert stem("class") == "class"  # never strips -ss
    assert stem("run") == "run"  # short words untouched


def test_tokenize_removes_stopwords_and_stems():
    toks = tokenize("The cats are running")
    assert "the" not in toks and "are" not in toks
    assert "cat" in toks and "runn" in toks


def test_idf_monotonic_in_document_frequency():
    idx = _index()
    # "quantum" appears in 1 doc, "sat" appears in 2 docs -> quantum rarer -> higher idf.
    assert idx.idf(stem("quantum")) > idx.idf(stem("sat"))


def test_idf_floored_at_zero():
    idx = _index()
    for term in idx.postings:
        assert idx.idf(term) >= 0.0


def test_idf_smoothing_formula():
    idx = _index()
    term = stem("quantum")
    n, df = idx.n_docs, idx.document_frequency(term)
    expected = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
    assert abs(idx.idf(term) - expected) < 1e-12


def test_length_normalization_penalizes_long_docs():
    idx = BM25Index(k1=1.5, b=0.75)
    idx.add("short", "retrieval")
    idx.add("long", "retrieval " + "filler " * 40)
    # Same single query term, tf 1 in both; the short doc must score higher.
    assert idx.score("retrieval", "short") > idx.score("retrieval", "long")


def test_search_ranks_relevant_first():
    idx = _index()
    results = idx.search("cat mat", k=3)
    assert results[0][0] == "a"


def test_phrase_search_exact_consecutive():
    idx = BM25Index()
    idx.add("x", "reciprocal rank fusion combines lists")
    idx.add("y", "rank and reciprocal are different words here")
    hits = idx.phrase_search("reciprocal rank")
    assert hits == ["x"]


def test_phrase_search_absent_returns_empty():
    idx = _index()
    assert idx.phrase_search("purple monkey dishwasher") == []


def test_duplicate_doc_id_rejected():
    idx = BM25Index()
    idx.add("a", "one")
    try:
        idx.add("a", "two")
        raised = False
    except ValueError:
        raised = True
    assert raised
