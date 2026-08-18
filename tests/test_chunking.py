from hybridrag.chunking import Chunk, chunk_corpus, chunk_document

DOC = {
    "id": "d1",
    "title": "Title Words",
    "text": " ".join(f"tok{i}" for i in range(100)),
}


def test_fixed_chunk_size_guarantee():
    chunks = chunk_document(DOC, "fixed", size=20)
    assert all(c.n_tokens <= 20 for c in chunks)
    # 100 tokens / 20 == exactly 5 chunks with no remainder.
    assert len(chunks) == 5
    # Fixed chunks are contiguous and non overlapping.
    assert chunks[0].end_token == chunks[1].start_token


def test_overlapping_boundary_behavior():
    chunks = chunk_document(DOC, "overlapping", size=20, overlap=5)
    assert all(c.n_tokens <= 20 for c in chunks)
    # Step is size - overlap == 15.
    assert chunks[1].start_token - chunks[0].start_token == 15
    # Consecutive windows share exactly `overlap` tokens.
    shared = chunks[0].end_token - chunks[1].start_token
    assert shared == 5


def test_overlapping_covers_all_tokens():
    chunks = chunk_document(DOC, "overlapping", size=20, overlap=5)
    assert chunks[-1].end_token == 100


def test_semantic_hard_max_guarantee():
    text = ". ".join("word " * 30 for _ in range(4)) + "."
    doc = {"id": "d2", "title": "T", "text": text}
    chunks = chunk_document(doc, "semantic", target=40, max_tokens=60)
    assert all(c.n_tokens <= 60 for c in chunks)


def test_semantic_packs_multiple_sentences():
    text = "Alpha one two. Beta three four. Gamma five six."
    doc = {"id": "d3", "title": "T", "text": text}
    chunks = chunk_document(doc, "semantic", target=100, max_tokens=100)
    # Everything fits in the target, so it packs into a single chunk.
    assert len(chunks) == 1


def test_indexable_text_prepends_title():
    chunks = chunk_document(DOC, "fixed", size=20)
    c = chunks[0]
    assert c.indexable_text.startswith("Title Words")


def test_chunk_is_frozen():
    c = chunk_document(DOC, "fixed", size=20)[0]
    assert isinstance(c, Chunk)
    try:
        c.text = "mutated"  # type: ignore[misc]
        raised = False
    except Exception:
        raised = True
    assert raised


def test_chunk_corpus_flattens():
    docs = [DOC, {"id": "d4", "title": "T2", "text": "one two three"}]
    all_chunks = chunk_corpus(docs, "fixed", size=20)
    assert {c.doc_id for c in all_chunks} == {"d1", "d4"}
