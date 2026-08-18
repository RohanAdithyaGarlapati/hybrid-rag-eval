"""Document chunking strategies.

Three strategies sit behind one interface (``chunk_document`` / ``chunk_corpus``):

* ``fixed``       -- consecutive non overlapping windows of ``size`` tokens.
* ``overlapping`` -- a sliding window of ``size`` tokens advancing by ``size - overlap``.
* ``semantic``    -- pack whole sentences up to ``target`` tokens, never exceeding a
                     hard ``max_tokens`` (a single oversized sentence is split).

Tokens are whitespace separated words. Chunks are returned as frozen dataclasses
whose ``indexable_text`` prepends the document title so the title is searchable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    """An immutable passage carved from a source document."""

    doc_id: str
    chunk_id: str
    title: str
    text: str
    strategy: str
    start_token: int
    end_token: int

    @property
    def indexable_text(self) -> str:
        """Text used for indexing, with the document title prepended."""
        title = self.title.strip()
        body = self.text.strip()
        if not title:
            return body
        return f"{title}\n{body}"

    @property
    def n_tokens(self) -> int:
        return len(self.text.split())


def _tokenize(text: str) -> list[str]:
    return text.split()


def _split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _fixed_chunks(tokens: list[str], size: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for start in range(0, len(tokens), size):
        spans.append((start, min(start + size, len(tokens))))
    return spans


def _overlapping_chunks(tokens: list[str], size: int, overlap: int) -> list[tuple[int, int]]:
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    step = size - overlap
    spans: list[tuple[int, int]] = []
    start = 0
    n = len(tokens)
    while start < n:
        end = min(start + size, n)
        spans.append((start, end))
        if end == n:
            break
        start += step
    return spans


def _semantic_chunks(text: str, target: int, max_tokens: int) -> list[tuple[str, int, int]]:
    """Return (chunk_text, start_token, end_token) packing sentences to target/max."""
    if target > max_tokens:
        raise ValueError("target must be <= max_tokens")
    sentences = _split_sentences(text)
    chunks: list[tuple[str, int, int]] = []
    cur: list[str] = []
    cur_len = 0
    token_cursor = 0

    def flush() -> None:
        nonlocal cur, cur_len, token_cursor
        if cur:
            chunk_text = " ".join(cur)
            n = len(chunk_text.split())
            chunks.append((chunk_text, token_cursor, token_cursor + n))
            token_cursor += n
            cur = []
            cur_len = 0

    for sentence in sentences:
        s_tokens = sentence.split()
        # A single sentence longer than the hard max is split into token windows.
        if len(s_tokens) > max_tokens:
            flush()
            for start in range(0, len(s_tokens), max_tokens):
                piece = s_tokens[start:start + max_tokens]
                chunks.append((" ".join(piece), token_cursor, token_cursor + len(piece)))
                token_cursor += len(piece)
            continue
        if cur and cur_len + len(s_tokens) > target:
            flush()
        cur.append(sentence)
        cur_len += len(s_tokens)
    flush()
    return chunks


def chunk_document(
    doc: dict,
    strategy: str = "overlapping",
    *,
    size: int = 60,
    overlap: int = 15,
    target: int = 60,
    max_tokens: int = 90,
) -> list[Chunk]:
    """Chunk a single document dict with keys ``id``, ``title``, ``text``."""
    doc_id = doc["id"]
    title = doc.get("title", "")
    text = doc.get("text", "")
    tokens = _tokenize(text)
    chunks: list[Chunk] = []

    if strategy == "fixed":
        for i, (a, b) in enumerate(_fixed_chunks(tokens, size)):
            chunks.append(_mk(doc_id, i, title, " ".join(tokens[a:b]), strategy, a, b))
    elif strategy == "overlapping":
        for i, (a, b) in enumerate(_overlapping_chunks(tokens, size, overlap)):
            chunks.append(_mk(doc_id, i, title, " ".join(tokens[a:b]), strategy, a, b))
    elif strategy == "semantic":
        for i, (ctext, a, b) in enumerate(_semantic_chunks(text, target, max_tokens)):
            chunks.append(_mk(doc_id, i, title, ctext, strategy, a, b))
    else:
        raise ValueError(f"unknown chunking strategy: {strategy!r}")

    # A document with no text still yields one (empty) chunk so it stays retrievable.
    if not chunks:
        chunks.append(_mk(doc_id, 0, title, "", strategy, 0, 0))
    return chunks


def _mk(doc_id: str, i: int, title: str, text: str, strategy: str, a: int, b: int) -> Chunk:
    return Chunk(
        doc_id=doc_id,
        chunk_id=f"{doc_id}::{strategy}::{i}",
        title=title,
        text=text,
        strategy=strategy,
        start_token=a,
        end_token=b,
    )


def chunk_corpus(docs: Iterable[dict], strategy: str = "overlapping", **kwargs) -> list[Chunk]:
    """Chunk an iterable of document dicts, flattening the result."""
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, strategy, **kwargs))
    return out
