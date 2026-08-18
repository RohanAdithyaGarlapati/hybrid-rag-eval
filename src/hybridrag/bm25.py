"""A from scratch inverted index and BM25 ranker. No external search library.

The index stores, for every term, a postings list of ``Posting`` records carrying
the document id, the term frequency, and the token positions. Positions power an
exact ``phrase_search`` using set lookups. BM25 uses a smoothed inverse document
frequency that is floored at zero and configurable ``k1`` and ``b`` parameters.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A short, deliberately conservative stopword list.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an and are as at be by for from has have in into is it its of on or that the
    their to was were will with what how why does do this these those than then
    """.split()
)


def stem(word: str) -> str:
    """A conservative suffix stemmer.

    It strips a few common inflectional endings while avoiding aggressive rewrites
    that would merge unrelated words. Short words are left untouched.
    """
    w = word
    if len(w) <= 3:
        return w
    # Order matters: longest, most specific suffixes first.
    for suf in ("ingly", "edly"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    if w.endswith("ing") and len(w) - 3 >= 3:
        return w[:-3]
    if w.endswith("edly"):
        return w[:-4]
    if w.endswith("ed") and len(w) - 2 >= 3:
        return w[:-2]
    if w.endswith("ly") and len(w) - 2 >= 3:
        return w[:-2]
    if w.endswith("es") and len(w) - 2 >= 3:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) - 1 >= 3:
        return w[:-1]
    return w


def tokenize(text: str, *, remove_stopwords: bool = True, do_stem: bool = True) -> list[str]:
    """Lowercase, split on non alphanumerics, drop stopwords, and stem."""
    tokens = _TOKEN_RE.findall(text.lower())
    out: list[str] = []
    for tok in tokens:
        if remove_stopwords and tok in STOPWORDS:
            continue
        out.append(stem(tok) if do_stem else tok)
    return out


@dataclass
class Posting:
    doc_id: str
    tf: int
    positions: list[int] = field(default_factory=list)


@dataclass
class BM25Index:
    """Inverted index plus BM25 scoring, built entirely from scratch."""

    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        self.postings: dict[str, list[Posting]] = {}
        self.doc_len: dict[str, int] = {}
        self.doc_ids: list[str] = []
        self._doc_set: set[str] = set()
        self.avgdl: float = 0.0

    # -- construction -------------------------------------------------------
    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self._doc_set:
            raise ValueError(f"duplicate doc_id {doc_id!r}")
        tokens = tokenize(text)
        self.doc_ids.append(doc_id)
        self._doc_set.add(doc_id)
        self.doc_len[doc_id] = len(tokens)
        positions: dict[str, list[int]] = {}
        for pos, tok in enumerate(tokens):
            positions.setdefault(tok, []).append(pos)
        for tok, pos_list in positions.items():
            self.postings.setdefault(tok, []).append(
                Posting(doc_id=doc_id, tf=len(pos_list), positions=pos_list)
            )
        self._refresh_avgdl()

    def add_many(self, items: list[tuple[str, str]]) -> None:
        for doc_id, text in items:
            self.add(doc_id, text)

    def _refresh_avgdl(self) -> None:
        n = len(self.doc_len)
        self.avgdl = (sum(self.doc_len.values()) / n) if n else 0.0

    # -- statistics ---------------------------------------------------------
    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)

    def document_frequency(self, term: str) -> int:
        return len(self.postings.get(term, []))

    def idf(self, term: str) -> float:
        """Smoothed IDF, floored at zero.

        ``log(1 + (N - df + 0.5) / (df + 0.5))`` is strictly positive and decreases
        monotonically as document frequency rises, so rarer terms weigh more.
        """
        n = self.n_docs
        df = self.document_frequency(term)
        if df == 0:
            return 0.0
        val = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        return max(0.0, val)

    # -- scoring ------------------------------------------------------------
    def score(self, query: str, doc_id: str) -> float:
        terms = tokenize(query)
        dl = self.doc_len.get(doc_id, 0)
        if dl == 0 or not terms:
            return 0.0
        score = 0.0
        for term in terms:
            plist = self.postings.get(term)
            if not plist:
                continue
            tf = next((p.tf for p in plist if p.doc_id == doc_id), 0)
            if tf == 0:
                continue
            idf = self.idf(term)
            denom = tf + self.k1 * (1.0 - self.b + self.b * dl / (self.avgdl or 1.0))
            score += idf * (tf * (self.k1 + 1.0)) / denom
        return score

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Return the top k ``(doc_id, score)`` pairs for a query."""
        terms = tokenize(query)
        candidates: set[str] = set()
        for term in terms:
            for posting in self.postings.get(term, []):
                candidates.add(posting.doc_id)
        scored = [(doc_id, self.score(query, doc_id)) for doc_id in candidates]
        scored = [(d, s) for d, s in scored if s > 0.0]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]

    # -- phrase search ------------------------------------------------------
    def phrase_search(self, phrase: str) -> list[str]:
        """Return doc ids containing the exact consecutive phrase.

        Uses positional postings: a phrase of terms t0 t1 ... occurs when some
        position p has t0 at p, t1 at p+1, and so on. Position membership is tested
        with set lookups for speed.
        """
        terms = tokenize(phrase)
        if not terms:
            return []
        # Build per-term {doc_id: set(positions)} maps, intersecting candidate docs.
        term_maps: list[dict[str, set[int]]] = []
        candidate_docs: set[str] | None = None
        for term in terms:
            pmap: dict[str, set[int]] = {}
            for posting in self.postings.get(term, []):
                pmap[posting.doc_id] = set(posting.positions)
            term_maps.append(pmap)
            docs = set(pmap.keys())
            candidate_docs = docs if candidate_docs is None else (candidate_docs & docs)
            if not candidate_docs:
                return []
        results: list[str] = []
        for doc_id in candidate_docs or set():
            starts = term_maps[0][doc_id]
            for start in starts:
                if all((start + offset) in term_maps[offset][doc_id] for offset in range(len(terms))):
                    results.append(doc_id)
                    break
        results.sort()
        return results
